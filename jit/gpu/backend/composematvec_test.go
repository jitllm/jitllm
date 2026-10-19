package backend_test

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestComposedMatVecMatchesTheMenu holds the matvec former (compose.MatVec:
// the loop nest around the format's SubDot fragment) to the menu's MatVecEpi
// bit for bit, for every format at decode, an in-group k split and two rows a
// thread, with a residual epilogue; a subtracted residual is its violation.
func TestComposedMatVecMatchesTheMenu(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const rows, k = 128, 512
	rng := rand.New(rand.NewSource(13))
	resid := randActs(rows, rng)
	epi := func(op compose.Op) *compose.Epi {
		g := &compose.Row{Width: rows, Rows: 1}
		in := g.Add(compose.Node{Kind: compose.NInput})
		return &compose.Epi{G: g, Out: g.Bin(op, in, g.Load(g.Buf(compose.BufRow)))}
	}
	formats := []kernels.Quant{kernels.Q4_0, kernels.Q8_0, kernels.Q4_K, kernels.Q3_K, kernels.Q5_K, kernels.Q6_K,
		kernels.Q5_0, kernels.MXFP4, kernels.Float32, kernels.Float16, kernels.BFloat16, kernels.Q5_1, kernels.TQ2, kernels.TQ2F}
	scheds := []struct {
		name        string
		split, rowt int
		group       bool
	}{{"decode", 1, 1, false}, {"group4", 4, 1, true}, {"rowt2", 1, 2, false}}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			g := newGPU(t, d)
			defer g.free()
			acts := randActs(k, rng)
			bX := g.up(f32bytes(acts))
			bA, bAX := g.up(make([]byte, k)), g.up(make([]byte, k/32*3*4))
			qk, err := kernels.Quantize(k, 32)
			if err != nil {
				t.Fatal(err)
			}
			qc, err := d.Compile(qk)
			if err != nil {
				t.Fatal(err)
			}
			if err := qc.Launch((kernels.QuantizeThreads(k/32)+127)/128, 128, bX, bA, bAX); err != nil {
				t.Fatal(err)
			}
			qc.Close()
			bR := g.up(f32bytes(resid))
			ran := 0
			for _, q := range formats {
				if kernels.DeviceWhyNot(q) != "" {
					continue
				}
				qs, dw, sc, err := kernels.PackWeights(q, randWeights(q, rows, k, rng), rows, k)
				if err != nil {
					t.Fatal(err)
				}
				bQS, bD, bSC := g.up(u32bytes(qs)), g.up(u32bytes(dw)), g.up(u32bytes(sc))
				dArg := bD
				if kernels.IsFloat(q) {
					dArg = bX
				}
				launch := func(ker *ir.Kernel, threads, width int) []byte {
					c, err := d.Compile(ker)
					if err != nil {
						t.Fatalf("%v: compile: %v", q, err)
					}
					defer c.Close()
					out := g.up(make([]byte, rows*4))
					if err := c.Launch((threads+width-1)/width, width, bQS, dArg, bSC, bA, bAX, out, bR); err != nil {
						t.Fatal(err)
					}
					raw := make([]byte, rows*4)
					if err := out.Read(raw); err != nil {
						t.Fatal(err)
					}
					return raw
				}
				for _, sc := range scheds {
					s := kernels.MatVecShape{T: q, K: k, Rows: rows, Split: sc.split, GroupSplit: sc.group, Rowt: sc.rowt}
					menu, err := kernels.MatVecEpi(s, epi(compose.OpAdd))
					if err != nil {
						t.Logf("%v %s: the menu refuses: %v", q, sc.name, err)
						continue
					}
					width := 128
					if sc.group {
						width = kernels.GroupSplitWidth(sc.split)
					}
					want := launch(menu, rows/sc.rowt*sc.split, width)
					mv := &compose.MatVec{T: q, K: k, Rows: rows, Split: sc.split, Group: sc.group, Rowt: sc.rowt, Epi: epi(compose.OpAdd)}
					ker, err := mv.Emit()
					if err != nil {
						t.Errorf("%v %s: the former refuses what the menu builds: %v", q, sc.name, err)
						continue
					}
					if got := launch(ker, mv.Threads(), mv.Width()); !bytes.Equal(got, want) {
						t.Errorf("%v %s: the composed matvec parts from the menu's", q, sc.name)
					}
					bad := &compose.MatVec{T: q, K: k, Rows: rows, Split: sc.split, Group: sc.group, Rowt: sc.rowt, Epi: epi(compose.OpSub)}
					bk, err := bad.Emit()
					if err != nil {
						t.Fatal(err)
					}
					if got := launch(bk, bad.Threads(), bad.Width()); bytes.Equal(got, want) {
						t.Errorf("%v %s: a subtracted residual matched: the gate cannot see the epilogue", q, sc.name)
					}
					ran++
				}
			}
			if ran < 30 {
				t.Fatalf("only %d format and schedule pairs ran", ran)
			}
			t.Logf("%d format and schedule pairs bit-exact", ran)
		})
	}
}
