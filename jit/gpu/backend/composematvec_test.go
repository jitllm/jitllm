package backend_test

import (
	"bytes"
	"encoding/binary"
	"math"
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

// TestComposedSplitMatVecMatchesTheMenu holds a k split across threads to the
// menu's: compose.MatVec's partial plane to MatVec's at Split 4, and
// compose.SplitSum to Reduce, then with a residual epilogue to Reduce's sum
// plus the residual in float32; a subtracted residual is its violation.
func TestComposedSplitMatVecMatchesTheMenu(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const rows, k, split = 128, 512, 4
	rng := rand.New(rand.NewSource(17))
	resid := randActs(rows, rng)
	epi := func(op compose.Op) *compose.Epi {
		g := &compose.Row{Width: rows, Rows: 1}
		in := g.Add(compose.Node{Kind: compose.NInput})
		return &compose.Epi{G: g, Out: g.Bin(op, in, g.Load(g.Buf(compose.BufRow)))}
	}
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
			epiArg := false
			run := func(ker *ir.Kernel, threads, width, outBytes int, args ...backend.Buf) []byte {
				c, err := d.Compile(ker)
				if err != nil {
					t.Fatal(err)
				}
				defer c.Close()
				out := g.up(make([]byte, outBytes))
				all := append(append([]backend.Buf(nil), args...), out)
				if epiArg {
					all = append(all, bR)
				}
				if err := c.Launch((threads+width-1)/width, width, all...); err != nil {
					t.Fatal(err)
				}
				raw := make([]byte, outBytes)
				if err := out.Read(raw); err != nil {
					t.Fatal(err)
				}
				return raw
			}
			for _, q := range []kernels.Quant{kernels.Q4_0, kernels.Q8_0, kernels.Q4_K, kernels.Q6_K, kernels.Float16} {
				qs, dw, sc, err := kernels.PackWeights(q, randWeights(q, rows, k, rng), rows, k)
				if err != nil {
					t.Fatal(err)
				}
				bQS, bD, bSC := g.up(u32bytes(qs)), g.up(u32bytes(dw)), g.up(u32bytes(sc))
				dArg := bD
				if kernels.IsFloat(q) {
					dArg = bX
				}
				menuMV, err := kernels.MatVec(kernels.MatVecShape{T: q, K: k, Rows: rows, Split: split})
				if err != nil {
					t.Fatal(err)
				}
				wantPart := run(menuMV, rows*split, 128, rows*split*4, bQS, dArg, bSC, bA, bAX)
				mv := &compose.MatVec{T: q, K: k, Rows: rows, Split: split}
				ker, err := mv.Emit()
				if err != nil {
					t.Fatal(err)
				}
				gotPart := run(ker, mv.Threads(), mv.Width(), rows*split*4, bQS, dArg, bSC, bA, bAX)
				if !bytes.Equal(gotPart, wantPart) {
					t.Errorf("%v: the composed partial plane parts from the menu's", q)
				}
				bP := g.up(wantPart)
				red, err := kernels.Reduce(rows, split)
				if err != nil {
					t.Fatal(err)
				}
				wantSum := run(red, rows, 128, rows*4, bP)
				ss := &compose.SplitSum{Rows: rows, Split: split}
				sk, err := ss.Emit()
				if err != nil {
					t.Fatal(err)
				}
				if got := run(sk, rows, compose.SplitSumWidth, rows*4, bP); !bytes.Equal(got, wantSum) {
					t.Errorf("%v: SplitSum parts from Reduce", q)
				}
				sums := bytesF32(wantSum)
				want := make([]float32, rows)
				for i := range want {
					want[i] = sums[i] + resid[i]
				}
				epiArg = true
				se := &compose.SplitSum{Rows: rows, Split: split, Epi: epi(compose.OpAdd)}
				ek, err := se.Emit()
				if err != nil {
					t.Fatal(err)
				}
				if got := run(ek, rows, compose.SplitSumWidth, rows*4, bP); !bytes.Equal(got, f32bytes(want)) {
					t.Errorf("%v: SplitSum with a residual parts from Reduce's sum plus the residual", q)
				}
				bad := &compose.SplitSum{Rows: rows, Split: split, Epi: epi(compose.OpSub)}
				bk, err := bad.Emit()
				if err != nil {
					t.Fatal(err)
				}
				if got := run(bk, rows, compose.SplitSumWidth, rows*4, bP); bytes.Equal(got, f32bytes(want)) {
					t.Errorf("%v: a subtracted residual matched", q)
				}
				epiArg = false
			}
		})
	}
}

func bytesF32(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}

// TestComposedChunkMatVecMatchesTheMenu holds the matvec former over a chunk
// of tokens (the dp4a batched form) to the menu's MatVec bit for bit: one
// token a thread, four, eight with centering, two rows a thread, and a k
// split's partial planes; a moved token group (one token a thread over a
// kernel built for four) is its violation.
func TestComposedChunkMatVecMatchesTheMenu(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const rows, k, ntok = 64, 512, 8
	rng := rand.New(rand.NewSource(19))
	scheds := []struct {
		name             string
		tok, rowt, split int
	}{{"tok1", 1, 1, 1}, {"tok4", 4, 1, 1}, {"tok8", 8, 1, 1}, {"tok4rowt2", 4, 2, 1}, {"tok2split4", 2, 1, 4}}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			g := newGPU(t, d)
			defer g.free()
			acts := randActs(ntok*k, rng)
			bX := g.up(f32bytes(acts))
			bA, bAX := g.up(make([]byte, ntok*k)), g.up(make([]byte, ntok*k/32*3*4))
			qk, err := kernels.Quantize(ntok*k, 32)
			if err != nil {
				t.Fatal(err)
			}
			qc, err := d.Compile(qk)
			if err != nil {
				t.Fatal(err)
			}
			if err := qc.Launch((kernels.QuantizeThreads(ntok*k/32)+127)/128, 128, bX, bA, bAX); err != nil {
				t.Fatal(err)
			}
			qc.Close()
			ran := 0
			for _, q := range []kernels.Quant{kernels.Q4_0, kernels.Q8_0, kernels.Q4_K, kernels.Q3_K, kernels.Q5_K, kernels.Q6_K,
				kernels.Q5_0, kernels.MXFP4, kernels.Q5_1, kernels.TQ2, kernels.Float16} {
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
				run := func(ker *ir.Kernel, threads, width, n int) []byte {
					c, err := d.Compile(ker)
					if err != nil {
						t.Fatalf("%v: compile: %v", q, err)
					}
					defer c.Close()
					out := g.up(make([]byte, n*4))
					if err := c.Launch((threads+width-1)/width, width, bQS, dArg, bSC, bA, bAX, out); err != nil {
						t.Fatal(err)
					}
					raw := make([]byte, n*4)
					if err := out.Read(raw); err != nil {
						t.Fatal(err)
					}
					return raw
				}
				for _, sc := range scheds {
					s := kernels.MatVecShape{T: q, K: k, Rows: rows, NTok: ntok, Tok: sc.tok, Rowt: sc.rowt, Split: sc.split}
					menu, err := kernels.MatVec(s)
					if err != nil {
						t.Logf("%v %s: the menu refuses: %v", q, sc.name, err)
						continue
					}
					n := rows * ntok * sc.split
					want := run(menu, rows/sc.rowt*(ntok/sc.tok)*sc.split, 128, n)
					mv := &compose.MatVec{T: q, K: k, Rows: rows, NTok: ntok, Tok: sc.tok, Rowt: sc.rowt, Split: sc.split}
					ker, err := mv.Emit()
					if err != nil {
						t.Errorf("%v %s: the former refuses what the menu builds: %v", q, sc.name, err)
						continue
					}
					if got := run(ker, mv.Threads(), mv.Width(), n); !bytes.Equal(got, want) {
						t.Errorf("%v %s: the composed chunk matvec parts from the menu's", q, sc.name)
					}
					if sc.tok > 1 {
						bad := &compose.MatVec{T: q, K: k, Rows: rows, NTok: ntok, Tok: 1, Rowt: sc.rowt, Split: sc.split}
						bk, err := bad.Emit()
						if err != nil {
							t.Fatal(err)
						}
						// Launched at the four-token kernel's grid: tokens past it go unwritten.
						if got := run(bk, mv.Threads(), mv.Width(), n); bytes.Equal(got, want) {
							t.Errorf("%v %s: a kernel for one token a thread on the four-token grid matched", q, sc.name)
						}
					}
					ran++
				}
			}
			if ran < 40 {
				t.Fatalf("only %d format and schedule pairs ran", ran)
			}
			t.Logf("%d format and schedule pairs bit-exact", ran)
		})
	}
}
