package backend_test

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// TestComposedGemmMatchesTheMenu holds the GEMM former (compose.Gemm: the
// staged binary16 loop nest around the format's VoltaDeq fragment) to the
// menu's GemmVoltaEpi bit for bit, for every format the staged GEMM takes, at
// each instruction shape the device compiles, single and double staging,
// with a residual epilogue; and to GemmVolta's partial planes at a k split.
// A subtracted residual is its violation.
func TestComposedGemmMatchesTheMenu(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const rows, k, ntok = 256, 1024, 128
	rng := rand.New(rand.NewSource(23))
	resid := randActs(rows*ntok, rng)
	epi := func(op compose.Op) *compose.Epi {
		g := &compose.Row{Width: rows, Rows: 1}
		in := g.Add(compose.Node{Kind: compose.NInput})
		return &compose.Epi{G: g, Out: g.Bin(op, in, g.Load(g.Buf(compose.BufRow)))}
	}
	tiles := []oldkernels.VoltaTile{
		{MT: 4, NT: 4, WM: 2, WN: 2, KB: 1, F16K: 16},
		{MT: 2, NT: 4, WM: 2, WN: 2, KB: 1, F16K: 8},
		{MT: 4, NT: 4, WM: 2, WN: 4, KB: 2, F16K: 16, Single: true},
		{MT: 2, NT: 2, WM: 2, WN: 2, KB: 1},
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			g := newGPU(t, d)
			defer g.free()
			acts := randActs(ntok*k, rng)
			bX := g.up(f32bytes(acts))
			bR := g.up(f32bytes(resid))
			ran := 0
			for q := kernels.Q4_0; q <= kernels.TQ2F; q++ {
				if !oldkernels.GemmVoltaOK(q) || kernels.DeviceWhyNot(q) != "" {
					continue
				}
				sub, _, _, _, _ := oldkernels.GemmFormat(q)
				ak, err := oldkernels.ActF16T(q, ntok, k)
				if err != nil {
					t.Fatal(err)
				}
				ac, err := d.Compile(ak)
				if err != nil {
					t.Logf("%v: no binary16 conversion here: %v", q, err)
					continue
				}
				bB := g.up(make([]byte, ntok*k*2))
				if err := ac.Launch((ntok*k/sub+127)/128, 128, bX, bB); err != nil {
					t.Fatal(err)
				}
				ac.Close()
				qs, dw, sc, err := kernels.PackWeights(q, randWeights(q, rows, k, rng), rows, k)
				if err != nil {
					t.Fatal(err)
				}
				bQS, bD, bSC := g.up(u32bytes(qs)), g.up(u32bytes(dw)), g.up(u32bytes(sc))
				run := func(ker *ir.Kernel, groups, width, n int, extra ...backend.Buf) ([]byte, bool) {
					c, err := d.Compile(ker)
					if err != nil {
						return nil, false
					}
					defer c.Close()
					out := g.up(make([]byte, n*4))
					args := append([]backend.Buf{bQS, bD, bSC, bB, out}, extra...)
					if err := c.Launch(groups, width, args...); err != nil {
						t.Fatal(err)
					}
					raw := make([]byte, n*4)
					if err := out.Read(raw); err != nil {
						t.Fatal(err)
					}
					return raw, true
				}
				for _, tl := range tiles {
					s := oldkernels.MatVecShape{T: q, K: k, Rows: rows, NTok: ntok}
					menu, err := oldkernels.GemmVoltaEpi(s, tl, epi(compose.OpAdd))
					if err != nil {
						continue
					}
					want, ok := run(menu, oldkernels.GemmVoltaGroups(s, tl), tl.Threads(), rows*ntok, bR)
					if !ok {
						continue // the device compiles no such instruction shape
					}
					gm := &compose.Gemm{T: q, K: k, Rows: rows, NTok: ntok, Tile: tl, Epi: epi(compose.OpAdd)}
					ker, err := gm.Emit()
					if err != nil {
						t.Errorf("%v %+v: the former refuses what the menu builds: %v", q, tl, err)
						continue
					}
					if got, _ := run(ker, gm.Groups(), gm.Width(), rows*ntok, bR); !bytes.Equal(got, want) {
						t.Errorf("%v %+v: the composed GEMM parts from the menu's", q, tl)
					}
					bad := &compose.Gemm{T: q, K: k, Rows: rows, NTok: ntok, Tile: tl, Epi: epi(compose.OpSub)}
					bk, err := bad.Emit()
					if err != nil {
						t.Fatal(err)
					}
					if got, _ := run(bk, bad.Groups(), bad.Width(), rows*ntok, bR); bytes.Equal(got, want) {
						t.Errorf("%v %+v: a subtracted residual matched", q, tl)
					}
					ss := s
					ss.Split = 2
					if sm, err := oldkernels.GemmVolta(ss, tl); err == nil {
						wantP, _ := run(sm, oldkernels.GemmVoltaGroups(ss, tl), tl.Threads(), 2*rows*ntok)
						sg := &compose.Gemm{T: q, K: k, Rows: rows, NTok: ntok, Tile: tl, Split: 2}
						sk, err := sg.Emit()
						if err != nil {
							t.Errorf("%v %+v: the former refuses a split the menu builds: %v", q, tl, err)
						} else if got, _ := run(sk, sg.Groups(), sg.Width(), 2*rows*ntok); !bytes.Equal(got, wantP) {
							t.Errorf("%v %+v: the composed split's partial planes part from the menu's", q, tl)
						}
					}
					ran++
				}
			}
			if ran == 0 {
				t.Skipf("no staged binary16 GEMM compiles on %s", d.API())
			}
			t.Logf("%d format and tile pairs bit-exact", ran)
		})
	}
}

// TestComposedGemmOperandMatchesActF16T holds the GEMM's operand group
// (compose.Gemm.Operand: two indexed loads, a pack and a store) to ActF16T bit
// for bit for every format the staged GEMM takes; a swapped pair is its
// violation.
func TestComposedGemmOperandMatchesActF16T(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const k, ntok = 1024, 16
	rng := rand.New(rand.NewSource(29))
	acts := randActs(ntok*k, rng)
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			g := newGPU(t, d)
			defer g.free()
			bX := g.up(f32bytes(acts))
			ran := 0
			for q := kernels.Q4_0; q <= kernels.TQ2F; q++ {
				if !oldkernels.GemmVoltaOK(q) {
					continue
				}
				sub, _, _, _, _ := oldkernels.GemmFormat(q)
				run := func(ker *ir.Kernel, groups, width int) ([]byte, bool) {
					c, err := d.Compile(ker)
					if err != nil {
						return nil, false
					}
					defer c.Close()
					out := g.up(make([]byte, ntok*k*2))
					if err := c.Launch(groups, width, bX, out); err != nil {
						t.Fatal(err)
					}
					raw := make([]byte, ntok*k*2)
					if err := out.Read(raw); err != nil {
						t.Fatal(err)
					}
					return raw, true
				}
				ak, err := oldkernels.ActF16T(q, ntok, k)
				if err != nil {
					t.Fatal(err)
				}
				want, ok := run(ak, (ntok*k/sub+127)/128, 128)
				if !ok {
					continue
				}
				gm := &compose.Gemm{T: q, K: k, NTok: ntok}
				op := gm.Operand()
				ker, err := op.Emit(compose.Schedule{Group: 128})
				if err != nil {
					t.Fatal(err)
				}
				if got, _ := run(ker, ntok, 128); !bytes.Equal(got, want) {
					t.Errorf("%v: the composed operand parts from ActF16T", q)
				}
				bad := gm.Operand()
				for i, n := range bad.Nodes {
					if n.Kind == compose.NBinary && n.Op == compose.OpPackF16 {
						bad.Nodes[i].A, bad.Nodes[i].B = n.B, n.A
					}
				}
				bk, err := bad.Emit(compose.Schedule{Group: 128})
				if err != nil {
					t.Fatal(err)
				}
				if got, _ := run(bk, ntok, 128); bytes.Equal(got, want) {
					t.Errorf("%v: a swapped pair matched", q)
				}
				ran++
			}
			if ran == 0 {
				t.Skipf("no binary16 conversion compiles on %s", d.API())
			}
			t.Logf("%d formats bit-exact", ran)
		})
	}
}
