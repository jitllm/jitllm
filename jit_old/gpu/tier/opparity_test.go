package tier

import (
	"flag"
	"fmt"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit_old/gpu/backend/recorder"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/paritytest"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

var opExplain = flag.Bool("explain", false, "write out every missing instruction of main's")

// opKernels is the kernels the tier's lowering forms for op o on profile
// prof: inputs and outputs of the widths given, a weight vector bound where
// vec, at rows rows of plan.
func opKernels(t *testing.T, prof recorder.Profile, o devOp, ins, outs []int, vec bool, plan nn.LayerPlan, rows int) []*ir.Kernel {
	t.Helper()
	rec := recorder.New(prof, 1<<40)
	g := &devTier{devShared: &devShared{dev: rec}, devSess: &devSess{}, lane: &lane{}, Stats: &Stats{}}
	bs := &blockScratch{p: plan}
	prog := &devProgram{}
	for _, w := range ins {
		o.In = append(o.In, prog.val("in", devType{Elem: elemF32, Layout: layoutRow, Width: w}, slotAuto))
	}
	for _, w := range outs {
		o.Out = append(o.Out, prog.val("out", devType{Elem: elemF32, Layout: layoutRow, Width: w}, slotAuto))
	}
	if vec {
		b, err := rec.Alloc(4 * ins[0])
		if err != nil {
			t.Fatal(err)
		}
		o.V = b
	}
	if ls := g.genOp(&o, prog, bs, rows); len(ls) == 0 {
		t.Fatalf("no kernel formed: %v; %s", o.Why, g.LastErr)
	}
	var ks []*ir.Kernel
	for _, k := range rec.IR() {
		if strings.HasPrefix(k.Comment, "composed") {
			ks = append(ks, k)
		}
	}
	return ks
}

// chainKernels is the composed kernels of a program of kinds over rows of
// width w, each op reading the last's value (an add reads a second row
// too) and through the composer's program: lowered each alone, then fused.
// A norm binds a vector.
func chainKernels(t *testing.T, prof recorder.Profile, kinds []devKind, w int, plan nn.LayerPlan, rows int) []*ir.Kernel {
	t.Helper()
	rec := recorder.New(prof, 1<<40)
	g := &devTier{devShared: &devShared{dev: rec}, devSess: &devSess{}, lane: &lane{}, Stats: &Stats{}}
	bs := &blockScratch{p: plan}
	prog := &devProgram{}
	row := func(e devElem) devVal {
		return prog.val("v", devType{Elem: e, Layout: layoutRow, Width: w}, slotAuto)
	}
	x := row(elemF32)
	for _, k := range kinds {
		o := devOp{Kind: k, Act: graph.ActSiLU, In: []devVal{x}}
		switch k {
		case dkAdd, dkActMul:
			o.In = append(o.In, row(elemF32))
		case dkNorm:
			b, err := rec.Alloc(4 * w)
			if err != nil {
				t.Fatal(err)
			}
			o.V = b
		}
		switch k {
		case dkQuant:
			x = row(elemQ8)
		case dkActMul:
			x = prog.val("act", devType{Elem: elemF32, Layout: layoutRow, Width: w}, slotAct)
		default:
			x = row(elemF32)
		}
		o.Out = []devVal{x}
		prog.ops = append(prog.ops, o)
	}
	g.genProgram(prog, bs, rows)
	for i, o := range prog.ops {
		if len(o.Gen) == 0 && o.Why != whyFused {
			t.Fatalf("op %d (%v) formed nothing: %v; %s", i, o.Kind, o.Why, g.LastErr)
		}
	}
	var ks []*ir.Kernel
	for _, k := range rec.IR() {
		if strings.HasPrefix(k.Comment, "composed") {
			ks = append(ks, k)
		}
	}
	return ks
}

// sameAsMain holds composed to main's kernels, instruction by instruction.
func sameAsMain(t *testing.T, main, composed []*ir.Kernel) {
	t.Helper()
	total, n, miss := paritytest.Compare(main, composed)
	ctotal, cn, extra := paritytest.Compare(composed, main)
	in := paritytest.NewInterner()
	if *opExplain {
		for _, m := range extra {
			t.Logf("extra %%%d %s", m.Op, paritytest.Explain(in, m.K, m.Op, 4))
		}
	}
	if n == total && cn != ctotal {
		t.Errorf("every value of main's, and %d of the composer's %d more: first %%%d %s", ctotal-cn, ctotal,
			extra[0].Op, paritytest.Explain(in, extra[0].K, extra[0].Op, 3))
	}
	if n == total {
		return
	}
	if *opExplain {
		for _, m := range miss {
			t.Logf("missing %%%d %s", m.Op, paritytest.Explain(in, m.K, m.Op, 4))
		}
		for _, k := range composed {
			for i := range k.Ops {
				t.Logf("composed %%%d %s", i, paritytest.Explain(in, k, i, 4))
			}
		}
	}
	t.Errorf("%d of %d of main's values (%.1f%%); first missing %%%d %s", n, total,
		100*float64(n)/float64(total), miss[0].Op, paritytest.Explain(in, miss[0].K, miss[0].Op, 3))
}

// TestElementwiseIsMains holds the tier's lowering of the elementwise ops
// and the quantizers to main's kernels for them on a ptx target.
func TestElementwiseIsMains(t *testing.T) {
	prof := recorder.Profiles()["cuda-sm70"]
	for _, w := range []int{64, 256, 2048} {
		for _, rows := range []int{1, 32} {
			plan := nn.LayerPlan{ActWin: 32, NEmbd: w}
			n, win := rows*w, min(32, w)
			cases := []struct {
				name      string
				op        devOp
				ins, outs []int
				main      func() (*ir.Kernel, error)
			}{
				{"add", devOp{Kind: dkAdd}, []int{w, w}, []int{w}, func() (*ir.Kernel, error) { return mainkernels.Add(n) }},
				{"act_mul", devOp{Kind: dkActMul, Act: graph.ActSiLU}, []int{w, w}, []int{w}, func() (*ir.Kernel, error) {
					return mainkernels.ActMul(n, mainkernels.ActSiLU)
				}},
				{"act", devOp{Kind: dkAct, Act: graph.ActSiLU}, []int{w}, []int{w}, func() (*ir.Kernel, error) {
					return mainkernels.Act(n, mainkernels.ActSiLU)
				}},
				{"quantize", devOp{Kind: dkQuant}, []int{w}, []int{w}, func() (*ir.Kernel, error) {
					return mainkernels.Quantize(n, win)
				}},
			}
			t.Run(fmt.Sprintf("act_quant_%dx%d", rows, w), func(t *testing.T) {
				mk, err := mainkernels.QuantizeAct(n, win, mainkernels.ActSiLU)
				if err != nil {
					t.Fatal(err)
				}
				sameAsMain(t, []*ir.Kernel{mk}, chainKernels(t, prof, []devKind{dkActMul, dkQuant}, w, plan, rows))
			})
			for _, c := range cases {
				t.Run(fmt.Sprintf("%s_%dx%d", c.name, rows, w), func(t *testing.T) {
					mk, err := c.main()
					if err != nil {
						t.Fatal(err)
					}
					sameAsMain(t, []*ir.Kernel{mk}, opKernels(t, prof, c.op, c.ins, c.outs, false, plan, rows))
				})
			}
		}
	}
}

// TestRotaryIsMains holds the tier's lowering of q's rotary to main's
// RoPERows (after its CopyAtRows where the rotary is partial), NORM and
// NEOX, whole and partial, at decode and a chunk's rows.
func TestRotaryIsMains(t *testing.T) {
	prof := recorder.Profiles()["cuda-sm70"]
	for _, c := range []struct{ heads, hd, nrot int }{{4, 64, 64}, {8, 128, 128}, {4, 64, 32}, {2, 128, 64}} {
		for _, neox := range []bool{false, true} {
			for _, rows := range []int{1, 32} {
				layout := graph.RopeNorm
				if neox {
					layout = graph.RopeNeox
				}
				w := c.heads * c.hd
				name := fmt.Sprintf("h%d_hd%d_rot%d_neox%v_%d", c.heads, c.hd, c.nrot, neox, rows)
				t.Run(name, func(t *testing.T) {
					var main []*ir.Kernel
					if c.nrot < c.hd {
						k, err := mainkernels.CopyAtRows(w, rows)
						if err != nil {
							t.Fatal(err)
						}
						main = append(main, k)
					}
					k, err := mainkernels.RoPERows(c.heads, c.hd, c.nrot, neox, rows)
					if err != nil {
						t.Fatal(err)
					}
					main = append(main, k)
					op := devOp{Kind: dkRopeQ, A: graph.Attrs{HeadDim: c.hd, NRot: c.nrot, Layout: layout}}
					sameAsMain(t, main, opKernels(t, prof, op, []int{w}, []int{w}, false, nn.LayerPlan{NEmbd: w}, rows))
				})
			}
		}
	}
}

// TestNormsAreMains holds the tier's lowering of the RMSNorm ops -- alone,
// after an add, with the int8 operand -- to main's warp norms on a ptx
// target with 32 lanes, at decode and at a chunk's rows.
func TestNormsAreMains(t *testing.T) {
	prof := recorder.Profiles()["cuda-sm70"]
	const eps = 1e-6
	for _, w := range []int{64, 256, 1024, 2048} {
		for _, rows := range []int{1, 32} {
			plan := nn.LayerPlan{RMSEps: eps, ActWin: 32, NEmbd: w}
			win := min(32, w)
			cases := []struct {
				kinds []devKind
				main  func() (*ir.Kernel, error)
			}{
				{[]devKind{dkNorm}, func() (*ir.Kernel, error) {
					return mainkernels.RMSNormRowsWarp(w, rows, eps, false, false)
				}},
				{[]devKind{dkNorm, dkQuant}, func() (*ir.Kernel, error) {
					return mainkernels.RMSNormQuantRowsWarp(w, rows, eps, false, false, win)
				}},
				{[]devKind{dkAdd, dkNorm}, func() (*ir.Kernel, error) {
					return mainkernels.RMSNormRowsWarp(w, rows, eps, false, true)
				}},
				{[]devKind{dkAdd, dkNorm, dkQuant}, func() (*ir.Kernel, error) {
					return mainkernels.RMSNormQuantRowsWarp(w, rows, eps, false, true, win)
				}},
			}
			for _, c := range cases {
				t.Run(fmt.Sprintf("%v_%dx%d", c.kinds, rows, w), func(t *testing.T) {
					mk, err := c.main()
					if err != nil {
						t.Fatal(err)
					}
					sameAsMain(t, []*ir.Kernel{mk}, chainKernels(t, prof, c.kinds, w, plan, rows))
				})
			}
		}
	}
}

// TestRowWindowNormQuantIsMains holds a norm's int8 operand over a window as
// wide as the row (one scale a row, BitNet's activations) to main's, on a
// target whose norm reduces by a warp and on one whose norm reduces through
// its workgroup.
func TestRowWindowNormQuantIsMains(t *testing.T) {
	const eps = 1e-6
	for _, pc := range []struct {
		prof string
		main func(k, rows int, eps float32, addOne, add bool, window int) (*ir.Kernel, error)
	}{{"cuda-sm70", mainkernels.RMSNormQuantRowsWarp}, {"hip-gfx942", mainkernels.RMSNormQuantRows}} {
		prof := recorder.Profiles()[pc.prof]
		for _, w := range []int{64, 256, 512, 2048, 8192} {
			for _, rows := range []int{1, 8, 32} {
				plan := nn.LayerPlan{RMSEps: eps, ActWin: 1 << 16, NEmbd: 256} // BitNet's: a window past every row, rows wider than the embedding
				for _, add := range []bool{false, true} {
					kinds := []devKind{dkNorm, dkQuant}
					if add {
						kinds = []devKind{dkAdd, dkNorm, dkQuant}
					}
					t.Run(fmt.Sprintf("%s_%v_%dx%d", pc.prof, kinds, rows, w), func(t *testing.T) {
						mk, err := pc.main(w, rows, eps, false, add, w)
						if err != nil {
							t.Fatal(err)
						}
						sameAsMain(t, []*ir.Kernel{mk}, chainKernels(t, prof, kinds, w, plan, rows))
					})
				}
			}
		}
	}
}
