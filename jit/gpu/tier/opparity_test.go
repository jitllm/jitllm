package tier

import (
	"flag"
	"fmt"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend/recorder"
	"github.com/jitllm/jitllm/jit/gpu/ir"
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

// sameAsMain holds composed to main's kernels, instruction by instruction.
func sameAsMain(t *testing.T, main, composed []*ir.Kernel) {
	t.Helper()
	total, n, miss := paritytest.Compare(main, composed)
	if n == total {
		return
	}
	in := paritytest.NewInterner()
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

// TestNormsAreMains holds the tier's lowering of the RMSNorm ops -- alone,
// after an add, with the int8 operand -- to main's warp norms on a ptx
// target with 32 lanes, at decode and at a chunk's rows.
func TestNormsAreMains(t *testing.T) {
	prof := recorder.Profiles["cuda-sm70"]
	const eps = 1e-6
	for _, w := range []int{64, 256, 1024, 2048} {
		for _, rows := range []int{1, 32} {
			plan := nn.LayerPlan{RMSEps: eps, ActWin: 32, NEmbd: w}
			win := min(32, w)
			cases := []struct {
				kind      devKind
				ins, outs []int
				main      func() (*ir.Kernel, error)
			}{
				{dkNorm, []int{w}, []int{w}, func() (*ir.Kernel, error) {
					return mainkernels.RMSNormRowsWarp(w, rows, eps, false, false)
				}},
				{dkNormQuant, []int{w}, []int{w}, func() (*ir.Kernel, error) {
					return mainkernels.RMSNormQuantRowsWarp(w, rows, eps, false, false, win)
				}},
				{dkAddNorm, []int{w, w}, []int{w, w}, func() (*ir.Kernel, error) {
					return mainkernels.RMSNormRowsWarp(w, rows, eps, false, true)
				}},
				{dkAddNormQuant, []int{w, w}, []int{w, w}, func() (*ir.Kernel, error) {
					return mainkernels.RMSNormQuantRowsWarp(w, rows, eps, false, true, win)
				}},
			}
			for _, c := range cases {
				t.Run(fmt.Sprintf("%v_%dx%d", c.kind, rows, w), func(t *testing.T) {
					mk, err := c.main()
					if err != nil {
						t.Fatal(err)
					}
					sameAsMain(t, []*ir.Kernel{mk}, opKernels(t, prof, devOp{Kind: c.kind}, c.ins, c.outs, true, plan, rows))
				})
			}
		}
	}
}
