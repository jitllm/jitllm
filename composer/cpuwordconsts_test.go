package composer

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUWordConstantsReload executes on the CPU a row of words
// through sixteen adds of sixteen constants, summed pairwise: more constants
// held in vectors than the vector registers left beside the sums, so the
// binding reads some of them again where they are used. It forms one kernel,
// each word exact at ragged widths, a guard of poisoned words past the row.
// An integer constant held as an immediate cannot be read again, and the
// binding refused the map for register pressure.
func TestComposedCPUWordConstantsReload(t *testing.T) {
	gateComposedCPUWordConstantsReload(t, hostArm())
}

func gateComposedCPUWordConstantsReload(t *testing.T, cpuOn cpuArm) {
	const terms, guard, poison = 16, 16, 0x7FC00000
	U := ir.U32
	consts := make([]uint32, terms)
	var all uint32
	for j := range consts {
		consts[j] = 0x9E3779B9 * uint32(j+1)
		all += consts[j]
	}
	for _, W := range []int{1, 7, 8, 9, 31, 64, 133} {
		t.Run(fmt.Sprint(W), func(t *testing.T) {
			var x exprs
			r, i := x.coord(0), x.coord(1)
			w := x.at(0, []int{1, W}, r, i)
			sums := make([]int, terms)
			for j := range sums {
				sums[j] = x.op(OpAdd, U, w, x.u(consts[j]))
			}
			for len(sums) > 1 {
				var next []int
				for j := 0; j+1 < len(sums); j += 2 {
					next = append(next, x.op(OpAdd, U, sums[j], sums[j+1]))
				}
				sums = next
			}
			out := Operand{Kind: OperandValue, Value: 1, Name: "o", Elem: U, Out: true}
			p := Prim{Kind: PrimMap, Shape: []int{1, W}, Expr: x.e,
				In:     []Operand{{Kind: OperandValue, Value: 0, Name: "w", Elem: U}},
				Stores: []PrimStore{{Out: out, Ax: []int{r, i}, Ext: []int{1, W}, Value: sums[0]}}}
			k, err := generate(p)
			if err != nil {
				t.Fatal(err)
			}
			k.Apart = true
			dev := cpuDevice(cpuOn)
			reg := Default()
			ks, err := reg.Profiles[DeviceCPU].on(dev).run([]*Kernel{k}, dev)
			if err != nil {
				t.Fatal(err)
			}
			pcs, err := reg.Lower(ks, dev)
			if err != nil {
				t.Fatal(err)
			}
			if len(pcs) != 1 {
				t.Fatalf("%d kernels, want one", len(pcs))
			}
			words := make([]uint32, W)
			for j := range words {
				words[j] = 0xFFFFFFF0 + uint32(j)*0x01000193
			}
			got := make([]uint32, W+guard)
			for j := range got {
				got[j] = poison
			}
			g := &graph.Graph{Values: []graph.Value{{Name: "w", Width: W, Elem: graph.ElemU32}, {Name: "o", Width: W, Elem: graph.ElemU32}}}
			if err := runCPUCalls(cpuOn, []Code{pcs[0].Code}, g, map[string][]uint32{"w": words, "o": got}); err != nil {
				t.Fatal(err)
			}
			for j, v := range words {
				if want := terms*v + all; got[j] != want {
					t.Fatalf("word %d of %d is %#x, want %#x", j, W, got[j], want)
				}
			}
			for j := W; j < W+guard; j++ {
				if got[j] != poison {
					t.Fatalf("the code wrote past the row, at %d", j)
				}
			}
		})
	}
}
