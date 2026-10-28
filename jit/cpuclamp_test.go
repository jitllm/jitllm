package jit

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit/cpu"
)

// clampTable is a bounds table of n pairs: pair i at 2i, [lo, hi] with
// lo < hi, some bounds infinite (a matrix whose source states none).
func clampTable(rng *rand.Rand, n int) []float32 {
	t := make([]float32, 2*n)
	for i := range n {
		lo, hi := float32(-0.5-rng.Float64()), float32(0.5+rng.Float64())
		switch i % 4 {
		case 1:
			lo = float32(math.Inf(-1))
		case 2:
			hi = float32(math.Inf(1))
		}
		t[2*i], t[2*i+1] = lo, hi
	}
	return t
}

// clampGraph is OpClamp over w elements: to [-bound, bound], or with a
// table of k words to the pair at off.
func clampGraph(w, k, off int, bound float64) *graph.Graph {
	b := graph.New()
	x := b.In("x", w)
	var bind []graph.Binding
	if k > 0 {
		bind = []graph.Binding{{Role: jlm.RoleVClamp, K: k}}
	}
	b.Out(b.Emit(graph.OpClamp, []graph.ValueID{x}, []graph.Result{{Name: "x.clamp", Width: w}}, bind,
		graph.Attrs{Offset: off, Bound: bound}, nil, nil)[0])
	return b.G
}

// TestComposedCPUClamp executes the composer's clamp on the CPU, bounded by
// a table's pair (a clipped linear's, jlm.RoleVClamp) at every pair of a
// block's 28 words and by a symmetric bound, at ragged widths and several
// rows, each held to the oracle's clamp exactly with a guard of poisoned
// words after the result. The violation reads the pair one word over: the
// gate must see it.
func TestComposedCPUClamp(t *testing.T) {
	const guard, pairs = 64, 14
	rng := rand.New(rand.NewSource(29))
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}
	tab := clampTable(rng, pairs)
	nan := float32(math.NaN())
	right, seen := 0, 0
	for _, rows := range []int{1, 2, 3, 9} {
		for _, w := range []int{1, 2, 3, 7, 8, 9, 15, 16, 17, 33, 100, 257} {
			for _, off := range []int{-1, 0, 2, 12, 26} {
				name := fmt.Sprintf("%d rows/width %d/pair at %d", rows, w, off)
				k, at, bound := len(tab), off, 0.0
				if off < 0 {
					k, at, bound = 0, 0, 0.75
				}
				_, codes, err := r.Form(Request{Graph: clampGraph(w, k, at, bound),
					Formats: map[jlm.Role]jlm.Type{jlm.RoleVClamp: jlm.TypeF32},
					Step:    Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
				if err != nil {
					t.Errorf("%s: %v", name, err)
					continue
				}
				x := make([]float32, rows*w)
				for i := range x {
					x[i] = float32(3 * rng.NormFloat64())
				}
				lo, hi := float32(-bound), float32(bound)
				if k > 0 {
					lo, hi = tab[at], tab[at+1]
				}
				want := make([]float32, len(x))
				oracle.Clamp32(want, x, lo, hi)
				run := func(tab []float32) ([]float32, error) {
					in := append(append([]float32(nil), x...), make([]float32, guard)...)
					for i := len(x); i < len(in); i++ {
						in[i] = nan
					}
					out, err := runCPUVectors(codes, in, len(x)+guard, [3][]uint32{}, map[jlm.Role][]float32{jlm.RoleVClamp: tab})
					if err != nil {
						return nil, err
					}
					if writesInput(codes) {
						out = in
					}
					for i := len(x); i < len(out); i++ {
						if !math.IsNaN(float64(out[i])) {
							return nil, fmt.Errorf("the guard word %d past the result was written", i-len(x))
						}
					}
					return out[:len(x)], nil
				}
				got, err := run(tab)
				if err != nil {
					t.Errorf("%s: %v", name, err)
					continue
				}
				bad := 0
				for i := range want {
					if got[i] != want[i] {
						bad++
					}
				}
				if bad != 0 {
					t.Errorf("%s: %d of %d elements differ from the oracle's clamp", name, bad, len(want))
					continue
				}
				right++
				if k == 0 {
					continue
				}
				moved, err := run(append(append([]float32(nil), tab[1:]...), tab[0]))
				if err != nil {
					t.Errorf("%s: the violation did not run: %v", name, err)
					continue
				}
				differ := false
				for i := range want {
					differ = differ || moved[i] != want[i]
				}
				if differ {
					seen++
				} else if len(want) >= 16 {
					t.Errorf("%s: the table read one word over clamps the same %d elements: the gate cannot see it", name, len(want))
				}
			}
		}
	}
	t.Logf("%d clamps exact against the oracle; the table read one word over seen at %d", right, seen)
	if right == 0 || seen == 0 {
		t.Fatal("no clamp held to the oracle, or no violation seen: this gate proved nothing")
	}
}
