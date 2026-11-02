package composer

import (
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/composer/ir"
)

// TestCoarsenClassesAreTheUnitsValues holds Coarsen's classes of a unit's
// index expressions to the expressions' values at every unit: two nodes
// share a class exactly where they agree at every one, and a node is a
// value exactly where no division meets a zero. The index expressions a
// chain's rows decode -- (u*a+j)/n, its remainder, a shift and a mask --
// stay P words a period, whatever the rows.
func TestCoarsenClassesAreTheUnitsValues(t *testing.T) {
	ops := []Op{OpAdd, OpSub, OpMul, OpDiv, OpRem, OpMin, OpMax, OpShl, OpShr, OpAnd}
	rng := rand.New(rand.NewSource(1))
	for _, shape := range [][2]int{{1, 96}, {2, 48}, {7, 12}, {64, 24}, {512, 8}} {
		R, P := shape[0], shape[1]
		k := &coarse{m: &Kernel{}, T: R * P, R: R, P: P, values: map[int]*uval{}, classOf: map[int]int{}, classes: map[string]int{}}
		k.u = k.m.add(Node{Op: OpCoord, Type: ir.U32})
		brute := map[int][]uint32{}
		u := make([]uint32, k.T)
		for i := range u {
			u[i] = uint32(i)
		}
		brute[k.u] = u
		konst := func(v int) int {
			i := k.m.u32(v)
			b := make([]uint32, k.T)
			for t := range b {
				b[t] = uint32(v)
			}
			brute[i] = b
			return i
		}
		eval := func(op Op, a, b []uint32) []uint32 {
			if a == nil || b == nil {
				return nil
			}
			out := make([]uint32, len(a))
			for t := range out {
				x, y := a[t], b[t]
				switch op {
				case OpAdd:
					out[t] = x + y
				case OpSub:
					out[t] = x - y
				case OpMul:
					out[t] = x * y
				case OpDiv, OpRem:
					if y == 0 {
						return nil
					}
					if op == OpDiv {
						out[t] = x / y
					} else {
						out[t] = x % y
					}
				case OpMin:
					out[t] = min(x, y)
				case OpMax:
					out[t] = max(x, y)
				case OpShl:
					out[t] = x << (y & 31)
				case OpShr:
					out[t] = x >> (y & 31)
				case OpAnd:
					out[t] = x & y
				}
			}
			return out
		}
		bin := func(op Op, a, b int) int {
			i := k.m.bin(op, ir.U32, a, b)
			brute[i] = eval(op, brute[a], brute[b])
			return i
		}
		// The decodes a chain makes of a member's elements.
		nodes := []int{k.u}
		var decodes []int
		for _, a := range []int{1, 2, 4} {
			for j := 0; j < a; j++ {
				idx := bin(OpAdd, bin(OpMul, k.u, konst(a)), konst(j))
				// Periodic where the divisor divides a row's elements, as a
				// chain's inner extents do; the others are values all the same.
				for _, n := range []int{P * a, P * a / 2, 4, 8} {
					if n > 0 {
						d, r := bin(OpDiv, idx, konst(n)), bin(OpRem, idx, konst(n))
						nodes = append(nodes, d, r)
						if P*a%n == 0 {
							decodes = append(decodes, d, r)
						}
					}
				}
				sh, and := bin(OpShr, idx, konst(2)), bin(OpAnd, idx, konst(3))
				nodes = append(nodes, sh, and, bin(OpShl, idx, konst(1)))
				if P*a%4 == 0 {
					decodes = append(decodes, sh, and)
				}
			}
		}
		for range 400 {
			a := nodes[rng.Intn(len(nodes))]
			b := nodes[rng.Intn(len(nodes))]
			if rng.Intn(2) == 0 {
				b = konst([]int{0, 1, 2, 3, 4, 7, 8, 31, 32, P, 2 * P, P * R}[rng.Intn(12)])
			}
			nodes = append(nodes, bin(ops[rng.Intn(len(ops))], a, b))
		}
		seen := map[int][]uint32{}
		for _, i := range nodes {
			id, ok := k.class(i)
			if ok != (brute[i] != nil) {
				t.Fatalf("R %d, P %d: node %d is a value %v, its units say %v", R, P, i, ok, brute[i] != nil)
			}
			if !ok {
				continue
			}
			if want, ok := seen[id]; ok && !slices.Equal(want, brute[i]) {
				t.Fatalf("R %d, P %d: node %d shares class %d with a node of other values", R, P, i, id)
			}
			seen[id] = brute[i]
		}
		for id, v := range seen {
			for id2, w := range seen {
				if id != id2 && slices.Equal(v, w) {
					t.Fatalf("R %d, P %d: classes %d and %d are one function", R, P, id, id2)
				}
			}
		}
		for _, i := range decodes {
			if v, ok := k.value(i); ok && v.full != nil {
				t.Fatalf("R %d, P %d: a decode of the unit (node %d, %v) holds every unit's word", R, P, i, k.m.Nodes[i].Op)
			}
		}
		t.Logf("R %d, P %d: %d nodes, %d classes", R, P, len(nodes), len(seen))
	}
}
