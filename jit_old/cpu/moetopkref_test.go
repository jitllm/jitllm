//go:build amd64 || arm64

package cpu

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// The MoE router top-k gate's oracle and its fixtures, shared by the amd64 and
// arm64 arms (moetopk_test.go and moetopka64_test.go) so the fixture
// table is written once.

// moeFault is a deliberate defect in the oracle: the three ways this kernel
// could be wrong while still looking right, which only an equality comparison
// sees.
type moeFault int

const (
	moeFaultNone moeFault = iota
	// The fold keeps the last of a tie instead of the first. Invisible on any
	// input without equal probabilities, and a divergence for a whole
	// generation on a freshly converted router that emits identical logits.
	moeFaultLastOfTie
	// The divisor is summed in ascending id order instead of selection order:
	// the same k terms, a different float32 parenthesisation.
	moeFaultAscendingSum
	// ord is left in selection order. The right experts with the right
	// weights, accumulated into the residual in the wrong order.
	moeFaultOrdInSelectionOrder
)

func (f moeFault) String() string {
	switch f {
	case moeFaultNone:
		return "none"
	case moeFaultLastOfTie:
		return "a fold that keeps the LAST of a tie"
	case moeFaultAscendingSum:
		return "the divisor summed in ASCENDING id order"
	case moeFaultOrdInSelectionOrder:
		return "ord left in SELECTION order"
	}
	return "?"
}

// moeRoute is one call's outputs, from either the oracle or a kernel.
type moeRoute struct {
	sel, ord []int32
	wt, ow   []float32
	sum      float32
}

// moeTopKGo is engine/model/moe.go's four loops as they stood before this kernel,
// transcribed rather than re-derived: the selection (moeTopK and its `used`
// helper), the renormalising sum, the ascending insertion sort, and the
// per-expert division. f injects one defect.
func moeTopKGo(p []float32, k int, norm bool, f moeFault) moeRoute {
	selI := make([]int, k)
	for i := range selI {
		best := -1
		for e := range p {
			used := false
			for _, v := range selI[:i] {
				if v == e {
					used = true
					break
				}
			}
			if used {
				continue
			}
			// The tie rule: `>` scanning e ascending keeps the lowest index;
			// `>=` keeps the last, which is the violation.
			if best < 0 || p[e] > p[best] ||
				(f == moeFaultLastOfTie && p[e] == p[best]) {
				best = e
			}
		}
		selI[i] = best
	}
	ordI := make([]int, k)
	copy(ordI, selI)
	for i := 1; i < k; i++ {
		for j := i; j > 0 && ordI[j] < ordI[j-1]; j-- {
			ordI[j], ordI[j-1] = ordI[j-1], ordI[j]
		}
	}
	sum := float32(1)
	if norm {
		sum = 0
		src := selI
		if f == moeFaultAscendingSum {
			src = ordI
		}
		for _, e := range src {
			sum += p[e]
		}
		// The floor the kernel adds, so an all-underflowed selection is not
		// 0/0. See oracle.MoEDivergences for why it is DeepSeek's addition
		// and not llama.cpp's clamp.
		sum += MoESumFloor()
	}
	if f == moeFaultOrdInSelectionOrder {
		copy(ordI, selI)
	}
	r := moeRoute{
		sel: make([]int32, k), ord: make([]int32, k),
		wt: make([]float32, k), ow: make([]float32, k), sum: sum,
	}
	for i, e := range selI {
		r.sel[i], r.wt[i] = int32(e), p[e]/sum
	}
	for i, e := range ordI {
		r.ord[i], r.ow[i] = int32(e), p[e]/sum
	}
	return r
}

// moeDiff reports the first place two routes disagree, "" when they are equal.
// The weights and the divisor are compared bitwise: a last-bit difference in a
// renormalised weight propagates through every later layer.
func moeDiff(want, got moeRoute) string {
	for i := range want.sel {
		if want.sel[i] != got.sel[i] {
			return fmt.Sprintf("sel[%d] = %d, want %d (sel %v, want %v)",
				i, got.sel[i], want.sel[i], got.sel, want.sel)
		}
	}
	for i := range want.ord {
		if want.ord[i] != got.ord[i] {
			return fmt.Sprintf("ord[%d] = %d, want %d (ord %v, want %v)",
				i, got.ord[i], want.ord[i], got.ord, want.ord)
		}
	}
	if a, b := math.Float32bits(want.sum), math.Float32bits(got.sum); a != b {
		return fmt.Sprintf("sum = %v (%#08x), want %v (%#08x)", got.sum, b, want.sum, a)
	}
	for i := range want.wt {
		if a, b := math.Float32bits(want.wt[i]), math.Float32bits(got.wt[i]); a != b {
			return fmt.Sprintf("wt[%d] = %v (%#08x), want %v (%#08x)", i, got.wt[i], b, want.wt[i], a)
		}
	}
	for i := range want.ow {
		if a, b := math.Float32bits(want.ow[i]), math.Float32bits(got.ow[i]); a != b {
			return fmt.Sprintf("ow[%d] = %v (%#08x), want %v (%#08x)", i, got.ow[i], b, want.ow[i], a)
		}
	}
	return ""
}

var moeTestConsts = MoETopKConsts()

// moeCall runs a mapped kernel over p, in the ABI nn uses.
func moeCall(c *Code, p []float32, k int) moeRoute {
	r := moeRoute{
		sel: make([]int32, k), ord: make([]int32, k),
		wt: make([]float32, k), ow: make([]float32, k),
	}
	sum := make([]float32, 1)
	scr := make([]float32, MoETopKScratch(k))
	// The engine reuses scratch across calls, so it is poisoned here: a kernel
	// that read a pad lane it did not write would pass on a zeroed buffer.
	for i := range scr {
		scr[i] = math.Float32frombits(0xDEADBEEF)
	}
	args := Args{
		Q32:      &p[0],
		Scr:      (*byte)(unsafe.Pointer(&moeTestConsts[0])),
		Scratch:  (*byte)(unsafe.Pointer(&scr[0])),
		ASum:     &r.sel[0],
		AHalfSum: &r.ord[0],
		Out:      &r.wt[0],
		Out2:     &r.ow[0],
		AScale:   &sum[0],
	}
	c.Call(&args)
	r.sum = sum[0]
	return r
}

// moeShapes is the (n, k) matrix. The expert counts are the ones that ship --
// 4 on Qwen3-MoE-4x0.6B, 32 on gpt-oss, 64 on olmoe, 128 on Qwen3-30B-A3B, 512
// on Qwen3-Next-80B -- plus 8, and 63/65 either side of a vector boundary,
// because the vector width divides none of the real ones either.
var moeShapes = func() [][2]int {
	var out [][2]int
	for _, n := range []int{4, 8, 32, 63, 64, 65, 128, 512} {
		for _, k := range []int{1, 2, 4, 6, 8, 10} {
			if k > n {
				// k > n is not a configuration; the emitters refuse it
				// (TestMoETopKRefusesAShapeItCannotServe).
				continue
			}
			out = append(out, [2]int{n, k})
		}
	}
	return out
}()

// moeInputs is the probability vectors for one expert count.
//
// A lane-parallel selection can only go wrong about ties, differently by where
// the tie sits, so two-way ties are placed at 0/1, at the lane boundary, at the
// vector boundary and at the last two elements, and "all equal" makes every
// pair a tie.
func moeInputs(n int) []struct {
	name string
	p    []float32
} {
	rng := rand.New(rand.NewSource(int64(n)*7919 + 13))
	mk := func(f func(i int) float32) []float32 {
		p := make([]float32, n)
		for i := range p {
			p[i] = f(i)
		}
		return p
	}
	var out []struct {
		name string
		p    []float32
	}
	add := func(name string, p []float32) {
		out = append(out, struct {
			name string
			p    []float32
		}{name, p})
	}
	add("random", mk(func(int) float32 { return rng.Float32() }))
	add("allEqual", mk(func(int) float32 { return 0.25 }))
	add("allZero", mk(func(int) float32 { return 0 }))
	add("allNegative", mk(func(int) float32 { return -1 - rng.Float32() }))
	add("ascending", mk(func(i int) float32 { return float32(i) * 1e-3 }))
	add("descending", mk(func(i int) float32 { return float32(n-i) * 1e-3 }))
	// Two values only: every element ties with half the vector, so the answer
	// is decided entirely by index order.
	add("twoValues", mk(func(i int) float32 {
		if i%2 == 0 {
			return 0.5
		}
		return 0.25
	}))
	// Pairwise ties at the boundaries a vector fold crosses.
	pairs := mk(func(i int) float32 { return rng.Float32() * 0.1 })
	for _, at := range []int{0, 3, 7, 8, 15, 31, 32, n - 2} {
		if at >= 0 && at+1 < n {
			pairs[at+1] = pairs[at]
		}
	}
	add("tiesAtBoundaries", pairs)
	// A softmax over random logits: what the engine actually hands the kernel,
	// with most of the mass on a few experts and the rest near zero.
	logits := mk(func(int) float32 { return rng.Float32()*8 - 4 })
	sm := make([]float32, n)
	mx := float32(math.Inf(-1))
	for _, v := range logits {
		if v > mx {
			mx = v
		}
	}
	var s float32
	for i, v := range logits {
		sm[i] = float32(math.Exp(float64(v - mx)))
		s += sm[i]
	}
	for i := range sm {
		sm[i] /= s
	}
	add("softmax", sm)
	// A softmax whose tail underflows to +0: the divisor is then a sum of a few
	// terms and several exact zeros, and the division of a zero by it must come
	// out +0 rather than a NaN.
	flat := mk(func(i int) float32 {
		if i < 3 {
			return 0.3
		}
		return 0
	})
	add("threeHotRestZero", flat)
	add("infinities", mk(func(i int) float32 {
		switch i {
		case 1:
			return float32(math.Inf(1))
		case 2:
			return float32(math.Inf(-1))
		}
		return rng.Float32()
	}))
	return out
}

// moeCheckAllEqual is the assertion the "every element ties" fixture exists
// for, stated separately because it is checkable without the oracle: with all
// probabilities equal the first k indices must win, in order.
func moeCheckAllEqual(t *testing.T, what string, r moeRoute) {
	t.Helper()
	for i := range r.sel {
		if r.sel[i] != int32(i) || r.ord[i] != int32(i) {
			t.Errorf("%s: every probability equal and sel = %v, ord = %v; "+
				"want 0..%d in both -- the lowest index must win every tie",
				what, r.sel, r.ord, len(r.sel)-1)
			return
		}
	}
}
