//go:build amd64 || arm64

package nn

import (
	"math"
	"math/rand"
	"testing"
)

// The router entry point's gate. jit_old/cpu/moetopk_test.go gates the KERNELS
// over the whole shape matrix and every tie shape; this gates the SEAM -- that
// MoETopK32JIT puts the right pointers in the right Args fields, that it sizes
// its scratch the way NewMoERoute does, and that it refuses rather than
// half-filling a route it cannot serve.
//
// A right kernel wired wrongly is invisible to a kernel gate: swapping Out and
// Out2 (the two weight orders) produces a fluent, wrong model.

// moeTopKRef is the router's original four Go loops, kept as the reference.
func moeTopKRef(p []float32, k int, norm bool) (sel, ord []int32, wt, ow []float32, sum float32) {
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
			if best < 0 || p[e] > p[best] {
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
	sum = 1
	if norm {
		sum = 0
		for _, e := range selI {
			sum += p[e]
		}
	}
	sel, ord = make([]int32, k), make([]int32, k)
	wt, ow = make([]float32, k), make([]float32, k)
	for i, e := range selI {
		sel[i], wt[i] = int32(e), p[e]/sum
	}
	for i, e := range ordI {
		ord[i], ow[i] = int32(e), p[e]/sum
	}
	return
}

func TestMoETopK32JITMatchesTheGoLoops(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, n := range []int{4, 8, 63, 64, 128, 512} {
		for _, k := range []int{1, 2, 4, 8, 10} {
			if k > n {
				continue
			}
			for _, norm := range []bool{true, false} {
				p := make([]float32, n)
				for i := range p {
					p[i] = rng.Float32()
				}
				// The second half ties with the first, pair for pair, so the
				// tie rule is exercised at every lane and vector boundary.
				for i := n / 2; i < n; i++ {
					p[i] = p[i-n/2]
				}
				r := NewMoERoute(k)
				if !MoETopK32JIT(p, norm, r) {
					t.Fatalf("n=%d k=%d norm=%v: no kernel", n, k, norm)
				}
				sel, ord, wt, ow, sum := moeTopKRef(p, k, norm)
				for i := 0; i < k; i++ {
					if r.Sel[i] != sel[i] || r.Ord[i] != ord[i] {
						t.Fatalf("n=%d k=%d norm=%v: sel %v ord %v, want %v %v",
							n, k, norm, r.Sel, r.Ord, sel, ord)
					}
					if math.Float32bits(r.Wt[i]) != math.Float32bits(wt[i]) {
						t.Fatalf("n=%d k=%d norm=%v: wt[%d] = %v, want %v", n, k, norm, i, r.Wt[i], wt[i])
					}
					if math.Float32bits(r.OWt[i]) != math.Float32bits(ow[i]) {
						t.Fatalf("n=%d k=%d norm=%v: ow[%d] = %v, want %v", n, k, norm, i, r.OWt[i], ow[i])
					}
				}
				if math.Float32bits(r.Sum[0]) != math.Float32bits(sum) {
					t.Fatalf("n=%d k=%d norm=%v: sum = %v, want %v", n, k, norm, r.Sum[0], sum)
				}
			}
		}
	}
}

// TestMoETopK32JITRefusesRatherThanHalfFilling: k > n has no answer -- the Go
// loop it replaces wrote -1 into sel and the caller then indexed p with it --
// so the entry point says false and the caller returns an error. It must not
// leave a route that reads like a selection.
func TestMoETopK32JITRefusesRatherThanHalfFilling(t *testing.T) {
	r := NewMoERoute(8)
	for i := range r.Sel {
		r.Sel[i] = -7
	}
	if MoETopK32JIT(make([]float32, 4), true, r) {
		t.Fatal("k = 8 over 4 experts was served")
	}
	for i := range r.Sel {
		if r.Sel[i] != -7 {
			t.Fatalf("the refused call wrote sel[%d] = %d", i, r.Sel[i])
		}
	}
	if MoETopK32JIT(nil, true, r) {
		t.Fatal("an empty probability vector was served")
	}
}

// TestMoETopK32JITPanicsOnAShortRoute is the contract violation: a route whose
// buffers disagree with its Sel length is a caller bug, and the kernel would
// otherwise write past one of them.
func TestMoETopK32JITPanicsOnAShortRoute(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("a route with a short scratch was accepted")
		}
	}()
	r := NewMoERoute(8)
	r.Scr = r.Scr[:1]
	MoETopK32JIT(make([]float32, 64), true, r)
}
