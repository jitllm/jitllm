//go:build amd64 || arm64

package nn_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestAttnAccIntoSplitsBitIdentically: accumulating a window in pieces must give
// the same bits as accumulating it in one call. That is the precondition for a
// paged KV cache agreeing exactly with an unpaged one, rather than within a
// bound that could hide a page-boundary bug.
//
// It holds because EmitAttnAcc nests the output dimension outer and the
// position inner, so seeding the accumulators from Out reproduces the FMA order
// across a split. Against the violation (AttnAcc, which zeroes, for the second
// piece) it reads a delta the size of the first piece.
func TestAttnAccIntoSplitsBitIdentically(t *testing.T) {
	f := nn.NewJIT(256, 256, []quant.Type{quant.F32})
	defer f.Close()

	for _, hd := range []int{64, 128, 256, 12, 17} {
		for _, npos := range []int{16, 129, 512} {
			f.AddAttn(hd, hd, cpu.KVF32)
			r := rand.New(rand.NewSource(int64(hd*1000 + npos)))
			v := make([]float32, npos*hd)
			for i := range v {
				v[i] = float32(r.NormFloat64())
			}
			att := make([]float32, npos)
			for i := range att {
				att[i] = float32(r.Float64())
			}

			whole := make([]float32, hd)
			f.AttnAcc(whole, v, att, npos)

			// Split at every page size that divides the window sensibly, plus a
			// ragged one -- a page walk's last piece is short by construction.
			tried := 0
			for _, P := range []int{8, 16, 64, 100} {
				if P >= npos {
					continue // no boundary to cross; the guard below would fire
				}
				tried++
				split := make([]float32, hd)
				ran := 0
				for lo := 0; lo < npos; lo += P {
					n := min(P, npos-lo)
					if lo == 0 {
						f.AttnAcc(split, v[lo*hd:], att[lo:], n)
					} else {
						f.AttnAccInto(split, v[lo*hd:], att[lo:], n)
					}
					ran++
				}
				// A split that never crossed a boundary would pass vacuously.
				if ran < 2 {
					t.Fatalf("hd=%d npos=%d P=%d: %d call(s) -- the gate never split",
						hd, npos, P, ran)
				}
				for i := range whole {
					if whole[i] != split[i] {
						t.Fatalf("hd=%d npos=%d P=%d: element %d is %v split and %v whole "+
							"(delta %g) -- the page walk does not reproduce the contiguous call",
							hd, npos, P, i, split[i], whole[i],
							math.Abs(float64(split[i]-whole[i])))
					}
				}
			}
			if tried == 0 {
				t.Fatalf("hd=%d npos=%d: no page size split the window", hd, npos)
			}
		}
	}
}

// TestAttnAcc2IntoSplitsBitIdentically is the paired twin of the gate above.
// The paged walk falls back silently to two single-head walks when the paired
// kernel is not selected, so a model-level gate cannot see AttnAcc2Into being
// wrong; the equality is asserted on the kernel.
func TestAttnAcc2IntoSplitsBitIdentically(t *testing.T) {
	f := nn.NewJIT(256, 256, []quant.Type{quant.F32})
	defer f.Close()

	ran := 0
	for _, hd := range []int{64, 128, 256, 12, 17} {
		for _, npos := range []int{16, 129, 512} {
			f.AddAttn(hd, hd, cpu.KVF32)
			r := rand.New(rand.NewSource(int64(hd*7919 + npos)))
			v := make([]float32, npos*hd)
			for i := range v {
				v[i] = float32(r.NormFloat64())
			}
			a0, a1 := make([]float32, npos), make([]float32, npos)
			for i := range a0 {
				a0[i], a1[i] = float32(r.Float64()), float32(r.Float64())
			}

			w0, w1 := make([]float32, hd), make([]float32, hd)
			if !f.AttnAcc2(w0, w1, v, a0, a1, npos) {
				continue // no paired kernel for this shape on this build
			}
			for _, P := range []int{8, 16, 64, 100} {
				if P >= npos {
					continue
				}
				s0, s1 := make([]float32, hd), make([]float32, hd)
				calls := 0
				for lo := 0; lo < npos; lo += P {
					n := min(P, npos-lo)
					var ok bool
					if lo == 0 {
						ok = f.AttnAcc2(s0, s1, v[lo*hd:], a0[lo:], a1[lo:], n)
					} else {
						ok = f.AttnAcc2Into(s0, s1, v[lo*hd:], a0[lo:], a1[lo:], n)
					}
					if !ok {
						t.Fatalf("hd=%d npos=%d P=%d: declined at lo=%d", hd, npos, P, lo)
					}
					calls++
				}
				if calls < 2 {
					t.Fatalf("hd=%d npos=%d P=%d: %d call(s) -- the gate never split",
						hd, npos, P, calls)
				}
				for i := range w0 {
					if s0[i] != w0[i] || s1[i] != w1[i] {
						t.Fatalf("hd=%d npos=%d P=%d: element %d is (%v,%v) split and "+
							"(%v,%v) whole (delta %g, %g) -- the paired page walk does not "+
							"reproduce the contiguous call",
							hd, npos, P, i, s0[i], s1[i], w0[i], w1[i],
							math.Abs(float64(s0[i]-w0[i])), math.Abs(float64(s1[i]-w1[i])))
					}
				}
				ran++
			}
		}
	}
	// If the paired kernel exists nowhere, say so rather than passing.
	if ran == 0 {
		t.Fatal("no paired accumulate ran on any shape: AttnAcc2 declined everywhere, so " +
			"this gate asserted nothing about AttnAcc2Into")
	}
	t.Logf("%d paired split(s) reproduced the contiguous call exactly", ran)
}
