//go:build amd64 && jitllmtest

package nn

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestSSEAttnUnderTheForcedTier is the attention family's nn gate: with the
// SSE tier forced, a JIT's attention comes from the SSE table and does what
// the model and the page walk rely on.
//
//   - AddAttn maps all six kernels as SSE-tier code (cpu.MappedByTier moves by
//     six on the SSE line and not at all on the AVX2 one), at both KV widths --
//     f16 = true is a basic kernel on this tier, never a refusal (RULE 8).
//   - KVWidthPaysOff declines the f16 cache: with no F16C the f16 kernels carry
//     a software widening and grow far past its 5% band, so an SSE host keeps
//     an f32 cache unless the caller forces f16.
//   - TestAttnAccIntoSplitsBitIdentically and its paired twin, through the nn
//     entry points, on the SSE kernels: a window accumulated in pieces is one
//     call over the whole window, bit for bit.
//   - AddAttnTiled provisions the tower's qt=8 kernel, and its scores agree
//     with the single-query kernel's.
func TestSSEAttnUnderTheForcedTier(t *testing.T) {
	old := cpu.ForceTierForTest(cpu.TierSSE)
	defer cpu.ForceTierForTest(old)
	if cpu.HostTier() != cpu.TierSSE {
		t.Fatalf("the tier could not be forced to SSE (host tier %v)", cpu.HostTier())
	}
	f := NewJIT(256, 256, []quant.Type{quant.F32})
	defer f.Close()
	if f.tier != cpu.TierSSE {
		t.Fatalf("a JIT built under the force records tier %v", f.tier)
	}

	for _, hd := range []int{64, 80, 128, 256} {
		if f.KVWidthPaysOff(hd, 4*hd) {
			t.Errorf("hd=%d: KVWidthPaysOff chose an f16 cache on the SSE tier, which has no F16C -- "+
				"it measured kernels this host does not run", hd)
		}
	}

	splits := 0
	for _, f16 := range []bool{false, true} {
		for _, hd := range []int{64, 128, 256, 12, 17} {
			before := cpu.MappedByTier()
			f.AddAttn(hd, hd, cpu.KVOf(f16))
			after := cpu.MappedByTier()
			if d := after[cpu.TierSSE] - before[cpu.TierSSE]; d != 6 {
				t.Fatalf("hd=%d f16=%v: AddAttn mapped %d SSE-tier kernels, want 6", hd, f16, d)
			}
			if d := after[cpu.TierAVX2] - before[cpu.TierAVX2]; d != 0 {
				t.Fatalf("hd=%d f16=%v: AddAttn mapped %d AVX2 kernels under the forced SSE tier", hd, f16, d)
			}
			for _, npos := range []int{16, 129, 512} {
				splits += attnSplitsSSE(t, f, hd, npos, f16)
			}
		}
	}
	t.Logf("%d split walks through nn reproduced the contiguous call exactly", splits)

	// The tower's tiled scores.
	const hd, stride, npos, qt = 64, 64, 37, 8
	f.AddAttn(hd, stride, cpu.KVF32)
	f.AddAttnTiled(hd, stride, stride, npos, qt, cpu.KVF32)
	if f.AttnTiledQt() != qt {
		t.Fatalf("AddAttnTiled(qt=%d) on the SSE tier provisioned qt=%d", qt, f.AttnTiledQt())
	}
	r := rand.New(rand.NewSource(7))
	k := make([]float32, npos*stride)
	q := make([]float32, qt*stride)
	for i := range k {
		k[i] = float32(r.NormFloat64())
	}
	for i := range q {
		q[i] = float32(r.NormFloat64())
	}
	tiled := make([]float32, qt*npos)
	if !f.AttnScoresTiled(tiled, k, q, npos) {
		t.Fatal("AttnScoresTiled declined although AttnTiledQt reports a kernel")
	}
	one := make([]float32, npos)
	for j := 0; j < qt; j++ {
		f.AttnScores(one, k, q[j*stride:], npos)
		for p := range one {
			got, want := float64(tiled[j*npos+p]), float64(one[p])
			if d := math.Abs(got - want); d > 1e-5*(math.Abs(want)+1) || math.IsNaN(d) {
				t.Fatalf("tiled query %d key %d: %v, the single-query kernel gives %v", j, p, got, want)
			}
		}
	}
}

// attnSplitsSSE runs one (hd, npos) window through f's single and paired
// accumulates whole and split at several page sizes, and fails unless every
// split gives the whole call's bits. It returns how many splits ran.
func attnSplitsSSE(t *testing.T, f *JIT, hd, npos int, f16 bool) int {
	t.Helper()
	r := rand.New(rand.NewSource(int64(hd*1000 + npos)))
	elem := 4
	if f16 {
		elem = 2
	}
	raw := make([]byte, npos*hd*elem+16)
	for i := 0; i < npos*hd; i++ {
		v := float32(r.NormFloat64())
		if f16 {
			*(*uint16)(unsafe.Pointer(&raw[2*i])) = quant.EncodeHalf(v)
		} else {
			*(*float32)(unsafe.Pointer(&raw[4*i])) = v
		}
	}
	// The cache at position lo, as the []float32 nn's entry points take (they
	// only ever hand its first element's address to the kernel).
	cache := func(lo int) []float32 {
		return unsafe.Slice((*float32)(unsafe.Pointer(&raw[lo*hd*elem])), 1)
	}
	a0, a1 := make([]float32, npos), make([]float32, npos)
	for i := range a0 {
		a0[i], a1[i] = float32(r.Float64()), float32(r.Float64())
	}
	whole, w0, w1 := make([]float32, hd), make([]float32, hd), make([]float32, hd)
	f.AttnAcc(whole, cache(0), a0, npos)
	if !f.AttnAcc2(w0, w1, cache(0), a0, a1, npos) {
		t.Fatalf("hd=%d f16=%v: the paired accumulate declined on the SSE tier", hd, f16)
	}
	ran := 0
	for _, P := range []int{8, 16, 64, 100} {
		if P >= npos {
			continue
		}
		s, p0, p1 := make([]float32, hd), make([]float32, hd), make([]float32, hd)
		calls := 0
		for lo := 0; lo < npos; lo += P {
			n := min(P, npos-lo)
			ok := true
			if lo == 0 {
				f.AttnAcc(s, cache(lo), a0[lo:], n)
				ok = f.AttnAcc2(p0, p1, cache(lo), a0[lo:], a1[lo:], n)
			} else {
				f.AttnAccInto(s, cache(lo), a0[lo:], n)
				ok = f.AttnAcc2Into(p0, p1, cache(lo), a0[lo:], a1[lo:], n)
			}
			if !ok {
				t.Fatalf("hd=%d f16=%v P=%d: the paired accumulate declined at lo=%d", hd, f16, P, lo)
			}
			calls++
		}
		if calls < 2 {
			t.Fatalf("hd=%d npos=%d P=%d: %d call(s) -- the gate never split", hd, npos, P, calls)
		}
		for i := range whole {
			if s[i] != whole[i] || p0[i] != w0[i] || p1[i] != w1[i] {
				t.Fatalf("hd=%d npos=%d f16=%v P=%d: element %d split (%v | %v, %v) whole (%v | %v, %v) "+
					"-- the page walk does not reproduce the contiguous call",
					hd, npos, f16, P, i, s[i], p0[i], p1[i], whole[i], w0[i], w1[i])
			}
		}
		if whole[0] != w0[0] {
			t.Fatalf("hd=%d npos=%d f16=%v: AttnAcc %v and AttnAcc2's head 0 %v differ -- the pair "+
				"must be two single calls", hd, npos, f16, whole[0], w0[0])
		}
		ran++
	}
	return ran
}
