//go:build amd64 || arm64

package nn

import (
	"fmt"
	"hash/fnv"
	"maps"
	"slices"
	"time"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// The fused packed kernel's software-prefetch distance is measured per host,
// on real decode tokens, and cached beside the pack width. On a high-latency
// part the hardware prefetcher re-trains at every sub-block and the kernel is
// bound by misses in flight, so a prefetch pays a lot; on a host whose
// prefetcher keeps up it costs an instruction per word for nothing.
//
// prefetchDistances is the ladder, in row groups; the duel starts from the
// first rung and tries every other against it (fpfTuner sweeps).
// cpu.FusedAheadWords is a different kind of rung: eight payload words ahead in
// the same row group, reaching the next sub-block's lines. It is first because
// the incumbent is what a process decodes at until the duel settles, and an
// unsettled duel is not cached, so short sessions should start on a prefetch;
// a host where none wins still reaches it past the first-step margin.
var prefetchDistances = []int{cpu.FusedAheadWords, 2, 4, 8, 0}

// fpfTuner wraps a duelTuner, whose values must be >= 1 to mean "chosen", so
// every distance travels as distance+1.
type fpfTuner struct {
	d    *duelTuner
	skip int
	t0   time.Time
	cur  int // the distance packedFused currently holds
	done bool
	// discard drops the token under way (DiscardToken).
	discard bool
}

func (f *JIT) newFPFTuner() *fpfTuner {
	// The key carries a fingerprint of the kernels being dueled: hostSig hashes
	// the row-major emitter, and a cached distance must not outlive an edit to
	// the fused one it was measured against.
	fp := fnv.New64a()
	for _, q := range slices.Sorted(maps.Keys(f.packedFused)) {
		if b, err := f.em.PackedFusedAhead(q, cpu.FusedAheadWords); err == nil {
			fp.Write(b)
		}
	}
	if fp.Sum64() == fnv.New64a().Sum64() {
		return nil // no fused kernel, or this tier's has no prefetch form
	}
	cands := make([]int, len(prefetchDistances))
	for i, d := range prefetchDistances {
		cands[i] = d + 1
	}
	pin := 0
	if f.cfg.FusedPrefetch >= 0 {
		pin = f.cfg.FusedPrefetch + 1
	}
	mode := f.cfg.Tune
	if f.distPick() {
		// The shapes choose (mvpick.go); this only holds the default the
		// batched paths start from, the first rung, until a shape settles.
		mode = TuneOff
	}
	d := newDuelTuner("fused prefetch (row groups ahead)", pin,
		tuneKey(fmt.Sprintf("fpf:%016x", fp.Sum64()), f.keyShapes, f.pool.Max()), cands, mode, f.cfg.Quiet)
	// More rounds and two margins: the row-group distances issue the same
	// instructions, so 0.5% decides between them, but leaving the first rung
	// keeps the 2% margin so noise cannot pick a loser where every distance is
	// within 1% (TestTunerOnThisHardware).
	d.rounds, d.margin, d.firstMargin, d.offset = 24, 1.005, tuneMargin2, 1
	// Every rung is tried: the rungs are different prefetches, not a climb,
	// and one losing says nothing about the next.
	d.sweep = true
	return &fpfTuner{d: d}
}

// fusedKernels is the fused kernels at prefetch distance d, emitted the first
// time. A type whose emitter refuses is absent.
func (f *JIT) fusedKernels(d int) map[quant.Type]*cpu.Code {
	if f.fusedAt[d] == nil {
		m := map[quant.Type]*cpu.Code{}
		for q := range f.fusedAt[0] {
			if b, err := f.em.PackedFusedAhead(q, d); err == nil {
				if c, err := cpu.MapNamed(b, q.String()+"_packed_fused"); err == nil {
					m[q] = c
				}
			}
		}
		f.fusedAt[d] = m
	}
	return f.fusedAt[d]
}

// useFused points packedFused at the kernels for distance d. A type whose
// emitter refuses keeps what it had.
func (f *JIT) useFused(d int) {
	t := f.fpf
	if d == t.cur {
		return
	}
	for q, c := range f.fusedKernels(d) {
		f.packedFused[q] = c
	}
	t.cur = d
}

// settleDist records a shape's chosen distance. The largest shape of a type
// sets packedFused, which the batched paths (MatVecPackedMulti, the prefill
// GEMM's fused fallback) run at.
func (f *JIT) settleDist(t quant.Type, size, d int) {
	if f.distSize == nil {
		f.distSize = map[quant.Type]int{}
	}
	if size <= f.distSize[t] {
		return
	}
	if c := f.fusedKernels(d)[t]; c != nil {
		f.distSize[t] = size
		f.packedFused[t] = c
	}
}

// distPick reports whether MatVecPacked chooses the fused kernel's prefetch
// distance per shape (mvpick.go) instead of the whole-token duel. The duel
// compares token rates, which on a two-socket host could not resolve a
// 4-6% difference: two identical runs settled on different distances, and the
// worse one cost several percent of decode. A shape's own calls,
// timed in place, are tight to a few percent over thousands of samples. A
// tier with no prefetch form (SSE) has nothing to pick between.
func (f *JIT) distPick() bool {
	return f.tier != cpu.TierNEON && f.aheadForm && f.cfg.FusedPrefetch < 0 && f.cfg.MatVecPick == 0 && f.cfg.Tune != TuneOff
}

// fpfStart runs only once the pack width has settled: two duels at once would
// each measure the other.
func (f *JIT) fpfStart() {
	t := f.fpf
	if t == nil || t.done || f.tuner.on {
		return
	}
	if !t.d.on {
		t.done = true
		if f.distPick() {
			// The shapes' picks run every distance, and one that has settled
			// already set packedFused; the default is only for before that.
			if len(f.distSize) == 0 {
				f.useFused(t.d.best - 1)
			}
			return
		}
		f.useFused(t.d.best - 1)
		for d, m := range f.fusedAt {
			if d != t.cur {
				for _, c := range m {
					c.Close()
				}
				delete(f.fusedAt, d)
			}
		}
		return
	}
	f.useFused(t.d.pick() - 1)
	t.t0 = time.Now()
}

func (f *JIT) fpfEnd() {
	t := f.fpf
	if t == nil || t.t0.IsZero() {
		return
	}
	dt := time.Since(t.t0)
	t.t0 = time.Time{}
	if t.discard {
		t.discard = false
		return
	}
	if t.skip < tuneSkip {
		t.skip++
		return
	}
	t.d.observe(1 / dt.Seconds())
}
