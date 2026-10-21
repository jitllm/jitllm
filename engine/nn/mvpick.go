//go:build amd64 || arm64

package nn

import (
	"slices"
	"time"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// On arm64 no one packed matvec wins every shape, so each shape is timed. The
// tiled kernel walks the plane-major runs super-block by super-block and suits
// long runs; the fused one keeps its sums in registers and walks all of k per
// row group, suiting short runs and long k, and splitting k across workers
// makes a worker's planes contiguous. The winner moves with rows, k and format
// (nn.TestShapeCost), so each (format, rows, k, stride) runs every candidate in
// turn for its first calls, in place, and keeps the fastest median. On x86
// the fused kernel measured best at every shape, so the kernel keeps its fixed
// rule and what each shape times is the fused kernel's prefetch distance
// (JIT.distPick).

// pickArms are the candidates: 0 the tiled kernel, s > 0 the fused one over s
// k-slices.
var pickArms = []int{0, 1, 2, 4}

const (
	pickWarm       = 1  // rounds discarded: first touches, emitted code, page faults
	pickRounds     = 5  // rounds kept per candidate
	pickDistRounds = 15 // the same for prefetch distances
)

type pickKey struct {
	t               quant.Type
	rows, k, stride int
}

// mvPick is one shape's choice: arms in turn until each has pickRounds samples
// past the warm-up, then the lowest median for good.
type mvPick struct {
	arms []int
	ns   [][]int64
	n    int // calls so far
	best int // chosen arm value, or -1 while timing
	// dist: the arms are fused prefetch distances rather than kernels
	// (JIT.distPick), timed over more rounds because they differ by a few
	// percent; applied once the choice has reached packedFused.
	dist    bool
	rounds  int
	applied bool
}

// pickFor returns the chooser for a shape, or nil when there is nothing to
// choose: timing off, or only one kernel can serve it.
func (f *JIT) pickFor(t quant.Type, nrows, k, stride int) *mvPick {
	if f.cfg.MatVecPick < 0 {
		return nil
	}
	if f.cfg.MatVecPick == 0 && f.tier != cpu.TierNEON {
		if !f.distPick() {
			return nil
		}
		key := pickKey{t, nrows, k, stride}
		if pk := f.picks[key]; pk != nil {
			return pk
		}
		pk := newPick(slices.Clone(prefetchDistances), pickDistRounds)
		pk.dist = true
		if f.picks == nil {
			f.picks = map[pickKey]*mvPick{}
		}
		f.picks[key] = pk
		return pk
	}
	// A caller that pinned the fused kernel's k walk asked for the fused
	// kernel; the fixed rule serves it.
	if f.cfg.MatVecPick == 0 && (f.cfg.FusedKSlices != 0 || f.cfg.FusedKSplit != 0) {
		return nil
	}
	// Tuning off is a promise of the same answer every run, and the k-sliced
	// arms reassociate the sum; the tiled kernel is the measured default.
	pin := f.cfg.MatVecPick
	if pin == 0 && f.cfg.Tune == TuneOff {
		pin = 1
	}
	key := pickKey{t, nrows, k, stride}
	if pk := f.picks[key]; pk != nil {
		return pk
	}
	nsup := k / cpu.PackedOuterElems(t)
	var arms []int
	for i, s := range pickArms {
		if pin > 0 && i != pin-1 {
			continue
		}
		// GEMMExact sums over the whole of k (fusedSlices), so a timed choice
		// under it never picks a split; a pinned one still may.
		if s > 1 && (s > nsup || f.pool.N()%s != 0 || (pin == 0 && f.cfg.GEMMExact)) {
			continue
		}
		arms = append(arms, s)
	}
	if len(arms) == 0 {
		arms = []int{0}
	}
	pk := newPick(arms, 0)
	if len(arms) == 1 {
		pk.best = arms[0]
	}
	if f.picks == nil {
		f.picks = map[pickKey]*mvPick{}
	}
	f.picks[key] = pk
	return pk
}

// newPick is a chooser over arms timed for rounds (0: pickRounds), its sample
// slices sized up front so the timing window appends without allocating.
func newPick(arms []int, rounds int) *mvPick {
	pk := &mvPick{arms: arms, ns: make([][]int64, len(arms)), best: -1, rounds: rounds}
	if rounds <= 0 {
		rounds = pickRounds
	}
	for i := range pk.ns {
		pk.ns[i] = make([]int64, 0, rounds)
	}
	return pk
}

// arm is the candidate this call runs.
func (pk *mvPick) arm() int {
	if pk.best >= 0 {
		return pk.best
	}
	return pk.arms[pk.n%len(pk.arms)]
}

// observe records the call that arm() chose, and decides once every candidate
// has its samples.
func (pk *mvPick) observe(d time.Duration) {
	if pk == nil || pk.best >= 0 {
		return
	}
	i := pk.n % len(pk.arms)
	if pk.n >= pickWarm*len(pk.arms) {
		pk.ns[i] = append(pk.ns[i], int64(d))
	}
	pk.n++
	rounds := pickRounds
	if pk.rounds > 0 {
		rounds = pk.rounds
	}
	if pk.n < (pickWarm+rounds)*len(pk.arms) {
		return
	}
	best, bestMed := 0, int64(-1)
	for i, s := range pk.ns {
		slices.Sort(s)
		if m := s[len(s)/2]; bestMed < 0 || m < bestMed {
			best, bestMed = i, m
		}
	}
	pk.best, pk.ns = pk.arms[best], nil
}
