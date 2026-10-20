package maintier

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	kernels "github.com/jitllm/jitllm/paritytest/mainkernels"
)

// Split width, measured on the device that will run it rather than tabulated:
// chooseSplit's fitted rule missed the optimum on most shapes, even on the
// card it was fitted on.
//
// The sweep runs at load against the real resident weights, so it measures the
// footprint that will actually be read, and costs a few milliseconds once.
//
// A dispersed measurement is refused, not rounded: above IQR/median 0.10 the
// table's answer stands.
const splitTuneIQR = 0.10

// splitTieBand is the margin by which tuneSplit's later candidate must beat
// the best so far: the narrower split and the partial-writing form win a tie.
const splitTieBand = 0.01

type splitKey struct {
	q       kernels.Quant
	rows, k int
}

// tuneSplit returns the best split for one shape, measuring once per shape.
func (g *devTier) tuneSplit(q kernels.Quant, rows, k int, r *resident) int {
	if g.kb.split >= 1 || g.kb.tune == TuneOff {
		return chooseSplit(rows, k/32, g.kb.split)
	}
	// Metal is measured like every other backend; this depends on Session's
	// Sync actually waiting for the GPU (see backend/metal.go).
	key := splitKey{q, rows, k}
	if s, ok := g.splits[key]; ok {
		return s
	}
	fallback := chooseSplit(rows, k/32, g.kb.split)
	g.splits[key] = fallback // so a failure below is not retried per layer

	// A measured answer persisted by an earlier process on this device, driver
	// and executable (WithTuneCache). An entry that does not parse, or names a
	// split this shape cannot take, is measured again and overwritten.
	ck := g.tuneKey(fmt.Sprintf("%v-%dx%d", q, rows, k))
	if v, ok := lookupCache(g.kb.cacheDir, "gpusplit1", ck); ok && g.kb.tune != TuneForce {
		var s, gi int
		if n, err := fmt.Sscanf(v, "%d %d", &s, &gi); n == 2 && err == nil && s >= 1 && nb32ok(k, s) &&
			(gi == 0 || gi == 1 && !g.NoGroupSplit && kernels.GroupSplitOK(rows, s)) &&
			// A partial-writing split needs its partial plane, which the
			// measurement below would have grown.
			(s == 1 || gi == 1 || g.sizePart(rows*s)) {
			g.splits[key], g.groupSplit[key] = s, gi == 1
			return s
		}
	}

	if g.kb.lazyTune && ck != "" && r != nil {
		g.queueTune(tuneJob{key: key, ck: ck, r: r, cands: g.splitCands(q, rows, k)})
		return fallback
	}

	g.trialQuiet()
	nb := k / 32
	out, err := g.dev.Alloc(rows * 4)
	if err != nil {
		return fallback
	}
	defer out.Free()

	// A float weight reads a K-float activation where a quant's scale plane
	// goes; its own d buffer is one word.
	d := r.d
	if kernels.IsFloat(q) {
		xf, err := g.dev.Alloc(k * 4)
		if err != nil {
			return fallback
		}
		defer xf.Free()
		if xf.Write(make([]byte, k*4)) != nil {
			return fallback
		}
		d = xf
	}
	best, bestRate, bestGroup := 0, 0.0, false
	for _, split := range splitCands(nb, splitMax(rows)) {
		if rows*split > 1<<22 {
			continue
		}
		// Both reductions are timed at every split: which wins is a property of
		// the shape, not the device (on Metal in-group wins big on q/o and
		// loses on a Q6_K head).
		for _, grp := range []bool{false, true} {
			if grp && g.NoGroupSplit {
				continue
			}
			if grp && !kernels.GroupSplitOK(rows, split) {
				continue
			}
			// The split tuner times the unbiased shape; a bias is one add per
			// row and does not move the split.
			kern := g.kernelMode(q, k, rows, split, 1, false, grp)
			if kern == nil {
				continue
			}
			var red backend.Kernel
			if split > 1 && !grp {
				if red = g.reduceKernel(rows, split); red == nil {
					continue
				}
				if !g.sizePart(rows * split) {
					continue
				}
			}
			// The in-group kernel writes the final row, so it needs no partial
			// plane, but it launches the same grid as the arm it is compared
			// against: `grp` says the output contract, `split` the grid.
			rate, ok := splitTrial(g, kern, red, r, d, out, rows, split, grp)
			// A later candidate (a wider split, or the in-group form) must
			// beat the best by splitTieBand: within it the trial cannot tell
			// them apart, and the pick went by noise. Ministral-3-3B's Q6_K head
			// measured 308/s at split 3 and 309/s at 24 in-group in one trial,
			// and the token ran 1.5% slower when the wide one won.
			if ok && rate > bestRate*(1+splitTieBand) {
				best, bestRate, bestGroup = split, rate, grp
			}
		}
	}
	if best == 0 {
		return fallback
	}
	g.groupSplit[key] = bestGroup
	g.splits[key] = best
	gi := 0
	if bestGroup {
		gi = 1
	}
	saveCache(g.kb.cacheDir, "gpusplit1", ck, fmt.Sprintf("%d %d", best, gi))
	if backend.Verbose() {
		fmt.Fprintf(os.Stderr, "tier: %s %dx%d split %d%s measured (%.0f/s), table says %d\n",
			q, rows, k, best, map[bool]string{true: " in-group"}[bestGroup], bestRate, fallback)
	}
	return best
}

var tuneTrials, bgTuneTrials atomic.Int64

// BackgroundTuneTrials counts the trials WithLazyTune's background tuner has
// timed in this process, which TuneTrials does not: they run beside the
// device's work, not on its path.
func BackgroundTuneTrials() int64 { return bgTuneTrials.Load() }

// bgTrial is one round of a background candidate's trial; a gate swaps it.
var bgTrial = (*devTier).timeSplitRound

// tuneJob is one shape the background tuner measures.
type tuneJob struct {
	key   splitKey
	ck    string
	r     *resident
	cands []splitCand
}

// splitCands compiles the candidates tuneSplit times for one shape: here, at
// placement, so the background tuner compiles nothing. Callers hold g.mu.
func (g *devTier) splitCands(q kernels.Quant, rows, k int) []splitCand {
	var cs []splitCand
	for _, split := range splitCands(k/32, splitMax(rows)) {
		if rows*split > 1<<22 {
			continue
		}
		for _, grp := range []bool{false, true} {
			if grp && (g.NoGroupSplit || !kernels.GroupSplitOK(rows, split)) {
				continue
			}
			c := splitCand{split: split, grp: grp}
			if c.kern = g.kernelMode(q, k, rows, split, 1, false, grp); c.kern == nil {
				continue
			}
			if split > 1 && !grp {
				if c.red = g.reduceKernel(rows, split); c.red == nil {
					continue
				}
			}
			cs = append(cs, c)
		}
	}
	return cs
}

// queueTune hands a shape to the background tuner, starting it if it is not
// running. Callers hold g.mu.
func (g *devTier) queueTune(j tuneJob) {
	if len(j.cands) < 2 || g.tuneStop {
		return
	}
	g.tuneQ = append(g.tuneQ, j)
	g.lastSub.Store(time.Now().UnixNano()) // placement is work too: wait past it
	if !g.tuneRunning {
		g.tuneRunning = true
		g.tuneWG.Add(1)
		// The tuner is a session of its own over lane0, never the view of
		// the session that queued the shape: that session's devSess says
		// whether ITS submission is running (inSub), which the tuner's
		// quiesce would read as its own and refuse.
		tv := &devTier{devShared: g.devShared, devSess: &devSess{}, lane: g.lane0, Stats: &g.tot}
		go tv.tuneWorker()
	}
}

// tuneWorker drains the queue. Each trial runs under g.mu, so it never meets
// a weight being freed, a scratch being rebuilt or Close; between trials the
// lock is free and the device's own work goes first.
func (g *devTier) tuneWorker() {
	defer g.tuneWG.Done()
	for {
		g.mu.Lock()
		if g.tuneStop || len(g.tuneQ) == 0 {
			g.tuneRunning = false
			g.mu.Unlock()
			return
		}
		j := g.tuneQ[0]
		g.tuneQ = g.tuneQ[1:]
		g.mu.Unlock()
		g.runTuneJob(j)
	}
}

// runTuneJob times one shape's candidates into buffers of its own (the
// partial sums of a running submission are g.partBuf) and adopts the winner.
func (g *devTier) runTuneJob(j tuneJob) {
	rows, k := j.key.rows, j.key.k
	maxSplit := 1
	for _, c := range j.cands {
		maxSplit = max(maxSplit, c.split)
	}
	var bufs []backend.Buf
	defer func() {
		for _, b := range bufs {
			b.Free()
		}
	}()
	alloc := func(n int) backend.Buf {
		b, err := g.dev.Alloc(n)
		if err != nil {
			return nil
		}
		bufs = append(bufs, b)
		return b
	}
	out, part := alloc(rows*4), alloc(rows*maxSplit*4)
	if out == nil || part == nil {
		return
	}
	var xf backend.Buf
	if kernels.IsFloat(j.key.q) {
		if xf = alloc(k * 4); xf == nil || xf.Write(make([]byte, k*4)) != nil {
			return
		}
	}
	best, bestRate, bestGroup := 0, 0.0, false
	for _, c := range j.cands {
		// The candidate is timed a round at a time, each round under g.mu
		// after the device has been quiet: a whole trial is ten rounds, and
		// on an output head (Q6_K 128256x4096) it held g.mu for 21 ms, which
		// a prompt arriving meanwhile waited for -- 64-token prompts on a
		// V100 read 830 and 520 tok/s against 1550 beside them. Between
		// rounds the lock is free and a prompt goes first.
		rates := make([]float64, 0, splitRounds)
		ok := true
		for round := 0; ok && round < splitRounds+1; round++ {
			if !g.waitQuiet() {
				return
			}
			g.mu.Lock()
			// No submission may be in flight under a trial (trialQuiet).
			g.trialQuiet()
			if g.tuneStop || g.closed || !j.r.ok {
				g.mu.Unlock()
				return
			}
			d := j.r.d
			if xf != nil {
				d = xf
			}
			if round == 0 {
				bgTuneTrials.Add(1)
			}
			var rate float64
			rate, ok = bgTrial(g, c.kern, c.red, j.r, d, out, part, rows, c.split, c.grp)
			g.mu.Unlock()
			if ok && round > 0 { // the first round pays first-touch
				rates = append(rates, rate)
			}
		}
		if !ok {
			continue
		}
		if rate, ok := medianRate(rates); ok && rate > bestRate {
			best, bestRate, bestGroup = c.split, rate, c.grp
		}
	}
	if best == 0 {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.tuneStop {
		return
	}
	// Adopted here only where the partial plane already holds it: growing it
	// would reallocate a buffer a submission may be writing. The cache takes
	// it either way, and the next placement sizes the plane for it.
	if best == 1 || bestGroup || rows*best <= g.partCap {
		g.splits[j.key], g.groupSplit[j.key] = best, bestGroup
	}
	gi := 0
	if bestGroup {
		gi = 1
	}
	saveCache(g.kb.cacheDir, "gpusplit1", j.ck, fmt.Sprintf("%d %d", best, gi))
}

// tuneQuiet is how long the device must have run no submission before the
// background tuner times a trial: a prompt's submissions come closer together
// than this, so a trial lands between requests, not inside one. A trial holds
// g.mu for its ten timed rounds, which a request arriving meanwhile waits for.
const tuneQuiet = 50 * time.Millisecond

// waitQuiet blocks until the device has been quiet for tuneQuiet, and reports
// false when the tuner is being stopped.
func (g *devTier) waitQuiet() bool {
	for {
		g.mu.Lock()
		stop := g.tuneStop
		g.mu.Unlock()
		if stop {
			return false
		}
		idle := time.Since(time.Unix(0, g.lastSub.Load()))
		if idle >= tuneQuiet {
			return true
		}
		time.Sleep(tuneQuiet - idle)
	}
}

// trialQuiet waits until no submission is in flight, before a split trial
// that holds g.mu. A trial ends in Session.Sync, which on CUDA is
// cuCtxSynchronize: it waits for every stream of the context, and the backend
// holds it off while any queue is recording (cudaDev.capturing). A submission
// runs outside g.mu (runUnlocked) and, while recording, may take g.mu again
// for a build reached from inside it (subLock). With g.mu held here and a
// submission in flight, the trial waits for the recording to end and the
// recording for g.mu: the deadlock a server's warm-up prefill met beside the
// background tuner on a T4. The placement-path trial also writes g.partBuf,
// which a submission in flight may be reading. Callers hold g.mu, and are not
// inside a submission or its prologue (quiesce refuses).
func (g *devTier) trialQuiet() { g.quiesce() }

// stopTune stops the background tuner and waits for it. Callers do not hold
// g.mu.
func (g *devTier) stopTune() {
	g.mu.Lock()
	g.tuneStop = true
	g.tuneQ = nil
	g.mu.Unlock()
	g.tuneWG.Wait()
}

// TuneTrials counts the timed trials the device tuners have run in this
// process (the split tuner, the sequence retunes, the backend probe): what a
// benchmark reads to show its timed span tuned nothing.
func TuneTrials() int64 { return tuneTrials.Load() }

// splitTrial is one timed candidate of tuneSplit; a gate swaps it to count
// trials and fix the winner.
var splitTrial = (*devTier).timeSplitMode

// nb32ok reports that split divides a k-wide row's 32-element blocks the way
// splitCands offers it.
func nb32ok(k, split int) bool { return split <= k/32 && (k/32)%split == 0 }

// tuneKey is a cache key for one shape on this tier's device: the
// executable's fingerprint, the device's UUID and driver, and the shape. ""
// when any part is unknown, and then nothing is read or written.
func (g *devTier) tuneKey(shape string) string {
	id, dk := binaryID(), deviceKey([]backend.Device{g.dev})
	if id == "" || dk == "" {
		return ""
	}
	return id + "-" + dk + "-" + shape
}

func (g *devTier) timeSplitMode(kern, red backend.Kernel, r *resident, d, out backend.Buf, rows, split int, grp bool) (float64, bool) {
	tuneTrials.Add(1)
	return g.timeSplitOn(kern, red, r, d, out, g.partBuf, rows, split, grp)
}

// timeSplitOn is one timed candidate writing its partial sums to part.
func (g *devTier) timeSplitOn(kern, red backend.Kernel, r *resident, d, out, part backend.Buf, rows, split int, grp bool) (float64, bool) {
	rates := make([]float64, 0, splitRounds)
	for round := 0; round < splitRounds+1; round++ {
		rate, ok := g.timeSplitRound(kern, red, r, d, out, part, rows, split, grp)
		if !ok {
			return 0, false
		}
		if round > 0 { // the first round pays compilation and first-touch
			rates = append(rates, rate)
		}
	}
	return medianRate(rates)
}

// splitRounds is how many timed rounds a split trial takes, after one
// untimed.
const splitRounds = 9

// medianRate is a trial's verdict over its rounds' rates: the median, or
// false when they spread past splitTuneIQR.
func medianRate(rates []float64) (float64, bool) {
	if len(rates) == 0 {
		return 0, false
	}
	sort.Float64s(rates)
	med := rates[len(rates)/2]
	iqr := rates[len(rates)*3/4] - rates[len(rates)/4]
	if med <= 0 || iqr/med > splitTuneIQR {
		return 0, false
	}
	return med, true
}

// timeSplitRound times one round of a split trial: four launches of the
// candidate and a synchronise, as launches a second.
func (g *devTier) timeSplitRound(kern, red backend.Kernel, r *resident, d, out, part backend.Buf, rows, split int, grp bool) (float64, bool) {
	if g.bs == nil {
		return 0, false // no activation to read: a scratch rebuilt or closed
	}
	const iters = 4
	t0 := time.Now()
	var err error
	g.dev.Session(func(s backend.Session) {
		backend.MarkTrial(s) // a profile tells a tuner's trial from the work
		for i := 0; i < iters; i++ {
			dst := out
			if split > 1 && !grp {
				dst = part
			}
			if err = s.Launch(kern, groupsOf(rows*split, split, grp), widthOf(split, grp),
				r.qs, d, r.sc, g.bs.a, g.bs.ax, dst); err != nil {
				return
			}
			if split > 1 && !grp {
				if err = s.Launch(red, (rows+127)/128, 128, part, out); err != nil {
					return
				}
			}
		}
		err = s.Sync()
	})
	if err != nil {
		return 0, false
	}
	return float64(iters) / time.Since(t0).Seconds(), true
}

// retuneDecode re-picks each dense decode matvec's split once every block is
// placed, by timing it the way a token runs it: one launch per placed block's
// OWN weights, in order, so each launch streams a matrix the one before it did
// not touch.
//
// tuneSplit times one matrix repeatedly, which a small matrix serves from L2,
// while a decode token reads every weight from DRAM; the cache-resident answer
// can be the slowest in a token. A sequence over the real blocks is the regime
// itself (flushing L2 per trial measured worse; see
// docs/engineering-history/gpu-kernels.md, "Flushing L2 before each
// split-tuner trial was tried").
//
// It runs from Layers, before the first decode token is recorded, and again
// whenever the number of placed blocks has doubled since. Callers hold no lock
// and are outside any backend Session.
func (g *devTier) retuneDecode() {
	g.mu.Lock()
	defer g.mu.Unlock()
	// Doubling rather than every change: -gpu-grow adopts one block a token,
	// and a pass is tens of milliseconds a shape.
	if g.kb.split >= 1 || g.kb.tune == TuneOff || g.bs == nil || len(g.layers) < 2*g.retunedAt || len(g.layers) == 0 {
		return
	}
	g.retunedAt = len(g.layers)
	// The retune rewrites the blocks' matvecs, which a submission in flight
	// launches.
	g.quiesce()
	type use struct {
		m *mv
		r *resident
	}
	by := map[splitKey][]use{}
	for _, l := range g.layers {
		if l == nil || !l.ok || l.nonCausal || l.linear || (l.pg != nil && !l.pg.in) {
			continue
		}
		for _, u := range []use{{&l.mvq, l.wq}, {&l.mvk, l.wk}, {&l.mvv, l.wv}, {&l.mvo, l.wo},
			{&l.mvg, l.gate}, {&l.mvu, l.up}, {&l.mvd, l.down},
			// MLA's projections and the shared expert are dense decode shapes
			// too, and the single-matrix trial picks them unreliably.
			{&l.mvQA, l.wqa}, {&l.mvQB, l.wqb}, {&l.mvKVA, l.wkva},
			{&l.mvsg, l.shGate}, {&l.mvsu, l.shUp}, {&l.mvsd, l.shDown}} {
			m, r := u.m, u.r
			if r == nil || m.kern == nil || m.slots > 0 || m.groups > 0 || m.threads > 0 ||
				m.rowt > 1 || m.bias != nil || kernels.IsFloat(m.q) || r.nrows != m.rows {
				continue
			}
			key := splitKey{m.q, m.rows, m.k}
			by[key] = append(by[key], u)
		}
	}
	changed := false
	for key, us := range by {
		if len(us) < 2 {
			continue
		}
		// 16 blocks stream well past a mid-size card's L2 but not far past a large one's;
		// widen it if a card's L2 ever holds the whole sequence.
		rs := make([]*resident, 0, 16)
		for _, u := range us[:min(len(us), 16)] {
			rs = append(rs, u.r)
		}
		ck := g.tuneKey(fmt.Sprintf("%v-%dx%d", key.q, key.rows, key.k))
		var cands []splitCand
		nb := key.k / 32
		for _, split := range splitCands(nb, 64) {
			if key.rows*split > 1<<22 {
				continue
			}
			for _, grp := range []bool{false, true} {
				if grp && (g.NoGroupSplit || !kernels.GroupSplitOK(key.rows, split)) {
					continue
				}
				c := splitCand{split: split, grp: grp}
				if c.kern = g.kernelMode(key.q, key.k, key.rows, split, 1, false, grp); c.kern == nil {
					continue
				}
				if split > 1 && !grp {
					if c.red = g.reduceKernel(key.rows, split); c.red == nil || !g.sizePart(key.rows*split) {
						continue
					}
				}
				cands = append(cands, c)
			}
		}
		// A candidate with a reduce launches twice per matvec and must win by
		// more than the timing can resolve (~1 us a launch) to be taken over one
		// that writes its row itself.
		cost := func(c splitCand) float64 {
			if c.split > 1 && !c.grp {
				return c.t * 1.03
			}
			return c.t
		}
		best, bestT, bestGrp := 0, 0.0, false
		if v, ok := lookupCache(g.kb.cacheDir, "gpusplit", ck); ok && g.kb.tune != TuneForce {
			var gi int
			if n, _ := fmt.Sscanf(v, "%d %d", &best, &gi); n != 2 {
				best = 0
			}
			bestGrp = gi == 1
		} else {
			for _, c := range g.timeSequences(cands, rs, key.rows, nil) {
				if c.t > 0 && (best == 0 || cost(c) < bestT) {
					best, bestT, bestGrp = c.split, cost(c), c.grp
				}
			}
			if best > 0 && ck != "" {
				gi := 0
				if bestGrp {
					gi = 1
				}
				saveCache(g.kb.cacheDir, "gpusplit", ck, fmt.Sprintf("%d %d", best, gi))
			}
		}
		m0 := us[0].m
		if best == 0 || (best == m0.split && bestGrp == m0.group) {
			continue
		}
		kern := g.kernelMode(key.q, key.k, key.rows, best, 1, false, bestGrp)
		var red backend.Kernel
		if best > 1 && !bestGrp {
			red = g.reduceKernel(key.rows, best)
		}
		if kern == nil || (best > 1 && !bestGrp && red == nil) {
			continue
		}
		if backend.Verbose() {
			fmt.Fprintf(os.Stderr, "tier: %s %dx%d retuned over %d blocks: split %d%s -> %d%s (%.1f us a launch)\n",
				key.q, key.rows, key.k, len(us), m0.split, map[bool]string{true: " in-group"}[m0.group],
				best, map[bool]string{true: " in-group"}[bestGrp], bestT*1e6)
		}
		for _, u := range us {
			u.m.kern, u.m.red, u.m.split, u.m.group = kern, red, best, bestGrp
			if u.m.res != nil {
				u.m.res = g.kernelMode(key.q, key.k, key.rows, best, 1, true, bestGrp)
			}
			if u.m.gated != nil || u.m.wantGated {
				u.m.gated = g.gatedKernel(key.q, key.k, key.rows, best, bestGrp, g.bs.p.Act)
				u.m.gate2 = g.gate2Kernel(key.q, key.k, key.rows, best, bestGrp, g.bs.p.Act)
			}
		}
		g.splits[key], g.groupSplit[key] = best, bestGrp
		changed = true
	}
	if g.retuneIndexed() {
		changed = true
	}
	if changed {
		for _, l := range g.layers {
			if l != nil && l.ok {
				g.fuseQKV(l)
			}
		}
		g.dropGraph()
	}
	// After the splits settle: the operand prologue is timed at the split
	// each matvec now runs.
	g.tuneActIn()
	// After the prologue's verdicts: the FFN's recipes run down in the
	// form the prologue tuner chose.
	g.tuneFFN()
	g.tuneGemm()
}

// splitMax is the widest split the tuner offers a shape of rows output rows.
// 32 covers a matrix whose rows fill the card; a matrix of a few rows over a
// long k (a hyper-connection's mix: 4 rows of 10240) runs 4*32 threads at 32,
// one multiprocessor of eighty, so it is offered splits up to 512 and the
// measurement decides.
func splitMax(rows int) int {
	if rows < 64 {
		return 512
	}
	return 32
}

// splitCands is every split up to max that divides nb activation blocks,
// ascending. Divisors, not powers of two: gpt-oss's k of 2880 is 90 blocks,
// which only 2 of the powers of two divide. For a power-of-two nb the lists
// are the same.
func splitCands(nb, max int) []int {
	var out []int
	for s := 1; s <= max && s <= nb; s++ {
		if nb%s == 0 {
			out = append(out, s)
		}
	}
	return out
}

// retuneIndexed does for a mixture's EXPERT matvecs what retuneDecode does for
// the dense ones: it times every split -- partial-writing and in-group -- over
// a sequence of placed blocks' own banks, the way a token runs them, and adopts
// the fastest, where the table (fitted to one small card, powers of two only)
// badly under-filled larger cards. It reports whether any matvec changed. A
// streamed bank and a biased one are left to the table. Caller holds g.mu.
func (g *devTier) retuneIndexed() bool {
	if g.bs == nil {
		return false
	}
	type idKey struct {
		q                       kernels.Quant
		rows, k, experts, slots int
		slotAct                 bool
	}
	type use struct {
		m *mv
		r *resident
	}
	by := map[idKey][]use{}
	var order []idKey
	for _, l := range g.layers {
		if l == nil || !l.ok || l.stream != nil || (l.pg != nil && !l.pg.in) {
			continue
		}
		// A mixture's three banks, and MLA's two per-head absorb banks, which
		// are the same indexed shape with the identity as the selection.
		var cand []use
		if l.router != nil {
			cand = append(cand, use{&l.mvg, l.gate}, use{&l.mvu, l.up}, use{&l.mvd, l.down})
		}
		if l.mla {
			cand = append(cand, use{&l.mvKB, l.wkb}, use{&l.mvVB, l.wvb})
		}
		for _, u := range cand {
			m, r := u.m, u.r
			if r == nil || m.kern == nil || m.slots < 1 || m.experts < 2 || m.bias != nil ||
				kernels.IsFloat(m.q) {
				continue
			}
			key := idKey{m.q, m.rows, m.k, m.experts, m.slots, m.slotAct}
			if by[key] == nil {
				order = append(order, key)
			}
			by[key] = append(by[key], u)
		}
	}
	changed := false
	for _, key := range order {
		us := by[key]
		if len(us) < 2 {
			continue
		}
		nout := key.rows * key.slots
		ck := g.tuneKey(fmt.Sprintf("%v-%dx%d-e%dx%d-%v", key.q, key.rows, key.k, key.experts, key.slots, key.slotAct))
		mk := func(split int, grp bool) (backend.Kernel, backend.Kernel, bool) {
			kern, err := kernels.MatVec(kernels.MatVecShape{Center: g.center, T: key.q, K: key.k, Rows: key.rows,
				Split: split, Experts: key.experts, Slots: key.slots, SlotAct: key.slotAct, GroupSplit: grp})
			if err != nil {
				return nil, nil, false
			}
			c, err := g.dev.Compile(kern)
			if err != nil {
				return nil, nil, false
			}
			var red backend.Kernel
			if split > 1 && !grp {
				if red = g.reduceKernel(nout, split); red == nil || !g.sizePart(nout*split) {
					return nil, nil, false
				}
			}
			return c, red, true
		}
		best, bestGrp := 0, false
		if v, ok := lookupCache(g.kb.cacheDir, "gpusplitid", ck); ok && g.kb.tune != TuneForce {
			var gi int
			if n, _ := fmt.Sscanf(v, "%d %d", &best, &gi); n != 2 {
				best = 0
			}
			bestGrp = gi == 1
		} else {
			var cands []splitCand
			for _, split := range splitCands(key.k/32, 64) {
				if nout*split > 1<<22 {
					continue
				}
				for _, grp := range []bool{false, true} {
					if grp && (g.NoGroupSplit || !kernels.GroupSplitOK(key.rows, split)) {
						continue
					}
					if kern, red, ok := mk(split, grp); ok {
						cands = append(cands, splitCand{kern: kern, red: red, split: split, grp: grp})
					}
				}
			}
			// The selection: distinct experts spread over the bank, the same
			// for every block of the sequence -- each block's bank is its own
			// memory, so the sequence still streams from DRAM.
			ids := make([]uint32, key.slots)
			for i := range ids {
				ids[i] = uint32(i * key.experts / key.slots)
			}
			sel, err := g.dev.Alloc(len(ids) * 4)
			if err != nil {
				continue
			}
			if sel.Write(u32b(ids)) != nil {
				sel.Free()
				continue
			}
			rs := make([]*resident, 0, 16)
			for _, u := range us[:min(len(us), 16)] {
				rs = append(rs, u.r)
			}
			bestT := 0.0
			cost := func(c splitCand) float64 {
				if c.split > 1 && !c.grp {
					return c.t * 1.03
				}
				return c.t
			}
			for _, c := range g.timeSequences(cands, rs, nout, sel) {
				if c.t > 0 && (best == 0 || cost(c) < bestT) {
					best, bestT, bestGrp = c.split, cost(c), c.grp
				}
			}
			sel.Free()
			// The candidates were compiled for the trial alone; the winner is
			// compiled again below. Reductions are the tier's cache and stay.
			for _, c := range cands {
				c.kern.Close()
			}
			if best > 0 && ck != "" {
				gi := 0
				if bestGrp {
					gi = 1
				}
				saveCache(g.kb.cacheDir, "gpusplitid", ck, fmt.Sprintf("%d %d", best, gi))
			}
			if backend.Verbose() && best > 0 {
				fmt.Fprintf(os.Stderr, "tier: %s %dx%d e%dx%d indexed, retuned over %d blocks: split %d%s -> %d%s (%.1f us a launch)\n",
					key.q, key.rows, key.k, key.experts, key.slots, len(rs), us[0].m.split,
					map[bool]string{true: " in-group"}[us[0].m.group],
					best, map[bool]string{true: " in-group"}[bestGrp], bestT*1e6)
			}
		}
		m0 := us[0].m
		if best == 0 || (best == m0.split && bestGrp == m0.group) {
			continue
		}
		// The winner from the device's cache, which Close closes; mk's kernels
		// are the trial's own.
		kern, err := g.idKernel(kernels.MatVecShape{Center: g.center, T: key.q, K: key.k, Rows: key.rows,
			Split: best, Experts: key.experts, Slots: key.slots, SlotAct: key.slotAct, GroupSplit: bestGrp})
		if err != nil {
			continue
		}
		var red backend.Kernel
		if best > 1 && !bestGrp {
			if red = g.reduceKernel(nout, best); red == nil || !g.sizePart(nout*best) {
				continue
			}
		}
		for _, u := range us {
			if bestGrp && !u.m.group {
				g.IndexedGroup++
			}
			u.m.kern, u.m.red, u.m.split, u.m.group = kern, red, best, bestGrp
		}
		changed = true
	}
	return changed
}

// splitCand is one split retuneDecode considers, and t its median seconds a
// launch (0 when it could not be timed).
type splitCand struct {
	kern, red backend.Kernel
	split     int
	grp       bool
	t         float64
}

// timeSequences times every candidate run once over each of rs in turn, the
// candidates interleaved round by round so drift lands on all of them alike.
// Where the device records, each sequence is a captured graph and a trial is
// its replay -- the way a decode token is launched -- so host launch cost does
// not enter the comparison.
//
// sel is the expert selection an INDEXED matvec reads after its output, or nil
// for a dense one; rows is then the launch's output rows, Rows*Slots.
func (g *devTier) timeSequences(cs []splitCand, rs []*resident, rows int, sel backend.Buf) []splitCand {
	tuneTrials.Add(1)
	// Three replays a sample so a Sync's jitter is small against the work, and
	// seven rounds; the differences being decided are ~1 us a launch.
	const rounds, reps = 7, 3
	out, err := g.dev.Alloc(rows * 4)
	if err != nil {
		return nil
	}
	defer out.Free()
	seq := func(s backend.Session, c splitCand) error {
		for _, r := range rs {
			dst := out
			if c.split > 1 && !c.grp {
				dst = g.partBuf
			}
			args := []backend.Buf{r.qs, r.d, r.sc, g.bs.a, g.bs.ax, dst}
			if sel != nil {
				args = append(args, sel)
			}
			if err := s.Launch(c.kern, groupsOf(rows*c.split, c.split, c.grp), widthOf(c.split, c.grp),
				args...); err != nil {
				return err
			}
			if c.split > 1 && !c.grp {
				if err := s.Launch(c.red, (rows+127)/128, 128, g.partBuf, out); err != nil {
					return err
				}
			}
		}
		return nil
	}
	recs := make([]backend.Recording, len(cs))
	defer func() {
		for _, r := range recs {
			if r != nil {
				r.Free()
			}
		}
	}()
	// A failed capture is freed after the session, never inside it: the
	// callback runs on CUDA's owner goroutine, and cudaGraph.Free posts to
	// that same goroutine and waits, which deadlocks.
	var failed []backend.Recording
	defer func() {
		for _, r := range failed {
			r.Free()
		}
	}()
	g.dev.Session(func(s backend.Session) {
		backend.MarkTrial(s) // a profile tells a tuner's trial from the work
		rec, ok := s.(backend.Recorder)
		if !ok || g.NoGraph {
			return
		}
		for i, c := range cs {
			var inner error
			r, e := backend.Record(rec, func() { inner = seq(s, c) })
			if e == nil && inner == nil {
				recs[i] = r
			} else if r != nil {
				failed = append(failed, r)
			}
		}
	})
	per := make([][]float64, len(cs))
	dead := make([]bool, len(cs))
	for round := 0; round < rounds+1; round++ {
		for i, c := range cs {
			if dead[i] {
				continue
			}
			t0 := time.Now()
			g.dev.Session(func(s backend.Session) {
				backend.MarkTrial(s) // a profile tells a tuner's trial from the work
				for r := 0; r < reps && err == nil; r++ {
					if recs[i] != nil {
						err = s.(backend.Recorder).Replay(recs[i])
					} else {
						err = seq(s, c)
					}
				}
				if err == nil {
					err = s.Sync()
				}
			})
			if err != nil {
				dead[i], err = true, nil
				continue
			}
			if round > 0 { // the first pays first touch
				per[i] = append(per[i], time.Since(t0).Seconds()/float64(reps*len(rs)))
			}
		}
		if round == 1 {
			// A candidate half again slower than the best after one round is
			// not going to win, and a split-1 sequence over a 4096x14336 matrix
			// is 3 ms a replay: stop timing it.
			best := 0.0
			for i := range cs {
				if !dead[i] && (best == 0 || per[i][0] < best) {
					best = per[i][0]
				}
			}
			for i := range cs {
				if !dead[i] && per[i][0] > 1.5*best {
					dead[i] = true
				}
			}
		}
	}
	for i, p := range per {
		if len(p) < rounds {
			continue
		}
		sort.Float64s(p)
		if med := p[len(p)/2]; med > 0 && (p[len(p)*3/4]-p[len(p)/4])/med <= splitTuneIQR {
			cs[i].t = med
		}
		if backend.Verbose() {
			fmt.Fprintf(os.Stderr, "tier: sequence trial rows=%d split %d grp=%v: median %.2f us, IQR/median %.3f\n",
				rows, cs[i].split, cs[i].grp, p[len(p)/2]*1e6, (p[len(p)*3/4]-p[len(p)/4])/p[len(p)/2])
		}
	}
	return cs
}

// binaryID fingerprints the running executable, so a cached split does not
// outlive the kernels it was measured on: any rebuild tunes once again. Empty
// when the executable cannot be read, and then nothing is cached.
var binaryID = sync.OnceValue(func() string {
	p, err := os.Executable()
	if err != nil {
		return ""
	}
	f, err := os.Open(p)
	if err != nil {
		return ""
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return ""
	}
	return fmt.Sprintf("%x", h.Sum(nil)[:8])
})

// fuseMoE builds a mixture block's expert matvecs with their epilogues fused
// -- the idea of llama.cpp's MUL_MAT_ID with glu fusion (mmvq.cu,
// ggml_cuda_mm_fusion_args: the up projection applies the gate, the biases and
// the GLU in its own epilogue) -- or leaves them nil. It needs the final-row
// forms (split 1, or in-group): a partial-writing split has no row to apply an
// epilogue to until its Reduce.
//
// It removes one to four launches a layer: the gated activation always
// (kernels.ActMul over the slots), and with expert biases the three
// kernels.IndexedBiasAdd. A streamed bank is left alone: its weights are
// compacted 0..k-1 while its biases need the true ids.
func (g *devTier) fuseMoE(l *layer) {
	l.moeGate, l.moeUp, l.moeDown = nil, nil, nil
	if g.NoMoEFuse || l.router == nil || l.stream != nil || g.bs == nil {
		return
	}
	// Llama 4 weights the expert's input (ActMulWeighted), which the fused
	// up epilogue does not carry.
	if g.bs.p.ExpertWeightIn {
		return
	}
	act := g.bs.p.Act
	if act != kernels.ActSiLU && act != kernels.ActGELU && act != kernels.ActSwiGLUOAI {
		return
	}
	final := func(m mv) bool {
		return m.kern != nil && m.slots > 0 && m.experts > 1 && m.bias == nil &&
			(m.split <= 1 || m.group) && !kernels.IsFloat(m.q)
	}
	build := func(m mv, bias, gate bool) backend.Kernel {
		s := kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, Split: m.split, GroupSplit: m.group,
			Experts: m.experts, Slots: m.slots, SlotAct: m.slotAct, Bias: bias, Gate: gate, GateAct: act}
		key := fmt.Sprintf("%+v", s)
		if k, ok := g.segKerns[key]; ok {
			return k
		}
		var k backend.Kernel
		ker, err := kernels.MatVec(s)
		if err == nil {
			k, err = g.dev.Compile(ker)
		}
		if err != nil {
			g.LastErr, k = err.Error(), nil
		}
		if g.segKerns == nil {
			g.segKerns = map[string]backend.Kernel{}
		}
		g.segKerns[key] = k
		return k
	}
	// The up projection is the one that must fuse; the gate's bias only
	// matters if it does, since the fused up reads the gate's output as is.
	if !final(l.mvg) || !final(l.mvu) {
		return
	}
	up := build(l.mvu, l.expUpB != nil, true)
	if up == nil {
		return
	}
	var gate backend.Kernel
	if l.expGateB != nil {
		if gate = build(l.mvg, true, false); gate == nil {
			return
		}
	}
	l.moeGate, l.moeUp = gate, up
	if l.expDownB != nil && final(l.mvd) {
		l.moeDown = build(l.mvd, true, false)
	}
}

// warpNorm reports whether the whole-row RMSNorm kernels take the
// subgroup-shuffle reduction (kernels.RMSNormRowsWarp): a device that
// guarantees 32 lanes AND whose subgroups are contiguous runs of 32 threads,
// which CUDA's warps are by architecture and nothing promises elsewhere.
// Config.NoWarpNorm keeps the shared-memory tree.
func (g *devTier) warpNorm() bool {
	return !g.NoWarpNorm && g.dev.API() == "ptx" && g.pickLanes() == ir.SubgroupLanes
}

func (g *devTier) rmsRows() func(k, rows int, eps float32, addOne, add bool) (*ir.Kernel, error) {
	if g.warpNorm() {
		return kernels.RMSNormRowsWarp
	}
	return kernels.RMSNormRows
}

func (g *devTier) rmsQuantRows() func(k, rows int, eps float32, addOne, add bool, window int) (*ir.Kernel, error) {
	if g.warpNorm() {
		return kernels.RMSNormQuantRowsWarp
	}
	return kernels.RMSNormQuantRows
}

// segLaunch is a kernels.MatVecSegments launch: the kernel and its grid.
type segLaunch struct {
	kern      backend.Kernel
	blocks, w int
}

// fuseSegs builds one launch for several plain, unbiased decode matvecs over
// the SAME activation, at the first one's split -- fuseQKV's recipe for any
// such group. nil where they cannot share a launch. A pair saves a launch
// each time and the small member rides the big one's grid (MLA's 576-row kv_a
// alone reads far below the bandwidth wall).
func (g *devTier) fuseSegs(ms []mv) *segLaunch {
	if len(ms) < 2 || g.NoSegFuse {
		return nil
	}
	m0 := ms[0]
	if m0.split > 1 && !m0.group {
		return nil
	}
	var segs []kernels.MatVecShape
	for _, m := range ms {
		if m.kern == nil || m.slots > 0 || m.groups > 0 || m.threads > 0 || m.rowt > 1 ||
			m.k != m0.k || m.bias != nil || kernels.IsFloat(m.q) {
			return nil
		}
		segs = append(segs, kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows})
	}
	blocks, w, err := kernels.SegmentsGrid(segs, m0.split, m0.group)
	if err != nil {
		return nil
	}
	key := fmt.Sprintf("%v/%d/%v", segs, m0.split, m0.group)
	k, ok := g.segKerns[key]
	if !ok {
		ker, err := kernels.MatVecSegments(segs, m0.split, m0.group)
		if err == nil {
			k, err = g.dev.Compile(ker)
		}
		if err != nil {
			g.LastErr = err.Error()
			k = nil
		}
		if g.segKerns == nil {
			g.segKerns = map[string]backend.Kernel{}
		}
		g.segKerns[key] = k
	}
	if k == nil {
		return nil
	}
	n := 0
	for _, b := range blocks {
		n += b
	}
	return &segLaunch{kern: k, blocks: n, w: w}
}

// widthOf is a matvec's workgroup width, and groupsOf the groups that cover
// threads at it; see kernels.GroupSplitWidth.
func widthOf(split int, grp bool) int {
	if grp {
		return kernels.GroupSplitWidth(split)
	}
	return 128
}

func groupsOf(threads, split int, grp bool) int {
	w := widthOf(split, grp)
	return (threads + w - 1) / w
}

// fuseQKV builds block l's q/k/v launch (kernels.MatVecSegments) at the split
// its q projection runs, or leaves l.qkv nil where the three cannot share one:
// a batched, indexed, tiled or float matvec, a split that needs a separate
// reduction, or projections that read different activations. MLA, recurrent
// and vision blocks never reach it with three plain decode matvecs.
func (g *devTier) fuseQKV(l *layer) {
	defer g.fuseActIn(l)
	g.fuseMoE(l)
	l.mlaQK, l.shGU = nil, nil
	if l.mla {
		q, wq := l.mvq, l.wq
		if l.wqa != nil {
			q, wq = l.mvQA, l.wqa
		}
		if wq != nil && l.wkva != nil {
			l.mlaQK = g.fuseSegs([]mv{q, l.mvKVA})
		}
	}
	if l.shGate != nil && l.shUp != nil {
		l.shGU = g.fuseSegs([]mv{l.mvsg, l.mvsu})
	}
	g.fuseGateUp(l)
	l.qkv = nil
	if l.mla || l.linear || l.nonCausal || l.wq == nil || l.wk == nil || l.wv == nil {
		return
	}
	q := l.mvq
	if q.split > 1 && !q.group {
		return
	}
	var segs []kernels.MatVecShape
	for _, m := range []mv{l.mvq, l.mvk, l.mvv} {
		if m.kern == nil || m.slots > 0 || m.groups > 0 || m.threads > 0 || m.rowt > 1 || m.k != q.k {
			return
		}
		segs = append(segs, kernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, Bias: m.bias != nil})
	}
	blocks, w, err := kernels.SegmentsGrid(segs, q.split, q.group)
	if err != nil {
		return
	}
	key := fmt.Sprintf("%v/%d/%v", segs, q.split, q.group)
	k, ok := g.segKerns[key]
	if !ok {
		ker, err := kernels.MatVecSegments(segs, q.split, q.group)
		if err == nil {
			k, err = g.dev.Compile(ker)
		}
		if err != nil {
			g.LastErr = err.Error()
			k = nil
		}
		if g.segKerns == nil {
			g.segKerns = map[string]backend.Kernel{}
		}
		g.segKerns[key] = k
	}
	if k == nil {
		return
	}
	n := 0
	for _, b := range blocks {
		n += b
	}
	l.qkv, l.qkvBlocks, l.qkvW = k, n, w
	// The launch appends a buffer per bias it finds, and CUDA does not check the
	// count; it runs this kernel only while these still match.
	l.qkvBias = [3]bool{segs[0].Bias, segs[1].Bias, segs[2].Bias}
}

// fuseGateUp builds block l's dense gate/up launch and the down kernel that
// forms act(gate)*up from their rows (Config.GateUpFuse), or leaves both nil.
// The two must run at one split and one reduction form, so each row's sum is
// the one its own launch computed; down's prologue is QuantizeAct's
// arithmetic, the same element and the same rounding as up's gated store
// followed by down's quantize.
func (g *devTier) fuseGateUp(l *layer) {
	l.gu, l.guDown = nil, nil
	why := func(w string) { g.GateUpWhy = w }
	// Built whether or not the launch takes it, so the switch can be thrown
	// between tokens (an A/B) without a retune.
	if g.bs == nil || l.router != nil || l.mla || l.linear || l.nonCausal ||
		l.gate == nil || l.up == nil || l.down == nil || ungatedFFN(&g.bs.p) || lnBlock(&g.bs.p) {
		return
	}
	mg, mu, md := l.mvg, l.mvu, l.mvd
	switch {
	case mg.split != mu.split || mg.group != mu.group:
		why(fmt.Sprintf("gate split %d/%v, up %d/%v", mg.split, mg.group, mu.split, mu.group))
		return
	case mg.bias != nil || mu.bias != nil:
		why("a biased gate or up")
		return
	case md.res == nil || md.bias != nil || md.slots > 0 || md.groups > 0 || md.threads > 0 || md.act != nil:
		why("down is not a plain residual matvec")
		return
	case g.pickLanes() != ir.SubgroupLanes || kernels.WideWindow(g.bs.p.ActWin):
		why("no 32-lane subgroup, or a wide quantization window")
		return
	}
	gu := g.fuseSegs([]mv{mg, mu})
	if gu == nil {
		why(fmt.Sprintf("gate and up cannot share a launch at split %d/%v (rowt %d/%d)", mg.split, mg.group, mg.rowt, mu.rowt))
		return
	}
	in := kernels.ActIn{On: true, Gated: true, Act: g.bs.p.Act, Window: max(g.bs.p.ActWin, 32), Prefetch: 8}
	if g.ActInPrefetch != 0 {
		in.Prefetch = max(g.ActInPrefetch, 0)
	}
	s := kernels.MatVecShape{Center: g.center, T: md.q, K: md.k, Rows: md.rows, Split: md.split,
		GroupSplit: md.group, Rowt: max(md.rowt, 1), Bias: true, In: in}
	down, err := g.idKernel(s)
	if err != nil {
		why("down's gated prologue: " + err.Error())
		return
	}
	l.gu, l.guDown = gu, down
	g.GateUpWhy = ""
}

// gateUpSwapped hands down's prologue up's row as the gate and the gate's as
// up: the violation the fused FFN's gate must see (SwapGateUpForTest,
// jitllmtest builds only).
var gateUpSwapped bool
