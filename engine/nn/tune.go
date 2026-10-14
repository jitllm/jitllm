//go:build amd64 || arm64

package nn

import (
	"bufio"
	"fmt"
	"hash/fnv"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit/cpu"
)

// Interleave width is chosen by measuring real tokens, then cached per
// (CPU, cores, model shape set).
//
// N interleaved rows are N concurrent read streams per worker, and what runs
// out is the hardware prefetcher: a wider pack issues fewer instructions and
// still loses once its loads become stalling demand loads. That balance
// depends on the prefetcher, the cache hierarchy, the core count and the
// model's shapes, so it is measured rather than tabulated -- on real tokens,
// because a load-time microbenchmark on a slab that fits in cache picks the
// wrong answer. One width serves every quant type: the stream count is a
// property of the pool, not the format. See docs/engineering-history/cpu-kernels.md.
const (
	// tuneRound is ratios per duel, and an ABBA quad yields two, so a duel
	// costs six tokens plus the skipped ones. A tuner that needs many tokens
	// never settles on a short generation.
	tuneRound  = 6    // paired ratios per duel, two per ABBA quad
	tuneSkip   = 2    // tokens discarded first: page faults, cold caches, ramp
	tuneMargin = 0.98 // a challenger must win by >2% to justify another step

	// tuneMaxIQR is the same dispersion gate the bench harness applies. Above
	// it the host is too noisy to separate the candidates, so the tuner falls
	// back to the default and caches nothing.
	tuneMaxIQR = 0.10
)

// Tuner knobs come from the owning JIT's f.cfg, not package variables, so two
// models in one process do not share a tuner mode or pins.

var (
	tuneCacheOnce sync.Once
	tuneCache     map[string]int
	tuneCacheMu   sync.Mutex
)

// packTuner runs a sequence of paired duels across consecutive tokens: best vs
// best+1, interleaved token by token, climbing while the challenger wins. The
// curve is unimodal in practice, so this normally settles within ten tokens.
// Interleaving rather than running A then B is what cancels thermal drift.
type packTuner struct {
	on    bool
	quiet bool // Config.Quiet, carried so settle/end need no package variable
	best  int
	chal  int // challenger; 0 once settled
	cur   int // width this token is using
	keyed bool
	skip  int
	turn  int
	quad  [4]time.Duration
	rat   []float64
	t0    time.Time
	key   string
}

func newPackTuner(static, pin int, mode TuneMode, quiet bool) *packTuner {
	t := &packTuner{best: static, cur: static, quiet: quiet}
	if static < 2 || pin != 0 || mode == TuneOff {
		return t // no interleaved kernel, or an explicit width was demanded
	}
	t.on, t.chal = true, static+1
	return t
}

// width is the interleave width this token should use. Zero means the dispatch
// has no interleaved kernel to reach for.
func (t *packTuner) width() int {
	if t == nil {
		return 0
	}
	return t.cur
}

// initialWidths is what AddShape generates up front: the incumbent, plus the
// first challenger when a climb is pending.
func (t *packTuner) initialWidths() []int {
	if t == nil || t.best < 2 {
		return nil
	}
	if !t.on {
		return []int{t.best}
	}
	return []int{t.best, t.chal}
}

// begin picks the width for this token and starts the clock.
func (t *packTuner) begin() {
	if t == nil || !t.on {
		return
	}
	// ABBA, not ABAB: a decode token gets more expensive as the KV cache fills,
	// and strict alternation would always charge that growth to the challenger.
	// Best on turns 0 and 3, challenger on 1 and 2, so drift that is linear
	// across four tokens cancels exactly.
	if q := t.turn % 4; q == 0 || q == 3 {
		t.cur = t.best
	} else {
		t.cur = t.chal
	}
	t.t0 = time.Now()
}

// end records the token and, when a duel is complete, decides it.
func (t *packTuner) end(f *JIT) {
	if t == nil || !t.on || t.t0.IsZero() {
		return
	}
	d := time.Since(t.t0)
	t.t0 = time.Time{}
	if t.skip < tuneSkip {
		t.skip++
		return
	}
	q := t.turn % 4
	t.quad[q] = d
	t.turn++
	if q != 3 {
		return
	}
	// The median of per-quad ratios, not a ratio of medians: pairing inside the
	// quad cancels drift before the division. Two ratios per quad, (0,1) with
	// the incumbent first and (3,2) with it second, so the later slot's extra
	// cost appears in both directions.
	if t.quad[0] > 0 && t.quad[1] > 0 && t.quad[2] > 0 && t.quad[3] > 0 {
		t.rat = append(t.rat,
			float64(t.quad[0])/float64(t.quad[1]),
			float64(t.quad[3])/float64(t.quad[2]))
	}
	if len(t.rat) < tuneRound {
		return
	}
	med, ok := medianIQRf(t.rat)
	t.rat = t.rat[:0]
	if !ok {
		t.settle(f, false) // too noisy to believe; keep the table's answer
		return
	}
	if !t.quiet {
		fmt.Fprintf(os.Stderr, "jitllm: tune pack%d vs pack%d: %.4fx (>1 means the challenger is faster)\n",
			t.best, t.chal, med)
	}
	// med is incumbent time over challenger time, so the challenger wins by more
	// than the margin exactly when med exceeds 1/tuneMargin.
	if med <= 1/tuneMargin {
		t.settle(f, true)
		return
	}
	t.best = t.chal
	if t.chal >= cpu.Pack8 {
		t.settle(f, true)
		return
	}
	t.chal++
	f.ensureWidth(t.chal)
}

func (t *packTuner) settle(f *JIT, trust bool) {
	t.on, t.chal, t.cur = false, 0, t.best
	if trust && t.key != "" {
		storeTuned(t.key, t.best)
	}
	if !t.quiet {
		fmt.Fprintf(os.Stderr, "jitllm: tune settled on pack%d\n", t.best)
	}
	f.dropUnusedWidths(t.best)
}

// medianIQRf is the median of per-round ratios, and whether their IQR over that
// median is within tuneMaxIQR.
func medianIQRf(r []float64) (float64, bool) {
	s := append([]float64(nil), r...)
	sort.Float64s(s)
	med := s[len(s)/2]
	iqr := s[len(s)*3/4] - s[len(s)/4]
	return med, med > 0 && iqr/med <= tuneMaxIQR
}

// tuneKey is (parameter, CPU, cores, shape set), the cache key of every tuned
// choice. The shape set stands in for "the model" deliberately: two files with
// the same (type, blocks-per-row) inventory pose the identical problem, and a
// finetune should not force a re-tune.
//
// cores must be the pool's maximum, never its current participant count: that
// count is itself tuned, so keying on it would move the key mid-tune.
func tuneKey(param string, shapes []shapeKey, cores int) string {
	h := fnv.New64a()
	sigs := make([]string, 0, len(shapes))
	for _, s := range shapes {
		sigs = append(sigs, fmt.Sprintf("%s:%d", s.t, s.nb))
	}
	sort.Strings(sigs)
	for _, s := range sigs {
		h.Write([]byte(s))
		h.Write([]byte{0})
	}
	h.Write([]byte(hostSig()))
	return fmt.Sprintf("%s|%s|%s|%d|%d|%016x",
		param, runtime.GOOS, runtime.GOARCH, runtime.NumCPU(), cores, h.Sum64())
}

// probeTypes is a fixed probe set, not the model's types: hostSig describes
// the host, so asking about a format this model does not contain is the point.
var probeTypes = []quant.Type{quant.Q4_0, quant.Q5_0, quant.Q8_0, quant.Q4_K, quant.Q3_K, quant.Q5_K, quant.Q6_K}

// hostSig describes the machine the way the tuner cares about it: what code
// will be generated here, not what the chip is called. It combines the
// emitter's capability answers with a hash of natively emitted reference
// kernels, so editing any emitter invalidates answers measured against the old
// code; tuneKey adds GOOS, GOARCH and NumCPU.
//
// It does not capture the microarchitecture: two chips with the same CPU count
// collide. The fix for that is re-validating a cached answer on load, not a
// longer key.
func hostSig() string {
	var b strings.Builder
	// Asked with a fixed probe type, so this reports the emitter's capability,
	// not an answer about one file.
	fmt.Fprintf(&b, "w%d;", cpu.WideActWindow([]quant.Type{quant.Q4_K}))
	// The unpinned native default, not this JIT's pin: a pin must not change
	// every tuner's cache key.
	for _, t := range probeTypes {
		fmt.Fprintf(&b, "%d,", cpu.PackWidthNative(0, t))
	}
	// The fingerprint must come from the native emitter (EmitNative), not a
	// host-pure one that emits the same bytes on every architecture.
	f := fnv.New64a()
	for _, t := range probeTypes {
		if code, err := cpu.EmitNative(cpu.Spec{W: t, Rows: 1, Cols: 1, Accs: 1}); err == nil {
			f.Write(code)
		}
	}
	fmt.Fprintf(&b, "e%016x", f.Sum64())
	// The tier, when it is not the primary one: EmitNative's quantized bytes
	// are the AVX2 code, so a forced-SSE run would otherwise share the AVX2
	// run's answers. Appended only off the primary tier so existing keys stay
	// valid.
	if t := cpu.HostTier(); t == cpu.TierSSE || t == cpu.TierNone {
		fmt.Fprintf(&b, ";tier=%s", t)
	}
	return b.String()
}

func tuneCachePath() string {
	dir, err := os.UserCacheDir()
	if err != nil {
		return ""
	}
	return filepath.Join(dir, "jitllm", "packwidth")
}

func lookupTuned(key string, mode TuneMode) (int, bool) {
	tuneCacheOnce.Do(func() { tuneCache = loadPackCache() })
	if mode == TuneForce {
		return 0, false // forced re-tune
	}
	tuneCacheMu.Lock()
	defer tuneCacheMu.Unlock()
	w, ok := tuneCache[key]
	return w, ok
}

// storeTuned is best-effort. A read-only HOME costs a re-tune next run, which
// is a handful of slightly-wrong iterations, not a failure worth surfacing.
func storeTuned(key string, w int) {
	tuneCacheOnce.Do(func() { tuneCache = loadPackCache() })
	tuneCacheMu.Lock()
	defer tuneCacheMu.Unlock()
	tuneCache[key] = w
	p := tuneCachePath()
	if p == "" || os.MkdirAll(filepath.Dir(p), 0o755) != nil {
		return
	}
	keys := make([]string, 0, len(tuneCache))
	for k := range tuneCache {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var sb strings.Builder
	for _, k := range keys {
		fmt.Fprintf(&sb, "%s\t%d\n", k, tuneCache[k])
	}
	os.WriteFile(p, []byte(sb.String()), 0o644)
}

func loadPackCache() map[string]int {
	m := map[string]int{}
	p := tuneCachePath()
	if p == "" {
		return m
	}
	f, err := os.Open(p)
	if err != nil {
		return m
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		key, val, ok := strings.Cut(sc.Text(), "\t")
		if !ok {
			continue
		}
		var w int
		if _, err := fmt.Sscanf(val, "%d", &w); err == nil && w >= 1 {
			m[key] = w
		}
	}
	return m
}

// gemmCandidates are the prefill GEMM register tiles the tuner tries: mr weight
// rows by nr groups of eight tokens. Which wins is a property of the register
// file, ports and caches, so the list spans the design space (tall, square,
// wide, and 1x1 as the floor) rather than one machine's ranking.
//
// Every token width 8*nr must divide the prefill chunk: MatMul zero-pads a
// short tile, so a width that does not divide the chunk does real work on
// zeros in every chunk (a 24-token tile lost to 8- and 16-token ones end to
// end despite winning the single-core kernel sweep). init enforces it.
var gemmCandidates = []gemmTile{
	{4, 2}, {6, 2}, {2, 2}, {3, 4}, {2, 4}, {8, 1}, {4, 1}, {1, 1},
}

// init refuses a candidate whose token width does not tile a chunk exactly;
// see gemmCandidates.
func init() {
	for _, c := range gemmCandidates {
		if PrefillChunkTokens%(8*c.nr) != 0 {
			panic(fmt.Sprintf("nn: gemm candidate %dx%d is %d tokens wide, which does not divide the %d-token prefill chunk",
				c.mr, c.nr, 8*c.nr, PrefillChunkTokens))
		}
	}
}

// PrefillChunkTokens is the SMALLEST prefill batch width. Every candidate width
// is a multiple of it, so a tile that divides this divides all of them. It is a
// constant here rather than imported because model imports nn, not the reverse.
const PrefillChunkTokens = 32

// defaultTile is used when tuning is off or a measurement was refused. It is
// mid-range on purpose: register-feasible everywhere, wide enough to amortize
// the broadcast, narrow enough not to depend on twelve accumulators fitting.
var defaultTile = gemmTile{4, 2}

// gemmTuner runs one shape's candidates round-robin across real MatMul calls.
// It compares rates (macs/second), not durations, so calls of different sizes
// are comparable.
type gemmTuner struct {
	on    bool
	cands []gemmTile
	i     int
	best  gemmTile
	rates map[gemmTile][]float64
	key   string
	quiet bool // Config.Quiet, carried like packTuner's
}

// gemmTile picks the tile for one shape, consulting the cache before measuring.
func (f *JIT) gemmTile(t quant.Type, nb, rows, ntok int) gemmTile {
	if tl, ok := parseTile(f.cfg.Tile); ok {
		return fitTile(tl, ntok)
	}
	key := rateKey{t: t, nb: nb, rows: rows, ntok: ntok}
	if tl, ok := f.gtile[key]; ok {
		return tl
	}
	// One tuner at a time, in order: participants, then batch width, then
	// tiles. Each earlier one changes the conditions the later ones measure
	// under, and two cycling at once make the IQR gate refuse both.
	prefillDuels.Lock()
	cycling := (f.ptune != nil && f.ptune.on) || (f.ctune != nil && f.ctune.on) ||
		(f.wtune != nil && f.wtune.on) || (f.rtune != nil && f.rtune.on)
	prefillDuels.Unlock()
	if cycling {
		return fitTile(defaultTile, ntok)
	}
	if f.gtune == nil {
		f.gtune = map[rateKey]*gemmTuner{}
	}
	g := f.gtune[key]
	if g == nil {
		// Only candidates this target can emit (a candidate that cannot would
		// never collect samples), narrowed to widths the batch can fill.
		cands := make([]gemmTile, 0, len(gemmCandidates))
		for _, c := range gemmCandidates {
			c = fitTile(c, ntok)
			// Feasibility is asked at the window gemmKernel will emit with:
			// the wide arm64 path needs more registers, and a tile refused
			// there would silently drop prefill to the per-token loop.
			if _, err := cpu.EmitGEMMWindow(t, c.mr, c.nr, nb, f.actWindow); err != nil {
				continue
			}
			if !slices.Contains(cands, c) {
				cands = append(cands, c)
			}
		}
		g = &gemmTuner{on: f.cfg.Tune != TuneOff && len(cands) > 1, best: fitTile(defaultTile, ntok),
			cands: cands, rates: map[gemmTile][]float64{}, quiet: f.cfg.Quiet,
			key: tuneKey("gemm"+itoaN(nb)+"x"+itoaN(rows)+"n"+itoaN(ntok)+"_"+t.String(), f.keyShapes, f.pool.Max())}
		if len(cands) > 0 {
			g.best = cands[0]
		}
		if w, ok := lookupTuned(g.key, f.cfg.Tune); ok {
			if tl, ok := decodeTile(w); ok {
				g.on = false
				g.best = tl
				f.gtile[key] = tl
			}
		}
		f.gtune[key] = g
	}
	if !g.on {
		return g.best
	}
	return g.cands[g.i%len(g.cands)]
}

// gemmObserve records one real call's throughput and settles when every
// candidate has enough samples.
func (f *JIT) gemmObserve(t quant.Type, nb, rows, ntok int, tl gemmTile, macs float64, d time.Duration) {
	key := rateKey{t: t, nb: nb, rows: rows, ntok: ntok}
	g := f.gtune[key]
	if g == nil || !g.on || d <= 0 {
		return
	}
	g.rates[tl] = append(g.rates[tl], macs/d.Seconds())
	g.i++
	if len(g.rates[tl]) < tuneRound || len(g.rates) < len(g.cands) {
		return
	}
	for _, c := range g.cands {
		if len(g.rates[c]) < tuneRound {
			return
		}
	}
	best, bestRate, ok := g.best, 0.0, false
	for _, c := range g.cands {
		r, stable := medianRate(g.rates[c])
		if !stable {
			continue
		}
		if r > bestRate {
			best, bestRate, ok = c, r, true
		}
	}
	g.on = false
	if !ok {
		// Every candidate was too dispersed to believe. Keep the default,
		// cache nothing, and say so: a silent decline looks like no tuner.
		f.gtile[key] = g.best
		if !g.quiet {
			fmt.Fprintf(os.Stderr, "jitllm: tune %s k%d rows%d gemm tile: no candidate stable enough, keeping %dx%d\n",
				t, nb, rows, g.best.mr, g.best.nr)
		}
		return
	}
	g.best = best
	f.gtile[key] = best
	storeTuned(g.key, encodeTile(best))
	if !g.quiet {
		fmt.Fprintf(os.Stderr, "jitllm: tune %s k%d rows%d gemm tile %dx%d at %.1f Gmac/s\n",
			t, nb, rows, best.mr, best.nr, bestRate/1e9)
	}
}

// fitTile narrows a tile's token width to one the batch can fill. nr is 1, 2 or
// 4 -- 8, 16 or 32 tokens -- and the columns a short batch does not supply are
// zeros the kernel multiplies anyway.
func fitTile(tl gemmTile, ntok int) gemmTile {
	for tl.nr > 1 && cpu.GEMMTokens(tl.nr) > ntok {
		tl.nr /= 2
	}
	return tl
}

// medianRate reports the median and whether the spread is tight enough to act
// on -- the same IQR/median gate the rest of the harness uses.
func medianRate(v []float64) (float64, bool) {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	med := s[len(s)/2]
	iqr := s[len(s)*3/4] - s[len(s)/4]
	return med, med > 0 && iqr/med <= tuneMaxIQR
}

func encodeTile(t gemmTile) int { return t.mr*100 + t.nr }

func decodeTile(v int) (gemmTile, bool) {
	mr, nr := v/100, v%100
	if mr < 1 || nr < 1 || mr*nr > 12 {
		return gemmTile{}, false
	}
	return gemmTile{mr, nr}, true
}

// parseTile reads WithGEMMTile's "<mr>x<nr>", the manual override for someone who
// already knows the answer for their machine.
func parseTile(s string) (gemmTile, bool) {
	mr, nr, ok := strings.Cut(s, "x")
	if !ok {
		return gemmTile{}, false
	}
	a, err1 := strconv.Atoi(mr)
	b, err2 := strconv.Atoi(nr)
	if err1 != nil || err2 != nil || a < 1 || b < 1 || a*b > 12 {
		return gemmTile{}, false
	}
	return gemmTile{a, b}, true
}

// PrefillChunkRounds is the ratios a prefill duel collects per step. It is
// lower than tuneRound because a prompt supplies only a handful of chunks;
// three is enough for a median and an IQR, and the cache carries the answer to
// later requests.
const PrefillChunkRounds = 3

// PrefillChunks are the batch widths Prefill may use. Every one is a multiple
// of 32, the largest tile width the GEMM candidates offer, so the no-padding
// invariant holds at any of them. The wide ones pay with the weight-stationary
// GEMM, which reads each weight once per chunk; a mixture's expert sees only
// chunk*used/experts rows of it.
var PrefillChunks = []int{32, 64, 128, 256, 512}

// MaxHostPrefillChunk sizes the batched scratch in model: the largest
// candidate, not the tuned value, because the tuner may still be cycling.
const MaxHostPrefillChunk = 512

// MaxPrefillChunk is the narrower sub-chunk a device that cannot hold its
// wide scratch falls back to (tier.Layers).
const MaxPrefillChunk = 128

// MaxDevicePrefillChunk is the widest chunk a device's batched prefill takes in
// one submission: every chunk re-reads the weights and ends in a host round
// trip, so fewer, wider chunks pay. A device that cannot hold the scratch for
// it runs narrower sub-chunks (tier.Layers).
const MaxDevicePrefillChunk = 512

// ensureTuners builds the pool-level tuners on first use. They cannot be built
// in NewJIT because the shape inventory that keys their cache is not complete
// until every AddShape has run.
func (f *JIT) ensureTuners() {
	prefillDuels.Lock()
	defer prefillDuels.Unlock()
	f.ensureTunersLocked()
}

// ensureTunersLocked is ensureTuners with prefillDuels held.
//
// A duel in progress belongs to the process, not to the JIT that started it:
// every model.State builds its own JIT and only a settled answer reaches the
// disk cache, so a duel longer than one prompt would restart forever. Keyed
// like the answer (shapes and cores), the next State continues it.
func (f *JIT) ensureTunersLocked() {
	if f.ptune != nil {
		return
	}
	f.ptune = sharedDuel(tuneKey("part", f.keyShapes, f.pool.Max()), func() *duelTuner {
		return f.newPartDuel(f.pool.Max(), tuneKey("part", f.keyShapes, f.pool.Max()))
	})
	f.ctune = sharedDuel(tuneKey("chunk", f.keyShapes, f.pool.Max()), func() *duelTuner {
		return f.newChunkDuel(f.keyShapes, f.pool.Max())
	})
	f.wtune = sharedDuel(tuneKey("gemmrows", f.keyShapes, f.pool.Max()), func() *duelTuner {
		return f.newGEMMRowsDuel(f.keyShapes, f.pool.Max())
	})
	f.rtune = sharedDuel(tuneKey("rowchunk", f.keyShapes, f.pool.Max()), func() *duelTuner {
		return f.newRowChunkDuel(f.keyShapes, f.pool.Max())
	})
}

// prefillDuels guards the prefill duels, which several JITs may share, and the
// registry of the ones still running.
var prefillDuels struct {
	sync.Mutex
	live map[string]*duelTuner
}

// sharedDuel returns the running duel for key, or makes one with mk and
// registers it while it runs. Callers hold prefillDuels.
func sharedDuel(key string, mk func() *duelTuner) *duelTuner {
	if d := prefillDuels.live[key]; d != nil && d.on {
		return d
	}
	d := mk()
	if d.on {
		if prefillDuels.live == nil {
			prefillDuels.live = map[string]*duelTuner{}
		}
		prefillDuels.live[key] = d
	}
	return d
}

// RowChunkKiB are the batched row chunk caps the duel tries, in KiB of a
// worker's weight slice; the first is no cap at all.
var RowChunkKiB = []int{1 << 20, 256, 64, 16}

// newRowChunkDuel measures the cap on a batched matmul's row chunk
// (Config.ChunkBytes) on whole prefill chunks, after the participants and the
// token chunk have settled.
//
// The answer depends on the pool: on identical cores a cap only shortens the
// contiguous runs and loses, while on a P+E pool it balances slow cores
// against fast ones and wins (see docs/engineering-history/cpu-kernels.md).
// Every rung is tried (sweep) because the curve is not monotone.
func (f *JIT) newRowChunkDuel(shapes []shapeKey, cores int) *duelTuner {
	pin := 0
	if f.cfg.ChunkBytes > 0 {
		pin = max(1, f.cfg.ChunkBytes>>10)
	}
	d := newDuelTuner("prefill row chunk KiB", pin, tuneKey("rowchunk", shapes, cores),
		RowChunkKiB, f.cfg.Tune, f.cfg.Quiet)
	d.sweep = true
	return d
}

// GEMMRowsCands are the weight-stationary GEMM's rows a call the gemmrows duel
// tries; gemmOff is "no GEMM, the tiled path". Candidates are >= 1 because
// that is how the duel encodes a pin and the cache an answer.
var GEMMRowsCands = []int{gemmRows, 32, 64, gemmOff}

const gemmOff = 1

// newGEMMRowsDuel measures the weight-stationary GEMM's rows a call, and
// whether to run it at all, on whole prefill chunks. Both answers depend on the
// host (narrow rows keep many workers busy but share cache lines on a 128-byte
// line host) and on the model's format mix.
func (f *JIT) newGEMMRowsDuel(shapes []shapeKey, cores int) *duelTuner {
	pin := 0
	switch {
	case f.cfg.GEMMRows > 0:
		pin = max(8, f.cfg.GEMMRows)
	case f.cfg.GEMMTok < 0 || f.cfg.NoGEMM || f.em.PackedGEMM == nil:
		pin = gemmRows // the GEMM cannot run, so there is nothing to measure
	}
	d := newDuelTuner("prefill GEMM rows", pin, tuneKey("gemmrows", shapes, cores),
		GEMMRowsCands, f.cfg.Tune, f.cfg.Quiet)
	d.sweep = true
	return d
}

// PrefillChunk is how many tokens the next prefill batch should carry.
//
// Wider batches amortize the weight read over more tokens but cost scratch
// memory linearly, and the returns flatten; which width wins is a property of
// the cache hierarchy, so it is measured.
func (f *JIT) PrefillChunk() int {
	if f == nil {
		return PrefillChunks[0]
	}
	prefillDuels.Lock()
	defer prefillDuels.Unlock()
	f.ensureTunersLocked()
	if f.onE {
		// The E-cores' pool is not the one the tuners climb on: every region
		// takes its whole crew, at the width and sizes already settled.
		f.pool.SetParticipants(f.pool.Max())
		f.chunkBytes, f.gemmRows = f.rtune.best<<10, f.wtune.best
		return f.ctune.best
	}
	// Applying the participant count here, per chunk, is what lets it be
	// measured on the same end-to-end number as the batch width: it governs
	// every region of the chunk, not just the matmul.
	f.pool.SetParticipants(f.ptune.pick())
	// One at a time, in order: participants, batch width, GEMM rows, row
	// chunk -- each is measured with the ones before it settled.
	f.chunkBytes, f.gemmRows = f.rtune.best<<10, f.wtune.best
	switch {
	case f.ptune.on:
		return f.ctune.best
	case f.ctune.on:
		return f.ctune.pick()
	case f.wtune.on:
		f.gemmRows = f.wtune.pick()
	default:
		f.chunkBytes = f.rtune.pick() << 10
	}
	return f.ctune.best
}

// ObserveChunk records one prefill chunk's end-to-end throughput. The pool
// parameters are tuned on the whole chunk, not the matmul rate: batch width
// and participant count also govern attention, norms and elementwise ops,
// which the matmul rate cannot see.
func (f *JIT) ObserveChunk(width, ntok int, d time.Duration) {
	if f == nil || f.ctune == nil || f.onE || d <= 0 || ntok != width {
		return // a short final chunk is not a sample
	}
	prefillDuels.Lock()
	defer prefillDuels.Unlock()
	rate := float64(ntok) / d.Seconds()
	if f.ptune.on {
		f.ptune.observe(rate)
		return
	}
	if f.ctune.on {
		f.ctune.observe(rate)
		return
	}
	if f.wtune.on {
		f.wtune.observe(rate)
		return
	}
	f.rtune.observe(rate)
}

// duelTuner hill-climbs one pool-level parameter using paired adjacent chunks.
// Chunk throughput falls through a prompt as attention grows, so absolute
// rates measure position; adjacent chunks' ratio cancels that drift.
type duelTuner struct {
	label  string
	on     bool
	best   int
	chal   int
	cands  []int
	turn   int
	quad   [4]float64 // one ABBA quad of chunk rates
	ratios []float64
	key    string
	quiet  bool // Config.Quiet, carried like the other two tuners'
	// rounds and margin override PrefillChunkRounds and tuneMargin2 when set.
	rounds int
	margin float64
	offset int // subtracted from best when printed (fpfTuner stores d+1)
	// firstMargin, when set, is the margin for leaving cands[0] -- a step
	// that can cost something the later steps do not.
	firstMargin float64
	// sweep keeps dueling the remaining candidates against the incumbent when
	// a challenger loses, where the default climb stops at the first loss.
	// For a ladder whose rungs are not ordered by cost (the fused prefetch).
	sweep bool
}

// newDuelTuner builds a duel over cands. A pin >= 1 fixes the answer; a cached
// answer is used when present.
func newDuelTuner(label string, pin int, key string, cands []int, mode TuneMode, quiet bool) *duelTuner {
	c := &duelTuner{label: label, key: key, best: cands[0], cands: cands, quiet: quiet}
	if pin >= 1 {
		c.best = pin
		return c
	}
	if w, ok := lookupTuned(key, mode); ok && w >= 1 {
		c.best = w
		return c
	}
	if len(cands) < 2 || mode == TuneOff {
		return c
	}
	c.on, c.chal = true, cands[1]
	return c
}

// newChunkDuel climbs the prefill batch width from the narrowest candidate.
func (f *JIT) newChunkDuel(shapes []shapeKey, cores int) *duelTuner {
	return newDuelTuner("prefill chunk", f.cfg.Chunk, tuneKey("chunk", shapes, cores),
		PrefillChunks, f.cfg.Tune, f.cfg.Quiet)
}

// newPartDuel climbs the participant count downward from every core. Using
// every core can lose when the workers, the caller and Go's runtime threads do
// not all fit, since one descheduled worker holds up every barrier.
func (f *JIT) newPartDuel(max int, key string) *duelTuner {
	cands := []int{max}
	for n := max - 1; n >= max-3 && n >= 2; n-- {
		cands = append(cands, n)
	}
	d := newDuelTuner("participants", f.cfg.Part, key, cands, f.cfg.Tune, f.cfg.Quiet)
	// Dropping a worker needs 5% over six ratios: the answer also applies to
	// decode, where one worker fewer is one core's worth of outstanding misses
	// fewer, and a 2% bar settled below the full count on noise.
	d.rounds, d.margin = 2*PrefillChunkRounds, partMargin
	return d
}

// partMargin is what one participant fewer must buy; see newPartDuel.
const partMargin = 1.05

func (c *duelTuner) pick() int {
	if c == nil {
		return 0
	}
	// ABBA for the same reason packTuner uses it: chunks get more expensive as
	// the context grows, so the incumbent must not always take the earlier slot.
	if q := c.turn % 4; !c.on || q == 0 || q == 3 {
		return c.best
	}
	return c.chal
}

// observe takes one chunk's throughput and, on every challenger chunk, the
// ratio against the incumbent chunk that immediately preceded it.
func (c *duelTuner) observe(rate float64) {
	if c == nil || !c.on || rate <= 0 {
		return
	}
	q := c.turn % 4
	c.quad[q] = rate
	c.turn++
	if q != 3 {
		return
	}
	// Two ratios per quad, incumbent first in one and second in the other.
	if c.quad[0] > 0 && c.quad[1] > 0 && c.quad[2] > 0 && c.quad[3] > 0 {
		c.ratios = append(c.ratios, c.quad[1]/c.quad[0], c.quad[2]/c.quad[3])
	}
	rounds, margin := PrefillChunkRounds, tuneMargin2
	if c.rounds > 0 {
		rounds, margin = c.rounds, c.margin
	}
	if c.firstMargin > 0 && c.best == c.cands[0] {
		margin = c.firstMargin
	}
	if len(c.ratios) < rounds {
		return
	}
	med, _ := medianRate(c.ratios) // paired inside the quad, so the drift is already out
	c.ratios = c.ratios[:0]
	if med <= margin {
		if next := c.after(c.chal); c.sweep && next != 0 {
			c.chal = next
			return
		}
		c.settle()
		return
	}
	c.best = c.chal
	next := c.after(c.best)
	if next == 0 {
		c.settle()
		return
	}
	c.chal = next
}

// after is the candidate after v in the ladder, or 0 at its end.
func (c *duelTuner) after(v int) int {
	for i, x := range c.cands {
		if x == v && i+1 < len(c.cands) {
			return c.cands[i+1]
		}
	}
	return 0
}

func (c *duelTuner) settle() {
	c.on = false
	storeTuned(c.key, c.best)
	if !c.quiet {
		fmt.Fprintf(os.Stderr, "jitllm: tune %s = %d\n", c.label, c.best-c.offset)
	}
}

// tuneMargin2 is the improvement a wider batch must show to justify the memory
// it costs. Biased toward the narrow end, which is the cheap direction.
const tuneMargin2 = 1.02
