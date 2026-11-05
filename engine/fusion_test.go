//go:build amd64 || arm64

package engine

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestFusionCeiling bounds what whole-layer fusion could buy without building
// it. Fusing a block runs its regions as one dispatch; the bytes, FMAs and
// kernels are unchanged, so the whole prize is regions saved times the cost of
// one region. A small model is the case most favourable to fusion (dispatch is
// the largest share of its token), so a ceiling measured there can reject
// fusion soundly but never accept it.
func TestFusionCeiling(t *testing.T) {
	// Opt-in: it loads every model and decodes 72 tokens through each, which
	// does not belong in the default correctness run.
	if os.Getenv("JITLLM_FUSION") == "" {
		t.Skip("set JITLLM_FUSION=1 to run the fusion-ceiling measurement")
	}
	if testing.Short() {
		t.Skip("allocates and measures")
	}
	// The pressure gate: a busy box once read a 16x slower rate that looked
	// like a regression.
	if p, err := cpuPressure(); err == nil && p > 25 {
		t.Skipf("CPU pressure is %.1f%% over the last 10s; a dispatch measurement "+
			"needs a quiet box (see bench.Guard on why this is PSI and not loadavg)", p)
	}
	// A sweep contaminates itself: under scripts/cap's MemoryMax, loading a
	// large model evicts the previous one's pages from the cgroup, and the
	// pressure gate does not see it. Run one model per process when the rate
	// matters:
	//     JITLLM_FUSION=1 go test ./engine/model -run 'TestFusionCeiling/tinyllama'
	// The sweep is for comparing region counts, which paging does not affect.
	paths := modelFiles()
	if p := os.Getenv("JITLLM_MODEL"); p != "" {
		paths = []string{testmodels.Resolve(p)}
	}
	if len(paths) == 0 {
		t.Skip("MODEL MISSING: no models in " + testmodels.Dir() + " (set JITLLM_MODELS to the model directory) -- this measurement needs a real one")
	}
	sort.Strings(paths)
	for _, p := range paths {
		t.Run(filepath.Base(p), func(t *testing.T) { fusionCeiling(t, p) })
	}
}

func fusionCeiling(t *testing.T, path string) {
	m, err := model.Open(jlmOf(t, path))
	if err != nil {
		t.Skipf("cannot load: %v", err)
	}
	defer m.Close()
	s := testInstance(t, m, 256)
	defer s.Close()
	// The ceiling is of the per-op path: a fused run (fuse.go) is what it
	// prices, and with it on the counters below would read one region a token.
	s.SetBlockFusion(false)

	// Warm: codegen, tuning and first-touch page faults are not per-token costs.
	for i := 0; i < 8; i++ {
		if _, err := s.Forward(int32(1 + i%16)); err != nil {
			t.Fatal(err)
		}
	}

	const tokens = 64
	s.Pool().ResetRegions()
	start := time.Now()
	for i := 0; i < tokens; i++ {
		if _, err := s.Forward(int32(1 + i%16)); err != nil {
			t.Fatal(err)
		}
	}
	elapsed := time.Since(start)
	par, ser := s.Pool().Regions()
	perTok := elapsed / tokens
	parPerTok := float64(par) / tokens
	t.Logf("%d layers, %.1f parallel regions/token, %.1f inline regions/token, %v/token (%.1f tok/s)",
		m.Cfg.NLayer, parPerTok, float64(ser)/tokens, perTok, float64(tokens)/elapsed.Seconds())

	// Per-region dispatch cost on this pool at decode's participant count, with
	// a trivial body so the number is the round trip and not the work.
	cost := dispatchCost(t)
	t.Logf("one parallel region round trip: %v", cost)

	// The cost of splitting real work, including the barrier straggler an empty
	// region cannot show. Sized like a decode elementwise region.
	split, splitIQR := splitCost(t, m.Cfg.NEmbd*8, 8)
	if split > 0 && splitIQR <= 0.10 {
		t.Logf("one region of real work (dispatch + straggler): %v  IQR/med %.1f%%", split, 100*splitIQR)
		if split > cost {
			cost = split
		}
	} else {
		t.Logf("split-cost measurement rejected (IQR/median %.0f%%); using the empty-region round trip, "+
			"which is the CONSERVATIVE choice -- see below", 100*splitIQR)
	}

	// Fusion is several levers, so report a curve. A decode block issues roughly
	// fourteen parallel regions:
	//
	//	norm | q k v | rope | attn | o | resid | norm | gate up | act | down | resid
	//
	//  1. Independent siblings: q/k/v read one normed vector, gate/up another
	//     (3 saved).
	//  2. Elementwise epilogues (rope, two residual adds, SiLU-multiply) are per
	//     element over rows the producing worker already owns, so they fold in
	//     without synchronisation (4 more).
	//  3. Reductions (two RMSNorms, the softmax) are genuine barriers and cannot
	//     fold.
	perLayer := parPerTok / float64(m.Cfg.NLayer)
	t.Logf("~%.1f parallel regions per layer", perLayer)
	for _, d := range []struct {
		saved float64
		what  string
	}{
		{3, "siblings only (q/k/v, gate/up)"},
		{7, "siblings + elementwise epilogues (rope, 2 residuals, SiLU)"},
		{perLayer - 3, "everything but the three reductions (2 norms + softmax)"},
	} {
		if d.saved <= 0 || d.saved > perLayer {
			continue
		}
		pct := 100 * d.saved * float64(m.Cfg.NLayer) * float64(cost) / float64(perTok)
		verdict := "below RULE 6's 5% bar"
		if pct >= 5 {
			verdict = "CLEARS the 5% bar"
		}
		t.Logf("  save %4.1f regions/layer -> %5.2f%% of a token   %s  (%s)",
			d.saved, pct, verdict, d.what)
	}
	ceiling := (perLayer - 3) * float64(m.Cfg.NLayer) * float64(cost) / float64(perTok)
	// Not captured: fusion could also keep the residual resident across a block
	// and fold activation quantization into the producing op. Both are small
	// while NEmbd floats split across workers stay L1-resident (~1.3 KB each at
	// gemma's width); a much wider model would change that.
	if ceiling >= 0.05 {
		t.Logf("  >= 5%%: INCONCLUSIVE here. Re-measure on a model RULE 1 names " +
			"before building anything -- this box's only model is the friendliest " +
			"case, not the shipping one.")
	} else {
		t.Logf("  < 5%%: REJECTED. Fusion cannot clear RULE 6's bar even where " +
			"dispatch is the largest share of a token it can be.")
	}
}

// splitCost measures what N regions cost against one region doing the same
// total work, which is fusion's actual prize. In situ it came out lower than an
// empty round trip (dispatch overlaps computation), so the empty-region figure
// is the conservative one and the ceiling uses whichever is larger.
func splitCost(t *testing.T, elems, nRegions int) (perRegion time.Duration, iqr float64) {
	t.Helper()
	p := sched.New(sched.DecodeCores())
	defer p.Close()
	buf := make([]float32, elems)
	for i := range buf {
		buf[i] = float32(i & 255)
	}
	// Memory-shaped work, like the elementwise regions between matvecs.
	body := func(_, lo, hi int) {
		var s float32
		for i := lo; i < hi; i++ {
			s += buf[i] * 1.000001
		}
		buf[lo] = s
	}
	timeIt := func(regions int) (time.Duration, float64) {
		per := elems / regions
		run := func() {
			for r := 0; r < regions; r++ {
				lo := r * per
				hi := lo + per
				if r == regions-1 {
					hi = elems
				}
				p.Do(hi-lo, max(1, (hi-lo)/(4*p.N())), func(w, a, b int) { body(w, lo+a, lo+b) })
			}
		}
		for i := 0; i < 50; i++ {
			run()
		}
		var ds []time.Duration
		for r := 0; r < 21; r++ {
			const reps = 20
			st := time.Now()
			for i := 0; i < reps; i++ {
				run()
			}
			ds = append(ds, time.Since(st)/reps)
		}
		sort.Slice(ds, func(i, j int) bool { return ds[i] < ds[j] })
		med := ds[len(ds)/2]
		return med, float64(ds[len(ds)*3/4]-ds[len(ds)/4]) / float64(med)
	}
	one, oneIQR := timeIt(1)
	many, manyIQR := timeIt(nRegions)
	// Refuse a dispersed answer: this is a difference of two timings, so on a
	// small model it can be a difference of noise.
	if many <= one || oneIQR > 0.10 || manyIQR > 0.10 {
		return 0, 1
	}
	per := (many - one) / time.Duration(nRegions-1)
	return per, (oneIQR + manyIQR) * float64(one+many) / float64(many-one)
}

// dispatchCost times an empty parallel region, median of 21 rounds.
func dispatchCost(t *testing.T) time.Duration {
	t.Helper()
	cores := sched.DecodeCores()
	p := sched.New(cores)
	defer p.Close()
	// Enough chunks that Do dispatches rather than running inline, with a body
	// that costs nothing.
	const total, chunk = 64, 1
	sink := make([]int64, total)
	run := func() { p.Do(total, chunk, func(_, lo, hi int) { sink[lo] += int64(hi) }) }
	for i := 0; i < 200; i++ {
		run()
	}
	var ds []time.Duration
	for r := 0; r < 21; r++ {
		const reps = 500
		s := time.Now()
		for i := 0; i < reps; i++ {
			run()
		}
		ds = append(ds, time.Since(s)/reps)
	}
	sort.Slice(ds, func(i, j int) bool { return ds[i] < ds[j] })
	med := ds[len(ds)/2]
	iqr := float64(ds[len(ds)*3/4]-ds[len(ds)/4]) / float64(med)
	if iqr > 0.10 {
		t.Logf("dispatch cost IQR/median %.1f%% -- above rule 2's gate, treat the ceiling as soft", 100*iqr)
	}
	return med
}

// cpuPressure is the fraction of the last ten seconds in which something waited
// for a CPU, the gate bench.Guard applies. Not the load average, which reads
// high on an idle box.
func cpuPressure() (float64, error) {
	b, err := os.ReadFile("/proc/pressure/cpu")
	if err != nil {
		return 0, err
	}
	for _, ln := range strings.Split(string(b), "\n") {
		if !strings.HasPrefix(ln, "some ") {
			continue
		}
		for _, f := range strings.Fields(ln) {
			if v, ok := strings.CutPrefix(f, "avg10="); ok {
				return strconv.ParseFloat(v, 64)
			}
		}
	}
	return 0, fmt.Errorf("no `some avg10` in /proc/pressure/cpu")
}
