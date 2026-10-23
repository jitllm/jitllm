package tier

import (
	"fmt"
	"os"
	"sort"
	"time"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// gemmDeepMargin is how far under f16Tiles' pick the deep tiles' median
// ratio must fall to be taken: inside it the trial cannot tell the two apart.
const gemmDeepMargin = 0.98

// tuneGemm times, per prompt GEMM shape of the placed blocks, the deep m16n8
// tiles (f16DeepTiles: eight warps, 64 of k a trip, one buffer) against
// f16Tiles' pick at the reserved prompt width, on the resident weights, and
// records the winner in gemmDeep for f16Gemm. Neither wins everywhere: on a
// T4 the deep tile read 0.91-0.95 of the shipping one on Llama-3.1-8B's k and
// Q4_K down projections and 1.00-1.02 on gate/up and the Q6_K down
// (gpu-kernels.md, "The deep stage"). A verdict is cached per device, driver
// and executable (WithTuneCache). Callers hold g.mu, outside any Session.
func (g *devTier) tuneGemm() {
	if g.bs == nil || g.promptW <= 0 || g.NoVolta || g.Config.GemmDeep != 0 || gemmDeepDefault != 0 {
		return
	}
	k16 := g.f16GemmK()
	if k16 == 0 {
		return
	}
	if g.gemmDeep == nil {
		g.gemmDeep = map[splitKey]bool{}
	}
	W := g.promptW
	type use struct {
		m mv
		r *resident
	}
	by := map[splitKey]use{}
	var keys []splitKey
	for _, l := range g.layers {
		if l == nil || !l.ok || l.nonCausal || l.linear || (l.pg != nil && !l.pg.in) {
			continue
		}
		for _, u := range []use{{l.mvq, l.wq}, {l.mvk, l.wk}, {l.mvv, l.wv}, {l.mvo, l.wo},
			{l.mvg, l.gate}, {l.mvu, l.up}, {l.mvd, l.down}} {
			if u.r == nil || u.m.kern == nil || u.m.slots > 0 || u.m.bias != nil || kernels.IsFloat(u.m.q) ||
				!kernels.Volta70OK(u.m.q) || u.r.nrows != u.m.rows {
				continue
			}
			key := splitKey{u.m.q, u.m.rows, u.m.k}
			if _, done := g.gemmDeep[key]; done {
				continue
			}
			if _, ok := by[key]; !ok {
				by[key], keys = u, append(keys, key)
			}
		}
	}
	// The trial's builds count nothing: the stats count what the prompt runs.
	f16n, voltan, deepn := g.GemmF16, g.VoltaMV, g.Stats.GemmDeep
	defer func() { g.GemmF16, g.VoltaMV, g.Stats.GemmDeep = f16n, voltan, deepn }()
	for _, key := range keys {
		ck := g.tuneKey(fmt.Sprintf("gemm-%v-%dx%d-t%d", key.q, key.rows, key.k, W))
		if v, ok := lookupCache(g.kb.cacheDir, "gpugemm1", ck); ok && g.kb.tune != TuneForce {
			g.gemmDeep[key] = v == "1"
			continue
		}
		u := by[key]
		fill := g.dev.Slots() / 16
		ship, ok1 := g.stagedGemm(u.m, W, f16Tiles, k16, fill)
		deep, ok2 := g.stagedGemm(u.m, W, f16DeepTiles, k16, fill)
		if !ok1 || !ok2 {
			continue
		}
		r, ok := g.timeGemm(ship, deep, u.r, W)
		win := ok && r < gemmDeepMargin
		g.gemmDeep[key] = win
		if backend.Verbose() {
			fmt.Fprintf(os.Stderr, "tier: prompt GEMM %v %dx%d at %d rows: deep tile %.3f of the shipping one (resolved %v): %v\n",
				key.q, key.rows, key.k, W, r, ok, win)
		}
		if ok && ck != "" {
			v := "0"
			if win {
				v = "1"
			}
			saveCache(g.kb.cacheDir, "gpugemm1", ck, v)
		}
	}
}

// timeGemm is the median per-round ratio of b's time to a's, interleaved
// over rounds, and whether it resolved (IQR/median within splitTuneIQR).
// Each arm is its GEMM launch and, split, its Reduce, over the resident
// weights and the prompt operand buffer.
func (g *devTier) timeGemm(a, b mv, r *resident, ntok int) (float64, bool) {
	tuneTrials.Add(1)
	out, err := g.dev.Alloc(a.rows * ntok * 4)
	if err != nil {
		return 0, false
	}
	defer out.Free()
	launch := func(s backend.Session, m mv) error {
		dst := out
		if m.split > 1 {
			dst = g.partBuf
		}
		w := max(m.width, 128)
		if err := s.Launch(m.kern, (m.threads+w-1)/w, w, r.qs, r.d, r.sc, g.f16Buf, dst); err != nil {
			return err
		}
		if m.split > 1 {
			return s.Launch(m.red, (m.redN+127)/128, 128, g.partBuf, out)
		}
		return nil
	}
	timed := func(m mv) (float64, error) {
		const reps = 3
		t0 := time.Now()
		var err error
		g.dev.Session(func(s backend.Session) {
			backend.MarkTrial(s) // a profile tells a tuner's trial from the work
			for i := 0; i < reps && err == nil; i++ {
				err = launch(s, m)
			}
			if err == nil {
				err = s.Sync()
			}
		})
		return time.Since(t0).Seconds(), err
	}
	const rounds = 9
	var ratios []float64
	for round := 0; round < rounds+1; round++ {
		var ta, tb float64
		var ea, eb error
		if round%2 == 0 {
			ta, ea = timed(a)
			tb, eb = timed(b)
		} else {
			tb, eb = timed(b)
			ta, ea = timed(a)
		}
		if ea != nil || eb != nil {
			return 0, false
		}
		if round == 0 || ta <= 0 {
			continue // the first round pays first-touch
		}
		ratios = append(ratios, tb/ta)
	}
	sort.Float64s(ratios)
	med := ratios[len(ratios)/2]
	iqr := ratios[len(ratios)*3/4] - ratios[len(ratios)/4]
	return med, med > 0 && iqr/med <= splitTuneIQR
}
