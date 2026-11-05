//go:build amd64 || arm64

package engine

import (
	"os"
	"runtime"
	"slices"
	"testing"
	"time"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestRegionLabelCost prices what a pprof region label costs a token, in
// allocations rather than time, so it needs no quiet box. Each labelled region
// builds its label by concatenation and runs pprof.Do for every participant
// (a label map and a context each). The arms differ only in their
// instance's pool's labels (cpu.Pool.SetRegionLabels).
func TestRegionLabelCost(t *testing.T) {
	if os.Getenv("JITLLM_LABELCOST") == "" {
		t.Skip("set JITLLM_LABELCOST=1 to price the pprof region labels")
	}
	name := envOr("JITLLM_LABELCOST_MODEL", "tinyllama-1.1b-q3_K_M.gguf")
	path := jlmOf(t, testmodels.Path(name))
	m, err := model.Open(path)
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()

	// allocs and bytes for `n` decode tokens, after a warm-up that settles the
	// tuner and the kernel caches so their one-off allocations are not counted.
	perToken := func(labels bool, n int) (allocs, bytes float64, regions float64) {
		s := testInstance(t, m, 64)
		defer s.Close()
		s.Pool().SetRegionLabels(labels)
		for i := 0; i < 8; i++ {
			if _, err := s.Forward(int32(5 + i)); err != nil {
				t.Fatal(err)
			}
		}
		var a, b runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&a)
		r0, _ := s.Pool().Regions()
		for i := 0; i < n; i++ {
			if _, err := s.Forward(int32(3 + i%64)); err != nil {
				t.Fatal(err)
			}
		}
		runtime.ReadMemStats(&b)
		r1, _ := s.Pool().Regions()
		return float64(b.Mallocs-a.Mallocs) / float64(n),
			float64(b.TotalAlloc-a.TotalAlloc) / float64(n),
			float64(r1-r0) / float64(n)
	}

	const n = 24
	onA, onB, reg := perToken(true, n)
	offA, offB, _ := perToken(false, n)

	t.Logf("%s, %.1f parallel regions/token", name, reg)
	t.Logf("labels ON   %8.1f allocs/token  %10.0f B/token", onA, onB)
	t.Logf("labels OFF  %8.1f allocs/token  %10.0f B/token", offA, offB)
	t.Logf("the label   %8.1f allocs/token  %10.0f B/token  (%.1f per region)",
		onA-offA, onB-offB, (onA-offA)/max(reg, 1))

	// The arms must actually differ, or this priced one configuration twice.
	if onA <= offA {
		t.Fatalf("labels ON allocated %.1f/token and OFF %.1f -- the switch did not "+
			"reach the pool, so nothing was priced", onA, offA)
	}
	if reg < 1 {
		t.Fatalf("%.1f regions per token: the model ran no parallel region and there was "+
			"no label to pay for", reg)
	}
}

// TestRegionLabelRate asks the question the allocation count cannot: do the
// per-token allocations cost any time? The bytes are negligible against weight
// traffic; this asks whether the GC pressure reaches the stopwatch.
//
// ABBA in one process, A/A self-control first, two passes, IQR/median gated at
// 0.10, cores recorded at each pass's start and finish.
func TestRegionLabelRate(t *testing.T) {
	if os.Getenv("JITLLM_LABELRATE") == "" {
		t.Skip("set JITLLM_LABELRATE=1 to price the labels in tok/s")
	}
	if b, err := coreBusy(time.Second); err == nil && b > 20 {
		t.Skipf("the cores this pins to are %.1f%% busy; a rate row needs them idle", b)
	}
	name := envOr("JITLLM_LABELRATE_MODEL", "tinyllama-1.1b-q3_K_M.gguf")
	rounds := envInt("JITLLM_LABELRATE_ROUNDS", 12)
	path := jlmOf(t, testmodels.Path(name))
	m, err := model.Open(path)
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()

	sample := func(labels bool) float64 {
		s := testInstance(t, m, 128)
		defer s.Close()
		s.Pool().SetRegionLabels(labels)
		for i := 0; i < 8; i++ {
			if _, err := s.Forward(int32(5 + i)); err != nil {
				t.Fatal(err)
			}
		}
		const meas = 48
		t0 := time.Now()
		for i := 0; i < meas; i++ {
			if _, err := s.Forward(int32(3 + i%64)); err != nil {
				t.Fatal(err)
			}
		}
		return float64(meas) / time.Since(t0).Seconds()
	}

	pass := func(a, b bool) ([]float64, float64, float64, float64, float64) {
		b0, _ := coreBusy(300 * time.Millisecond)
		var r, ra, rb []float64
		for i := 0; i < rounds; i++ {
			x1, y1 := sample(a), sample(b)
			y2, x2 := sample(b), sample(a)
			r = append(r, x1/y1, x2/y2)
			ra = append(ra, x1, x2)
			rb = append(rb, y1, y2)
		}
		b1, _ := coreBusy(300 * time.Millisecond)
		return r, median(ra), median(rb), b0, b1
	}

	ctl, _, _, c0, c1 := pass(false, false)
	cm, craw := medIQR(ctl)
	t.Logf("A/A self-control (labels OFF both arms)   %.4f   IQR/median %.3f   n=%d   cores %.1f%% -> %.1f%%",
		cm, craw/cm, len(ctl), c0, c1)

	for p := 1; p <= 2; p++ {
		// OFF / ON: above 1 means removing the labels is FASTER.
		r, roff, ron, b0, b1 := pass(false, true)
		med, raw := medIQR(r)
		gate := "GATES"
		if raw/med > 0.10 {
			gate = "REJECTED (IQR/median over 0.10)"
		}
		if b0 > 20 || b1 > 20 {
			gate = "REJECTED (a tenant was on the cores)"
		}
		t.Logf("pass %d  labels OFF / ON   %.4f   IQR/median %.3f   n=%d   cores %.1f%% -> %.1f%%   %s",
			p, med, raw/med, len(r), b0, b1, gate)
		t.Logf("        off %.2f tok/s   on %.2f tok/s", roff, ron)
	}
}

// medIQR is v's median and interquartile range.
func medIQR(v []float64) (med, iqr float64) {
	s := slices.Sorted(slices.Values(v))
	return s[len(s)/2], s[3*len(s)/4] - s[len(s)/4]
}
