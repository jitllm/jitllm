package model

import (
	"math"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// samplerPerfRatios drives the sampler A/B: the generated kernels against the
// Go implementation they replaced, ABBA in one process, returning the median
// per-round ratio and IQR/median. TestSamplerPerf is opt-in and needs a real
// model so the logits row and the token are the engine's own.
//
//	JITLLM_SAMPLEPERF=1 ./scripts/cap 24G -- taskset -c 0,2,4,6,8,10 \
//	  go test ./engine/model -run TestSamplerPerf -timeout 30m -v
func samplerPerfRatios(rounds int, a, b func() time.Duration) (float64, float64) {
	var r []float64
	for i := 0; i < rounds; i++ {
		// ABBA: the two orders alternate, so a monotone drift cancels.
		var ta, tb time.Duration
		if i%2 == 0 {
			ta, tb = a(), b()
			tb2, ta2 := b(), a()
			ta += ta2
			tb += tb2
		} else {
			tb, ta = b(), a()
			ta2, tb2 := a(), b()
			ta += ta2
			tb += tb2
		}
		r = append(r, float64(tb)/float64(ta)) // b over a: how much the A arm buys
	}
	sort.Float64s(r)
	med := r[len(r)/2]
	iqr := r[3*len(r)/4] - r[len(r)/4]
	return med, iqr / med
}

func TestSamplerPerf(t *testing.T) {
	if os.Getenv("JITLLM_SAMPLEPERF") == "" {
		t.Skip("set JITLLM_SAMPLEPERF=1 to run the sampler A/B")
	}
	if p, err := cpuPressure(); err == nil && p > 25 {
		t.Skipf("CPU pressure is %.1f%% over the last 10s", p)
	}
	path := testmodels.Resolve(os.Getenv("JITLLM_MODEL"))
	if path == "" {
		t.Skip("set JITLLM_MODEL to a container; this measures a real logits row")
	}
	m, err := Open(path)
	if err != nil {
		t.Fatalf("cannot load %s: %v", path, err)
	}
	defer m.Close()
	st := m.NewInstance(2048)
	defer st.Close()
	for i := 0; i < 8; i++ {
		if _, err := st.Forward(int32(1 + i%16)); err != nil {
			t.Fatal(err)
		}
	}
	logits, err := st.Forward(7)
	if err != nil {
		t.Fatal(err)
	}
	row := append([]float32(nil), logits...)
	// The realism check: a flat row and a peaked one are different measurements,
	// so say which this is.
	mx := float32(math.Inf(-1))
	for _, v := range row {
		if v > mx {
			mx = v
		}
	}
	var z float64
	for _, v := range row {
		z += math.Exp(float64(v - mx))
	}
	t.Logf("%d logits, p(top1) = %.4f", len(row), 1/z)

	cfg := struct {
		topk, lastn           int
		temp, topp, minp, pen float64
	}{40, 64, 0.8, 0.95, 0.05, 1.1}

	mkNew := func() *Sampler {
		return &Sampler{Temp: cfg.temp, TopK: cfg.topk, TopP: cfg.topp, MinP: cfg.minp,
			RepeatPen: cfg.pen, RepeatLastN: cfg.lastn, Seed: 11}
	}
	mkOld := func() *oldSampler {
		return &oldSampler{Temp: cfg.temp, TopK: cfg.topk, TopP: cfg.topp, MinP: cfg.minp,
			RepeatPen: cfg.pen, RepeatLastN: cfg.lastn, Seed: 11}
	}

	// ---- the OP, on a real logits row ----
	const perRound = 20
	drawn := 0
	opNew := func() time.Duration {
		s := mkNew()
		start := time.Now()
		for i := 0; i < perRound; i++ {
			s.Observe(s.Sample(row))
			drawn++
		}
		return time.Since(start)
	}
	opOld := func() time.Duration {
		s := mkOld()
		start := time.Now()
		for i := 0; i < perRound; i++ {
			s.Observe(s.Sample(row))
			drawn++
		}
		return time.Since(start)
	}
	aa, aaIQR := samplerPerfRatios(16, opNew, opNew)
	t.Logf("OP   A/A self-control   %.4f  IQR/median %.3f", aa, aaIQR)
	for pass := 1; pass <= 2; pass++ {
		r, iqr := samplerPerfRatios(16, opNew, opOld)
		t.Logf("OP   pass %d  generated/Go  %.2fx  IQR/median %.3f", pass, r, iqr)
	}
	if drawn == 0 {
		t.Fatal("no draw ran: this measured nothing")
	}

	// ---- the TOKEN: Forward plus the sampler ----
	const tokPerRound = 8
	sampled := 0
	tokNew := func() time.Duration {
		s := mkNew()
		start := time.Now()
		tok := int32(7)
		for i := 0; i < tokPerRound; i++ {
			l, err := st.Forward(tok)
			if err != nil {
				t.Fatal(err)
			}
			tok = s.Sample(l)
			s.Observe(tok)
			sampled++
		}
		return time.Since(start)
	}
	tokOld := func() time.Duration {
		s := mkOld()
		start := time.Now()
		tok := int32(7)
		for i := 0; i < tokPerRound; i++ {
			l, err := st.Forward(tok)
			if err != nil {
				t.Fatal(err)
			}
			tok = s.Sample(l)
			s.Observe(tok)
			sampled++
		}
		return time.Since(start)
	}
	aa, aaIQR = samplerPerfRatios(10, tokNew, tokNew)
	t.Logf("TOK  A/A self-control   %.4f  IQR/median %.3f", aa, aaIQR)
	for pass := 1; pass <= 2; pass++ {
		r, iqr := samplerPerfRatios(10, tokNew, tokOld)
		t.Logf("TOK  pass %d  generated/Go  %.3fx  IQR/median %.3f", pass, r, iqr)
	}
	// And the absolute rates.
	tg := tokNew() / tokPerRound
	to := tokOld() / tokPerRound
	t.Logf("TOK  generated %v/token (%.2f tok/s)   Go %v/token (%.2f tok/s)",
		tg, float64(time.Second)/float64(tg), to, float64(time.Second)/float64(to))
	if sampled == 0 {
		t.Fatal("no token was sampled: this measured nothing")
	}
}
