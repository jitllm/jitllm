package tier

import "testing"

// mvPays is called with g.mu held; these construct a GPU nothing else touches,
// so the lock is not taken.
//
// TestMVPaysIsPerModel: the decision is all-or-nothing per model (a
// per-tensor threshold measured wrong; see mvPays). The shapes are two real
// models that bracket the threshold.
func TestMVPaysIsPerModel(t *testing.T) {
	// tinyllama q3_K_M and gemma-2b, one layer's matvec weights in KiB, read
	// off JITLLM_GPU_VERBOSE on the real files: q, k, v, o, gate, up, down.
	kib := func(v ...int) []int {
		for i := range v {
			v[i] <<= 10
		}
		return v
	}
	tiny := kib(1761, 215, 287, 2304, 4844, 4844, 6339)
	gem := kib(2304, 287, 287, 2304, 18432, 18432, 18432)

	for _, c := range []struct {
		name  string
		layer []int
		want  bool
	}{
		{"tinyllama", tiny, false}, // 2.87 MB per crossing
		{"gemma-2b", gem, true},    // 8.44 MB per crossing
	} {
		g := newDevice(&devShared{Config: &Config{mvMin: 3300 << 10}})
		var served, declined int
		// Two full tokens' worth of offers, which is well past mvSample.
		for i := 0; i < 4*mvSample; i++ {
			if g.mvPays(0, c.layer[i%len(c.layer)]) {
				served++
			} else {
				declined++
			}
		}
		// The sample is taken by serving, so the first mvSample offers are
		// served whatever the verdict; the steady state is what is asserted.
		got := declined == 0
		if got != c.want {
			t.Errorf("%s: served %d declined %d, want serve=%v", c.name, served, declined, c.want)
		}
		if !c.want && served != mvSample-1 {
			t.Errorf("%s: sampled %d offers, want %d", c.name, served, mvSample-1)
		}
	}
}

// TestMVPaysDoesNotDrift: a verdict, once reached, must not drift within a
// session (a tied lm_head at the end of every token would flip it and make the
// seam non-deterministic). The sample is per session (mvFor(sid)), so every
// session reaches the same answer.
func TestMVPaysDoesNotDrift(t *testing.T) {
	g := newDevice(&devShared{Config: &Config{mvMin: 3300 << 10}})
	for i := 0; i < mvSample; i++ {
		g.mvPays(0, 1<<10)
	}
	if v := g.mvFor(0).verdict; v >= 0 {
		t.Fatalf("verdict %d, want negative after a sample of 1 KiB matvecs", v)
	}
	for i := 0; i < 100; i++ {
		if g.mvPays(0, 512<<20) {
			t.Fatalf("a 512 MiB matvec reopened a settled verdict at offer %d", i)
		}
	}
}

// TestMVPaysZeroServesEverything: JITLLM_MV_MIN_KB=0, the unpriced behaviour
// every A/B in mvPays' comment ran against, must stay reachable.
func TestMVPaysZeroServesEverything(t *testing.T) {
	g := newDevice(&devShared{Config: &Config{mvMin: 0}})
	for i := 0; i < 4*mvSample; i++ {
		if !g.mvPays(0, 1) {
			t.Fatalf("declined a 1-byte matvec at mvMin 0, offer %d", i)
		}
	}
}
