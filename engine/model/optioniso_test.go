package model

import (
	"sync"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestTwoModelsKeepTheirOwnLoadOptions checks two models in one process each
// keep their own load options, rather than the second Open reconfiguring the
// first through package state (a silent failure). The three knobs are read back
// through the instance APIs that consume them, and the arms must disagree on
// every one.
func TestTwoModelsKeepTheirOwnLoadOptions(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))

	type arm struct {
		name  string
		opts  []Option
		pair  int
		kvF16 bool
	}
	arms := []arm{
		{"A", []Option{WithAttnPair(0), WithKVF16(false)}, 0, false},
		{"B", []Option{WithAttnPair(3), WithKVF16(true)}, 3, true},
	}
	// The arms must differ in every axis, or an axis proves nothing.
	if arms[0].pair == arms[1].pair || arms[0].kvF16 == arms[1].kvF16 {
		t.Fatal("the two arms must disagree on every knob; this gate would prove nothing")
	}

	// Open both before observing either: a clobber happens at the second Open.
	ms := make([]*Model, len(arms))
	for i, a := range arms {
		m, err := Open(path, a.opts...)
		if err != nil {
			for _, p := range ms[:i] {
				p.Close()
			}
			t.Skip(err)
		}
		ms[i] = m
	}
	defer func() {
		for _, m := range ms {
			m.Close()
		}
	}()

	for i, a := range arms {
		s := ms[i].NewInstance(32)
		if got := s.AttnPair(); got != a.pair {
			t.Errorf("%s: AttnPair() = %d, want %d -- it read the other model's option",
				a.name, got, a.pair)
		}
		if got := s.KVIsF16(); got != a.kvF16 {
			t.Errorf("%s: KVIsF16() = %v, want %v -- it read the other model's option",
				a.name, got, a.kvF16)
		}
		s.Close()
	}
}

// TestTwoModelsKeepTheirOwnUnobservableOptions checks the options whose effect
// needs a device, a tower or a stopwatch to observe, by reading them off each
// Model. That is weaker than running them: it shows the option landed on this
// model and not on the package.
func TestTwoModelsKeepTheirOwnUnobservableOptions(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))

	a, err := Open(path,
		WithHeadPlacement(false), WithTowerTiling(false), WithTowerGPULayers(0),
		WithGrowDebug(true), WithProfile(true), WithSeamTuneSchedule(7, 5, 2, true))
	if err != nil {
		t.Skip(err)
	}
	defer a.Close()
	b, err := Open(path,
		WithHeadPlacement(true), WithTowerTiling(true), WithTowerGPULayers(-1),
		WithGrowDebug(false), WithProfile(false), WithSeamTuneSchedule(32, 24, 3, false))
	if err != nil {
		t.Skip(err)
	}
	defer b.Close()

	for _, c := range []struct {
		name string
		got  [2]any
		want [2]any
	}{
		{"headPlace", [2]any{a.opt.headPlace, b.opt.headPlace}, [2]any{false, true}},
		{"elemPool", [2]any{a.opt.elemPool, b.opt.elemPool}, [2]any{false, true}},
		{"towerGPULayers", [2]any{a.opt.towerGPULayers, b.opt.towerGPULayers}, [2]any{0, -1}},
		{"growDebug", [2]any{a.opt.growDebug, b.opt.growDebug}, [2]any{true, false}},
		{"profile", [2]any{a.opt.profile, b.opt.profile}, [2]any{true, false}},
		{"seamWarmup", [2]any{a.opt.seamWarmup, b.opt.seamWarmup}, [2]any{7, 32}},
		{"seamRun", [2]any{a.opt.seamRun, b.opt.seamRun}, [2]any{5, 24}},
		{"seamRounds", [2]any{a.opt.seamRounds, b.opt.seamRounds}, [2]any{2, 3}},
		{"seamVerbose", [2]any{a.opt.seamVerbose, b.opt.seamVerbose}, [2]any{true, false}},
	} {
		if c.want[0] == c.want[1] {
			t.Errorf("%s: the two arms agree; this axis proves nothing", c.name)
		}
		for i, who := range []string{"A", "B"} {
			if c.got[i] != c.want[i] {
				t.Errorf("%s on %s = %v, want %v -- it read the other model's option",
					c.name, who, c.got[i], c.want[i])
			}
		}
	}
}

// TestConcurrentOpenKeepsEachModelsOptions opens models concurrently, as a
// server does, and checks each keeps its own options. Run it under -race; the
// assertion holds without it too.
func TestConcurrentOpenKeepsEachModelsOptions(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))

	const n = 4
	ms := make([]*Model, n)
	errs := make([]error, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			// Alternating arms, so a clobber lands on a neighbour rather than
			// on an identical value.
			ms[i], errs[i] = Open(path, WithAttnPair(3*(i%2)), WithKVF16(i%2 == 1))
		}(i)
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			for _, m := range ms {
				if m != nil {
					m.Close()
				}
			}
			t.Skipf("open %d: %v", i, err)
		}
	}
	defer func() {
		for _, m := range ms {
			m.Close()
		}
	}()
	for i, m := range ms {
		s := m.NewInstance(32)
		if got, want := s.AttnPair(), 3*(i%2); got != want {
			t.Errorf("model %d: AttnPair() = %d, want %d", i, got, want)
		}
		if got, want := s.KVIsF16(), i%2 == 1; got != want {
			t.Errorf("model %d: KVIsF16() = %v, want %v", i, got, want)
		}
		s.Close()
	}
}
