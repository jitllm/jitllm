package engine

import (
	"sync"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestTwoInstancesKeepTheirOwnOptions checks two instances of one model each
// keep their own options, rather than the second NewInstance reconfiguring
// the first through shared state (a silent failure). The two knobs are read
// back through the instance APIs that consume them, and the arms must
// disagree on every one.
func TestTwoInstancesKeepTheirOwnOptions(t *testing.T) {
	m, err := model.Open(jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf")))
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()

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

	// Make both before observing either: a clobber happens at the second.
	ins := make([]*Instance, len(arms))
	for i, a := range arms {
		ins[i] = testInstance(t, m, 32, a.opts...)
		defer ins[i].Close()
	}
	for i, a := range arms {
		s := ins[i]
		if got := s.AttnPair(); got != a.pair {
			t.Errorf("%s: AttnPair() = %d, want %d -- it read the other instance's option",
				a.name, got, a.pair)
		}
		if got := s.KVIsF16(); got != a.kvF16 {
			t.Errorf("%s: KVIsF16() = %v, want %v -- it read the other instance's option",
				a.name, got, a.kvF16)
		}
	}
}

// TestTwoInstancesKeepTheirOwnUnobservableOptions checks the options whose
// effect needs a device, a tower or a stopwatch to observe, by reading them
// off each instance. That is weaker than running them: it shows the option
// landed on this instance and not on its model or the package.
func TestTwoInstancesKeepTheirOwnUnobservableOptions(t *testing.T) {
	m, err := model.Open(jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf")))
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()
	a := testInstance(t, m, 32, WithHeadPlacement(false), WithTowerTiling(false), WithTowerGPULayers(0),
		WithGrowDebug(true), WithProfile(true), WithSeamTuneSchedule(7, 5, 2, true))
	defer a.Close()
	b := testInstance(t, m, 32, WithHeadPlacement(true), WithTowerTiling(true), WithTowerGPULayers(-1),
		WithGrowDebug(false), WithProfile(false), WithSeamTuneSchedule(32, 24, 3, false))
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
				t.Errorf("%s on %s = %v, want %v -- it read the other instance's option",
					c.name, who, c.got[i], c.want[i])
			}
		}
	}
}

// TestConcurrentInstancesKeepTheirOptions makes instances of one model
// concurrently, as a server does, and checks each keeps its own options. Run
// it under -race; the assertion holds without it too.
func TestConcurrentInstancesKeepTheirOptions(t *testing.T) {
	m, err := model.Open(jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf")))
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()

	const n = 4
	ins := make([]*Instance, n)
	errs := make([]error, n)
	var wg sync.WaitGroup
	for i := range n {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			// Alternating arms, so a clobber lands on a neighbour rather than
			// on an identical value.
			ins[i], errs[i] = NewInstance(m, 32, WithAttnPair(3*(i%2)), WithKVF16(i%2 == 1))
		}(i)
	}
	wg.Wait()
	defer func() {
		for _, s := range ins {
			if s != nil {
				s.Close()
			}
		}
	}()
	for i, err := range errs {
		if err != nil {
			t.Fatalf("instance %d: %v", i, err)
		}
	}
	for i, s := range ins {
		if got, want := s.AttnPair(), 3*(i%2); got != want {
			t.Errorf("instance %d: AttnPair() = %d, want %d", i, got, want)
		}
		if got, want := s.KVIsF16(), i%2 == 1; got != want {
			t.Errorf("instance %d: KVIsF16() = %v, want %v", i, got, want)
		}
	}
}
