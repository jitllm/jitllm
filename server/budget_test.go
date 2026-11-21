package server

import (
	"testing"

	"github.com/jitllm/jitllm/sched"
)

// Whatever the policy, the shares must never exceed the machine's budget.
//
// That is why the function exists: sched.MemBudget() is stateless, so without
// a division two models would each claim the whole machine.
func TestSharesNeverExceedTheBudget(t *testing.T) {
	total := sched.MemBudget()
	if total == 0 {
		t.Skip("no host budget on this platform")
	}
	for _, c := range []struct {
		name     string
		paths    []string
		active   string
		priority bool
		pins     map[string]uint64
	}{
		{"one model", []string{"a"}, "a", false, nil},
		{"two, first come", []string{"a", "b"}, "a", false, nil},
		{"two, priority", []string{"a", "b"}, "a", true, nil},
		{"four, priority", []string{"a", "b", "c", "d"}, "c", true, nil},
		{"a pin", []string{"a", "b"}, "a", true, map[string]uint64{"b": total / 2}},
		{"a pin bigger than the machine", []string{"a", "b"}, "a", true,
			map[string]uint64{"b": total * 4}},
		{"every model pinned", []string{"a", "b"}, "a", true,
			map[string]uint64{"a": total / 4, "b": total / 4}},
		{"the ACTIVE model is pinned", []string{"a", "b", "c"}, "a", true,
			map[string]uint64{"a": total / 4}},
	} {
		t.Run(c.name, func(t *testing.T) {
			got := shares(total, c.paths, c.active, c.priority, c.pins)
			var sum uint64
			for _, v := range got {
				sum += v
			}
			if sum > total {
				t.Errorf("granted %d of a %d budget: the models have each been "+
					"handed memory the machine does not have", sum, total)
			}
			if len(got) != len(c.paths) {
				t.Errorf("granted %d share(s) for %d model(s)", len(got), len(c.paths))
			}
		})
	}
}

// Priority must actually favour the active model, and first-come must not.
func TestPriorityFavoursTheActiveModel(t *testing.T) {
	total := sched.MemBudget()
	if total == 0 {
		t.Skip("no host budget on this platform")
	}
	paths := []string{"a", "b", "c"}

	fair := shares(total, paths, "a", false, nil)
	if fair["a"] != fair["b"] || fair["b"] != fair["c"] {
		t.Errorf("first-come gave unequal shares: %v", fair)
	}

	pri := shares(total, paths, "a", true, nil)
	if pri["a"] <= pri["b"] {
		t.Errorf("with priority the active model got %d against a background %d",
			pri["a"], pri["b"])
	}
	if pri["b"] != pri["c"] {
		t.Errorf("the two backgrounded models got %d and %d", pri["b"], pri["c"])
	}
	if pri["b"] == 0 {
		t.Error("a backgrounded model got nothing: it is still open, its pager " +
			"is still alive, and a budget below one page cannot be recovered from")
	}
}

// A pin is honoured exactly and comes off the top.
func TestAPinIsHonouredExactly(t *testing.T) {
	total := sched.MemBudget()
	if total == 0 {
		t.Skip("no host budget on this platform")
	}
	const want = 1 << 30
	got := shares(total, []string{"a", "b"}, "a", true, map[string]uint64{"b": want})
	if got["b"] != want {
		t.Errorf("a pinned model got %d, want exactly %d", got["b"], want)
	}
	if got["a"]+got["b"] > total {
		t.Errorf("the pin was not taken off the top: %d + %d > %d", got["a"], got["b"], total)
	}
}
