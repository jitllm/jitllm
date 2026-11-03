package server

import (
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// TestStepBudgetFollowsTheMeasurement drives the prompt budget with step
// times it is told, a line of the prompt tokens a step carries: it starts
// small; prompt tokens that cost nothing double it a step up to the ceiling;
// at 0.5 ms a token beside a 10 ms decode step it settles where the tokens
// cost cost-1 decode steps (20 at a cost of 2, 60 at 4); a step the budget
// did not limit does not raise it; slow tokens take it to its floor; and a
// fixed budget ignores the clock. Against a budget that never reads its
// measurements, every case after the first reads budgetStart.
func TestStepBudgetFollowsTheMeasurement(t *testing.T) {
	const ms = time.Millisecond
	line := func(per time.Duration) func(int) time.Duration {
		return func(p int) time.Duration { return 10*ms + time.Duration(p)*per }
	}
	drive := func(b *stepBudget, steps int, cost func(int) time.Duration) []int {
		var seen []int
		for range steps {
			p := b.tokens()
			b.observe(4, p, cost(p))
			seen = append(seen, b.tokens())
		}
		return seen
	}
	b := newStepBudget(0, 512, 2)
	if got := b.tokens(); got != budgetStart {
		t.Fatalf("unmeasured budget %d, want %d", got, budgetStart)
	}
	if seen := drive(b, 6, line(0)); seen[0] != 2*budgetStart || seen[len(seen)-1] != 512 {
		t.Fatalf("free prompt tokens grew the budget %v, want doubling to 512", seen)
	}
	for _, c := range []struct {
		cost float64
		want int
	}{{2, 20}, {4, 60}} {
		f := newStepBudget(0, 512, c.cost)
		seen := drive(f, 40, line(ms/2))
		if got := f.tokens(); got < c.want-1 || got > c.want+1 {
			t.Fatalf("cost %.0f: budget %v, want it at %d (10 ms / 0.5 ms x %.0f)", c.cost, seen, c.want, c.cost-1)
		}
		f.observe(4, 2, line(ms/2)(2)) // under the budget: says nothing about more
		if got := f.tokens(); got < c.want-1 || got > c.want+1 {
			t.Fatalf("cost %.0f: an unlimited step moved the budget to %d", c.cost, got)
		}
	}
	slow := newStepBudget(0, 512, 2)
	drive(slow, 40, line(ms/2))
	if seen := drive(slow, 60, line(100*ms)); seen[len(seen)-1] != budgetFloor {
		t.Fatalf("slow prompt tokens took the budget %v, want the floor %d", seen, budgetFloor)
	}
	// Once the tokens are cheap again the budget leaves its floor: a fit
	// taken across slow steps (a compile, a pool growth) must not hold it
	// there once its steps spend under half their cost, so it comes back to
	// at least half of what the fit gives (20). Against the fit that
	// stood whenever a step was within its cost, it stays at the floor.
	if seen := drive(slow, 60, line(ms/2)); seen[len(seen)-1] < 20/2 {
		t.Fatalf("cheap prompt tokens after slow ones left the budget %v, want it back within half of 20", seen)
	}
	if got := newStepBudget(16, 512, 2).tokens(); got != 16 {
		t.Fatalf("a fixed budget reads %d", got)
	}
}

// TestHostBatchPromptBesideDecodeIsBounded: a long prompt admitted beside two
// decoding rows is fed at most StepPromptTokens tokens a step while they
// decode -- counted, the most prompt tokens any step carried beside a
// decoding row -- it still finishes, and every row comes out as it does
// alone. Then the measured budget (StepPromptTokens 0) on the same requests:
// bounded by the chunk, rows equal to alone, the budget it settled on logged.
// The violation lifts the bound (the budget fixed at the prompt chunk): a
// step then carries more than the bound beside decoding rows.
func TestHostBatchPromptBesideDecodeIsBounded(t *testing.T) {
	const bound = 16
	e, lm, c := hostBatchEngine(t, "Llama-3.2-1B-Instruct-Q4_K_M.jlm",
		Config{JointSteps: JointAlways, PromptChunk: 64, StepPromptTokens: bound, DefaultMaxSeq: 1024,
			// Each run sends the same prompts: the memory cache would restore them.
			NoMemCache: true})
	long := strings.Repeat("The little dog ran to the park and played with a red ball. ", 12)
	reqs := []*v1.GenerateRequest{
		{ModelId: "dev", Prompt: text(hostPrompts()[0]), MaxTokens: 40},
		{ModelId: "dev", Prompt: text(hostPrompts()[1]), MaxTokens: 40},
		{ModelId: "dev", Prompt: text(long), MaxTokens: 8},
	}
	prompts := []string{hostPrompts()[0], hostPrompts()[1], long}
	alone := make([]completion, len(reqs))
	for i, r := range reqs {
		if alone[i] = complete(c, r); alone[i].err != nil {
			t.Fatal(alone[i].err)
		}
	}
	// The two short prompts are admitted first and decode; the long one
	// arrives once they do.
	run := func() ([]completion, int64) {
		lm.loop.stats.maxPromptBeside.Store(0)
		got := make([]completion, len(reqs))
		release := holdGates(t, e, lm)
		var wg sync.WaitGroup
		for i := range 2 {
			wg.Add(1)
			go func() { defer wg.Done(); got[i] = complete(c, reqs[i]) }()
		}
		waitFor(t, "the short requests to wait", func() bool { return waitingFor(lm.loop) == 2 })
		release()
		waitFor(t, "the short rows to decode", func() bool { return lm.loop.stats.steps.Load() > 0 && waitingFor(lm.loop) == 0 })
		got[2] = complete(c, reqs[2])
		wg.Wait()
		for i := range got {
			if got[i].err != nil {
				t.Fatal(got[i].err)
			}
			if j := parted(t, lm, prompts[i], got[i].ids, alone[i].ids); j >= 0 {
				t.Fatalf("row %d parts from alone at token %d\nbatched %v\nalone   %v", i, j, got[i].ids, alone[i].ids)
			}
		}
		return got, lm.loop.stats.maxPromptBeside.Load()
	}
	_, most := run()
	t.Logf("fixed bound %d: at most %d prompt tokens beside decoding rows", bound, most)
	if most == 0 || most > bound {
		t.Fatalf("%d prompt tokens in one step beside decoding rows, want 1..%d", most, bound)
	}

	lm.loop.budget = newStepBudget(0, 64, 0)
	_, most = run()
	t.Logf("measured: at most %d prompt tokens beside decoding rows; budget now %d",
		most, lm.loop.budget.tokens())
	if most == 0 || most > 64 {
		t.Fatalf("measured budget: %d prompt tokens beside decoding rows", most)
	}

	lm.loop.budget = newStepBudget(64, 64, 0)
	_, most = run()
	if most <= bound {
		t.Fatalf("the bound lifted to 64, a step still carried at most %d beside decoding rows: "+
			"this gate cannot see the bound", most)
	}
	t.Logf("violation: the bound lifted, %d prompt tokens beside decoding rows", most)
}

// TestHostBatchBudgetTrajectoryUnderArrivals is not a timing gate: it runs a
// closed loop of sixteen clients on the host, each sending a fresh prompt of
// about 128 tokens for 64 tokens as soon as its last ends, and logs the
// measured budget's trajectory over the steps beside decoding rows (the
// deterministic counters: steps, least, mean, most), so a controller stuck
// at its floor shows. It holds the budget off the floor on average and every
// request to its 64 tokens.
func TestHostBatchBudgetTrajectoryUnderArrivals(t *testing.T) {
	_, lm, c := hostBatchEngine(t, "Llama-3.2-1B-Instruct-Q4_K_M.jlm", Config{DefaultMaxSeq: 512})
	const clients, each = 16, 3
	base := strings.Repeat("The little dog ran to the park and played with a red ball. ", 10)
	var wg sync.WaitGroup
	errs := make(chan error, clients*each)
	for i := range clients {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range each {
				r := complete(c, &v1.GenerateRequest{ModelId: "dev", Prompt: text(fmt.Sprintf("%d.%d %s", i, j, base)),
					MaxTokens: 64, IgnoreEos: true})
				if r.err == nil && len(r.ids) != 64 {
					r.err = fmt.Errorf("client %d request %d: %d tokens", i, j, len(r.ids))
				}
				if r.err != nil {
					errs <- r.err
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	st := &lm.loop.stats
	n := st.budgetSteps.Load()
	if n == 0 {
		t.Fatal("no step ran beside decoding rows: the trajectory measured nothing")
	}
	mean := float64(st.budgetSum.Load()) / float64(n)
	t.Logf("budget over %d steps beside decoding rows: least %d, mean %.1f, most %d; at most %d prompt tokens beside them",
		n, st.budgetMin.Load(), mean, st.budgetMax.Load(), st.maxPromptBeside.Load())
	if mean <= budgetFloor {
		t.Fatalf("the budget sat at its floor (mean %.1f): the controller is stuck low", mean)
	}
}
