package bench

import (
	"bufio"
	"os"
	"testing"
	"time"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/tok/jinja"
)

// TestDecidePrefixIsFasterThanCold: a Decider sharing its pages through the
// prefix cache (Decider.SetPromptCache) answers a game's run of requests --
// boards that differ in a few tokens, as examples/jivman asks them -- in less
// time than one that prefills every prompt whole, on the host or the named
// device. The arms share the model and alternate request by request in one
// process; each round is the next requests of the run, so the warm arm never
// sees a request twice. An A/A of two cold Deciders runs first.
//
//	JITLLM_DECIDE_PREFIX_REQS  a run of request bodies, one JSON a line
//	                           (examples/jivman writes them); unset, skipped
func TestDecidePrefixIsFasterThanCold(t *testing.T) {
	path := os.Getenv("JITLLM_DECIDE_PREFIX_REQS")
	if path == "" {
		t.Skip("JITLLM_DECIDE_PREFIX_REQS names no request run: this floor proved nothing")
	}
	reqs := prefixBenchRequests(t, path)
	m := decideBenchModel(t)
	defer m.Close()
	g := decideBenchDevice(t)
	if g != nil {
		defer g.Close()
	}
	per := max(1, len(reqs)/(decideSoak+decideRounds))
	run := func(d *model.Decider, at *int) func() {
		return func() {
			for range per {
				r := reqs[*at%len(reqs)]
				*at++
				if _, err := d.Decide(r.State, r.Questions, nil); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
	cold1, cold2, warm := newBenchDecider(t, m, g), newBenchDecider(t, m, g), newBenchDecider(t, m, g)
	defer cold1.Close()
	defer cold2.Close()
	defer warm.Close()
	if on, err := warm.SetPromptCache(model.NewSharedStore(), "bench-prefix"); err != nil || !on {
		t.Fatalf("the warm Decider did not take the store: %v", err)
	}
	var a1, a2 int
	med, spread := pairedRatio(t, run(cold1, &a1), run(cold2, &a2))
	t.Logf("A/A, cold against cold: median %.4fx, IQR/median %.4f, n=%d", med, spread, decideRounds)
	var c, w int
	t0 := time.Now()
	med, spread = pairedRatio(t, run(cold1, &c), run(warm, &w))
	t.Logf("warm against cold: median %.4fx, IQR/median %.4f, n=%d, %d request(s) a round, %d position(s) restored, %v",
		med, spread, decideRounds, per, warm.PromptRestored(), time.Since(t0).Round(time.Millisecond))
	if med >= 1 {
		t.Fatalf("the prefix cache took %.3fx a cold prompt's time", med)
	}
}

// prefixBenchRequests reads a run of request bodies, one a line.
func prefixBenchRequests(t *testing.T, path string) []model.DecisionRequest {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var reqs []model.DecisionRequest
	sc := bufio.NewScanner(f)
	sc.Buffer(nil, 1<<22)
	for sc.Scan() {
		v, err := jinja.FromJSON(sc.Bytes())
		if err != nil {
			t.Fatal(err)
		}
		state, _ := v.AsDict().Get("state")
		qv, _ := v.AsDict().Get("questions")
		qs, err := model.ParseDecisionQuestions(qv)
		if err != nil {
			t.Fatal(err)
		}
		reqs = append(reqs, model.DecisionRequest{State: state, Questions: qs})
	}
	return reqs
}
