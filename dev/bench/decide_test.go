package bench

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
	"github.com/jitllm/jitllm/tok/jinja"
)

// The perf floors of the decision readouts' serving paths, beside the gates
// that hold their answers (model.TestDecisionBatchMatchesAlone) and not
// inside them (AGENTS.md RULE 4). Each alternates its arms round by round in
// one process and holds the median of the per-round ratios (RULE 2).
//
//	JITLLM_DECIDE_BENCH_MODEL   the decision model (default d1/d1-3B-Q8_0.jlm)
//	JITLLM_DECIDE_BENCH_DEVICE  a device spec (metal:0, cuda:0) for the
//	                            device arms; unset, they do not run

const decideRounds, decideSoak = 20, 3

// decideBenchModel opens the decision model the floors run on.
func decideBenchModel(t *testing.T) *model.Model {
	t.Helper()
	path := os.Getenv("JITLLM_DECIDE_BENCH_MODEL")
	if path == "" {
		path = testmodels.Path("d1/d1-3B-Q8_0.jlm")
	}
	path = testmodels.Resolve(path)
	if _, err := os.Stat(path); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS) -- this floor proved nothing", err)
	}
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// decideBenchRequests is the decision gates' two requests.
func decideBenchRequests(t *testing.T) []model.DecisionRequest {
	t.Helper()
	var reqs []model.DecisionRequest
	for _, name := range []string{"support", "ambiguous"} {
		b, err := os.ReadFile(filepath.Join("..", "..", "engine", "model", "testdata", "decision", name+".json"))
		if err != nil {
			t.Fatal(err)
		}
		v, err := jinja.FromJSON(b)
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

// decideBenchDevice is the device the device arms run on, nil when none is
// named.
func decideBenchDevice(t *testing.T) *tier.GPU {
	t.Helper()
	spec := os.Getenv("JITLLM_DECIDE_BENCH_DEVICE")
	if spec == "" {
		return nil
	}
	g, err := tier.OpenWith(tier.WithDevices(spec))
	if err != nil || g == nil {
		t.Fatalf("%s: %v", spec, err)
	}
	return g
}

// newBenchDecider is a Decider placed on g, every block it takes (nil: the
// host).
func newBenchDecider(t *testing.T, m *model.Model, g *tier.GPU) *model.Decider {
	t.Helper()
	d, err := m.NewDecider(4096)
	if err != nil {
		t.Fatal(err)
	}
	if g != nil {
		if err := d.SetDevice(g, -1); err != nil {
			t.Fatal(err)
		}
		if d.DeviceBlocks() == 0 {
			t.Fatalf("no block on the device: %s", g.Err())
		}
	}
	return d
}

// pairedRatio runs a then b, the order alternating by round, and returns the
// median of b's time over a's with its IQR/median.
func pairedRatio(t *testing.T, a, b func()) (med, spread float64) {
	t.Helper()
	var ratios []float64
	for r := range decideSoak + decideRounds {
		first, second := a, b
		if r%2 == 1 {
			first, second = b, a
		}
		t0 := time.Now()
		first()
		t1 := time.Now()
		second()
		t2 := time.Now()
		ta, tb := t1.Sub(t0), t2.Sub(t1)
		if r%2 == 1 {
			ta, tb = tb, ta
		}
		if r >= decideSoak {
			ratios = append(ratios, tb.Seconds()/ta.Seconds())
		}
	}
	med, q1, q3 := ratioMedian(ratios)
	return med, (q3 - q1) / med
}

// TestDecideBatchIsNotSlowerThanAlone: the requests answered as one batch
// (DecideBatch, the server's path when requests queue) take no longer than
// each answered alone, on the host and on the named device. A batch step
// once ran a recurrent block's projections as a matvec a row, and a batch on
// a device was many times slower than its prompts alone.
func TestDecideBatchIsNotSlowerThanAlone(t *testing.T) {
	m := decideBenchModel(t)
	defer m.Close()
	reqs := decideBenchRequests(t)
	g := decideBenchDevice(t)
	if g != nil {
		defer g.Close()
	}
	for _, arm := range []struct {
		name string
		g    *tier.GPU
	}{{"host", nil}, {"device", g}} {
		if arm.name == "device" && g == nil {
			continue
		}
		t.Run(arm.name, func(t *testing.T) {
			alone, batch := newBenchDecider(t, m, arm.g), newBenchDecider(t, m, arm.g)
			defer alone.Close()
			defer batch.Close()
			med, spread := pairedRatio(t, func() {
				for _, r := range reqs {
					if _, err := alone.Decide(r.State, r.Questions, nil); err != nil {
						t.Fatal(err)
					}
				}
			}, func() {
				out, err := batch.DecideBatch(reqs)
				if err != nil {
					t.Fatal(err)
				}
				for _, r := range out {
					if r.Err != nil {
						t.Fatal(r.Err)
					}
				}
			})
			t.Logf("batch against each alone: median %.4fx, IQR/median %.4f, n=%d", med, spread, decideRounds)
			if spread > 0.10 {
				t.Skipf("IQR/median %.3f > 0.10: the box is too noisy to hold the floor (RULE 2)", spread)
			}
			if med > 1.05 {
				t.Fatalf("the batch took %.3fx the requests alone", med)
			}
		})
	}
}

// TestSecondDeciderIsNotSlower: two identical Deciders on one device take
// the same time, whichever came second. The second session on a Metal card
// once paid 73% more for the same request.
func TestSecondDeciderIsNotSlower(t *testing.T) {
	m := decideBenchModel(t)
	defer m.Close()
	reqs := decideBenchRequests(t)
	g := decideBenchDevice(t)
	if g == nil {
		t.Skip("JITLLM_DECIDE_BENCH_DEVICE names no device: this floor proved nothing")
	}
	defer g.Close()
	first, second := newBenchDecider(t, m, g), newBenchDecider(t, m, g)
	defer first.Close()
	defer second.Close()
	run := func(d *model.Decider) func() {
		return func() {
			for _, r := range reqs {
				if _, err := d.Decide(r.State, r.Questions, nil); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
	med, spread := pairedRatio(t, run(first), run(second))
	t.Logf("second Decider against the first: median %.4fx, IQR/median %.4f, n=%d", med, spread, decideRounds)
	if spread > 0.10 {
		t.Skipf("IQR/median %.3f > 0.10: the box is too noisy to hold the floor (RULE 2)", spread)
	}
	if med > 1.05 || med < 1/1.05 {
		t.Fatalf("the second Decider took %.3fx the first", med)
	}
}
