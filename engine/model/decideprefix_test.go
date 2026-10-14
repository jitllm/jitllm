package model

import (
	"math"
	"path/filepath"
	"testing"
	"time"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestDecisionTrunkMatchesWhole holds a request answered from one shared
// trunk -- prefilled once, each question's tail run on from it, the sequence
// rewound and the recurrent summary put back between tails -- to the same
// request with every prompt run whole, on the host and with every block a
// card holds on each device. On the host the two are the same to the bit (a
// row's arithmetic does not depend on the chunk it rides in). On a device the
// tail is a chunk of its own where the whole prompt was one, and the
// attention's splits follow the chunk's shape: lev's nouls on CUDA move by up
// to 1e-2 with the shape alone (the same tail first after the trunk moves as
// much as after a rewind, and a one-token tail moves differently from the
// whole one), the f32 reduction band RULE 11c names. The violation (the
// summary not put back, so each tail runs on from the last one's state) must
// read a decade past the clean arm.
func TestDecisionTrunkMatchesWhole(t *testing.T) {
	for _, c := range decisionCases[:2] { // d1 and lev: the label readouts
		t.Run(c.name, func(t *testing.T) {
			m := openDecision(t, c.gguf, c.kind)
			defer m.Close()
			arms := []*tier.GPU{nil}
			arms = append(arms, decisionDevices(t)...)
			for _, g := range arms {
				name := "host"
				if g != nil {
					name = g.Name()
				}
				// The host's equality and the device's band.
				bound := 1e-9
				if g != nil {
					bound = 2e-2
				}
				t.Run(name, func(t *testing.T) {
					d, err := m.NewDecider(4096)
					if err != nil {
						t.Fatal(err)
					}
					defer d.Close()
					if g != nil {
						if err := d.SetDevice(g, -1); err != nil {
							t.Fatal(err)
						}
						t.Logf("%s holds %d of %d blocks", name, d.DeviceBlocks(), m.Cfg.NLayer)
						if d.DeviceBlocks() == 0 {
							t.Skipf("%s took no block", name)
						}
					}
					for _, req := range decisionRequests {
						state, qs := decisionRequest(t, filepath.Join("testdata", "decision", req+".json"))
						run := func(violation string) ([]DecisionAnswer, time.Duration, int) {
							d.violation = violation
							defer func() { d.violation = "" }()
							before := d.trunks
							t0 := time.Now()
							ans, err := d.Decide(state, qs, nil)
							if err != nil {
								t.Fatal(err)
							}
							return ans, time.Since(t0), d.trunks - before
						}
						run("") // warm
						whole, tw, _ := run("no-trunk")
						shared, ts, n := run("")
						if n != 1 {
							t.Fatalf("%s: %d trunk run(s); the request's prompts did not share their trunk", req, n)
						}
						worst := 0.0
						for i := range qs {
							worst = math.Max(worst, probDiff(shared[i], whole[i]))
						}
						t.Logf("%s: %d questions, trunk %v against whole %v; worst probability difference %.2e",
							req, len(qs), ts.Round(time.Millisecond), tw.Round(time.Millisecond), worst)
						if math.IsNaN(worst) || worst > bound {
							t.Errorf("%s: the trunk's answers leave the whole prompts' by %.3e", req, worst)
						}
						bad, _, _ := run("trunk-no-rec")
						vb := 0.0
						for i := range qs {
							vb = math.Max(vb, probDiff(bad[i], whole[i]))
						}
						t.Logf("%s: violation, the summary not put back: worst %.2e", req, vb)
						if !(vb > 10*max(worst, 1e-6)) {
							t.Errorf("%s: the summary not put back reads %.3e against a clean %.3e: the gate cannot see it",
								req, vb, worst)
						}
					}
				})
			}
		})
	}
}

// probDiff is the largest difference between two answers' probabilities,
// P(true) for a noul.
func probDiff(a, b DecisionAnswer) float64 {
	w := math.Abs(a.Noul - b.Noul)
	for i := range a.Probs {
		if i < len(b.Probs) {
			w = math.Max(w, math.Abs(a.Probs[i]-b.Probs[i]))
		}
	}
	return w
}
