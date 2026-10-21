package model

import (
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestDecisionBatchOfOnePromptRunsAlone sends a batch of one request with one
// question -- what a server's lone request is -- and holds it to Decide's
// answer, run on the Decider's own State: the batch session takes no step. A
// lone prompt stepped through the batch session reads every logit of its row
// back where Decide reads the labels' alone, and on the V100 that put lev's
// /v1/systemone p50 at 84 ms against Decide's 42. The violation runs it as a
// row of the batch session, which the step count sees.
func TestDecisionBatchOfOnePromptRunsAlone(t *testing.T) {
	for _, c := range decisionCases {
		t.Run(c.name, func(t *testing.T) {
			m := openDecision(t, c.gguf, c.kind, noTune)
			defer m.Close()
			if m.IsEncoder() || c.kind == jlm.DecisionClef || c.kind == jlm.DecisionKev {
				t.Skipf("%s runs a request one pass of its own, never a batch session", c.name)
			}
			m.SetKVF16(false)
			state, qs := decisionRequest(t, filepath.Join("testdata", "decision", decisionRequests[0]+".json"))
			// A noul: one prompt on every readout (lev asks a choice twice).
			var one []DecisionQuestion
			for i := range qs {
				if qs[i].Type == jlm.QuestionNoul {
					one = qs[i : i+1]
					break
				}
			}
			if one == nil {
				t.Fatalf("%s carries no noul question", decisionRequests[0])
			}
			reqs := []DecisionRequest{{State: state, Questions: one}}
			arms := []*tier.GPU{nil}
			arms = append(arms, decisionDevices(t)...)
			for _, g := range arms {
				name := "host"
				if g != nil {
					name = g.Name()
				}
				t.Run(name, func(t *testing.T) {
					d, err := m.NewDecider(4096)
					if err != nil {
						t.Fatal(err)
					}
					defer d.Close()
					if g != nil {
						if err := d.SetDevice(g, decisionBlocks(m)/2); err != nil {
							t.Fatal(err)
						}
						if d.DeviceBlocks() == 0 {
							t.Fatalf("%s took no block: %s", g.Name(), g.Err())
						}
					}
					if n := d.variants(&one[0]); n != 1 {
						t.Fatalf("question %s asks %d variants: not one prompt", one[0].ID, n)
					}
					alone, err := d.Decide(state, one, nil)
					if err != nil {
						t.Fatal(err)
					}
					tokens := d.InputTokens()
					s0 := d.BatchStats().Steps
					got, err := d.DecideBatch(reqs)
					if err != nil {
						t.Fatal(err)
					}
					if got[0].Err != nil {
						t.Fatal(got[0].Err)
					}
					if n := d.BatchStats().Steps - s0; n != 0 {
						t.Fatalf("one prompt took %d batch-session step(s)", n)
					}
					if got[0].InputTokens != tokens || d.InputTokens() != tokens {
						t.Fatalf("the batch ran %d tokens (%d counted), alone %d", got[0].InputTokens, d.InputTokens(), tokens)
					}
					if diff := decisionDiffRow(t, [][]DecisionAnswer{alone}, [][]DecisionAnswer{got[0].Answers}); diff != 0 {
						t.Fatalf("a batch of one moved the answer %.6f from Decide's", diff)
					}
					d.violation = "batch-alone"
					s0 = d.BatchStats().Steps
					_, err = d.DecideBatch(reqs)
					d.violation = ""
					if err != nil {
						t.Fatal(err)
					}
					if d.BatchStats().Steps == s0 {
						t.Errorf("the violation took no batch step either: the gate does not see it")
					}
				})
			}
		})
	}
}
