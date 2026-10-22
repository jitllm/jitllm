package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// d1OmniGold is scripts/d1omnigold.py's output: per question the reference's
// ids and markers (prompt.encode), the trunk's first 32 values at each
// marker, the head's logits and system_one's probabilities.
type d1OmniGold struct {
	Usage struct {
		InputTokens int `json:"input_tokens"`
	} `json:"usage"`
	Questions []struct {
		ID      string      `json:"id"`
		IDs     []int32     `json:"ids"`
		Markers []int       `json:"markers"`
		Prefix  int         `json:"prefix"`
		Trunk   [][]float64 `json:"trunk_at_markers"`
		Logits  []float64   `json:"logits"`
		Probs   []float64   `json:"probabilities"`
	} `json:"questions"`
}

const d1OmniModel = "d1-omni/d1-omni-600M-F16.gguf"

func readD1OmniGold(t *testing.T, req string) d1OmniGold {
	t.Helper()
	var g d1OmniGold
	b, err := os.ReadFile(filepath.Join("testdata", "decision", "d1-omni."+req+".reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

// TestD1OmniMatchesReference holds d1-omni's text readout to its own code
// (modeling_d1.py, on the same F16 weights in f32): the prompt token for
// token, the trunk's rows at the markers, the head's logits and the answer's
// probabilities.
func TestD1OmniMatchesReference(t *testing.T) {
	m := openDecision(t, d1OmniModel, jlm.DecisionD1Omni)
	defer m.Close()
	d1OmniMatchesReference(t, m)
}

// d1OmniMatchesReference is TestD1OmniMatchesReference on model m: the
// trunk's rows are the host's, the logits and probabilities those m's
// Decider gives, on the composer's plan where m was opened on one.
func d1OmniMatchesReference(t *testing.T, m *Model) {
	d, err := m.NewDecider(0)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	for _, req := range decisionRequests {
		t.Run(req, func(t *testing.T) {
			g := readD1OmniGold(t, req)
			state, qs := decisionRequest(t, filepath.Join("testdata", "decision", req+".json"))
			byID := map[string]*DecisionQuestion{}
			for i := range qs {
				byID[qs[i].ID] = &qs[i]
			}
			worstT, worstL, worstP := 0.0, 0.0, 0.0
			for _, gq := range g.Questions {
				q := byID[gq.ID]
				if q == nil {
					t.Fatalf("the reference answers %q, the request has no such question", gq.ID)
				}
				ids, markers, err := d.d1OmniSequence(state, q, d.d1OmniMaxLen())
				if err != nil {
					t.Fatal(err)
				}
				if !equalIDs(ids, gq.IDs) || !equalInts(markers, gq.Markers) {
					t.Fatalf("%s: the sequence differs from the reference's\n got %v %v\nwant %v %v",
						q.ID, ids, markers, gq.IDs, gq.Markers)
				}
				d.emb.afterBlock = nil
				if err := d.emb.lfm2Encode(ids, nil); err != nil {
					t.Fatal(err)
				}
				dm := m.Cfg.NEmbd
				for j, r := range markers {
					for k, want := range gq.Trunk[j] {
						worstT = max(worstT, math.Abs(float64(d.emb.x[r*dm+k])-want))
					}
				}
				s, err := d.d1OmniForward(ids, q.Type, markers, 0)
				if err != nil {
					t.Fatal(err)
				}
				for j := range s {
					if math.IsNaN(float64(s[j])) || math.IsInf(float64(s[j]), 0) {
						t.Fatalf("%s: score %d is %v", q.ID, j, s[j])
					}
					worstL = max(worstL, math.Abs(float64(s[j])-gq.Logits[j]))
				}
				a, err := d.answer(q, [][]float32{s})
				if err != nil {
					t.Fatal(err)
				}
				probs := a.Probs
				if q.Type == jlm.QuestionNoul {
					probs = []float64{a.Noul, 1 - a.Noul}
				}
				for j := range probs {
					worstP = max(worstP, math.Abs(probs[j]-gq.Probs[j]))
				}
			}
			t.Logf("worst difference from the reference: trunk %.2e, logit %.2e, probability %.2e", worstT, worstL, worstP)
			if m.opt.planDev != nil && len(d.plans) == 0 {
				t.Errorf("the questions never ran on the plan")
			}
			// F16 weights rounded alike on both sides (D1OMNI_F16): what is
			// left is f32 order (worst 4.6e-5 on the trunk, 2.8e-5 on a logit).
			if worstT > 5e-4 || worstL > 3e-4 || worstP > 1e-4 {
				t.Errorf("past the bound (trunk 5e-4, logit 3e-4, probability 1e-4)")
			}
		})
	}
}

// TestD1OmniFeaturesAreLoadBearing removes each feature of d1-omni's graph
// that LFM2's decoder or Laya's head does not share and requires the logits
// to move past ten times the bound the whole graph meets: the convolution's
// right tap, attention to later rows, and the question type's row.
func TestD1OmniFeaturesAreLoadBearing(t *testing.T) {
	m := openDecision(t, d1OmniModel, jlm.DecisionD1Omni)
	defer m.Close()
	d, err := m.NewDecider(0)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	g := readD1OmniGold(t, "support")
	state, qs := decisionRequest(t, filepath.Join("testdata", "decision", "support.json"))
	for _, v := range []string{"causal-conv", "causal-attn", "type-row-0"} {
		t.Run(v, func(t *testing.T) {
			d.emb.violation = v
			defer func() { d.emb.violation = "" }()
			worst := 0.0
			for _, gq := range g.Questions {
				for i := range qs {
					if qs[i].ID != gq.ID {
						continue
					}
					ids, markers, err := d.d1OmniSequence(state, &qs[i], d.d1OmniMaxLen())
					if err != nil {
						t.Fatal(err)
					}
					s, err := d.d1OmniForward(ids, qs[i].Type, markers, 0)
					if err != nil {
						t.Fatal(err)
					}
					for j := range s {
						worst = max(worst, math.Abs(float64(s[j])-gq.Logits[j]))
					}
				}
			}
			t.Logf("worst logit difference without it: %.3g", worst)
			if !(worst > 3e-3) {
				t.Errorf("removing %s moved no logit past 3e-3 (worst %.3g): the gate cannot see it", v, worst)
			}
		})
	}
}

// TestD1OmniKeepsThePrinciples holds d1-omni's readout to two of the five
// principles on the host: a warm pass over every prompt makes no engine
// allocation, and a page budget of half its blocks evicts, re-reads and
// gives the same scores to the bit; on each device, a seam and every block
// placed make no allocation either (relocation: TestDecisionRelocatesBetweenDevices).
func TestD1OmniKeepsThePrinciples(t *testing.T) {
	t.Run("allocation", func(t *testing.T) {
		m := openDecision(t, d1OmniModel, jlm.DecisionD1Omni, noTune)
		defer m.Close()
		t.Run("host", func(t *testing.T) { decisionAllocs(t, m, nil, 0) })
		for _, arm := range layaArms(t) {
			if arm.gpu == nil {
				continue
			}
			t.Run(arm.name, func(t *testing.T) {
				t.Run("seam", func(t *testing.T) { decisionAllocs(t, m, arm.gpu, decisionBlocks(m)/2) })
				t.Run("every-block", func(t *testing.T) { decisionAllocs(t, m, arm.gpu, -1) })
			})
		}
	})
	t.Run("paging", func(t *testing.T) {
		state, qs := decisionRequest(t, filepath.Join("testdata", "decision", "support.json"))
		run := func(paged bool) ([]float32, int64, int) {
			m := openDecision(t, d1OmniModel, jlm.DecisionD1Omni)
			defer m.Close()
			if paged {
				m.SetPageBudget(uint64(m.container.NPages()/2) * m.PageSize())
			}
			d, err := m.NewDecider(0)
			if err != nil {
				t.Fatal(err)
			}
			defer d.Close()
			var s []float32
			for range 2 {
				if s, err = d.run(state, &qs[0], 0); err != nil {
					t.Fatal(err)
				}
			}
			return append([]float32(nil), s...), m.PagerReads(), m.container.NPages()
		}
		whole, _, _ := run(false)
		paged, reads, pages := run(true)
		if reads <= int64(pages) {
			t.Fatalf("a budget of half the %d block pages read %d: nothing was evicted", pages, reads)
		}
		for i := range whole {
			if whole[i] != paged[i] {
				t.Errorf("option %d: %v resident, %v paged", i, whole[i], paged[i])
			}
		}
	})
}
