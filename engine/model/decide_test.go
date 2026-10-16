package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/tok/jinja"
)

// The decision models' label readouts against llama.cpp's /v1/systemone on
// the same GGUF (b11514; scripts/decisiongold.sh writes the goldens). RULE 7m:
// llama.cpp is the oracle for the readout here because it reads the GGUF's own
// template and metadata, which is what jitllm's container carries.

type decisionGold struct {
	Answers map[string]struct {
		Type          string             `json:"type"`
		Noul          float64            `json:"noul"`
		Choice        string             `json:"choice"`
		Score         float64            `json:"score"`
		Confidence    float64            `json:"confidence"`
		Probabilities map[string]float64 `json:"probabilities"`
	} `json:"answers"`
	Usage struct {
		InputTokens int `json:"input_tokens"`
	} `json:"usage"`
}

// decisionRequest reads a /v1/systemone body into the Decider's inputs.
func decisionRequest(t testing.TB, path string) (jinja.Value, []DecisionQuestion) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	v, err := jinja.FromJSON(b)
	if err != nil {
		t.Fatal(err)
	}
	state, _ := v.AsDict().Get("state")
	qv, _ := v.AsDict().Get("questions")
	qs, err := ParseDecisionQuestions(qv)
	if err != nil {
		t.Fatal(err)
	}
	return state, qs
}

// decisionCases are the models and the llama.cpp answers they are held to.
var decisionCases = []struct {
	name, gguf, gold string // gold is the golden's prefix: <gold>.<request>.llamacpp.json
	kind             jlm.DecisionKind
	// ownConfidence: the model's reference computes its confidence its own
	// way (Laya: one minus normalised entropy) and jitllm follows it, where
	// llama.cpp gives TypeSafe's formula (RULE 7m); the field is not compared.
	ownConfidence bool
	// tol is how far a probability may sit from llama.cpp's on the same
	// bytes, on either kv cache and either host architecture; the note
	// before decisionSeen has the measurements.
	tol float64
}{
	{"d1-3B", "d1/d1-3B-Q8_0.gguf", "d1-3B-Q8_0", jlm.DecisionLFM2D1, false, 0.03},
	{"lev", "lev/lev-Q8_0.gguf", "lev-Q8_0", jlm.DecisionLev, false, 0.10},
	{"laya", "laya/gguf/Laya-BF16.gguf", "Laya-BF16", jlm.DecisionLaya, true, 0.01},
	{"clef", "clef/Clef-Flash-Q8_0.gguf", "Clef-Flash-Q8_0", jlm.DecisionClef, true, 0.03},
	// clefgold.py's random fixture: f32, so llama.cpp, Clef's own code and
	// jitllm agree to the reference's rounding of its answers (6.5e-5); it
	// is the Clef every gate can run on a card the release does not fit.
	{"clef-fixture", "clef/fixture/clef-fixture.gguf", "clef-fixture", jlm.DecisionClef, true, 0.001},
	// d1-omni's F16 GGUF: jitllm sits 5.5e-6 from the model's own code on
	// these weights (TestD1OmniMatchesReference) and llama.cpp 0.0023 from
	// both (its F16 products). Its confidence is its own code's (the chosen
	// option's probability).
	{"d1-omni", d1OmniModel, "d1-omni-600M-F16", jlm.DecisionD1Omni, true, 0.005},
	// llama.cpp's own openjev test model (ggml-org/tinyopenjev-for-testing-gguf):
	// a random qwen35 whose answers sit far from saturation.
	{"openjev", "openjev/tinyopenjev-for-testing-Q8_0.gguf", "tinyopenjev-for-testing-Q8_0", jlm.DecisionOpenJev, false, 0.01},
	// vllm-sr's d3-edge (ggml-org's GGUF): lev's codes over a qwen35 whose
	// attention blocks see the whole prompt; its confidence is its own
	// reference's (one minus the normalised entropy), which llama.cpp's
	// server also gives for it. The tolerance is lev's kind: the ambiguous
	// request's issue question is a near-tie (late 0.50 against damaged 0.50
	// here, 0.42 against 0.57 at llama.cpp's f16 cache), where the f32 arm
	// reads 0.078 off and the f16 arm 0.015.
	{"vllm-d3", "vllm-d3/vllm-d3-edge-Q8_0.gguf", "vllm-d3-edge-Q8_0", jlm.DecisionVllmD3, false, 0.08},
	// jaredpalmer's Kev 4B (ggml-org's BF16): the pointer head over the
	// normed hidden state, the question's last row against each option's
	// <|box_end|> row. BF16 (worst 0.0026 on either cache): at Q4_K_M both
	// engines' projections sit 5.6% from the BF16 ones and 2% from each
	// other, a band as wide as some of the readout's own violations.
	{"kev", "kev/Kev-4B-BF16.gguf", "Kev-4B-BF16", jlm.DecisionKev, false, 0.01},
	// perplexity-ai's pplx-decider v1.1 27B through llama.cpp's own converter
	// (Q8_0; no published GGUF states its readout): lev's codes, its output
	// projection the readout.
	{"pplx-decider", "pplx-decider/pplx-decider-v1.1-27b-Q8_0.gguf", "pplx-decider-v1.1-27b-Q8_0", jlm.DecisionPplxDecider, false, 0.05},
}

// decisionRequests are the request bodies in testdata/decision: an object
// state with every question type, and a string state whose answers are not
// saturated, where a feature's removal shows.
var decisionRequests = []string{"support", "ambiguous"}

// Why each decision case's tolerance is what it is. llama-server's kv
// cache is f16 (its answers did not move with -ctk f32). Measured worst, on
// the Linux laptop (amd64) and the M4 (arm64), f32 and f16 kv:
//
//	d1    0.021 (f16, the billing/returns near-tie) on amd64, 0.022 on arm64
//	lev   0.090 (f32, amd64) and 0.073 (f16, arm64), both on the ambiguous
//	      request's tone, neutral 0.42 against annoyed 0.54 at llama.cpp
//	laya  0.0039 on both
//
// lev's spread is Q8_0 rounding meeting a near-tie, not a kv path's fault:
// the raw logits of the arms sit 0.2-0.4 apart on a scale of 20, and the
// BF16 adapter-merged model in transformers (f32, the reference graph on the
// same prompt ids) is no nearer either -- neutral leads annoyed by 1.57 there,
// 2.00 at f32 kv and 1.44 at f16 kv on amd64.
// decisionSeen is how far a removed feature must move an answer to count as
// seen: twice the worst f32 difference on the questions the violations ask
// (0.0044 on d1, 0.0145 on lev's object state).
const decisionSeen = 0.03

// openDecision opens a case's model, or skips it as missing.
func openDecision(t *testing.T, gguf string, kind jlm.DecisionKind, opts ...Option) *Model {
	t.Helper()
	src := testmodels.Path(gguf)
	if _, err := os.Stat(src); err != nil {
		testmodels.Missing(t, "%s (scripts/decisiongold.sh names its source)", src)
	}
	m, err := Open(jlmOf(t, src), opts...)
	if err != nil {
		t.Fatal(err)
	}
	if got := m.Decision(); got != kind {
		m.Close()
		t.Fatalf("the container says readout %v, want %v", got, kind)
	}
	return m
}

func readDecisionGold(t *testing.T, name string) decisionGold {
	t.Helper()
	var gold decisionGold
	b, err := os.ReadFile(filepath.Join("testdata", "decision", name))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &gold); err != nil {
		t.Fatal(err)
	}
	return gold
}

type decisionArm struct {
	name string
	f16  bool
	tol  float64
}

func TestDecisionMatchesLlamaCpp(t *testing.T) {
	for _, c := range decisionCases {
		t.Run(c.name, func(t *testing.T) {
			m := openDecision(t, c.gguf, c.kind)
			defer m.Close()
			// A decoder runs both kv caches; an encoder keeps none.
			arms := []decisionArm{{"kv-f32", false, c.tol}, {"kv-f16", true, c.tol}}
			if m.IsEncoder() {
				arms = []decisionArm{{"encoder", false, c.tol}}
			}
			for _, req := range decisionRequests {
				gold := readDecisionGold(t, c.gold+"."+req+".llamacpp.json")
				for _, arm := range arms {
					t.Run(req+"/"+arm.name, func(t *testing.T) {
						if !m.IsEncoder() {
							m.SetKVF16(arm.f16)
						}
						state, qs := decisionRequest(t, filepath.Join("testdata", "decision", req+".json"))
						d, err := m.NewDecider(4096)
						if err != nil {
							t.Fatal(err)
						}
						defer d.Close()
						if d.st != nil && d.st.KVIsF16() != arm.f16 {
							t.Fatalf("the kv cache is f16=%v, the arm asked for %v", d.st.KVIsF16(), arm.f16)
						}
						ans, err := d.Decide(state, qs, nil)
						if err != nil {
							t.Fatal(err)
						}
						if d.InputTokens() != gold.Usage.InputTokens {
							t.Errorf("ran %d prompt tokens, llama.cpp %d: the prompts differ", d.InputTokens(), gold.Usage.InputTokens)
						}
						checkDecisions(t, qs, ans, gold, arm.tol, !c.ownConfidence)
					})
				}
			}
		})
	}
}

// TestDecisionReadoutIsLoadBearing removes each feature of a readout and
// requires the answers to leave llama.cpp's by more than the gate's tolerance:
// a feature whose removal the gate cannot see is not gated (RULE 10). d1's
// GGUF states no temperature, so calibration is a feature of lev's alone; and
// d1's second code form (" A" beside "A") moved no answer past 0.004 on
// either request, so it is the reference's behaviour kept without a gate.
func TestDecisionReadoutIsLoadBearing(t *testing.T) {
	for _, c := range []struct {
		name, gguf, gold, req string
		kind                  jlm.DecisionKind
		violation             string
		ask                   []string // the questions the feature touches
	}{
		{"d1-false-first", "d1/d1-3B-Q8_0.gguf", "d1-3B-Q8_0", "ambiguous", jlm.DecisionLFM2D1, "false-first", []string{"compensate", "churn", "damage"}},
		{"lev-one-order", "lev/lev-Q8_0.gguf", "lev-Q8_0", "support", jlm.DecisionLev, "one-order", []string{"intent", "team"}},
		{"lev-unsorted", "lev/lev-Q8_0.gguf", "lev-Q8_0", "support", jlm.DecisionLev, "unsorted", []string{"intent", "urgent", "frustration", "refund", "team"}},
		{"lev-uncalibrated", "lev/lev-Q8_0.gguf", "lev-Q8_0", "support", jlm.DecisionLev, "uncalibrated", []string{"intent", "urgent"}},
		{"vllm-d3-causal", "vllm-d3/vllm-d3-edge-Q8_0.gguf", "vllm-d3-edge-Q8_0", "support", jlm.DecisionVllmD3, "causal", []string{"intent", "team"}},
		{"kev-same-half", "kev/Kev-4B-BF16.gguf", "Kev-4B-BF16", "support", jlm.DecisionKev, "same-half", []string{"intent", "team"}},
		{"kev-key-before-end", "kev/Kev-4B-BF16.gguf", "Kev-4B-BF16", "support", jlm.DecisionKev, "key-before-end", []string{"intent", "team"}},
		{"kev-uncalibrated", "kev/Kev-4B-BF16.gguf", "Kev-4B-BF16", "support", jlm.DecisionKev, "uncalibrated", []string{"intent", "team"}},
		{"pplx-decider-codes-shifted", "pplx-decider/pplx-decider-v1.1-27b-Q8_0.gguf", "pplx-decider-v1.1-27b-Q8_0", "support", jlm.DecisionPplxDecider, "codes-shifted", []string{"intent", "team"}},
		{"openjev-false-first", "openjev/tinyopenjev-for-testing-Q8_0.gguf", "tinyopenjev-for-testing-Q8_0", "ambiguous", jlm.DecisionOpenJev, "false-first", []string{"compensate", "churn", "damage"}},
		{"openjev-uncalibrated", "openjev/tinyopenjev-for-testing-Q8_0.gguf", "tinyopenjev-for-testing-Q8_0", "ambiguous", jlm.DecisionOpenJev, "uncalibrated", []string{"compensate", "churn", "damage"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			m := openDecision(t, c.gguf, c.kind)
			defer m.Close()
			m.SetKVF16(false)
			gold := readDecisionGold(t, c.gold+"."+c.req+".llamacpp.json")
			state, all := decisionRequest(t, filepath.Join("testdata", "decision", c.req+".json"))
			var qs []DecisionQuestion
			for _, q := range all {
				for _, id := range c.ask {
					if q.ID == id {
						qs = append(qs, q)
					}
				}
			}
			d, err := m.NewDecider(4096)
			if err != nil {
				t.Fatal(err)
			}
			defer d.Close()
			d.violation = c.violation
			ans, err := d.Decide(state, qs, nil)
			if err != nil {
				t.Fatal(err)
			}
			worst := 0.0
			for i, q := range qs {
				g := gold.Answers[q.ID]
				if q.Type == jlm.QuestionNoul {
					worst = max(worst, math.Abs(ans[i].Noul-g.Noul))
					continue
				}
				for j, o := range q.Options {
					worst = max(worst, math.Abs(ans[i].Probs[j]-g.Probabilities[o.Key]))
				}
			}
			t.Logf("%s moves the answers by %.4f", c.violation, worst)
			if worst <= decisionSeen {
				t.Errorf("removing %s moved no answer past %.2f: the gate does not see it", c.violation, decisionSeen)
			}
		})
	}
}

// checkDecisions holds every answer to the golden's within tol.
func checkDecisions(t *testing.T, qs []DecisionQuestion, ans []DecisionAnswer, gold decisionGold, tol float64, confidence bool) {
	t.Helper()
	worst := 0.0
	for i, q := range qs {
		g, ok := gold.Answers[q.ID]
		if !ok {
			t.Fatalf("the golden has no answer for %s", q.ID)
		}
		a := ans[i]
		if a.Type.String() != g.Type {
			t.Errorf("%s: type %v, golden %s", q.ID, a.Type, g.Type)
		}
		diff := func(what string, got, want float64) {
			if math.IsNaN(got) || math.IsInf(got, 0) {
				t.Errorf("%s %s: %v is not finite", q.ID, what, got)
				return
			}
			d := math.Abs(got - want)
			worst = max(worst, d)
			if d > tol {
				t.Errorf("%s %s: %.4f, llama.cpp %.4f", q.ID, what, got, want)
			}
		}
		switch q.Type {
		case jlm.QuestionNoul:
			diff("noul", a.Noul, g.Noul)
		default:
			// The pick is held unless the two options are a tie within the
			// tolerance: llama.cpp's own probabilities of its choice and
			// ours less than two tolerances apart, where both picks sit
			// inside the band every probability is held to (vllm-d3's
			// ambiguous issue question: 0.42 against 0.57 at llama.cpp,
			// 0.50 against 0.50 on the f32 cache). The values are held
			// either way.
			if q.Type == jlm.QuestionChoice && a.Choice != g.Choice {
				if m := g.Probabilities[g.Choice] - g.Probabilities[a.Choice]; m < 2*tol {
					t.Logf("%s: chose %q, llama.cpp %q, a tie: llama.cpp's margin %.4f is under %.4f",
						q.ID, a.Choice, g.Choice, m, 2*tol)
				} else {
					t.Errorf("%s: chose %q, llama.cpp %q", q.ID, a.Choice, g.Choice)
				}
			}
			if q.Type == jlm.QuestionScore {
				diff("score", a.Score, g.Score)
			}
			if confidence {
				diff("confidence", a.Confidence, g.Confidence)
			}
			for j, o := range q.Options {
				diff("p("+o.Key+")", a.Probs[j], g.Probabilities[o.Key])
			}
		}
	}
	t.Logf("worst difference from llama.cpp: %.5f", worst)
}

// TestDecisionPicksOnTheHost holds the host's label pick (State.pickOnHost):
// each decoder readout's prompts project the labels' rows alone on the host
// (counted, so a pick that never ran is a failure), its answers are the whole
// head's within the matvec's reduction order, and a pick one row past each
// label moves them -- the answers are what the pick read.
func TestDecisionPicksOnTheHost(t *testing.T) {
	for _, c := range decisionCases {
		if c.kind == jlm.DecisionLaya || c.kind == jlm.DecisionClef || c.kind == jlm.DecisionD1Omni || c.kind == jlm.DecisionKev {
			continue // an encoder's scorer and Clef's span head read no label logits
		}
		t.Run(c.name, func(t *testing.T) {
			m := openDecision(t, c.gguf, c.kind)
			defer m.Close()
			run := func(violation string) ([][]DecisionAnswer, int) {
				d, err := m.NewDecider(4096)
				if err != nil {
					t.Fatal(err)
				}
				defer d.Close()
				d.violation = violation
				var out [][]DecisionAnswer
				for _, req := range decisionRequests {
					state, qs := decisionRequest(t, filepath.Join("testdata", "decision", req+".json"))
					ans, err := d.Decide(state, qs, nil)
					if err != nil {
						t.Fatal(err)
					}
					out = append(out, ans)
				}
				return out, d.picked
			}
			got, picked := run("")
			if picked == 0 {
				t.Fatal("no prompt's labels were picked on the host: the whole head ran")
			}
			whole, wp := run("no-pick")
			if wp != 0 {
				t.Fatalf("the no-pick arm picked %d prompts", wp)
			}
			w := decisionDiff(t, whole, got)
			t.Logf("%d prompts picked on the host; against the whole head %.2e", picked, w)
			if w > 1e-4 {
				t.Errorf("the picked labels answer %.6f from the whole head's", w)
			}
			off, _ := run("pick-off-by-one")
			if seen := decisionDiff(t, got, off); seen <= c.tol {
				t.Errorf("a pick one past each label moved the answers %.5f: the gate does not see the pick", seen)
			}
		})
	}
}
