package engine

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/tok/jinja"
)

// The label readouts against llama.cpp's /v1/systemone on the same GGUF
// (scripts/decisiongold.sh writes the goldens). RULE 7m: llama.cpp is the
// oracle for the readout here because it reads the GGUF's own template and
// metadata, which is what the container carries.

type decisionGold struct {
	Answers map[string]struct {
		Type          string             `json:"type"`
		Noul          float64            `json:"noul"`
		Choice        string             `json:"choice"`
		Score         float64            `json:"score"`
		Confidence    float64            `json:"confidence"`
		Probabilities map[string]float64 `json:"probabilities"`
	} `json:"answers"`
}

// decisionCases are the label readouts and the llama.cpp answers they are
// held to: gold is the golden's prefix, <gold>.<request>.llamacpp.json, and
// tol how far a probability may sit from llama.cpp's on the same bytes on
// either KV cache. d1 measured 0.021 at worst (the billing/returns near-tie),
// lev 0.090 (the ambiguous request's tone, a near-tie that Q8_0 rounding
// moves), vllm-d3 0.078 (its ambiguous issue question, a near-tie).
type decisionCase struct {
	name, gguf, gold string
	kind             jlm.DecisionKind
	tol              float64
}

func decisionCases() []decisionCase {
	return []decisionCase{
		{"d1-3B", "d1/d1-3B-Q8_0.gguf", "d1-3B-Q8_0", jlm.DecisionLFM2D1, 0.03},
		{"lev", "lev/lev-Q8_0.gguf", "lev-Q8_0", jlm.DecisionLev, 0.10},
		{"openjev", "openjev/tinyopenjev-for-testing-Q8_0.gguf", "tinyopenjev-for-testing-Q8_0", jlm.DecisionOpenJev, 0.01},
		{"vllm-d3", "vllm-d3/vllm-d3-edge-Q8_0.gguf", "vllm-d3-edge-Q8_0", jlm.DecisionVllmD3, 0.08},
	}
}

// decisionSeen is how far a rewritten readout must move an answer to count
// as seen.
const decisionSeen = 0.03

func openDecision(t *testing.T, c decisionCase) *model.Model {
	t.Helper()
	src, ok := existingModel(testmodels.Path(c.gguf))
	if !ok {
		testmodels.Missing(t, "MODEL MISSING: %s (scripts/decisiongold.sh names its source)", testmodels.Path(c.gguf))
	}
	m, err := model.Open(jlmOf(t, src))
	if err != nil {
		t.Fatal(err)
	}
	if cfg := m.Container.Config(); cfg.Decision != c.kind || cfg.DecisionLabels == nil {
		m.Close()
		t.Fatalf("the container says readout %v (labels %v), want %v", cfg.Decision, cfg.DecisionLabels != nil, c.kind)
	}
	return m
}

func decisionRequest(t testing.TB, name string) (jinja.Value, []DecisionQuestion) {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "decision", name+".json"))
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

// TestDecisionMatchesLlamaCpp holds each label readout's answers to
// llama.cpp's on both requests, on an f32 and an f16 KV cache: every
// prompt's labels picked by the head and read by the composed readout part.
// A readout whose calibration is rewritten (every question at temperature 8)
// must move the answers past decisionSeen.
func TestDecisionMatchesLlamaCpp(t *testing.T) {
	for _, c := range decisionCases() {
		t.Run(c.name, func(t *testing.T) {
			m := openDecision(t, c)
			defer m.Close()
			for _, f16 := range []bool{false, true} {
				in, err := NewInstance(m, 4096, WithKVF16(f16))
				if err != nil {
					t.Fatal(err)
				}
				d, err := NewDecider(in)
				if err != nil {
					in.Close()
					t.Fatal(err)
				}
				for _, req := range []string{"support", "ambiguous"} {
					state, qs := decisionRequest(t, req)
					ans, err := d.Decide(state, qs)
					if err != nil {
						t.Fatalf("kv f16 %v, %s: %v", f16, req, err)
					}
					worst := checkDecisions(t, qs, ans, readDecisionGold(t, c.gold+"."+req+".llamacpp.json"), c.tol)
					t.Logf("kv f16 %v, %s: %d prompt tokens, worst difference from llama.cpp %.5f", f16, req, d.InputTokens(), worst)
				}
				if !f16 {
					cfg := *d.cfg
					cfg.DecisionTemps = []jlm.DecisionTemp{{Type: jlm.QuestionChoice, T: 8},
						{Type: jlm.QuestionScore, T: 8}, {Type: jlm.QuestionNoul, T: 8}}
					d.cfg = &cfg
					state, qs := decisionRequest(t, "support")
					ans, err := d.Decide(state, qs)
					if err != nil {
						t.Fatal(err)
					}
					gold := readDecisionGold(t, c.gold+".support.llamacpp.json")
					if moved := decisionDiff(qs, ans, gold); moved <= decisionSeen {
						t.Errorf("the calibration at 8 moved no answer past %.2f (%.4f): the gate does not see it", decisionSeen, moved)
					}
				}
				in.Close()
			}
		})
	}
}

// decisionDiff is the largest difference of ans from gold.
func decisionDiff(qs []DecisionQuestion, ans []DecisionAnswer, gold decisionGold) float64 {
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
	return worst
}

// checkDecisions holds every answer to the golden's within tol and returns
// the largest difference.
func checkDecisions(t *testing.T, qs []DecisionQuestion, ans []DecisionAnswer, gold decisionGold, tol float64) float64 {
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
		if q.Type == jlm.QuestionNoul {
			diff("noul", a.Noul, g.Noul)
			continue
		}
		// The pick is held unless the two options are a tie within the
		// tolerance: llama.cpp's own probabilities of its choice and ours
		// less than two tolerances apart. The values are held either way.
		if q.Type == jlm.QuestionChoice && a.Choice != g.Choice {
			if m := g.Probabilities[g.Choice] - g.Probabilities[a.Choice]; m < 2*tol {
				t.Logf("%s: chose %q, llama.cpp %q, a tie: llama.cpp's margin %.4f is under %.4f", q.ID, a.Choice, g.Choice, m, 2*tol)
			} else {
				t.Errorf("%s: chose %q, llama.cpp %q", q.ID, a.Choice, g.Choice)
			}
		}
		if q.Type == jlm.QuestionScore {
			diff("score", a.Score, g.Score)
		}
		diff("confidence", a.Confidence, g.Confidence)
		for j, o := range q.Options {
			diff("p("+o.Key+")", a.Probs[j], g.Probabilities[o.Key])
		}
	}
	return worst
}

// TestDecisionResponseMatchesLlamaCpp holds the response body to llama.cpp's
// for vllm-d3's legend request: the answer's shape, and a score's legend
// with each description that is not text as its sorted-key JSON, a string,
// as the container states. Without the statement the legend is the
// request's own JSON, which must fail it.
func TestDecisionResponseMatchesLlamaCpp(t *testing.T) {
	c := decisionCases()[3]
	m := openDecision(t, c)
	defer m.Close()
	in, err := NewInstance(m, 4096)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	d, err := NewDecider(in)
	if err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(filepath.Join("testdata", "decision", "legend.json"))
	if err != nil {
		t.Fatal(err)
	}
	_, state, qs, err := ParseDecisionRequest(body)
	if err != nil {
		t.Fatal(err)
	}
	ans, err := d.Decide(state, qs)
	if err != nil {
		t.Fatal(err)
	}
	type resp struct {
		Answers map[string]struct {
			Type   string            `json:"type"`
			Legend map[string]string `json:"legend"`
		} `json:"answers"`
	}
	var gold resp
	b, err := os.ReadFile(filepath.Join("testdata", "decision", c.gold+".legend.llamacpp.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &gold); err != nil {
		t.Fatal(err)
	}
	legend := func(out []byte) (map[string]string, error) {
		var r resp
		if err := json.Unmarshal(out, &r); err != nil {
			return nil, err
		}
		return r.Answers["anger"].Legend, nil
	}
	got, err := legend(d.DecisionResponse(c.gguf, qs, ans))
	if err != nil {
		t.Fatal(err)
	}
	want := gold.Answers["anger"].Legend
	if len(got) != len(want) || len(want) == 0 {
		t.Fatalf("legend %v, llama.cpp %v", got, want)
	}
	for k, w := range want {
		if got[k] != w {
			t.Errorf("legend %s: %q, llama.cpp %q", k, got[k], w)
		}
	}
	lab := *d.lab
	lab.SortedLegend = false
	d.lab = &lab
	if off, err := legend(d.DecisionResponse(c.gguf, qs, ans)); err == nil && off["0"] == want["0"] {
		t.Errorf("the legend without the statement is llama.cpp's too: the gate does not see it")
	}
}
