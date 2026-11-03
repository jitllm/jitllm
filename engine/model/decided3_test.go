package model

import (
	"encoding/json"
	"image"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestVllmD3LegendIsCanonical holds vllm-d3's score legend to llama-server's
// on a request whose levels are described by objects (testdata/decision
// legend.json; the golden is llama-server's answer to it on the same GGUF):
// a description that is not text is given as its JSON with the keys sorted,
// in a string, and a text one as itself. The violation writes them as the
// request gave them, which must differ.
func TestVllmD3LegendIsCanonical(t *testing.T) {
	m := openDecision(t, "vllm-d3/vllm-d3-edge-Q8_0.gguf", jlm.DecisionVllmD3)
	defer m.Close()
	state, qs := decisionRequest(t, filepath.Join("testdata", "decision", "legend.json"))
	d, err := m.NewDecider(4096)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	ans, err := d.Decide(state, qs, nil)
	if err != nil {
		t.Fatal(err)
	}
	gold := readDecisionGold(t, "vllm-d3-edge-Q8_0.legend.llamacpp.json")
	checkDecisions(t, qs, ans, gold, 0.03, true)
	b, err := os.ReadFile(filepath.Join("testdata", "decision", "vllm-d3-edge-Q8_0.legend.llamacpp.json"))
	if err != nil {
		t.Fatal(err)
	}
	type legends struct {
		Answers map[string]struct {
			Legend map[string]json.RawMessage `json:"legend"`
		} `json:"answers"`
	}
	var want, got legends
	if err := json.Unmarshal(b, &want); err != nil {
		t.Fatal(err)
	}
	legendOf := func(ans []DecisionAnswer) map[string]json.RawMessage {
		if err := json.Unmarshal(DecisionResponse("m", qs, ans, 0), &got); err != nil {
			t.Fatal(err)
		}
		return got.Answers["anger"].Legend
	}
	w := want.Answers["anger"].Legend
	if len(w) != 3 {
		t.Fatalf("the golden's legend has %d levels", len(w))
	}
	same := func(l map[string]json.RawMessage) bool {
		for k, v := range w {
			if string(l[k]) != string(v) {
				return false
			}
		}
		return true
	}
	if l := legendOf(ans); !same(l) {
		t.Errorf("legend %s, llama.cpp %s", l, w)
	}
	ans[0].CanonicalLegend = false
	if same(legendOf(ans)) {
		t.Errorf("the request's own descriptions match llama.cpp's legend: the gate does not see the canonical form")
	}
}

// TestVllmD3PicturesMatchLlamaCpp holds vllm-d3's picture requests -- two
// pictures over a null state, and a chat-message state carrying one -- to
// llama-server's answers on the same GGUF and mmproj: the qwen3vl tower's
// rows between their markers, and the whole prompt, pictures and text, one
// bidirectional run (Instance.bidirAll). The violation runs it causally, which
// must move the answers.
func TestVllmD3PicturesMatchLlamaCpp(t *testing.T) {
	text, proj := testmodels.Path("vllm-d3/vllm-d3-edge-Q8_0.gguf"), testmodels.Path("vllm-d3/mmproj-vllm-d3-edge-Q8_0.gguf")
	if !vlmAvailable(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (RULE 11)", text)
	}
	m, err := Open(jlmOfPair(t, text, proj), noTune(), WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.Tower() == nil || m.Decision() != jlm.DecisionVllmD3 {
		t.Fatalf("the container carries tower %v and readout %v", m.Tower() != nil, m.Decision())
	}
	for _, req := range []string{"pictures", "chatpicture"} {
		t.Run(req, func(t *testing.T) {
			body, err := os.ReadFile(filepath.Join("testdata", "decision", req+".json"))
			if err != nil {
				t.Fatal(err)
			}
			r, err := ParseDecisionRequest(body)
			if err != nil {
				t.Fatal(err)
			}
			if len(r.Images) == 0 {
				t.Fatal("the request carries no picture, so this gate proves nothing")
			}
			imgs := make([]image.Image, len(r.Images))
			for i, u := range r.Images {
				if imgs[i], err = DecodeImageDataURL(u); err != nil {
					t.Fatal(err)
				}
			}
			gold := readDecisionGold(t, "vllm-d3-edge-Q8_0."+req+".llamacpp.json")
			decide := func(violation string) ([]DecisionAnswer, int) {
				d, err := m.NewDecider(4096)
				if err != nil {
					t.Fatal(err)
				}
				defer d.Close()
				d.violation = violation
				ans, err := d.Decide(r.State, r.Questions, imgs)
				if err != nil {
					t.Fatal(err)
				}
				return ans, d.InputTokens()
			}
			ans, tokens := decide("")
			if tokens != gold.Usage.InputTokens {
				t.Errorf("ran %d positions, llama.cpp %d: the prompts differ", tokens, gold.Usage.InputTokens)
			}
			// The text gate's band: the tower's int8 activations against
			// llama.cpp's float ones move the chat picture's photo/shapes
			// split 0.053 (0.012 on the two-picture request).
			checkDecisions(t, r.Questions, ans, gold, 0.08, true)
			bad, _ := decide("causal")
			seen := 0.0
			for i, q := range r.Questions {
				if q.Type == jlm.QuestionNoul {
					seen = max(seen, math.Abs(bad[i].Noul-ans[i].Noul))
					continue
				}
				for j := range q.Options {
					seen = max(seen, math.Abs(bad[i].Probs[j]-ans[i].Probs[j]))
				}
			}
			t.Logf("run causally, the answers move %.4f", seen)
			if seen <= 0.08 {
				t.Errorf("a causal prompt moved the answers %.4f: the gate does not see the run", seen)
			}
		})
	}
}
