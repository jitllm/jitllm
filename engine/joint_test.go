package engine

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// jointGold is joint_schema_model.py's run of a request: its prompt's ids,
// each question's type row and spans, and each option's logit.
type jointGold struct {
	IDs       []int32 `json:"ids"`
	Questions []struct {
		ID      string   `json:"id"`
		Type    int      `json:"type"`
		Span    [2]int   `json:"span"`
		Options [][2]int `json:"options"`
		Keys    []string `json:"option_ids"`
	} `json:"questions"`
	Logits map[string][]float64 `json:"logits"`
}

// TestJointDecisionMatchesTheReference holds a joint head (jlm.JointHead) to
// Clef's own JointSchemaHead on its fixture -- the prompt's ids and every
// span to the token, each option's logit, the answers -- and the release
// to its own code's run of it alike, on the CPU and with every block on each
// card.
func TestJointDecisionMatchesTheReference(t *testing.T) {
	cases := []struct {
		gguf, gold, ref string
		tol, logit      float64
	}{
		{"clef/fixture/clef-fixture.gguf", "clef-fixture", "clef-fixture", 1e-3, 2e-3},
		// The release: its prompts, logits and answers held to Clef's own code
		// (f32 on the BF16 weights, clefgold.py real). The band is Q8_0's
		// rounding with the CPU's int8 activations through the backbone, as
		// llama.cpp's: logits 0.106 and 0.181, answers 0.026 and 0.053 on the
		// CPU (a near tie), 0.040 and 0.058, 0.014 and 0.011 on a card;
		// llama.cpp's answers sit 0.017 and 0.038 from the reference's. Not
		// llama.cpp's answers: its confidence is not the chosen option's
		// probability (docs/divergences.md).
		{"clef/Clef-Flash-Q8_0.gguf", "clef", "clef", 0.06, 0.2},
	}
	for _, c := range cases {
		t.Run(c.gold, func(t *testing.T) {
			src, ok := existingModel(testmodels.Path(c.gguf))
			if !ok {
				testmodels.Missing(t, "MODEL MISSING: %s (scripts/decisiongold.sh names its source)", testmodels.Path(c.gguf))
			}
			m, err := model.Open(jlmOf(t, src))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if cfg := m.Container.Config(); cfg.Decision != jlm.DecisionClef || cfg.JointHead == nil || !m.Cfg.Joint {
				t.Fatalf("the container says readout %v, joint head %v", cfg.Decision, cfg.JointHead != nil)
			}
			arm := func(t *testing.T, g *gpu.GPU) {
				in, err := NewInstance(m, 4096, WithKVF16(false))
				if err != nil {
					t.Fatal(err)
				}
				defer in.Close()
				if g != nil {
					if err := in.SetDevice(g); err != nil {
						t.Fatal(err)
					}
					if in.GPULayers() != m.Cfg.NLayer {
						t.Fatalf("the card took %d of %d blocks: %s", in.GPULayers(), m.Cfg.NLayer, g.Err())
					}
				}
				d, err := NewDecider(in)
				if err != nil {
					t.Fatal(err)
				}
				for _, req := range []string{"support", "ambiguous"} {
					state, qs := decisionRequest(t, req)
					if c.ref != "" {
						checkJointPrompt(t, d, req, c.ref, c.logit)
					}
					ans, err := d.Decide(state, qs)
					if err != nil {
						t.Fatalf("%s: %v", req, err)
					}
					kind := ".llamacpp.json"
					if c.ref == c.gold {
						kind = ".reference.json"
					}
					worst := checkDecisions(t, qs, ans, readDecisionGold(t, c.gold+"."+req+kind), max(c.tol, 1e-3))
					t.Logf("%s: %d prompt tokens, worst difference %.5f", req, d.InputTokens(), worst)
				}
			}
			t.Run("cpu", func(t *testing.T) { arm(t, nil) })
			for _, spec := range stepDevices() {
				t.Run(spec, func(t *testing.T) {
					g, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
					if err != nil || g == nil {
						noDevice(t, spec, err)
					}
					defer g.Close()
					arm(t, g)
				})
			}
		})
	}
}

// checkJointPrompt holds req's prompt, spans and logits to the reference's
// run of it.
func checkJointPrompt(t *testing.T, d *Decider, req, gold string, tol float64) {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "decision", gold+"."+req+".reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var g jointGold
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	state, qs := decisionRequest(t, req)
	ids, jq, shown, err := d.jointPrompt(state, qs)
	if err != nil {
		t.Fatalf("%s: %v", req, err)
	}
	if !slices.Equal(ids, g.IDs) {
		at := 0
		for at < min(len(ids), len(g.IDs)) && ids[at] == g.IDs[at] {
			at++
		}
		t.Fatalf("%s: %d prompt ids, the reference's %d, first apart at %d", req, len(ids), len(g.IDs), at)
	}
	if len(jq) != len(g.Questions) {
		t.Fatalf("%s: %d questions, the reference's %d", req, len(jq), len(g.Questions))
	}
	no := 0
	for i, q := range g.Questions {
		keys := make([]string, len(shown[i]))
		for k, oi := range shown[i] {
			keys[k] = qs[i].Options[oi].Key
		}
		if jq[i].Type != q.Type || jq[i].Span != q.Span || !slices.Equal(jq[i].Options, q.Options) || !slices.Equal(keys, q.Keys) {
			t.Fatalf("%s: question %s is type %d spans %v %v options %v, the reference's %d %v %v %v",
				req, q.ID, jq[i].Type, jq[i].Span, jq[i].Options, keys, q.Type, q.Span, q.Options, q.Keys)
		}
		no += len(q.Options)
	}
	logits := make([]float32, no)
	if err := d.in.JointScores(ids, jq, logits); err != nil {
		t.Fatalf("%s: %v", req, err)
	}
	worst, at := 0.0, 0
	for _, q := range g.Questions {
		for k, want := range g.Logits[q.ID] {
			got := float64(logits[at+k])
			if math.IsNaN(got) || math.Abs(got-want) > tol {
				t.Errorf("%s: %s option %s logit %.5f, the reference's %.5f", req, q.ID, q.Keys[k], got, want)
			}
			worst = max(worst, math.Abs(got-want))
		}
		at += len(q.Options)
	}
	t.Logf("%s: the prompt, its spans and %d logits match the reference's, worst logit %.2e", req, no, worst)
}
