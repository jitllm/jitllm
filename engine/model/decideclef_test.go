package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// Clef against its own code (joint_schema_model.py in Cloudflare/clef-flash),
// the goldens written by scripts/clefgold.py: a Qwen3.5 backbone and a joint
// schema head built from the release's classes with every parameter random,
// converted by llama.cpp's own converter. Held: the prompt token for token,
// every question's and option's span, and the head's logits.

type clefGold struct {
	IDs       []int32 `json:"ids"`
	Questions []struct {
		ID      string   `json:"id"`
		Span    [2]int   `json:"span"`
		Options [][2]int `json:"options"`
	} `json:"questions"`
	Logits map[string][]float64 `json:"logits"`
}

const clefFixture = "clef/fixture/clef-fixture.gguf"

// clefFixtureTol is how far a logit may sit from the reference's: f32 weights
// on both sides, so what is left is reduction order (worst 1.7e-6 on the
// laptop's host).
const clefFixtureTol = 5e-6

// clefSeen is how far a removed feature must move a logit to count as seen:
// five times the tolerance. The tanh GELU in place of the erf one is the
// smallest at 5.3e-5 -- thirty times the worst f32 difference.
const clefSeen = 5 * clefFixtureTol

// clefRealTol is how far the release's logit may sit from its reference: Q8_0
// weights here against BF16 read as f32 there, through 32 blocks and the head:
// worst 0.076 and 0.156 of a logit on the two requests (logits from -4.8 to
// 1.3) on the laptop. llama.cpp on the same Q8_0 sits as far from the
// reference in probability (0.017 and 0.038) as jitllm does.
const clefRealTol = 0.2

func readClefGold(t *testing.T, req string) clefGold { return readClefGoldOf(t, "clef-fixture", req) }

func readClefGoldOf(t *testing.T, prefix, req string) clefGold {
	t.Helper()
	var g clefGold
	b, err := os.ReadFile(filepath.Join("testdata", "decision", prefix+"."+req+".reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

// clefLogits runs one request through d and returns its prompt and the
// head's logits in the prompt's option order.
func clefLogits(t *testing.T, d *Decider, req string) (*clefPrompt, []float32) {
	t.Helper()
	state, qs := decisionRequest(t, filepath.Join("testdata", "decision", req+".json"))
	if err := d.Prepare(state, qs, 0); err != nil {
		t.Fatal(err)
	}
	p, err := d.clefPrompt(state, qs)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(p.opts))
	if err := d.clefForward(p, out); err != nil {
		t.Fatal(err)
	}
	return p, out
}

// clefWorst is the largest distance of got from the golden's logits, in the
// golden's question order.
func clefWorst(t *testing.T, g clefGold, got []float32) float64 {
	t.Helper()
	worst, at := 0.0, 0
	for _, q := range g.Questions {
		for _, want := range g.Logits[q.ID] {
			x := float64(got[at])
			if math.IsNaN(x) || math.IsInf(x, 0) || math.IsNaN(want) {
				t.Fatalf("%s: logit %v against %v is not finite", q.ID, x, want)
			}
			worst = max(worst, math.Abs(x-want))
			at++
		}
	}
	if at != len(got) {
		t.Fatalf("%d logits, the reference %d", len(got), at)
	}
	return worst
}

// clefRefCases are the models held to Clef's own code: the random fixture
// (f32 on both sides) and the release, its reference run in f32 from the
// BF16 safetensors (clefgold.py real) against the Q8_0 GGUF here.
var clefRefCases = []struct {
	name, gguf, gold string
	tol              float64
}{
	{"fixture", clefFixture, "clef-fixture", clefFixtureTol},
	{"clef", "clef/Clef-Flash-Q8_0.gguf", "clef", clefRealTol},
}

func TestClefMatchesReference(t *testing.T) {
	for _, c := range clefRefCases {
		t.Run(c.name, func(t *testing.T) {
			m := openDecision(t, c.gguf, jlm.DecisionClef)
			defer m.Close()
			m.SetKVF16(false)
			d, err := m.NewDecider(4096)
			if err != nil {
				t.Fatal(err)
			}
			defer d.Close()
			for _, req := range decisionRequests {
				t.Run(req, func(t *testing.T) {
					g := readClefGoldOf(t, c.gold, req)
					p, got := clefLogits(t, d, req)
					if len(p.ids) != len(g.IDs) {
						t.Fatalf("the prompt is %d tokens, the reference's %d", len(p.ids), len(g.IDs))
					}
					for i := range p.ids {
						if p.ids[i] != g.IDs[i] {
							t.Fatalf("token %d is %d, the reference's %d", i, p.ids[i], g.IDs[i])
						}
					}
					at := 0
					for qi, q := range g.Questions {
						if s := p.qs[qi]; s.lo != q.Span[0] || s.hi != q.Span[1] {
							t.Errorf("%s: span [%d,%d), the reference's %v", q.ID, s.lo, s.hi, q.Span)
						}
						for _, o := range q.Options {
							if s := p.opts[at]; s.lo != o[0] || s.hi != o[1] {
								t.Errorf("%s option %d: span [%d,%d), the reference's %v", q.ID, at, s.lo, s.hi, o)
							}
							at++
						}
					}
					w := clefWorst(t, g, got)
					t.Logf("worst logit difference from Clef's own code: %.2e over %d options", w, len(got))
					if w > c.tol {
						t.Errorf("worst logit %.2e from the reference, tolerance %.0e", w, c.tol)
					}
				})
			}
		})
	}
}

// TestClefFeaturesAreLoadBearing removes each feature of Clef's head and
// requires the logits to leave the reference's by far more than the
// tolerance: a feature the gate cannot see is not gated (RULE 10).
func TestClefFeaturesAreLoadBearing(t *testing.T) {
	m := openDecision(t, clefFixture, jlm.DecisionClef)
	defer m.Close()
	m.SetKVF16(false)
	for _, v := range []string{"false-first", "unsorted", "no-hidden-norm", "no-lexical", "no-type",
		"no-self", "no-prior", "no-cosine", "first-row", "tanh-gelu"} {
		t.Run(v, func(t *testing.T) {
			d, err := m.NewDecider(4096)
			if err != nil {
				t.Fatal(err)
			}
			defer d.Close()
			d.violation = v
			worst := 0.0
			for _, req := range decisionRequests {
				g := readClefGold(t, req)
				_, got := clefLogits(t, d, req)
				if len(got) == countClefLogits(g) {
					worst = max(worst, clefWorstLoose(g, got))
				} else {
					worst = math.Inf(1)
				}
			}
			t.Logf("%s moves the logits by %.2e", v, worst)
			if worst <= clefSeen {
				t.Errorf("removing %s moved no logit past %.0e: the gate does not see it", v, clefSeen)
			}
		})
	}
}

func countClefLogits(g clefGold) int {
	n := 0
	for _, l := range g.Logits {
		n += len(l)
	}
	return n
}

// clefWorstLoose is clefWorst without the checks a violation may break.
func clefWorstLoose(g clefGold, got []float32) float64 {
	worst, at := 0.0, 0
	for _, q := range g.Questions {
		for _, want := range g.Logits[q.ID] {
			worst = max(worst, math.Abs(float64(got[at])-want))
			at++
		}
	}
	return worst
}
