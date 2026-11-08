package engine

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// markerAnswers is one request's goldens from Laya's own code
// (scripts/layagold.py): RLAgent.system_one's answers, and each question's
// ids and marker rows as build_sequence made them.
type markerAnswers struct {
	Answers map[string]struct {
		Choice        string             `json:"choice"`
		Score         *float64           `json:"score"`
		Noul          *float64           `json:"noul"`
		Confidence    *float64           `json:"confidence"`
		Probabilities map[string]float64 `json:"probabilities"`
	} `json:"answers"`
	IDs       map[string][]int32 `json:"ids"`
	Markers   map[string][]int   `json:"markers"`
	Questions []struct {
		ID      string  `json:"id"`
		IDs     []int32 `json:"ids"`
		Markers []int   `json:"markers"`
	} `json:"questions"`
}

// TestMarkerPromptMatchesTheReference holds a marker head's prompt, the
// container's program (jlm.MarkerPrompt), to Laya's build_sequence -- every
// question's ids and marker rows, exactly -- and Decide's answers to
// RLAgent.system_one's, on the CPU and with every block on each card. The
// requests' questions cut the head's budget (ten options), write JSON
// instructions and leave descriptions out, so each branch of the program
// runs.
func TestMarkerPromptMatchesTheReference(t *testing.T) {
	for _, mc := range markerCases() {
		name, base, tol := mc.src, mc.gold, mc.tol
		t.Run(filepath.Base(name), func(t *testing.T) {
			src, ok := existingModel(testmodels.Path(name))
			if !ok {
				testmodels.Missing(t, "MODEL MISSING: %s (scripts/decisiongold.sh names its source)", testmodels.Path(name))
			}
			m, err := model.Open(jlmOf(t, src))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if m.Container.Config().MarkerPrompt == nil {
				t.Fatalf("%s states no marker prompt", name)
			}
			t.Run("cpu", func(t *testing.T) { markerPromptArm(t, m, nil, base, tol) })
			for _, spec := range stepDevices() {
				t.Run(spec, func(t *testing.T) {
					g, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
					if err != nil || g == nil {
						noDevice(t, spec, err)
					}
					defer g.Close()
					markerPromptArm(t, m, g, base, tol)
				})
			}
		})
	}
}

func markerPromptArm(t *testing.T, m *model.Model, g *gpu.GPU, base string, tol float64) {
	in, err := NewInstance(m, 1024, WithKVF16(false))
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
	// A probability is the scores' softmax at a temperature near 1 or
	// above: it moves by no more than the scores' band, plus the rounding
	// to four places the reference writes.
	ptol := tol + 1e-4
	asked := 0
	for _, req := range []string{"support", "ambiguous"} {
		b, err := os.ReadFile(filepath.Join("testdata", "decision", base+"."+req+".reference.json"))
		if err != nil {
			t.Fatal(err)
		}
		ref := markerAnswers{IDs: map[string][]int32{}, Markers: map[string][]int{}}
		if err := json.Unmarshal(b, &ref); err != nil {
			t.Fatal(err)
		}
		for _, q := range ref.Questions {
			ref.IDs[q.ID], ref.Markers[q.ID] = q.IDs, q.Markers
		}
		state, qs := decisionRequest(t, req)
		for i := range qs {
			q := &qs[i]
			ids, marks, err := markedPrompt(d.mp, d.encodePlain, state, q, in.MaxSeq(), 0)
			if err != nil {
				t.Fatalf("%s %s: %v", req, q.ID, err)
			}
			if want := ref.IDs[q.ID]; !slices.Equal(ids, want) {
				at := 0
				for at < min(len(ids), len(want)) && ids[at] == want[at] {
					at++
				}
				t.Errorf("%s %s: %d ids, the reference's %d, apart from %d: %v against %v", req, q.ID, len(ids), len(want), at,
					ids[at:min(len(ids), at+8)], want[at:min(len(want), at+8)])
			}
			if want := ref.Markers[q.ID]; !slices.Equal(marks, want) {
				t.Errorf("%s %s: markers %v, the reference's %v", req, q.ID, marks, want)
			}
		}
		answers, err := d.Decide(state, qs)
		if err != nil {
			t.Fatalf("%s: %v", req, err)
		}
		for i, q := range qs {
			a, want := answers[i], ref.Answers[q.ID]
			near := func(what string, got, ref, band float64) {
				if math.IsNaN(got) || math.Abs(got-ref) > band {
					t.Errorf("%s %s: %s %.4f, the reference's %.4f (band %.3g)", req, q.ID, what, got, ref, band)
				}
			}
			k := float64(len(q.Options))
			switch {
			case want.Noul != nil:
				near("noul", a.Noul, *want.Noul, ptol)
			default:
				best, second := 0.0, 0.0
				for j, o := range q.Options {
					p := want.Probabilities[o.Key]
					near("P("+o.Key+")", a.Probs[j], p, ptol)
					if p > best {
						best, second = p, best
					} else if p > second {
						second = p
					}
				}
				if want.Score != nil {
					near("score", a.Score, *want.Score, k*ptol)
				} else if best-second > 2*ptol && a.Choice != want.Choice {
					t.Errorf("%s %s: chose %q, the reference %q", req, q.ID, a.Choice, want.Choice)
				}
				if want.Confidence != nil {
					near("confidence", a.Confidence, *want.Confidence, k*ptol)
				}
			}
			asked++
		}
	}
	if asked == 0 {
		t.Fatal("no question was asked")
	}
	t.Logf("%d questions: the reference's ids, markers and answers", asked)
}
