package engine

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// markerReference is one request's goldens from the model's own code
// (scripts/layagold.py, scripts/d1omnigold.py): each question's prompt ids,
// its markers' rows and the scorer's raw logits at them, keyed by question
// (Laya's) or listed (d1-omni's).
type markerReference struct {
	IDs       map[string][]int32   `json:"ids"`
	Markers   map[string][]int     `json:"markers"`
	Logits    map[string][]float64 `json:"logits"`
	Questions []struct {
		ID      string    `json:"id"`
		IDs     []int32   `json:"ids"`
		Markers []int     `json:"markers"`
		Logits  []float64 `json:"logits"`
	} `json:"questions"`
}

// markerCase is a marker-head model the gate holds to its reference: its
// source, its goldens' prefix and the band its weights' format allows.
type markerCase struct {
	src, gold string
	tol       float64
}

// markerCases is Laya's fixture (f32 against the reference's f32) and the
// published Laya (converted to BF16).
func markerCases() []markerCase {
	return []markerCase{{"laya/fixture/laya-fixture.gguf", "laya-fixture", 2e-3}, {"laya/gguf/Laya-BF16.gguf", "laya", 5e-2},
		{"d1-omni/d1-omni-600M-F16.gguf", "d1-omni", 5e-2}}
}

// TestMarkerHeadMatchesTheReference holds a marker head (jlm.Flag2HeadBlocks)
// to Laya's own model: each question's prompt ids, as the reference built
// them, through the trunk, the question type's row and the head's blocks to
// the scorer at its markers, on the CPU and with every block on each card,
// against the reference's raw logits. The fixture's weights are all drawn at
// random, so a dropped type row, norm or head block moves every score.
func TestMarkerHeadMatchesTheReference(t *testing.T) {
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
			if m.Cfg.Head == nil {
				t.Fatalf("%s states no head blocks: the gate would run the trunk alone", name)
			}
			t.Run("cpu", func(t *testing.T) { markerArm(t, m, nil, base, tol) })
			for _, spec := range stepDevices() {
				t.Run(spec, func(t *testing.T) {
					g, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
					if err != nil || g == nil {
						noDevice(t, spec, err)
					}
					defer g.Close()
					markerArm(t, m, g, base, tol)
				})
			}
		})
	}
}

func markerArm(t *testing.T, m *model.Model, g *gpu.GPU, base string, tol float64) {
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
	asked := 0
	for _, req := range []string{"support", "ambiguous"} {
		b, err := os.ReadFile(filepath.Join("testdata", "decision", base+"."+req+".reference.json"))
		if err != nil {
			t.Fatal(err)
		}
		ref := markerReference{IDs: map[string][]int32{}, Markers: map[string][]int{}, Logits: map[string][]float64{}}
		if err := json.Unmarshal(b, &ref); err != nil {
			t.Fatal(err)
		}
		for _, q := range ref.Questions {
			ref.IDs[q.ID], ref.Markers[q.ID], ref.Logits[q.ID] = q.IDs, q.Markers, q.Logits
		}
		_, qs := decisionRequest(t, req)
		for _, q := range qs {
			ids, marks, want := ref.IDs[q.ID], ref.Markers[q.ID], ref.Logits[q.ID]
			if len(ids) == 0 || len(marks) != len(want) {
				t.Fatalf("%s %s: %d ids, %d markers, %d logits in the reference", req, q.ID, len(ids), len(marks), len(want))
			}
			got := make([]float32, len(marks))
			if err := in.MarkScores(ids, int(q.Type), marks, got); err != nil {
				t.Fatalf("%s %s: %v", req, q.ID, err)
			}
			worst := 0.0
			for i := range want {
				if math.IsNaN(float64(got[i])) || math.IsNaN(want[i]) {
					t.Fatalf("%s %s: score %d is %v, the reference's %v", req, q.ID, i, got[i], want[i])
				}
				worst = max(worst, math.Abs(float64(got[i])-want[i]))
			}
			if worst > tol {
				t.Errorf("%s %s: scores %v, the reference's %v (%.3g apart, the band %.3g)", req, q.ID, got, want, worst, tol)
			} else {
				t.Logf("%s %s: %d scores within %.3g of the reference", req, q.ID, len(want), worst)
			}
			asked++
		}
	}
	if asked == 0 {
		t.Fatal("no question was asked")
	}
}
