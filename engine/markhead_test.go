package engine

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
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

// TestLayaFeaturesAreLoadBearing removes each of the graph's features from
// Laya's fixture -- the sliding window (every layer global), the local
// rotary base (one base), NEOX pairs, the LayerNorm, GELU -- by restating
// its container, and asks every question that is not a choice through the
// choice's type row: each moves a score past ten times the band
// TestMarkerHeadMatchesTheReference holds the container to, so none is a
// feature the gate cannot see.
func TestLayaFeaturesAreLoadBearing(t *testing.T) {
	mc := markerCases()[0]
	src, ok := existingModel(testmodels.Path(mc.src))
	if !ok {
		testmodels.Missing(t, "MODEL MISSING: %s (scripts/decisiongold.sh names its source)", testmodels.Path(mc.src))
	}
	path := jlmOf(t, src)
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	c := m.Container.Config()
	feats := jlm.FlagRopeNeox | jlm.FlagLayerNorm
	// The gated SiLU, stated as the converter states it, in place of the
	// GELU (erf) the container states.
	silu := actsWith(m, 0)
	if c.SWAWindow == 0 || c.RopeBaseSWA == c.RopeBase || c.Flags&feats != feats || reflect.DeepEqual(c.Acts, silu.Acts) {
		t.Fatalf("the fixture states a window of %d, bases %v and %v, flags %#x, activations %v: a feature the gate would remove is not there",
			c.SWAWindow, c.RopeBase, c.RopeBaseSWA, c.Flags, c.Acts)
	}
	same := func(q DecisionQuestion) (jlm.QuestionType, bool) { return q.Type, true }
	if clean := markerWorst(t, m, mc.gold, same); clean > mc.tol {
		t.Fatalf("the fixture's scores sit %.3g from the reference, past the band %.3g", clean, mc.tol)
	}
	for _, v := range []struct {
		name    string
		restate func(c *jlm.Config)
	}{
		{"every layer global", func(c *jlm.Config) { c.SWAWindow = 0 }},
		{"one rotary base", func(c *jlm.Config) { c.RopeBaseSWA = c.RopeBase }},
		{"pairs interleaved", func(c *jlm.Config) { c.Flags &^= jlm.FlagRopeNeox }},
		{"an RMS norm", func(c *jlm.Config) { c.Flags &^= jlm.FlagLayerNorm }},
		{"SiLU for GELU", func(c *jlm.Config) { c.Acts, c.ActLayers = silu.Acts, silu.ActLayers }},
	} {
		t.Run(v.name, func(t *testing.T) {
			rm := restatedCopy(t, path, v.restate)
			defer rm.Close()
			bad := markerWorst(t, rm, mc.gold, same)
			t.Logf("%s: scores %.4f from the reference", v.name, bad)
			if !(bad > 10*mc.tol) {
				t.Errorf("VIOLATION: %s moves no score past %.3g: the gate does not see it", v.name, 10*mc.tol)
			}
		})
	}
	t.Run("the choice's type row", func(t *testing.T) {
		choice := func(q DecisionQuestion) (jlm.QuestionType, bool) {
			return jlm.QuestionChoice, q.Type != jlm.QuestionChoice
		}
		bad := markerWorst(t, m, mc.gold, choice)
		t.Logf("the choice's type row for every other question: scores %.4f from the reference", bad)
		if !(bad > 10*mc.tol) {
			t.Errorf("VIOLATION: the choice's type row moves no score past %.3g: the gate does not see it", 10*mc.tol)
		}
	})
}

// markerWorst is how far m's scores of the support and ambiguous requests
// sit from base's reference at worst, each question asked through the type
// row typ gives it, none where it says not to, on the CPU device.
func markerWorst(t *testing.T, m *model.Model, base string, typ func(DecisionQuestion) (jlm.QuestionType, bool)) float64 {
	t.Helper()
	in, err := NewInstance(m, 1024, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	worst, asked := 0.0, 0
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
			ty, ask := typ(q)
			ids, marks, want := ref.IDs[q.ID], ref.Markers[q.ID], ref.Logits[q.ID]
			if !ask {
				continue
			}
			if len(ids) == 0 || len(marks) != len(want) {
				t.Fatalf("%s %s: %d ids, %d markers, %d logits in the reference", req, q.ID, len(ids), len(marks), len(want))
			}
			got := make([]float32, len(marks))
			if err := in.MarkScores(ids, int(ty), marks, got); err != nil {
				t.Fatalf("%s %s: %v", req, q.ID, err)
			}
			for j := range got {
				d := math.Abs(float64(got[j]) - want[j])
				if math.IsNaN(d) {
					return math.Inf(1)
				}
				worst = max(worst, d)
			}
			asked++
		}
	}
	if asked == 0 {
		t.Fatal("no question was asked")
	}
	return worst
}

// restatedCopy is the container at src copied and restated by restate,
// opened.
func restatedCopy(t *testing.T, src string, restate func(c *jlm.Config)) *model.Model {
	t.Helper()
	b, err := os.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "restated"+model.Ext)
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	if changed, err := jlm.Restate(path, func(_ *jlm.File, c *jlm.Config) (bool, error) {
		restate(c)
		return true, nil
	}); err != nil || !changed {
		t.Fatalf("restating %s: changed %v, %v", src, changed, err)
	}
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	return m
}
