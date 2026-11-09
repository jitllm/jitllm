package engine

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// decisionPictures is request name's pictures, each as the span the prefill
// encodes (Model.PictureSpan), decoded from the request's data URLs.
func decisionPictures(t *testing.T, m *model.Model, name string) []model.Span {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "decision", name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var req struct {
		Images []string `json:"images"`
	}
	if err := json.Unmarshal(b, &req); err != nil {
		t.Fatal(err)
	}
	if len(req.Images) == 0 {
		t.Fatalf("request %s holds no pictures: the gate would ask about none", name)
	}
	var out []model.Span
	for i, u := range req.Images {
		_, data, ok := strings.Cut(u, ",")
		if !ok {
			t.Fatalf("picture %d is no data URL", i)
		}
		raw, err := base64.StdEncoding.DecodeString(data)
		if err != nil {
			t.Fatal(err)
		}
		img, _, err := image.Decode(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		sp, err := m.PictureSpan(img)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, sp)
	}
	return out
}

// TestDecisionOverPicturesMatchesItsGoldens holds a decision readout asked
// about pictures to its golden: the label readouts' answers to llama.cpp's
// on the same GGUF and its projector, and Clef's joint head's to its own
// code's on its VL fixture; on the CPU and with every block on each card.
// The pictures are spans of the prompt where the systemone template writes
// them, through the model's tower.
func TestDecisionOverPicturesMatchesItsGoldens(t *testing.T) {
	cases := []struct {
		name, gguf, gold string
		tol              float64
		// cpuOnly names why the case runs no card arm yet.
		cpuOnly string
	}{
		{"d1-3B", "d1/d1-3B-Q8_0.gguf", "d1-3B-Q8_0.pictures.llamacpp.json", 0.05, ""},
		// A non-causal decoder: its prompt is one bidirectional run, the
		// pictures' rows among the text's. 0.039, 0.030 and 0.028 on the CPU,
		// its float forms and a card; with each span its own run the text
		// before a picture never saw it and read 0.092 and 0.070.
		{"vllm-d3", "vllm-d3/vllm-d3-edge-Q8_0.gguf", "vllm-d3-edge-Q8_0.pictures.llamacpp.json", 0.05, ""},
		// The same request against the model's own runtime (d3_runtime.py in
		// float32 on the CPU; scripts/d3gold.py): 0.068 on the CPU's float
		// forms and 0.070 on a card, where the checkpoint's own bfloat16 run
		// stands 0.068 from its float32 one; this GGUF's Q8_0 weights are
		// 0.036 of it (an F32 conversion read 0.032).
		{"vllm-d3-runtime", "vllm-d3/vllm-d3-edge-Q8_0.gguf", "vllm-d3-edge.pictures.reference.json", 0.08, ""},
		{"clef-fixture-vl", "clef/fixture-vl/clef-fixture-vl.gguf", "clef-fixture-vl.pictures.reference.json", 2e-3, ""},
		// A marker head: the pictures a prefix that reads itself alone
		// (jlm.Flag2MediaPrefix), the text after it, uncalibrated. 0.026
		// with the CPU's int8 activations, 0.019 with its float forms (the
		// per-family readout read 0.02); llama.cpp sits 0.0003 from the
		// model's own code here.
		{"d1-omni", "d1-omni/d1-omni-600M-F16.gguf", "d1-omni-600M-F16.pictures.llamacpp.json", 0.03, ""},
	}
	// int8Tol is a case's tolerance on the CPU's int8 activations where it is
	// wider than its float forms': vllm-d3's runtime case read 0.137 there.
	int8Tol := map[string]float64{"vllm-d3-runtime": 0.15}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			src, ok := existingModel(testmodels.Path(c.gguf))
			if !ok {
				testmodels.Missing(t, "MODEL MISSING: %s (scripts/decisiongold.sh names its source)", testmodels.Path(c.gguf))
			}
			proj := towerOf(src)
			if proj == "" {
				t.Fatalf("%s has no projector beside it: the pictures would reach no tower", src)
			}
			m, err := model.Open(jlmOfPair(t, src, proj))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if !m.SeesPictures() {
				t.Fatal("the container states no tower")
			}
			arm := func(t *testing.T, tol float64, g *gpu.GPU, opts ...Option) {
				in, err := NewInstance(m, 4096, append([]Option{WithKVF16(false)}, opts...)...)
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
				state, qs := decisionRequest(t, "pictures")
				pics := decisionPictures(t, m, "pictures")
				ans, err := d.Decide(state, qs, pics...)
				if err != nil {
					t.Fatal(err)
				}
				worst := checkDecisions(t, qs, ans, readDecisionGold(t, c.gold), tol)
				t.Logf("%d pictures, %d prompt rows, worst difference %.5f", len(pics), d.InputTokens(), worst)
				// The violation: the same request with no pictures is not
				// the golden's answer.
				blind, err := d.Decide(state, qs)
				if err != nil {
					t.Fatal(err)
				}
				if differs := decisionsDiffer(qs, ans, blind); !differs {
					t.Fatal("VIOLATION arm: the answers without the pictures are the answers with them")
				}
			}
			t.Run("cpu", func(t *testing.T) { arm(t, max(c.tol, int8Tol[c.name]), nil) })
			// The CPU's float forms: no int8 activations, the band's floor.
			t.Run("cpu-float", func(t *testing.T) { arm(t, c.tol, nil, withCPUFloat()) })
			if c.cpuOnly != "" {
				t.Logf("no card arm: %s", c.cpuOnly)
				return
			}
			for _, spec := range stepDevices() {
				t.Run(spec, func(t *testing.T) {
					g, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
					if err != nil || g == nil {
						noDevice(t, spec, err)
					}
					defer g.Close()
					arm(t, c.tol, g)
				})
			}
		})
	}
}

// decisionsDiffer reports whether two runs' answers to qs differ in any
// probability by more than 1e-4.
func decisionsDiffer(qs []DecisionQuestion, a, b []DecisionAnswer) bool {
	for i := range qs {
		if d := a[i].Noul - b[i].Noul; d > 1e-4 || d < -1e-4 {
			return true
		}
		for j := range a[i].Probs {
			if d := a[i].Probs[j] - b[i].Probs[j]; d > 1e-4 || d < -1e-4 {
				return true
			}
		}
	}
	return false
}
