package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestDecisionOverVideoMatchesItsGolden asks a decision model the video
// request -- a clip's frames with no rate stated -- and holds its answers to
// the golden its own code gave: Clef's VL fixture to Clef's code on its
// release's processor, vllm-d3 to its runtime (d3_runtime.py; scripts/
// d3gold.py), which reads a list of frames as a clip at 2 a second
// (VIDEO_FPS) -- the rate its case states; on the CPU device (int8
// activations and float forms) and on each card with every block placed.
// The violation: the same request with no video is not the golden's answer.
func TestDecisionOverVideoMatchesItsGolden(t *testing.T) {
	for _, c := range []struct {
		name, gguf, gold string
		tol, fps         float64
	}{
		{"clef-fixture-vl", "clef/fixture-vl/clef-fixture-vl.gguf", "clef-fixture-vl.video.reference.json", 2e-3, 0},
		{"vllm-d3-runtime", "vllm-d3/vllm-d3-edge-Q8_0.gguf", "vllm-d3-edge.video.reference.json", 0.05, 2},
	} {
		t.Run(c.name, func(t *testing.T) { videoMatchesItsGolden(t, c.gguf, c.gold, c.tol, c.fps) })
	}
}

func videoMatchesItsGolden(t *testing.T, gguf, gold string, tol, fps float64) {
	src, ok := existingModel(testmodels.Path(gguf))
	if !ok {
		testmodels.Missing(t, "MODEL MISSING: %s (scripts/decisiongold.sh names its source)", testmodels.Path(gguf))
	}
	m, err := model.Open(jlmOfPair(t, src, towerOf(src)))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	video, err := m.VideoSpans(decisionVideo(t, "video"), fps)
	if err != nil {
		t.Fatal(err)
	}
	arm := func(t *testing.T, g *gpu.GPU, opts ...Option) {
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
		state, qs := decisionRequest(t, "video")
		ans, err := d.DecideMedia(state, qs, nil, [][]model.Span{video})
		if err != nil {
			t.Fatal(err)
		}
		worst := checkDecisions(t, qs, ans, readDecisionGold(t, gold), tol)
		t.Logf("%d prompt rows, worst difference %.5f", d.InputTokens(), worst)
		blind, err := d.Decide(state, qs)
		if err != nil {
			t.Fatal(err)
		}
		if !decisionsDiffer(qs, ans, blind) {
			t.Fatal("VIOLATION arm: the answers without the video are the answers with it")
		}
	}
	t.Run("cpu", func(t *testing.T) { arm(t, nil) })
	t.Run("cpu-float", func(t *testing.T) { arm(t, nil, withCPUFloat()) })
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
}
