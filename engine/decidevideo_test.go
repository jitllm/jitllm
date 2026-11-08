package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestDecisionOverVideoMatchesItsGolden asks Clef's VL fixture the video
// request -- a clip's frames with no rate stated -- and holds its answers to
// the golden Clef's own code gave on its release's processor, on the CPU
// device (int8 activations and float forms) and on each card with every
// block placed. The violation: the same request with no video is not the
// golden's answer.
func TestDecisionOverVideoMatchesItsGolden(t *testing.T) {
	src, ok := existingModel(testmodels.Path("clef/fixture-vl/clef-fixture-vl.gguf"))
	if !ok {
		testmodels.Missing(t, "MODEL MISSING: %s (scripts/clefgold.py makes it)", testmodels.Path("clef/fixture-vl/clef-fixture-vl.gguf"))
	}
	m, err := model.Open(jlmOfPair(t, src, towerOf(src)))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	video, err := m.VideoSpans(decisionVideo(t, "video"), 0)
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
		worst := checkDecisions(t, qs, ans, readDecisionGold(t, "clef-fixture-vl.video.reference.json"), 2e-3)
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
