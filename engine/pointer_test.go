package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestPointerDecisionMatchesLlamaCpp holds a pointer readout
// (jlm.Flag2MarkerPointer) to llama.cpp's answers on the same GGUF: Kev's
// prompt from the container's program, its scores the last row's query
// against each option's <|box_end|> row, on the CPU and with every block on
// each card.
func TestPointerDecisionMatchesLlamaCpp(t *testing.T) {
	c := decisionCase{"kev", "kev/Kev-4B-BF16.gguf", "Kev-4B-BF16", jlm.DecisionKev, 0.05}
	src, ok := existingModel(testmodels.Path(c.gguf))
	if !ok {
		testmodels.Missing(t, "MODEL MISSING: %s (scripts/decisiongold.sh names its source)", testmodels.Path(c.gguf))
	}
	m, err := model.Open(jlmOf(t, src))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if cfg := m.Container.Config(); cfg.Decision != c.kind || cfg.MarkerPrompt == nil || !m.Cfg.MarkPointer {
		t.Fatalf("the container says readout %v, prompt %v, pointer %v", cfg.Decision, cfg.MarkerPrompt != nil, m.Cfg.MarkPointer)
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
			ans, err := d.Decide(state, qs)
			if err != nil {
				t.Fatalf("%s: %v", req, err)
			}
			worst := checkDecisions(t, qs, ans, readDecisionGold(t, c.gold+"."+req+".llamacpp.json"), c.tol)
			t.Logf("%s: %d prompt tokens, worst difference from llama.cpp %.5f", req, d.InputTokens(), worst)
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
}
