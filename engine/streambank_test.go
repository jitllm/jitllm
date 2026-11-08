//go:build linux

package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestAReleasedStreamedBlockGivesBackItsBank streams two mixtures of
// different shapes through one card, one after the other, as a server holding
// several models does. A model's blocks leave the card when the next model is
// admitted (GPU.ReleaseModel), and its shared compact banks must leave with
// them: the second model placed after the first holds exactly the banks it
// holds on a card that saw nothing before (Stats.SharedBankBytes). A bank kept
// until the card closes is the violation: each model's shapes took new ones,
// and the card census ran out of room for gpt-oss-120b behind them.
func TestAReleasedStreamedBlockGivesBackItsBank(t *testing.T) {
	q8 := jlmOf(t, testmodels.Path("synth-nemotronhmoe-q8.gguf"))
	open := func() *gpu.GPU {
		g, err := openGPU(t, gpu.WithDevices("gpu:0"), gpu.WithDeviceTune(gpu.TuneOff),
			gpu.WithConfig(func(c *gpu.Config) { c.StreamExperts = true }))
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		return g
	}
	// run streams m's blocks on g, decodes a token and reports the banks the
	// card held meanwhile, then closes the instance and the model.
	run := func(g *gpu.GPU, m *model.Model, what string, opts ...Option) uint64 {
		t.Helper()
		defer m.Close()
		s := testInstance(t, m, 16, opts...)
		defer s.Close()
		s.SetDeviceLayers(g, m.Cfg.NLayer)
		if s.GPULayers() != m.Cfg.NLayer {
			t.Fatalf("%s: %d of %d blocks placed: %s", what, s.GPULayers(), m.Cfg.NLayer, g.Err())
		}
		if _, err := s.Forward(1); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		st := g.Stats()
		if st.StreamBlocks == 0 || st.SharedBankBytes == 0 {
			t.Fatalf("%s: %d blocks streamed, %d B of shared banks: the arm streamed nothing",
				what, st.StreamBlocks, st.SharedBankBytes)
		}
		return st.SharedBankBytes
	}
	nemotron := func() (*model.Model, []Option) {
		m, err := model.Open(q8)
		if err != nil {
			t.Fatal(err)
		}
		return m, []Option{WithPageBudget(2 * m.PageSize())}
	}
	g := open()
	m, opts := nemotron()
	alone := run(g, m, "the nemotron mixture alone", opts...)
	g.Close()
	g = open()
	defer g.Close()
	wm, wopts := hybridModelOpt(t, hyOpt{moe: true, wideMoE: true})
	first := run(g, wm, "the wide mixture", wopts...)
	m, opts = nemotron()
	after := run(g, m, "the nemotron mixture after the wide one", opts...)
	t.Logf("shared banks: the wide mixture %d B; the nemotron mixture %d B alone, %d B after the wide one", first, alone, after)
	if after != alone {
		t.Errorf("the nemotron mixture placed after the wide one holds %d B of shared banks, %d B alone: "+
			"the wide mixture's banks stayed on the card", after, alone)
	}
}
