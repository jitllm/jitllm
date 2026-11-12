//go:build linux

package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestStreamedVerbatimBankMatchesTheResidentOne is the streamed-bank gate
// (TestStreamedExpertBankMatchesTheResidentOne) on a mixture whose routed
// banks the container carries verbatim: synth-qwen2moe quantized to IQ4_NL by
// llama.cpp's llama-quantize. Its sheets are the bank's own blocks, an
// expert's rows a run of them, as a float bank's are its rows; gathered and
// sent direct, each streamed arm is the resident bank's answer bit for bit,
// under a page budget that evicts so the routed sheets are read.
func TestStreamedVerbatimBankMatchesTheResidentOne(t *testing.T) {
	path := jlmOf(t, testmodels.Path("synth-qwen2moe-iq4nl.gguf"))
	probe, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	page := probe.Container.H.PageSize
	probe.Close()
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	budget := 2 * page
	if !budgetEvicts(m, budget) {
		t.Fatalf("the budget is %d bytes over pages of %d and nothing is evicted", budget, page)
	}
	if m.Cfg.NExpertUsed >= m.Cfg.NExpert || m.Cfg.NExpertUsed < 1 {
		t.Fatalf("the fixture routes %d of %d experts; streaming needs used < total", m.Cfg.NExpertUsed, m.Cfg.NExpert)
	}
	ids := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	run := func(stream func(*gpu.Config)) ([][]float32, gpu.Stats, int, uint64) {
		t.Helper()
		opts := []gpu.Option{gpu.WithDevices("gpu:0"), gpu.WithDeviceTune(gpu.TuneOff)}
		if stream != nil {
			opts = append(opts, gpu.WithConfig(func(c *gpu.Config) { c.StreamExperts = true; stream(c) }))
		}
		g, err := openGPU(t, opts...)
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		defer g.Close()
		s := testInstance(t, m, 16, WithPageBudget(budget))
		defer s.Close()
		s.SetDeviceLayers(g, m.Cfg.NLayer)
		placed := s.GPULayers()
		if placed != m.Cfg.NLayer {
			t.Fatalf("stream=%v: the device took %d of %d blocks: %s", stream != nil, placed, m.Cfg.NLayer, g.Err())
		}
		out := make([][]float32, len(ids))
		for i, id := range ids {
			l, err := s.Forward(id)
			if err != nil {
				t.Fatalf("stream=%v: %v", stream != nil, err)
			}
			out[i] = append([]float32(nil), l...)
		}
		return out, g.Stats(), placed, g.Bytes()
	}
	want, rs, rp, rb := run(nil)
	for _, arm := range []struct {
		name   string
		cfg    func(*gpu.Config)
		direct bool
	}{
		{"gathered", func(c *gpu.Config) { c.StreamGroups = 1; c.StreamCacheSlots = -1 }, false},
		{"direct", func(c *gpu.Config) { c.StreamDirectBytes = 1; c.StreamGroups = 1; c.StreamCacheSlots = -1 }, true},
	} {
		t.Run(arm.name, func(t *testing.T) {
			streamedMatches(t, run, arm.cfg, arm.direct, false, ids, want, rs, rp, rb)
		})
	}
}
