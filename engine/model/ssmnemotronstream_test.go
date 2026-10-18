//go:build linux

package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestUngatedStreamedBankMatchesTheResidentOne is
// TestStreamedExpertBankMatchesTheResidentOne on Nemotron 3's mixture, whose
// experts have no gate bank: the streamed arm uploads up and down sheets
// alone, and a fill that sent a gate it does not have, or skipped up, reads
// another expert's bytes. Bit equality, since the compact bank runs the same
// kernels over the same bytes; the budget evicts so the sheets are read. The
// fixture's weights at Q8_0 (scripts/ssmgold.py synth-nemotronhmoe-q8): a
// float bank has no packed layout to stream.
func TestUngatedStreamedBankMatchesTheResidentOne(t *testing.T) {
	path := jlmOf(t, testmodels.Path("synth-nemotronhmoe-q8.gguf"))
	ids := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	run := func(stream bool) ([][]float32, tier.Stats, int) {
		t.Helper()
		m, err := Open(path, noTune)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		m.SetPageBudget(2 * m.PageSize())
		if !m.container.CanEvict() {
			t.Fatalf("a budget of two pages evicts nothing: the sheets would never be read")
		}
		opts := []tier.Option{tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff)}
		if stream {
			opts = append(opts, tier.WithConfig(func(c *tier.Config) { c.StreamExperts = true }))
		}
		g, err := openTier(t, opts...)
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		defer g.Close()
		s := m.NewState(16)
		defer s.Close()
		s.SetDeviceLayers(g, m.Cfg.NLayer)
		placed := s.GPULayers()
		if placed != m.Cfg.NLayer {
			t.Fatalf("stream=%v: %d of %d blocks placed: %s", stream, placed, m.Cfg.NLayer, g.Err())
		}
		out := make([][]float32, len(ids))
		for i, id := range ids {
			l, err := s.Forward(id)
			if err != nil {
				t.Fatalf("stream=%v: %v", stream, err)
			}
			out[i] = append([]float32(nil), l...)
		}
		return out, g.Stats(), placed
	}
	want, rs, _ := run(false)
	got, ss, _ := run(true)
	// Three of the four blocks are mixtures.
	if ss.StreamBlocks != 3 || ss.StreamFills != 3*len(ids) || rs.StreamBlocks != 0 {
		t.Fatalf("streamed arm %d blocks and %d fills, resident arm %d streamed blocks: "+
			"the configuration under test never ran", ss.StreamBlocks, ss.StreamFills, rs.StreamBlocks)
	}
	for p := range ids {
		for i := range want[p] {
			w, gv := want[p][i], got[p][i]
			if math.IsNaN(float64(gv)) || math.IsInf(float64(gv), 0) || w != gv {
				t.Fatalf("pos %d logit %d: streamed %v, resident %v", p, i, gv, w)
			}
		}
	}
	t.Logf("%d streamed blocks, %d fills over %d tokens, logits bit-identical", ss.StreamBlocks, ss.StreamFills, len(ids))
}
