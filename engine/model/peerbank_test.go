//go:build amd64 || arm64

package model

import (
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// peerTokens is how many decode tokens follow the prompt in the expert-card
// gate.
const peerTokens = 6

// peerRun prefills ids and decodes peerTokens greedy tokens on a tier over
// specs, whose experts run hybrid through a cache too small for a chunk (so
// every wave and every token misses). card names the expert card ("" for
// none). It returns every step's logits and the device's counters.
func peerRun(t *testing.T, m *Model, ids []int32, specs []string, card string, extra ...tier.Option) ([][]float32, tier.Stats) {
	t.Helper()
	c := m.Cfg
	g, err := openTier(t, append([]tier.Option{tier.WithDevices(strings.Join(specs, ",")), tier.WithDeviceTune(tier.TuneOff),
		tier.WithConfig(func(cf *tier.Config) {
			cf.StreamExperts, cf.HybridExperts = true, true
			cf.SplitAdmit = c.NExpertUsed
			cf.StreamCacheSlots = 2 * c.NExpertUsed
			cf.ExpertCard = card
		})}, extra...)...)
	if err != nil || g == nil {
		noDevice(t, strings.Join(specs, ","), err)
	}
	defer g.Close()
	s := m.NewState(len(ids) + peerTokens + 2)
	defer s.Close()
	s.SetDeviceLayers(g, c.NLayer)
	if s.GPULayers() != c.NLayer {
		t.Skipf("CARD TOO SMALL: %d of %d blocks placed (%s) -- this arm proved nothing", s.GPULayers(), c.NLayer, g.Stats().LastErr)
	}
	defer m.enterPager()()
	lg, err := s.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	out := [][]float32{append([]float32(nil), lg...)}
	for range peerTokens {
		if lg, err = s.Forward(Greedy(lg)); err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	return out, g.Stats()
}

// TestExpertCardServesCacheMisses runs a hybrid mixture whose expert cache
// misses on every wave and token, once with the host serving the misses and
// once with a second card holding the experts, and demands the same logits
// from both (a sheet copied card to card is the host's bytes, however it
// travels), with the copies counted: blocks given a bank, sheets taken from
// it (RULE 10: the path ran).
func TestExpertCardServesCacheMisses(t *testing.T) {
	specs := testdevices.Take(t, 2, "cuda:0,cuda:1")
	path, ok := existingModel(testmodels.Path("kimik3/Kimi-K3-0.40B.Q8_0.gguf"))
	if !ok {
		t.Skipf("MODEL MISSING: kimik3/Kimi-K3-0.40B.Q8_0.gguf -- this gate proved nothing")
	}
	m, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := prefillWavesIDs(m)
	want, base := peerRun(t, m, ids, specs[:1], "")
	if base.PeerSheets != 0 || base.PeerBlocks != 0 {
		t.Fatalf("the control arm took %d sheets from %d expert banks", base.PeerSheets, base.PeerBlocks)
	}
	got, st := peerRun(t, m, ids, specs, specs[1])
	t.Logf("expert card %s: %d block(s) with a bank (%d declined), %d sheets, %d bytes taken from it, %d planes staged through the host",
		specs[1], st.PeerBlocks, st.PeerDeclined, st.PeerSheets, st.PeerBytes, st.PeerStaged)
	if st.PeerBlocks == 0 || st.PeerSheets == 0 {
		t.Fatalf("no sheet came from the expert card (%d banks, %d sheets): the path never ran", st.PeerBlocks, st.PeerSheets)
	}
	for i := range want {
		nm, _ := nmse32(want[i], got[i])
		if math.IsNaN(nm) || nm > 1e-8 || Greedy(got[i]) != Greedy(want[i]) {
			t.Errorf("step %d: NMSE %.3e against the host-served run, argmax %d against %d", i, nm, Greedy(got[i]), Greedy(want[i]))
		}
	}
}
