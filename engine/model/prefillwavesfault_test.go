//go:build jitllmfault && (amd64 || arm64)

package model

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestPrefillWavesGateDiscriminates runs TestHybridPrefillWavesMatchTheHost's
// multi-wave arm with each of the waves' two violations armed: the chunk's
// logits must part from the host's past the gate's band.
func TestPrefillWavesGateDiscriminates(t *testing.T) {
	for _, md := range prefillWavesModels {
		if md.name == "kimik3" {
			// Its last row barely reads the later waves' experts (dropping them
			// moves the logits 4e-5): the violation is held on the other model.
			continue
		}
		path, ok := existingModel(testmodels.Path(md.path))
		if !ok {
			t.Skipf("MODEL MISSING: %s -- this gate proved nothing", md.path)
		}
		m, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		ids := prefillWavesIDs(m)
		want := prefillWavesHost(t, m, ids)
		for _, fault := range []string{"prefillwave", "prefillunnamed"} {
			t.Run(md.name+"/"+fault, func(t *testing.T) {
				got, st := prefillWavesCase(t, m, ids, md.blocks, 2*m.Cfg.NExpertUsed, false, tier.WithFault("moe", fault))
				nmse := wavesNMSE(got, want)
				t.Logf("%s: NMSE %.2e (band %.0e), %d waves", fault, nmse, md.band, st.PrefillWaves)
				if st.PrefillWaves <= st.PrefillBlocks || nmse <= md.band {
					t.Fatalf("with %s armed the chunk reads NMSE %.2e over %d waves: the gate cannot see it", fault, nmse, st.PrefillWaves)
				}
			})
		}
	}
}

// TestPrefillWavesAllocGateDiscriminates arms an allocation in the waves'
// chunk loop: the warm count TestHybridPrefillWavesMatchTheHost holds at zero
// must see it.
func TestPrefillWavesAllocGateDiscriminates(t *testing.T) {
	md := prefillWavesModels[1]
	path, ok := existingModel(testmodels.Path(md.path))
	if !ok {
		t.Skipf("MODEL MISSING: %s -- this gate proved nothing", md.path)
	}
	m, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := prefillWavesIDs(m)
	_, st, allocs := prefillWavesAllocs(t, m, ids, md.blocks, 0, false, tier.WithFault("moe", "prefillalloc"))
	t.Logf("prefillalloc: %d warm allocations over %d block-chunks", allocs, st.PrefillBlocks)
	if st.PrefillBlocks == 0 || allocs == 0 {
		t.Fatalf("with prefillalloc armed the warm chunk made %d allocations over %d block-chunks: the gate cannot see it", allocs, st.PrefillBlocks)
	}
}
