package model

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestWideQKNormOnDevice is the end-to-end gate on the device's whole-row q/k
// norm (bs.qkRms): olmoe normalises all of q and all of k, and the tier runs
// that as one RMSNorm launch per vector instead of a part/apply pair. The arms
// differ only in that norm's reduction order; a norm with the wrong weight or
// buffer lands far outside the band. Stats.QKRms is the selection check.
//
// The model is olmoe itself (JITLLM_QKRMS_MODEL, default the Q4_K_M GGUF in
// JITLLM_MODELS): the tiny HF olmoe fixture has an expert width of 16, which
// the device declines, and no other fixture carries a wide q/k norm. Whatever
// the card holds is placed; the gate needs the norm to run, not every block.
func TestWideQKNormOnDevice(t *testing.T) {
	path := testmodels.Resolve(os.Getenv("JITLLM_QKRMS_MODEL"))
	if path == "" {
		// The file is spelt both ways depending on where it was downloaded
		// from (the servers carry the lower-case name).
		for _, name := range []string{"olmoe-1b-7b-0924-instruct-q4_k_m.gguf", "olmoe-1b-7b-0924-instruct-Q4_K_M.gguf"} {
			if path = testmodels.Path(name); fileExists(path) {
				break
			}
		}
	}
	if _, ok := existingModel(path); !ok {
		testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS or JITLLM_QKRMS_MODEL) (RULE 11)", path)
	}
	m, err := Open(jlmOf(t, path), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !m.Cfg.QKNorm || !m.Cfg.QKNormWide {
		t.Fatal("the model has no wide q/k norm: this gate would prove nothing")
	}
	ids := m.Vocab.Encode("The capital of France is Paris, and the capital of", true)
	var arms [2][][]float32
	for i, rms := range []bool{false, true} {
		gpu, err := openTier(t, tier.WithDevices(testdevices.Like(t, "cuda:0")), tier.WithDeviceTune(tier.TuneOff),
			tier.WithConfig(func(c *tier.Config) { c.NoQKRms = !rms }))
		if err != nil || gpu == nil {
			t.Skipf("no cuda device (%v)", err)
		}
		st := m.NewState(len(ids) + 1)
		st.SetDeviceLayers(gpu, -1)
		placed := st.GPULayers()
		arms[i] = runIDs(t, st, ids)
		s := gpu.Stats()
		st.Close()
		gpu.Close()
		if placed == 0 {
			t.Fatalf("the device placed no block: %v", declineList(st))
		}
		if (s.QKRms > 0) != rms {
			t.Fatalf("selection: %d whole-row q/k norm launch(es) with qkRms=%v", s.QKRms, rms)
		}
		t.Logf("qkRms=%v: %d of %d blocks placed, %d whole-row q/k launch(es)", rms, placed, m.Cfg.NLayer, s.QKRms)
	}
	// Held to the host, relative to the other arm's own distance from it, not
	// to each other: a reduction-order change travels sixteen blocks of int8
	// activations and amplifies.
	host := teacherForce(t, m, ids)
	ref, refD := worstOf(host, arms[0])
	got, gotD := worstOf(host, arms[1])
	between, betweenD := worstOf(arms[0], arms[1])
	t.Logf("against the host: part/apply NMSE %.3e (|dlogit| %.3f), whole-row %.3e (%.3f); between them %.3e (%.3f)",
		ref, refD, got, gotD, between, betweenD)
	if !(got < 3*ref+1e-6) || !(got < 1e-2) {
		t.Fatalf("the whole-row q/k norm sits NMSE %.3e from the host where the part/apply pair sits %.3e",
			got, ref)
	}
}
