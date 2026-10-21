//go:build amd64 || arm64

package model

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestMoEFFNRunsBatched asserts which dispatch shape a mixture's decode took,
// by counter: the per-expert loop computes the same answer, so moeFFN quietly
// declining (type or shape mismatch, unpacked bank) is invisible to
// correctness gates. It is two-sided: the F32 mixture has no packed bank and
// must take the loop.
func TestMoEFFNRunsBatched(t *testing.T) {
	const steps = 4
	for _, tc := range []struct {
		name  string
		batch bool // the packed bank batches; the F32 one cannot
	}{
		{"olmoe-1b-7b-0924-instruct-Q4_K_M.gguf", true},
		{"Qwen3-MOE-4x0.6B-Q4_K_M.gguf", true},
		{"tiny-qwen3moe-f32.gguf", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, err := Open(jlmOf(t, testmodels.Path(tc.name)))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if !m.Cfg.MoE() {
				t.Fatalf("%s is not a mixture -- this gate proved nothing", tc.name)
			}
			s := m.NewState(64)
			defer s.Close()
			for i := 0; i < steps; i++ {
				if _, err := s.Forward(int32(1 + i%16)); err != nil {
					t.Fatal(err)
				}
			}
			batched, looped := s.MoEDispatch()
			if batched+looped == 0 {
				t.Fatal("no mixture FFN ran at all -- this gate proved nothing")
			}
			// The expectation comes from the JIT's capability
			// (MixtureBatches), not from the architecture: where batching is
			// off (arm64 by default) the loop is correct.
			gt := m.layers[0].experts[0].gate.typ
			canBatch := tc.batch && s.jit.MixtureBatches(gt)
			// Every layer of every step runs exactly one mixture FFN, so the
			// total is not merely nonzero, it is known.
			if want := int64(m.Cfg.NLayer * steps); batched+looped != want {
				t.Errorf("%d mixture FFNs over %d layers x %d steps, want %d",
					batched+looped, m.Cfg.NLayer, steps, want)
			}
			if canBatch && looped != 0 {
				t.Errorf("packed bank, %s has a fused kernel: %d FFNs took the "+
					"per-expert loop (batched %d) -- the batch was not selected",
					gt, looped, batched)
			}
			if !canBatch && batched != 0 {
				t.Errorf("no fused %s kernel on this build, yet %d FFNs reported "+
					"batched -- the counter is not distinguishing", gt, batched)
			}
			// The down projection is counted separately: it batches under a
			// different contract (one activation per expert) and declines on
			// its own.
			dbatched, dlooped := s.MoEDownDispatch()
			if dbatched+dlooped != batched+looped {
				t.Errorf("down dispatch total %d != FFN total %d",
					dbatched+dlooped, batched+looped)
			}
			if canBatch && dlooped != 0 {
				t.Errorf("packed bank: %d down projections took the per-expert "+
					"loop (batched %d)", dlooped, dbatched)
			}
			if !canBatch && dbatched != 0 {
				t.Errorf("no fused %s kernel: %d down projections reported batched",
					gt, dbatched)
			}
			t.Logf("%s: %s, fused kernel %v -> gate/up batched %d looped %d, "+
				"down batched %d looped %d (k=%d of %d experts)", tc.name, gt,
				cpu.PackedFusedSupported(gt), batched, looped, dbatched, dlooped,
				m.Cfg.NExpertUsed, m.Cfg.NExpert)
		})
	}
}
