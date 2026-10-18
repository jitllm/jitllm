package model

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestStepFusesAsDecodeDoes: a step of a few sequences runs decode's fusions
// over every row -- q, k and v in one launch, the attention output and the
// FFN down projection adding the residual themselves, the up projection
// writing act(gate)*up -- and its logits are those of the same step with each
// one a separate launch (tier.Config.NoRagFuse), within the bound every step
// gate holds a device to (mlaDeviceNMSE). Not bit for bit: k and v run at q's
// split in the one launch, as decode's do, and at their own apart, and a
// change of summation order alone moves this model's logits by NMSE ~2e-03
// (pagedBand). The residual and gate epilogues alone are bit exact
// (backend.TestGroupEpiloguesAreTheSeparateLaunches); a token reading another
// token's residual or gate is NMSE of order one. The fused arm's counters are
// the selection check and the other arm's zeros say the knob reached the
// tier. Steps of two, three and four sessions, since the kernels carry every
// token in one thread and a per-token offset is wrong only past the first.
func TestStepFusesAsDecodeDoes(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := Open(jlmOf(t, p), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompts := []string{"The capital of France is", "Water boils at a temperature of",
		"The clerk counted barrels of salt on the upper floor, and", "Once upon a time"}
	const steps = 3
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			arm := func(noFuse bool) ([][]float32, tier.Stats) {
				g, err := openTier(t, tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff),
					tier.WithConfig(func(c *tier.Config) { c.NoRagFuse = noFuse }))
				if err != nil || g == nil {
					noDevice(t, spec, err)
				}
				defer g.Close()
				var out [][]float32
				for n := 2; n <= len(prompts); n++ {
					var sts []*State
					var toks []int32
					var fed [][]int32
					for _, pr := range prompts[:n] {
						ids := m.Vocab.Encode(pr, true)
						st := m.NewState(len(ids) + steps + 2)
						defer st.Close()
						if err := st.SetDevice(g); err != nil {
							t.Fatal(err)
						}
						if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
							t.Skipf("CARD TOO SMALL: %d of %d blocks -- this gate proved nothing here (%s)",
								st.GPULayers(), m.Cfg.NLayer, g.Err())
						}
						lg, err := st.Prefill(ids)
						if err != nil {
							t.Fatal(err)
						}
						sts, toks, fed = append(sts, st), append(toks, Greedy(lg)), append(fed, ids)
					}
					// The tokens fed are the prompt's own again, not each arm's
					// greedy choice: a near tie would otherwise feed the two arms
					// different tokens and compare different rows.
					for s := range steps {
						lgs, err := Step(sts, toks)
						if err != nil {
							t.Fatal(err)
						}
						for i, lg := range lgs {
							out = append(out, append([]float32(nil), lg...))
							toks[i] = fed[i][s%len(fed[i])]
						}
					}
				}
				return out, g.Stats()
			}
			fused, fs := arm(false)
			apart, as := arm(true)
			if fs.RagQKVFused == 0 || fs.RagResFused == 0 || fs.RagGateFused == 0 {
				t.Fatalf("the fused arm ran %d q/k/v, %d residual and %d gate fusions: one never ran (%s)",
					fs.RagQKVFused, fs.RagResFused, fs.RagGateFused, fs.LastErr)
			}
			if as.RagQKVFused+as.RagResFused+as.RagGateFused != 0 {
				t.Fatalf("NoRagFuse still ran %d q/k/v, %d residual and %d gate fusions",
					as.RagQKVFused, as.RagResFused, as.RagGateFused)
			}
			if len(fused) != len(apart) {
				t.Fatalf("%d rows fused against %d apart", len(fused), len(apart))
			}
			worst, at := 0.0, 0
			for r := range fused {
				e := logitNMSE(fused[r], apart[r])
				if !(e < mlaDeviceNMSE) {
					t.Fatalf("row %d: fused against apart NMSE %.3e, bound %.0e", r, e, mlaDeviceNMSE)
				}
				if e > worst {
					worst, at = e, r
				}
			}
			t.Logf("%d rows, worst NMSE %.3e (row %d) against apart; %d q/k/v, %d residual, %d gate fusions",
				len(fused), worst, at, fs.RagQKVFused, fs.RagResFused, fs.RagGateFused)
		})
	}
}
