package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestPrefillFoldsALayerNormHead holds the folded head to a LayerNorm model
// with a quantized head (the float fixtures refuse the fold): the output
// projection rides a prompt's last chunk, and its norm must be the model's,
// the mean-centred one with its bias, launched with its six buffers. A device
// that refuses the folded chunk runs the prompt a row at a time and reads right
// with no fold at all, and one that does not count its buffers reads zeros, so
// the fold must move, no chunk may retry by rows, and the logits must equal
// the tail call's.
func TestPrefillFoldsALayerNormHead(t *testing.T) {
	_, golden := openC6(t, "synth-phi2")
	path := testmodels.Path("synth-phi2-q8.gguf")
	m, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := golden.IDs
	if len(ids) <= tailRowsMax {
		t.Fatalf("a prompt of %d rows is a ragged step, not a chunk with a fold", len(ids))
	}
	if !m.Cfg.LayerNorm || m.outNormB == nil {
		t.Fatalf("the fixture is not a LayerNorm model with a head norm bias: %+v", m.Cfg)
	}
	ran := 0
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			gpu, err := tier.OpenWith(tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
			if err != nil || gpu == nil {
				noDevice(t, spec, err)
			}
			defer gpu.Close()
			run := func(fold bool) ([]float32, int) {
				st := m.NewState(len(ids) + 4)
				defer st.Close()
				st.SetDeviceFallback(false)
				if err := st.SetDevice(gpu); err != nil {
					t.Fatal(err)
				}
				if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
					t.Fatalf("placed %d of %d blocks, head %v: %s", st.GPULayers(), m.Cfg.NLayer, st.HeadOnDevice(), gpu.Err())
				}
				st.noHeadFold = !fold
				before := gpu.Stats().HeadFolds
				lg, err := st.Prefill(ids)
				if err != nil {
					t.Fatal(err)
				}
				if st.devRowChunks != 0 {
					t.Fatalf("%d chunk(s) ran a row at a time after the device refused them: %s", st.devRowChunks, gpu.Err())
				}
				return append([]float32(nil), lg...), gpu.Stats().HeadFolds - before
			}
			want, tailFolds := run(false)
			got, folds := run(true)
			if tailFolds != 0 || folds == 0 {
				t.Fatalf("folds: %d on the tail-call arm, %d on the folded one -- the arms are not two arms", tailFolds, folds)
			}
			nmse := logitNMSE(got, want)
			if math.IsNaN(nmse) || !(nmse < 1e-10) {
				t.Fatalf("prefill logits NMSE %.3e between the folded head and the tail call", nmse)
			}
			t.Logf("%d fold(s); logits NMSE %.3e", folds, nmse)
			ran++
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}
