package engine

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// TestASteppedDownCacheIsAnF32Cache holds an instance whose empty binary16
// cache stepped down to f32 (setKVFmt: an attach, and Pool.Serve taking the
// pool's first instance's format) to one opened at f32: the same forms over
// the same history, so the same logits. A retired row restarts on the cache.
func TestASteppedDownCacheIsAnF32Cache(t *testing.T) {
	for _, name := range []string{"stories15M-q4_0.gguf", "synth-phimoe.gguf"} {
		t.Run(name, func(t *testing.T) {
			p, ok := existingModel(testmodels.Path(name))
			if !ok {
				testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path(name)+" (set JITLLM_MODELS to the model directory)")
			}
			m, err := model.Open(jlmOf(t, p))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			const rows, steps = 3, 12
			want := newSessBatch(m, rows, 64, WithKVPrewarm(false), WithKVType(model.KVF32))
			defer want.Close()
			got := newSessBatch(m, rows, 64, WithKVPrewarm(false))
			defer got.Close()
			if got.kvFmt != nn.KVF16 {
				t.Fatalf("the default cache is %v here: the step down is not exercised", got.kvFmt)
			}
			got.stepDownKVToF32()
			if got.kvFmt != want.kvFmt {
				t.Fatalf("stepped down to %v, opened at %v", got.kvFmt, want.kvFmt)
			}
			toks := m.Vocab.Encode("Once upon a time, in a small village by the sea, there lived an old fisherman", true)
			worst := 0.0
			for step := 0; step < steps; step++ {
				if step == 4 {
					want.Retire(1)
					got.Retire(1)
				}
				in := make([]int32, rows)
				for r := range in {
					in[r] = toks[(want.SeqPos(r)+r)%len(toks)]
				}
				a, err := want.ForwardBatch(in)
				if err != nil {
					t.Fatalf("step %d, opened at f32: %v", step, err)
				}
				a = append([]float32(nil), a...)
				b, err := got.ForwardBatch(in)
				if err != nil {
					t.Fatalf("step %d, stepped down: %v", step, err)
				}
				for r := range rows {
					nv := m.Cfg.NVocab
					e, _ := nmseOf(b[r*nv:(r+1)*nv], a[r*nv:(r+1)*nv])
					if e != e {
						t.Fatalf("step %d row %d: a non-finite NMSE", step, r)
					}
					if e > worst {
						worst = e
					}
					if e > 1e-10 {
						t.Errorf("step %d row %d: logit NMSE %.3e between a stepped-down cache and an f32 one", step, r, e)
					}
				}
			}
			t.Logf("worst logit NMSE %.3e over %d steps of %d rows", worst, steps, rows)
		})
	}
}
