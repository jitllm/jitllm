package engine

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestAStateThatLedAHostStepDecodesAlone: the instance leading a host step across
// sessions lends the model its host scratch for the step and hands it back
// (borrowStep, returnStep), and a decode token it runs alone afterwards -- the
// other row having ended, or still sealing its prompt -- finds the scratch it
// needs: the gated query of a hybrid is one buffer a single token reads. Both
// instances are checked against the same tokens run on instances that never stepped
// together.
func TestAStateThatLedAHostStepDecodesAlone(t *testing.T) {
	for _, c := range []struct{ name, file string }{
		{"hybrid", "qwen35/Qwen3.5-0.8B-Q4_K_M.gguf"},
		{"dense", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := testmodels.Path(c.file)
			if _, err := os.Stat(p); err != nil {
				t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
			}
			// The float forms: a step's and a row's apart by their folding
			// order alone (floatFormsNMSE).
			instOpts := []Option{WithKVF16(false), withCPUFloat(), sharedCPU(t)}
			m, err := model.Open(jlmOf(t, p))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			prompts := [][]int32{m.Vocab.Encode("The capital of France is", true), m.Vocab.Encode("Water boils at a temperature of", true)}
			next := func(lg []float32) int32 { return int32(argmax(lg)) }

			// The truth: each prompt run alone, then three tokens.
			var want [2][]float32
			for i, pr := range prompts {
				s := testInstance(t, m, 64, instOpts...)
				lg, err := s.Prefill(pr)
				if err != nil {
					t.Fatal(err)
				}
				for k := 0; k < 3; k++ {
					if lg, err = s.Forward(next(lg)); err != nil {
						t.Fatal(err)
					}
				}
				want[i] = append([]float32(nil), lg...)
				s.Close()
			}

			a, b := testInstance(t, m, 64, instOpts...), testInstance(t, m, 64, instOpts...)
			defer a.Close()
			defer b.Close()
			if err := a.HostRefusal(); err != nil {
				t.Skipf("the host step refuses this model: %v", err)
			}
			lga, err := a.Prefill(prompts[0])
			if err != nil {
				t.Fatal(err)
			}
			lgb, err := b.Prefill(prompts[1])
			if err != nil {
				t.Fatal(err)
			}
			ta, tb := []int32{next(lga)}, []int32{next(lgb)}
			joint := []Run{{Instance: a, Tokens: ta, Logits: true}, {Instance: b, Tokens: tb, Logits: true}}
			r0 := hostStepRows(joint)
			// a leads one joint step, then b's row is done and a decodes alone.
			out, err := StepRuns(joint)
			if err != nil {
				t.Fatal(err)
			}
			if n := hostStepRows(joint) - r0; n != 2 {
				t.Fatalf("the step ran %d rows across sessions, want 2: it was not a host step", n)
			}
			lga, lgb = append([]float32(nil), out[0]...), append([]float32(nil), out[1]...)
			for k := 0; k < 2; k++ {
				if lga, err = a.Forward(next(lga)); err != nil {
					t.Fatal(err)
				}
			}
			for k := 0; k < 2; k++ {
				if lgb, err = b.Forward(next(lgb)); err != nil {
					t.Fatal(err)
				}
			}
			for i, got := range [][]float32{lga, lgb} {
				if w, at := worstStep([][]float32{got}, [][]float32{want[i]}); !(w <= floatFormsNMSE) {
					t.Fatalf("state %d: a token run alone after leading a host step differs from one that never stepped jointly: NMSE %.3e at %d (bound %.0e)", i, w, at, floatFormsNMSE)
				}
			}
		})
	}
}
