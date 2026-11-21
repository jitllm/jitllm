//go:build jitllmfault

package engine

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// TestAPromptOfSeveralStreamsRunsWithoutTheBatchedScratch prefills a model
// of several residual streams on a card where no batched scratch wider than
// 64 rows fits (Faults.SetRoomFault("narrow")) and no reserved width serves
// the prompt: the chunk takes the fallbacks -- its rows one at a time, the
// head a call of its own on the last row -- each its rows of every stream
// and of the per-layer inputs, every block on the card, and the logits agree
// with the composed CPU's.
func TestAPromptOfSeveralStreamsRunsWithoutTheBatchedScratch(t *testing.T) {
	for _, name := range []string{"synth-gemma3n"} {
		t.Run(name, func(t *testing.T) {
			p, ok := existingModel(testmodels.Path(name + ".gguf"))
			if !ok {
				testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path(name+".gguf")+" (set JITLLM_MODELS to the model directory)")
			}
			m, err := model.Open(jlmOf(t, p))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if m.Cfg.Streams() < 2 {
				t.Fatalf("%s carries %d residual stream(s): the configuration under test is not selected", name, m.Cfg.Streams())
			}
			text := strings.Repeat("The river rose every spring until the bridges were islands, and the clerk counted barrels of salt. ", 4)
			all := m.Vocab.Encode(text, true)
			const prompt, steps = 40, 4
			if len(all) < prompt+steps {
				t.Fatalf("the text is %d tokens; the gate needs %d", len(all), prompt+steps)
			}
			ids, toks := all[:prompt], all[prompt:prompt+steps]
			opts := []Option{WithKVF16(false)}
			want := cpuDecode(t, m, ids, toks, opts...)
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					var f gpu.Faults
					f.SetRoomFault("narrow")
					got, st, _ := composedDecode(t, m, opts, dev, ids, toks, nn.ComposedBlock, false, false, gpu.WithFaults(&f),
						gpu.WithConfig(func(c *gpu.Config) { c.ComposedWidths, c.NoTail, c.NoExactPrompt = true, true, true }))
					if st.PromptSplits == 0 {
						t.Fatalf("the %d-row prompt ran whole: no fallback was taken (%s)", prompt, st.LastErr)
					}
					w, at := worstStep(got, want)
					t.Logf("%d streams, %d fallback split(s): worst NMSE %.3e (row %d) against the composed CPU",
						m.Cfg.Streams(), st.PromptSplits, w, at)
					if !(w < mlaDeviceNMSE) {
						t.Fatalf("the prompt's fallbacks on the card disagree with the CPU: NMSE %.3e at row %d", w, at)
					}
				})
			}
		})
	}
}
