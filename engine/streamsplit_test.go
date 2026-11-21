package engine

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// TestAPromptOfSeveralStreamsSplitsOnTheCard prefills a model of several
// residual streams (AltUp's, DeepSeek V4's hyper-connections) with a prompt
// longer than the card's reserved width, every block on the card: the chunk
// is cut into calls of that width as one stream's is, each call its rows of
// every stream (Stats.PromptSplits), and the card's logits, prompt and decode,
// agree with the composed CPU's.
func TestAPromptOfSeveralStreamsSplitsOnTheCard(t *testing.T) {
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
			text := strings.Repeat("The river rose every spring until the bridges were islands, and the clerk counted barrels of salt. ", 12)
			all := m.Vocab.Encode(text, true)
			const prompt, steps = 150, 6
			if len(all) < prompt+steps {
				t.Fatalf("the text is %d tokens; the gate needs %d", len(all), prompt+steps)
			}
			ids, toks := all[:prompt], all[prompt:prompt+steps]
			opts := []Option{WithKVF16(false)}
			want := cpuDecode(t, m, ids, toks, opts...)
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					got, st, _ := composedDecode(t, m, opts, dev, ids, toks, nn.ComposedBlock, false, false)
					if st.PromptSplits == 0 {
						t.Fatalf("a %d-row prompt over a reserved width of %d was not split (%s)", prompt, st.ReservedPrompt, st.LastErr)
					}
					w, at := worstStep(got, want)
					t.Logf("%d streams, %d split chunk(s): worst NMSE %.3e (row %d) against the composed CPU",
						m.Cfg.Streams(), st.PromptSplits, w, at)
					if !(w < mlaDeviceNMSE) {
						t.Fatalf("the split prompt on the card disagrees with the CPU: NMSE %.3e at row %d", w, at)
					}
				})
			}
		})
	}
}
