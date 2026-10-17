package model

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// The two families whose mixture FFN is lowered through the block graph
// (Qwen3-MoE and OLMoE) join the five principles' gates by name: the
// relocation, batch-fault, page-budget and allocation gates run every
// principle mixture, and TestGraphMixturesPageWithTheSameAnswer is their
// paging gate (a mixture's expert pages are not TestPagingDoesNotChangeTheAnswer's
// frames). Both are the real checkpoints: the random transformers OLMoE
// (tiny-random-OlmoeForCausalLM) has a 16-wide expert, which every device
// declines by name today ("Quantize: k=16 is not a multiple of 32"), so on
// the relocation gates it would place nothing. It is the paging gate's,
// where the 7B release's one-page arm would re-read 4 GiB per pass.
var graphMixtures = []string{"Qwen3-MOE-4x0.6B-Q4_K_M.gguf", "olmoe-1b-7b-0924-instruct-Q4_K_M.gguf"}

func init() { principleMixtures = append(principleMixtures, graphMixtures...) }

func TestGraphMixturesPageWithTheSameAnswer(t *testing.T) {
	for _, name := range []string{"Qwen3-MOE-4x0.6B-Q4_K_M.gguf", "tiny-random-OlmoeForCausalLM.jlm"} {
		t.Run(name, func(t *testing.T) {
			src := jlmOf(t, testmodels.Path(name))
			m, err := Open(src, noTune)
			if err != nil {
				t.Fatal(err)
			}
			if m.Cfg.NExpert == 0 {
				t.Fatalf("%s carries no mixture: this gate would prove nothing", name)
			}
			ids := m.Vocab.Encode("The capital of France is", true)
			m.Close()
			pagesWithTheSameAnswer(t, ids, src)
		})
	}
}
