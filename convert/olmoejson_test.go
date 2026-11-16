package convert

import (
	"errors"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestOLMoEThroughTheLibrary: olmoe.json states norm_topk_prob, which decides
// the renormalisation over the graph's default, and does not read clip_qkv,
// which OLMoE applies after the q/k norm where the engine's clamp (DBRX's)
// comes before it, so a clamp is refused as a graph key the file leaves unread.
func TestOLMoEThroughTheLibrary(t *testing.T) {
	mj := libraryFile(t, "OlmoeForCausalLM")
	js := `{"architectures": ["OlmoeForCausalLM"], "vocab_size": 100, "hidden_size": 32, "intermediate_size": 16,
	  "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4, "num_experts": 8,
	  "num_experts_per_tok": 2, "norm_topk_prob": false, "clip_qkv": null, "hidden_act": "silu",
	  "tie_word_embeddings": false}`
	names := []string{"lm_head.weight"}
	c, err := mj.config(hfConfigOf(t, js), names)
	if err != nil {
		t.Fatal(err)
	}
	if c.Name != "olmoe" || !c.Flags.Has(jlm.FlagQKNormWide) || !c.Flags.Has(jlm.FlagNoExpertNorm) {
		t.Errorf("arch %v, flags %v", c.Name, c.Flags)
	}
	c, err = mj.config(hfConfigOf(t, strings.Replace(js, `"norm_topk_prob": false`, `"norm_topk_prob": true`, 1)), names)
	if err != nil || c.Flags.Has(jlm.FlagNoExpertNorm) {
		t.Errorf("norm_topk_prob true does not renormalise: %v", err)
	}
	if _, err := mj.config(hfConfigOf(t, strings.Replace(js, `"clip_qkv": null`, `"clip_qkv": 8`, 1)), names); !errors.Is(err, ErrNotImplemented) || !strings.Contains(err.Error(), "clip_qkv") {
		t.Errorf("clip_qkv 8: %v, want the clamp refused", err)
	}
}
