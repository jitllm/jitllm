package jinja_test

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/tok/jinja"
)

// pinned holds exact renderings of the chat templates in real GGUFs. A stray
// newline is a different prompt, so the pins are the whole output, not a
// "contains" check.
func pinned() map[string]string {
	return map[string]string{
	"tinyllama-1.1b-q3_K_M": "<|user|>\nhello</s>\n<|assistant|>\n",
	"Qwen2-1.5B-Instruct-Q4_K_M": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
		"<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
	"tiny-qwen3moe-f32":            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
	"Phi-3.5-mini-instruct-Q4_K_M": "<|user|>\nhello<|end|>\n<|assistant|>\n",
	"Mixtral-8x7B-Instruct-Q3_K_M": "<s>[INST] hello [/INST]",
	"SmolLM2-360M-Instruct-Q8_0": "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, " +
		"trained by Hugging Face<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
}
}

// TestGGUFChatTemplatesRender compiles and renders every chat template in the model directory.
func TestGGUFChatTemplatesRender(t *testing.T) {
	paths := testmodels.Glob("*.gguf")
	if len(paths) == 0 {
		testmodels.Missing(t, "%s", "no models in "+testmodels.Dir()+" (set JITLLM_MODELS to the model directory) -- RULE 11: a missing artefact is a task, not a constraint")
	}
	// No size cap: a metadata read never touches the weights, so even a large
	// file opens at once, and a cap would drop pinned shapes.
	// JITLLM_TPL_MODEL renders exactly one file when bisecting.
	only := os.Getenv("JITLLM_TPL_MODEL")

	rendered, hit := 0, 0
	for _, p := range paths {
		name := strings.TrimSuffix(filepath.Base(p), ".gguf")
		if only != "" && !strings.Contains(name, only) {
			continue
		}
		// Part 2 onward of a split model carries tensors and no metadata, and
		// gguf.Open refuses it by name; part 1 renders for the whole model.
		if m := splitPart().FindStringSubmatch(filepath.Base(p)); m != nil && m[2] != "00001" {
			continue
		}
		f, err := gguf.Open(p)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		v, ok := f.KV["tokenizer.chat_template"]
		src, _ := v.String()
		bos, eos := specialText(f)
		f.Close()
		if !ok {
			continue
		}

		tpl, err := jinja.Compile(src)
		if err != nil {
			t.Errorf("%s: compile (%d bytes): %v", name, len(src), err)
			continue
		}
		out, err := tpl.Render(map[string]any{
			"messages":              []any{map[string]any{"role": "user", "content": "hello"}},
			"add_generation_prompt": true,
			"bos_token":             bos,
			"eos_token":             eos,
		})
		if err != nil {
			t.Errorf("%s: render: %v", name, err)
			continue
		}
		rendered++
		if out == "" {
			t.Errorf("%s: rendered empty -- a template that produces nothing is not a pass", name)
		}
		if want, ok := pinned()[name]; ok {
			hit++
			if out != want {
				t.Errorf("%s: rendering moved\n got %q\nwant %q", name, out, want)
			}
		}
	}

	// A green line that rendered nothing is the failure this guards.
	if rendered < 6 {
		testmodels.Missing(t, "only %d templates rendered; this gate is vacuous below 6", rendered)
	}
	if hit < len(pinned()) && only == "" {
		t.Fatalf("only %d of %d pinned models were reached -- the pins are not being checked", hit, len(pinned()))
	}
	t.Logf("rendered %d templates, %d of them pinned exactly", rendered, hit)
}

// specialText resolves the BOS/EOS token strings the template interpolates.
func specialText(f *meta.File) (bos, eos string) {
	b, _ := f.KV["tokenizer.ggml.bos_token_id"].Int()
	e, _ := f.KV["tokenizer.ggml.eos_token_id"].Int()
	toks, ok := f.KV["tokenizer.ggml.tokens"]
	if !ok {
		return "", ""
	}
	toks.EachString(func(i int, s []byte) bool {
		if int64(i) == b {
			bos = string(s)
		}
		if int64(i) == e {
			eos = string(s)
		}
		return true
	})
	return bos, eos
}

// splitPart is llama.cpp's gguf-split naming, NAME-00001-of-00003.gguf.
func splitPart() *regexp.Regexp {
	return regexp.MustCompile(`^(.*)-(\d{5})-of-(\d{5})\.gguf$`)
}
