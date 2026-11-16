package convert

import (
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
)

// TestDefaultPreTokenizerIsTheLibrarys holds a BPE file naming no
// pre-tokenizer of its own to the one its architecture's library file
// states, with that name's booleans (pretokenizers.json): Llama 4's o200k
// split, its BOS and its ignore_merges. A file that names one keeps it, and
// an architecture the library states none for keeps llama.cpp's default,
// with neither boolean.
func TestDefaultPreTokenizerIsTheLibrarys(t *testing.T) {
	file := func(arch, pre string) *meta.File {
		kv := map[string]meta.Value{
			"general.architecture":  meta.MakeString(arch),
			"tokenizer.ggml.model":  meta.MakeString("gpt2"),
			"tokenizer.ggml.tokens": meta.MakeStrings([]string{"a", "b", "ab"}),
			"tokenizer.ggml.merges": meta.MakeStrings([]string{"a b"}),
		}
		if pre != "" {
			kv["tokenizer.ggml.pre"] = meta.MakeString(pre)
		}
		return meta.New(arch+".gguf", kv, nil)
	}
	for _, tc := range []struct {
		arch, pre, want string
		bos, ignore     bool
	}{
		{"llama4", "default", "llama4", true, true},
		{"llama4", "", "llama4", true, true},
		{"llama4", "llama-bpe", "llama-bpe", true, true},
		{"qwen2", "default", "default", false, false},
		{"llama", "falcon3", "falcon3", true, true},
	} {
		v, err := VocabOf(file(tc.arch, tc.pre))
		if err != nil {
			t.Fatalf("%s %q: %v", tc.arch, tc.pre, err)
		}
		if v.PreName != tc.want || v.AddBOS != tc.bos || v.IgnoreMerges != tc.ignore || len(v.Pre) == 0 {
			t.Errorf("%s %q: pre-tokenizer %q (%d stages), add_bos %v, ignore_merges %v; want %q, %v, %v",
				tc.arch, tc.pre, v.PreName, len(v.Pre), v.AddBOS, v.IgnoreMerges, tc.want, tc.bos, tc.ignore)
		}
	}
}
