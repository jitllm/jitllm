package convert

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/pretok"
)

// TestAStageTheTokenizerCannotBuildIsRefusedAtConversion holds the stage
// translation to the tokenizer's own check: a pattern that does not compile,
// and one that compiles past the engine's bound, are refused while the file
// is in front of the person converting it, and a pattern that compiles is
// stored as written.
func TestAStageTheTokenizerCannotBuildIsRefusedAtConversion(t *testing.T) {
	for _, bad := range []string{`(a`, `(?:(?:ab){1000}){100}`} {
		_, err := containerPreOps([]pretok.Op{{Kind: pretok.OpPattern, Pattern: bad, Behavior: "Isolated"}})
		if err == nil || !strings.Contains(err.Error(), bad) {
			t.Errorf("%q: %v; want a refusal naming the pattern", bad, err)
		}
	}
	got, err := containerPreOps([]pretok.Op{{Kind: pretok.OpPattern, Pattern: `\p{N}{1,3}`, Behavior: "Isolated"}})
	if err != nil || len(got) != 1 || got[0].Kind != jlm.PrePattern || got[0].Pattern != `\p{N}{1,3}` {
		t.Errorf("a pattern that compiles: %+v, %v", got, err)
	}
}

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
