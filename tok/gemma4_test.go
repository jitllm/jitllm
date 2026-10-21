package tok

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestGemma4EncodesAsItsTokenizerDoes holds the SentencePiece-style BPE
// (jlm.VocabBPESPM) to HuggingFace tokenizers running Gemma 4's own
// tokenizer.json (scripts/gemma4tokgold.py) on a corpus of escaped spaces,
// byte fallback, newlines beside other characters and long prose and code. The
// vocabulary is the one llama.cpp's converter wrote into the synth-gemma4
// fixture, read through convert.VocabOf, so it is what a container carries.
func TestGemma4EncodesAsItsTokenizerDoes(t *testing.T) {
	path := testmodels.Path("synth-gemma4.gguf")
	f, err := gguf.Open(path)
	if err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v -- run scripts/gemma4gold.py synth-gemma4 (RULE 11)", err)
	}
	defer f.Close()
	vc := vocabOf(t, f)
	if vc.Kind != jlm.VocabBPESPM || !vc.ByteFallback || vc.AddSpacePrefix || vc.Merges.Len() == 0 {
		t.Fatalf("the vocabulary is %v (byte fallback %v, space prefix %v, %d merges), want bpe-spm "+
			"with byte fallback and no space prefix", vc.Kind, vc.ByteFallback, vc.AddSpacePrefix, vc.Merges.Len())
	}
	v, err := New(vc)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join("testdata", "gemma4_encode.json"))
	if err != nil {
		t.Fatalf("golden: %v (RULE 11: regenerate it with scripts/gemma4tokgold.py, do not skip)", err)
	}
	var golden []struct {
		Text string  `json:"text"`
		IDs  []int32 `json:"ids"`
	}
	if err := json.Unmarshal(b, &golden); err != nil {
		t.Fatal(err)
	}
	if len(golden) < 20 {
		t.Fatalf("golden has %d case(s) -- too few to prove anything", len(golden))
	}
	ids, fallback := 0, 0
	for _, c := range golden {
		got := v.Encode(c.Text, false)
		if !slices.Equal(got, c.IDs) {
			i := 0
			for i < len(got) && i < len(c.IDs) && got[i] == c.IDs[i] {
				i++
			}
			t.Errorf("%.60q: %d ids against tokenizers' %d, first difference at %d", c.Text, len(got), len(c.IDs), i)
			continue
		}
		ids += len(got)
		for _, id := range got {
			if strings.HasPrefix(v.Text(id), "<0x") {
				fallback++
			}
		}
		// Decode is the inverse wherever no byte was lost: not across a
		// carriage return or a "▁" the text already carried.
		if d := v.Decode(got); d != c.Text && !strings.ContainsAny(c.Text, "\r▁") {
			t.Errorf("Decode(Encode(%.60q)) = %.60q", c.Text, d)
		}
	}
	// The corpus must reach the byte fallback, or the gate never ran it.
	if fallback == 0 {
		t.Errorf("no byte-fallback token in %d ids: the corpus does not reach it", ids)
	}
	t.Logf("%d case(s), %d ids, %d byte-fallback tokens, against tokenizers", len(golden), ids, fallback)
}
