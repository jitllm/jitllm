package convert

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/tok"
)

// TestTiktokenVocabMatchesLlamaCpp builds Kimi's vocabulary from tiktoken.model
// and holds it to what llama.cpp's converter wrote for the same file: Moonlight
// ships the identical tiktoken.model (sha256 b6c497a7...), and its GGUF carries
// llama.cpp's token table and reconstructed merges. Every base token and every
// merge must agree, in order.
func TestTiktokenVocabMatchesLlamaCpp(t *testing.T) {
	dir := testmodels.Path("Kimi-Linear-48B-A3B-Instruct")
	gg := testmodels.Path("Moonlight-16B-A3B-Instruct-Q4_K_M.gguf")
	if !fileExists(filepath.Join(dir, "tiktoken.model")) || !fileExists(gg) {
		t.Skipf("MODEL MISSING: %s/tiktoken.model or %s (set JITLLM_MODELS to the model directory) -- this gate proved nothing", dir, gg)
	}
	ours, err := hfTiktokenVocab(dirFiles(dir), 163840)
	if err != nil {
		t.Fatal(err)
	}
	f, err := gguf.Open(gg)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	ref, err := VocabOf(f)
	if err != nil {
		t.Fatal(err)
	}
	const base = 163584
	if ref.Tokens.Len() < base {
		t.Fatalf("llama.cpp's table has %d tokens", ref.Tokens.Len())
	}
	bad := 0
	for i := 0; i < base; i++ {
		if ours.Tokens.At(i) != ref.Tokens.At(i) {
			if bad++; bad <= 5 {
				t.Errorf("token %d: ours %q, llama.cpp %q", i, ours.Tokens.At(i), ref.Tokens.At(i))
			}
		}
	}
	if ours.Merges.Len() != ref.Merges.Len() {
		t.Errorf("%d merges, llama.cpp %d", ours.Merges.Len(), ref.Merges.Len())
	}
	for i := 0; i < min(ours.Merges.Len(), ref.Merges.Len()); i++ {
		if ours.Merges.At(i) != ref.Merges.At(i) {
			if bad++; bad <= 10 {
				t.Errorf("merge %d: ours %v, llama.cpp %v", i, ours.Merges.At(i), ref.Merges.At(i))
			}
		}
	}
	for _, id := range []int32{ours.BOS, ours.EOS} {
		if id < base || ours.Tokens.At(int(id)) != ref.Tokens.At(int(id)) || ours.Kinds[id] != jlm.TokenControl {
			t.Errorf("special %d: ours %q, llama.cpp %q", id, ours.Tokens.At(int(id)), ref.Tokens.At(int(id)))
		}
	}
	t.Logf("%d tokens and %d merges agree; BOS %d %q, EOS %d %q; llama.cpp pre %q ignore_merges %v",
		base, ours.Merges.Len(), ours.BOS, ours.Tokens.At(int(ours.BOS)), ours.EOS, ours.Tokens.At(int(ours.EOS)),
		ref.PreName, ref.IgnoreMerges)
}

// TestTiktokenVocabEncodesLikeMoonshot runs the container's tokenizer, built
// from tiktoken.model, over tok/testdata/kimi_tiktoken.json -- ids produced by
// Moonshot's own TikTokenTokenizer (scripts/tiktokgold.py). The corpus reaches
// camelCase (where llama.cpp's kimi-k2 splitter departs from the pat_str),
// Han against other scripts, digit groups, contractions, the long-s and Kelvin
// case-fold, and specials written literally in the text.
func TestTiktokenVocabEncodesLikeMoonshot(t *testing.T) {
	dir := testmodels.Path("Kimi-Linear-48B-A3B-Instruct")
	if !fileExists(filepath.Join(dir, "tiktoken.model")) {
		t.Skipf("MODEL MISSING: %s/tiktoken.model (set JITLLM_MODELS to the model directory) -- this gate proved nothing", dir)
	}
	b, err := os.ReadFile(filepath.Join("..", "tok", "testdata", "kimi_tiktoken.json"))
	if err != nil {
		t.Fatal(err)
	}
	var gold struct {
		Cases []struct {
			Text string  `json:"text"`
			IDs  []int32 `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(b, &gold); err != nil || len(gold.Cases) == 0 {
		t.Fatalf("golden: %v, %d cases", err, len(gold.Cases))
	}
	jv, err := hfTiktokenVocab(dirFiles(dir), 163840)
	if err != nil {
		t.Fatal(err)
	}
	v, err := tok.New(jv)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range gold.Cases {
		if got := v.Encode(c.Text, true); !slices.Equal(got, c.IDs) {
			t.Errorf("Encode(%q)\n  ours      %v\n  Moonshot  %v", c.Text, got, c.IDs)
		}
	}
}
