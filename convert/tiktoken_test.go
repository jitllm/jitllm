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
	for _, p := range []string{filepath.Join(dir, "tiktoken.model"), filepath.Join(dir, "tokenizer_config.json"), gg} {
		if !fileExists(p) {
			testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS to the model directory) -- this gate proved nothing", p)
		}
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
	if len(ref.Tokens) < base {
		t.Fatalf("llama.cpp's table has %d tokens", len(ref.Tokens))
	}
	bad := 0
	for i := 0; i < base; i++ {
		if ours.Tokens[i] != ref.Tokens[i] {
			if bad++; bad <= 5 {
				t.Errorf("token %d: ours %q, llama.cpp %q", i, ours.Tokens[i], ref.Tokens[i])
			}
		}
	}
	if len(ours.Merges) != len(ref.Merges) {
		t.Errorf("%d merges, llama.cpp %d", len(ours.Merges), len(ref.Merges))
	}
	for i := 0; i < min(len(ours.Merges), len(ref.Merges)); i++ {
		if ours.Merges[i] != ref.Merges[i] {
			if bad++; bad <= 10 {
				t.Errorf("merge %d: ours %v, llama.cpp %v", i, ours.Merges[i], ref.Merges[i])
			}
		}
	}
	for _, id := range []int32{ours.BOS, ours.EOS} {
		if id < base || int(id) >= len(ours.Tokens) || int(id) >= len(ref.Tokens) {
			t.Fatalf("special %d is not one of the %d tokens past the %d base ones", id, len(ours.Tokens)-base, base)
		}
		if ours.Tokens[id] != ref.Tokens[id] || ours.Kinds[id] != jlm.TokenControl {
			t.Errorf("special %d: ours %q, llama.cpp %q", id, ours.Tokens[id], ref.Tokens[id])
		}
	}
	t.Logf("%d tokens and %d merges agree; BOS %d %q, EOS %d %q; llama.cpp pre %q ignore_merges %v",
		base, len(ours.Merges), ours.BOS, ours.Tokens[ours.BOS], ours.EOS, ours.Tokens[ours.EOS],
		ref.PreName, ref.IgnoreMerges)
}

// TestTiktokenVocabEncodesLikeMoonshot runs the container's tokenizer, built
// from tiktoken.model, over tok/testdata/kimi_tiktoken.json -- ids produced by
// Moonshot's own TikTokenTokenizer (scripts/tiktokgold.py). The corpus reaches
// camelCase (where llama.cpp's kimi-k2 splitter departs from the pat_str),
// Han against other scripts, digit groups, contractions, the long-s and Kelvin
// case-fold, and specials written literally in the text.
func TestTiktokenVocabEncodesLikeMoonshot(t *testing.T) {
	testmodels.SourceTree(t)
	dir := testmodels.Path("Kimi-Linear-48B-A3B-Instruct")
	if !fileExists(filepath.Join(dir, "tiktoken.model")) {
		testmodels.Missing(t, "MODEL MISSING: %s/tiktoken.model (set JITLLM_MODELS to the model directory) -- this gate proved nothing", dir)
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
