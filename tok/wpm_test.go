package tok

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// wpmVocab is a hand-built WordPiece vocabulary in llama.cpp's spelling: a
// word-initial piece carries "▁", a continuation carries nothing.
func wpmVocab(lower, keepMarks bool) *jlm.Vocab {
	toks := []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]",
		"▁hello", "▁cafe", "▁café", ",", "▁,", "▁!", "▁play", "ing", "▁日", "▁本", "▁x"}
	kinds := make([]jlm.TokenKind, len(toks))
	for i := range kinds {
		kinds[i] = jlm.TokenNormal
	}
	for i := 0; i < 4; i++ {
		kinds[i] = jlm.TokenControl
	}
	return &jlm.Vocab{Kind: jlm.VocabWPM, Tokens: jlm.NewStrings(toks), Kinds: kinds,
		BOS: 2, EOS: 3, Unk: 1, Sep: 3, Pad: 0, Mask: -1, AddBOS: true, AddEOS: true,
		Pre: []jlm.PreOp{{Kind: jlm.PreBert, CaseFold: lower, LetterMarks: keepMarks}}}
}

// TestWordPieceNormalizesAndSplits is the unit half of the WordPiece gate (the
// model half is model.TestEmbeddingsMatchReference, against llama.cpp and the
// models' own tokenizers). Each case is one BertNormalizer / BertPreTokenizer
// rule, and each rule's violation is the second vocabulary below.
func TestWordPieceNormalizesAndSplits(t *testing.T) {
	v, err := New(wpmVocab(true, false))
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		text string
		want []int32
	}{
		// lowercase, accent stripped: "Café" -> "cafe" (NFD base, mark dropped)
		{"HELLO Café", []int32{2, 4, 5, 3}},
		// punctuation is its own word, so it is word-initial: "▁,"
		{"hello,hello!", []int32{2, 4, 8, 4, 9, 3}},
		// greedy longest match, then a continuation piece
		{"playing", []int32{2, 10, 11, 3}},
		// every CJK ideograph is a word of its own
		{"日本", []int32{2, 12, 13, 3}},
		// a word no piece covers is one [UNK], not a partial match
		{"zzz x", []int32{2, 1, 14, 3}},
		// control characters vanish; a literal special is that token
		{"hello\x07 [SEP]", []int32{2, 4, 3, 3}},
	} {
		if got := v.Encode(c.text, true); !slices.Equal(got, c.want) {
			t.Errorf("%q: %v, want %v", c.text, got, c.want)
		}
	}
	// The violations: a cased vocabulary keeps "HELLO" whole (so it is
	// unknown here) and keeps the accent (so "café" is its own piece).
	vc, err := New(wpmVocab(false, true))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := vc.Encode("HELLO café", true), []int32{2, 1, 6, 3}; !slices.Equal(got, want) {
		t.Errorf("cased, accents kept: %v, want %v", got, want)
	}
	if got := v.Decode([]int32{2, 10, 11, 8, 4, 3}); got != "playing , hello" {
		t.Errorf("decode: %q", got)
	}
}
