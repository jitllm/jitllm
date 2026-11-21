package convert

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestTokenizerModelsAreKinds holds tokenizers.json to the kinds the
// container defines: each model of a known kind, a WordPiece one naming its
// pre-tokenizer and no other naming one, and llama's SentencePiece with its
// byte fallback, space prefix and BOS.
func TestTokenizerModelsAreKinds(t *testing.T) {
	tms, err := tokenizerModels()
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []struct {
		model string
		kind  jlm.VocabKind
	}{{"llama", jlm.VocabSPM}, {"gpt2", jlm.VocabBPE}, {"bert", jlm.VocabWPM}, {"wordpiece", jlm.VocabWPM}, {"gemma4", jlm.VocabBPESPM}} {
		tm, ok := tms[want.model]
		if !ok || vocabKinds()[tm.Kind] != want.kind {
			t.Errorf("%s is %+v, want kind %v", want.model, tm, want.kind)
		}
	}
	if l := tms["llama"]; !l.ByteFallback || !l.SpacePrefix || !l.AddBOS {
		t.Errorf("llama's SentencePiece is %+v", l)
	}
	for _, bad := range []string{`{"x": {"kind": "unigram"}}`, `{"x": {"kind": "wpm"}}`, `{"x": {"kind": "bpe", "pre": "bert"}}`,
		`{"x": {"kind": "bpe", "bos": true}}`, `{}`} {
		if _, err := tokenizerModelsOf([]byte(bad)); err == nil {
			t.Errorf("%s was read", bad)
		}
	}
}
