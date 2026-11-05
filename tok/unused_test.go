package tok_test

import (
	"testing"

	"github.com/jitllm/jitllm/tok"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestDecodeRendersUnusedAsNothing: a token of type UNUSED is a converter's
// padding placeholder ("[PAD50954]" in phi-2) and decodes to nothing, as in
// llama.cpp. Both tokenizer families, because Decode is two functions.
func TestDecodeRendersUnusedAsNothing(t *testing.T) {
	for _, c := range []struct{ stem, unused string }{
		{testmodels.Path("synth-phi2"), "[PAD50954]"},    // BPE
		{testmodels.Path("gemma-2b"), "<unused_256000>"}, // SentencePiece
	} {
		s := openSrc(t, c.stem)
		if s == nil {
			testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS to the model directory) -- this gate would prove nothing (RULE 11)", c.stem)
		}
		v, err := tok.New(s.vocab(t))
		s.Close()
		if err != nil {
			t.Fatal(err)
		}
		id, ok := v.ID(c.unused)
		if !ok {
			t.Fatalf("%s: vocabulary has no %s", c.stem, c.unused)
		}
		if v.Kind(id) != tok.TypeUnused {
			t.Fatalf("%s: %s is type %d, not UNUSED -- the gate is not testing what it names", c.stem, c.unused, v.Kind(id))
		}
		// The placeholder must vanish and its neighbours must not.
		word := v.Encode(" Paris", false)
		both := append(append(append([]int32{}, word...), id), word...)
		if got, want := v.Decode(both), v.Decode(word)+v.Decode(word); got != want {
			t.Errorf("%s: Decode(%q + %s + %q) = %q, want %q", c.stem, " Paris", c.unused, " Paris", got, want)
		}
	}
}
