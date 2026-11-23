package tok_test

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/tok"
)

// TestIDFindsTheFirstTokenOfADecodedPiece: on SentencePiece and byte-level
// BPE vocabularies, ID of every token's decoded piece is the token whose
// text it is, else the lowest id decoding to it, as a scan of the
// vocabulary finds it, and ID of the fill-in-the-middle names a vocabulary
// does not carry misses. On the BPE vocabulary some pieces are reached only
// through their decoding ("\n\n" is stored as "ĊĊ"), and the gate fails if
// none is. The violation, an index of the raw texts, fails it.
func TestIDFindsTheFirstTokenOfADecodedPiece(t *testing.T) {
	for _, stem := range []string{"stories15M-q8_0", "Llama-3.2-1B-Instruct-Q4_K_M"} {
		s := openSrc(t, testmodels.Path(stem))
		if s == nil {
			testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS to the model directory) -- this gate would prove nothing (RULE 11)", stem)
		}
		v, err := tok.New(s.vocab(t))
		s.Close()
		if err != nil {
			t.Fatal(err)
		}
		raw, first := map[string]int32{}, map[string]int32{}
		for i := range int32(v.Size()) {
			if _, ok := raw[v.Text(i)]; !ok {
				raw[v.Text(i)] = i
			}
			d := v.Decode([]int32{i})
			if _, ok := first[d]; !ok {
				first[d] = i
			}
		}
		byDecoding := 0
		for i := range int32(v.Size()) {
			d := v.Decode([]int32{i})
			want, isRaw := raw[d]
			if !isRaw {
				want = first[d]
				byDecoding++
			}
			if id, ok := v.ID(d); !ok || id != want {
				t.Fatalf("%s: ID(%q) is %d %v, the scan's %d", stem, d, id, ok, want)
			}
		}
		if stem != "stories15M-q8_0" && byDecoding == 0 {
			t.Fatalf("%s: no piece was found through its decoding: the gate checked nothing", stem)
		}
		for _, name := range []string{"<|fim_prefix|>", "<fim-prefix>", "▁<PRE>", "<｜fim▁begin｜>"} {
			_, isRaw := raw[name]
			if _, dec := first[name]; dec || isRaw {
				continue
			}
			if id, ok := v.ID(name); ok {
				t.Fatalf("%s: ID(%q) is %d, and no token decodes to it", stem, name, id)
			}
		}
		t.Logf("%s: %d pieces found through their decoding", stem, byDecoding)
	}
}
