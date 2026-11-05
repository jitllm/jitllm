package tok_test

import (
	"math/rand"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/tok"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// requireStreamIsDecodeChat checks that ChatStream's pieces, concatenated, are
// DecodeChat of every prefix of ids, byte for byte.
func requireStreamIsDecodeChat(t *testing.T, name string, v *tok.Vocab, ids []int32) {
	t.Helper()
	cs := v.NewChatStream()
	var b strings.Builder
	for i, id := range ids {
		b.WriteString(cs.Next(id))
		if want := v.DecodeChat(ids[:i+1]); b.String() != want {
			t.Fatalf("%s: after %d ids %v the stream reads %q, DecodeChat %q", name, i+1, ids[:i+1], b.String(), want)
		}
	}
}

// TestChatStreamIsDecodeChat: the incremental decoder a server streams through
// is DecodeChat at every prefix, on every vocabulary kind -- SentencePiece,
// byte-level BPE (multi-byte text split across ids), harmony's channels and
// WordPiece's dropped opening space -- over encoded text and random ids.
//
// VIOLATION SIGNATURE. Drop piece's TrimPrefix and the WordPiece arm fails
// with `the stream reads " hello", DecodeChat "hello"`; drop the thinkClose
// and the harmony arm fails at its first <|end|>.
func TestChatStreamIsDecodeChat(t *testing.T) {
	text := "Once upon a time, café 日本語 — naïve “quotes” 🙂 the end.\n"
	w, err := tok.New(tok.WPMVocab(true, false))
	if err != nil {
		t.Fatal(err)
	}
	requireStreamIsDecodeChat(t, "wordpiece", w, []int32{2, 4, 5, 7, 11, 12, 3, 14})
	requireStreamIsDecodeChat(t, "wordpiece, control first", w, []int32{0, 2, 13, 4, 11})

	for _, stem := range []string{"stories15M-q8_0", "Llama-3.2-1B-Instruct-Q4_K_M", "gpt-oss-20b-Q4_K_M"} {
		s := openSrc(t, testmodels.Path(stem))
		if s == nil {
			testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS to the model directory) -- this gate would prove nothing (RULE 11)", stem)
		}
		v, err := tok.New(s.vocab(t))
		s.Close()
		if err != nil {
			t.Fatal(err)
		}
		ids := v.Encode(text, false)
		requireStreamIsDecodeChat(t, stem+" text", v, ids)
		r := rand.New(rand.NewSource(1))
		rnd := make([]int32, 300)
		for i := range rnd {
			rnd[i] = int32(r.Intn(v.Size()))
		}
		requireStreamIsDecodeChat(t, stem+" random", v, rnd)
		if !v.Harmony() {
			continue
		}
		sp := func(name string) int32 {
			id, ok := v.ID(name)
			if !ok {
				t.Fatalf("%s: no %s", stem, name)
			}
			return id
		}
		var turn []int32
		turn = append(turn, sp("<|channel|>"))
		turn = append(turn, v.Encode("analysis", false)...)
		turn = append(turn, sp("<|message|>"))
		turn = append(turn, ids...)
		turn = append(turn, sp("<|end|>"), sp("<|start|>"))
		turn = append(turn, v.Encode("assistant", false)...)
		turn = append(turn, sp("<|channel|>"))
		turn = append(turn, v.Encode("final", false)...)
		turn = append(turn, sp("<|message|>"))
		turn = append(turn, ids...)
		turn = append(turn, sp("<|return|>"))
		if got := v.DecodeChat(turn); !strings.Contains(got, "</think>") {
			t.Fatalf("%s: the harmony turn renders no reasoning block (%q), so this arm proves nothing", stem, got)
		}
		requireStreamIsDecodeChat(t, stem+" harmony", v, turn)
		// Random ids through the state machine: the specials scattered.
		specials := []int32{sp("<|start|>"), sp("<|channel|>"), sp("<|message|>"), sp("<|end|>"), sp("<|return|>")}
		for i := range rnd {
			if r.Intn(8) == 0 {
				rnd[i] = specials[r.Intn(len(specials))]
			}
		}
		requireStreamIsDecodeChat(t, stem+" harmony random", v, rnd)
	}
}
