package engine

import (
	"bytes"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestResetForgetsTheTokenStream: a reused session must not name the NEXT
// sequence's pages after the PREVIOUS one's tokens. Reset must clear kc.seq
// (the ids page keys are hashed from): paths that record no ids (ForwardEmbd,
// PrefillMixed) would otherwise seal pages under the old prompt's hash. It
// fails against deleting `kc.seq = kc.seq[:0]` from reset.
func TestResetForgetsTheTokenStream(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	// A small page so several seal inside a short prompt; otherwise nothing
	// would be stored.
	const maxSeq, page, prompt = 128, 8, 40

	ms := NewMemStore()
	s := testInstance(t, m, maxSeq, withKVPage(page))
	defer s.Close()
	s.SetKVStore(ms)
	mustKey(t, s, "tinyllama/q3_K_M/f32")

	ids := make([]int32, prompt)
	for i := range ids {
		ids[i] = int32(300 + i*7)
	}
	if _, err := s.Prefill(ids); err != nil {
		t.Fatal(err)
	}
	key := s.kv.pageKey(0, 0)
	if key == "" {
		t.Fatal("layer 0 page 0 has no key, so nothing was stored and this gate would pass " +
			"against any violation")
	}
	var before bytes.Buffer
	if err := ms.Get(key, 0, 0, &before); err != nil {
		t.Fatalf("the store holds no page 0 under %q: %v", key, err)
	}
	if before.Len() == 0 {
		t.Fatal("page 0 is empty")
	}

	// Reuse the session for a sequence of embeddings, which note() never sees.
	s.Reset()
	if n := len(s.kv.seq); n != 0 {
		t.Fatalf("Reset left %d noted id(s) behind; the next sequence's pages would be named "+
			"after the last one's tokens", n)
	}
	e := make([]float32, prompt*m.Cfg.NEmbd)
	for i := range e {
		e[i] = float32(i%17) * 0.01
	}
	if _, err := s.PrefillMixed(model.Span{Embd: e}); err != nil {
		t.Fatal(err)
	}

	var after bytes.Buffer
	if err := ms.Get(key, 0, 0, &after); err != nil {
		t.Fatalf("the page named by the first prompt vanished: %v", err)
	}
	if !bytes.Equal(before.Bytes(), after.Bytes()) {
		t.Fatalf("the page named by H(tokens[0:%d]) now holds an embedding run's keys -- a later "+
			"run of that prompt would attend over a history the model never wrote for it", page)
	}
}

// TestNoteRefusesToAppendAcrossAGap: seq[i] must be the id at position i, or
// every page key past the gap names the wrong window. It asserts the invariant
// rather than a token diff, since a wrongly named page only shows when another
// session presents the shifted window's prompt. It fails against note()
// appending at len(seq) instead of at pos.
func TestNoteRefusesToAppendAcrossAGap(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	const maxSeq, page, prompt = 128, 8, 40

	s := testInstance(t, m, maxSeq, withKVPage(page))
	defer s.Close()
	s.SetKVStore(NewMemStore())
	mustKey(t, s, "tinyllama/q3_K_M/f32")

	ids := make([]int32, prompt)
	for i := range ids {
		ids[i] = int32(300 + i*7)
	}
	if _, err := s.Prefill(ids); err != nil {
		t.Fatal(err)
	}
	if got := len(s.kv.seq); got != prompt {
		t.Fatalf("a pure-token prefill noted %d ids for %d positions", got, prompt)
	}
	// One position that has no id at all.
	e := make([]float32, m.Cfg.NEmbd)
	for i := range e {
		e[i] = float32(i%17) * 0.01
	}
	if _, err := s.ForwardEmbd(e); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 20; i++ {
		if _, err := s.Forward(int32(500 + i)); err != nil {
			t.Fatal(err)
		}
	}
	// Either seq tracks the position exactly, or it stopped at the gap and names
	// nothing past it. What it must never do is carry ids at the wrong index.
	if n := len(s.kv.seq); n != s.Pos() && n != prompt {
		t.Fatalf("position %d, %d noted ids, gap at %d -- seq[i] is no longer the id at "+
			"position i, so every page key past the gap names a shifted window",
			s.Pos(), n, prompt)
	}
	// And the ids it did keep are the ones that were really there.
	for i, id := range s.kv.seq {
		if id != ids[i] {
			t.Fatalf("noted id %d at position %d, the prompt has %d", id, i, ids[i])
		}
	}
}
