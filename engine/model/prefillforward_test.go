package model

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestPrefillForwardRowsIsTheForwardLoop: with WithPrefillForwardRows(n), a
// prompt of up to n tokens is the decode path's Forward over each token --
// the same logits, bit for bit -- and a longer one, or any prompt without the
// option, still takes the batched chunk (State.forwardPrompts counts which).
func TestPrefillForwardRowsIsTheForwardLoop(t *testing.T) {
	m, err := Open(jlmOf(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")), noTune, WithPrefillForwardRows(6))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	loop := func(ids []int32) []float32 {
		s := m.NewState(len(ids) + 2)
		defer s.Close()
		var lg []float32
		for _, id := range ids {
			if lg, err = s.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return slices.Clone(lg)
	}
	pre := func(ids []int32, forward int) []float32 {
		s := m.NewState(len(ids) + 2)
		defer s.Close()
		lg, err := s.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		if s.forwardPrompts != forward {
			t.Fatalf("a %d-token prompt ran %d prompts token by token, want %d", len(ids), s.forwardPrompts, forward)
		}
		return slices.Clone(lg)
	}
	short := m.Vocab.Encode("The capital of France is", true)
	if len(short) > 6 {
		t.Fatalf("the short prompt is %d tokens", len(short))
	}
	if a, b := loop(short), pre(short, 1); !slices.Equal(a, b) {
		t.Fatal("a prompt under the forward-row limit is not the Forward loop bit for bit")
	}
	pre(m.Vocab.Encode("The capital of France is Paris, and the capital of Italy is Rome.", true), 0)
	m2, err := Open(jlmOf(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m2.Close()
	s := m2.NewState(len(short) + 2)
	defer s.Close()
	if _, err := s.Prefill(short); err != nil {
		t.Fatal(err)
	}
	if s.forwardPrompts != 0 {
		t.Fatal("without the option a short prompt ran token by token")
	}
}
