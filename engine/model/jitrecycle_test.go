package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// declining is an accelerator that serves nothing.
type declining struct{}

func (declining) MatVec(out []float32, t quant.Type, w []byte, x []float32, nrows, k int) bool {
	return false
}

// TestASessionRecyclesTheJITItClosed holds the JIT a closed State hands the
// next to a fresh one: the second session gets the first one's JIT, with no
// device although the first had attached one, its counters where a fresh
// JIT's start, and the same logits to the bit; a session opened beside it
// builds its own.
func TestASessionRecyclesTheJITItClosed(t *testing.T) {
	m, err := Open(jlmOf(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := make([]int32, 70)
	for i := range ids {
		ids[i] = int32(11 + 7*i%900)
	}
	run := func(s *State) []float32 {
		out, err := s.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		for i := range 6 {
			if out, err = s.Forward(int32(30 + i)); err != nil {
				t.Fatal(err)
			}
		}
		return append([]float32(nil), out...)
	}
	s1 := m.NewState(128)
	j1 := s1.jit
	p1, _ := s1.jit.Regions()
	want := run(s1)
	s1.jit.SetDevice(declining{})
	s1.jit.SetLookahead(false)
	if err := s1.Close(); err != nil {
		t.Fatal(err)
	}

	s2 := m.NewState(128)
	defer s2.Close()
	if s2.jit != j1 {
		t.Fatal("the second session built a JIT; the first one's was not recycled")
	}
	if d := s2.jit.Device(); d != nil {
		t.Fatalf("the recycled JIT carries the closed session's device %T", d)
	}
	if p2, _ := s2.jit.Regions(); p2 != p1 {
		t.Fatalf("the recycled JIT starts at %d parallel regions, a fresh one at %d", p2, p1)
	}
	s3 := m.NewState(128)
	if s3.jit == s2.jit {
		t.Fatal("two open sessions share one JIT")
	}
	s3.Close()
	got := run(s2)
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("logit %d: %v from the recycled JIT, %v from a fresh one", i, got[i], want[i])
		}
	}
}
