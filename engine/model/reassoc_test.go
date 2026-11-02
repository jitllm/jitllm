//go:build amd64 && linux

package model

import (
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
)

// TestGreedyDivergenceControl calibrates how long two numerically-equivalent
// configurations stay on the same greedy path.
//
// It exists because "Prefill and Forward produce different token 26" is not by
// itself evidence of a bug. Greedy decoding is autoregressive: a float32
// reassociation shifts the logits by ~1e-5, and the first time two candidates
// are within that of each other the argmax flips and the two runs never rejoin.
// The question is not whether that happens but whether it happens SOONER than
// it does for two configurations nobody suspects.
//
// Both arms here are the ordinary decode path. They differ only in interleave
// width, which changes the summation order and nothing else.
func TestGreedyDivergenceControl(t *testing.T) {
	path := benchModel(t)
	m, err := Open(jlmOf(t, path))
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()

	ids := m.Vocab.Encode("The capital of France is Paris, and the capital of "+
		"Germany is Berlin, and the capital of Italy is Rome, and the capital "+
		"of Spain is Madrid, and the capital of Portugal is", true)

	run := func(pack int) []int32 {
		// Per-instance options: NewInstance builds its own JIT from m.jit, so the two
		// widths are two emitted kernels in ONE process.
		m.jit = []nn.Option{nn.WithPackWidth(pack), nn.WithTune(nn.TuneOff)}
		s := m.NewInstance(len(ids) + 64)
		defer s.Close()
		var lg []float32
		for _, id := range ids {
			if lg, err = s.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		out := make([]int32, 0, 48)
		for i := 0; i < 48; i++ {
			tok := Greedy(lg)
			out = append(out, tok)
			if lg, err = s.Forward(tok); err != nil {
				t.Fatal(err)
			}
		}
		return out
	}
	a, b := run(2), run(4)
	diverged := -1
	for i := range a {
		if a[i] != b[i] {
			diverged = i
			break
		}
	}
	if diverged < 0 {
		t.Logf("pack2 and pack4 agree for all %d greedy steps", len(a))
	} else {
		t.Logf("pack2 and pack4 -- two ordinary decode configs -- diverge at greedy step %d", diverged)
	}
}
