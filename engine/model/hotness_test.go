//go:build amd64 || arm64

package model

import (
	"math"
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestHotnessIsAProbability holds the measured profile to the shape tier.Place
// needs: every expert's Uses in [0,1], and NExpertUsed of them per token.
//
// It guards the denominator: Uses is a per-token probability and attention is
// exactly 1.0, so counting prefill rows against one token would put experts
// above 1 and rank them above weights every token reads in full.
func TestHotnessIsAProbability(t *testing.T) {
	path := testmodels.Resolve(os.Getenv("JITLLM_ROUTE_MODEL"))
	if path == "" {
		t.Skip("set JITLLM_ROUTE_MODEL to a mixture-of-experts gguf")
	}
	m, err := Open(jlmOf(t, path))
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()
	if !m.Cfg.MoE() {
		t.Skipf("%s is not a mixture of experts", path)
	}
	const n = 24
	ids := m.Vocab.Encode("The capital of France is", true)
	s := m.NewState(len(ids) + n + 1)
	defer s.Close()
	var lg []float32
	for _, id := range ids {
		if lg, err = s.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < n; i++ {
		if lg, err = s.Forward(Greedy(lg)); err != nil {
			t.Fatal(err)
		}
	}

	uses, tokens := s.ExpertUses()
	if tokens == 0 {
		t.Fatal("no tokens were counted; the hotness hook is not firing")
	}
	c := m.Cfg
	if len(uses) != c.NExpert*c.NLayer {
		t.Fatalf("got %d entries, want %d", len(uses), c.NExpert*c.NLayer)
	}
	for li := 0; li < c.NLayer; li++ {
		// A dense lead block (Kimi-K3's, DeepSeek's) routes nothing.
		if !c.MoEAt(li) {
			continue
		}
		var sum float64
		for e := 0; e < c.NExpert; e++ {
			u := uses[li*c.NExpert+e]
			if u < 0 || u > 1 {
				t.Fatalf("layer %d expert %d: Uses %.4f outside [0,1] over %d tokens",
					li, e, u, tokens)
			}
			sum += float64(u)
		}
		// Every token selects exactly NExpertUsed experts per layer, so the
		// layer's probabilities sum to that.
		if math.Abs(sum-float64(c.NExpertUsed)) > 1e-9 {
			t.Fatalf("layer %d uses sum to %.4f over %d tokens, want NExpertUsed = %d",
				li, sum, tokens, c.NExpertUsed)
		}
	}
	t.Logf("%d tokens, %d layers x %d experts, %d used", tokens, c.NLayer, c.NExpert, c.NExpertUsed)
}
