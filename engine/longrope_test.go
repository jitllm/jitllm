package engine

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestLongRopeTurnsPastTheOriginalContext holds LongRoPE's two factor sets
// to the instance's context (model.RopeFor): synth-phimoe trains its short
// factors to 512 positions and states long ones beside them. An instance of
// a context within 512 turns the short factors, one past it the long, at
// every position -- its decode parts from the short instance's from the
// first rotated position -- and the rotary's table the instance composes is
// the long factors'. An instance that ignored its context would decode as
// the short one, the violation the comparison sees.
func TestLongRopeTurnsPastTheOriginalContext(t *testing.T) {
	src := testmodels.Path("synth-phimoe.gguf")
	path, ok := existingModel(src)
	if !ok {
		testmodels.Missing(t, "MODEL MISSING: %s", src)
	}
	m, err := model.Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.RopeLong == nil {
		t.Fatal("the container's rope_factors_long reached no rotary")
	}
	if m.LongCtx != 512 {
		t.Fatalf("the short factors' context reads %d, want the container's 512", m.LongCtx)
	}
	if slices.Equal(m.RopeLong.Freqs, m.Rope.Freqs) {
		t.Fatal("the fixture's two factor sets agree: the comparison below proves nothing")
	}
	if m.RopeFor(512) != &m.Rope || m.RopeFor(513) != m.RopeLong {
		t.Fatal("RopeFor does not key the long factors on a context past 512")
	}
	ids := []int32{1, 7, 11, 13, 17}
	decode := func(ctx int) []float32 {
		s := testInstance(t, m, ctx)
		defer s.Close()
		var last []float32
		for _, id := range ids {
			lg, err := s.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			last = slices.Clone(lg)
		}
		return last
	}
	short, long := decode(256), decode(1024)
	shortAgain := decode(512)
	if n, _ := nmseOf(shortAgain, short); n > 1e-10 {
		t.Fatalf("two instances within the original context decode apart (NMSE %.3e)", n)
	}
	n, _ := nmseOf(long, short)
	if n < 1e-6 {
		t.Fatalf("an instance past the original context decodes as one within it (NMSE %.3e): the long factors are not turned", n)
	}
	t.Logf("the long factors past 512 move the last logits by NMSE %.3e; within it two contexts agree", n)
}
