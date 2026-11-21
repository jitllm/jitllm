package engine

import (
	"context"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/model"
)

// TestJointGroupsOnOneInstanceKeepTheirResults steps two groups of sessions
// leasing one instance as the server's decode groups do (StepJoint), group A,
// then group B before A's results are read: A's results are still A's
// sessions' logits -- each its decode alone -- not B's, which the instance's
// own result list (out nil) would hand it. The violation arm reads that list
// and must see B's logits under A.
func TestJointGroupsOnOneInstanceKeepTheirResults(t *testing.T) {
	m, err := model.Open(jlmOf(t, models()["stories15M"]))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	in := testInstance(t, m, 64)
	defer in.Close()
	pool := NewPool()
	if err := pool.Serve(in, 4); err != nil {
		t.Fatal(err)
	}
	defer pool.Unserve(in)
	alone := func(id int32) []float32 {
		s := testInstance(t, m, 64)
		defer s.Close()
		lg, err := s.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		return slices.Clone(lg)
	}
	session := func() *Session {
		s, err := pool.NewSession(64)
		if err != nil {
			t.Fatal(err)
		}
		if err := s.Acquire(context.Background(), PhaseGenerate); err != nil {
			t.Fatal(err)
		}
		return s
	}
	a, b := session(), session()
	defer a.Close()
	defer b.Close()
	wantA, wantB := alone(2000), alone(3000)
	if n, _ := nmseOf(wantA, wantB); n < 1e-3 {
		t.Fatalf("the two tokens' logits stand %.3e apart: the gate could not tell A's result from B's", n)
	}
	ga := []Run{{Session: a, Tokens: []int32{2000}, Logits: true}}
	gb := []Run{{Session: b, Tokens: []int32{3000}, Logits: true}}
	resA, err := StepJoint(ga, make([][]float32, 0, 1))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := StepJoint(gb, make([][]float32, 0, 1)); err != nil {
		t.Fatal(err)
	}
	if n, _ := nmseOf(resA[0], wantA); n > 1e-10 {
		t.Fatalf("group A's result after group B stepped is not A's logits (NMSE %.3e)", n)
	}
	if n, _ := nmseOf(resA[0], wantB); n < 1e-6 {
		t.Fatal("group A's result is group B's logits")
	}
	// The violation: the instance's own list, held across B's step.
	a2, b2 := session(), session()
	defer a2.Close()
	defer b2.Close()
	sharedA, err := StepJoint([]Run{{Session: a2, Tokens: []int32{2000}, Logits: true}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := StepJoint([]Run{{Session: b2, Tokens: []int32{3000}, Logits: true}}, nil); err != nil {
		t.Fatal(err)
	}
	if n, _ := nmseOf(sharedA[0], wantB); n > 1e-10 {
		t.Fatalf("VIOLATION arm: the instance's own list held across B's step still reads A's logits (NMSE %.3e to B's)", n)
	}
}
