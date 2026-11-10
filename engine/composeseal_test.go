package engine

import (
	"errors"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// TestTheCPUDeviceComposesNothingAtAStep is the composition rule made a
// property of the CPU device rather than a count: once the instance is
// placed and has stepped, a form it never made -- block 0's graph at a row
// count no placement formed -- is declined as not formed, and the model's
// forms are not even asked. Inside a placement the same request forms: that
// is the gate seeing the guard and not a form that cannot be made.
func TestTheCPUDeviceComposesNothingAtAStep(t *testing.T) {
	m, err := model.Open(jlmOf(t, models()["stories260K"]))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	s := testInstance(t, m, 32)
	defer s.Close()
	if _, err := s.Forward(1); err != nil {
		t.Fatal(err)
	}
	g := s.planFor(0).Graph
	if g == nil {
		t.Fatal("block 0 states no graph: the gate would ask for nothing")
	}
	x := execOf(s)
	asked := s.Account(nn.CPU).Asks
	_, err = x.FormGraph(s.partPlan(g, 7, nil))
	var d cpudev.DeclineError
	if !errors.As(err, &d) || d.Why != nn.DeclineNotFormed {
		t.Fatalf("a form asked after placement: %v, want a not-formed decline", err)
	}
	if n := s.Account(nn.CPU).Asks - asked; n != 0 {
		t.Fatalf("the declined form asked the model's forms %d time(s)", n)
	}
	t.Logf("after placement: %v", err)
	done := x.Composer().Place()
	f, err := x.FormGraph(s.partPlan(g, 7, nil))
	done()
	if err != nil || f == nil {
		t.Fatalf("VIOLATION arm: inside a placement the same form did not form: %v", err)
	}
}
