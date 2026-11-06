package engine

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// testFrames is frames over m of the test's own, for a gate that pages the
// model's blocks without an instance (jlm.Budget gives them a budget); closed
// when the test ends.
func testFrames(tb testing.TB, m *model.Model, opts ...jlm.FramesOption) *jlm.Frames {
	tb.Helper()
	fr, err := m.NewFrames(opts...)
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(fr.Close)
	return fr
}

// budgetEvicts reports whether a page budget of b bytes holds fewer than
// every page of m: a page-in under it evicts.
func budgetEvicts(m *model.Model, b uint64) bool {
	var all uint64
	for i := 0; i < m.Container.NPages(); i++ {
		all += m.Container.PageBytes(i)
	}
	return b > 0 && b < all
}

// instancesOf is the instances runs step.
func instancesOf(runs []Run) []*Instance {
	out := make([]*Instance, len(runs))
	for i, r := range runs {
		out[i] = r.Instance
	}
	return out
}

// framesOfInstance is every frames s pages its blocks through: the CPU
// device's and each card's it placed a block on.
func framesOfInstance(s *Instance) []*jlm.Frames {
	out := []*jlm.Frames{s.fr}
	seen := map[string]bool{}
	for _, at := range s.Locations() {
		if at == "" || onCPU(at) || seen[at] {
			continue
		}
		seen[at] = true
		if fr := s.FramesOn(nn.DeviceName(at)); fr != nil {
			out = append(out, fr)
		}
	}
	return out
}
