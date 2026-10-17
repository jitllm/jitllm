package graph

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// gate2in is found by its edges in both covered graphs, once each, and the
// region is the FFN chain; a graph with a norm between down and the residual
// (gemma's post-norm) is a near miss refused with its reason, as is a chain
// whose gate has a second reader.
func TestGate2InIsFoundByPattern(t *testing.T) {
	for _, qk := range []bool{false, true} {
		f := stated(qk)
		g := f.Config.Build()
		ds := Apply(g, Gate2In)
		if len(ds) != 1 || ds[0].Err != nil {
			t.Fatalf("qk=%v: %v", qk, ds)
		}
		roles := []jlm.Role{}
		for _, i := range ds[0].Region.Ops {
			if len(g.Ops[i].Weight) > 0 {
				roles = append(roles, g.Ops[i].Weight[0].Role)
			}
		}
		want := []jlm.Role{jlm.RoleFFNNorm, jlm.RoleFFNGate, jlm.RoleFFNUp, jlm.RoleFFNDown}
		if len(roles) != len(want) {
			t.Fatalf("qk=%v: region binds %v", qk, roles)
		}
		for i := range want {
			if roles[i] != want[i] {
				t.Fatalf("qk=%v: region binds %v, want %v", qk, roles, want)
			}
		}
	}
	post := stated(false)
	post.Config.PostNorm = true
	ds := Apply(post.Config.Build(), Gate2In)
	if len(ds) != 1 || ds[0].Err == nil || !strings.Contains(ds[0].Err.Error(), "through a norm") {
		t.Fatalf("post-norm near miss: %v", ds)
	}
	base := stated(false)
	g := base.Config.Build()
	for i, o := range g.Ops {
		if o.Kind == OpActMul {
			// A second reader of gate: a probe that keeps its pre-activation.
			gv := o.In[0]
			out := g.value("probe", g.Values[gv].Width)
			g.Ops = append(g.Ops[:i+1], append([]Op{{Kind: OpAdd, In: []ValueID{gv, gv}, Out: []ValueID{out}}}, g.Ops[i+1:]...)...)
			break
		}
	}
	ds = Apply(g, Gate2In)
	if len(ds) != 1 || ds[0].Err == nil || !strings.Contains(ds[0].Err.Error(), "reader besides") {
		t.Fatalf("second reader of gate: %v", ds)
	}
}
