package graph

import (
	"fmt"
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

// The operand-prologue sites are found by their edges in both covered
// graphs, once each; a second reader of the attention norm's output is a near
// miss refused with its reason; and each is refused at the chunk row class,
// on a subgroup that is not 32 lanes and on a backend with no contract.
func TestPrologueSitesAreFoundByPattern(t *testing.T) {
	dec := Target{API: "ptx", Subgroup: 32, Rows: RowsDecode}
	for _, qk := range []bool{false, true} {
		f := stated(qk)
		g := f.Config.Build()
		for _, rw := range []*Rewrite{ActInQKV, ActInO, ActInDown} {
			os := Match(g, rw)
			if len(os) != 1 {
				t.Fatalf("qk=%v %s: %d regions", qk, rw.Name, len(os))
			}
			if err := os[0].Offer(dec); err != nil {
				t.Fatalf("qk=%v %s at decode: %v", qk, rw.Name, err)
			}
			for _, bad := range []Target{{API: "ptx", Subgroup: 32, Rows: RowsChunk},
				{API: "ptx", Subgroup: 64, Rows: RowsDecode}, {API: "host", Subgroup: 32, Rows: RowsDecode}} {
				if os[0].Offer(bad) == nil {
					t.Fatalf("qk=%v %s offered at %s", qk, rw.Name, bad.Key())
				}
			}
		}
	}
	f := stated(false)
	g := f.Config.Build()
	// A second reader of the attention norm: a probe of the normed row.
	h := g.Ops[0].Out[0]
	out := g.value("probe", g.Values[h].Width)
	g.Ops = append(g.Ops[:1], append([]Op{{Kind: OpAdd, In: []ValueID{h, h}, Out: []ValueID{out}}}, g.Ops[1:]...)...)
	os := Match(g, ActInQKV)
	if len(os) != 1 || !strings.Contains(fmt.Sprint(os[0].Offer(dec)), "readers") {
		t.Fatalf("second reader of the attention norm: %v", os)
	}
}
