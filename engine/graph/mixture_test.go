package graph

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// mixFacts is a mixture sublayer as a container would state it: 64 experts
// of 256 over a row of 512, top 8, with or without a gated shared expert and
// the banks' biases.
func mixFacts(shared, biased bool) Facts {
	inv := map[jlm.Role]Shape{
		jlm.RoleRouter:      {K: 512, Rows: 64},
		jlm.RoleExpGateBank: {K: 512, Rows: 256, Sheets: 64}, jlm.RoleExpUpBank: {K: 512, Rows: 256, Sheets: 64},
		jlm.RoleExpDownBank: {K: 256, Rows: 512, Sheets: 64},
	}
	m := Mixture{Block: 2, NEmbd: 512, NExpert: 64, K: 8, NFF: 256, Act: ActSiLU, Router: RouterSoftmax, Norm: true,
		Inv: inv}
	if shared {
		m.SharedFF, m.SharedGated = 384, true
		inv[jlm.RoleShExpGate], inv[jlm.RoleShExpUp] = Shape{K: 512, Rows: 384}, Shape{K: 512, Rows: 384}
		inv[jlm.RoleShExpDown], inv[jlm.RoleShRouter] = Shape{K: 384, Rows: 512}, Shape{K: 512}
	}
	if biased {
		inv[jlm.RoleRouterBias] = Shape{K: 64}
		inv[jlm.RoleExpGateBias], inv[jlm.RoleExpUpBias] = Shape{K: 256, Rows: 64}, Shape{K: 256, Rows: 64}
		inv[jlm.RoleExpDownBias] = Shape{K: 512, Rows: 64}
	}
	return Facts{Mixture: &m, Inventory: inv}
}

// A mixture sublayer verifies with and without a shared expert and the
// banks' biases, and both mixture rewrites find its routed experts by their
// edges: route, gather, gate, up, act_mul, down and combine. The verifier
// refuses a top-k the configuration does not state, a bank of the wrong
// expert count, and a shared gate no op binds.
func TestMixtureGraphVerifies(t *testing.T) {
	for _, c := range []struct{ shared, biased bool }{{false, false}, {true, false}, {true, true}} {
		f := mixFacts(c.shared, c.biased)
		g := f.Mixture.Build()
		if err := Verify(g, f); err != nil {
			t.Fatalf("shared %v biased %v: %v\n%s", c.shared, c.biased, err, g)
		}
		for _, rw := range []*Rewrite{MoEFewRow, MoEWaves} {
			rs := rw.Match(g)
			if len(rs) != 1 {
				t.Fatalf("%s: %d regions", rw.Name, len(rs))
			}
			var kinds []string
			for _, i := range rs[0].Ops {
				kinds = append(kinds, g.Ops[i].Kind.String())
			}
			if got := strings.Join(kinds, " "); got != "route gather expert_matvec expert_matvec act_mul expert_matvec combine" {
				t.Fatalf("%s: region %s", rw.Name, got)
			}
		}
	}
	f := mixFacts(true, false)
	g := f.Mixture.Build()
	g.Ops[0].Attr.TopK = 6
	if err := Verify(g, f); err == nil || !strings.Contains(err.Error(), "top 6 of 64") {
		t.Fatalf("a top-k the configuration does not state verified: %v", err)
	}
	f = mixFacts(false, false)
	f.Inventory[jlm.RoleExpUpBank] = Shape{K: 512, Rows: 256, Sheets: 32}
	if err := Verify(f.Mixture.Build(), f); err == nil || !strings.Contains(err.Error(), "exp_up_bank") {
		t.Fatalf("a bank of 32 experts in a 64-expert mixture verified: %v", err)
	}
	f = mixFacts(true, false)
	f.Mixture.SharedGated = false
	if err := Verify(f.Mixture.Build(), f); err == nil || !strings.Contains(err.Error(), "sh_router") {
		t.Fatalf("a shared gate no op binds verified: %v", err)
	}
}

// The few-row form and the waves are guarded by the target, not by their
// call sites: the few-row form only at the few-row class with the experts on
// the card, on a backend whose contract names it; the waves only at a prompt
// chunk's rows through an expert cache, and never over a bank with biases.
// A near miss (the combine reading another selection) is refused by both.
func TestMixtureRewritesAreGuardedByTheTarget(t *testing.T) {
	f := mixFacts(true, false)
	g := f.Mixture.Build()
	os := Match(g, MoEFewRow, MoEWaves)
	few, ok1 := Find(os, MoEFewRow.Name)
	waves, ok2 := Find(os, MoEWaves.Name)
	if !ok1 || !ok2 {
		t.Fatal("a mixture rewrite did not match")
	}
	at := func(api string, rows RowClass, res Residency) Target {
		return Target{API: api, Rows: rows, Residency: res}
	}
	for _, c := range []struct {
		o     Offered
		t     Target
		offer bool
	}{
		{few, at("ptx", RowsFew, ResidencyCard), true},
		{few, at("ptx", RowsFew, ResidencyCached), true},
		{few, at("msl", RowsFew, ResidencyCard), true},
		{few, at("ptx", RowsChunk, ResidencyCard), false},
		{few, at("ptx", RowsDecode, ResidencyCard), false},
		{few, at("ptx", RowsFew, ResidencyHost), false},
		{few, at("host", RowsFew, ResidencyCard), false},
		{waves, at("ptx", RowsChunk, ResidencyCached), true},
		{waves, at("spirv", RowsChunk, ResidencyCached), true},
		{waves, at("ptx", RowsFew, ResidencyCached), false},
		{waves, at("ptx", RowsChunk, ResidencyCard), false},
		{waves, at("ptx", RowsChunk, ResidencyHost), false},
	} {
		if err := c.o.Offer(c.t); (err == nil) != c.offer {
			t.Errorf("%s at %s: offered %v (%v), want %v", c.o.Rewrite.Name, c.t.Key(), err == nil, err, c.offer)
		}
	}
	// The verdict keys differ by residency.
	if few.Key(at("ptx", RowsFew, ResidencyCard)) == few.Key(at("ptx", RowsFew, ResidencyCached)) {
		t.Error("a verdict at one residency is keyed as another's")
	}
	// Biased banks: the waves refuse with the reason, the few-row form does not.
	fb := mixFacts(true, true)
	gb := fb.Mixture.Build()
	ob := Match(gb, MoEFewRow, MoEWaves)
	wb, _ := Find(ob, MoEWaves.Name)
	if err := wb.Offer(at("ptx", RowsChunk, ResidencyCached)); err != errWaveBias {
		t.Errorf("waves over biased banks: %v", err)
	}
	fbw, _ := Find(ob, MoEFewRow.Name)
	if err := fbw.Offer(at("ptx", RowsFew, ResidencyCard)); err != nil {
		t.Errorf("few-row over biased banks: %v", err)
	}
	// The near miss: the combine reads the gather's selection twice removed.
	g2 := f.Mixture.Build()
	for i := range g2.Ops {
		if g2.Ops[i].Kind == OpCombine {
			g2.Ops[i].In[1] = g2.In[0]
		}
	}
	for _, rw := range []*Rewrite{MoEFewRow, MoEWaves} {
		ds := Apply(g2, rw)
		if len(ds) != 1 || ds[0].Err == nil || !strings.Contains(ds[0].Err.Error(), "another selection") {
			t.Errorf("%s on a combine of another selection: %v", rw.Name, ds)
		}
	}
	// A dense graph has no mixture to match.
	if len(Match((&Dense{Block: 0, NEmbd: 64, NHead: 4, NKVHead: 2, HeadDim: 16, NFFN: 96, Eps: 1e-6, NRot: 16,
		Base: 1e6, Layout: RopeNorm, Scale: 0.25, Act: ActSiLU}).Build(), MoEFewRow, MoEWaves)) != 0 {
		t.Error("a mixture rewrite matched a dense block")
	}
}
