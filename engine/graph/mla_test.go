package graph

import "testing"

// TestMLAAbsorbMatchesTheLatentAttention: mla-absorb is matched on the
// latent attention of an MLA block, legal there and decided at compile time,
// and nowhere on a dense block.
func TestMLAAbsorbMatchesTheLatentAttention(t *testing.T) {
	d := Dense{Block: 1, NEmbd: 64, NHead: 4, NKVHead: 4, HeadDim: 24, NFFN: 96, Eps: 1e-6, NRot: 8, Base: 1e4,
		Layout: RopeNorm, Scale: 0.2, Act: ActSiLU,
		MLA: &MLA{NHead: 4, HeadDim: 24, NRot: 8, KVLoraRank: 32, QLoraRank: 16, HeadDimV: 16, LatentEps: 1e-6}}
	g := d.Build()
	got := Apply(g, MLAAbsorb)
	if len(got) != 1 || got[0].Err != nil || got[0].When != WhenCompile {
		t.Fatalf("mla-absorb on a latent attention: %+v", got)
	}
	if g.Ops[got[0].Region.Ops[0]].Kind != OpAttendLatent {
		t.Fatalf("matched %s", g.Ops[got[0].Region.Ops[0]].Kind)
	}
	// The violation: an attention that reads no latent cache is refused.
	bad := *g
	bad.Ops = append([]Op(nil), g.Ops...)
	o := &bad.Ops[got[0].Region.Ops[0]]
	o.Reads = []StateRef{{"rec", 1}}
	if err := MLAAbsorb.Legal(&bad, got[0].Region); err == nil {
		t.Fatal("mla-absorb is legal on an attention reading no latent cache")
	}
	d.MLA = nil
	if len(Apply(d.Build(), MLAAbsorb)) != 0 {
		t.Fatal("mla-absorb matched a dense attention")
	}
}
