package graph

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// deltaFacts is a gated delta sublayer as a container would state it: 16
// value heads of 128 over a row of 2048, the gate one matrix or two factors.
func deltaFacts(gate Act, rank int) Facts {
	d := Delta{Block: 3, NEmbd: 2048, VHeads: 16, VDim: 128, Eps: 1e-6, Gate: gate, GateRank: rank,
		NormW: Shape{K: 128}, OutRow: Shape{K: 2048, Rows: 2048}}
	inv := map[jlm.Role]Shape{
		jlm.RoleAttnQKV: {K: 2048, Rows: 8192}, jlm.RoleSSMBA: {K: 2048, Rows: 32},
		jlm.RoleSSMConv1d: {K: 4, Rows: 8192}, jlm.RoleSSMA: {K: 16}, jlm.RoleSSMDtBias: {K: 16},
		jlm.RoleSSMNorm: d.NormW, jlm.RoleSSMOut: d.OutRow,
	}
	switch {
	case gate == 0:
	case rank > 0:
		d.GateA, d.GateB = Shape{K: 2048, Rows: rank}, Shape{K: rank, Rows: 2048}
		inv[jlm.RoleSSMGA], inv[jlm.RoleSSMGB] = d.GateA, d.GateB
	default:
		d.GateA = Shape{K: 2048, Rows: 2048}
		inv[jlm.RoleAttnGate] = d.GateA
	}
	for _, r := range DeltaRuleRoles {
		if s, ok := inv[r]; ok {
			d.Rule = append(d.Rule, Binding{Role: r, K: s.K, Rows: s.Rows})
		}
	}
	return Facts{Delta: &d, Inventory: inv}
}

// The gated delta sublayer verifies under each gate form, and normgate is
// found there by its edges, once, legal; the dense graphs' head norms (q/k,
// read by the rotary) are no match. Near misses are refused with the reason:
// a second reader of the normalised row, an activation with no epilogue. A
// sublayer with no gate has nothing to match. The verifier refuses a head
// norm at the wrong value width and a rule with no recurrence.
func TestNormGateIsFoundByPattern(t *testing.T) {
	for _, c := range []struct {
		gate Act
		rank int
	}{{ActSiLU, 0}, {ActSigmoid, 0}, {ActSigmoid, 32}} {
		f := deltaFacts(c.gate, c.rank)
		g := f.Delta.Build()
		if err := Verify(g, f); err != nil {
			t.Fatalf("gate %d rank %d: %v", c.gate, c.rank, err)
		}
		ds := Apply(g, NormGate)
		if len(ds) != 1 || ds[0].Err != nil {
			t.Fatalf("gate %d rank %d: %v", c.gate, c.rank, ds)
		}
		r := ds[0].Region
		if g.Ops[r.Ops[0]].Weight[0].Role != jlm.RoleSSMNorm || g.Ops[r.Ops[1]].Attr.Act != c.gate {
			t.Fatalf("gate %d: region %v is not the output norm and its gate", c.gate, r.Ops)
		}
		if b := NormGate.Bounds(g, r); b.Launches != 1 || b.TrafficBytes != 8*16*128 {
			t.Fatalf("bounds %+v", b)
		}
		o := Offered{Rewrite: NormGate, Graph: g, Region: r}
		for _, api := range []string{"ptx", "spirv"} {
			for _, rows := range []RowClass{RowsDecode, RowsChunk} {
				if err := o.Offer(Target{API: api, Rows: rows}); err != nil {
					t.Fatalf("%s rows %d: %v", api, rows, err)
				}
			}
		}
		if err := o.Offer(Target{API: "host", Rows: RowsDecode}); err == nil {
			t.Fatal("offered on a backend with no contract")
		}
	}
	for _, qk := range []bool{false, true} {
		sf := stated(qk)
		if ds := Apply(sf.Config.Build(), NormGate); len(ds) != 0 {
			t.Fatalf("dense qk=%v: %v", qk, ds)
		}
	}
	if ds := Apply(deltaFacts(0, 0).Delta.Build(), NormGate); len(ds) != 0 {
		t.Fatalf("no gate: %v", ds)
	}

	// A second reader of the normalised row.
	f := deltaFacts(ActSiLU, 0)
	g := f.Delta.Build()
	for _, o := range g.Ops {
		if o.Kind == OpHeadNorm {
			probe := g.value("probe", g.Values[o.Out[0]].Width)
			g.op(Op{Kind: OpAdd, In: []ValueID{o.Out[0], o.Out[0]}, Out: []ValueID{probe}})
			break
		}
	}
	if ds := Apply(g, NormGate); len(ds) != 1 || ds[0].Err == nil || !strings.Contains(ds[0].Err.Error(), "readers") {
		t.Fatalf("second reader: %v", ds)
	}
	// An activation with no epilogue.
	g = f.Delta.Build()
	for i := range g.Ops {
		if g.Ops[i].Kind == OpActMul {
			g.Ops[i].Attr.Act = ActGELUTanh
		}
	}
	if ds := Apply(g, NormGate); len(ds) != 1 || ds[0].Err == nil || !strings.Contains(ds[0].Err.Error(), "epilogue") {
		t.Fatalf("gelu gate: %v", ds)
	}

	// The verifier's violations.
	bad := deltaFacts(ActSiLU, 0)
	g = bad.Delta.Build()
	wrong := *bad.Delta
	wrong.VDim = 64
	if err := Verify(g, Facts{Delta: &wrong, Inventory: bad.Inventory}); err == nil {
		t.Fatal("a head norm over heads of 128 verified against a value width of 64")
	}
	g = bad.Delta.Build()
	g.Ops[0].Reads, g.Ops[0].Writes = nil, nil
	if err := Verify(g, bad); err == nil || !strings.Contains(err.Error(), "recurrence") {
		t.Fatalf("a rule with no recurrence: %v", err)
	}
	g = bad.Delta.Build()
	g.Ops[0].Weight = g.Ops[0].Weight[1:]
	if err := Verify(g, bad); err == nil || !strings.Contains(err.Error(), "no op binds") {
		t.Fatalf("a rule that drops a weight: %v", err)
	}
	if err := Verify(bad.Delta.Build(), Facts{Config: stated(false).Config, Inventory: bad.Inventory}); err == nil {
		t.Fatal("a gated delta rule verified against dense facts")
	}
}
