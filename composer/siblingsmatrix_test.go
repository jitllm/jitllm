package composer

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
)

// TestSiblingsCarryTheMatrixTiles forms a mixture's gate and up projections
// over a prompt's picked experts, on a device with a matrix unit and 32 KiB
// of workgroup memory (OLMoE's prompt block on an Apple GPU), up to the
// siblings pass; runs the two matrix kernels as one launch, as the pass does;
// and lowers it through the passes after. Each share's tile loads, scales and
// stores are carried into the launch -- a store that lost its tile mark
// stores a tile as a scalar, which no lowering takes -- and, one share's body
// running in a workgroup, the launch declares the workgroup memory one share
// does, not the sum.
func TestSiblingsCarryTheMatrixTiles(t *testing.T) {
	jt, ok := jlm.TypeOf(quant.Q4_K)
	if !ok {
		t.Fatal("Q4_K has no container type")
	}
	const K, N, E, P = 2048, 1024, 64, 8
	dev := Device{Kind: DeviceGPU, Lanes: 32, Units: 10, Shared: 32 << 10, Matrix: [3]int{16, 16, 16},
		ScaleRows: true, HalfPairs: true, VectorLoads: true}
	b := graph.New()
	x := b.In("x", K)
	sel := b.In("sel", P)
	b.G.Values[sel].Elem = graph.ElemU32
	in := b.Emit(graph.OpTile, []graph.ValueID{x}, []graph.Result{{Name: "experts.in", Width: P * K}}, nil, graph.Attrs{Span: K}, nil, nil)[0]
	formats := map[jlm.Role]jlm.Type{}
	for _, r := range []jlm.Role{jlm.RoleExpGateBank, jlm.RoleExpUpBank} {
		b.Out(b.Emit(graph.OpExpertMatVec, []graph.ValueID{in, sel}, []graph.Result{{Name: "y", Width: P * N}},
			[]graph.Binding{{Role: r, K: K, Rows: N, Sheets: E}}, graph.Attrs{}, nil, nil)[0])
		formats[r] = jt
	}
	r := Default()
	prof := r.Profiles[DeviceGPU]
	at := slices.IndexFunc(prof.Passes, func(p Pass) bool { return p.Name() == Siblings{}.Name() })
	if at < 0 {
		t.Fatal("the card's profile runs no siblings pass")
	}
	after := prof
	after.Passes = prof.Passes[at+1:]
	prof.Passes = prof.Passes[:at]
	r.Profiles[DeviceGPU] = prof
	ks, err := r.formKernels(Request{Graph: b.G, Formats: formats, Step: Step{Rows: 64, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
	if err != nil {
		t.Fatal(err)
	}
	var ms []*Kernel
	for _, k := range ks {
		if k.Matrix[0] > 0 && slices.ContainsFunc(k.Stores, func(s Store) bool { return s.Tile }) {
			ms = append(ms, k)
		}
	}
	if len(ms) != 2 {
		t.Fatalf("%d matrix kernels storing tiles formed, want the two projections'", len(ms))
	}
	lower := func(ks []*Kernel) int {
		out, err := after.on(dev).run(ks, dev)
		if err != nil {
			t.Fatal(err)
		}
		codes, err := r.Lower(out, dev)
		if err != nil {
			t.Fatal(err)
		}
		most := 0
		for _, c := range codes {
			if c.GPU == nil {
				continue
			}
			n := 0
			for _, o := range c.GPU.Ops {
				if o.Kind == ir.OpShared {
					n += 4 * int(o.Imm)
				}
			}
			most = max(most, n)
		}
		return most
	}
	k := Siblings{}.merge(ms)
	k.compact()
	two := lower([]*Kernel{k})
	one := lower(ms[:1])
	if one == 0 {
		t.Fatal("a projection alone declares no workgroup memory: the gate bounds nothing")
	}
	if two > one {
		t.Errorf("the two projections as one launch declare %d bytes of workgroup memory, past the %d one declares", two, one)
	}
	t.Logf("one projection %d bytes of workgroup memory, the two as one launch %d", one, two)
}
