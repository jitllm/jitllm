package composer

import (
	"maps"
	"testing"

	"github.com/jitllm/jitllm/graph"
)

type seenPass struct {
	p       Pass
	in, out *[][]*Kernel
}

func (a seenPass) Name() string { return a.p.Name() }

func (a seenPass) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	o, err := a.p.Apply(ks, d)
	*a.in, *a.out = append(*a.in, ks), append(*a.out, o)
	return o, err
}

// TestDeadStoresKeepAReadRunningBest forms a greedy argmax whose largest
// value is unread, handed back, and read by a later op. The argmax's fold
// carries its best value either way; DeadStores drops the value's store only
// where nothing reads it, and the index-only argmax stays one SplitExtreme
// and GroupExtreme take. The violation: a rule that ignores the graph's
// outputs or the part's loads drops a store the reader needs.
func TestDeadStoresKeepAReadRunningBest(t *testing.T) {
	const n = 151936
	dev := Device{Kind: DeviceGPU, Slots: 163840, Lanes: 32, Units: 80, Shared: 96 << 10, Regs: 65536}
	build := func(use string) *graph.Graph {
		g := (&graph.Sample{NVocab: n}).Build()
		best := g.Ops[0].Out[1]
		switch use {
		case "out":
			g.Out = append(g.Out, best)
		case "read":
			b := &graph.Builder{G: g}
			g.Out = append(g.Out, b.Emit(graph.OpMap, []graph.ValueID{best}, []graph.Result{{Name: "twice", Width: 1}}, nil,
				graph.Attrs{Fn: graph.MapExp}, nil, nil)[0])
		}
		return g
	}
	for _, tc := range []struct {
		use  string
		kept bool
	}{{"none", false}, {"out", true}, {"read", true}} {
		t.Run(tc.use, func(t *testing.T) {
			g := build(tc.use)
			best := g.Ops[0].Out[1]
			var in, out [][]*Kernel
			r := Default()
			r.Profiles = maps.Clone(r.Profiles)
			p := r.Profiles[DeviceGPU]
			var ps []Pass
			for _, q := range p.Passes {
				if q.Name() == "deadstores" {
					q = seenPass{q, &in, &out}
				}
				ps = append(ps, q)
			}
			p.Passes = ps
			r.Profiles[DeviceGPU] = p
			if _, _, err := r.Form(Request{Graph: g, Step: Step{Rows: 1, Page: 64, Keys: 1}, Device: dev}); err != nil {
				t.Fatal(err)
			}
			if len(in) == 0 || len(out) == 0 {
				t.Fatal("deadstores never ran")
			}
			stores := func(ks []*Kernel) (argmax *Kernel, ofBest int) {
				for _, k := range ks {
					for _, s := range k.Stores {
						if o := k.Bufs[s.Buf].Operand; !s.Barrier && !s.Run && o.Kind == OperandValue && o.Value == best {
							ofBest++
							argmax = k
						}
					}
				}
				return
			}
			if k, c := stores(in[0]); c == 0 || k == nil {
				t.Fatalf("before deadstores no kernel stores the best value")
			}
			k, c := stores(out[0])
			if tc.kept != (c > 0) {
				t.Fatalf("after deadstores %d stores of the best value, want kept=%v", c, tc.kept)
			}
			var arg *Kernel
			for _, k := range out[0] {
				if _, _, _, _, _, ok := extremeSplittable(k); ok {
					arg = k
				}
			}
			if arg == nil {
				t.Fatalf("after deadstores no argmax a split takes (best's kernel %v)", k != nil)
			}
		})
	}
}
