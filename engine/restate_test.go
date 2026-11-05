package engine

import (
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/model"
)

// A gate's violation is a rewrite of the graphs the model states: the model
// carries no switch a violation sets. A rewrite that reaches nothing leaves
// the violation without effect, and the gate it serves fails for not seeing
// it; one that does not verify fails where the graph is composed.

// restated states every block of m, and the head's collapse of the streams,
// as rewrite makes of its graph, until the returned undo. A graph the rewrite
// returns as it was given stays.
func restated(m *model.Model, rewrite func(g *graph.Graph) *graph.Graph) func() {
	s := m.FrontGraphs()
	set := *s
	set.Blocks = slices.Clone(s.Blocks)
	for li, g := range set.Blocks {
		if g != nil {
			set.Blocks[li] = rewrite(g)
		}
	}
	if s.Collapse != nil {
		set.Collapse = rewrite(s.Collapse)
	}
	m.Front.Mu.Lock()
	was := m.Front.Set
	m.Front.Set = &set
	m.Front.Mu.Unlock()
	return func() {
		m.Front.Mu.Lock()
		m.Front.Set = was
		m.Front.Mu.Unlock()
	}
}

// cloned is g with ops a rewrite may change in place: each op's inputs and
// the graph's outputs copied.
func cloned(g *graph.Graph) *graph.Graph {
	out := *g
	out.Values = slices.Clone(g.Values)
	out.Ops = slices.Clone(g.Ops)
	for i := range out.Ops {
		out.Ops[i].In = slices.Clone(out.Ops[i].In)
	}
	out.Out = slices.Clone(g.Out)
	return &out
}

// writer is the index of the op writing v.
func writer(g *graph.Graph, v graph.ValueID) int {
	for i, o := range g.Ops {
		if slices.Contains(o.Out, v) {
			return i
		}
	}
	return -1
}

// readers reports whether any op from index from on, or the graph's output,
// reads v.
func readers(g *graph.Graph, v graph.ValueID, from int) bool {
	for _, o := range g.Ops[from:] {
		if slices.Contains(o.In, v) {
			return true
		}
	}
	return slices.Contains(g.Out, v)
}

// redirect makes every op from index from on, and the graph's output, read
// to where they read v.
func redirect(g *graph.Graph, v, to graph.ValueID, from int) {
	for i := from; i < len(g.Ops); i++ {
		for j, in := range g.Ops[i].In {
			if in == v {
				g.Ops[i].In[j] = to
			}
		}
	}
	for j, o := range g.Out {
		if o == v {
			g.Out[j] = to
		}
	}
}

// insert emits ops through emit at the end of g and moves them to just after
// op k, returning emit's value and the index of the first op after them.
func insert(g *graph.Graph, k int, emit func(b *graph.Builder) graph.ValueID) (graph.ValueID, int) {
	b := &graph.Builder{G: g}
	n := len(g.Ops)
	v := emit(b)
	added := slices.Clone(g.Ops[n:])
	g.Ops = slices.Insert(g.Ops[:n], k+1, added...)
	return v, k + 1 + len(added)
}

// skip makes op k's readers read its input in, its result still read --
// summed, times zero, added to that input -- so the op's weights stay bound.
func skip(g *graph.Graph, k, in int) {
	y, x := g.Ops[k].Out[0], g.Ops[k].In[in]
	kept, from := insert(g, k, func(b *graph.Builder) graph.ValueID {
		z := b.Emit(graph.OpReduce, []graph.ValueID{y}, []graph.Result{{Name: "kept", Width: 1}}, nil,
			graph.Attrs{Fn: graph.MapAdd}, nil, nil)[0]
		z = b.Emit(graph.OpMap, []graph.ValueID{z}, []graph.Result{{Name: "kept.zero", Width: 1}}, nil,
			graph.Attrs{Fn: graph.MapMul, Const: 0}, nil, nil)[0]
		return b.Emit(graph.OpMap, []graph.ValueID{x, z}, []graph.Result{{Name: "kept.in", Width: g.Values[x].Width}}, nil,
			graph.Attrs{Fn: graph.MapAdd}, nil, nil)[0]
	})
	redirect(g, y, kept, from)
}

// zero makes op k's readers read its result times zero.
func zero(g *graph.Graph, k int) {
	y := g.Ops[k].Out[0]
	z, from := insert(g, k, func(b *graph.Builder) graph.ValueID {
		return b.Emit(graph.OpMap, []graph.ValueID{y}, []graph.Result{{Name: "zeroed", Width: g.Values[y].Width}}, nil,
			graph.Attrs{Fn: graph.MapMul, Const: 0}, nil, nil)[0]
	})
	redirect(g, y, z, from)
}

// written is the values of g each op match picks writes first, in op order.
func written(g *graph.Graph, match func(i int, o *graph.Op) bool) []graph.ValueID {
	var vs []graph.ValueID
	for i := range g.Ops {
		if len(g.Ops[i].Out) > 0 && match(i, &g.Ops[i]) {
			vs = append(vs, g.Ops[i].Out[0])
		}
	}
	return vs
}

// named matches an op writing a value of g named name.
func named(g *graph.Graph, name string) func(int, *graph.Op) bool {
	return func(_ int, o *graph.Op) bool { return len(o.Out) > 0 && g.Values[o.Out[0]].Name == name }
}

// binding matches an op binding role.
func binding(role jlm.Role) func(int, *graph.Op) bool {
	return func(_ int, o *graph.Op) bool {
		return slices.ContainsFunc(o.Weight, func(w graph.Binding) bool { return w.Role == role })
	}
}

// skipping is the rewrite skipping (skip) every op match picks, its readers
// reading its input in; g itself where it picks none.
func skipping(match func(g *graph.Graph) func(int, *graph.Op) bool, in int) func(*graph.Graph) *graph.Graph {
	return func(g *graph.Graph) *graph.Graph {
		vs := written(g, match(g))
		if len(vs) == 0 {
			return g
		}
		out := cloned(g)
		for _, v := range vs {
			skip(out, writer(out, v), in)
		}
		return out
	}
}

// zeroing is the rewrite zeroing (zero) every op match picks; g itself where
// it picks none.
func zeroing(match func(g *graph.Graph) func(int, *graph.Op) bool) func(*graph.Graph) *graph.Graph {
	return func(g *graph.Graph) *graph.Graph {
		vs := written(g, match(g))
		if len(vs) == 0 {
			return g
		}
		out := cloned(g)
		for _, v := range vs {
			zero(out, writer(out, v))
		}
		return out
	}
}

// editing is the rewrite applying edit to every op match picks; g itself
// where it picks none.
func editing(match func(g *graph.Graph) func(int, *graph.Op) bool, edit func(o *graph.Op)) func(*graph.Graph) *graph.Graph {
	return func(g *graph.Graph) *graph.Graph {
		hit := false
		for i := range g.Ops {
			hit = hit || match(g)(i, &g.Ops[i])
		}
		if !hit {
			return g
		}
		out := cloned(g)
		for i := range out.Ops {
			if match(out)(i, &out.Ops[i]) {
				edit(&out.Ops[i])
			}
		}
		return out
	}
}

// opsNamed, opsBinding and opsOfKind are the matches a rewrite takes.
func opsNamed(name string) func(*graph.Graph) func(int, *graph.Op) bool {
	return func(g *graph.Graph) func(int, *graph.Op) bool { return named(g, name) }
}

func opsBinding(role jlm.Role) func(*graph.Graph) func(int, *graph.Op) bool {
	return func(*graph.Graph) func(int, *graph.Op) bool { return binding(role) }
}

func opsOfKind(k graph.OpKind) func(*graph.Graph) func(int, *graph.Op) bool {
	return func(*graph.Graph) func(int, *graph.Op) bool {
		return func(_ int, o *graph.Op) bool { return o.Kind == k }
	}
}

// inBlocks narrows match to the blocks keep picks.
func inBlocks(keep func(li int) bool, match func(*graph.Graph) func(int, *graph.Op) bool) func(*graph.Graph) func(int, *graph.Op) bool {
	return func(g *graph.Graph) func(int, *graph.Op) bool {
		if g.Block < 0 || !keep(g.Block) {
			return func(int, *graph.Op) bool { return false }
		}
		return match(g)
	}
}

// modelWide narrows match to the graphs over every block (the head's
// collapse), no block's own.
func modelWide(match func(*graph.Graph) func(int, *graph.Op) bool) func(*graph.Graph) func(int, *graph.Op) bool {
	return func(g *graph.Graph) func(int, *graph.Op) bool {
		if g.Block >= 0 {
			return func(int, *graph.Op) bool { return false }
		}
		return match(g)
	}
}

// lastNamed matches an op writing a value named name that no other such op
// reads: a chain's last.
func lastNamed(name string) func(*graph.Graph) func(int, *graph.Op) bool {
	return func(g *graph.Graph) func(int, *graph.Op) bool {
		return func(_ int, o *graph.Op) bool {
			if len(o.Out) == 0 || g.Values[o.Out[0]].Name != name {
				return false
			}
			for _, r := range g.Ops {
				if len(r.Out) > 0 && g.Values[r.Out[0]].Name == name && slices.Contains(r.In, o.Out[0]) {
					return false
				}
			}
			return true
		}
	}
}

// reading matches an op of kind k reading a value named name.
func reading(k graph.OpKind, name string) func(*graph.Graph) func(int, *graph.Op) bool {
	return func(g *graph.Graph) func(int, *graph.Op) bool {
		return func(_ int, o *graph.Op) bool {
			return o.Kind == k && slices.ContainsFunc(o.In, func(v graph.ValueID) bool { return g.Values[v].Name == name })
		}
	}
}

// recurrent and attends pick m's blocks by kind.
func recurrent(m *model.Model) func(int) bool {
	return func(li int) bool { return m.Cfg.LayerKind(li).Recurrent() }
}

func attends(m *model.Model) func(int) bool {
	return func(li int) bool { return m.Cfg.LayerKind(li).Attends() }
}

// The pieces the gates' violations take out, each as the rewrite of the
// stated graphs that removes it.

// unbanked is a residual bank read as nothing: each mix over the bank is the
// running residual, its last row (a bank's mix's last FMA reads it).
func unbanked(m *model.Model) func() { return restated(m, skipping(lastNamed("bank.mix"), 1)) }

// headUnbanked is the head's mix over the streams read as the running
// residual alone, the blocks' mixes kept.
func headUnbanked(m *model.Model) func() {
	return restated(m, skipping(modelWide(lastNamed("bank.mix")), 1))
}

// rawBankScores scores a bank on its rows unnormed.
func rawBankScores(m *model.Model) func() { return restated(m, skipping(opsNamed("bank.normed"), 0)) }

// latentUnnormed skips the norm over a mixture's routed sum before its up
// projection.
func latentUnnormed(m *model.Model) func() {
	return restated(m, skipping(opsBinding(jlm.RoleFFNRoutedNorm), 0))
}

// latentAttnUngated leaves a latent attention's output ungated.
func latentAttnUngated(m *model.Model) func() { return restated(m, skipping(opsNamed("att.gated"), 1)) }

// recurrentUngated leaves a recurrent block's output ungated.
func recurrentUngated(m *model.Model) func() {
	return restated(m, skipping(inBlocks(recurrent(m), opsNamed("gated")), 1))
}

// recurrentSiLUGated gates a recurrent block's output by silu of its gate
// where it states a sigmoid.
func recurrentSiLUGated(m *model.Model) func() {
	silu := []graph.Term{
		{Fn: graph.MapSigmoid, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}}},
		{Fn: graph.MapMul, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}, {Kind: graph.ArgTerm, Index: 0}}},
		{Fn: graph.MapMul, Args: [3]graph.Arg{{Kind: graph.ArgTerm, Index: 1}, {Kind: graph.ArgIn, Index: 1}}},
	}
	return restated(m, editing(inBlocks(recurrent(m), opsNamed("gated")), func(o *graph.Op) { o.Attr.Expr = silu }))
}

// unrestarted makes a checkpoint block -- one whose attention sublayer
// restarts the residual, its sum stated and read by nothing -- add its
// attention: what reads the sublayer's output reads the sum.
func unrestarted(m *model.Model) func() {
	return restated(m, func(g *graph.Graph) *graph.Graph {
		if g.Block < 0 {
			return g
		}
		if _, ok := m.Cfg.ResCheckpoint(g.Block); !ok {
			return g
		}
		p := written(g, binding(jlm.RoleAttnOut))
		if len(p) != 1 {
			return g
		}
		for i, o := range g.Ops {
			if o.Kind != graph.OpAdd || len(o.Out) == 0 || !slices.Contains(o.In, p[0]) || readers(g, o.Out[0], 0) {
				continue
			}
			out := cloned(g)
			redirect(out, p[0], o.Out[0], i+1)
			return out
		}
		return g
	})
}

// selection edits every block selection's attributes.
func selection(m *model.Model, edit func(a *graph.Attrs)) func() {
	return restated(m, editing(opsOfKind(graph.OpSelect), func(o *graph.Op) { edit(&o.Attr) }))
}

// sharedSelection selects once, with group 0's head of the indexer's query,
// for every kv group.
func sharedSelection(m *model.Model) func() {
	c := m.Cfg
	return restated(m, func(g *graph.Graph) *graph.Graph {
		k := slices.IndexFunc(g.Ops, func(o graph.Op) bool { return o.Kind == graph.OpSelect })
		if k < 0 {
			return g
		}
		out := cloned(g)
		q := out.Ops[k].In[0]
		q0, _ := insert(out, k-1, func(b *graph.Builder) graph.ValueID {
			return b.Emit(graph.OpSlice, []graph.ValueID{q}, []graph.Result{{Name: "idx.q.0", Width: c.IdxHeadDim}}, nil,
				graph.Attrs{HeadDim: c.IdxHeads * c.IdxHeadDim, Width: c.IdxHeadDim}, nil, nil)[0]
		})
		sel := &out.Ops[k+1]
		sel.In[0], sel.Attr.NHead, sel.Attr.NKVHead = q0, 1, 1
		return out
	})
}

// halfDropped drops one half of every block that attends beside a second
// mixer: the attention's output (attn) or the mixer's projected one, times
// zero, still computed and bound.
func halfDropped(m *model.Model, attn bool) func() {
	return restated(m, func(g *graph.Graph) *graph.Graph {
		if g.Block < 0 || !m.Cfg.LayerKind(g.Block).Attends() || len(written(g, binding(jlm.RoleSSMOut))) == 0 {
			return g
		}
		out := cloned(g)
		if !attn {
			zero(out, writer(out, written(out, binding(jlm.RoleSSMOut))[0]))
			return out
		}
		// The attention's output enters the sublayer's norm, or its output
		// projection where the block stores no norm.
		k := slices.IndexFunc(out.Ops, func(o graph.Op) bool { return binding(jlm.RoleAttnSubNorm)(0, &o) })
		if k < 0 {
			k = slices.IndexFunc(out.Ops, func(o graph.Op) bool { return binding(jlm.RoleAttnOut)(0, &o) })
		}
		att := out.Ops[k].In[0]
		zero(out, writer(out, att))
		return out
	})
}
