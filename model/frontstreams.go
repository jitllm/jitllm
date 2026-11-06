package model

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// A block over several residual streams that it predicts from each other
// and corrects by what it did to the active one (Config.PredStreams): the
// graph reads each stream as an input of its own, a row of n_embd, and
// writes each as an output, stream k's output over stream k's input. The
// call site binds them at their offsets in main's stream-major residual.
//
//	w      = tanh(R . norm(x_0) / n_embd)              the router's weights
//	pred_k = x_k + sum_j c_kj x_j,  c = T_pred^T w     the predictions
//	a      = pred_0, through the block, the attention's residual joined by
//	         the side branch: (a + attn + s) / sqrt(2), s = n + norm(U D n)
//	out_k  = pred_k + (T_corr^T w' + 1)_k (y - pred_0), w' the router over y
//	out_k += norm(P (act(G (out_0 * scale)) * pl))     for k >= 1

// streams is a block's residual streams as its graph reads them: the
// inputs, their predictions and the side branch beside the attention's
// residual (0 for none).
type streams struct {
	in, pred []graph.ValueID
	side     graph.ValueID
	// mixed is streams the converter states (mixesStreams), and bank the
	// stream the block banks its input into, written before any stream is
	// read (0 for none).
	mixed bool
	bank  graph.ValueID
	// cur is the streams after a sublayer that writes every one, and ex
	// the values the sublayer's read exports to its writes.
	cur []graph.ValueID
	ex  []graph.ValueID
}

// ResidStreams says the residual is every stream, which the input rows
// expand into (jlm.Streams.Expand): a row's hidden state is all of them, as
// the head's mix reads them, not the output norm's row.
func (c *Config) ResidStreams() bool { return c.stated != nil && len(c.stated.Expand) != 0 }

// predictsStreams says the model's several residual streams are predicted
// and corrected (PredStreams), and nothing else carries a stream.
func (c *Config) PredictsStreams() bool {
	return c.PredStreams > 1 && c.Streams() == c.PredStreams && c.HCMult <= 1 && c.statedStreams() <= 1
}

// A block over a bank of checkpoints (Config.ResAttn): stream 0 is the
// running residual and stream 1+j checkpoint j. Each sublayer reads a
// softmax mix of the checkpoints banked so far and the running residual,
// each scored by its stored vector's dot with the stream RMS-normed with no
// weight; a checkpoint block banks its input first and restarts the running
// residual from its attention's output.
//
//	in_a = mix(x_1..x_nb, x_0)                     the attention's input
//	x_0' = attn(in_a), or x_0 + attn(in_a)          a checkpoint block, or not
//	in_f = mix(x_1..x_nb', x_0')                    with the input banked
//	x_0'' = x_0' + ffn(in_f)

// mixesStreams says the model's several residual streams are the ones the
// converter states, and nothing else carries a stream.
func (c *Config) mixesStreams() bool {
	return c.statedStreams() > 1 && c.Streams() == c.statedStreams() && c.PredStreams == 0
}

// streamRead is the input u reads -- its program over the streams it names:
// stream k as it is now (xs), stream 0 as the block received it (in), or the
// running residual (run) -- or run where it states none, and the values its
// program exports.
func (m *Model) streamRead(b *graph.Builder, f graph.Facts, u *jlm.SubStreams, xs []graph.ValueID, in, run graph.ValueID) (graph.ValueID, []graph.ValueID, error) {
	if u.Read == nil {
		return run, nil, nil
	}
	ins := make([]graph.ValueID, len(u.From))
	for j, k := range u.From {
		switch k {
		case jlm.StreamIn:
			ins[j] = in
		case jlm.StreamRun:
			ins[j] = run
		default:
			ins[j] = xs[k]
		}
	}
	if len(u.Read.Ops) == 0 {
		return ins[u.Read.Out], nil, nil
	}
	vals, err := lowerValues(b, nil, u.Read, ins, 1, f)
	if err != nil {
		return 0, nil, err
	}
	ex := make([]graph.ValueID, len(u.Exports))
	for i, e := range u.Exports {
		ex[i] = vals[e]
	}
	return vals[u.Read.Out], ex, nil
}

// streamWrites is every stream after a sublayer whose output is y: u's
// write of stream k over it as it is (xs[k]), y and the read's exports, the
// stream as it is where u writes none.
func (m *Model) streamWrites(b *graph.Builder, f graph.Facts, u *jlm.SubStreams, xs []graph.ValueID, y graph.ValueID, ex []graph.ValueID) ([]graph.ValueID, error) {
	out := slices.Clone(xs)
	for k, p := range u.Writes {
		if p == nil {
			continue
		}
		v, err := lowerStated(b, p, append([]graph.ValueID{xs[k], y}, ex...), f)
		if err != nil {
			return nil, fmt.Errorf("stream %d's write: %w", k, err)
		}
		out[k] = v
	}
	return out, nil
}

// streamJoin is the running residual after a sublayer: u's join over run
// and the sublayer's output y, or their sum where it states none.
func (m *Model) streamJoin(b *graph.Builder, f graph.Facts, u *jlm.SubStreams, run, y graph.ValueID, sum func(x, y graph.ValueID) graph.ValueID) (graph.ValueID, error) {
	if u.Join == nil {
		return sum(run, y), nil
	}
	return lowerStated(b, u.Join, []graph.ValueID{run, y}, f)
}

// lowerStated lowers a stated program over in; one of no op returns the
// input it names.
func lowerStated(b *graph.Builder, p *jlm.Program, in []graph.ValueID, f graph.Facts) (graph.ValueID, error) {
	if len(p.Ops) == 0 {
		return in[p.Out], nil
	}
	return LowerProgram(b, nil, p, in, 1, f)
}

// statesStreams says the front end states the model's residual streams.
func (c *Config) StatesStreams() bool {
	return c.PredictsStreams() || c.mixesStreams()
}

// banked writes the block's streams: the running residual y, the banked
// stream its input, every other stream its input itself, which no kernel
// writes.
func (m *Model) banked(b *graph.Builder, f graph.Facts, li int, st *streams, y graph.ValueID, has func(jlm.Role) bool) (*graph.Graph, Uncovered, error) {
	out := slices.Clone(st.in)
	out[0] = outScaled(b, has, y)
	if k := m.Cfg.stated.Blocks[li].Bank; k != 0 {
		out[k] = st.bank
	}
	b.Out(out...)
	return verified(b, f)
}

// streamCopy is a row of v as a value of its own.
func streamCopy(b *graph.Builder, v graph.ValueID) graph.ValueID {
	w := b.G.Values[v].Width
	return b.Emit(graph.OpSlice, []graph.ValueID{v}, []graph.Result{{Name: "stream", Width: w}}, nil, graph.Attrs{Width: w}, nil, nil)[0]
}

// slice1 is element i of each row of v.
func slice1(b *graph.Builder, name string, v graph.ValueID, i int) graph.ValueID {
	return b.Emit(graph.OpSlice, []graph.ValueID{v}, []graph.Result{{Name: name, Width: 1}}, nil,
		graph.Attrs{Offset: i, Width: 1}, nil, nil)[0]
}

// routed is the router's weight for each stream over the row x.
func (m *Model) routed(b *graph.Builder, x graph.ValueID) graph.ValueID {
	c := m.Cfg
	h := b.Norm(x, jlm.RoleStreamRouterNorm, c.RMSEps)
	h = mapRow(b, "router.in", graph.Attrs{Fn: graph.MapMul, Const: 1 / float64(c.NEmbd)}, nil, c.NEmbd, h)
	r := b.MatVec(h, jlm.RoleStreamRouter, c.PredStreams)
	return mapRow(b, "router.w", graph.Attrs{Fn: graph.MapTanh}, nil, c.PredStreams, r)
}

// coefs is the table role's rows, each w wide, summed at the router's
// weights over x.
func (m *Model) coefs(b *graph.Builder, f graph.Facts, x graph.ValueID, role jlm.Role, w int) graph.ValueID {
	rw := m.routed(b, x)
	sh := f.Inventory[role]
	bind := []graph.Binding{{Role: role, K: sh.K, Rows: sh.Rows}}
	var acc graph.ValueID
	for i := range m.Cfg.PredStreams {
		t := mapRow(b, "coef", graph.Attrs{Fn: graph.MapMul, Offset: i * w, Width: w}, bind, w, slice1(b, "router.wi", rw, i))
		if i == 0 {
			acc = t
			continue
		}
		acc = b.Add(acc, t)
	}
	return acc
}

// predicted is each stream's prediction from every stream.
func (m *Model) predicted(b *graph.Builder, f graph.Facts, xs []graph.ValueID) []graph.ValueID {
	s, d := len(xs), m.Cfg.NEmbd
	c := m.coefs(b, f, xs[0], jlm.RoleStreamPredictCoef, s*s)
	pred := make([]graph.ValueID, s)
	for k := range s {
		acc := xs[k]
		for j := range s {
			acc = mapRow(b, "pred", graph.Attrs{Fn: graph.MapFma}, nil, d, slice1(b, "pred.c", c, k*s+j), xs[j], acc)
		}
		pred[k] = acc
	}
	return pred
}

// lowRank is the side branch beside the attention's residual over its
// normed row n: n plus the normed up projection of its down projection.
func (m *Model) lowRank(b *graph.Builder, f graph.Facts, n graph.ValueID) graph.ValueID {
	c := m.Cfg
	h := b.MatVec(n, jlm.RoleLowRankIn, f.Inventory[jlm.RoleLowRankIn].Rows)
	return b.Add(n, b.Norm(b.MatVec(h, jlm.RoleLowRankOut, c.NEmbd), jlm.RoleLowRankNorm, c.RMSEps))
}

// corrected writes each stream's prediction corrected by what the block
// did to the active one (its output y), with the per-layer input gated by
// the corrected active stream at its scale into every other stream, and
// the block's output scale on the active stream.
func (m *Model) corrected(b *graph.Builder, f graph.Facts, li int, st *streams, y graph.ValueID, has func(jlm.Role) bool) (*graph.Graph, Uncovered, error) {
	c := m.Cfg
	s, d := len(st.in), c.NEmbd
	cc := mapRow(b, "corr", graph.Attrs{Fn: graph.MapAdd, Const: 1}, nil, s, m.coefs(b, f, y, jlm.RoleStreamCorrectCoef, s))
	inn := mapRow(b, "innov", graph.Attrs{Fn: graph.MapSub}, nil, d, y, st.pred[0])
	out := make([]graph.ValueID, s)
	for k := range s {
		out[k] = mapRow(b, "corrected", graph.Attrs{Fn: graph.MapFma}, nil, d, slice1(b, "corr.c", cc, k), inn, st.pred[k])
	}
	if c.PLEDim != 0 {
		g := mapRow(b, "ple.from", graph.Attrs{Fn: graph.MapMul}, []graph.Binding{{Role: jlm.RoleStreamCorrectScale, K: d}}, d, out[0])
		o, why, err := m.perLayerOf(b, f, li, g)
		if err != nil {
			return nil, why, err
		}
		for k := 1; k < s; k++ {
			out[k] = b.Add(out[k], o)
		}
	}
	out[0] = outScaled(b, has, out[0])
	b.Out(out...)
	return verified(b, f)
}

// The input rows' streams and the head's row (Config.PredStreams): stream 0
// is the scaled embedding and stream k its projection matched to its
// magnitude; the head reads the mean of stream 0 and each stream's
// projection matched to stream 0's.
//
//	mag(p, r) = p * sqrt(mean(r^2)) / sqrt(max(mean(p^2), eps))
//	x_k = mag(E_k x_0, x_0)                          the expansion
//	y   = (x_0 + sum_k mag(U_k x_k, x_0)) / S        the collapse

// meanSquare is the mean of the squares of each row of v, a row of one.
func meanSquare(b *graph.Builder, name string, v graph.ValueID, w int) graph.ValueID {
	sq := mapRow(b, name+".sq", graph.Attrs{Fn: graph.MapMul}, nil, w, v, v)
	s := b.Emit(graph.OpReduce, []graph.ValueID{sq}, []graph.Result{{Name: name + ".sum", Width: 1}}, nil,
		graph.Attrs{Fn: graph.MapAdd}, nil, nil)[0]
	return mapRow(b, name, graph.Attrs{Fn: graph.MapMul, Const: 1 / float64(w)}, nil, 1, s)
}

// magMatched is p matched to the magnitude of the row r.
func (m *Model) magMatched(b *graph.Builder, p, r graph.ValueID) graph.ValueID {
	d := m.Cfg.NEmbd
	num := mapRow(b, "mag.ref", graph.Attrs{Fn: graph.MapSqrt}, nil, 1, meanSquare(b, "mag.refms", r, d))
	den := mapRow(b, "mag.floor", graph.Attrs{Fn: graph.MapMax, Const: magEps}, nil, 1, meanSquare(b, "mag.ms", p, d))
	den = mapRow(b, "mag.root", graph.Attrs{Fn: graph.MapSqrt}, nil, 1, den)
	k := mapRow(b, "mag.k", graph.Attrs{Fn: graph.MapDiv}, nil, 1, num, den)
	return mapRow(b, "mag", graph.Attrs{Fn: graph.MapMul}, nil, d, k, p)
}

// expandsStreams says the input rows' streams past the first are made from
// it: predicted from it, or copies of it.
func (m *Model) ExpandsStreams() bool {
	c := m.Cfg
	return c.PredictsStreams() || c.stated != nil && len(c.stated.Expand) != 0
}

// streamExpand is the input rows' streams 1 and up from stream 0, x:
// copies of it, or its predictions matched to its magnitude.
func (m *Model) streamExpand(b *graph.Builder, f graph.Facts, x graph.ValueID) ([]graph.ValueID, error) {
	c := m.Cfg
	s, d := c.Streams(), c.NEmbd
	var out []graph.ValueID
	if c.stated != nil && len(c.stated.Expand) != 0 {
		for k, p := range c.stated.Expand {
			if p == nil {
				out = append(out, streamCopy(b, x))
				continue
			}
			v, err := lowerStated(b, p, []graph.ValueID{x}, f)
			if err != nil {
				return nil, fmt.Errorf("stream %d's expansion: %w", k+1, err)
			}
			out = append(out, v)
		}
		return out, nil
	}
	p := b.MatVec(x, jlm.RoleStreamExpand, (s-1)*d)
	for k := 1; k < s; k++ {
		pk := b.Emit(graph.OpSlice, []graph.ValueID{p}, []graph.Result{{Name: "expand", Width: d}}, nil,
			graph.Attrs{Offset: (k - 1) * d, Width: d}, nil, nil)[0]
		out = append(out, m.magMatched(b, pk, x))
	}
	return out, nil
}

// streamExpandGraph is the input rows' streams 1 and up from stream 0:
// its one input stream 0, an output a stream past it; nil where the model
// predicts no streams and copies none. A bank of checkpoints has none: a
// block writes each checkpoint before any block reads it.
func (m *Model) streamExpandGraph() (*graph.Graph, error) {
	if !m.ExpandsStreams() {
		return nil, nil
	}
	f, _ := m.Inventory(jlm.DenseBlock)
	b := graph.NewBlock(-1)
	x := b.In("x", m.Cfg.NEmbd)
	xs, err := m.streamExpand(b, f, x)
	if err != nil {
		return nil, err
	}
	b.Out(xs...)
	if err := graph.Verify(b.G, f); err != nil {
		return nil, fmt.Errorf("the streams' expansion does not verify: %w", err)
	}
	return b.G, nil
}

// streamCollapseGraph is the row the head reads from every stream: an
// input a stream, one output; nil where the front end states no streams.
// A bank of checkpoints is mixed with the running residual by the head's
// score vector, and gated streams through the head's own gate.
func (m *Model) streamCollapseGraph() (*graph.Graph, error) { return m.streamCollapseGraphAt(-1) }

// streamCollapseGraphAt is streamCollapseGraph over block li's head roles:
// a prediction block's own head mix, which its draft's logits read; -1 is
// the model's head.
func (m *Model) streamCollapseGraphAt(li int) (*graph.Graph, error) {
	c := m.Cfg
	if !c.StatesStreams() {
		return nil, nil
	}
	f, _ := m.Inventory(jlm.DenseBlock)
	if li >= 0 {
		f, _ = m.Inventory(int32(li))
	}
	s, d := c.Streams(), c.NEmbd
	b := graph.NewBlock(li)
	xs := make([]graph.ValueID, s)
	xs[0] = b.In("x", d)
	for k := 1; k < s; k++ {
		xs[k] = b.In(fmt.Sprintf("x.%d", k), d)
	}
	switch {
	case c.mixesStreams():
		y, _, err := m.streamRead(b, f, &c.stated.Head, xs, xs[0], xs[0])
		if err != nil {
			return nil, fmt.Errorf("the streams' stated head read: %w", err)
		}
		b.Out(y)
	default:
		acc := xs[0]
		for k := 1; k < s; k++ {
			h := b.MatVec(xs[k], jlm.RoleStreamCollapse1+jlm.Role(k-1), d)
			acc = b.Add(acc, m.magMatched(b, h, xs[0]))
		}
		b.Out(mapRow(b, "collapsed", graph.Attrs{Fn: graph.MapMul, Const: 1 / float64(s)}, nil, d, acc))
	}
	if err := graph.Verify(b.G, f); err != nil {
		return nil, fmt.Errorf("the streams' collapse does not verify: %w", err)
	}
	return b.G, nil
}

// plainActAt is block li's stated activation without the sparsity the
// container states on its FFN's gate (unsparsified): the activation a gate
// that is not an FFN's (the per-layer input's) reads.
func (m *Model) plainActAt(li int) *jlm.Expr {
	e := m.Container.Config().ActAt(li)
	if e == nil {
		return nil
	}
	return Unsparsified(e)
}

// unsparsified is e with each term keeping max(g - t, 0) of its gate g
// (operand 0), t a value reduced over the gate -- Gemma 3n's gaussian top-k,
// t the mean and SparseStd deviations -- read as the gate itself, and the
// terms nothing reads then dropped: e itself where it has no such term.
func Unsparsified(e *jlm.Expr) *jlm.Expr {
	gate := jlm.Operand(0)
	ts := slices.Clone(e.Terms)
	// overGate reports whether a reads a reduction over the gate, gated
	// whether it reads the gate at all.
	var overGate, gated func(a jlm.Arg) bool
	gated = func(a jlm.Arg) bool {
		if a == gate {
			return true
		}
		if a.Kind != jlm.ArgTerm {
			return false
		}
		return slices.ContainsFunc(ts[a.Index].Args[:], gated)
	}
	overGate = func(a jlm.Arg) bool {
		if a.Kind != jlm.ArgTerm {
			return false
		}
		t := ts[a.Index]
		if t.Reduce && gated(t.Args[0]) {
			return true
		}
		return slices.ContainsFunc(t.Args[:], overGate)
	}
	cut := false
	for k, t := range ts {
		if t.Fn != jlm.FnMax || t.Reduce || t.Args[1] != jlm.Num(0) || t.Args[0].Kind != jlm.ArgTerm {
			continue
		}
		s := ts[t.Args[0].Index]
		if s.Fn != jlm.FnSub || s.Reduce || s.Args[0] != gate || !overGate(s.Args[1]) {
			continue
		}
		at := jlm.Arg{Kind: jlm.ArgTerm, Index: uint32(k)}
		for j := k + 1; j < len(ts); j++ {
			for i, a := range ts[j].Args {
				if a == at {
					ts[j].Args[i] = gate
				}
			}
		}
		cut = true
	}
	if !cut {
		return e
	}
	live := make([]bool, len(ts))
	live[len(ts)-1] = true
	for k := len(ts) - 1; k >= 0; k-- {
		if !live[k] {
			continue
		}
		for _, a := range ts[k].Args {
			if a.Kind == jlm.ArgTerm {
				live[a.Index] = true
			}
		}
	}
	at := make([]uint32, len(ts))
	out := &jlm.Expr{}
	for k, t := range ts {
		if !live[k] {
			continue
		}
		for i, a := range t.Args {
			if a.Kind == jlm.ArgTerm {
				t.Args[i].Index = at[a.Index]
			}
		}
		at[k] = uint32(len(out.Terms))
		out.Terms = append(out.Terms, t)
	}
	return out
}
