package model

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
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
	// mixed is a bank of checkpoints (mixesStreams), and bank the stream
	// the block banks its input into, written before any stream is read
	// (0 for none).
	mixed bool
	bank  graph.ValueID
	// gated is streams each sublayer reads through a low-rank gate
	// (gatesStreams): cur the streams after the attention's placement, w
	// the placement weights of the sublayer being stated.
	gated bool
	cur   []graph.ValueID
	w     graph.ValueID
}

// predictsStreams says the model's several residual streams are predicted
// and corrected (PredStreams), and nothing else carries a stream.
func (c *Config) predictsStreams() bool {
	return c.PredStreams > 1 && c.Streams() == c.PredStreams && c.HCMult <= 1 && !c.ResAttn()
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

// mixesStreams says the model's several residual streams are a bank of
// checkpoints, and nothing else carries a stream.
func (c *Config) mixesStreams() bool {
	return c.ResAttn() && c.Streams() == c.resStreams() && c.PredStreams == 0 && c.HCMult <= 1
}

// A block over several streams that each sublayer reads through a low-rank
// gate and writes into every stream at a weight of its own (HCLowRank): with
// xn the S streams side by side, each RMS-normed against its own row of the
// stored weight,
//
//	g   = sigmoid(U silu(D xn / S)) * xn
//	in  = the mean over the streams k of g_k
//	w   = 2 sigmoid(J xn / S)                  one weight a stream
//	x_k = x_k + w_k sublayer(in)
//
// The input rows' streams are S copies of the embedding, and the head reads
// the mix through a gate of its own with no placement and no norm.

// gatesStreams says the model's several residual streams are mixed through
// a low-rank gate, and nothing else carries a stream.
func (c *Config) gatesStreams() bool {
	return c.HCLowRank > 0 && c.Streams() == c.HCMult && c.PredStreams == 0 && !c.ResAttn()
}

// statesStreams says the front end states the model's residual streams.
func (c *Config) statesStreams() bool {
	return c.predictsStreams() || c.mixesStreams() || c.gatesStreams()
}

// gatedMix is the sublayer input the low-rank gate reads from the streams
// xs, and with an inject role stored the streams' placement weights (0
// for none).
func (m *Model) gatedMix(b *graph.Builder, f graph.Facts, xs []graph.ValueID, norm, down, up, inject jlm.Role) (mix, w graph.ValueID) {
	c := m.Cfg
	s, d := len(xs), c.NEmbd
	sd, r, inv := s*d, c.HCLowRank, 1/float64(s)
	cat := b.Emit(graph.OpConcat, xs, []graph.Result{{Name: "hc.streams", Width: sd}}, nil, graph.Attrs{}, nil, nil)[0]
	xn := b.Emit(graph.OpHeadNorm, []graph.ValueID{cat}, []graph.Result{{Name: "hc.normed", Width: sd}}, nil,
		graph.Attrs{Eps: c.RMSEps, HeadDim: d}, nil, nil)[0]
	xn = mapRow(b, "hc.xn", graph.Attrs{Fn: graph.MapMul}, []graph.Binding{{Role: norm, K: sd}}, sd, xn)
	lo := mapRow(b, "hc.lo", graph.Attrs{Fn: graph.MapMul, Const: inv}, nil, r, b.MatVec(xn, down, r))
	g := b.Expr("hc.gate", sd, []graph.Term{
		{Fn: graph.MapSigmoid, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}}},
		{Fn: graph.MapMul, Args: [3]graph.Arg{{Kind: graph.ArgTerm, Index: 0}, {Kind: graph.ArgIn, Index: 1}}},
	}, nil, b.MatVec(siluOf(b, "hc.act", lo), up, sd), xn)
	acc := b.Emit(graph.OpSlice, []graph.ValueID{g}, []graph.Result{{Name: "hc.g", Width: d}}, nil, graph.Attrs{Width: d}, nil, nil)[0]
	for k := 1; k < s; k++ {
		gk := b.Emit(graph.OpSlice, []graph.ValueID{g}, []graph.Result{{Name: "hc.g", Width: d}}, nil,
			graph.Attrs{Offset: k * d, Width: d}, nil, nil)[0]
		acc = b.Add(acc, gk)
	}
	mix = mapRow(b, "hc.mix", graph.Attrs{Fn: graph.MapMul, Const: inv}, nil, d, acc)
	if _, ok := f.Inventory[inject]; !ok {
		return mix, 0
	}
	w = mapRow(b, "hc.w.in", graph.Attrs{Fn: graph.MapMul, Const: inv}, nil, s, b.MatVec(xn, inject, s))
	w = mapRow(b, "hc.w.sig", graph.Attrs{Fn: graph.MapSigmoid}, nil, s, w)
	return mix, mapRow(b, "hc.w", graph.Attrs{Fn: graph.MapMul, Const: 2}, nil, s, w)
}

// placed is each stream of xs plus y at its placement weight, at weight
// one where the sublayer states none.
func (m *Model) placed(b *graph.Builder, xs []graph.ValueID, w, y graph.ValueID) []graph.ValueID {
	d := m.Cfg.NEmbd
	out := make([]graph.ValueID, len(xs))
	for k, x := range xs {
		if w == 0 {
			out[k] = b.Add(x, y)
			continue
		}
		out[k] = mapRow(b, "hc.placed", graph.Attrs{Fn: graph.MapFma}, nil, d, slice1(b, "hc.wk", w, k), y, x)
	}
	return out
}

// bankMix is the mix of the banked rows and the running residual x: each
// scored by the sum of role's vector times the row RMS-normed with no
// weight, the scores' softmax weighting the rows. With no bank it is x
// times its softmax of one, exactly x; a block that stores no score vector
// reads x.
func (m *Model) bankMix(b *graph.Builder, f graph.Facts, bank []graph.ValueID, x graph.ValueID, role jlm.Role) graph.ValueID {
	sh, ok := f.Inventory[role]
	if !ok {
		return x
	}
	c := m.Cfg
	d := c.NEmbd
	rows := append(slices.Clone(bank), x)
	w := []graph.Binding{{Role: role, K: sh.K, Rows: sh.Rows}}
	reduce := func(name string, v graph.ValueID, fn graph.MapFn) graph.ValueID {
		return b.Emit(graph.OpReduce, []graph.ValueID{v}, []graph.Result{{Name: name, Width: 1}}, nil, graph.Attrs{Fn: fn}, nil, nil)[0]
	}
	scores := make([]graph.ValueID, len(rows))
	for j, v := range rows {
		nv := v
		if c.k3Fault != k3FaultRawScores {
			nv = b.Emit(graph.OpHeadNorm, []graph.ValueID{v}, []graph.Result{{Name: "bank.normed", Width: d}}, nil,
				graph.Attrs{Eps: c.RMSEps, HeadDim: d}, nil, nil)[0]
		}
		scores[j] = reduce("bank.score", mapRow(b, "bank.scored", graph.Attrs{Fn: graph.MapMul}, w, d, nv), graph.MapAdd)
	}
	ns := len(rows)
	sc := b.Emit(graph.OpConcat, scores, []graph.Result{{Name: "bank.scores", Width: ns}}, nil, graph.Attrs{}, nil, nil)[0]
	ex := b.Expr("bank.exp", ns, []graph.Term{
		{Fn: graph.MapSub, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}, {Kind: graph.ArgIn, Index: 1}}},
		{Fn: graph.MapExp, Args: [3]graph.Arg{{Kind: graph.ArgTerm, Index: 0}}},
	}, nil, sc, reduce("bank.max", sc, graph.MapMax))
	p := mapRow(b, "bank.p", graph.Attrs{Fn: graph.MapDiv}, nil, ns, ex, reduce("bank.sum", ex, graph.MapAdd))
	acc := mapRow(b, "bank.mix", graph.Attrs{Fn: graph.MapMul}, nil, d, slice1(b, "bank.pj", p, 0), rows[0])
	for j := 1; j < ns; j++ {
		acc = mapRow(b, "bank.mix", graph.Attrs{Fn: graph.MapFma}, nil, d, slice1(b, "bank.pj", p, j), rows[j], acc)
	}
	if c.k3Fault == k3FaultNoBank {
		return keepBound(b, x, acc)
	}
	return acc
}

// bankAfter is the bank block li's FFN mixes: its attention's, with the
// stream it banks its input into read as that input.
func (st *streams) bankAfter(c *Config, li int) []graph.ValueID {
	bank := slices.Clone(st.in[1 : 1+c.resBank(li, true)])
	if k, ok := c.resCheckpoint(li); ok {
		bank[k-1] = st.in[0]
	}
	return bank
}

// banked writes the block's streams: the running residual y, the banked
// stream its input, every other stream its input itself, which no kernel
// writes.
func (m *Model) banked(b *graph.Builder, f graph.Facts, li int, st *streams, y graph.ValueID, has func(jlm.Role) bool) (*graph.Graph, Uncovered, error) {
	out := slices.Clone(st.in)
	out[0] = outScaled(b, has, y)
	if k, ok := m.Cfg.resCheckpoint(li); ok {
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
func (m *Model) expandsStreams() bool {
	c := m.Cfg
	return c.predictsStreams() || c.gatesStreams()
}

// streamExpand is the input rows' streams 1 and up from stream 0, x:
// copies of it, or its predictions matched to its magnitude.
func (m *Model) streamExpand(b *graph.Builder, x graph.ValueID) []graph.ValueID {
	c := m.Cfg
	s, d := c.Streams(), c.NEmbd
	var out []graph.ValueID
	if c.gatesStreams() {
		for range s - 1 {
			out = append(out, streamCopy(b, x))
		}
		return out
	}
	p := b.MatVec(x, jlm.RoleStreamExpand, (s-1)*d)
	for k := 1; k < s; k++ {
		pk := b.Emit(graph.OpSlice, []graph.ValueID{p}, []graph.Result{{Name: "expand", Width: d}}, nil,
			graph.Attrs{Offset: (k - 1) * d, Width: d}, nil, nil)[0]
		out = append(out, m.magMatched(b, pk, x))
	}
	return out
}

// streamExpandGraph is the input rows' streams 1 and up from stream 0:
// its one input stream 0, an output a stream past it; nil where the model
// predicts no streams and copies none. A bank of checkpoints has none: a
// block writes each checkpoint before any block reads it.
func (m *Model) streamExpandGraph() (*graph.Graph, error) {
	if !m.expandsStreams() {
		return nil, nil
	}
	f, _ := m.inventory(jlm.DenseBlock)
	b := graph.NewBlock(-1)
	x := b.In("x", m.Cfg.NEmbd)
	b.Out(m.streamExpand(b, x)...)
	if err := graph.Verify(b.G, f); err != nil {
		return nil, fmt.Errorf("the streams' expansion does not verify: %w", err)
	}
	return b.G, nil
}

// streamCollapseGraph is the row the head reads from every stream: an
// input a stream, one output; nil where the front end states no streams.
// A bank of checkpoints is mixed with the running residual by the head's
// score vector, and gated streams through the head's own gate.
func (m *Model) streamCollapseGraph() (*graph.Graph, error) {
	c := m.Cfg
	if !c.statesStreams() {
		return nil, nil
	}
	f, _ := m.inventory(jlm.DenseBlock)
	s, d := c.Streams(), c.NEmbd
	b := graph.NewBlock(-1)
	xs := make([]graph.ValueID, s)
	xs[0] = b.In("x", d)
	for k := 1; k < s; k++ {
		xs[k] = b.In(fmt.Sprintf("x.%d", k), d)
	}
	switch {
	case c.mixesStreams() && c.k3Fault == k3FaultNoHeadMix:
		b.Out(streamCopy(b, xs[0]))
	case c.mixesStreams():
		b.Out(m.bankMix(b, f, xs[1:1+c.resBankFinal()], xs[0], jlm.RoleOutputResScore))
	case c.gatesStreams():
		mix, _ := m.gatedMix(b, f, xs, jlm.RoleHCHeadNorm, jlm.RoleHCHeadDown, jlm.RoleHCHeadUp, 0)
		b.Out(mix)
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
	e := m.container.Config().ActAt(li)
	if e == nil {
		return nil
	}
	return unsparsified(e)
}

// unsparsified is e with each term keeping max(g - t, 0) of its gate g
// (operand 0), t a value reduced over the gate -- Gemma 3n's gaussian top-k,
// t the mean and SparseStd deviations -- read as the gate itself, and the
// terms nothing reads then dropped: e itself where it has no such term.
func unsparsified(e *jlm.Expr) *jlm.Expr {
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

// keepBound is x with v read and weighed by zero: a gate's violation that
// drops what v computes keeps the tensors v reads bound, as the container
// stores them. v must be finite.
func keepBound(b *graph.Builder, x, v graph.ValueID) graph.ValueID {
	z := b.Emit(graph.OpReduce, []graph.ValueID{v}, []graph.Result{{Name: "kept", Width: 1}}, nil,
		graph.Attrs{Fn: graph.MapAdd}, nil, nil)[0]
	z = mapRow(b, "kept.zero", graph.Attrs{Fn: graph.MapMul, Const: 0}, nil, 1, z)
	return mapRow(b, "kept.in", graph.Attrs{Fn: graph.MapAdd}, nil, b.G.Values[x].Width, x, z)
}
