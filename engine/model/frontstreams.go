package model

import (
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
}

// predictsStreams says the model's several residual streams are predicted
// and corrected (PredStreams), and nothing else carries a stream.
func (c *Config) predictsStreams() bool {
	return c.PredStreams > 1 && c.Streams() == c.PredStreams && c.HCMult <= 1 && !c.ResAttn()
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

// plainActAt is block li's stated activation without the sparsity the
// container states on its first NSparse blocks' FFN gates: the activation
// of the first block past them, nil where every block is sparse. A gate
// that is not an FFN's (the per-layer input's) reads it.
func (m *Model) plainActAt(li int) *jlm.Expr {
	c := m.Cfg
	if li < c.NSparse {
		if c.NSparse >= c.NLayer {
			return nil
		}
		li = c.NSparse
	}
	return m.container.Config().ActAt(li)
}
