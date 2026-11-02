package model

import (
	"math"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// compRow is where a compressing block's history row keeps its parts: the
// key, then the compressor's kv and gate, coff heads each (two where its
// windows overlap), then on a block selecting its entries the indexer's kv
// and gate, two of its heads each.
type compRow struct {
	hd, ihd, coff, width int
	kv, gate, ikv, igate int
}

// compRowAt is block li's row.
func (c *Config) compRowAt(li int) compRow {
	r := compRow{hd: c.HeadDim, ihd: c.IdxHeadDim, width: c.HeadDim}
	switch c.CompAt(li) {
	case jlm.CompHCA:
		r.coff = 1
	case jlm.CompCSA:
		r.coff = 2
	default:
		return r
	}
	r.kv = r.hd
	r.gate = r.kv + r.coff*r.hd
	r.width = r.gate + r.coff*r.hd
	if c.CompAt(li) == jlm.CompCSA {
		r.ikv = r.width
		r.igate = r.ikv + 2*r.ihd
		r.width = r.igate + 2*r.ihd
	}
	return r
}

// compressedRow is block li's history row over its key k: the compressor's
// kv and gate of n, each gate plus the position bias of the row's place in
// its window, then the indexer's where the block selects its entries.
func (m *Model) compressedRow(b *graph.Builder, li int, n, k graph.ValueID, matvec func(graph.ValueID, jlm.Role, int, ...jlm.Role) graph.ValueID) graph.ValueID {
	c := m.Cfg
	r, rate := c.compRowAt(li), c.CompRateAt(li)
	parts := []graph.ValueID{k}
	pair := func(kv, gate, bias jlm.Role, w int) {
		at := b.Emit(graph.OpEmbed, nil, []graph.Result{{Name: "comp.bias", Width: w}},
			[]graph.Binding{{Role: bias, K: w, Rows: rate}}, graph.Attrs{Positional: true, Span: rate}, nil, nil)[0]
		parts = append(parts, matvec(n, kv, w), b.Add(matvec(n, gate, w), at))
	}
	pair(jlm.RoleCompKV, jlm.RoleCompGate, jlm.RoleCompAPE, r.coff*r.hd)
	if r.ikv != 0 {
		pair(jlm.RoleIdxCompKV, jlm.RoleIdxCompGate, jlm.RoleIdxCompAPE, 2*r.ihd)
	}
	return b.Emit(graph.OpConcat, parts, []graph.Result{{Name: "k.row", Width: r.width}}, nil, graph.Attrs{}, nil, nil)[0]
}

// compressedAttend is block li's attention of q over its window of hist and
// the entries its compressor folds, under a's heads and the sinks. A row's
// entry is the window of the rate rows ending at it pooled column by column
// under the softmax of their gates (the windows overlapping by a rate where
// the row holds two halves), RMS-normed and turned by rope at the window's
// first position; the engine's entry descriptor names the rows that
// complete one. A block selecting its entries keeps the indexer's entry
// beside it and attends the entries the indexer's scores keep: each of its
// heads' query (of the query latent qIn, turned at the row's position)
// dotted with an entry's key through a ReLU, weighted per head by n's
// projection, summed.
func (m *Model) compressedAttend(b *graph.Builder, li int, n, qIn, q graph.ValueID, hist graph.StateRef, a, rope graph.Attrs,
	matvec func(graph.ValueID, jlm.Role, int, ...jlm.Role) graph.ValueID, sinks []jlm.Role) graph.ValueID {
	c := m.Cfg
	r, rate := c.compRowAt(li), c.CompRateAt(li)
	turn := func(v graph.ValueID, hd, shift int) graph.ValueID {
		ra := rope
		ra.HeadDim, ra.Offset, ra.Shift = hd, hd-rope.NRot, shift
		return b.Emit(graph.OpRope, []graph.ValueID{v}, []graph.Result{{Name: "comp.rope", Width: b.G.Values[v].Width}}, nil, ra, nil, nil)[0]
	}
	pool := func(off, gate, hd int, overlap bool, norm jlm.Role) graph.ValueID {
		e := b.ColPool(hist, graph.Attrs{HeadDim: hd, Span: rate, Offset: off, GateAt: gate, Width: r.width, Overlap: overlap})
		return turn(b.Norm(e, norm, c.RMSEps), hd, rate-1)
	}
	ent := pool(r.kv, r.gate, r.hd, r.coff == 2, jlm.RoleCompNorm)
	if r.coff == 2 && c.ds4Fault == ds4FaultNoOverlap {
		ent = pool(r.kv+r.hd, r.gate+r.hd, r.hd, false, jlm.RoleCompNorm)
	}
	selects := r.ikv != 0
	if selects {
		ie := pool(r.ikv, r.igate, r.ihd, true, jlm.RoleIdxCompNorm)
		ent = b.Emit(graph.OpConcat, []graph.ValueID{ent, ie}, []graph.Result{{Name: "ent.row", Width: r.hd + r.ihd}}, nil, graph.Attrs{}, nil, nil)[0]
	}
	entries := graph.StateRef{State: graph.StateEnt, Version: b.EntAppend(ent, 0)}
	a.EntWidth = b.G.Values[ent].Width
	var sel *graph.ValueID
	var iq, iw graph.ValueID
	if selects {
		h := c.IdxHeads
		iq = turn(matvec(qIn, jlm.RoleIdxQB, h*r.ihd), r.ihd, 0)
		iw = mapRow(b, "idx.w", graph.Attrs{Fn: graph.MapMul, Const: 1 / math.Sqrt(float64(h*r.ihd))}, nil, h, matvec(n, jlm.RoleIdxProj, h))
		keep := c.IdxTopK
		if c.ds4Fault == ds4FaultAllEntries {
			keep = math.MaxInt32
		}
		s := b.Select(iq, entries, graph.Attrs{NHead: h, NKVHead: 1, HeadDim: r.ihd, Offset: r.hd, Width: a.EntWidth, Span: 1,
			TopK: keep, Fn: graph.MapMax}, iw)
		sel = &s
	}
	if c.ds4Fault == ds4FaultWindowOnly {
		att := keepBound(b, b.AttendOf(q, hist, a, sinks...), ent)
		if selects {
			att = keepBound(b, keepBound(b, att, iq), iw)
		}
		return att
	}
	return b.AttendTwo(q, hist, entries, sel, a, sinks...)
}
