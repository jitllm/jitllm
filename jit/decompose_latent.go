package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/ir"
)

// decomposeLatentAppend writes each row's latent and rotary key as one row
// of the key pool, the latent first: the value is the row's latent prefix,
// read from the same pool (decomposeAttendLatent).
func decomposeLatentAppend(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 2 || c.Step.Page <= 0 {
		return nil, fmt.Errorf("a latent append of %d values over pages of %d", len(o.In), c.Step.Page)
	}
	U := ir.U32
	page := c.Step.Page
	lat, rot := c.width(o.In[0]), c.width(o.In[1])
	w := lat + rot
	var prims []Prim
	for i, from := range []int{0, lat} {
		v := o.In[i]
		sh := c.rows(v)
		if sh[1] == 0 {
			continue
		}
		var x exprs
		row := func(k int) int {
			return x.load(1, x.op(OpAdd, U, x.op(OpMul, U, x.coord(0), x.u(rowWords)), x.u(uint32(k))))
		}
		d := x.coord(1)
		tab, pos := row(rowTab), row(rowWrite)
		pid := x.load(2, x.op(OpAdd, U, tab, x.op(OpDiv, U, pos, x.u(uint32(page)))))
		off := x.op(OpRem, U, pos, x.u(uint32(page)))
		at := x.op(OpAdd, U, x.u(uint32(from)), d)
		val := x.at(0, sh, x.coord(0), d)
		prims = append(prims, Prim{Kind: PrimMap, Shape: sh, In: []Operand{c.value(v), tableRows, tablePages}, Expr: x.e,
			Facts:  Facts{Extents: map[int]int{1: c.Step.Rows * rowWords, 2: c.pages()}},
			Stores: []PrimStore{{Out: tableKeys, Ax: []int{pid, off, at}, Ext: []int{c.pages(), page, w}, Value: val}}})
	}
	return prims, nil
}

// decomposeAttendLatent is the absorbed latent attention (graph.MLAAbsorb):
// each head's unrotated query through its sheet of the k up-projection into
// the latent's width, the rotary query after it, attention over the key
// pool's latent rows as one shared key head with the value each row's
// latent prefix -- the attention reads the whole row as the value, the
// work passes taking one head width, and the up-projection reads its
// latent prefix -- and each head's result through its sheet of the v
// up-projection.
func decomposeAttendLatent(o graph.Op, c *Context) ([]Prim, error) {
	a := o.Attr
	if len(o.In) != 2 || len(o.Out) != 1 || len(o.Weight) != 2 || a.NHead <= 0 || c.Step.Page <= 0 || c.Step.Keys <= 0 {
		return nil, fmt.Errorf("a latent attention of %d operands, %d weights", len(o.In), len(o.Weight))
	}
	U, F := ir.U32, ir.F32
	R, H := c.Step.Rows, a.NHead
	kb, vb := o.Weight[0], o.Weight[1]
	nope, lat, dv := kb.K, kb.Rows, vb.Rows
	rot := c.width(o.In[1]) / H
	if kb.Sheets != H || vb.Sheets != H || vb.K != lat || c.width(o.In[0]) != H*nope || c.width(o.In[1]) != H*rot ||
		c.width(o.Out[0]) != H*dv {
		return nil, fmt.Errorf("a latent attention of %d heads of %d over a latent of %d", H, nope, lat)
	}
	kbo, err := c.weight(kb)
	if err != nil {
		return nil, err
	}
	vbo, err := c.weight(vb)
	if err != nil {
		return nil, err
	}
	out := c.value(o.Out[0])
	tmp := func(name string) Operand {
		return Operand{Kind: OperandScratch, Name: out.Name + "." + name, Temp: true}
	}
	qAbs, qFull, oLat := tmp("q_abs"), tmp("q_full"), tmp("o_lat")
	w := lat + rot

	var qx exprs
	qr, qh, ql, qn := qx.coord(0), qx.coord(1), qx.coord(2), qx.coord(3)
	qin := []Operand{c.value(o.In[0])}
	qf := Facts{Extents: map[int]int{0: R * H * nope}}
	kv, err := weightElement(&qx, kbo, H*lat, nope, &qin, &qf, qx.op(OpAdd, U, qx.op(OpMul, U, qh, qx.u(uint32(lat))), ql), qn)
	if err != nil {
		return nil, err
	}
	qx.op(OpMul, F, kv, qx.at(0, []int{R, H * nope}, qr, qx.op(OpAdd, U, qx.op(OpMul, U, qh, qx.u(uint32(nope))), qn)))
	prims := []Prim{{Kind: PrimReduce, Shape: []int{R, H, lat, nope}, In: qin, Out: qAbs, Combine: OpAdd, Expr: qx.e, Facts: qf}}

	query := qAbs
	if rot > 0 {
		var fx exprs
		fr, fh, fe := fx.coord(0), fx.coord(1), fx.coord(2)
		ab := fx.at(0, []int{R, H, lat}, fr, fh, fx.op(OpMin, U, fe, fx.u(uint32(lat-1))))
		pe := fx.at(1, []int{R, H * rot}, fr, fx.op(OpAdd, U, fx.op(OpMul, U, fh, fx.u(uint32(rot))),
			fx.op(OpSub, U, fx.op(OpMax, U, fe, fx.u(uint32(lat))), fx.u(uint32(lat)))))
		fx.sel(F, fx.op(OpLess, ir.Pred, fe, fx.u(uint32(lat))), ab, pe)
		prims = append(prims, Prim{Kind: PrimMap, Shape: []int{R, H, w}, In: []Operand{qAbs, c.value(o.In[1])}, Out: qFull, Expr: fx.e})
		query = qFull
	}

	shared := graph.Op{Attr: graph.Attrs{NHead: H, NKVHead: 1, HeadDim: w, Scale: a.Scale}}
	prims = append(prims, attend(shared, c, query, oLat, attendGeom{D: w, KW: w, Values: tableKeys})...)

	var vx exprs
	vr, vh, vd, vl := vx.coord(0), vx.coord(1), vx.coord(2), vx.coord(3)
	vin := []Operand{oLat}
	vf := Facts{Extents: map[int]int{0: R * H * w}}
	vv, err := weightElement(&vx, vbo, H*dv, lat, &vin, &vf, vx.op(OpAdd, U, vx.op(OpMul, U, vh, vx.u(uint32(dv))), vd), vl)
	if err != nil {
		return nil, err
	}
	vx.op(OpMul, F, vv, vx.at(0, []int{R, H, w}, vr, vh, vl))
	return append(prims, Prim{Kind: PrimReduce, Shape: []int{R, H, dv, lat}, In: vin, Out: out, Combine: OpAdd, Expr: vx.e, Facts: vf}), nil
}
