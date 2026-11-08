package composer

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// decompositions is every op kind the composer decomposes, each into
// primitives over coordinates: a value of the step is [rows][width].
func decompositions() map[graph.OpKind]Decomposition {
	return map[graph.OpKind]Decomposition{
		graph.OpNorm:         decomposeNorm,
		graph.OpHeadNorm:     decomposeHeadNorm,
		graph.OpMatVec:       decomposeMatVec,
		graph.OpAdd:          decomposeAdd,
		graph.OpExpr:         decomposeExpr,
		graph.OpMap:          decomposeMap,
		graph.OpLayerNorm:    decomposeLayerNorm,
		graph.OpRope:         decomposeRope,
		graph.OpKVAppend:     decomposeKVAppend,
		graph.OpAttend:       decomposeAttend,
		graph.OpRopeTable:    decomposeRopeTable,
		graph.OpRopePos:      decomposeRopePos,
		graph.OpTopK:         decomposeTopK,
		graph.OpReduce:       decomposeReduce,
		graph.OpTake:         decomposeTake,
		graph.OpExpertMatVec: decomposeExpertMatVec,
		graph.OpRank:         decomposeRank,
		graph.OpClamp:        decomposeClamp,
		graph.OpScale:        decomposeScale,
		graph.OpStateConv:    decomposeStateConv,
		graph.OpDeltaRule:    decomposeDeltaRule,
		graph.OpPosition:     decomposePosition,
		graph.OpToken:        decomposeToken,
		graph.OpSelect:       decomposeSelect,
		graph.OpHistoryPool:  decomposeHistoryPool,
		graph.OpNGramHash:    decomposeNGramHash,
		graph.OpNGramWindow:  decomposeNGramWindow,
		graph.OpHalves:       decomposeHalves,
		graph.OpWidenHalves:  decomposeWidenHalves,
		graph.OpQ8Rows:       decomposeQ8Rows,
		graph.OpWidenQ8Rows:  decomposeWidenQ8Rows,
		graph.OpUniform:      decomposeUniform,
		graph.OpPut:          decomposePut,
		graph.OpPool:         decomposePool,
		graph.OpTakeRows:     decomposeTakeRows,
		graph.OpSpanPool:     decomposeSpanPool,
		graph.OpRowConv:      decomposeRowConv,
		graph.OpConv:         decomposeConv,
		graph.OpUnfold:       decomposeUnfold,
		graph.OpPatchify:     decomposePatchify,
		graph.OpRegroup:      decomposeRegroup,
		graph.OpLeadRow:      decomposeLeadRow,
		graph.OpTailRow:      decomposeTailRow,
		graph.OpPosAxes:      decomposePosAxes,
		graph.OpPosBuckets:   decomposePosBuckets,
		graph.OpTableRows:    decomposeTableRows,
		graph.OpRowsAttend:   decomposeRowsAttend,
		graph.OpUpsample:     decomposeUpsample,
		graph.OpAppendRow:    decomposeAppendRow,
		graph.OpDropRows:     decomposeDropRows,
		graph.OpRowBreak:     decomposeRowBreak,
		graph.OpMarkRow:      decomposeMarkRow,
		graph.OpWrapRows:     decomposeWrapRows,
		graph.OpFold:         decomposeFold,
		graph.OpUntile:       decomposeUntile,
		graph.OpSplice:       decomposeSplice,
		graph.OpPosResample:  decomposePosResample,
		graph.OpPicture:      decomposePicture,
		graph.OpFrames:       decomposeFrames,
		graph.OpClipMask:     decomposeClipMask,
		graph.OpSplitRows:    decomposeSplitRows,
		graph.OpFeatureNorm:  decomposeFeatureNorm,
		graph.OpSlice:        decomposeSlice,
		graph.OpTile:         decomposeTile,
		graph.OpConcat:       decomposeConcat,
		graph.OpEmbed:        decomposeEmbed,
	}
}

func (c *Context) value(v graph.ValueID) Operand {
	o := Operand{Kind: OperandValue, Value: v, Name: c.Graph.Values[v].Name, Out: slices.Contains(c.Graph.Out, v)}
	if g := c.Graph; c.Part == PartBlock && len(g.Out) > 1 {
		if k := slices.Index(g.Out, v); k >= 0 {
			o.Stream = k + 1
		} else if k := slices.Index(g.In, v); k >= 0 && k < len(g.Out) {
			o.Stream = k + 1
		}
	}
	if c.Graph.Values[v].Elem == graph.ElemU32 {
		o.Elem = ir.U32
	}
	return o
}

func (c *Context) width(v graph.ValueID) int { return c.Graph.Values[v].Width }

// rows is value v's shape: its rows (the step's, or its grid's where it
// states them, graph.Value.Rows) and its width.
func (c *Context) rows(v graph.ValueID) []int {
	if r := c.Graph.Values[v].Rows; r > 0 {
		return []int{r, c.width(v)}
	}
	return []int{c.Step.Rows, c.width(v)}
}

func (c *Context) weight(w graph.Binding) (Operand, error) {
	f, ok := c.Formats[w.Role]
	if !ok {
		return Operand{}, fmt.Errorf("no format for %s", w.Role)
	}
	return Operand{Kind: OperandWeight, Role: w.Role, Format: f, Name: w.Role.String(), Instance: w.From}, nil
}

// decomposeNorm is an RMS norm: each row's sum of squares, then each element
// scaled by the row's reciprocal root mean square and its weight.
func decomposeNorm(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.Weight) != 1 || len(o.In) != 1 || len(o.Out) != 1 {
		return nil, fmt.Errorf("a norm of %d weights", len(o.Weight))
	}
	F := ir.F32
	sh := c.rows(o.In[0])
	R, W := sh[0], sh[1]
	sum := Operand{Kind: OperandScratch, Name: c.Graph.Values[o.Out[0]].Name + ".sumsq"}
	wt := Operand{Kind: OperandVector, Role: o.Weight[0].Role, Name: o.Weight[0].Role.String()}

	var s exprs
	xs := s.at(0, sh, s.coord(0), s.coord(1))
	s.op(OpMul, F, xs, xs)

	var m exprs
	r, i := m.coord(0), m.coord(1)
	ms := m.op(OpMul, F, m.at(1, []int{R}, r), m.f(1/float32(W)))
	inv := m.op(OpDiv, F, m.f(1), m.un(OpSqrt, F, m.op(OpAdd, F, ms, m.f(float32(o.Attr.Eps)))))
	m.op(OpMul, F, m.op(OpMul, F, m.at(0, sh, r, i), inv), m.at(2, []int{W}, i))

	return []Prim{
		{Kind: PrimReduce, Shape: sh, In: []Operand{c.value(o.In[0])}, Out: sum, Combine: OpAdd, Expr: s.e},
		{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), sum, wt}, Out: c.value(o.Out[0]), Expr: m.e},
	}, nil
}

// decomposeHeadNorm is an RMS norm over each head of a row: the row seen as
// heads of HeadDim, each head's sum of squares, then each element scaled by
// its head's reciprocal root mean square and the weight every head shares,
// ones where there is none.
func decomposeHeadNorm(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.Weight) > 1 || len(o.In) != 1 || len(o.Out) != 1 {
		return nil, fmt.Errorf("a head norm of %d weights", len(o.Weight))
	}
	F := ir.F32
	rows := c.rows(o.In[0])
	R, W, D := rows[0], rows[1], o.Attr.HeadDim
	if D <= 0 || W%D != 0 {
		return nil, fmt.Errorf("a head norm of heads of %d over a row of %d", D, W)
	}
	sh := []int{R, W / D, D}
	sum := Operand{Kind: OperandScratch, Name: c.Graph.Values[o.Out[0]].Name + ".sumsq"}

	var s exprs
	xs := s.at(0, sh, s.coord(0), s.coord(1), s.coord(2))
	s.op(OpMul, F, xs, xs)

	var m exprs
	r, h, i := m.coord(0), m.coord(1), m.coord(2)
	ms := m.op(OpMul, F, m.at(1, sh[:2], r, h), m.f(1/float32(D)))
	inv := m.op(OpDiv, F, m.f(1), m.un(OpSqrt, F, m.op(OpAdd, F, ms, m.f(float32(o.Attr.Eps)))))
	y := m.op(OpMul, F, m.at(0, sh, r, h, i), inv)
	in := []Operand{c.value(o.In[0]), sum}
	if len(o.Weight) == 1 {
		m.op(OpMul, F, y, m.at(2, []int{D}, i))
		in = append(in, Operand{Kind: OperandVector, Role: o.Weight[0].Role, Name: o.Weight[0].Role.String()})
	}
	return []Prim{
		{Kind: PrimReduce, Shape: sh, In: []Operand{c.value(o.In[0])}, Out: sum, Combine: OpAdd, Expr: s.e},
		{Kind: PrimMap, Shape: sh, In: in, Out: c.value(o.Out[0]), Expr: m.e},
	}, nil
}

// decomposeMatVec is each output row's sum over K of the weight's row times
// the input.
func decomposeMatVec(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) == 2 {
		return decomposeMatVecRows(o, c)
	}
	if len(o.Weight) == 2 && len(o.In) == 1 && len(o.Out) == 1 {
		return decomposeBiased(o, c)
	}
	if len(o.Weight) != 1 || len(o.In) != 1 || len(o.Out) != 1 {
		return nil, fmt.Errorf("a matvec of %d weights", len(o.Weight))
	}
	w := o.Weight[0]
	wo, err := c.weight(w)
	if err != nil {
		return nil, err
	}
	codes := c.Graph.Values[o.In[0]].Elem == graph.ElemI8
	if c.matrix && !codes && wo.Format != jlm.TypeF32 {
		if p, err := decomposeMatVecDecoded(w.Rows, w.K, wo, c.value(o.In[0]), c.value(o.Out[0]), c.Step.Rows); err == nil {
			return []Prim{p}, nil
		}
	}
	switch wo.Format {
	case jlm.TypeF32:
	case jlm.TypeQ8:
		// The int8 form is the layout's whole blocks; a K padded to them (a
		// tower 144 wide) is decoded.
		if c.Int8 != nil && !codes && w.K%q8Block == 0 {
			return decomposeMatVecQ8(o, c, wo)
		}
		fallthrough
	default:
		if l, ok := jlm.Layout(wo.Format); ok {
			if prims, ok := decomposeMatVecInt8(o, c, wo, l); ok {
				return prims, nil
			}
		}
		p, err := decomposeMatVecDecoded(w.Rows, w.K, wo, c.value(o.In[0]), c.value(o.Out[0]), c.Step.Rows)
		if err != nil {
			return nil, fmt.Errorf("a matvec over %s weights: %w", wo.Format, err)
		}
		return []Prim{p}, nil
	}
	var x exprs
	r, row, k := x.coord(0), x.coord(1), x.coord(2)
	x.op(OpMul, ir.F32, x.at(0, []int{w.Rows, w.K}, row, k), x.at(1, c.rows(o.In[0]), r, k))
	var prims []Prim
	if v := o.In[0]; c.Int8 != nil && c.Part == PartBlock && !codes && !c.formed[v] && c.producedByRow(v) && c.width(v)%c.Int8.Window == 0 {
		c.formed[v] = true
		prims = c.int8(v)
	}
	return append(prims, Prim{Kind: PrimContract, Shape: []int{c.Step.Rows, w.Rows, w.K}, In: []Operand{wo, c.value(o.In[0])},
		Out: c.value(o.Out[0]), Expr: x.e}), nil
}

// decomposeBiased is a matvec with a bias: the product, then each output
// element plus the bias's.
func decomposeBiased(o graph.Op, c *Context) ([]Prim, error) {
	b := o.Weight[1]
	if f := c.Formats[b.Role]; f != jlm.TypeF32 {
		return nil, fmt.Errorf("a bias %s stored as %s", b.Role, f)
	}
	inner := o
	inner.Weight = o.Weight[:1]
	prims, err := decomposeMatVec(inner, c)
	if err != nil {
		return nil, err
	}
	out := c.value(o.Out[0])
	raw := Operand{Kind: OperandScratch, Name: out.Name + ".raw"}
	for i := range prims {
		if prims[i].Out == out {
			prims[i].Out = raw
		}
	}
	sh := c.rows(o.Out[0])
	var x exprs
	r, i := x.coord(0), x.coord(1)
	x.op(OpAdd, ir.F32, x.at(0, sh, r, i), x.at(1, []int{sh[1]}, i))
	bias := Operand{Kind: OperandVector, Role: b.Role, Name: b.Role.String()}
	return append(prims, Prim{Kind: PrimMap, Shape: sh, In: []Operand{raw, bias}, Out: out, Expr: x.e}), nil
}

// producedByRow reports whether value v is the output of an op computing a
// row from the row alone: a norm or an elementwise expression.
func (c *Context) producedByRow(v graph.ValueID) bool {
	for _, o := range c.Graph.Ops {
		for _, out := range o.Out {
			if out == v {
				return o.Kind == graph.OpNorm || o.Kind == graph.OpExpr
			}
		}
	}
	return false
}

func decomposeAdd(o graph.Op, c *Context) ([]Prim, error) {
	if len(o.In) != 2 || len(o.Out) != 1 {
		return nil, fmt.Errorf("an add of %d operands", len(o.In))
	}
	F := ir.F32
	sh := c.rows(o.Out[0])
	var x exprs
	r, i := x.coord(0), x.coord(1)
	a, b := x.at(0, sh, r, i), x.at(1, sh, r, i)
	if al := o.Attr.Alpha; al != 0 && al != 1 {
		// A scaled residual: x + Alpha*y.
		b = x.op(OpMul, F, b, x.f(float32(al)))
	}
	sum := x.op(OpAdd, F, a, b)
	if o.Attr.Scale != 0 {
		// A scaled add, (x + y) * Scale.
		x.op(OpMul, F, sum, x.f(float32(o.Attr.Scale)))
	}
	return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0]), c.value(o.In[1])}, Out: c.value(o.Out[0]), Expr: x.e}}, nil
}

// sliceSum is a left fold of adds of a value's k slices, w wide at j*w.
type sliceSum struct {
	v    graph.ValueID
	w, k int
}

// sliceSums is every slice sum by its root op, every value between read by
// the fold alone, and the fold's other ops.
func sliceSums(g *graph.Graph) (map[int]sliceSum, map[int]bool) {
	readers := make([]int, len(g.Values))
	made := make([]int, len(g.Values))
	for i, o := range g.Ops {
		for _, v := range o.In {
			readers[v]++
		}
		for _, v := range o.Out {
			made[v] = i + 1
		}
	}
	for _, v := range g.Out {
		readers[v]++
	}
	plainAdd := func(o graph.Op) bool {
		return o.Kind == graph.OpAdd && len(o.In) == 2 && len(o.Out) == 1 && (o.Attr.Alpha == 0 || o.Attr.Alpha == 1) && o.Attr.Scale == 0
	}
	leaf := func(v graph.ValueID) (graph.Op, bool) {
		if made[v] == 0 || readers[v] != 1 {
			return graph.Op{}, false
		}
		o := g.Ops[made[v]-1]
		return o, o.Kind == graph.OpSlice && len(o.In) == 1 && len(o.Weight) == 0 && o.Attr.HeadDim == 0 &&
			g.Values[o.In[0]].Rows == g.Values[v].Rows
	}
	roots, members := map[int]sliceSum{}, map[int]bool{}
	for i := len(g.Ops) - 1; i >= 0; i-- {
		o := g.Ops[i]
		if members[i] || !plainAdd(o) {
			continue
		}
		var terms []graph.Op
		var ops []int
		cur := o
		for {
			t, ok := leaf(cur.In[1])
			if !ok {
				break
			}
			terms = append(terms, t)
			ops = append(ops, made[cur.In[1]]-1)
			l := cur.In[0]
			if t, ok := leaf(l); ok {
				terms = append(terms, t)
				ops = append(ops, made[l]-1)
				cur = graph.Op{}
				break
			}
			if made[l] == 0 || readers[l] != 1 || !plainAdd(g.Ops[made[l]-1]) {
				break
			}
			ops = append(ops, made[l]-1)
			cur = g.Ops[made[l]-1]
		}
		if cur.Kind != graph.OpInvalid || len(terms) < 2 {
			continue
		}
		v, w, k := terms[0].In[0], terms[0].Attr.Width, len(terms)
		ok := g.Values[v].Width == k*w && g.Values[o.Out[0]].Rows == g.Values[v].Rows
		for j, t := range terms {
			ok = ok && t.In[0] == v && t.Attr.Width == w && t.Attr.Offset == (k-1-j)*w
		}
		if !ok {
			continue
		}
		roots[i] = sliceSum{v: v, w: w, k: k}
		for _, m := range ops {
			members[m] = true
		}
	}
	return roots, members
}

func decomposeSliceSum(o graph.Op, s sliceSum, c *Context) []Prim {
	U := ir.U32
	sh := c.rows(o.Out[0])
	var x exprs
	x.at(0, c.rows(s.v), x.coord(0), x.op(OpAdd, U, x.op(OpMul, U, x.coord(2), x.u(uint32(s.w))), x.coord(1)))
	return []Prim{{Kind: PrimReduce, Shape: []int{sh[0], s.w, s.k}, In: []Operand{c.value(s.v)}, Out: c.value(o.Out[0]),
		Combine: OpAdd, Expr: x.e}}
}
