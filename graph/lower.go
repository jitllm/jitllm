package graph

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// The stated programs' lowering: a container states a tower's entry, block
// pieces and head, a block's stream reads and writes, and its recurrent
// statements as jlm programs, each op lowered here to the graph op of the
// same meaning.

// TowerTiles is the nested order the tower's patch rows are in: merge x
// merge merge windows (TOpPatchify's Width, 1 for none) in windows of window
// x window of them (TowerBlock.Window over the merge, 0 for none).
func TowerTiles(t *jlm.Tower) (merge, window int) {
	merge = 1
	if t == nil {
		return merge, 0
	}
	if ops := t.Entry.Ops; len(ops) > 0 && ops[0].Code == jlm.TOpPatchify && ops[0].Width > 1 {
		merge = int(ops[0].Width)
	}
	return merge, int(t.Block.Window) / merge
}

// resampleOf is a record's resample (jlm.ResampleAligned, jlm.TapsLinear) as
// OpPosResample states it.
func resampleOf(chain, filter uint8) (uint8, uint8, error) {
	var c, f uint8
	switch chain {
	case jlm.ResampleAligned:
		c = ChainAligned
	case jlm.ResampleHalfPixel:
		c = ChainHalfPixel
	}
	switch filter {
	case jlm.TapsBilinear:
		f = FilterBilinear
	case jlm.TapsLinear:
		f = FilterLinear
	case jlm.TapsCubic:
		f = FilterCubic
	case jlm.TapsAntialias:
		f = FilterAntialias
	}
	if c == 0 || f == 0 {
		return 0, 0, fmt.Errorf("a resample by chain %d and taps %d", chain, filter)
	}
	return c, f, nil
}

// LowerProgram adds p's ops to b over its inputs in, each op the graph's of
// the same meaning, and is p's result. A value's width is its op's: a
// matrix's rows, a regroup's input times its group, and a patch gather's the
// width the matrix reading it takes.
func LowerProgram(b *Builder, t *jlm.Tower, p *jlm.Program, in []ValueID, rows int, f Facts) (ValueID, error) {
	vals, err := lowerValuesOf(b, t, p, in, rows, f)
	if err != nil {
		return 0, err
	}
	return vals[p.Out], nil
}

// lowerValuesOf is LowerProgram's every value: the inputs, then each op's.
func lowerValuesOf(b *Builder, t *jlm.Tower, p *jlm.Program, in []ValueID, rows int, f Facts) ([]ValueID, error) {
	return LowerValues(b, t, p, in, rows, f, nil)
}

// LowerValues is lowerValuesOf where blockOps lowers the ops only a block
// has (the model's blockOps).
func LowerValues(b *Builder, t *jlm.Tower, p *jlm.Program, in []ValueID, rows int, f Facts,
	blockOps func([]ValueID, *jlm.TowerOp) (ValueID, error)) ([]ValueID, error) {
	merge, window := TowerTiles(t)
	if len(in) != int(p.NIn) {
		return nil, fmt.Errorf("a program of %d inputs over %d", p.NIn, len(in))
	}
	vals := append([]ValueID(nil), in...)
	shape := func(r jlm.Role) (Shape, error) {
		s, ok := f.Inventory[r]
		if !ok {
			return s, fmt.Errorf("binds %v, which the container does not hold", r)
		}
		return s, nil
	}
	// The width a value is read at by the matrix that reads it.
	readAt := map[int]int{}
	for _, op := range p.Ops {
		if op.Code == jlm.TOpLinear && len(op.Bind) > 0 {
			if s, ok := f.Inventory[op.Bind[0]]; ok {
				readAt[int(op.In[0])] = s.K
			}
		}
	}
	// An elementwise op's inputs are read at the width its result is: a
	// video's frames' difference is read by the second plane, padded as
	// the first is.
	for i := len(p.Ops) - 1; i >= 0; i-- {
		op := &p.Ops[i]
		if k := readAt[len(in)+i]; k > 0 && (op.Code == jlm.TOpSub || op.Code == jlm.TOpAdd) {
			for _, v := range op.In {
				readAt[int(v)] = max(readAt[int(v)], k)
			}
		}
	}
	width := func(v ValueID) int { return b.G.Values[v].Width }
	// rowsOf is a value's rows: its own, where an op that changes the count
	// made it, else the program's.
	rowsOf := func(v ValueID) int {
		if r := b.G.Values[v].Rows; r > 0 {
			return r
		}
		return rows
	}
	for i := range p.Ops {
		op := &p.Ops[i]
		ins := make([]ValueID, len(op.In))
		for j, v := range op.In {
			ins[j] = vals[v]
		}
		at := len(in) + i
		name := fmt.Sprintf("op%d", i)
		var out ValueID
		switch op.Code {
		case jlm.TOpPatchify:
			w := 3 * int(op.Span*op.Span)
			if k := readAt[at]; k > w {
				w = k
			}
			out = b.Emit(OpPatchify, ins, []Result{{Name: "patches", Width: w}}, nil,
				Attrs{Span: int(op.Span), Offset: int(op.Offset), Width: int(op.Width), Window: window}, nil, nil)[0]
		case jlm.TOpLinear:
			s, err := shape(op.Bind[0])
			if err != nil {
				return nil, err
			}
			out = b.MatVec(ins[0], op.Bind[0], s.Rows, op.Bind[1:]...)
		case jlm.TOpPosTable:
			s, err := shape(op.Bind[0])
			if err != nil {
				return nil, err
			}
			out = b.Emit(OpEmbed, nil, []Result{{Name: "pos", Width: s.K}},
				[]Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}}, Attrs{Positional: true, Offset: int(op.Offset)}, nil, nil)[0]
		case jlm.TOpPosResample:
			s, err := shape(op.Bind[0])
			if err != nil {
				return nil, err
			}
			if s.Rows != int(op.Span*op.Span) {
				return nil, fmt.Errorf("resamples a table %d a side from %d rows", op.Span, s.Rows)
			}
			chain, filter, err := resampleOf(op.Chain, op.Filter)
			if err != nil {
				return nil, err
			}
			out = b.Emit(OpPosResample, nil, []Result{{Name: "pos", Width: s.K}},
				[]Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}},
				Attrs{Span: int(op.Span), Offset: int(op.Offset), Width: int(op.Width), Window: window, Chain: chain, Filter: filter}, nil, nil)[0]
		case jlm.TOpPosAxes:
			s, err := shape(op.Bind[0])
			if err != nil {
				return nil, err
			}
			if s.Rows != int(2*op.Span) {
				return nil, fmt.Errorf("reads two tables of %d rows from %d rows", op.Span, s.Rows)
			}
			out = b.Emit(OpPosAxes, nil, []Result{{Name: "pos", Width: s.K}},
				[]Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}},
				Attrs{Span: int(op.Span), Offset: int(op.Offset)}, nil, nil)[0]
		case jlm.TOpAdd:
			out = b.Add(ins[0], ins[1])
		case jlm.TOpSub:
			out = b.Emit(OpMap, ins, []Result{{Name: "diff", Width: width(ins[0])}}, nil,
				Attrs{Fn: MapSub}, nil, nil)[0]
		case jlm.TOpLayerNorm:
			out = b.LayerNorm(ins[0], op.Bind[0], float64(op.Eps), op.Bind[1:]...)
		case jlm.TOpRMSNorm:
			if len(op.Bind) == 0 {
				w, hd := width(ins[0]), width(ins[0])
				if op.Span > 0 {
					hd = int(op.Span)
				}
				out = b.Emit(OpHeadNorm, ins, []Result{{Name: name, Width: w}}, nil,
					Attrs{Eps: float64(op.Eps), HeadDim: hd}, nil, nil)[0]
				break
			}
			if op.Span > 0 && int(op.Span) < width(ins[0]) {
				out = b.HeadNorm(ins[0], op.Bind[0], int(op.Span), float64(op.Eps))
				break
			}
			out = b.Norm(ins[0], op.Bind[0], float64(op.Eps))
		case jlm.TOpRope, jlm.TOpNGramRows, jlm.TOpTailConv:
			if blockOps == nil {
				return nil, fmt.Errorf("op %d is code %d, which only a block has", i, op.Code)
			}
			v, err := blockOps(ins, op)
			if err != nil {
				return nil, err
			}
			out = v
		case jlm.TOpScale, jlm.TOpShift:
			s, err := shape(op.Bind[0])
			if err != nil {
				return nil, err
			}
			fn := MapMul
			if op.Code == jlm.TOpShift {
				fn = MapAdd
			}
			out = b.Map(name, Attrs{Fn: fn}, []Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}},
				width(ins[0]), ins[0])
		case jlm.TOpSum, jlm.TOpMax:
			fn := MapAdd
			if op.Code == jlm.TOpMax {
				fn = MapMax
			}
			w, a := 1, Attrs{Fn: fn}
			if op.Span > 0 {
				w, a.Span = width(ins[0])/int(op.Span), int(op.Span)
			}
			out = b.Emit(OpReduce, ins, []Result{{Name: name, Width: w}}, nil, a, nil, nil)[0]
		case jlm.TOpConcat:
			w := 0
			for _, v := range ins {
				w += width(v)
			}
			out = b.Emit(OpConcat, ins, []Result{{Name: name, Width: w}}, nil, Attrs{}, nil, nil)[0]
		case jlm.TOpSlice:
			out = b.Emit(OpSlice, ins, []Result{{Name: name, Width: int(op.Width)}}, nil,
				Attrs{Offset: int(op.Offset), Width: int(op.Width)}, nil, nil)[0]
		case jlm.TOpRegroup:
			S := int(op.Span)
			out = b.Emit(OpRegroup, ins, []Result{{Name: "grouped", Width: S * S * width(ins[0]),
				Rows: rowsOf(ins[0]) / (S * S)}}, nil, Attrs{Span: S}, nil, nil)[0]
		case jlm.TOpPool:
			// The mean of each block's S*S rows: the block regrouped into one
			// row, its slots summed and scaled by 1/S^2.
			S, E := int(op.Span), width(ins[0])
			g := b.Emit(OpRegroup, ins, []Result{{Name: "pool.grouped", Width: S * S * E,
				Rows: rowsOf(ins[0]) / (S * S)}}, nil, Attrs{Span: S, GridW: int(op.GridW)}, nil, nil)[0]
			slot := func(k int) ValueID {
				return b.Emit(OpSlice, []ValueID{g}, []Result{{Name: "pool.slot", Width: E}}, nil,
					Attrs{Offset: k * E, Width: E}, nil, nil)[0]
			}
			acc := slot(0)
			for k := 1; k < S*S; k++ {
				acc = b.Add(acc, slot(k))
			}
			var e jlm.ExprBuilder
			e.Fn(jlm.FnMul, jlm.Operand(0), jlm.Num(1/float32(S*S)))
			mean := jlm.Expr{Terms: e.T}
			var err error
			if out, err = b.Stated("pooled", &mean, []ValueID{acc}, E, f); err != nil {
				return nil, fmt.Errorf("the pool's mean: %w", err)
			}
		case jlm.TOpFold:
			S := int(op.Span)
			out = b.Emit(OpFold, ins, []Result{{Name: "folded", Width: S * width(ins[0]),
				Rows: rowsOf(ins[0]) / S}}, nil, Attrs{Span: S}, nil, nil)[0]
		case jlm.TOpUntile:
			out = b.Emit(OpUntile, ins, []Result{{Name: "raster", Width: width(ins[0]), Rows: rowsOf(ins[0])}}, nil,
				Attrs{Width: merge, Window: window}, nil, nil)[0]
		case jlm.TOpLeadRow:
			out = b.Emit(OpLeadRow, ins, []Result{{Name: "class", Width: width(ins[0])}},
				[]Binding{{Role: op.Bind[0], K: width(ins[0])}}, Attrs{Offset: int(op.Offset)}, nil, nil)[0]
		case jlm.TOpMarkRow:
			s, err := shape(op.Bind[0])
			if err != nil {
				return nil, err
			}
			if s.K != width(ins[0]) || s.Rows < 1 {
				return nil, fmt.Errorf("marks rows %d wide from a table of %d rows %d wide", width(ins[0]), s.Rows, s.K)
			}
			if op.Span < 1 || op.Offset < 0 {
				return nil, fmt.Errorf("marks rows with runs of %d before and %d after", op.Span, op.Offset)
			}
			out = b.Emit(OpMarkRow, ins, []Result{{Name: "marked", Width: width(ins[0]),
				Rows: rowsOf(ins[0]) + int(op.Span+op.Offset)}},
				[]Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}},
				Attrs{Width: int(op.Width), Span: int(op.Span), Offset: int(op.Offset)}, nil, nil)[0]
		case jlm.TOpPosBuckets:
			s, err := shape(op.Bind[0])
			if err != nil {
				return nil, err
			}
			if s.Rows != int(op.Span*op.Span) {
				return nil, fmt.Errorf("buckets onto a table %d a side from %d rows", op.Span, s.Rows)
			}
			out = b.Emit(OpPosBuckets, nil, []Result{{Name: "pos", Width: s.K}},
				[]Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}},
				Attrs{Span: int(op.Span), Offset: int(op.Offset)}, nil, nil)[0]
		case jlm.TOpTableRows:
			s, err := shape(op.Bind[0])
			if err != nil {
				return nil, err
			}
			out = b.Emit(OpTableRows, nil, []Result{{Name: "table", Width: s.K, Rows: s.Rows}},
				[]Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}}, Attrs{}, nil, nil)[0]
		case jlm.TOpRowsAttend:
			w, h, g := width(ins[0]), int(op.Span), int(op.Width)
			if g == 0 {
				g = h
			}
			if w%h != 0 || width(ins[1]) != g*(w/h) || width(ins[2]) != g*(w/h) || len(ins) == 4 && width(ins[3]) != w {
				return nil, fmt.Errorf("attends rows %d, %d and %d wide in %d heads over %d", w, width(ins[1]), width(ins[2]), h, g)
			}
			// The relative form reads its positions' table.
			var bind []Binding
			if len(ins) == 4 {
				s, err := shape(op.Bind[0])
				if err != nil {
					return nil, err
				}
				bind = []Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}}
			}
			out = b.Emit(OpRowsAttend, ins, []Result{{Name: "attended", Width: w, Rows: rowsOf(ins[0])}}, bind,
				Attrs{NHead: h, NKVHead: g, HeadDim: w / h, Scale: 1 / math.Sqrt(float64(w/h))}, nil, nil)[0]
		case jlm.TOpFrames:
			out = b.Emit(OpFrames, nil, []Result{{Name: "frames", Width: int(op.Width), Rows: rows}}, nil,
				Attrs{Width: int(op.Width), Stride: int(op.Stride), Const: float64(op.Eps)}, nil, nil)[0]
		case jlm.TOpFeatureNorm:
			out = b.Emit(OpFeatureNorm, ins, []Result{{Name: name, Width: width(ins[0])}}, nil,
				Attrs{Offset: int(op.Offset), Eps: float64(op.Eps)}, nil, nil)[0]
		case jlm.TOpSplitRows:
			w := int(op.Width)
			if width(ins[0])%w != 0 {
				return nil, fmt.Errorf("splits rows of %d into rows of %d", width(ins[0]), w)
			}
			out = b.Emit(OpSplitRows, ins, []Result{{Name: name, Width: w, Rows: rowsOf(ins[0]) * (width(ins[0]) / w)}}, nil,
				Attrs{Width: w}, nil, nil)[0]
		case jlm.TOpClipMask:
			out = b.Emit(OpClipMask, ins, []Result{{Name: name, Width: width(ins[0])}}, nil,
				Attrs{Stride: int(op.Stride), Span: int(op.Span), GridW: int(op.GridW)}, nil, nil)[0]
		case jlm.TOpRowConv:
			s, err := shape(op.Bind[0])
			if err != nil {
				return nil, err
			}
			if s.K != int(op.Span) || s.Rows != width(ins[0]) {
				return nil, fmt.Errorf("convolves %d channels by %d taps from a %dx%d table", width(ins[0]), op.Span, s.Rows, s.K)
			}
			out = b.Emit(OpRowConv, ins, []Result{{Name: name, Width: width(ins[0])}},
				[]Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}}, Attrs{Taps: int(op.Span)}, nil, nil)[0]
		case jlm.TOpUnfold, jlm.TOpConv:
			T, st, W := int(op.Span), int(op.Stride), int(op.GridW)
			R := rowsOf(ins[0])
			if R%W != 0 {
				return nil, fmt.Errorf("convolves %d rows as a grid %d wide", R, W)
			}
			res := Result{Name: name, Width: width(ins[0])}
			if st > 1 {
				res.Rows = ((R/W + st - 1) / st) * ((W + st - 1) / st)
			}
			kind := OpConv
			var bind []Binding
			if op.Code == jlm.TOpUnfold {
				kind, res.Width = OpUnfold, int(op.Width)
			} else {
				bind = []Binding{{Role: op.Bind[0], K: width(ins[0]), Rows: T * T}}
			}
			out = b.Emit(kind, ins, []Result{res}, bind, Attrs{Taps: T, Stride: st, GridW: W, Offset: int(op.Offset)}, nil, nil)[0]
		case jlm.TOpUpsample:
			S := int(op.Span)
			out = b.Emit(OpUpsample, ins, []Result{{Name: name, Width: width(ins[0]), Rows: rowsOf(ins[0]) * S * S}}, nil,
				Attrs{Span: S, GridW: int(op.GridW)}, nil, nil)[0]
		case jlm.TOpAppendRow:
			s, err := shape(op.Bind[0])
			if err != nil {
				return nil, err
			}
			if s.K != width(ins[0]) || int(op.Offset) >= max(s.Rows, 1) {
				return nil, fmt.Errorf("appends row %d of a %dx%d table to rows %d wide", op.Offset, s.Rows, s.K, width(ins[0]))
			}
			out = b.Emit(OpAppendRow, ins, []Result{{Name: name, Width: width(ins[0]), Rows: rowsOf(ins[0]) + 1}},
				[]Binding{{Role: op.Bind[0], K: s.K, Rows: s.Rows}}, Attrs{Offset: int(op.Offset)}, nil, nil)[0]
		case jlm.TOpTailRow:
			out = b.Emit(OpTailRow, ins, []Result{{Name: "class", Width: width(ins[0])}},
				[]Binding{{Role: op.Bind[0], K: width(ins[0])}}, Attrs{}, nil, nil)[0]
		case jlm.TOpRowBreak:
			out = b.Emit(OpRowBreak, ins, []Result{{Name: "lines", Width: width(ins[0]), Rows: 2 * rowsOf(ins[0])}},
				[]Binding{{Role: op.Bind[0], K: width(ins[0])}}, Attrs{Span: int(op.Span), Offset: int(op.Offset)}, nil, nil)[0]
		case jlm.TOpWrapRows:
			line, _, _ := ProgramFrame(p)
			out = b.Emit(OpWrapRows, ins, []Result{{Name: "framed", Width: width(ins[0]), Rows: rowsOf(ins[0]) + 2}},
				[]Binding{{Role: op.Bind[0], K: width(ins[0])}, {Role: op.Bind[1], K: width(ins[0])}},
				Attrs{Span: int(op.Span), Offset: int(t.Entry.Ops[0].Offset), Width: line}, nil, nil)[0]
		case jlm.TOpDropRows:
			out = b.Emit(OpDropRows, ins, []Result{{Name: "rows", Width: width(ins[0]),
				Rows: rowsOf(ins[0]) - int(op.Offset)}}, nil, Attrs{Offset: int(op.Offset)}, nil, nil)[0]
		case jlm.TOpStated:
			w := int(op.Width)
			if w == 0 {
				w = width(ins[0])
			}
			v, err := b.Stated(name, op.Expr, ins, w, f)
			if err != nil {
				return nil, err
			}
			out = v
		default:
			return nil, fmt.Errorf("op %d is code %d, which this engine does not lower", i, op.Code)
		}
		vals = append(vals, out)
	}
	return vals, nil
}

// ProgramFrame is the rows a program's TOpRowBreak adds after each row of
// the merged grid -- 1 where it has one, 0 where not -- the merge it counts
// that grid over, and the breaks it leaves off the last row.
func ProgramFrame(p *jlm.Program) (line, merge, trim int) {
	for _, op := range p.Ops {
		if op.Code == jlm.TOpRowBreak {
			return 1, int(op.Span), int(op.Offset)
		}
	}
	return 0, 0, 0
}
