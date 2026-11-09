package graph

import (
	"errors"
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
)

// Shape is a stored tensor's shape: K its input width (a vector's length),
// Rows its output rows (0 for a vector), Sheets a bank's experts (0 for one
// matrix).
type Shape struct{ K, Rows, Sheets int }

// Facts is what the container stores for a block, independently of any graph
// built for it: its tensors by role. The verifier holds a graph to them.
type Facts struct {
	Inventory map[jlm.Role]Shape
}

// Verify checks g's structure before anything is formed, and returns every
// violation found, joined: the op language's version; that every value has
// one producer and is defined before it is read; that every state read sees
// the version in force and every write advances it by one; that every
// binding names a tensor the container stores at the shape the op binds; that
// a matrix product's widths are its weight's and a norm's scale its row's; and
// that the outputs are produced. What a graph computes is held elsewhere, by
// running it against a reference; this checks what can be checked without.
func Verify(g *Graph, f Facts) error {
	if g == nil {
		return errors.New("graph: no graph")
	}
	if g.Version != Version {
		return fmt.Errorf("graph: block %d is op language v%d; this build reads v%d", g.Block, g.Version, Version)
	}
	var errs []error
	bad := func(format string, a ...any) { errs = append(errs, fmt.Errorf(format, a...)) }
	n := len(g.Values)
	defined := make([]bool, n)
	valid := func(id ValueID) bool { return id >= 0 && int(id) < n }
	width := func(id ValueID) int {
		if valid(id) {
			return g.Values[id].Width
		}
		return -1
	}
	// A block reads the residual; a model-wide graph (the input rows) may
	// read nothing but its step.
	if len(g.In) == 0 && g.Block >= 0 {
		bad("graph: block %d has no input", g.Block)
	}
	for _, id := range g.In {
		if !valid(id) {
			bad("graph: block %d's input %d is not a value", g.Block, id)
			continue
		}
		defined[id] = true
	}
	state := map[StateRef]int{}
	for i, o := range g.Ops {
		at := fmt.Sprintf("graph: block %d op %d (%s)", g.Block, i, o.Kind)
		if o.Kind <= OpInvalid || o.Kind >= numOps {
			bad("%s: unknown op", at)
			continue
		}
		for _, id := range o.In {
			switch {
			case !valid(id):
				bad("%s: input %d is not a value", at, id)
			case !defined[id]:
				bad("%s: reads %%%s before anything writes it", at, g.Values[id].Name)
			}
		}
		for _, id := range o.Out {
			switch {
			case !valid(id):
				bad("%s: output %d is not a value", at, id)
			case defined[id]:
				bad("%s: writes %%%s, which already has a producer", at, g.Values[id].Name)
			default:
				defined[id] = true
			}
		}
		// An op's values are rows of one grid; an op that makes a grid -- a
		// strided convolution, a regroup of S*S rows into one, a fold of Span
		// consecutive rows, a row drop -- states its output's rows, and a
		// regroup's, a fold's and a drop's follow their input's where those
		// are the input's own.
		rows := func(id ValueID) int {
			if valid(id) {
				return g.Values[id].Rows
			}
			return 0
		}
		strided := (o.Kind == OpConv || o.Kind == OpUnfold) && o.Attr.Stride > 1
		grid := strided || o.Kind == OpRegroup || o.Kind == OpFold || o.Kind == OpSplitRows || o.Kind == OpDropRows ||
			o.Kind == OpRowBreak || o.Kind == OpWrapRows || o.Kind == OpMarkRow || o.Kind == OpTableRows || o.Kind == OpRowsAttend || o.Kind == OpUpsample || o.Kind == OpAppendRow
		// Attention over the call's own rows takes the call's rows.
		if o.Kind == OpRowsAttend && len(o.In) > 0 && rows(o.In[0]) == 0 {
			grid = false
		}
		for j, id := range o.Out {
			if r := rows(id); grid && j == 0 && r < 1 {
				bad("%s: a %s's %%%s states no rows", at, o.Kind, g.Values[id].Name)
			}
		}
		if grid && len(o.In) > 0 && len(o.Out) > 0 && rows(o.In[0]) > 0 {
			in, out, want := rows(o.In[0]), rows(o.Out[0]), -1
			switch o.Kind {
			case OpRegroup:
				want = in / max(o.Attr.Span*o.Attr.Span, 1)
			case OpFold:
				want = in / max(o.Attr.Span, 1)
			case OpSplitRows:
				if w := o.Attr.Width; w > 0 && len(o.In) > 0 {
					want = in * (g.Values[o.In[0]].Width / w)
				}
			case OpDropRows:
				want = in - o.Attr.Offset
			case OpRowBreak:
				want = 2 * in
			case OpWrapRows:
				want = in + 2
			case OpRowsAttend:
				want = in
			case OpUpsample:
				want = in * o.Attr.Span * o.Attr.Span
			case OpAppendRow:
				want = in + 1
			case OpMarkRow:
				want = in + o.Attr.Span + o.Attr.Offset
			}
			if want >= 0 && out != want {
				bad("%s: a %s of %d rows writes %d, not %d", at, o.Kind, in, out, want)
			}
		}
		for _, id := range append(slices.Clone(o.In), o.Out...) {
			if !grid && len(o.In) > 0 && rows(id) != rows(o.In[0]) {
				bad("%s: %%%s is of %d rows, %%%s of %d", at, g.Values[id].Name, rows(id), g.Values[o.In[0]].Name, rows(o.In[0]))
			}
		}
		for _, r := range o.Reads {
			if state[r.key()] != r.Version {
				bad("%s: reads %s@%d where %s@%d is in force", at, r, r.Version, r, state[r.key()])
			}
		}
		for _, w := range o.Writes {
			if w.Version != state[w.key()]+1 {
				bad("%s: writes %s@%d over %s@%d", at, w, w.Version, w, state[w.key()])
			}
			state[w.key()] = w.Version
		}
		for _, w := range o.Weight {
			s, ok := f.Inventory[w.Role]
			switch {
			case !ok:
				bad("%s: binds %v, which the container does not store for the block", at, w.Role)
			case w.From < 0 || w.From > 0 && (o.Kind != OpEmbed || w.Sheets != 0):
				bad("%s: binds %v from row %d, which only an embedding of one matrix reads", at, w.Role, w.From)
			case o.Kind == OpEmbed && w.Sheets == 0:
				// An embedding reads rows From to From+Rows of its table.
				if s.K != w.K || s.Sheets != 0 || w.From+w.Rows > s.Rows {
					bad("%s: binds rows %d to %d of %v, stored %dx%dx%d", at, w.From, w.From+w.Rows, w.Role, s.K, s.Rows, s.Sheets)
				}
			case s != (Shape{w.K, w.Rows, w.Sheets}):
				bad("%s: binds %v as %dx%dx%d, stored %dx%dx%d", at, w.Role, w.K, w.Rows, w.Sheets, s.K, s.Rows, s.Sheets)
			}
		}
		switch o.Kind {
		case OpMatVec:
			if len(o.In) == 2 && len(o.Out) == 1 && len(o.Weight) == 1 {
				if w := o.Weight[0]; w.K != width(o.In[0]) || width(o.In[1]) != width(o.Out[0]) || g.Values[o.In[1]].Elem != ElemU32 {
					bad("%s: %v of rows of %d selected by %d ids over a row of %d into %d", at, w.Role, w.K, width(o.In[1]), width(o.In[0]), width(o.Out[0]))
				}
				break
			}
			if len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) == 0 || len(o.Weight) > 2 {
				bad("%s: wants one input, one output, a matrix and at most a bias", at)
				break
			}
			if w := o.Weight[0]; w.K != width(o.In[0]) || w.Rows != width(o.Out[0]) {
				bad("%s: %v is %dx%d between widths %d and %d", at, w.Role, w.K, w.Rows, width(o.In[0]), width(o.Out[0]))
			}
			if len(o.Weight) == 2 && (o.Weight[1].Rows != 0 || o.Weight[1].K != width(o.Out[0])) {
				bad("%s: bias %v of %d on a row of %d", at, o.Weight[1].Role, o.Weight[1].K, width(o.Out[0]))
			}
		case OpNorm, OpLayerNorm:
			if len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) == 0 {
				bad("%s: wants one input, one output and a scale", at)
				break
			}
			if o.Weight[0].K != width(o.In[0]) || width(o.Out[0]) != width(o.In[0]) {
				bad("%s: a scale of %d over a row of %d into %d", at, o.Weight[0].K, width(o.In[0]), width(o.Out[0]))
			}
			if o.Kind == OpLayerNorm && (len(o.Weight) > 2 || len(o.Weight) == 2 && (o.Weight[1].Rows != 0 || o.Weight[1].K != width(o.In[0]))) {
				bad("%s: %d weights where a scale and a bias of the row's %d are", at, len(o.Weight), width(o.In[0]))
			}
		case OpHeadNorm:
			if len(o.In) != 1 || len(o.Out) != 1 || o.Attr.HeadDim <= 0 || width(o.In[0])%o.Attr.HeadDim != 0 {
				bad("%s: heads of %d over a row of %d", at, o.Attr.HeadDim, width(o.In[0]))
				break
			}
			if len(o.Weight) > 0 && o.Weight[0].K != o.Attr.HeadDim {
				bad("%s: a scale of %d over heads of %d", at, o.Weight[0].K, o.Attr.HeadDim)
			}
		case OpExpr:
			if len(o.Out) != 1 || len(o.Attr.Expr) == 0 {
				bad("%s: wants one output and a term", at)
				break
			}
			for _, id := range o.In {
				if w := width(id); w <= 0 || width(o.Out[0])%w != 0 {
					bad("%s: an operand of %d under a row of %d", at, w, width(o.Out[0]))
				}
			}
			for k, t := range o.Attr.Expr {
				n := t.Fn.Arity()
				if n == 0 {
					bad("%s: term %d has no function", at, k)
				}
				for j, a := range t.Args {
					ok := true
					switch {
					case j >= n:
						ok = a.Kind == ArgNone
					case a.Kind == ArgIn:
						ok = a.Index >= 0 && a.Index < len(o.In)
					case a.Kind == ArgWeight:
						ok = a.Index >= 0 && a.Index < len(o.Weight) && o.Weight[a.Index].Rows <= 1 && a.Elem >= 0 && a.Elem < o.Weight[a.Index].K
					case a.Kind == ArgTerm:
						ok = a.Index >= 0 && a.Index < k
					default:
						ok = a.Kind == ArgConst
					}
					if !ok {
						bad("%s: term %d's operand %d does not name one the op has", at, k, j)
					}
				}
			}
		case OpFrames:
			if len(o.In) != 0 || len(o.Out) != 1 || o.Attr.Width < 1 || o.Attr.Stride < 1 ||
				g.Values[o.Out[0]].Width != o.Attr.Width {
				bad("%s: frames of %d samples every %d into %%%s", at, o.Attr.Width, o.Attr.Stride, g.Values[o.Out[0]].Name)
			}
		case OpPicture:
			a := o.Attr
			switch {
			case len(o.In) != 0 || len(o.Out) != 1 || len(o.Weight) != 0:
				bad("%s: a picture reads the step's and writes one value", at)
			case a.Pixels < PixRGBA8 || a.Pixels >= PixFormats:
				bad("%s: a picture of sample format %d", at, a.Pixels)
			case a.PicW < 1 || a.PicH < 1:
				bad("%s: a picture resampled to %dx%d", at, a.PicW, a.PicH)
			case a.PicArea < 0 || a.PicArea > a.PicW*a.PicH || a.PicArea > 0 && !a.PicLive:
				bad("%s: a picture of at most %d pixels resampled to %dx%d, its size %v", at, a.PicArea, a.PicW, a.PicH, a.PicLive)
			case a.PicPad < 0 || a.PicPad > 0 && (!a.PicLive || a.PicW%a.PicPad != 0 || a.PicH%a.PicPad != 0):
				bad("%s: a picture padded to units of %d, its size %v and at most %dx%d", at, a.PicPad, a.PicLive, a.PicW, a.PicH)
			case a.PicBox && (!a.PicLive || a.PicPad != 0) || !a.PicBox && a.PicFill != [3]uint8{}:
				bad("%s: a picture boxed %v (filled %v), its size %v, padded to %d", at, a.PicBox, a.PicFill, a.PicLive, a.PicPad)
			case !(a.Chain == ChainHalfPixel && a.Filter == FilterLinear) &&
				!(a.Chain == ChainPIL && (a.Filter == FilterBilinear || a.Filter == FilterCubic) && a.Taps > 0 && a.PicSide > 0):
				bad("%s: a picture resampled by chain %d and filter %d (%d taps)", at, a.Chain, a.Filter, a.Taps)
			case a.Std[0] == 0 || a.Std[1] == 0 || a.Std[2] == 0:
				bad("%s: a picture normalised by a std of 0", at)
			case a.PicPass > PicCols || a.PicPass != PicWhole && a.Chain != ChainPIL:
				bad("%s: a picture's pass %d by chain %d", at, a.PicPass, a.Chain)
			case a.PicColumns && a.PicPass != PicBand:
				bad("%s: a picture's pass %d reading its column bands' sums", at, a.PicPass)
			}
		case OpAdd:
			if len(o.In) != 2 || len(o.Out) != 1 || width(o.In[0]) != width(o.In[1]) || width(o.Out[0]) != width(o.In[0]) {
				bad("%s: widths %d and %d into %d", at, width(o.In[0]), width(o.In[1]), width(o.Out[0]))
			}
		case OpUnfold:
			if t := o.Attr.Taps; len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) != 0 || o.Attr.GridW < 1 || t < 1 || t%2 == 0 ||
				width(o.Out[0]) < t*t*width(o.In[0]) {
				bad("%s: a %dx%d window over a grid %d wide of rows of %d into %d", at, t, t, o.Attr.GridW, width(o.In[0]), width(o.Out[0]))
			}
		case OpUniform:
			switch {
			case len(o.In) != 2 || len(o.Out) != 1 || len(o.Weight) != 0:
				bad("%s: %d inputs into %d outputs", at, len(o.In), len(o.Out))
			case width(o.In[1]) != 2 || width(o.Out[0]) != width(o.In[0]):
				bad("%s: %d keys and a counter of %d into %d uniforms", at, width(o.In[0]), width(o.In[1]), width(o.Out[0]))
			case g.Values[o.In[0]].Elem != ElemU32 || g.Values[o.In[1]].Elem != ElemU32 || g.Values[o.Out[0]].Elem != ElemF32:
				bad("%s: keys and a counter not of words, or uniforms not of floats", at)
			}
		case OpPut:
			switch {
			case len(o.In) != 3 || len(o.Out) != 1 || len(o.Weight) != 0:
				bad("%s: %d inputs into %d outputs", at, len(o.In), len(o.Out))
			case width(o.In[1]) != width(o.In[2]) || width(o.Out[0]) != width(o.In[0]):
				bad("%s: %d indices and %d values into a row of %d, out %d", at, width(o.In[1]), width(o.In[2]), width(o.In[0]), width(o.Out[0]))
			case g.Values[o.In[1]].Elem != ElemU32 || g.Values[o.In[2]].Elem != g.Values[o.In[0]].Elem:
				bad("%s: indices not of words, or values not of the row's elements", at)
			}
		case OpPosition:
			if len(o.In) != 0 || len(o.Out) != 1 || width(o.Out[0]) != 1 || o.Attr.Offset < 0 || o.Attr.Span < 0 {
				bad("%s: %d inputs into %d outputs, offset %d and span %d", at, len(o.In), len(o.Out), o.Attr.Offset, o.Attr.Span)
			}
		case OpNGramWindow:
			a := o.Attr
			switch {
			case len(o.In) != 1 || len(o.Out) != 1 || width(o.In[0]) != 1 || width(o.Out[0]) != a.Taps || a.Taps < 2:
				bad("%s: %d inputs into %d outputs, a window of %d", at, len(o.In), len(o.Out), a.Taps)
			case g.Values[o.In[0]].Elem != ElemU32 || g.Values[o.Out[0]].Elem != ElemU32 || len(a.Words) != 1:
				bad("%s: a token and window not of indices, or %d constants", at, len(a.Words))
			case len(o.Reads) != 1 || len(o.Writes) != 1 || o.Reads[0].State != StateRecState || o.Writes[0].State != StateRecState:
				bad("%s: a window reads and writes the recurrent state", at)
			case a.Offset < 0 || a.Span < a.Offset+a.Taps-1:
				bad("%s: %d predecessors at %d of a slot of %d", at, a.Taps-1, a.Offset, a.Span)
			}
		case OpToken:
			if len(o.In) != 0 || len(o.Out) != 1 || width(o.Out[0]) != 1 || g.Values[o.Out[0]].Elem != ElemU32 {
				bad("%s: %d inputs into %d outputs, not one index a row", at, len(o.In), len(o.Out))
			}
		case OpHalves, OpWidenHalves:
			var words, floats ValueID
			if len(o.In) == 1 && len(o.Out) == 1 {
				words, floats = o.Out[0], o.In[0]
				if o.Kind == OpWidenHalves {
					words, floats = o.In[0], o.Out[0]
				}
			}
			if len(o.In) != 1 || len(o.Out) != 1 || g.Values[words].Elem != ElemU32 || g.Values[floats].Elem == ElemU32 ||
				width(floats) != 2*width(words) || width(words) < 1 {
				bad("%s: %d inputs into %d outputs, not floats and their halves two a word", at, len(o.In), len(o.Out))
			}
		case OpQ8Rows, OpWidenQ8Rows:
			var words, floats ValueID
			if len(o.In) == 1 && len(o.Out) == 1 {
				words, floats = o.Out[0], o.In[0]
				if o.Kind == OpWidenQ8Rows {
					words, floats = o.In[0], o.Out[0]
				}
			}
			if len(o.In) != 1 || len(o.Out) != 1 || g.Values[words].Elem != ElemU32 || g.Values[floats].Elem == ElemU32 ||
				width(floats) < 1 || width(words) != Q8RowWords(width(floats)) {
				bad("%s: %d inputs into %d outputs, not rows of floats and their q8 rows", at, len(o.In), len(o.Out))
			}
		case OpNGramHash:
			n, per := 0, o.Attr.Span
			if len(o.In) == 1 {
				n = width(o.In[0])
			}
			switch {
			case len(o.In) != 1 || len(o.Out) != 2 || n < 2 || per < 1 || len(o.Attr.Words) != 1+n+2*(n-1)*per:
				bad("%s: %d inputs into %d outputs, a window of %d, %d heads a gram, %d constants", at, len(o.In), len(o.Out), n, per, len(o.Attr.Words))
			case width(o.Out[0]) != (n-1)*per || width(o.Out[1]) != n-1:
				bad("%s: %d rows and %d predecessors from a window of %d", at, width(o.Out[0]), width(o.Out[1]), n)
			case g.Values[o.In[0]].Elem != ElemU32 || g.Values[o.Out[0]].Elem != ElemU32 || g.Values[o.Out[1]].Elem != ElemU32:
				bad("%s: a window, rows and predecessors not of indices", at)
			}
		case OpKVAppend:
			if o.Attr.Span < 0 {
				bad("%s: an entry each %d positions", at, o.Attr.Span)
			}
		case OpSelect:
			a := o.Attr
			switch {
			case len(o.In) < 1 || len(o.In) > 2 || len(o.Out) != 1 || len(o.Reads) != 1:
				bad("%s: %d inputs, %d outputs and %d histories", at, len(o.In), len(o.Out), len(o.Reads))
			case a.NHead < 1 || a.NKVHead < 1 || a.NHead%a.NKVHead != 0 || a.HeadDim < 1 || a.Span < 1 || a.TopK < 1 || a.Local < 0:
				bad("%s: %d heads in %d groups of %d, blocks of %d, %d kept, %d local", at, a.NHead, a.NKVHead, a.HeadDim, a.Span, a.TopK, a.Local)
			case a.Offset < 0 || a.Offset+a.HeadDim > a.Width:
				bad("%s: a key of %d at %d in a row of %d", at, a.HeadDim, a.Offset, a.Width)
			case width(o.In[0]) != a.NHead*a.HeadDim || len(o.In) == 2 && width(o.In[1]) != a.NHead:
				bad("%s: a query of %d and %d weights over %d heads of %d", at, width(o.In[0]), len(o.In)-1, a.NHead, a.HeadDim)
			case width(o.Out[0]) != a.NKVHead || g.Values[o.Out[0]].Elem != ElemF32:
				bad("%s: a selection of %d over %d groups", at, width(o.Out[0]), a.NKVHead)
			case a.Pooled && (a.Local != 0 || len(o.Weight) != 1 || o.Weight[0].K != a.HeadDim || a.NRot < 0 || a.NRot%2 != 0 || a.NRot > a.HeadDim):
				bad("%s: pooled blocks with %d local, %d norm weights, a rotary of %d over keys of %d", at, a.Local, len(o.Weight), a.NRot, a.HeadDim)
			case !a.Pooled && len(o.Weight) != 0:
				bad("%s: %d weights on a selection that pools no key", at, len(o.Weight))
			case a.Pooled && a.Firsts:
				bad("%s: pooled blocks scored by their first positions", at)
			}
		case OpAttend:
			if len(o.In) == 2 {
				if p := producer(g, o.In[1]); p == nil || p.Kind != OpSelect || o.Attr.NHead%p.Attr.NKVHead != 0 || len(p.Reads) != 1 || len(o.Reads) < 1 || p.Reads[0] != o.Reads[len(o.Reads)-1] {
					bad("%s: a selection that is not one of its own history's groups", at)
				}
			}
			if a := o.Attr; a.Width != 0 && a.Width < a.NKVHead*a.HeadDim {
				bad("%s: a history row of %d under %d heads of %d", at, a.Width, a.NKVHead, a.HeadDim)
			}
			if a := o.Attr; len(o.Reads) == 2 && (a.Span < 1 || len(a.Words) != 1 || int(a.Words[0]) < a.NKVHead*a.HeadDim ||
				o.Reads[0].Hist == o.Reads[1].Hist) || len(o.Reads) > 2 {
				bad("%s: %d histories, the second an entry each %d positions of %d words", at, len(o.Reads), a.Span, len(a.Words))
			}
		case OpHistoryPool:
			a := o.Attr
			switch {
			case len(o.In) != 0 || len(o.Out) != 1 || len(o.Reads) != 1 || len(o.Weight) != 1:
				bad("%s: %d inputs, %d outputs, %d histories and %d biases", at, len(o.In), len(o.Out), len(o.Reads), len(o.Weight))
			case a.Span < 1 || len(a.Words) < 4 || len(a.Words)%4 != 0 || o.Weight[0].Rows != a.Span:
				bad("%s: windows of %d, %d segment words, a bias of %d rows", at, a.Span, len(a.Words), o.Weight[0].Rows)
			default:
				w := width(o.Out[0])
				for i := 0; i < len(a.Words); i += 4 {
					if a.Words[i] > 1 || int(a.Words[i+1])+w > a.Width || int(a.Words[i+2])+w > a.Width || int(a.Words[i+3])+w > o.Weight[0].K {
						bad("%s: segment %d of %d channels past a row of %d or a bias of %d", at, i/4, w, a.Width, o.Weight[0].K)
					}
				}
			}
		case OpDeltaRule:
			a := o.Attr
			switch {
			case len(o.In) != 5 || len(o.Out) != 1:
				bad("%s: %d inputs and %d outputs", at, len(o.In), len(o.Out))
			case a.NHead < 1 || a.NKVHead < 1 || a.HeadDim < 1 || a.NHead%a.NKVHead != 0:
				bad("%s: %d value heads over %d key heads of %d", at, a.NHead, a.NKVHead, a.HeadDim)
			case width(o.In[0]) != a.NKVHead*a.HeadDim || width(o.In[1]) != width(o.In[0]):
				bad("%s: q and k of %d and %d over %d heads of %d", at, width(o.In[0]), width(o.In[1]), a.NKVHead, a.HeadDim)
			case width(o.In[2]) < a.NHead || width(o.In[2])%a.NHead != 0 || width(o.Out[0]) != width(o.In[2]):
				bad("%s: v of %d into %d over %d heads", at, width(o.In[2]), width(o.Out[0]), a.NHead)
			case width(o.In[3]) != a.NHead && width(o.In[3]) != a.NHead*a.HeadDim:
				bad("%s: a decay of %d over %d heads of %d", at, width(o.In[3]), a.NHead, a.HeadDim)
			case width(o.In[4]) != a.NHead:
				bad("%s: a beta of %d over %d heads", at, width(o.In[4]), a.NHead)
			case a.Span != 0 && a.Span < width(o.In[2])*a.HeadDim:
				bad("%s: a state of %d floats a sequence, under its %d", at, a.Span, width(o.In[2])*a.HeadDim)
			}
		case OpMap:
			if a := o.Attr; a.Width != 0 {
				switch {
				case len(o.Weight) != 1:
					bad("%s: a window of %d on %d bound vectors", at, a.Width, len(o.Weight))
				case a.Width < 0 || a.Offset < 0 || a.Offset+a.Width > o.Weight[0].K*max(o.Weight[0].Rows, 1):
					// A stored table of several rows is its elements in order.
					bad("%s: a window of %d from %d on a vector of %d", at, a.Width, a.Offset, o.Weight[0].K*max(o.Weight[0].Rows, 1))
				}
			}
		}
	}
	for _, id := range g.Out {
		if !valid(id) || !defined[id] {
			bad("graph: block %d's output %d is never written", g.Block, id)
		}
	}
	return errors.Join(errs...)
}

// producer is the op that writes v, nil for none.
func producer(g *Graph, v ValueID) *Op {
	for i := range g.Ops {
		if slices.Contains(g.Ops[i].Out, v) {
			return &g.Ops[i]
		}
	}
	return nil
}

// Unbound is every tensor role f's inventory stores that no op of g binds,
// in role order: a stored tensor no op reads is a graph that dropped part of
// the block.
func Unbound(g *Graph, f Facts) []jlm.Role {
	bound := map[jlm.Role]bool{}
	for _, o := range g.Ops {
		for _, w := range o.Weight {
			bound[w.Role] = true
		}
	}
	var out []jlm.Role
	for r := range f.Inventory {
		if !bound[r] {
			out = append(out, r)
		}
	}
	slices.Sort(out)
	return out
}
