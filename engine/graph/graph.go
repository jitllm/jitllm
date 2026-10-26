// Package graph is the block graph: a versioned, typed SSA description of
// one block's values and persistent state effects, built from what the
// container states (docs/design/composer.md, "Above the recipes"). It holds
// data and a verifier and computes nothing: a graph is lowered to the
// generated operations the engine already runs, never interpreted here.
package graph

import (
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
)

// Version is the op language this package writes and the verifier reads. A
// graph of another version is refused, never guessed at.
const Version = 2

// OpKind is an operation. Each kind's semantics are fixed per Version: a
// change of meaning is a new kind or a new Version, not a reinterpretation.
type OpKind uint8

const (
	OpInvalid OpKind = iota
	// OpNorm is RMSNorm over a row: x * rsqrt(mean(x^2) + Eps) * w, the
	// epsilon inside the root, accumulated in f32. Weight[0] is the scale.
	OpNorm
	// OpHeadNorm is OpNorm over each head of HeadDim within the row, one
	// scale of HeadDim shared by every head; with no Weight the scale is
	// ones.
	OpHeadNorm
	// OpMatVec is out = W x, W's rows the output; Weight[0] is W, an
	// optional Weight[1] its bias added after. With a selection In[1] (u32)
	// and no bias, element j of Out[0] is W's row In[1][j] times x: the rows
	// it names alone, a name past W's rows read as its last.
	OpMatVec
	// OpRope turns each head's first NRot elements by the position, in the
	// pair Layout, at Base.
	OpRope
	// OpKVAppend writes k and v (In[0], In[1]) as this position's row of the
	// block's KV: it reads state version n and writes n+1.
	OpKVAppend
	// OpAttend is causal attention of q (In[0]) over the KV version it reads,
	// GQA by the head counts, scaled by Scale, over a Window (0: the whole
	// history), scores capped by Softcap (0: none); an optional Weight[0]
	// (RoleAttnSinks) is a per-head logit in the softmax's denominator.
	OpAttend
	// OpAdd is In[0] + In[1].
	OpAdd
	// OpExpr is an elementwise function of its operands, stated as terms
	// (Attr.Expr) in order: each term's Fn over its Args -- In[j] (an operand
	// narrower than Out[0] broadcast as OpMap's), element Elem of the bound
	// vector Weight[j] read as a scalar, a constant or an earlier term.
	// Out[0] is the last term. A primitive: an activation the container
	// states, whatever its formula.
	OpExpr
	// OpExpertMatVec is OpMatVec per pair: row j of In[0] against the sheet
	// of Weight[0]'s bank the selection In[1] names at pair j, an optional
	// Weight[1] the bank's bias (one row per expert) added after. With no
	// selection, pair j reads sheet j: a matvec per group of the row, the
	// bank's sheets in order, one a group.
	OpExpertMatVec
	// OpLayerNorm is LayerNorm over a row: (x - mean) * rsqrt(var + Eps) * w,
	// plus an optional bias Weight[1] (the classic block's, C6).
	OpLayerNorm
	// OpClamp is In[0] clamped to [-Bound, Bound], or with a bounds table
	// bound (Weight[0]) to [t[Offset], t[Offset+1]], bounds not necessarily
	// symmetric.
	OpClamp
	// OpSlice is Width elements of In[0] from Offset: of each head of
	// HeadDim where HeadDim is set (Out is the heads' slices in head
	// order), of the whole row where it is 0. An op with no input slices
	// the bound vector Weight[0] instead (a stored table's row).
	OpSlice
	// OpScale multiplies In[0] by the stored scalar Weight[0].
	OpScale
	// OpConv is a depthwise convolution of In[0] over the call's rows: each
	// channel's Taps taps (Weight[0], Taps rows of the width: tap-major),
	// centred, the rows past
	// either end zero.
	OpConv
	// OpAffine is In[0] times a per-channel scale plus a shift (Weight[0],
	// Weight[1]): a BatchNorm folded at conversion.
	OpAffine
	// OpMap is an elementwise function Fn of its operands: In[0], In[1] and
	// In[2] in order, then Weight[0] (a bound vector, a stored table of
	// several rows its elements in order), then Const, as many as
	// Fn takes. An operand narrower than Out[0] is broadcast: element i reads
	// its element i/(w/n), w the output's width and n the operand's (one
	// element a group of w/n; width 1 is a scalar). The bound vector is its
	// Width elements from Offset where Width is set, the whole vector where it
	// is 0: Width 1 reads one element of a stored table as a scalar, a
	// per-block constant a buffer holds. A primitive
	// (docs/design/composer.md, "Decomposition").
	OpMap
	// OpReduce combines each group of Span elements of In[0] (the whole row
	// where Span is 0) by Fn (MapAdd, MapMax or MapMin) in Order: Out[0] one
	// element a group. A primitive.
	OpReduce
	// OpTopK is the TopK largest elements of each group of Span of In[0]
	// (the whole row where Span is 0), largest first, equal values to the
	// lower index: Out[0] their indices within the group (u32), Out[1] their
	// values. A primitive.
	OpTopK
	// OpTake is In[0]'s element at each index of In[1] (u32): Out[0] one
	// element an index. A primitive.
	OpTake
	// OpRank is each element's place among its group of Span of In[0] (the
	// whole row where Span is 0), largest first, equal values to the lower
	// index (u32): OpTopK's order, for every element. A primitive.
	OpRank
	// OpTile is In[0] repeated: element i of Out[0] is In[0]'s element
	// i mod Span (Span 0: the whole row), Out[0]'s width a multiple of Span.
	// A primitive, a periodic index map: with OpSlice's offsets and OpMap's
	// group broadcast it states a row seen as several (a stream-major
	// residual) without naming what the groups are.
	OpTile
	// OpConcat is its inputs joined end to end, In[0] first: Out[0] the
	// sum of their widths. The same input twice repeats it. Where NHead is
	// set, each input is NHead heads of its width over NHead and they are
	// joined head by head: head h of Out[0] is each input's head h in turn.
	// A primitive.
	OpConcat
	// OpRopeTable is a step's rotary table: for each row's position and each
	// of NRot/2 pairs, the pair's cos and sin, Out[0] two elements a pair.
	// Where Sections states several axes, a row has a position on each, and
	// a pair's angle is its axis's position's.
	OpRopeTable
	// OpEmbed is a step's input rows: each row's token's row of Weight[0],
	// the token a word of the step's own a row; where Positional, each
	// row's position's row (a learned position table); where Request, the
	// row the step's request names, every row's (a decision's question
	// type's).
	OpEmbed
	// OpStateConv is a causal depthwise convolution of In[0] over the
	// block's window, the last Taps-1 rows it saw: each channel's Taps taps
	// (Weight[0], channel-major, the oldest first) against the window's rows
	// and the row, Out[0] a row of the input's width; the window then drops
	// its oldest row and takes the row. A primitive with a state of its own:
	// a recurrent mixer's convolution.
	OpStateConv
	// OpPool is the step's live rows of In[0] pooled into one row, Out[0]
	// it in every row: their mean, the first's or the last's (Pooling, a
	// jlm.Pooling). A primitive: an encoder's sequence into its embedding.
	OpPool
	// OpTakeRows is the rows of In[0] the step's request names, in its
	// order, Out[0]'s first rows (a decision's option markers); the rows
	// past them repeat the last.
	OpTakeRows
	// OpRowConv is a centred depthwise convolution of In[0] across the
	// step's live rows: each channel's Taps taps (Weight[0], channel-major)
	// against the rows Taps/2 before to Taps/2 after, zero past the live
	// rows -- a bidirectional model's short convolution (an encoder's).
	OpRowConv
	// OpPatchify is a picture's patches as rows: In[0] is the picture,
	// [y][x][channel] over the step's rows, a grid of Span x Span patches
	// whose width in patches is the request's first word; row Offset+p is
	// patch p in raster order, its Span*Span*3 elements channel-planar (a
	// convolution whose stride is its kernel, as a matrix product), zero past
	// them to the output's width and in the Offset rows before.
	OpPatchify
	// OpRegroup is space-to-depth over the step's grid of rows (the
	// request's first word its width): row g of the output is the Span x Span
	// block of rows at g's place, each Span*Span slot in raster order a run
	// of the input's width.
	OpRegroup
	// OpLeadRow is In[0] with its first Offset rows each Weight[0]'s one row
	// (a tower's class row).
	OpLeadRow
	// OpDropRows is In[0] from row Offset on: output row r is input row
	// r+Offset.
	OpDropRows
	// OpFold is each Span consecutive rows of In[0] as one row.
	OpFold
	// OpSplice is In[0] with a picture's rows in place: row r takes the row
	// of In[1] (the vision segment's projector, a picture's step's rows)
	// that the step's request names for it, where it names one.
	OpSplice
	// OpPosResample is each row's position-table row resampled to the
	// step's grid: Weight[0] a Span x Span table, the row's cell (its
	// position less Offset, a grid the request's first word wide and the
	// live rows past Offset tall) mapped to a source coordinate on each axis
	// by Chain and read through Filter's taps.
	OpPosResample
	// OpDeltaRule is the delta rule over the block's recurrent state: for
	// each value head h of NHead, a matrix S of Dv rows (v's width over
	// NHead) by HeadDim, paired with key head h/(NHead/NKVHead) of NKVHead,
	// or h mod NKVHead where Tiled. q and k (In[0], In[1]) are NKVHead heads
	// of HeadDim, v (In[2]) NHead heads of Dv, the decay In[3] one factor a
	// value head or one a key channel of each (NHead or NHead*HeadDim wide)
	// and beta In[4] one a value head. Each row, in order: S times the
	// decay (a channel's factor on its column), then S plus the outer
	// product of (v - S k) times beta and k (v times beta where Additive),
	// then Out[0] = S q. It reads
	// state version n and writes n+1. A primitive with a state of its own:
	// a linear-attention mixer's recurrence.
	OpDeltaRule
	// OpPosition is each row's position in the block's history, the one its
	// key is written at, plus Offset and in whole Spans where Span > 1 (an
	// integer division), as a float: Out[0] one element a row. A primitive.
	OpPosition
	numOps
)

var opNames = [...]string{"invalid", "norm", "head_norm", "matvec", "rope", "kv_append", "attend", "add", "expr", "expert_matvec", "layer_norm", "clamp", "slice", "scale", "conv", "affine", "map", "reduce", "top_k", "take", "rank", "tile", "concat", "rope_table", "embed", "state_conv", "pool", "take_rows", "row_conv", "patchify", "regroup", "lead_row", "drop_rows", "fold", "splice", "pos_resample", "delta_rule", "position"}

func (k OpKind) String() string {
	if int(k) < len(opNames) {
		return opNames[k]
	}
	return fmt.Sprintf("op%d", k)
}

// RopeLayout is how a rotary pairs a head's elements.
type RopeLayout uint8

const (
	// RopeNorm pairs (2i, 2i+1).
	RopeNorm RopeLayout = iota + 1
	// RopeNeox pairs (i, i+NRot/2).
	RopeNeox
	// RopeSplit pairs (i, i+NRot/2) as RopeNeox does, over a table of NRot
	// pairs: element i turns by the table's pair i and element i+NRot/2 by
	// its pair i+NRot/2, so a pair's two elements may turn by two angles
	// (a coordinate an element, XD-RoPE).
	RopeSplit
)

func (l RopeLayout) String() string {
	switch l {
	case RopeNorm:
		return "norm"
	case RopeNeox:
		return "neox"
	case RopeSplit:
		return "split"
	}
	return fmt.Sprintf("layout%d", l)
}

// Term is one function of an OpExpr: Fn over Fn.Arity() of Args.
type Term struct {
	Fn   MapFn
	Args [3]Arg
}

// Arg is a term's operand.
type Arg struct {
	Kind  ArgKind
	Index int     // the input, the bound vector or the earlier term
	Elem  int     // ArgWeight's element of the vector
	Const float64 // ArgConst's
}

// ArgKind is what a term's operand is.
type ArgKind uint8

const (
	ArgNone ArgKind = iota
	ArgIn
	ArgWeight
	ArgConst
	ArgTerm
)

// ValueID names a value in a Graph's Values.
type ValueID int32

// Value is one SSA value: a row of Width elements per token, f32 unless
// Elem says otherwise.
type Value struct {
	Name  string
	Width int
	Elem  Elem
}

// Elem is a value's element type.
type Elem uint8

const (
	// ElemF32 is a float row, every value's before decomposition.
	ElemF32 Elem = iota
	// ElemU32 is an index row: a selection's experts, a top-k's positions.
	ElemU32
	// ElemI8 is an int8 code row: every element an integer in [-128, 127],
	// held as its float. A matrix product over one is over the codes, its
	// scale applied by the ops after it.
	ElemI8
)

// MapFn is an OpMap's or an OpExpr term's function, or an OpReduce's
// combiner: the container's own codes (jlm.Fn), so a stated expression is
// read as it is written.
type MapFn = jlm.Fn

const (
	MapAdd      = jlm.FnAdd      // a + b
	MapSub      = jlm.FnSub      // a - b
	MapMul      = jlm.FnMul      // a * b
	MapDiv      = jlm.FnDiv      // a / b
	MapMax      = jlm.FnMax      // max(a, b)
	MapMin      = jlm.FnMin      // min(a, b)
	MapExp      = jlm.FnExp      // exp(a)
	MapSigmoid  = jlm.FnSigmoid  // 1 / (1 + exp(-a))
	MapSoftplus = jlm.FnSoftplus // log(1 + exp(a))
	MapSqrt     = jlm.FnSqrt     // sqrt(a)
	MapWhere    = jlm.FnWhere    // b where a > 0, else c
	MapLess     = jlm.FnLess     // 1 where a < b, else 0
	MapFma      = jlm.FnFma      // a*b + c, the product not rounded before the sum
	MapExpm1    = jlm.FnExpm1    // exp(a) - 1, exact near zero where the difference cancels
	MapAbs      = jlm.FnAbs      // |a|
	MapRound    = jlm.FnRound    // a to the nearest integer, halves away from zero
	MapRecip    = jlm.FnRecip    // 1 / a
	MapTanh     = jlm.FnTanh     // tanh(a)
	MapErf      = jlm.FnErf      // erf(a)
	MapEqual    = jlm.FnEqual    // 1 where a == b, else 0; over two index rows, their integers
	MapLog      = jlm.FnLog      // the natural logarithm of a
)

// Order is an OpReduce's: the order its additions run in is part of its
// arithmetic, so it is stated.
type Order uint8

const (
	// OrderSeq combines a group's elements one at a time from its first.
	OrderSeq Order = iota
	// OrderTree combines them pairwise, the first half with the second, down
	// to one.
	OrderTree
)

// Binding is a weight an op reads: its container role in the graph's block,
// with the shape the op requires (K the input width, Rows the output; Rows 0
// for a vector of K).
type Binding struct {
	Role    jlm.Role
	K, Rows int
	Sheets  int // a bank's experts, each a K x Rows sheet; 0 for one matrix
}

// StateRef is a version of the block's persistent state.
type StateRef struct {
	Name    string // "kv"
	Version int
}

// Attrs are an op's parameters. Only those its kind defines are read; the
// verifier refuses one set on a kind that does not define it.
type Attrs struct {
	Eps     float64
	HeadDim int
	NRot    int
	Base    float64
	Layout  RopeLayout
	Freqs   bool // the rotary divides each pair's theta by stored factors (the container's rope_freqs)
	// Sections splits the rotary's pairs between a row's position axes
	// (M-RoPE: time, height, width and a fourth); all zero is one axis.
	// Contiguous, Sections[i] consecutive pairs turn by axis i, the
	// sections covering every pair. Interleaved assigns the axes pair by
	// pair: pair p by axis 1 where p%3 == 1 and p < 3*Sections[1], by axis 2
	// where p%3 == 2 and p < 3*Sections[2], by axis 0 otherwise. Pair p's
	// frequency is p's either way. The row's positions on each axis are the
	// row's, so OpRope is the same rotation over a table built per row
	// (OpRopeTable's, which reads them).
	Sections    [4]int
	Interleaved bool
	// Mscale multiplies the rotary's cos and sin; 0 is one.
	Mscale  float64
	NHead   int
	NKVHead int
	Scale   float64
	Window  int
	// Chunked makes Window an aligned chunk: a position attends to the
	// earlier positions of its own chunk of Window, not the Window before it.
	Chunked bool
	// Bidir makes the attention non-causal over the whole sequence: every
	// position attends to every other (an embedding model's encoder pass).
	// With a Window it is the window's full width: a position attends the
	// keys within Window/2 of it either side.
	Bidir   bool
	Softcap float64
	// Expr is OpExpr's terms.
	Expr []Term
	// Experts and TopK are a bank's experts and how many pairs a selection
	// holds (OpExpertMatVec, OpTopK).
	Experts, TopK int
	// Alpha is OpAdd's scale on its second input: x + Alpha*y; 0 is one.
	Alpha float64
	Taps  int
	// Positional makes OpEmbed index its table by the row's position.
	Positional bool
	// Pooling is OpPool's: a jlm.Pooling.
	Pooling uint8
	// Grid is OpRopeTable's (RopeTable.Grid).
	Grid bool
	// Chain and Filter are OpPosResample's: the coordinate's convention
	// and the taps that read it.
	Chain, Filter uint8
	// Tiled pairs OpDeltaRule's value head h with key head h mod NKVHead.
	Tiled bool
	// Additive makes OpDeltaRule's update the decayed S plus the outer
	// product of v times beta and k, with no dot of S against k: a
	// selective state update.
	Additive bool
	// RowWindows makes OpAttend read each row's attended range from the
	// step's window table, not its history descriptor: a vision tower's
	// window attention, a row's keys its window's rows.
	RowWindows bool
	// Bound is OpClamp's.
	Bound float64
	// Offset and Width are OpSlice's; Offset is the leading rows of
	// OpPatchify, OpLeadRow and OpDropRows too, and both are the window of
	// OpMap's bound vector.
	Offset, Width int
	// Fn, Order, Span and Const are the primitives' (OpMap, OpReduce,
	// OpTopK, OpRank, OpTile): the function, the order a reduction runs in, the
	// group a reduction, top-k or rank runs over (0: the row), a constant
	// operand.
	Fn    MapFn
	Order Order
	Span  int
	Const float64
	// Exact asks a MapDiv, an OpMap's or an OpExpr term's, for the IEEE
	// quotient where the device's own division is not correctly rounded;
	// without it a division takes the device's, within the gates' band like
	// every other kernel form.
	Exact bool
}

// Op is one operation.
type Op struct {
	Kind   OpKind
	In     []ValueID
	Out    []ValueID
	Weight []Binding
	Reads  []StateRef
	Writes []StateRef
	Attr   Attrs
}

// Graph is one block.
type Graph struct {
	Version int
	Block   int
	Values  []Value
	Ops     []Op
	In, Out []ValueID // the residual at the block's boundary
}

func (g *Graph) value(name string, width int) ValueID {
	g.Values = append(g.Values, Value{Name: name, Width: width})
	return ValueID(len(g.Values) - 1)
}

func (g *Graph) op(o Op) { g.Ops = append(g.Ops, o) }

// String is the graph in the illustrative form docs/design/composer.md uses.
func (g *Graph) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "block %d (graph v%d)\n", g.Block, g.Version)
	name := func(ids []ValueID) string {
		s := make([]string, len(ids))
		for i, id := range ids {
			if id >= 0 && int(id) < len(g.Values) {
				s[i] = "%" + g.Values[id].Name
			} else {
				s[i] = fmt.Sprintf("%%?%d", id)
			}
		}
		return strings.Join(s, ", ")
	}
	for _, o := range g.Ops {
		fmt.Fprintf(&b, "  %s = %s(%s", name(o.Out), o.Kind, name(o.In))
		for _, w := range o.Weight {
			fmt.Fprintf(&b, ", %s", w.Role)
		}
		for _, r := range o.Reads {
			fmt.Fprintf(&b, ", reads %s@%d", r.Name, r.Version)
		}
		for _, w := range o.Writes {
			fmt.Fprintf(&b, ", writes %s@%d", w.Name, w.Version)
		}
		b.WriteString(")\n")
	}
	return b.String()
}

// A position table's resample to the grid (OpPosResample): the coordinate
// chain and the filter, each step one float32 rounding; 0 is unstated.
const (
	ChainAligned uint8 = 1 + iota
	ChainHalfPixel
)

const (
	FilterBilinear uint8 = 1 + iota
	FilterLinear
	FilterCubic
)
