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
const Version = 1

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
	// optional Weight[1] its bias added after.
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
	// OpActMul is Act(In[0]) * In[1], elementwise.
	OpActMul
	// OpExpertMatVec is OpMatVec per pair: row j of In[0] against the sheet
	// of Weight[0]'s bank the selection In[1] names at pair j, an optional
	// Weight[1] the bank's bias (one row per expert) added after.
	OpExpertMatVec
	// OpLayerNorm is LayerNorm over a row: (x - mean) * rsqrt(var + Eps) * w,
	// plus an optional bias Weight[1] (the classic block's, C6).
	OpLayerNorm
	// OpAct is Act(In[0]) elementwise.
	OpAct
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
	// In[2] in order, then Weight[0] (a bound vector), then Const, as many as
	// Fn takes. An operand narrower than Out[0] is broadcast: element i reads
	// its element i/(w/n), w the output's width and n the operand's (one
	// element a group of w/n; width 1 is a scalar). A primitive
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
	// sum of their widths. The same input twice repeats it. A primitive.
	OpConcat
	// OpRopeTable is a step's rotary table: for each row's position and each
	// of NRot/2 pairs, the pair's cos and sin, Out[0] two elements a pair.
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
	// a recurrent mixer's convolution (DecomposeRecurrent).
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
	// rows -- a bidirectional model's short convolution (LFM2's encoder).
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
	numOps
)

var opNames = [...]string{"invalid", "norm", "head_norm", "matvec", "rope", "kv_append", "attend", "add", "act_mul", "expert_matvec", "layer_norm", "act", "clamp", "slice", "scale", "conv", "affine", "map", "reduce", "top_k", "take", "rank", "tile", "concat", "rope_table", "embed", "state_conv", "pool", "take_rows", "row_conv", "patchify", "regroup", "lead_row", "drop_rows", "fold", "splice", "pos_resample"}

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
)

func (l RopeLayout) String() string {
	switch l {
	case RopeNorm:
		return "norm"
	case RopeNeox:
		return "neox"
	}
	return fmt.Sprintf("layout%d", l)
}

// Act is an activation an OpActMul applies to its first input.
type Act uint8

const (
	ActSiLU Act = iota + 1
	ActGELUTanh
	// ActSigmoid is sigma(In[0]) * In[1]: a gated norm's sigmoid gate.
	ActSigmoid
	// ActGELUErf is the exact (erf) GELU.
	ActGELUErf
	// ActReLU2 is relu(x)^2.
	ActReLU2
	// ActReLU is max(x, 0).
	ActReLU
	// ActQuickGELU is x*sigma(1.702x), OpenAI's CLIP's.
	ActQuickGELU
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
)

// MapFn is an OpMap's function, or an OpReduce's combiner.
type MapFn uint8

const (
	MapAdd      MapFn = iota + 1 // a + b
	MapSub                       // a - b
	MapMul                       // a * b
	MapDiv                       // a / b
	MapMax                       // max(a, b)
	MapMin                       // min(a, b)
	MapExp                       // exp(a)
	MapSigmoid                   // 1 / (1 + exp(-a))
	MapSoftplus                  // log(1 + exp(a))
	MapSqrt                      // sqrt(a)
	MapWhere                     // b where a > 0, else c
	MapLess                      // 1 where a < b, else 0
	MapFma                       // a*b + c, the product not rounded before the sum
)

// Arity is how many operands f takes.
func (f MapFn) Arity() int {
	switch f {
	case MapExp, MapSigmoid, MapSoftplus, MapSqrt:
		return 1
	case MapWhere, MapFma:
		return 3
	case MapAdd, MapSub, MapMul, MapDiv, MapMax, MapMin, MapLess:
		return 2
	}
	return 0
}

var mapFnNames = [...]string{"?", "add", "sub", "mul", "div", "max", "min", "exp", "sigmoid", "softplus", "sqrt", "where", "less", "fma"}

func (f MapFn) String() string {
	if int(f) < len(mapFnNames) {
		return mapFnNames[f]
	}
	return fmt.Sprintf("fn%d", f)
}

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
	// Interleaved assigns the axes pair by pair rather than in contiguous
	// runs. The row's positions on each axis are the row's, so
	// the op is the same rotation over a table built per row.
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
	Act     Act
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
	// RowWindows makes OpAttend read each row's attended range from the
	// step's window table, not its history descriptor: a vision tower's
	// window attention, a row's keys its window's rows.
	RowWindows bool
	// Bound is OpClamp's.
	Bound float64
	// Offset and Width are OpSlice's; Offset is the leading rows of
	// OpPatchify, OpLeadRow and OpDropRows too.
	Offset, Width int
	// Fn, Order, Span and Const are the primitives' (OpMap, OpReduce,
	// OpTopK, OpRank, OpTile): the function, the order a reduction runs in, the
	// group a reduction, top-k or rank runs over (0: the row), a constant
	// operand.
	Fn    MapFn
	Order Order
	Span  int
	Const float64
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
