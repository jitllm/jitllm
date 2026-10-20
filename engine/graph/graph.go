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
	// scale of HeadDim shared by every head (qwen3's q/k norm); with no
	// Weight the scale is ones (Llama 4's L2 norm).
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
	// OpGatedDelta is a gated delta rule over its input row (In[0], the
	// sublayer's normed input): the projections, causal convolution, decay
	// and scan that turn it into one output row per value head (Out[0],
	// VHeads*VDim), reading the recurrence's version n and writing n+1. It is
	// opaque in this version: the rule's own internals are one op, its
	// weights all bound by it, and only what follows it is matched.
	OpGatedDelta
	// OpRoute is a mixture's router over its input row (In[0]): the logits
	// against Weight[0] (Experts rows), an optional bias on them and a
	// selection bias on the choice (Weight[1:], by role), scored by Router,
	// within Groups, the top TopK kept, divided by their sum where RouteNorm,
	// times Scale. Out[0] is the selection: TopK (expert, weight) pairs in
	// the router's order.
	OpRoute
	// OpGather is the row In[0] once per pair of the selection In[1]: Out[0]
	// is TopK rows of the input's width.
	OpGather
	// OpExpertMatVec is OpMatVec per pair: row j of In[0] against the sheet
	// of Weight[0]'s bank the selection In[1] names at pair j, an optional
	// Weight[1] the bank's bias (one row per expert) added after.
	OpExpertMatVec
	// OpCombine is the selection's (In[1]) weighted sum of the pairs In[0],
	// in the selection's order: Out[0] is one row of a pair's width. An
	// optional Weight[0] is a per-expert scale on each pair's weight.
	OpCombine
	// OpGateScalar is sigma(dot(Weight[0], In[0])) * In[1]: a shared
	// expert's output scaled by its gate's sigmoid.
	OpGateScalar
	// OpLayerNorm is LayerNorm over a row: (x - mean) * rsqrt(var + Eps) * w,
	// plus an optional bias Weight[1] (the classic block's, C6).
	OpLayerNorm
	// OpAct is Act(In[0]) elementwise: an ungated FFN's activation. ActXIELU
	// binds its four numbers as Weight[0].
	OpAct
	// OpClamp is In[0] clamped to [-Bound, Bound] (DBRX's clip_qkv), or
	// with a bounds table bound (Weight[0]) to [t[Offset], t[Offset+1]]:
	// Gemma 4's clipped linears, each matrix's input and output bounded by
	// the block's own, not necessarily symmetric, values.
	OpClamp
	// OpSlice is Width elements of In[0] from Offset: of each head of
	// HeadDim where HeadDim is set (Out is the heads' slices in head
	// order), of the whole row where it is 0. An op with no input slices
	// the bound vector Weight[0] instead (a stored table's row).
	OpSlice
	// OpLatentAppend writes a latent row as this position's row of the
	// block's KV (MLA): the cached key is the latent In[0] then the shared
	// rotary key In[1], and the value is the key's latent prefix. It reads
	// state version n and writes n+1.
	OpLatentAppend
	// OpAttendLatent is causal attention of NHead query heads over the
	// latent rows of the KV version it reads, in the reference (naive)
	// form: head h's key is the transpose of Weight[0]'s sheet h (the k
	// up-projection, stored absorbed-side) over the latent, then the row's
	// shared rotary key; its value is Weight[1]'s sheet h over the latent.
	// In[0] is the queries' unrotated parts (NHead heads of HeadDim-NRot),
	// In[1] their rotary parts (NHead heads of NRot); scaled by Scale. Out is
	// NHead values of Weight[1]'s rows. mla-absorb is the rewrite that folds
	// both up-projections into the query and the output.
	OpAttendLatent
	// OpRecurrent is a recurrent mixer over its normed input row (In[0]):
	// Mixer names the kind (the container's layer kind: a state-space scan,
	// a short convolution, Mamba-1's selective scan), Out[0] the mixer's
	// output row at the residual's width, reading the recurrence's version n
	// and writing n+1. It is opaque in this version, as OpGatedDelta is: its
	// weights all bound by it.
	OpRecurrent
	// OpScale multiplies In[0] by the stored scalar Weight[0] (Gemma 4's
	// layer_output_scale on the block's output).
	OpScale
	// OpIndex is an attention indexer (DeepSeek V3.2's lightning indexer)
	// over the normed row (In[0]) and the query latent (In[1]): NHead heads
	// of HeadDim score every cached position against this row's query, and
	// Out[0] is the TopK positions the attention reads. Its key joins its
	// own history: it reads the idx state's version n and writes n+1. It is
	// opaque in this version: the projections, norm, rotary and scoring are
	// one op, its weights all bound by it.
	//
	// Over a GQA attention (MiniMax Sparse Attention, Dense.Select) it has
	// the normed row alone (In[0]): its key is one more kv head of the row
	// the block just appended (it reads kv's version n and writes n+1), and
	// Out[0] is the selection, per kv group the TopK blocks of Block
	// positions by their best score with the Local ending at the query's
	// forced in.
	OpIndex
	// OpStreamMix is a sublayer's read of a residual of Streams streams
	// (In[0], stream-major): Out[0] the sublayer's input row and, with a
	// placement, Out[1] the Streams weights OpStreamPlace adds its output
	// into each stream at (Qwen3.8-Flash-Next's low-rank gated
	// hyper-connection: the streams normed, gated by a low-rank sigmoid and
	// averaged; the placement 2*sigmoid of a projection). Opaque in this
	// version, its weights all bound by it.
	OpStreamMix
	// OpStreamPlace adds a sublayer's output row (In[1]) into every stream
	// of the residual (In[0]) at the weights In[2]: Out[0].
	OpStreamPlace
	// OpNGram adds the n-gram embedding into every stream of the residual
	// (In[0]): the row's token and the NGram-1 before it, hashed into table
	// rows, projected, gated per stream and convolved over the stream's
	// history. It reads and writes the ngram state (the earlier tokens and
	// the convolution's history). Opaque in this version.
	OpNGram
	// OpAltPredict is AltUp's block prologue over Streams streams (In[0],
	// stream-major; Gemma 3n's): a router over the active stream's norm
	// weighs the stored coefficients into each stream's prediction, Out[0]
	// every stream's, Out[1] the active one the block runs on. Opaque in
	// this version.
	OpAltPredict
	// OpAltCorrect is AltUp's epilogue: every prediction (In[0]) corrected
	// by what the block did to the active stream (In[1], its output), and
	// the per-layer input (In[2]) gated from the corrected active stream into
	// the others: Out[0] the streams. Opaque in this version.
	OpAltCorrect
	// OpLaurel is LAuReL's branch of the attention's normed input (In[0]):
	// In[0] + norm(R(L(In[0]))), joined to the attention's residual by an
	// add scaled by 1/sqrt(2).
	OpLaurel
	// OpGaussTopK keeps the FFN gate's values above its mean plus Bound
	// standard deviations, the rest zero (Gemma 3n's activation sparsity).
	OpGaussTopK
	// OpCompressedAttend is an attention whose history keeps pooled entries
	// beside its rows (DeepSeek V4's): over the normed row (In[0]) a
	// low-rank query, one key head that is also the value (turned on its
	// last NRot dimensions, and back after the softmax), sinks and a window;
	// on a compressing block every Block positions pooled into an entry of
	// its second history (Mixer the kind: every visible entry, or the
	// indexer's top-k of them); the heads' output through a grouped low-rank
	// projection: Out[0] the sublayer's output row. It reads and writes the
	// kv state. Opaque in this version, its weights all bound by it.
	OpCompressedAttend
	// OpPredict is a prediction block's prologue: In[0] the trunk's hidden
	// row (every stream of a residual of several), In[1] the next token's
	// embedding row; Out[0] the block's input, eh_proj([enorm(e) ; hnorm(h)])
	// (per stream). Its weights bind the block's own head too (its norm,
	// projection, embedding table and stream mix), which the draft reads
	// after the block. Opaque in this version.
	OpPredict
	// OpCrossAttend is the rows' (In[0], normed) attention over another
	// sequence's rows (In[1], the memory): q from the rows, k and v from the
	// memory, every row over every memory row, then the output projection,
	// each with its bias: Out[0]. Opaque in this version.
	OpCrossAttend
	// OpConv is a depthwise convolution of In[0] over the call's rows: each
	// channel's Taps taps (Weight[0], Taps rows of the width: tap-major),
	// centred, the rows past
	// either end zero.
	OpConv
	// OpAffine is In[0] times a per-channel scale plus a shift (Weight[0],
	// Weight[1]): a BatchNorm folded at conversion.
	OpAffine
	// OpRelAttend is self-attention over the call's rows with relative
	// positions (Transformer-XL's): q (In[0]) plus u (Weight[0]) against
	// the keys (In[1]), plus q plus v (Weight[1]) against the projected
	// positions (In[3]) at each pair's distance, softmaxed over the rows,
	// against the values (In[2]); NHead heads of HeadDim.
	OpRelAttend
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
	numOps
)

var opNames = [...]string{"invalid", "norm", "head_norm", "matvec", "rope", "kv_append", "attend", "add", "act_mul", "gated_delta", "route", "gather", "expert_matvec", "combine", "gate_scalar", "layer_norm", "act", "clamp", "slice", "latent_append", "attend_latent", "recurrent", "scale", "index", "stream_mix", "stream_place", "ngram", "alt_predict", "alt_correct", "laurel", "gauss_top_k", "compressed_attend", "predict", "cross_attend", "conv", "affine", "rel_attend", "map", "reduce", "top_k", "take", "rank", "tile", "concat"}

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
	// RopeXD pairs as NEOX does, each ELEMENT turned by its own coordinate
	// (HunyuanVL's XD-RoPE: sequence, column, row and picture), so a pair's
	// halves may turn by different angles.
	RopeXD
)

func (l RopeLayout) String() string {
	switch l {
	case RopeNorm:
		return "norm"
	case RopeNeox:
		return "neox"
	case RopeXD:
		return "xd"
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
	// ActXIELU is Apertus's xIELU, its four numbers a bound weight.
	ActXIELU
	// ActSwiGLUOAI is gpt-oss's clamped gate: In[0]*sigma(1.702*In[0]) with
	// In[0] clamped above at 7, times (In[1] clamped to [-7, 7]) + 1.
	ActSwiGLUOAI
	// ActSwiGLUClamp is DeepSeek V4's and MiniMax-M3's: SiLU of In[0]
	// clamped above at 10, times In[1] clamped to [-10, 10].
	ActSwiGLUClamp
	// ActSitu is Kimi-K3's: both operands bounded by a scaled tanh.
	ActSitu
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
	Freqs   bool // the rotary divides each pair's theta by stored factors (rope_freqs, YaRN)
	// Sections splits the rotary's pairs between a row's position axes
	// (M-RoPE: time, height, width and a fourth); all zero is one axis.
	// Interleaved assigns the axes pair by pair (Qwen3-VL's) rather than in
	// contiguous runs. The row's positions on each axis are the row's, so
	// the op is the same rotation over a table built per row.
	Sections    [4]int
	Interleaved bool
	// Mscale multiplies the rotary's cos and sin (YaRN's or LongRoPE's
	// magnitude); 0 is one.
	Mscale  float64
	NHead   int
	NKVHead int
	Scale   float64
	Window  int
	// Chunked makes Window an aligned chunk (Llama 4's): a position attends
	// to the earlier positions of its own chunk of Window, not the Window
	// before it.
	Chunked bool
	// BidirOff makes the attention causal over a picture's rows too, where
	// the prompt's spans mark them bidirectional (Gemma 4's global layers;
	// its sliding layers and every other model's honour the spans).
	BidirOff bool
	// Bidir makes the attention non-causal over the whole sequence: every
	// position attends to every other (an embedding model's encoder pass).
	Bidir   bool
	Softcap float64
	Act     Act
	// GLU is an OpActMul's bounds, the container's numbers (jlm.GLU): what
	// a generator composes the gated unit from. The zero value is SwiGLU.
	GLU jlm.GLU
	// A mixture's router and combine (OpRoute, OpCombine).
	Experts, TopK      int
	Router             Router
	RouteNorm          bool
	Groups, GroupsUsed int
	// Margin is OpRoute's relative band under RouterSparseMixer: an expert
	// is kept in a pick's softmax unless its gap to the pick's logit m,
	// (m - s)/max(|s|, m), exceeds twice Margin (phimoe's jitter_eps, as the
	// f32 number the file states). It is 0 on every other router.
	Margin float64
	// TempScale, TempFloor and TempOffs are OpAttend's query temperature:
	// the scores scaled by 1 + TempScale*ln(1 + floor((pos+TempOffs)/TempFloor))
	// (Llama 4's and Mistral 3's); TempScale 0 is none.
	TempScale, TempFloor, TempOffs float64
	// Mixer is OpRecurrent's kind (jlm.LayerKind's value).
	Mixer uint8
	// WeightIn is OpGather's and OpCombine's (Llama 4's): each pair's row is
	// scaled by its routed weight before its expert, and the combine sums
	// the experts' outputs unweighted.
	WeightIn bool
	// Alpha is a residual OpAdd's scale on its second input (Granite's
	// residual_scale: x + Alpha*y); 0 is one.
	Alpha float64
	// Taps is OpConv's kernel, and OpNGram's convolution's; Dil the
	// latter's spacing between taps.
	Taps, Dil int
	// Bound is OpClamp's.
	Bound float64
	// Offset and Width are OpSlice's.
	Offset, Width int
	// Block and Local are a block-selecting OpIndex's (Dense.Select);
	// Pooled is QSA's (the key the raw projection, each complete block
	// pooled).
	Block, Local int
	Pooled       bool
	// Streams is OpStreamMix's and OpStreamPlace's stream count; Sinkhorn
	// their DeepSeek V4 form (OpStreamMix's Out[1] the pre, post and
	// combination weights, Sinkhorn-normalised over Iters rounds).
	Streams  int
	Sinkhorn bool
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

// Opaque reports an op kind stated whole in this version of the language:
// its weights are bound and its effects versioned, but its inner arithmetic
// is one op, not ops of the graph's.
func Opaque(k OpKind) bool {
	switch k {
	case OpRecurrent, OpIndex, OpStreamMix, OpNGram, OpAltPredict, OpAltCorrect, OpLaurel, OpCompressedAttend,
		OpPredict, OpCrossAttend:
		return true
	}
	return false
}
