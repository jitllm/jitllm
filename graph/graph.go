// Package graph is the block graph: a versioned, typed SSA description of
// one block's values and persistent state effects, built from what the
// container states (.notes/design/composer.md, "Above the recipes"). It holds
// data and a verifier and computes nothing: a graph is lowered to the
// generated operations the engine already runs, never interpreted here.
package graph

import (
	"fmt"
	"math/bits"
	"slices"
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
	// optional Weight[1] its bias added after. With a selection In[1] (u32),
	// element j of Out[0] is W's row In[1][j] times x, plus the bias's element
	// In[1][j]: the rows it names alone, a name past W's rows read as its
	// last.
	OpMatVec
	// OpRope turns each head's first NRot elements by the position, in the
	// pair Layout, at Base: the NRot from Offset of the head where it is
	// set, and back (by the negated angle) where Inverse. With In[1] (an
	// OpRopeTable's value) it turns by that value's row rather than the
	// step's table.
	OpRope
	// OpKVAppend writes k and v (In[0], In[1]) as this position's row of the
	// block's KV: it reads state version n and writes n+1. With Span > 1 the
	// history keeps one entry each Span positions, the row closing a window
	// of them (its position+1 a multiple of Span) writing it, every other
	// row writing nothing of it.
	OpKVAppend
	// OpAttend is causal attention of q (In[0]) over the KV version it reads,
	// GQA by the head counts, scaled by Scale, over a Window (0: the whole
	// history), scores capped by Softcap (0: none); an optional Weight[0]
	// (RoleAttnSinks) is a per-head logit in the softmax's denominator.
	// Where Bidir, each row attends the span its row's descriptor names;
	// over a version the graph does not write (Builder.CrossAttend) the rows
	// are queries over keys and values other rows appended: a
	// cross-attention. With In[1], an OpSelect's Out[0], each group of its
	// query heads attends the positions the selection keeps for it alone.
	// Reading two histories, it is one softmax over both: the first's
	// positions as above, then the second's entries, one each Span positions
	// (an OpKVAppend's), the row seeing those its range's end over Span
	// closes, each of Words[0] wide; a selection masks the second alone.
	// Past the query and the selection, an Extent's input is a score bias
	// by backward distance and a RowSinks' input the sinks (AttendInputs).
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
	// either end zero. Where GridW is set the call's rows are a picture's
	// grid, GridW rows a grid row, and the taps a Taps x Taps square
	// (Weight[0] Taps*Taps rows of the width, row ky*Taps+kx), centred,
	// zero past the grid's edges: a depthwise convolution at SAME
	// padding. Where Stride is above one the output is the grid that
	// stride takes, ceil(H/Stride) x ceil(GridW/Stride) rows (its value's
	// Rows), padded as TensorFlow pads, the smaller half first -- or, where
	// Offset is set, Offset rows and columns before the grid as PyTorch's
	// padding pads (Offset Taps/2 keeps the same output grid).
	OpConv
	// OpMap is an elementwise function Fn of its operands: In[0], In[1] and
	// In[2] in order, then Weight[0] (a bound vector, a stored table of
	// several rows its elements in order), then Const, as many as
	// Fn takes. An operand narrower than Out[0] is broadcast: element i reads
	// its element i/(w/n), w the output's width and n the operand's (one
	// element a group of w/n; width 1 is a scalar). The bound vector is its
	// Width elements from Offset where Width is set, the whole vector where it
	// is 0: Width 1 reads one element of a stored table as a scalar, a
	// per-block constant a buffer holds. A primitive
	// (.notes/design/composer.md, "Decomposition").
	OpMap
	// OpReduce combines each group of Span elements of In[0] (the whole row
	// where Span is 0) by Fn (MapAdd, MapMax or MapMin) in Order: Out[0] one
	// element a group. Where HeadDim is set the row is rows of HeadDim and a
	// group is a column of Span of them: Out[0]'s element b*HeadDim+j
	// combines In[0]'s elements (b*Span+i)*HeadDim+j. A primitive.
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
	// With Span set it is a value, not the step's table: each row's at its
	// position floored to a multiple of Span (a window's first), of rotary
	// Rotary's planes.
	// Where Axes states several axes, a row has a position on each, and a
	// pair's angle is its axis's position's.
	OpRopeTable
	// OpEmbed is a step's input rows: each row's token's row of Weight[0],
	// the token a word of the step's own a row; where Positional, each
	// row's position's row (a learned position table), the position modulo
	// Span where Span is set; over an index row In[0] (u32), each of the
	// row's indices' rows side by side (an n-gram's table rows). Into an
	// index row (ElemU32), each element is an index: a token's experts.
	OpEmbed
	// OpStateConv is a causal depthwise convolution of In[0] over the
	// block's window, the last Taps-1 rows it saw: each channel's Taps taps
	// (Weight[0], channel-major, the oldest first) against the window's rows
	// and the row, Out[0] a row of the input's width; the window then drops
	// its oldest row and takes the row. Where Dil is above one the taps are
	// dilated: tap t reads the row (Taps-1-t)*Dil back, the window the last
	// (Taps-1)*Dil rows. The window is the recurrent convolution's state
	// (StateRecConv), or where the op reads StateRecState, Offset floats into
	// each of the state's slots of Span floats: a tail past a rule's state.
	// Where In[1] is given, it is each row's own taps added to the static
	// ones: per group of channels (In[0]'s width over In[1]'s in Taps), the
	// group's Taps taps side by side in the static taps' order, the oldest
	// first. A primitive with a state of its own: a recurrent mixer's
	// convolution, or a dynamic short convolution.
	OpStateConv
	// OpPool is the step's live rows of In[0] pooled into one row, Out[0]
	// it in every row: their mean, the first's or the last's (Pooling, a
	// jlm.Pooling). A primitive: an encoder's sequence into its embedding.
	OpPool
	// OpTakeRows is the rows of In[0] the step's request names, in its
	// order, Out[0]'s first rows (a decision's option markers); the rows
	// past them repeat the last. Offset, past 0, names the request's slot
	// the rows are read from (composer's request slots: after the marker
	// rows, each the call's live rows long).
	OpTakeRows
	// OpRowConv is a centred depthwise convolution of In[0] across the
	// step's live rows: each channel's Taps taps (Weight[0], channel-major)
	// against the rows Taps/2 before to Taps/2 after, zero past the live
	// rows -- a bidirectional model's short convolution (an encoder's).
	OpRowConv
	// OpPatchify is a picture's patches as rows: In[0] is the picture,
	// [y][x][channel] over the step's rows, a grid of Span x Span patches
	// whose width in patches is the request's first word and whose height is
	// the step's live rows past Offset over it; a row at position Offset+p
	// holds the patch at index p of the grid's nested order -- Width x Width
	// merge windows of patches (Width 0 or 1: none), the patches of one in
	// raster order and consecutive, in Window x Window windows of merge
	// windows (Window 0: one window, the grid), each window's merge windows
	// in raster order and consecutive, the windows in raster order and those
	// at the grid's right and lower edges cut to it -- its Span*Span*3
	// elements channel-planar (a convolution whose stride is its kernel, as a
	// matrix product), zero past them to the output's width and in the Offset
	// rows before.
	OpPatchify
	// OpRegroup is space-to-depth over the step's grid of rows (GridW its
	// width where set, the request's first word where not): row g of the
	// output is the Span x Span block of rows at g's place, each Span*Span
	// slot in raster order a run of the input's width.
	OpRegroup
	// OpFold is each Span consecutive rows of In[0] as one row.
	OpFold
	// OpSplitRows is each row of In[0] as rows of Width of its elements, in
	// order (OpFold's inverse): output row r*(w/Width)+k is elements
	// k*Width.. of input row r, w its width (a frame's features as a grid's
	// cells).
	OpSplitRows
	// OpSplice is In[0] with a picture's rows in place: row r takes the row
	// of In[1] (the vision segment's projector, a picture's step's rows)
	// that the step's request names for it, where it names one.
	OpSplice
	// OpPosResample is each row's position-table row resampled to the
	// step's grid: Weight[0] a Span x Span table, the row's cell (its
	// position less Offset, a grid the request's first word wide and the
	// live rows past Offset tall, in OpPatchify's order where Width or
	// Window is set) mapped to a source coordinate on each axis by Chain and
	// read through Filter's taps.
	OpPosResample
	// OpStateScan is a recurrence over the block's recurrent state, each row
	// in order: for each head h of NHead, a matrix S of Dv rows (Out[0]'s
	// width over NHead) by HeadDim columns. Its terms (Attr.Expr) are
	// evaluated at each element (h, j, i) of S as OpExpr's are, an operand
	// In[k] read at the axes Words[k] states (ScanHead, ScanGroup, ScanCycle, ScanRow,
	// ScanCol), ArgState the element of S before the row and ArgSum term
	// Index summed over S's row (every column i of (h, j)). Term Update is
	// S after the row, and Out[0] at (h, j) is the last term summed over
	// the row. It reads state version n and writes n+1. Where Span is set it
	// is the state's floats per sequence, the matrices' and a tail past them
	// the scan neither reads nor writes. A primitive with a state of its
	// own: a linear-attention or state-space mixer's recurrence.
	OpStateScan
	// OpPosition is each row's position in the block's history, the one its
	// key is written at, plus Offset and in whole Spans where Span > 1 (an
	// integer division), as a float: Out[0] one element a row. A primitive.
	OpPosition
	// OpSelect is a selection of the history's positions, which an OpAttend
	// reading Out[0] as its In[1] attends alone: each of NHead query heads of
	// In[0] (HeadDim wide) dots every position's key the row attends, the
	// HeadDim elements at Offset of a history key row Width wide; a head's
	// score passes Fn against Const where Fn is set (MapMax against 0 a
	// ReLU), is scaled by In[1]'s element for the head where given, and
	// each of NKVHead groups of NHead/NKVHead heads sums its heads' scores.
	// A group ranks the blocks of Span positions from the history's first,
	// a block by its best position, equal scores to the lower block, and
	// keeps TopK of them: the Local blocks ending at the row's own, then the
	// best of the rest -- every block where there are at most TopK. Pooled
	// (Attrs.Pooled) scores a block by one key pooled from its positions'.
	// Out[0] (one a group) is the positions each group keeps. A primitive
	// over the history, as OpAttend is.
	OpSelect
	// OpUnfold is each output row of In[0]'s grid (GridW rows a grid row)
	// as the Taps x Taps window of rows around it at SAME padding -- or, where
	// Offset is set, Offset rows and columns before the grid (PyTorch's) --
	// at Stride as OpConv takes it (Out[0] that grid's rows): tap ky*Taps+kx's
	// row of In[0]'s width at element (ky*Taps+kx)*width, zero past the
	// grid's edges and past Taps*Taps rows to Out[0]'s width. A full
	// convolution is OpMatVec over it.
	OpUnfold
	// OpUniform is a uniform in [0, 1) for each key of In[0] (u32), counter
	// based: Philox2x32-10 over the counter (the key, In[1]'s second word)
	// with In[1]'s first word as its key (In[1] u32, two words: the seed,
	// then the position), the first output word's top 24 bits over 2^24.
	// Out[0] one float a key. A primitive: a sampler's draw, the same on
	// every device.
	OpUniform
	// OpPut is In[0] with In[2]'s elements in place: element j of In[2]
	// takes the place of In[0]'s element at index j of In[1] (u32), an index
	// at or past In[0]'s width naming none. Two indices naming one element
	// carry equal values. Out[0] In[0]'s width. A primitive.
	OpPut
	// OpRopePos is each row's position on each of four rotary axes, from
	// its cache position and its sequence's rotary marks (the step's rows'
	// positions and the sequences' marks, as the engine keeps them): the
	// last mark at or before the position says how the row turns, each
	// axis an affine function of the position and of the row's index in the
	// mark, divided and wrapped (MarkPosition) -- and a row before every
	// mark turns at its position. Out[0] is four indices a row. A
	// primitive.
	OpRopePos
	// OpToken is each row's token, the step's own: Out[0] one index (u32) a
	// row. A primitive.
	OpToken
	// OpTokenWindow is each row's token window (Out[0], u32, Taps
	// words): the row's token In[0] (u32, one word; a row naming none, all
	// ones, reads as Words[0], a stated fill), then its Taps-1
	// predecessors nearest first, each its id plus one -- the earlier rows
	// of its run, then those the run's slot holds at Offset of the
	// recurrent state (Span floats a sequence, an id's word as a float's
	// bits). The run's predecessors after its last row are written to the
	// slot: its rows' ids plus one newest first, then the slot's. It reads
	// state version n and writes n+1. A primitive with a state of its own.
	OpTokenWindow
	// OpHalves is In[0]'s floats as binary16, two a word (Out[0], u32, half
	// In[0]'s width), the lower element in the low half, rounded to nearest
	// even as a binary16 history's pages store them; as bfloat16 where
	// Attr.BF16 says, a NaN quieted. A
	// primitive.
	OpHalves
	// OpWidenHalves is In[0]'s binary16 halves, two a word (u32), as floats
	// (Out[0], twice In[0]'s width), the low half first; bfloat16 halves
	// where Attr.BF16 says. A primitive.
	OpWidenHalves
	// OpQ8Rows is each row of In[0]'s floats as a q8 history's row holds it
	// (Out[0], u32): the row's nb = ceil(W/32) blocks' codes, four to a word
	// with the lower element low and the last block zero past W, then a pair
	// of floats a block, its scale amax/127 and the sum of its codes times
	// -16. A code is the element times the scale's reciprocal (zero where the
	// scale is), rounded half away from zero. A primitive.
	OpQ8Rows
	// OpWidenQ8Rows is OpQ8Rows' rows (In[0], u32) as floats (Out[0], the
	// rows' W elements): each element its code times its block's scale. A
	// primitive.
	OpWidenQ8Rows
	// OpPicture is the step's picture as a tower reads its pixels: the
	// decoded samples (the step's picture table, in Pixels' format, rows of
	// the request's stride in bytes, its width and height in pixels the
	// request's words) resampled to PicW x PicH by Chain and Filter's taps on
	// each axis -- where PicLive, to the request's output width and height,
	// at most PicW x PicH -- each sample read as [0, 1] and channel c
	// normalised as (v - Mean[c]) / Std[c]. Out[0] is the pixels
	// [y][x][channel] over the step's rows, as OpPatchify reads them, zero
	// past the output's pixels -- where PicCanvas, past its canvas, the
	// canvas's cells outside the picture its fill normalised. A primitive.
	OpPicture
	// OpHistoryRows is, for each row, the Span positions of the window of
	// Span holding its position, or of the window Local windows before it,
	// read from the history it reads (rows of Width): Out[0] position by
	// position, each its row's elements from Offset, Out[0]'s width over
	// Span of them. A position before the first, or past the row's own,
	// reads Const. A primitive over the history: its rows as a value.
	OpHistoryRows
	// OpRowsAttend is each row of In[0] (the queries) attending over the
	// step's live rows of In[1] (the keys) and In[2] (the values), every row
	// and both ways: NHead heads of HeadDim, the scores scaled by Scale, a
	// softmax over the live rows. The keys and values are NKVHead heads (0:
	// NHead), each read by NHead/NKVHead consecutive query heads. Out[0] has
	// In[0]'s rows. Offset, past 0, holds query row q to the keys from the
	// request's slot Offset to its slot Offset+1 (their row q). A fourth
	// operand, a row of NHead runs of the keys' rows a query row, is added to
	// the scores before they are scaled: element h*keys+j to head h's score
	// over key j (a score bias).
	OpRowsAttend
	// OpSpanPool is each row r of Out[0] the mean of In[0]'s rows from the
	// request's slot Offset to its slot Offset+1 (their row r), zero for an
	// empty span: a question's or an option's span of a prompt.
	OpSpanPool
	// OpSound is the step's clip as mono samples at a rate (Out[0], one row
	// of its Width samples, zero past the clip's outputs): the step's
	// picture table holds the clip's header, its PCM in Sound's sample
	// format and its resample's plan (composer.SoundTable), and each output
	// is the sum, over its window's taps and each tap's channels, of a
	// sample times its channel's weight in the mono mix times its phase's
	// filter at that tap -- blended toward the next phase's sum by the
	// output's fraction where the ratio interpolates -- libswresample's
	// rematrix and resample (composer.SWR, composer.MonoMix), at most Taps
	// taps times channels. A primitive.
	OpSound
	// OpBlockRows is each row of In[0]'s floats in GGUF blocks of
	// Attr.Block's format (Out[0], u32, BlockRowWords): the row's
	// ceil(W/32) blocks, the last zero past W, each quantized as the GGUF
	// type's reference quantizer does (internal/oracle), their fields in BlockRegions' order. A
	// primitive.
	OpBlockRows
	// OpWidenBlockRows is OpBlockRows' rows (In[0], u32) as floats (Out[0],
	// the rows' W elements), each element as the GGUF type decodes it.
	// A primitive.
	OpWidenBlockRows
	// OpGatherRows is each row of Out[0] a row of its sources -- In[0]'s rows
	// where it has one, then each of Weight's rows, end to end -- the one an
	// integer expression of the row names (Expr, its last term; Index builds
	// one, over the operands GatherRow to GatherTrailN), zero for one past
	// them all. Where Span is set, term Span-1 is the last row written: a row
	// past it is that row, recomputed and written again, and no row after
	// it is touched. A primitive.
	OpGatherRows
	// OpGatherElems is each element of Out[0] an element of In[0], its rows
	// read end to end, the one an integer expression of the element names
	// (Expr, its last term, over the operands GatherRow to GatherCol), zero
	// for one past them. A primitive.
	OpGatherElems
	// OpMeanRows is Out[0], one row, each element the sum of its column over
	// In[0]'s first n rows, over max(n - Const, 1): n an integer expression
	// (Expr, its last term, over GatherLive to GatherSamples). A primitive.
	OpMeanRows
	// OpHeadDots is each row of Out[0] In[0]'s row dotted, head by head, with
	// rows of Weight[0]: element h*Rows+t the dot of head h of the row (NHead
	// heads of HeadDim) with head h of the table's row t; where it states an
	// index (Expr, over the row and GatherCol), Span dots a head, element
	// h*Span+j the dot with the table's row the index names at column j, zero
	// where it names none. A primitive.
	OpHeadDots
	// OpPad is In[0] widened with zeros: Out[0]'s first elements In[0]'s,
	// zero past them -- a row read by a matrix stored wider than it, its k
	// padded to whole blocks. A primitive.
	OpPad
	// OpIterate is Taps values for each of In[0]'s elements carried through
	// Span trips in order: Expr's first Taps terms are the values before the
	// first trip, read at the element as OpExpr's are; the terms after them
	// are a trip's, ArgState term Index the value it carries into the trip,
	// and the values it carries out the terms Words names, value j term
	// Words[j], or the last Taps terms where Words is empty. Value j of
	// element i after the last trip is Out[j]'s element i where the op has
	// an output per value, each of its own type, else Out[0]'s element
	// i*Taps+j, every value of Out[0]'s type. A term's type follows from
	// what it reads (IterateElems). A trip follows the one before it: no
	// trips are taken apart or regrouped. Where Width is set, a trip's term
	// may read ArgSum term Index, a float, summed over the run of Width
	// elements of In[0]'s row the element is in (elements i-i%Width
	// onward): the run's elements take each trip together. Width is a power
	// of two dividing In[0]'s width. Its operands past In[0] are tables, read
	// whole and only by a MapTake term's first operand. A primitive.
	OpIterate
	numOps
)

func (k OpKind) String() string {
	switch k {
	case OpInvalid:
		return "invalid"
	case OpNorm:
		return "norm"
	case OpHeadNorm:
		return "head_norm"
	case OpMatVec:
		return "matvec"
	case OpRope:
		return "rope"
	case OpKVAppend:
		return "kv_append"
	case OpAttend:
		return "attend"
	case OpAdd:
		return "add"
	case OpExpr:
		return "expr"
	case OpExpertMatVec:
		return "expert_matvec"
	case OpLayerNorm:
		return "layer_norm"
	case OpClamp:
		return "clamp"
	case OpSlice:
		return "slice"
	case OpScale:
		return "scale"
	case OpConv:
		return "conv"
	case OpMap:
		return "map"
	case OpReduce:
		return "reduce"
	case OpTopK:
		return "top_k"
	case OpTake:
		return "take"
	case OpRank:
		return "rank"
	case OpTile:
		return "tile"
	case OpConcat:
		return "concat"
	case OpRopeTable:
		return "rope_table"
	case OpEmbed:
		return "embed"
	case OpStateConv:
		return "state_conv"
	case OpPool:
		return "pool"
	case OpSound:
		return "sound"
	case OpBlockRows:
		return "block_rows"
	case OpWidenBlockRows:
		return "widen_block_rows"
	case OpSpanPool:
		return "span_pool"
	case OpTakeRows:
		return "take_rows"
	case OpRowConv:
		return "row_conv"
	case OpPicture:
		return "picture"
	case OpHistoryRows:
		return "history_rows"
	case OpPatchify:
		return "patchify"
	case OpRegroup:
		return "regroup"
	case OpFold:
		return "fold"
	case OpSplitRows:
		return "split_rows"
	case OpSplice:
		return "splice"
	case OpPosResample:
		return "pos_resample"
	case OpStateScan:
		return "state_scan"
	case OpPosition:
		return "position"
	case OpSelect:
		return "select"
	case OpRowsAttend:
		return "rows_attend"
	case OpUnfold:
		return "unfold"
	case OpGatherRows:
		return "gather_rows"
	case OpGatherElems:
		return "gather_elems"
	case OpMeanRows:
		return "mean_rows"
	case OpHeadDots:
		return "head_dots"
	case OpPad:
		return "pad"
	case OpIterate:
		return "iterate"
	case OpUniform:
		return "uniform"
	case OpPut:
		return "put"
	case OpRopePos:
		return "rope_pos"
	case OpToken:
		return "token"
	case OpTokenWindow:
		return "token_window"
	case OpHalves:
		return "halves"
	case OpWidenHalves:
		return "widen_halves"
	case OpQ8Rows:
		return "q8_rows"
	case OpWidenQ8Rows:
		return "widen_q8_rows"
	}
	return fmt.Sprintf("op%d", k)
}

// Q8RowWords is the words OpQ8Rows writes for a row of w floats: the codes
// of ceil(w/32) blocks four to a word, then a pair a block.
func Q8RowWords(w int) int { return (w + 31) / 32 * 10 }

// BlockFmt is a GGUF block format a row of floats is held in
// (OpBlockRows): 32 elements a block, each block's fields the GGUF type's.
//
//	BlockQ4_0  binary16 d, 16 bytes of nibbles: (q-8)*d
//	BlockQ4_1  binary16 d and m, 16 bytes of nibbles: q*d+m
//	BlockQ5_0  binary16 d, 4 bytes of fifth bits, 16 of nibbles: (q-16)*d
//	BlockQ5_1  binary16 d and m, 4 bytes of fifth bits, 16 of nibbles: q*d+m
//	BlockIQ4NL binary16 d, 16 bytes of nibbles indexing IQ4_NL's
//	           code table: kv[q]*d
//
// Element j < 16 of a block is nibble byte j's low half and element j+16
// its high half; element j's fifth bit is bit j of the block's word. A row
// holds its blocks' fields by kind rather than block by block, so that no
// field crosses a word (BlockRegions): the same bytes as a GGUF row of
// blocks rounded up to a word.
type BlockFmt uint8

const (
	BlockQ4_0 BlockFmt = iota + 1
	BlockQ4_1
	BlockQ5_0
	BlockQ5_1
	BlockIQ4NL
)

// IQ4NLValues is IQ4_NL's code table, the values an iq4_nl code indexes.
func IQ4NLValues() [16]int8 {
	return [16]int8{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}
}

func (f BlockFmt) String() string {
	switch f {
	case BlockQ4_0:
		return "q4_0"
	case BlockQ4_1:
		return "q4_1"
	case BlockQ5_0:
		return "q5_0"
	case BlockQ5_1:
		return "q5_1"
	case BlockIQ4NL:
		return "iq4_nl"
	}
	return fmt.Sprintf("BlockFmt(%d)", uint8(f))
}

// Bytes is one block's bytes in format f, 0 for no format.
func (f BlockFmt) Bytes() int {
	switch f {
	case BlockQ4_0, BlockIQ4NL:
		return 18
	case BlockQ4_1:
		return 20
	case BlockQ5_0:
		return 22
	case BlockQ5_1:
		return 24
	}
	return 0
}

// Mins is whether f's block holds a minimum after its scale (q*d+m).
func (f BlockFmt) Mins() bool { return f == BlockQ4_1 || f == BlockQ5_1 }

// Fifths is whether f's block holds a fifth bit an element.
func (f BlockFmt) Fifths() bool { return f == BlockQ5_0 || f == BlockQ5_1 }

// BlockRegions is where the regions of a row of w floats in format f start,
// in words: the scales (a word a block, d low and m high, where the format
// has a minimum, else a word two blocks, the even block's d low), the
// fifth bits (a word a block) and the nibbles (four words a block, its 16
// bytes in order, the lower byte low).
func BlockRegions(f BlockFmt, w int) (scales, fifths, nibbles int) {
	nb := (w + 31) / 32
	fifths = (nb + 1) / 2
	if f.Mins() {
		fifths = nb
	}
	nibbles = fifths
	if f.Fifths() {
		nibbles += nb
	}
	return 0, fifths, nibbles
}

// BlockRowWords is the words OpBlockRows writes for a row of w floats in
// format f: ceil(w/32) blocks' bytes, rounded up to a word.
func BlockRowWords(f BlockFmt, w int) int {
	_, _, nibbles := BlockRegions(f, w)
	return nibbles + (w+31)/32*4
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
	// ArgState is an OpStateScan's state element before the row.
	ArgState
	// ArgSum is an OpStateScan's term Index summed over the state's row,
	// an OpIterate's over a run of its row (Attrs.Width).
	ArgSum
)

// ScanAxis is an axis an OpStateScan operand varies along (Attrs.Words, a
// mask an operand), its elements in the order the bits are listed.
type ScanAxis uint64

const (
	// ScanHead is the state's head, one of NHead.
	ScanHead ScanAxis = 1 << iota
	// ScanGroup is the head's group: head h reads h/(NHead/NKVHead) of
	// NKVHead.
	ScanGroup
	// ScanRow is the state's row, one of Out[0]'s width over NHead.
	ScanRow
	// ScanCol is the state's column, one of HeadDim.
	ScanCol
	// ScanCycle is the head's place in a cycle of NKVHead: head h reads
	// h mod NKVHead, an operand repeated whole across the heads where
	// ScanGroup repeats each of its heads in turn. It sits where ScanGroup
	// would in the operand's order.
	ScanCycle
)

// ScanWidth is the elements an operand along axes m holds, of heads heads
// in groups groups, rows rows and cols columns; false where m is not a set
// of axes (more than one of a head, its group and its cycle, or a bit past
// them).
func ScanWidth(m ScanAxis, heads, groups, rows, cols int) (int, bool) {
	if m&^(ScanHead|ScanGroup|ScanRow|ScanCol|ScanCycle) != 0 || bits.OnesCount64(uint64(m&(ScanHead|ScanGroup|ScanCycle))) > 1 {
		return 0, false
	}
	n := 1
	for _, d := range []struct {
		ax ScanAxis
		n  int
	}{{ScanHead, heads}, {ScanGroup, groups}, {ScanCycle, groups}, {ScanRow, rows}, {ScanCol, cols}} {
		if m&d.ax != 0 {
			n *= d.n
		}
	}
	return n, true
}

// ValueID names a value in a Graph's Values.
type ValueID int32

// Value is one SSA value: a row of Width elements per token, f32 unless
// Elem says otherwise.
type Value struct {
	Name  string
	Width int
	Elem  Elem
	// Rows, where set, is the value's own count of rows, a grid's that a
	// strided op made from the step's; 0 is the step's rows.
	Rows int
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
	MapAdd       = jlm.FnAdd       // a + b
	MapSub       = jlm.FnSub       // a - b
	MapMul       = jlm.FnMul       // a * b
	MapDiv       = jlm.FnDiv       // a / b
	MapMax       = jlm.FnMax       // max(a, b)
	MapMin       = jlm.FnMin       // min(a, b)
	MapExp       = jlm.FnExp       // exp(a)
	MapSigmoid   = jlm.FnSigmoid   // 1 / (1 + exp(-a))
	MapSoftplus  = jlm.FnSoftplus  // log(1 + exp(a))
	MapSqrt      = jlm.FnSqrt      // sqrt(a)
	MapWhere     = jlm.FnWhere     // b where a > 0, else c
	MapLess      = jlm.FnLess      // 1 where a < b, else 0
	MapFma       = jlm.FnFma       // a*b + c, the product not rounded before the sum
	MapExpm1     = jlm.FnExpm1     // exp(a) - 1, exact near zero where the difference cancels
	MapAbs       = jlm.FnAbs       // |a|
	MapRound     = jlm.FnRound     // a to the nearest integer, halves away from zero
	MapRecip     = jlm.FnRecip     // 1 / a
	MapTanh      = jlm.FnTanh      // tanh(a)
	MapErf       = jlm.FnErf       // erf(a)
	MapEqual     = jlm.FnEqual     // 1 where a == b, else 0; over two index rows, their integers
	MapLog       = jlm.FnLog       // the natural logarithm of a
	MapFloor     = jlm.FnFloor     // the greatest integer not above a
	MapCeil      = jlm.FnCeil      // the least integer not below a
	MapTrunc     = jlm.FnTrunc     // a's integer part, toward zero
	MapRoundEven = jlm.FnRoundEven // a to the nearest integer, halves to the even one
	MapF32       = jlm.FnF32       // a rounded to binary32
	MapRem       = jlm.FnRem       // a modulo b, over words
	MapAnd       = jlm.FnAnd       // a & b, over words
	MapOr        = jlm.FnOr        // a | b, over words
	MapXor       = jlm.FnXor       // a ^ b, over words
	MapShl       = jlm.FnShl       // a << b, over words
	MapShr       = jlm.FnShr       // a >> b, over words
	MapTake      = jlm.FnTake      // operand a's element at the word index b: a table read whole (OpIterate)
)

// Order is an OpReduce's: the order its additions run in is part of its
// arithmetic, so it is stated.
type Order uint8

const (
	// OrderSeq combines a group's elements one at a time from its first.
	OrderSeq Order = iota
	// OrderTree combines them pairwise, the first half with the second, down
	// to one. A group of a width not a power of two starts at the next power
	// of two: element i is combined with element i+P/2 where that is in the
	// group, and passed on alone where it is not.
	OrderTree
)

// Binding is a weight an op reads: its container role in the graph's block,
// with the shape the op requires (K the input width, Rows the output; Rows 0
// for a vector of K).
type Binding struct {
	Role    jlm.Role
	K, Rows int
	Sheets  int // a bank's experts, each a K x Rows sheet; 0 for one matrix
	// From, where set, is the first stored row the binding reads: an
	// embedding of rows From to From+Rows of its table (a row table's head).
	From int
}

// State is a block's persistent state an op reads or writes.
type State uint8

const (
	// StateKV is the block's paged attention history.
	StateKV State = iota
	// StateRecConv is a recurrent block's short convolution's tail.
	StateRecConv
	// StateRecState is a recurrent block's scan's state (OpStateScan).
	StateRecState
)

// StateRef is a version of the block's persistent state: State, and where
// From is set, block From-1's (a block reading another's history). Hist
// names which of the block's histories a StateKV is: 0 its history of rows,
// others histories of its own beside it (entries pooled from its rows).
type StateRef struct {
	State   State
	From    int
	Version int
	Hist    uint8
}

// String is the state's label, its version apart.
func (r StateRef) String() string {
	var s string
	switch r.State {
	case StateKV:
		s = "kv"
		if r.Hist > 0 {
			s = fmt.Sprintf("kv%d", r.Hist)
		}
	case StateRecConv:
		s = "rec.conv"
	case StateRecState:
		s = "rec.state"
	default:
		s = fmt.Sprintf("state%d", r.State)
	}
	if r.From > 0 {
		s += fmt.Sprintf(" of block %d", r.From-1)
	}
	return s
}

// HistoryRead is block Block's history Hist, as a block other than Block
// reads it.
type HistoryRead struct{ Block, Hist int }

// ReadsHistoryOf is the block whose histories g reads in place of its own (a
// StateKV naming another block), any of them by index: the first it names
// where it appends to no history of its own, -1 where it does or reads no
// other block's. A binding holds that block's histories at their indices
// (ReadsMore holds the rest).
func (g *Graph) ReadsHistoryOf() int {
	if g == nil {
		return -1
	}
	n := -1
	for _, o := range g.Ops {
		for _, w := range o.Writes {
			if o.Kind == OpKVAppend && w.State == StateKV && w.From == 0 {
				return -1
			}
		}
		for _, r := range o.Reads {
			if r.State == StateKV && r.From != 0 && n < 0 {
				n = int(r.From) - 1
			}
		}
	}
	return n
}

// ReadsMore is every history g reads of a block other than the one whose
// histories it binds at their indices (its own, or ReadsHistoryOf's), each
// once, in the order its ops first name them: a binding holds read j after
// that block's histories, at the table instance 1+kept+j, kept the
// histories past its rows' that block keeps.
func (g *Graph) ReadsMore() []HistoryRead {
	if g == nil {
		return nil
	}
	src := g.ReadsHistoryOf()
	var out []HistoryRead
	for _, o := range g.Ops {
		for _, r := range o.Reads {
			if r.State != StateKV || r.From == 0 || int(r.From)-1 == src {
				continue
			}
			if h := (HistoryRead{Block: int(r.From) - 1, Hist: int(r.Hist)}); !slices.Contains(out, h) {
				out = append(out, h)
			}
		}
	}
	return out
}

// MoreIndex is the index in ReadsMore of the history r names, -1 where r
// is one of the histories g binds at their indices.
func (g *Graph) MoreIndex(r StateRef) int {
	if r.State != StateKV || r.From == 0 || int(r.From)-1 == g.ReadsHistoryOf() {
		return -1
	}
	return slices.Index(g.ReadsMore(), HistoryRead{Block: int(r.From) - 1, Hist: int(r.Hist)})
}

// KeptHistory is a history a block keeps past its rows' (history 0), as
// its graph's append to it states it: an entry each Period positions,
// Width floats wide, its values its keys where the append's are.
type KeptHistory struct {
	Hist, Period, Width int
	ValuesInKeys        bool
}

// KeptHistories is every history past its rows' that g appends to, in the
// order of their indices; an index g appends to twice is listed once.
func (g *Graph) KeptHistories() []KeptHistory {
	if g == nil {
		return nil
	}
	var out []KeptHistory
	for _, o := range g.Ops {
		if o.Kind != OpKVAppend || len(o.Writes) != 1 || o.Writes[0].Hist == 0 || o.Writes[0].From != 0 {
			continue
		}
		k := KeptHistory{Hist: int(o.Writes[0].Hist), Period: max(1, o.Attr.Span), Width: g.Values[o.In[0]].Width, ValuesInKeys: o.In[0] == o.In[1]}
		i := 0
		for i < len(out) && out[i].Hist < k.Hist {
			i++
		}
		if i < len(out) && out[i].Hist == k.Hist {
			continue
		}
		out = append(out[:i], append([]KeptHistory{k}, out[i:]...)...)
	}
	return out
}

// key is the state r names, whatever its version.
func (r StateRef) key() StateRef { return StateRef{State: r.State, From: r.From, Hist: r.Hist} }

// Attrs are an op's parameters. Only those its kind defines are read; the
// verifier refuses one set on a kind that does not define it.
type Attrs struct {
	Eps     float64
	HeadDim int
	NRot    int
	Base    float64
	Layout  RopeLayout
	Freqs   bool // the rotary divides each pair's theta by stored factors (the container's rope_freqs)
	// Axes is OpRopeTable's pair map: pair p turns by the row's position on
	// axis Axes[p], at the frequency the rotary's planes fold for it; empty
	// is one axis. OpRope is the same rotation over a table built per row.
	Axes []uint8
	// Sections, Interleaved, Cycle and CycleAxes are read by nothing: a
	// graph stored before the table carried its pair map (Axes) stated its
	// axes here, and the table states them now. They stay because a stored
	// graph reads a field by its name (TestStoredFieldsOnlyGrow).
	Sections    [4]int
	Interleaved bool
	Cycle       int
	CycleAxes   [4]uint8
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
	Bidir bool
	// NoRuns keeps a causal attention causal inside the bidirectional key
	// runs a call hands its rows (a picture's rows that see each other,
	// nn.KeyRunDevice): each row counts keys to its own position.
	NoRuns  bool
	Softcap float64
	// Expr is OpExpr's terms, and OpGatherRows', OpGatherElems' and
	// OpMeanRows' integer expression.
	Expr []Term
	// Experts and TopK are a bank's experts and how many pairs a selection
	// holds (OpExpertMatVec, OpTopK).
	Experts, TopK int
	// Alpha is OpAdd's scale on its second input: x + Alpha*y; 0 is one.
	Alpha float64
	Taps  int
	// Dil is OpStateConv's dilation; 0 and 1 alike.
	Dil int
	// GridW is OpConv's grid width: the rows a grid row holds; Stride its
	// step over the grid, 0 and 1 alike.
	GridW, Stride int
	// Positional makes OpEmbed index its table by the row's position;
	// Request by the step request's question type, every row the same row
	// (a marker head's question type).
	Positional, Request bool
	// Pooling is OpPool's: a jlm.Pooling.
	Pooling uint8
	// Block is OpBlockRows' and OpWidenBlockRows' format.
	Block BlockFmt
	// BF16 makes OpHalves' and OpWidenHalves' halves bfloat16.
	BF16 bool
	// Rotary is OpRopeTable's: which of the model's rotaries the table is
	// (RopeTable.Index), the step's table it writes.
	Rotary int
	// Grid is OpRopeTable's (RopeTable.Grid).
	Grid bool
	// Chain and Filter are OpPosResample's: the coordinate's convention
	// and the taps that read it.
	Chain, Filter uint8
	// Extent, past 0, adds a row's score bias to OpAttend's scores: an
	// input of NHead*Extent a row, the key at backward distance d (the
	// row's position less the key's) gaining head h's element h*Extent+d
	// after the scale, before the softcap, where 0 <= d < Extent, nothing
	// elsewhere. Over two histories it biases the first's positions.
	Extent int
	// RowSinks takes OpAttend's sinks from an input of NHead a row, one
	// logit a head in the denominator alone, in place of Weight[0].
	RowSinks bool
	// RowWindows makes OpAttend read each row's attended range from the
	// step's window table, not its history descriptor: a vision tower's
	// window attention, a row's keys its window's rows.
	RowWindows bool
	// Bound is OpClamp's.
	Bound float64
	// Offset and Width are OpSlice's; Offset is the leading rows of
	// OpPatchify too, and both are the window of
	// OpMap's bound vector. Width is an OpAttend's history row where it is
	// wider than the heads it reads (its first), and OpSelect's, Offset
	// where OpSelect's key starts in it, and OpRope's where each head's turned
	// elements start.
	Offset, Width int
	// Origin is OpRopeTable's position of a grid's first cell on each axis
	// (graph.Rotary.Origin).
	Origin int
	// Local is OpSelect's blocks kept ending at the row's own, and the
	// windows before the row's own OpHistoryRows reads.
	Local int
	// Pooled scores OpSelect's blocks by their pooled keys: a complete
	// block's keys averaged, RMS-normed (Eps, Weight[0]) and turned at the
	// block's first position by the rotary its NRot, Base, Layout, Freqs,
	// Sections and Mscale name (none where NRot is 0). Only the complete
	// blocks rank, TopK of them kept, and the row's incomplete block is kept
	// beside them; Local is 0.
	Pooled bool
	// Firsts scores OpSelect's blocks by their first attended position's
	// score in place of their best; not with Pooled.
	Firsts bool
	// Inverse turns OpRope's elements back: by the negated angle.
	Inverse bool
	// Keeps makes OpStateScan, OpStateConv and OpTokenWindow store the
	// state after each of a step's rows besides each run's after its last:
	// the row at entry e of the run order to slot e of the state's keep
	// table, so a sequence can be returned to any row of a call with no row
	// run again (a speculation's rollback).
	Keeps bool
	// Fn, Order, Span and Const are the primitives' (OpMap, OpReduce,
	// OpTopK, OpRank, OpTile): the function, the order a reduction runs in, the
	// group a reduction, top-k or rank runs over (0: the row), a constant
	// operand. Span is OpGatherRows' bound too, Const the count OpMeanRows'
	// divisor is short of its rows.
	Fn    MapFn
	Order Order
	Span  int
	Const float64
	// Words are an op's constant words; OpStateScan's operands' axes
	// (ScanAxis), one an operand.
	Words []uint64
	// Reset is read by no op: the n-gram hash that read it is expressions
	// into words now, and a stored graph naming that op is refused by name.
	// It stays so a stored graph's fields only grow.
	Reset bool
	// Update is OpStateScan's term that is the state after the row.
	Update int
	// Exact asks a MapDiv, an OpMap's or an OpExpr term's, for the IEEE
	// quotient where the device's own division is not correctly rounded;
	// without it a division takes the device's, within the gates' band like
	// every other kernel form.
	Exact bool
	// Sound is OpSound's sample format (SoundU8 to SoundF64).
	Sound uint8
	// Pixels, PicW, PicH, Mean and Std are OpPicture's: the picture's sample
	// format, the size it is resampled to, and each channel's mean and std;
	// PicSide is the most a side of a picture resampled by Pillow's filters
	// (ChainPIL) may be, whose reduction Taps reaches. PicLive says the size
	// resampled to is the request's, PicW x PicH the most it is; PicCanvas
	// lays a live picture on a canvas the request states, its size and the
	// picture's corner on it (composer.RequestPicCW to RequestPicCY), at
	// most PicW x PicH, every cell outside the picture PicFill's sample of
	// its channel normalised (a pad or a letterbox, made before
	// normalising). PicArea, where set, is the most pixels a live picture
	// holds (Area), its sides any up to PicW and PicH.
	Pixels     uint8
	PicW, PicH int
	PicArea    int
	PicSide    int
	PicLive    bool
	PicCanvas  bool
	PicFill    [3]uint8
	// PicTorch says the taps are torchvision's precision, not Pillow's
	// (composer.PILTapsAt): the picture op runs the same passes on them.
	PicTorch bool
	// PicBytes writes the resampled picture's bytes rather than its
	// normalised samples (ChainPIL): Out[0] a word a pixel, its three
	// channels' bytes from the lowest and 255 above them, the live picture
	// dense over Out[0]'s rows of any width -- a picture the next resize
	// reads as the step's (PixRGBA8), as a processor resizing more than once
	// reads its own uint8 output.
	PicBytes bool
	// PicFrame reads the step's second frame, a video's, from the byte the
	// request's frame word names (composer.RequestPicFrame): its planes' and
	// palette's offsets the frame's own, its size and format the first's.
	PicFrame bool
	// PicPass is which of a Pillow resize's passes the op runs (ChainPIL):
	// PicWhole the picture's every row at once, PicBand one band of PicSide
	// source rows -- its horizontal sums, and each output's vertical taps
	// that fall in it added into the accumulator -- and PicFinish the
	// accumulator made the picture's bytes. A picture taller than PicSide is
	// its bands in order, then the finish. PicCols is one band of source
	// columns' horizontal sums over a band of source rows, added into the
	// horizontal accumulator (the first column band replacing what it held):
	// a picture whose windows outrun the entries' taps runs each row band's
	// column bands, then that row band's pass over their sums (PicColumns).
	PicPass uint8
	// PicColumns says a band pass reads its horizontal sums from the
	// horizontal accumulator its column bands (PicCols) added into, rather
	// than making them.
	PicColumns bool
	// PicBF16 says the normalisation is in bfloat16: the sample times
	// 1/255, less the mean, over the std, each result and the mean and std
	// rounded to bfloat16 (composer's normalized).
	PicBF16   bool
	Mean, Std [3]float64
}

// Area is the most pixels a picture op's output holds: PicArea where set,
// PicW x PicH where not.
func (a Attrs) Area() int {
	if a.PicArea > 0 {
		return a.PicArea
	}
	return a.PicW * a.PicH
}

// Holds says a picture op holds an output of w x h.
func (a Attrs) Holds(w, h int) bool { return w <= a.PicW && h <= a.PicH && w*h <= a.Area() }

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
	// Routed, where set, is the ops the front end states as the block's
	// routed experts: its router through their weighted sum, a piece a
	// placement may run on another device than the rest (Cut).
	Routed Region
}

// Region is the ops [Lo, Hi) of a graph; the zero Region is none.
type Region struct{ Lo, Hi int }

func (g *Graph) value(name string, width int) ValueID {
	g.Values = append(g.Values, Value{Name: name, Width: width})
	return ValueID(len(g.Values) - 1)
}

// StateVersion is the version of the block's state s its ops have written
// so far: the one the next op reading it reads.
func (g *Graph) StateVersion(s State) int {
	v := 0
	for _, o := range g.Ops {
		for _, w := range o.Writes {
			if w.State == s && w.From == 0 {
				v = max(v, w.Version)
			}
		}
	}
	return v
}

// AttendInputs is OpAttend o's inputs past the query, in their order: its
// selection, its score bias (Attrs.Extent) and its sinks (Attrs.RowSinks),
// each -1 where o has none; false where its inputs are not that list.
func AttendInputs(o Op) (sel, bias, sinks ValueID, ok bool) {
	sel, bias, sinks = -1, -1, -1
	at := len(o.In)
	take := func(on bool) ValueID {
		if !on || at < 2 {
			return -1
		}
		at--
		return o.In[at]
	}
	if sinks = take(o.Attr.RowSinks); o.Attr.RowSinks && sinks < 0 {
		return -1, -1, -1, false
	}
	if bias = take(o.Attr.Extent > 0); o.Attr.Extent > 0 && bias < 0 {
		return -1, -1, -1, false
	}
	if at == 2 {
		sel = o.In[1]
	}
	return sel, bias, sinks, at == 1 || at == 2
}

// AttendNoRuns reports that g's attention stays causal inside a call's
// bidirectional key runs (Attrs.NoRuns); false for a nil graph or one that
// does not attend.
func (g *Graph) AttendNoRuns() bool {
	if g != nil {
		for _, o := range g.Ops {
			if o.Kind == OpAttend {
				return o.Attr.NoRuns
			}
		}
	}
	return false
}

// ReadsRowWindows reports that an op of g reads each row's window from the
// step's window table (Attrs.RowWindows): its attention or its centred
// convolution. A block that states none reads its rows' runs, whatever
// windows its call was handed.
func (g *Graph) ReadsRowWindows() bool {
	if g != nil {
		for _, o := range g.Ops {
			if o.Attr.RowWindows {
				return true
			}
		}
	}
	return false
}

// AttendWindow is the window g's attention states (Attrs.Window) and whether
// it is an aligned chunk (Attrs.Chunked): 0 for a nil graph, one that does
// not attend, or one whose attention sees every key.
func (g *Graph) AttendWindow() (int, bool) {
	if g != nil {
		for _, o := range g.Ops {
			if o.Kind == OpAttend {
				return o.Attr.Window, o.Attr.Chunked
			}
		}
	}
	return 0, false
}

// ReadsRowCodes reports that a matvec of g projects a row's int8 codes
// (ElemI8, Builder.Int8Codes): g quantizes its projections' inputs per row
// by their absolute maximum. False for a nil graph.
func (g *Graph) ReadsRowCodes() bool {
	if g != nil {
		for _, o := range g.Ops {
			if o.Kind == OpMatVec && len(o.In) > 0 && g.Values[o.In[0]].Elem == ElemI8 {
				return true
			}
		}
	}
	return false
}

// op appends o, an output stating no rows over its first input's grid.
func (g *Graph) op(o Op) {
	if len(o.In) > 0 {
		for _, v := range o.Out {
			if g.Values[v].Rows == 0 {
				g.Values[v].Rows = g.Values[o.In[0]].Rows
			}
		}
	}
	g.Ops = append(g.Ops, o)
}

// String is the graph in the illustrative form .notes/design/composer.md uses.
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
			fmt.Fprintf(&b, ", reads %s@%d", r, r.Version)
		}
		for _, w := range o.Writes {
			fmt.Fprintf(&b, ", writes %s@%d", w, w.Version)
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
	// ChainPIL is Pillow's antialiased resample (a picture's, OpPicture):
	// each output's window and weights from the reduction's scale, in
	// Pillow's fixed point, a horizontal pass to 8 bits and then a vertical
	// one; Taps is the most weights an output reads.
	ChainPIL
)

const (
	FilterBilinear uint8 = 1 + iota
	FilterLinear
	FilterCubic
	// FilterAntialias is torch's antialiased bilinear: every sample of the
	// side weighted by a triangle as wide as the reduction (one sample at
	// least) about the cell's centre, normalised by the weights' sum.
	FilterAntialias
	// FilterLanczos is Pillow's LANCZOS: the sinc windowed by the sinc a
	// third as wide, over three samples each side, widened by the reduction.
	FilterLanczos
)

// PILFilter reports whether f is one of Pillow's filters a ChainPIL
// resample runs.
func PILFilter(f uint8) bool { return f == FilterBilinear || f == FilterCubic || f == FilterLanczos }

// A picture's sample formats (OpPicture), as an image decoder leaves them.
const (
	// PixRGBA8 is four samples a pixel, red, green, blue and an alpha the
	// picture drops, each a byte or, as the step's request says, two bytes
	// of which the first is read; or, as the request says, cyan, magenta,
	// yellow and key, made red, green and blue as Pillow's convert("RGB")
	// makes them.
	PixRGBA8 uint8 = 1 + iota
	// PixGray8 is one sample a pixel, every channel's, a byte or two as
	// PixRGBA8's; with the request's ink word set the sample is an alpha,
	// white where there is any and black where there is none, as
	// image/color's NRGBAModel makes it (image.Alpha, image.Alpha16).
	PixGray8
	// PixYCbCr8 is a luma plane and two chroma planes, a byte a sample, the
	// chroma subsampled as the step's request says; a pixel's red, green and
	// blue are Go's image/color.YCbCrToRGB of its three samples.
	PixYCbCr8
	// PixPaletted8 is one byte a pixel, an index into a palette of four
	// bytes an entry (red, green, blue, an alpha the picture drops).
	PixPaletted8
	// PixRGBAPremul is PixRGBA8's samples with red, green and blue
	// premultiplied by the alpha (image.RGBA, image.RGBA64), made straight
	// as image/color's NRGBAModel makes them.
	PixRGBAPremul
	// PixFormats is one past the last sample format.
	PixFormats
)

// A clip's sample formats (OpSound), as a WAV stores them, interleaved.
const (
	// SoundU8 is an unsigned byte a sample, 128 its zero.
	SoundU8 uint8 = 1 + iota
	// SoundS16, SoundS24 and SoundS32 are signed little-endian integers of
	// 16, 24 and 32 bits.
	SoundS16
	SoundS24
	SoundS32
	// SoundF32 and SoundF64 are little-endian IEEE floats of 32 and 64 bits.
	SoundF32
	SoundF64
	// SoundFormats is one past the last sample format.
	SoundFormats
)

// A Pillow resize's passes (Attrs.PicPass).
const (
	PicWhole uint8 = iota
	PicBand
	PicFinish
	PicCols
)
