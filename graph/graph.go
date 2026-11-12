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
	// With Span set it is a value, not the step's table: each row's at its
	// position floored to a multiple of Span (a window's first), of rotary
	// Rotary's planes.
	// Where Sections or Cycle states several axes, a row has a position on
	// each, and a pair's angle is its axis's position's.
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
	// each of the state's slots of Span floats: a tail past a rule's state. A
	// primitive with a state of its own: a recurrent mixer's convolution.
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
	// OpLeadRow is In[0] with its first Offset rows each Weight[0]'s one row
	// (a tower's class row).
	OpLeadRow
	// OpDropRows is In[0] from row Offset on: output row r is input row
	// r+Offset.
	OpDropRows
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
	// state version n and writes n+1. Where Span is set it is the state's
	// floats per sequence, the matrices' and a tail past them the rule
	// neither reads nor writes. A primitive with a state of its own: a
	// linear-attention mixer's recurrence.
	OpDeltaRule
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
	// OpTailRow is In[0] with its last live row (the step's positions'
	// first word less one) Weight[0]'s one row: a tower's class row after
	// its patches.
	OpTailRow
	// OpPosAxes is each row's position rows of a table of two axes: of
	// Weight[0], the row at the cell's column plus the row at Span plus its
	// row, the cell the row's position less Offset on a grid the request's
	// first word wide.
	OpPosAxes
	// OpNGramHash is each row's n-gram table rows from its window In[0]
	// (u32, n words: the row's token, then its n-1 predecessors nearest
	// first, each its id plus one, zero for none). A predecessor that is
	// none, or where Reset at or past one whose id is the end of sequence
	// (Words[0]), reads as that id; the token itself is read as it is. With
	// c the window's ids and m the n multipliers (Words[1:1+n]), head
	// h = (g-2)*Span+k of the g-grams (g from 2 to n, k below Span) reads
	// row (c0*m0 ^ c1*m1 ^ ... ^ c(g-1)*m(g-1)) mod voc[h] + off[h],
	// unsigned 64-bit wrapping, off[h] and voc[h] Words[1+n+2h] and
	// Words[2+n+2h]: Out[0] (u32, (n-1)*Span) the rows, Out[1] (u32, n-1)
	// the next position's predecessors, the token's id plus one then the
	// window's first n-2. A primitive.
	OpNGramHash
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
	// last mark at or before the position says how the row turns -- as
	// text (MarkText), at the position plus the mark's base less its first
	// position on every axis; as a cell of a picture's grid (MarkGrid), its
	// frame times the mark's stride, row and column after the mark's base;
	// as an XD-RoPE picture's
	// (MarkGridXD), its position, column, row and the picture's ordinal --
	// and a row before every mark turns at its position. Out[0] is four
	// indices a row. A primitive.
	OpRopePos
	// OpToken is each row's token, the step's own: Out[0] one index (u32) a
	// row. A primitive.
	OpToken
	// OpNGramWindow is each row's window for OpNGramHash (Out[0], u32, Taps
	// words): the row's token In[0] (u32, one word; a row naming none, all
	// ones, reads as Words[0], the model's image id), then its Taps-1
	// predecessors nearest first, each its id plus one -- the earlier rows
	// of its run, then those the run's slot holds at Offset of the
	// recurrent state (Span floats a sequence, an id's word as a float's
	// bits). The run's predecessors after its last row are written to the
	// slot: its rows' ids plus one newest first, then the slot's. It reads
	// state version n and writes n+1. A primitive with a state of its own.
	OpNGramWindow
	// OpHalves is In[0]'s floats as binary16, two a word (Out[0], u32, half
	// In[0]'s width), the lower element in the low half, rounded to nearest
	// even as a binary16 history's pages store them. A primitive.
	OpHalves
	// OpWidenHalves is In[0]'s binary16 halves, two a word (u32), as floats
	// (Out[0], twice In[0]'s width), the low half first. A primitive.
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
	// OpUntile is In[0]'s rows -- the cells of a grid in its nested order,
	// Window x Window windows of cells, each window's cells in raster order
	// and consecutive, the windows in raster order and those at the grid's
	// right and lower edges cut to it -- in the cells' raster order: output
	// row r is the input row of cell r. The grid of cells is the request's
	// first word over Width wide (a picture's patches over its merge
	// windows) and its live rows over Width*Width over that tall.
	OpUntile
	// OpHistoryPool is, for each row, the window of Span positions holding
	// its position, read from the history it reads (rows of Width), pooled
	// per channel of Out[0]: the segments of Words, four words each -- 1
	// for the window before the row's and 0 for its own, then the offsets
	// in a row of the segment's values and gates, and in a row of the bias
	// Weight[0] (Span rows, one an offset in the window) of its biases --
	// each channel the sum over every segment's positions of the value
	// weighted by the softmax of gate plus bias. A position before the
	// history's first, or past the row's own, is no term. Only a row
	// closing its window pools all of it.
	OpHistoryPool
	// OpRowBreak is In[0]'s rows -- a grid's raster, its width the step's
	// patch grid's (the request's) over Span -- with Weight[0]'s one row
	// after each of the grid's rows, or after each but the last where
	// Offset is 1: output row r of a grid W wide is input row
	// (r/(W+1))*W + r%(W+1), or the weight's where r%(W+1) is W. Rows past
	// the last are not written.
	OpRowBreak
	// OpWrapRows is Weight[0]'s one row, In[0]'s rows and Weight[1]'s one
	// row: the input a grid of the step's patches -- the step's rows less
	// Offset lead rows, its width the request's -- over Span, each of its
	// rows Width rows longer (an OpRowBreak's), the second weight's row after
	// its last.
	OpWrapRows
	// OpMarkRow is In[0]'s rows -- a piece's merged grid, its width the
	// step's patch grid's over Width -- after a run of Weight[0]'s
	// consecutive rows the request names for the piece (composer.RequestMark,
	// its first row plus one, 0 for none; RequestMarkN its length, at most
	// Span) and before the run it names after them (RequestTrail and
	// RequestTrailN, at most Offset); rows past them are not written.
	OpMarkRow
	// OpPosBuckets is each row's row of a Span x Span position table
	// (Weight[0]) at its cell of the step's grid -- the request's grid,
	// RequestGrid wide and RequestGridH tall, the cell the row's position
	// less Offset -- bucketed onto the table: row floor(Span*y/gh), column
	// floor(Span*x/gw).
	OpPosBuckets
	// OpTableRows is Weight[0]'s rows, each a row of Out[0]: a learned
	// table the reader copies out, read whole (a resampler's queries).
	OpTableRows
	// OpRowsAttend is each row of In[0] (the queries) attending over the
	// step's live rows of In[1] (the keys) and In[2] (the values), every row
	// and both ways: NHead heads of HeadDim, the scores scaled by Scale, a
	// softmax over the live rows. The keys and values are NKVHead heads (0:
	// NHead), each read by NHead/NKVHead consecutive query heads. Out[0] has
	// In[0]'s rows. Offset, past 0, holds query row q to the keys from the
	// request's slot Offset to its slot Offset+1 (their row q). With a fourth
	// operand and a weight it is a relative-position attention
	// (Transformer-XL's, a FastConformer's): row q's score over key j adds
	// In[3]'s row q dotted with Weight[0]'s row for the position q-j -- its
	// row k the position (rows+1)/2-1-k, the projected positions stated at
	// conversion.
	OpRowsAttend
	// OpUpsample is In[0], a grid of rows GridW wide, Span times as large
	// each way: row (y, x) of Out[0] is In[0]'s row (y/Span, x/Span), the
	// nearest.
	OpUpsample
	// OpAppendRow is In[0]'s rows, then row Offset of Weight[0]: a marker
	// after a picture's rows that the reference embeds through a table.
	OpAppendRow
	// OpFrames is the step's sound as frames: In[0]'s samples where it has
	// one (a sound op's), the step's picture table holding a clip's samples
	// as f32 words where not, as many as the request's
	// picture width (composer.RequestPicW); row t of Out[0] is the Width
	// samples from t*Stride - Width/2 (a centred frame), each sample
	// y[k] = x[k] - Const*x[k-1] (a preemphasis, x[-1] zero), zero past the
	// clip.
	OpFrames
	// OpFeatureNorm is each column of In[0] normalised over the step's
	// first live-Offset rows: (x - mean) / (std + Eps), std over n-1, a
	// row from live-Offset on zero (a clip's features over its valid
	// frames, NeMo's normalize_batch per_feature).
	OpFeatureNorm
	// OpClipMask is In[0] with the rows past a clip's valid length zero: the
	// rows a grid GridW wide (GridW 0: one row a grid row), a grid row kept
	// below L, L the request's sample count (composer.RequestPicW) over
	// Stride halved Span times as a stride-two convolution of three taps
	// padded by one halves it (L = (L-1)/2+1).
	OpClipMask
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
	case OpFrames:
		return "frames"
	case OpSound:
		return "sound"
	case OpFeatureNorm:
		return "feature_norm"
	case OpClipMask:
		return "clip_mask"
	case OpSpanPool:
		return "span_pool"
	case OpTakeRows:
		return "take_rows"
	case OpRowConv:
		return "row_conv"
	case OpPicture:
		return "picture"
	case OpUntile:
		return "untile"
	case OpHistoryPool:
		return "history_pool"
	case OpPatchify:
		return "patchify"
	case OpRegroup:
		return "regroup"
	case OpLeadRow:
		return "lead_row"
	case OpDropRows:
		return "drop_rows"
	case OpFold:
		return "fold"
	case OpSplitRows:
		return "split_rows"
	case OpSplice:
		return "splice"
	case OpPosResample:
		return "pos_resample"
	case OpDeltaRule:
		return "delta_rule"
	case OpPosition:
		return "position"
	case OpSelect:
		return "select"
	case OpTailRow:
		return "tail_row"
	case OpRowBreak:
		return "row_break"
	case OpWrapRows:
		return "wrap_rows"
	case OpMarkRow:
		return "mark_row"
	case OpPosBuckets:
		return "pos_buckets"
	case OpTableRows:
		return "table_rows"
	case OpUpsample:
		return "upsample"
	case OpAppendRow:
		return "append_row"
	case OpRowsAttend:
		return "rows_attend"
	case OpPosAxes:
		return "pos_axes"
	case OpNGramHash:
		return "ngram_hash"
	case OpUnfold:
		return "unfold"
	case OpUniform:
		return "uniform"
	case OpPut:
		return "put"
	case OpRopePos:
		return "rope_pos"
	case OpToken:
		return "token"
	case OpNGramWindow:
		return "ngram_window"
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
	// StateRecState is a recurrent block's delta rule's or scan's state.
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
	// Cycle, where set (Sections all zero), assigns the axes pair by pair
	// in a cycle of Cycle pairs: pair p turns by axis CycleAxes[p%Cycle], at
	// the frequency of its cycle's first pair (p - p%Cycle).
	Cycle     int
	CycleAxes [4]uint8
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
	// Rotary is OpRopeTable's: which of the model's rotaries the table is
	// (RopeTable.Index), the step's table it writes.
	Rotary int
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
	// OpMap's bound vector. Width is an OpAttend's history row where it is
	// wider than the heads it reads (its first), and OpSelect's, Offset
	// where OpSelect's key starts in it, and OpRope's where each head's turned
	// elements start.
	Offset, Width int
	// Origin is OpRopeTable's position of a grid's first cell on each axis
	// (graph.Rotary.Origin).
	Origin int
	// Local is OpSelect's blocks kept ending at the row's own.
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
	// Fn, Order, Span and Const are the primitives' (OpMap, OpReduce,
	// OpTopK, OpRank, OpTile): the function, the order a reduction runs in, the
	// group a reduction, top-k or rank runs over (0: the row), a constant
	// operand.
	Fn    MapFn
	Order Order
	Span  int
	Const float64
	// Words are OpNGramHash's constants, Reset its cut at an end of
	// sequence.
	Words []uint64
	Reset bool
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
	// PixRGBA8's.
	PixGray8
	// PixYCbCr8 is a luma plane and two chroma planes, a byte a sample, the
	// chroma subsampled as the step's request says; a pixel's red, green and
	// blue are Go's image/color.YCbCrToRGB of its three samples.
	PixYCbCr8
	// PixPaletted8 is one byte a pixel, an index into a palette of four
	// bytes an entry (red, green, blue, an alpha the picture drops).
	PixPaletted8
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
