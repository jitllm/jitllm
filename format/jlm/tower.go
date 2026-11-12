package jlm

import "fmt"

// TowerVersion is the stated tower's revision. A record of a later one is
// refused, not read as this one.
const TowerVersion = 10

// Tower is a vision tower as the converter states it: what the engine builds
// the tower from, with no family's name in it. The converter knows a family
// (RULE 7) and writes down what its reference does: the steps a picture
// passes through, the program that turns its pixels into the first block's
// rows (Entry), the repeated block (Block), the program that turns the last
// block's rows into the language model's (Head), and where the picture's rows
// stand in a prompt (Span). Projector stays the family's label, which the
// engine never switches on.
type Tower struct {
	Version uint16
	Pre     []PreStep
	Entry   Program
	Block   TowerBlock
	Head    Program
	Span    TowerSpan
	// Taps are rows of the picture beside the head's, each from the rows a
	// block leaves, in their blocks' order: text block i adds tap i's rows
	// into the picture's after it runs (Qwen3-VL's deepstack).
	Taps []TowerTap
	// Chain, where set, is the tower's blocks each a program of its own in
	// place of the repeated Block: block i runs Chain[i] over the rows the
	// block before it left (the entry's, for the first), a grid of its own
	// shape, its binds reading block i's tensors.
	Chain []ChainBlock
	// HeadFrom is the chained blocks whose rows the head reads beside the
	// last block's: block HeadFrom[j]'s are the head's value 1+j.
	HeadFrom []int32
	// Video is how the tower reads a video, nil for none. It is written
	// after the record, in the vision section's tail.
	Video *Video
	// Geometry is how a picture becomes the tower's squares, nil where the
	// steps say it (docs/design/picture-statement.md). It is written after
	// the video, in the tail.
	Geometry *PictureGeometry
}

// ChainBlock is one block of a chained tower: the rows and width of the
// grid it reads, and its program over them (value 0).
type ChainBlock struct {
	Rows, Width int32
	Prog        Program
}

// TowerTap is a program over the rows tower block Block leaves (value 0),
// whose result is as many rows as the head's at the text model's width. Its
// binds read the tensors stored at index Block of the dense region.
type TowerTap struct {
	Block int32
	Prog  Program
}

// PreKind is one preprocessing step: an algorithm with its parameters, never
// a family.
type PreKind uint16

const (
	// PreResize resamples the whole picture, aspect not kept, to Ints[0] x
	// Ints[1] pixels with filter Ints[2] (a Filter) in arithmetic Ints[3]
	// (a Precision).
	PreResize PreKind = 1
	// PreNormalize turns each channel c of a sample read as [0, 1] into
	// (v - Floats[c]) / Floats[3+c]: the mean, then the std.
	PreNormalize PreKind = 2
	// PreResizeShort resamples the picture, aspect kept, so its shorter side
	// is Ints[0] pixels and its longer int(Ints[0] * long / short), with
	// filter Ints[1] in arithmetic Ints[2].
	PreResizeShort PreKind = 3
	// PreCenterCrop is the Ints[0] x Ints[1] window of the picture as the
	// step before left it, at ((w - Ints[0]) / 2, (h - Ints[1]) / 2)
	// truncated.
	PreCenterCrop PreKind = 4
	// PreTile cuts the picture into tiles, each a picture the tower sees:
	// between Ints[0] and Ints[1] of them, the columns x rows grid whose
	// aspect is closest to the picture's (a tie to the larger grid while the
	// picture's area is more than half of it, the grids every columns x rows
	// pair, columns outer, stably sorted by tile count), the picture resized
	// by the resize step to the grid's size in squares and cut row-major;
	// then, where Ints[2] is 1 and the grid is more than one tile, the whole
	// picture resized to one square. A picture's rows are every tile's in
	// that order.
	PreTile PreKind = 5
	// PreSmartResize resamples the picture to a size of its own, each side a
	// multiple of Ints[0] pixels: each rounded half to even to one; then,
	// where the area passes Ints[2] pixels, both scaled down by
	// sqrt(area / Ints[2]) and floored to one, or where it falls short of
	// Ints[1], scaled up by sqrt(Ints[1] / area) and ceiled to one
	// (Qwen2-VL's smart_resize); with filter Ints[3] in arithmetic Ints[4].
	// Ints[5], where stated, is 1 for a picture a side shorter than Ints[0]
	// scaled up first so that side is Ints[0], each side truncated
	// (GLM-4V's smart_resize), 0 for none.
	PreSmartResize PreKind = 6
	// PreLimitPad keeps the picture at its size unless its whole Ints[0]
	// pixel patches pass Ints[1], where both sides are scaled by
	// sqrt(Ints[1] / patches) and truncated, with filter Ints[3] in
	// arithmetic Ints[4]; then pads it black, before it is normalised, on
	// the right and at the bottom to multiples of Ints[2] pixels (Kimi-VL's
	// rescale with pad_input).
	PreLimitPad PreKind = 7
	// PreLongBox resizes the picture so its longer side is Ints[0] pixels
	// and its shorter round-half-even(side * Ints[0] / longer), at least
	// Ints[1], with filter Ints[5] in arithmetic Ints[6]; then centres it
	// on an Ints[0]-pixel square of the colour Ints[2..4] (bytes, before
	// normalising), each margin the floor of half the slack (Janus's
	// resize and pad_to_square).
	PreLongBox PreKind = 8
	// PreNaFlex resamples the picture to a size of its own in whole
	// Ints[0]-pixel patches: its own whole patches' count clamped to
	// [Ints[1], Ints[2]], then the largest scale whose sides, each ceiled
	// to a patch and at least one, hold that many, found by bisection
	// (SigLIP2's get_image_size_for_max_num_patches); with filter Ints[3] in
	// arithmetic Ints[4].
	PreNaFlex PreKind = 9
	// PreCapUnit resamples the picture so neither side passes Ints[0]
	// pixels -- both divided by longer / Ints[0] and floored, where the
	// longer passes it -- then each side rounded up to a whole Ints[1]
	// pixels, with filter Ints[2] in arithmetic Ints[3] (Pixtral's resize).
	PreCapUnit PreKind = 10
	// PreAspectBudget resamples the picture to the largest size of its
	// aspect within Ints[0] patches of Ints[1] pixels, each side a whole
	// Ints[2] pixels: both scaled by sqrt(budget area / area) and floored
	// to the unit; a side that floors to nothing one unit, the other the
	// aspect's whole multiple of it, at most the budget's units a row
	// (Gemma 4's get_aspect_ratio_preserving_size); with filter Ints[3] in
	// arithmetic Ints[4].
	PreAspectBudget PreKind = 11
	// PreTileSmart is a picture one piece at its own size or cut: its own
	// size each side a whole Ints[4] pixels, rounded half to even, then
	// scaled into [Ints[5], Ints[6]] pixels as PreSmartResize scales it; a
	// picture whose sides rounded so (at least an Ints[3]-pixel patch each)
	// hold more than Floats[0] times Ints[6] pixels is cut into the grid of
	// Ints[1] to Ints[2] Ints[0]-pixel tiles closest to its aspect (as
	// PreTile chooses one), resized onto the grid and cut row-major, then
	// the whole at its own size as a thumbnail (LFM2-VL's); with filter
	// Ints[7] in arithmetic Ints[8].
	PreTileSmart PreKind = 12
	// PreBestFit cuts the picture into tiles as PreTile does, on the canvas
	// of at most Ints[0] tiles whose limiting scale is the smallest upscale,
	// or failing any the largest downscale, the smallest area breaking a tie
	// (the canvases every rows x columns pair by tile count from Ints[0]
	// down, grouped by aspect in the order each first appears): the picture
	// resized by the resize step onto it without distortion, at most its
	// canvas and upscaled to at most one tile, laid at its top left with
	// black past it and cut row-major; then, for more than one tile, the
	// whole picture resized to one tile more (Llama 4's). A head's marker
	// rows (TOpMarkRow) are a table of three: after a tile not its row's
	// last the first, after a row's last the second, before the picture's
	// last tile the third.
	PreBestFit PreKind = 13
	// PreSliceUHD is llava-uhd's cut (MiniCPM-V's): an overview at its own
	// size -- Ints[0]^2 pixels at the picture's aspect, each side a whole
	// Ints[1]-pixel patch, rounded half to even -- then, where the picture's
	// area over Ints[0]^2 rounds up past one, the slices of the grid of at
	// most Ints[2] whose aspect is closest to the picture's (of the counts
	// one either side of that multiple): the picture resized to a whole
	// number of best-sized slices and cut row-major; the overview upscaled
	// only where nothing is cut. With filter Ints[3] in arithmetic Ints[4].
	// A head's marker runs (TOpMarkRow) are a table of six: <image> and
	// </image> about the overview, <slice> and </slice> about a slice, a
	// slice row's first after the first taking the last two (a newline and
	// <slice>) before it.
	PreSliceUHD PreKind = 15
	// PreNormalizeBF16 is PreNormalize in bfloat16: the sample times 1/255,
	// less the mean, over the std, each result and the mean and std rounded
	// to bfloat16 as torch's arithmetic on a bfloat16 tensor rounds them.
	PreNormalizeBF16 PreKind = 14
)

// Filter is a resampling filter.
type Filter uint16

const (
	// FilterTwoTap is bilinear between the two source samples around each
	// output's centre ((i+0.5)*src/n - 0.5), clamped at the edges, with no
	// antialiasing on a reduction.
	FilterTwoTap Filter = 1
	// FilterPILBilinear and FilterPILBicubic are Pillow's antialiased
	// filters: the kernel widened by the reduction's scale.
	FilterPILBilinear Filter = 2
	FilterPILBicubic  Filter = 3
)

// Precision is the arithmetic a resample keeps its samples, taps and sums in.
type Precision uint16

const (
	// PrecFloat is float32 over the decoded samples at their own depth,
	// nothing rounded between the passes.
	PrecFloat Precision = 1
	// PrecPIL is Pillow's: 8-bit RGB in, fixed-point taps of the precision
	// Pillow picks for the filter, rounded to 8 bits after each pass.
	PrecPIL Precision = 2
	// PrecTorch is torchvision's antialiased resample of an 8-bit picture:
	// Pillow's passes and filters, the taps at the widest precision (to 21
	// bits) at which an axis's largest weight holds in an int16.
	PrecTorch Precision = 3
)

// PreStep is one step of the picture's preprocessing.
type PreStep struct {
	Kind   PreKind
	Ints   []int32
	Floats []float32
}

// OpCode is a program's operation. The codes are the container's: the
// engine maps each to its own graph's op, and a code it does not know refuses
// the container.
type OpCode uint16

const (
	// TOpPatchify is the picture's Span x Span patches as rows, raster
	// order -- where Width is set, in the raster order of the grid's Width x
	// Width windows, each window's patches in raster order -- each patch
	// channel-planar, after Offset rows left for a class row
	// (graph.OpPatchify).
	TOpPatchify OpCode = 1
	// TOpLinear is In[0] through the matrix Bind[0] (out width its rows),
	// plus the bias Bind[1] where bound.
	TOpLinear OpCode = 2
	// TOpPosTable is each row's row of the position table Bind[0], from row
	// Offset on (a table at the tower's own grid).
	TOpPosTable OpCode = 3
	// TOpAdd is In[0] plus In[1].
	TOpAdd OpCode = 4
	// TOpLayerNorm is In[0]'s LayerNorm, weight Bind[0], bias Bind[1]
	// where bound, epsilon Eps.
	TOpLayerNorm OpCode = 5
	// TOpRMSNorm is In[0]'s RMS norm, weight Bind[0] where bound, epsilon
	// Eps, each Span-wide head of it where Span is set.
	TOpRMSNorm OpCode = 6
	// TOpRegroup is space-to-depth: each Span x Span block of the grid's
	// rows one row, the block's rows in raster order (graph.OpRegroup).
	TOpRegroup OpCode = 7
	// TOpLeadRow puts Bind[0]'s one row in In[0]'s first Offset rows (a
	// class row before the patches).
	TOpLeadRow OpCode = 8
	// TOpDropRows is In[0] from row Offset on.
	TOpDropRows OpCode = 9
	// TOpStated is the expression Expr over the In values, Width wide, or as
	// wide as In[0] where Width is zero.
	TOpStated OpCode = 10
	// TOpPool is an average pool: each Span x Span block of the grid's rows
	// their mean, the blocks in raster order; the grid GridW rows wide where
	// set, the request's where not.
	TOpPool OpCode = 11
	// TOpFold is each Span consecutive rows of In[0] one row (graph.OpFold):
	// a merge window's patches, which the patch order made consecutive.
	TOpFold OpCode = 12
	// TOpUntile is In[0]'s rows, the picture's merge windows in the order
	// of the blocks' windows (TowerBlock.Window), in the merge windows'
	// raster order (graph.OpUntile).
	TOpUntile OpCode = 13
	// TOpPosResample is each row's row of the position table Bind[0], a
	// Span x Span grid, resampled to the picture's patch grid at the row's
	// cell (graph.OpPosResample): the source coordinate on each axis by
	// Chain, read through Filter's taps, the rows in TOpPatchify's order
	// where Width is set, from row Offset on.
	TOpPosResample OpCode = 14
	// TOpScale is In[0] times the stored vector Bind[0], element by
	// element.
	TOpScale OpCode = 15
	// TOpSum and TOpMax are In[0]'s elements summed, and their largest:
	// one wide, or one a Span-wide head where Span is set.
	TOpSum OpCode = 16
	TOpMax OpCode = 17
	// TOpConcat is the In values side by side, in order.
	TOpConcat OpCode = 18
	// TOpSlice is In[0]'s Width elements from Offset.
	TOpSlice OpCode = 19
	// TOpRope is In[0]'s Span-wide heads turned by the block's attention
	// rotary at the row's position, laid out as Chain says (RopeAsBlock,
	// RopeNormal, RopeNeox); a block with no rotary leaves them as they are.
	TOpRope OpCode = 20
	// TOpNGramRows is the row's n-gram table rows, Width wide: its token and
	// the Span-1 before it, held in the block's recurrent tail, hashed per
	// head (Hash) into a row of table Bind[0].
	TOpNGramRows OpCode = 21
	// TOpTailConv is In[0], Width channels, convolved depthwise by Bind[0]
	// over Span taps Offset positions apart, its history in the block's
	// recurrent tail.
	TOpTailConv OpCode = 22
	// TOpShift is In[0] plus the stored vector Bind[0], element by element.
	TOpShift OpCode = 23
	// TOpRowBreak is In[0]'s rows -- the merged grid's raster, its width the
	// picture's patch grid's over Span -- with Bind[0]'s one row after each
	// of the grid's rows (HunyuanVL's newline), or after each but the last
	// where Offset is 1 (Pixtral's [IMG_BREAK]).
	TOpRowBreak OpCode = 24
	// TOpWrapRows is Bind[0]'s one row, In[0]'s rows -- the patch grid over
	// Span, with a TOpRowBreak's row after each of its rows where one came
	// before in the program -- and Bind[1]'s one row.
	TOpWrapRows OpCode = 25
	// TOpPosAxes is each row's rows of two position tables of Span rows an
	// axis, back to back in Bind[0]: the first's at the patch's column plus
	// the second's at its row (graph.OpPosAxes), from row Offset on.
	TOpPosAxes OpCode = 26
	// TOpMarkRow is In[0]'s rows -- a piece's merged grid, its width the
	// patch grid's over Width, or where Width is 0 all of In[0]'s rows --
	// after a run of at most Span consecutive
	// rows of Bind[0] and before a run of at most Offset: the runs the
	// piece's place names under the picture's fit. Under PreTileSmart a tile
	// at grid row r and column c of a grid of at most M tiles takes row
	// r*M+c before it and the thumbnail row M*M; PreBestFit and PreSliceUHD
	// state theirs; a picture of one piece takes none.
	TOpMarkRow OpCode = 27
	// TOpTailRow is In[0] with its last row Bind[0]'s one row: a class row
	// after the patches, in the row the entry leaves for it.
	TOpTailRow OpCode = 28
	// TOpPosBuckets is each row's row of the position table Bind[0], a Span
	// x Span grid, at the patch's cell bucketed onto it: of a gh x gw grid,
	// row floor(Span*y/gh) and column floor(Span*x/gw) -- NaViT's bucketing
	// of the cell's fractional coordinates, at the rational -- from row
	// Offset on (graph.OpPosBuckets).
	TOpPosBuckets OpCode = 29
	// TOpTableRows is Bind[0]'s rows as a value of their own (graph.OpTableRows).
	TOpTableRows OpCode = 30
	// TOpRowsAttend is each row of In[0] attending over the live rows of
	// In[1] (keys) and In[2] (values), every row and both ways, Span heads
	// scaled by one over the root of their width, over Width key/value heads
	// (0: Span) each read by Span/Width of them (graph.OpRowsAttend): a value
	// of In[0]'s rows, whatever the picture's grid. With In[3] and Bind[0]
	// it is a relative-position attention: row q's score over key j adds
	// In[3]'s row q dotted with Bind[0]'s row for the position q-j.
	TOpRowsAttend OpCode = 31
	// TOpUnfold is each output row of a grid GridW rows wide as the Span x
	// Span window of rows around it at Stride, tap-major, Width wide, zero
	// past the taps (graph.OpUnfold), padded as TensorFlow pads or, where
	// Offset is set, by Offset all round (PyTorch's padding): a full
	// convolution's input, its matrix the next op's.
	TOpUnfold OpCode = 32
	// TOpConv is a depthwise convolution of a grid GridW rows wide by
	// Bind[0]'s Span x Span taps (tap-major rows of the width) at Stride,
	// padded as TensorFlow pads or by Offset all round where it is set
	// (graph.OpConv).
	TOpConv OpCode = 33
	// TOpUpsample is a grid GridW rows wide Span times as large each way,
	// each row its nearest (graph.OpUpsample).
	TOpUpsample OpCode = 34
	// TOpAppendRow is In[0]'s rows, then row Offset of Bind[0]
	// (graph.OpAppendRow): a marker the reference embeds through a table of
	// the tower's.
	TOpAppendRow OpCode = 35
	// TOpFrames is the call's clip as frames (graph.OpFrames): the samples
	// the step's picture table holds, as many as the request's picture
	// width, Width of them a frame every Stride, centred, each less Eps
	// times the one before it (a preemphasis), zero past the clip. It reads
	// no value: an audio tower's entry has none.
	TOpFrames OpCode = 36
	// TOpFeatureNorm is each column of In[0] normalised over the step's
	// first live-Offset rows by their mean and their std over n-1, plus Eps,
	// the rows after zero (graph.OpFeatureNorm).
	TOpFeatureNorm OpCode = 37
	// TOpSplitRows is each row of In[0] as rows of Width of its elements
	// (graph.OpSplitRows).
	TOpSplitRows OpCode = 38
	// TOpClipMask is In[0] with the rows of a grid GridW wide (0: one row a
	// grid row) zero from the clip's valid length on: its samples over
	// Stride, halved Span times as a stride-two convolution of three taps
	// padded by one halves it (graph.OpClipMask).
	TOpClipMask OpCode = 39
	// TOpRowConv is a centred depthwise convolution of In[0] across the
	// step's live rows by Bind[0]'s Span taps a channel, channel-major, zero
	// past the live rows (graph.OpRowConv).
	TOpRowConv OpCode = 40
	// TOpSub is In[0] less In[1].
	TOpSub   OpCode = 41
	topCodes OpCode = 42
)

// A stated rotary's layout (TOpRope's Chain).
const (
	// RopeAsBlock is the block's attention's own layout.
	RopeAsBlock uint8 = iota
	// RopeNormal turns adjacent pairs.
	RopeNormal
	// RopeNeox turns each element with the one half a head away.
	RopeNeox
)

// A position table's resample (TOpPosResample): where a cell of a side n
// cells long falls on the table's side, and the taps that read it.
const (
	// ResampleAligned is align_corners: cell i at i*(Span-1)/(n-1), the
	// points of torch.linspace(0, Span-1, n).
	ResampleAligned uint8 = 1 + iota
	// ResampleHalfPixel is (i+0.5)*Span/n - 0.5.
	ResampleHalfPixel
)

const (
	// TapsBilinear is the two samples around the coordinate, each weighted
	// by one less its distance, none below zero.
	TapsBilinear uint8 = 1 + iota
	// TapsLinear is TapsBilinear with the coordinate clamped at zero first.
	TapsLinear
	// TapsCubic is the four samples around it through the cubic
	// convolution of a = -0.75.
	TapsCubic
	// TapsAntialias is torch's antialiased bilinear (interpolate with
	// antialias=True, the half-pixel chain): a triangle over the samples
	// within the scale's support of the cell's centre, at least one
	// sample's, normalised by its sum.
	TapsAntialias
)

// TowerOp is one operation of a program: its inputs are values (a program's
// inputs first, then each op's output in order), its weights roles of the
// container's tensors.
type TowerOp struct {
	Code   OpCode
	In     []uint16
	Bind   []Role
	Span   int32
	Offset int32
	Width  int32
	Eps    float32
	Expr   *Expr
	// Chain and Filter are TOpPosResample's (ResampleAligned, TapsLinear),
	// zero on every other op.
	Chain, Filter uint8
	// Hash is TOpNGramRows's, nil on every other op.
	Hash *NGramHash
	// GridW is a grid op's input grid's width in rows (TOpUnfold, TOpConv,
	// TOpUpsample; TOpPool where set, the request's grid where not) and
	// Stride its step (TOpUnfold, TOpConv); zero on every other op.
	GridW, Stride int32
}

// NGramHash is an n-gram op's hash: for n = 2..Span, HeadsPer heads each,
// head h of a token reads table row
//
//	(t0*Mul[0] ^ ... ^ t(n-1)*Mul[n-1]) mod Voc[h] + Off[h]
//
// over the token and its n-1 predecessors; a predecessor at or before an EOS
// token, or before the sequence, reads as EOS, and Image stands for a
// picture's rows.
type NGramHash struct {
	EOS, Image, HeadsPer uint32
	Mul, Off, Voc        []uint64
}

// Program is a short graph: NIn inputs, then each op's output, and the value
// Out is the program's result.
type Program struct {
	NIn uint16
	Ops []TowerOp
	Out uint16
}

// NormKind is a block's normalization.
type NormKind uint8

const (
	NormLayer NormKind = 1
	NormRMS   NormKind = 2
)

// TowerBlock is the tower's repeated block. Which tensors a block has (the
// biases, a gate, post-norms) is its inventory; what that cannot say is here.
type TowerBlock struct {
	NHead, NKVHead, HeadDim uint32
	Norm                    NormKind
	Eps                     float32
	// AttnScale is the score scale, 0 for 1/sqrt(HeadDim).
	AttnScale float32
	// Act is the MLP's activation over its gate and up where gated, its up
	// alone where not.
	Act Expr
	// Rope turns q and k by the rows' places on the picture's grid; none
	// where it has no runs.
	Rope TowerRope
	// Window, where set, is the side in patches of the square windows the
	// patch rows are ordered in -- the grid's windows in raster order, those
	// at its right and lower edges cut to it, each window's merge windows in
	// raster order -- and a block attends within its row's; Full the
	// blocks, counted from the tower's first, that attend the whole picture.
	Window int32
	Full   []int32
	// VNorm says v takes an RMS norm over each head with no weight (Gemma
	// 4's); q's and k's norms are the block's tensors where it holds them.
	VNorm bool
}

// TowerRope is a tower block's rotary over the picture's grid: each head's
// pairs -- an element and the one half the turned width on (NEOX) where
// Neox, adjacent elements where not -- turned in runs, run i's Pairs pairs
// by the row's position on axis Axis (0 its grid row, 1 its column), pair j
// of a run at frequency Base^(-2(Freq+j)/Dim). The runs cover the head's
// first 2*sum(Pairs) elements; the rest are not turned. A cell's position
// on each axis counts from Origin; a row past the grid's cells (a class row
// after them) is not turned.
type TowerRope struct {
	Base   float32
	Dim    int32
	Neox   bool
	Runs   []TowerRopeRun
	Origin int32
}

// TowerRopeRun is one run of a tower's rotary.
type TowerRopeRun struct {
	Pairs, Axis, Freq int32
}

// TowerSpan is a picture's place in a prompt: the token ids before its rows,
// the ids after them, and text before and after those -- text the prompt's
// tokenizer encodes with the prompt beside it, as a reference processor
// expands its placeholder into a string before tokenizing the whole. The
// rows are the head's, one a token; Bidir says they attend to each other
// both ways in the language model. Mark is the text a message read as a
// string carries for each of its pictures, before its text -- what the
// reference processor's template writes for an image part -- for a chat
// template that reads no typed parts; empty where the family has none.
type TowerSpan struct {
	Before, After         []int32
	BeforeText, AfterText string
	// IndexBefore and IndexAfter are the text around the picture's number
	// in the prompt, counted from 0, ahead of BeforeText (MiniCPM-V's
	// processor's <image_id>N</image_id>); empty for none.
	IndexBefore, IndexAfter string
	Bidir                   bool
	Mark                    string
	// GridDiv says the rows are a grid for the language model's rotary
	// (M-RoPE): the patch grid with each side divided by GridDiv; 0 for rows
	// that advance as text.
	GridDiv int32
}

// Valid says the record is one this build can read: a known version, known
// steps and op codes, every value read after it is made, every op's
// required weights and attributes stated. Whether the bound tensors exist is
// the container's to say (ValidAgainst).
func (t *Tower) Valid() error {
	if t.Version != TowerVersion {
		return fmt.Errorf("jlm: the stated tower is revision %d and this build reads %d", t.Version, TowerVersion)
	}
	for i, s := range t.Pre {
		switch s.Kind {
		case PreResize:
			if len(s.Ints) != 4 || s.Ints[0] <= 0 || s.Ints[1] <= 0 || !resizes(Filter(s.Ints[2]), Precision(s.Ints[3])) {
				return fmt.Errorf("jlm: the tower's step %d resizes with %v", i, s.Ints)
			}
		case PreResizeShort:
			if len(s.Ints) != 3 || s.Ints[0] <= 0 || !resizes(Filter(s.Ints[1]), Precision(s.Ints[2])) {
				return fmt.Errorf("jlm: the tower's step %d resizes the shorter side with %v", i, s.Ints)
			}
		case PreCenterCrop:
			if len(s.Ints) != 2 || s.Ints[0] <= 0 || s.Ints[1] <= 0 {
				return fmt.Errorf("jlm: the tower's step %d crops to %v", i, s.Ints)
			}
		case PreSmartResize:
			if len(s.Ints) < 5 || len(s.Ints) > 6 || s.Ints[0] <= 0 || s.Ints[1] <= 0 || s.Ints[2] < s.Ints[1] ||
				!resizes(Filter(s.Ints[3]), Precision(s.Ints[4])) || len(s.Ints) == 6 && s.Ints[5]&^1 != 0 {
				return fmt.Errorf("jlm: the tower's step %d resizes to its own size with %v", i, s.Ints)
			}
		case PreLimitPad:
			if len(s.Ints) != 5 || s.Ints[0] <= 0 || s.Ints[1] <= 0 || s.Ints[2] <= 0 || s.Ints[2]%s.Ints[0] != 0 ||
				!resizes(Filter(s.Ints[3]), Precision(s.Ints[4])) {
				return fmt.Errorf("jlm: the tower's step %d limits and pads with %v", i, s.Ints)
			}
		case PreSliceUHD:
			if len(s.Ints) != 5 || s.Ints[0] <= 0 || s.Ints[1] <= 0 || s.Ints[2] < 1 ||
				!resizes(Filter(s.Ints[3]), Precision(s.Ints[4])) {
				return fmt.Errorf("jlm: the tower's step %d slices with %v", i, s.Ints)
			}
		case PreBestFit:
			if len(s.Ints) != 1 || s.Ints[0] < 1 {
				return fmt.Errorf("jlm: the tower's step %d fits a canvas of %v tiles", i, s.Ints)
			}
		case PreNormalizeBF16:
			if len(s.Floats) != 6 {
				return fmt.Errorf("jlm: the tower's step %d normalizes with %d numbers, want 6", i, len(s.Floats))
			}
		case PreTileSmart:
			if len(s.Ints) != 9 || len(s.Floats) != 1 || s.Ints[0] <= 0 || s.Ints[1] < 1 || s.Ints[2] < s.Ints[1] ||
				s.Ints[3] <= 0 || s.Ints[4] <= 0 || s.Ints[4]%s.Ints[3] != 0 || s.Ints[0]%s.Ints[4] != 0 ||
				s.Ints[5] <= 0 || s.Ints[6] < s.Ints[5] || !(s.Floats[0] > 0) || !resizes(Filter(s.Ints[7]), Precision(s.Ints[8])) {
				return fmt.Errorf("jlm: the tower's step %d tiles or sizes with %v %v", i, s.Ints, s.Floats)
			}
		case PreAspectBudget:
			if len(s.Ints) != 5 || s.Ints[0] <= 0 || s.Ints[1] <= 0 || s.Ints[2] <= 0 || s.Ints[2]%s.Ints[1] != 0 ||
				!resizes(Filter(s.Ints[3]), Precision(s.Ints[4])) {
				return fmt.Errorf("jlm: the tower's step %d resizes within a budget with %v", i, s.Ints)
			}
		case PreCapUnit:
			if len(s.Ints) != 4 || s.Ints[0] <= 0 || s.Ints[1] <= 0 || !resizes(Filter(s.Ints[2]), Precision(s.Ints[3])) {
				return fmt.Errorf("jlm: the tower's step %d caps the picture with %v", i, s.Ints)
			}
		case PreNaFlex:
			if len(s.Ints) != 5 || s.Ints[0] <= 0 || s.Ints[1] <= 0 || s.Ints[2] < s.Ints[1] ||
				!resizes(Filter(s.Ints[3]), Precision(s.Ints[4])) {
				return fmt.Errorf("jlm: the tower's step %d resizes by patch count with %v", i, s.Ints)
			}
		case PreLongBox:
			if len(s.Ints) != 7 || s.Ints[0] <= 0 || s.Ints[1] <= 0 || s.Ints[1] > s.Ints[0] ||
				s.Ints[2]&^255 != 0 || s.Ints[3]&^255 != 0 || s.Ints[4]&^255 != 0 || !resizes(Filter(s.Ints[5]), Precision(s.Ints[6])) {
				return fmt.Errorf("jlm: the tower's step %d boxes with %v", i, s.Ints)
			}
		case PreTile:
			if len(s.Ints) != 3 || s.Ints[0] < 1 || s.Ints[1] < s.Ints[0] || s.Ints[2]&^1 != 0 {
				return fmt.Errorf("jlm: the tower's step %d tiles by %v", i, s.Ints)
			}
		case PreNormalize:
			if len(s.Floats) != 6 {
				return fmt.Errorf("jlm: the tower's step %d normalizes with %d numbers, want 6", i, len(s.Floats))
			}
			for c := range 3 {
				if s.Floats[3+c] == 0 {
					return fmt.Errorf("jlm: the tower's step %d divides channel %d by a std of 0", i, c)
				}
			}
		default:
			return fmt.Errorf("jlm: the tower's step %d is kind %d, which this build does not know", i, s.Kind)
		}
	}
	if err := t.Entry.valid("entry"); err != nil {
		return err
	}
	if err := t.Head.valid("head"); err != nil {
		return err
	}
	for i, tp := range t.Taps {
		if tp.Block < 0 || i > 0 && tp.Block <= t.Taps[i-1].Block {
			return fmt.Errorf("jlm: the tower's tap %d reads block %d, not after the tap before it", i, tp.Block)
		}
		if tp.Prog.NIn != 1 {
			return fmt.Errorf("jlm: the tower's tap %d reads %d values, want its block's rows", i, tp.Prog.NIn)
		}
		if err := tp.Prog.valid(fmt.Sprintf("tap %d", i)); err != nil {
			return err
		}
	}
	switch {
	case len(t.Chain) > 0:
		if err := t.chainValid(); err != nil {
			return err
		}
	case len(t.HeadFrom) > 0:
		return fmt.Errorf("jlm: the tower's head reads %d blocks' rows and the tower has no chain", len(t.HeadFrom))
	}
	b := t.Block
	switch {
	case len(t.Chain) > 0:
	case b.NHead == 0 || b.NKVHead == 0 || b.HeadDim == 0 || b.NHead%b.NKVHead != 0:
		return fmt.Errorf("jlm: the tower's block has %d heads over %d kv heads of %d", b.NHead, b.NKVHead, b.HeadDim)
	case b.Norm != NormLayer && b.Norm != NormRMS:
		return fmt.Errorf("jlm: the tower's block normalizes by kind %d", b.Norm)
	case b.Eps <= 0:
		return fmt.Errorf("jlm: the tower's block has an epsilon of %v", b.Eps)
	}
	if err := b.Act.Valid(); len(t.Chain) == 0 && err != nil {
		return fmt.Errorf("jlm: the tower's block activation: %w", err)
	}
	if r := b.Rope; len(r.Runs) > 0 {
		pairs := int32(0)
		for _, run := range r.Runs {
			if run.Pairs <= 0 || run.Axis < 0 || run.Axis > 1 || run.Freq < 0 {
				return fmt.Errorf("jlm: the tower's rotary has a run %+v", run)
			}
			pairs += run.Pairs
		}
		if r.Base <= 0 || r.Dim <= 0 || 2*pairs > int32(b.HeadDim) {
			return fmt.Errorf("jlm: the tower's rotary turns %d pairs of a %d-wide head at base %v over %d", pairs, b.HeadDim, r.Base, r.Dim)
		}
	}
	if b.Window < 0 || len(b.Full) > 0 && b.Window == 0 {
		return fmt.Errorf("jlm: the tower's blocks attend windows of %d with %d whole-picture blocks", b.Window, len(b.Full))
	}
	for _, i := range b.Full {
		if i < 0 {
			return fmt.Errorf("jlm: the tower's whole-picture block %d", i)
		}
	}
	if t.Span.GridDiv < 0 {
		return fmt.Errorf("jlm: the tower's rows are a grid divided by %d", t.Span.GridDiv)
	}
	if t.Geometry != nil {
		if err := t.Geometry.Valid(); err != nil {
			return err
		}
	}
	if t.Video != nil {
		if err := t.Video.Valid(); err != nil {
			return err
		}
		return t.Video.PicturesValid(t)
	}
	return nil
}

// resizes says filter f is run in arithmetic p: the two-tap filter in float,
// Pillow's in Pillow's fixed point.
func resizes(f Filter, p Precision) bool {
	switch f {
	case FilterTwoTap:
		return p == PrecFloat
	case FilterPILBilinear, FilterPILBicubic:
		return p == PrecPIL || p == PrecTorch
	}
	return false
}

func (p *Program) valid(name string) error {
	if len(p.Ops) == 0 || p.NIn == 0 && p.Ops[0].Code != TOpFrames {
		return fmt.Errorf("jlm: the tower's %s has %d inputs and %d ops", name, p.NIn, len(p.Ops))
	}
	for i, op := range p.Ops {
		made := int(p.NIn) + i
		for _, v := range op.In {
			if int(v) >= made {
				return fmt.Errorf("jlm: the tower's %s op %d reads value %d before it is made", name, i, v)
			}
		}
		in, bind := op.arity()
		switch {
		case op.Code == 0 || op.Code >= topCodes:
			return fmt.Errorf("jlm: the tower's %s op %d is code %d, which this build does not know", name, i, op.Code)
		case in >= 0 && len(op.In) != in:
			return fmt.Errorf("jlm: the tower's %s op %d (code %d) reads %d values, want %d", name, i, op.Code, len(op.In), in)
		case len(op.Bind) < bind[0] || len(op.Bind) > bind[1]:
			return fmt.Errorf("jlm: the tower's %s op %d (code %d) binds %d weights, want %d to %d", name, i, op.Code, len(op.Bind), bind[0], bind[1])
		case (op.Code == TOpPatchify || op.Code == TOpRegroup || op.Code == TOpPool || op.Code == TOpFold ||
			op.Code == TOpRowBreak || op.Code == TOpWrapRows || op.Code == TOpPosAxes || op.Code == TOpMarkRow || op.Code == TOpPosBuckets ||
			op.Code == TOpRowsAttend) && op.Span <= 0:
			return fmt.Errorf("jlm: the tower's %s op %d groups by %d", name, i, op.Span)
		case op.Code == TOpRowBreak && op.Offset&^1 != 0:
			return fmt.Errorf("jlm: the tower's %s op %d breaks rows with an offset of %d", name, i, op.Offset)
		case op.Code == TOpRowsAttend && !(len(op.In) == 3 && len(op.Bind) == 0 || len(op.In) == 4 && len(op.Bind) == 1):
			return fmt.Errorf("jlm: the tower's %s op %d attends over %d values by %d tables", name, i, len(op.In), len(op.Bind))
		case op.Code == TOpFrames && (op.Width < 2 || op.Stride < 1):
			return fmt.Errorf("jlm: the tower's %s op %d frames %d samples every %d", name, i, op.Width, op.Stride)
		case op.Code == TOpFeatureNorm && op.Eps <= 0:
			return fmt.Errorf("jlm: the tower's %s op %d normalises features with an epsilon of %v", name, i, op.Eps)
		case op.Code == TOpSplitRows && op.Width < 1:
			return fmt.Errorf("jlm: the tower's %s op %d splits rows into %d elements", name, i, op.Width)
		case op.Code == TOpClipMask && (op.Stride < 1 || op.Span < 0):
			return fmt.Errorf("jlm: the tower's %s op %d masks a clip of a frame every %d halved %d times", name, i, op.Stride, op.Span)
		case op.Code == TOpRowConv && (op.Span < 1 || op.Span%2 == 0):
			return fmt.Errorf("jlm: the tower's %s op %d convolves rows by %d taps", name, i, op.Span)
		case (op.Code == TOpLayerNorm || op.Code == TOpRMSNorm) && op.Eps <= 0:
			return fmt.Errorf("jlm: the tower's %s op %d has an epsilon of %v", name, i, op.Eps)
		case op.Code == TOpNGramRows && (op.Span < 2 || op.Width <= 0 || !op.Hash.fits(op.Span, op.Width)):
			return fmt.Errorf("jlm: the tower's %s op %d gathers %d-grams %d wide by a hash of %+v", name, i, op.Span, op.Width, op.Hash)
		case op.Code != TOpNGramRows && op.Hash != nil:
			return fmt.Errorf("jlm: the tower's %s op %d (code %d) states a hash", name, i, op.Code)
		case op.Code == TOpTailConv && (op.Span < 1 || op.Offset < 1 || op.Width <= 0):
			return fmt.Errorf("jlm: the tower's %s op %d convolves %d taps %d apart over %d channels", name, i,
				op.Span, op.Offset, op.Width)
		case op.Code == TOpRope && (op.Span <= 0 || op.Chain > RopeNeox):
			return fmt.Errorf("jlm: the tower's %s op %d turns heads of %d laid out as %d", name, i, op.Span, op.Chain)
		case op.Code == TOpSlice && op.Width <= 0:
			return fmt.Errorf("jlm: the tower's %s op %d slices %d elements", name, i, op.Width)
		case op.Code == TOpStated && (op.Expr == nil || op.Width < 0 || op.Width == 0 && len(op.In) == 0):
			return fmt.Errorf("jlm: the tower's %s op %d states no expression or no width", name, i)
		case op.Offset < 0:
			return fmt.Errorf("jlm: the tower's %s op %d offsets by %d", name, i, op.Offset)
		case op.Code == TOpPosResample && (op.Span < 2 || op.Chain < ResampleAligned || op.Chain > ResampleHalfPixel ||
			op.Filter < TapsBilinear || op.Filter > TapsAntialias):
			return fmt.Errorf("jlm: the tower's %s op %d resamples a table %d a side by chain %d and taps %d",
				name, i, op.Span, op.Chain, op.Filter)
		case op.Code != TOpPosResample && (op.Chain != 0 && op.Code != TOpRope || op.Filter != 0):
			return fmt.Errorf("jlm: the tower's %s op %d (code %d) states a resample", name, i, op.Code)
		case (op.Code == TOpUnfold || op.Code == TOpConv) && (op.Span < 1 || op.Span%2 == 0 || op.Stride < 1 || op.GridW < 1):
			return fmt.Errorf("jlm: the tower's %s op %d convolves %d taps a side at stride %d over a grid %d wide", name, i,
				op.Span, op.Stride, op.GridW)
		case op.Code == TOpUnfold && op.Width < op.Span*op.Span:
			return fmt.Errorf("jlm: the tower's %s op %d unfolds %d taps into %d", name, i, op.Span*op.Span, op.Width)
		case op.Code == TOpUpsample && (op.Span < 2 || op.GridW < 1):
			return fmt.Errorf("jlm: the tower's %s op %d upsamples by %d a grid %d wide", name, i, op.Span, op.GridW)
		case op.Code == TOpRowsAttend && (op.Width < 0 || op.Width > 0 && op.Span%op.Width != 0):
			return fmt.Errorf("jlm: the tower's %s op %d attends %d heads over %d", name, i, op.Span, op.Width)
		case op.Code != TOpUnfold && op.Code != TOpConv && op.Code != TOpUpsample && op.Code != TOpPool && op.Code != TOpClipMask && op.GridW != 0:
			return fmt.Errorf("jlm: the tower's %s op %d (code %d) states a grid %d wide", name, i, op.Code, op.GridW)
		case op.Code != TOpUnfold && op.Code != TOpConv && op.Code != TOpFrames && op.Code != TOpClipMask && op.Stride != 0:
			return fmt.Errorf("jlm: the tower's %s op %d (code %d) states a stride of %d", name, i, op.Code, op.Stride)
		}
		if op.Expr != nil {
			if err := op.Expr.Valid(); err != nil {
				return fmt.Errorf("jlm: the tower's %s op %d: %w", name, i, err)
			}
			if n := op.Expr.Reads(); n != len(op.In) {
				return fmt.Errorf("jlm: the tower's %s op %d states an expression of %d operands over %d", name, i, n, len(op.In))
			}
		}
	}
	if int(p.Out) >= int(p.NIn)+len(p.Ops) {
		return fmt.Errorf("jlm: the tower's %s returns value %d, past its last", name, p.Out)
	}
	return nil
}

// arity is the values op reads (-1 for any number) and the least and most
// weights it binds.
func (op *TowerOp) arity() (in int, bind [2]int) {
	switch op.Code {
	case TOpPatchify, TOpDropRows, TOpRegroup, TOpPool, TOpFold, TOpUntile:
		return 1, [2]int{0, 0}
	case TOpPosAxes, TOpPosBuckets, TOpTableRows:
		return 0, [2]int{1, 1}
	case TOpRowsAttend:
		return -1, [2]int{0, 1}
	case TOpFrames:
		return 0, [2]int{0, 0}
	case TOpFeatureNorm, TOpSplitRows, TOpClipMask:
		return 1, [2]int{0, 0}
	case TOpRowConv:
		return 1, [2]int{1, 1}
	case TOpUnfold, TOpUpsample:
		return 1, [2]int{0, 0}
	case TOpConv, TOpAppendRow:
		return 1, [2]int{1, 1}
	case TOpMarkRow, TOpTailRow:
		return 1, [2]int{1, 1}
	case TOpLinear, TOpLayerNorm:
		return 1, [2]int{1, 2}
	case TOpRMSNorm:
		return 1, [2]int{0, 1}
	case TOpLeadRow, TOpScale, TOpShift, TOpTailConv, TOpRowBreak:
		return 1, [2]int{1, 1}
	case TOpWrapRows:
		return 1, [2]int{2, 2}
	case TOpNGramRows:
		return 0, [2]int{1, 1}
	case TOpSum, TOpMax, TOpSlice, TOpRope:
		return 1, [2]int{0, 0}
	case TOpConcat:
		return -1, [2]int{0, 0}
	case TOpPosTable, TOpPosResample:
		return 0, [2]int{1, 1}
	case TOpAdd, TOpSub:
		return 2, [2]int{0, 0}
	case TOpStated:
		return -1, [2]int{0, 0}
	}
	return -1, [2]int{0, 0}
}

// Roles is every role the tower's programs bind, each once, in order.
func (t *Tower) Roles() []Role {
	var out []Role
	seen := map[Role]bool{}
	for _, p := range []*Program{&t.Entry, &t.Head} {
		for _, op := range p.Ops {
			for _, r := range op.Bind {
				if !seen[r] {
					seen[r] = true
					out = append(out, r)
				}
			}
		}
	}
	return out
}

func encodeTower(w *wbuf, t *Tower) {
	w.u16(t.Version)
	w.u32(uint32(len(t.Pre)))
	for _, s := range t.Pre {
		w.u16(uint16(s.Kind))
		w.u32(uint32(len(s.Ints)))
		for _, v := range s.Ints {
			w.i32(v)
		}
		w.u32(uint32(len(s.Floats)))
		for _, v := range s.Floats {
			w.f32(v)
		}
	}
	encodeProgram(w, &t.Entry)
	b := &t.Block
	w.u32(b.NHead)
	w.u32(b.NKVHead)
	w.u32(b.HeadDim)
	w.u8(uint8(b.Norm))
	w.f32(b.Eps)
	w.f32(b.AttnScale)
	encodeExpr(w, &b.Act)
	w.f32(b.Rope.Base)
	w.i32(b.Rope.Dim)
	w.bool(b.Rope.Neox)
	w.u32(uint32(len(b.Rope.Runs)))
	for _, run := range b.Rope.Runs {
		w.i32(run.Pairs)
		w.i32(run.Axis)
		w.i32(run.Freq)
	}
	w.i32(b.Window)
	w.u32(uint32(len(b.Full)))
	for _, i := range b.Full {
		w.i32(i)
	}
	w.bool(b.VNorm)
	w.i32(b.Rope.Origin)
	encodeProgram(w, &t.Head)
	for _, ids := range [][]int32{t.Span.Before, t.Span.After} {
		w.u32(uint32(len(ids)))
		for _, id := range ids {
			w.i32(id)
		}
	}
	w.str(t.Span.BeforeText)
	w.str(t.Span.IndexBefore)
	w.str(t.Span.IndexAfter)
	w.str(t.Span.AfterText)
	w.bool(t.Span.Bidir)
	w.str(t.Span.Mark)
	w.i32(t.Span.GridDiv)
	w.u32(uint32(len(t.Taps)))
	for i := range t.Taps {
		w.i32(t.Taps[i].Block)
		encodeProgram(w, &t.Taps[i].Prog)
	}
	w.u32(uint32(len(t.Chain)))
	for i := range t.Chain {
		w.i32(t.Chain[i].Rows)
		w.i32(t.Chain[i].Width)
		encodeProgram(w, &t.Chain[i].Prog)
	}
	w.u32(uint32(len(t.HeadFrom)))
	for _, b := range t.HeadFrom {
		w.i32(b)
	}
}

func encodeProgram(w *wbuf, p *Program) {
	w.u16(p.NIn)
	w.u32(uint32(len(p.Ops)))
	for i := range p.Ops {
		op := &p.Ops[i]
		w.u16(uint16(op.Code))
		w.u32(uint32(len(op.In)))
		for _, v := range op.In {
			w.u16(v)
		}
		w.u32(uint32(len(op.Bind)))
		for _, r := range op.Bind {
			w.u16(uint16(r))
		}
		w.i32(op.Span)
		w.i32(op.Offset)
		w.i32(op.Width)
		w.f32(op.Eps)
		w.u8(op.Chain)
		w.u8(op.Filter)
		w.i32(op.GridW)
		w.i32(op.Stride)
		if op.Code == TOpNGramRows {
			encodeNGramHash(w, op.Hash)
		}
		w.bool(op.Expr != nil)
		if op.Expr != nil {
			encodeExpr(w, op.Expr)
		}
	}
	w.u16(p.Out)
}

func decodeTower(r *rbuf) *Tower {
	t := &Tower{Version: r.u16()}
	// The record is the section's last: another revision's is left unread
	// and refused by Valid by its number.
	if t.Version != TowerVersion {
		r.b = nil
		return t
	}
	// Each count is bounded by its element's least size in what is left, so
	// a corrupt record cannot size an allocation (RULE 9).
	for range r.count(10) {
		s := PreStep{Kind: PreKind(r.u16())}
		for range r.count(4) {
			s.Ints = append(s.Ints, r.i32())
		}
		for range r.count(4) {
			s.Floats = append(s.Floats, r.f32())
		}
		t.Pre = append(t.Pre, s)
	}
	decodeProgram(r, &t.Entry)
	b := &t.Block
	b.NHead, b.NKVHead, b.HeadDim = r.u32(), r.u32(), r.u32()
	b.Norm = NormKind(r.u8())
	b.Eps, b.AttnScale = r.f32(), r.f32()
	decodeExpr(r, &b.Act)
	b.Rope.Base, b.Rope.Dim, b.Rope.Neox = r.f32(), r.i32(), r.bool()
	for range r.count(12) {
		b.Rope.Runs = append(b.Rope.Runs, TowerRopeRun{Pairs: r.i32(), Axis: r.i32(), Freq: r.i32()})
	}
	b.Window = r.i32()
	for range r.count(4) {
		b.Full = append(b.Full, r.i32())
	}
	b.VNorm = r.bool()
	b.Rope.Origin = r.i32()
	decodeProgram(r, &t.Head)
	for _, ids := range []*[]int32{&t.Span.Before, &t.Span.After} {
		for range r.count(4) {
			*ids = append(*ids, r.i32())
		}
	}
	t.Span.BeforeText, t.Span.IndexBefore, t.Span.IndexAfter, t.Span.AfterText = r.str(), r.str(), r.str(), r.str()
	t.Span.Bidir = r.bool()
	t.Span.Mark = r.str()
	t.Span.GridDiv = r.i32()
	// A tap is at least its block and an empty program's three counts.
	for range r.count(12) {
		tp := TowerTap{Block: r.i32()}
		decodeProgram(r, &tp.Prog)
		t.Taps = append(t.Taps, tp)
	}
	// A chained block is at least its rows, its width and an empty
	// program's three counts.
	for range r.count(16) {
		cb := ChainBlock{Rows: r.i32(), Width: r.i32()}
		decodeProgram(r, &cb.Prog)
		t.Chain = append(t.Chain, cb)
	}
	for range r.count(4) {
		t.HeadFrom = append(t.HeadFrom, r.i32())
	}
	return t
}

func decodeProgram(r *rbuf, p *Program) {
	p.NIn = r.u16()
	for range r.count(37) {
		op := TowerOp{Code: OpCode(r.u16())}
		for range r.count(2) {
			op.In = append(op.In, r.u16())
		}
		for range r.count(2) {
			op.Bind = append(op.Bind, Role(r.u16()))
		}
		op.Span, op.Offset, op.Width = r.i32(), r.i32(), r.i32()
		op.Eps = r.f32()
		op.Chain, op.Filter = r.u8(), r.u8()
		op.GridW, op.Stride = r.i32(), r.i32()
		if op.Code == TOpNGramRows {
			op.Hash = decodeNGramHash(r)
		}
		if r.bool() {
			op.Expr = &Expr{}
			decodeExpr(r, op.Expr)
		}
		p.Ops = append(p.Ops, op)
	}
	p.Out = r.u16()
}

// Tail is how much of a block's recurrent tail, per sequence, the program's
// ops keep: an n-gram's tokens before the next position and a convolution's
// history, in the order the ops come.
func (p *Program) Tail() int {
	if p == nil {
		return 0
	}
	t := 0
	for _, op := range p.Ops {
		switch op.Code {
		case TOpNGramRows:
			t += int(op.Span) - 1
		case TOpTailConv:
			t += int(op.Span-1) * int(op.Offset) * int(op.Width)
		}
	}
	return t
}

// chainValid holds a chained tower to its shape: every block a program over
// one grid of rows, no repeated block or taps beside the chain, and the head
// reading the last block's rows and each HeadFrom block's, earlier ones.
func (t *Tower) chainValid() error {
	if b := t.Block; b.NHead != 0 || b.NKVHead != 0 || b.HeadDim != 0 || len(b.Rope.Runs) > 0 || b.Window != 0 || len(t.Taps) > 0 {
		return fmt.Errorf("jlm: the tower states a chain of %d blocks and a repeated block or taps beside it", len(t.Chain))
	}
	for i := range t.Chain {
		cb := &t.Chain[i]
		if cb.Rows < 1 || cb.Width < 1 || cb.Prog.NIn != 1 {
			return fmt.Errorf("jlm: the tower's chained block %d reads %d rows of %d in %d values", i, cb.Rows, cb.Width, cb.Prog.NIn)
		}
		if err := cb.Prog.valid(fmt.Sprintf("chained block %d", i)); err != nil {
			return err
		}
	}
	if int(t.Head.NIn) != 1+len(t.HeadFrom) {
		return fmt.Errorf("jlm: the tower's head reads %d values and %d blocks' rows beside the last's", t.Head.NIn, len(t.HeadFrom))
	}
	for j, b := range t.HeadFrom {
		if b < 0 || int(b) >= len(t.Chain)-1 || j > 0 && b <= t.HeadFrom[j-1] {
			return fmt.Errorf("jlm: the tower's head reads block %d's rows of a chain of %d", b, len(t.Chain))
		}
	}
	return nil
}
