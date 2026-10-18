package tier

import (
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// mv is a matvec's compiled kernel plus the split it was chosen for. alt is the
// same matvec at the split the fitted table would have picked, kept so
// tuneSplit's answer can be checked in a paired A/B. Codegen is microseconds, so
// a second kernel is free.
// mvTail is a batched twin's second launch (mv.tail).
type mvTail struct {
	kern    backend.Kernel
	threads int
	width   int
}

type mv struct {
	// pad is set on a matvec over a ragged tower's padded width
	// (nn.LayerPlan.KPad): its activation is the padded int8 rows, so a
	// batched twin that reads the float rows instead (an f16 conversion
	// at k) would read them at the wrong stride, and is not taken.
	pad  bool
	kern backend.Kernel
	// tail is a second launch of the batched twin over the tokens the first
	// one's block does not divide (tileMV); it takes the same parameters.
	tail *mvTail
	red  backend.Kernel
	rows int
	// slots is 0 for an ordinary matvec and NExpertUsed for an indexed one, in
	// which case the launch is rows*slots*split threads and the kernel takes a
	// pSel parameter after pOut.
	slots int
	split int
	// bias is the per-row addend, nil for every projection that has none. The
	// kernel declares pBias after pOut, so the launch appends it.
	bias backend.Buf
	alt  *mv
	// q/k are the shape this was built from, so the batched twin can be
	// generated later without re-reading the weight.
	q kernels.Quant
	k int
	// groups is NTok/Tok: how many token columns one thread carries, expressed
	// as the number of thread groups the launch needs. Zero on a decode matvec.
	groups int
	// threads overrides rows*groups when the launch is not one thread per
	// output row, as for the matrix-instruction kernel (a warp per tile).
	threads int
	// width is a staged GEMM's workgroup (its tile's threads) where that is
	// not 128; zero is 128.
	width int
	// rowt is how many rows one decode thread carries, so the grid is
	// rows/rowt rather than rows. Zero and one both mean one row per thread.
	rowt int
	// group says the split reduces inside the threadgroup, so this matvec
	// writes its final row directly: no partial plane and no red kernel.
	group bool
	// experts and slotAct are an indexed matvec's bank size and whether each
	// slot reads its own activation -- what retuneIndexed rebuilds it from.
	experts int
	slotAct bool
	// res is this projection's kernel with a bias slot, cached like every
	// matvec kernel; the tier passes the residual there so the projection
	// writes x + W*a and the separate Add launch goes. nil where the weight
	// has a bias of its own, or is not the attention output or FFN down.
	res backend.Kernel
	// gated is the Gate variant of the FFN up projection: it writes
	// act(gate)*up into the activation buffer itself, so the gated FFN's
	// ActMul launch goes. nil elsewhere, and on a split whose last pass is a
	// reduce (which would have to apply it instead).
	gated backend.Kernel
	// gate2 is gated summing the gate's row itself (kernels.MatVecShape.Gate2):
	// gate and up in one launch (Config.GateUp2). Built beside gated.
	gate2 backend.Kernel
	// gate2In is gate2 normalising and quantizing its own operand
	// (kernels.ActIn with Norm): the FFN norm, gate and up in one launch,
	// the FFN site's gate2in recipe. Built by fuseActIn beside gate2.
	gate2In backend.Kernel
	// resIn, gatedIn and kernIn are res, gated and kern building their own
	// operand (kernels.ActIn) from the float row: pIn [pNW] in pA/pAX's
	// place. Built by fuseActIn; nil where the block cannot take them.
	resIn, gatedIn, kernIn backend.Kernel
	// resActIn is res forming act(gate)*up and quantizing it in its prologue
	// (kernels.ActIn.Gated), for an FFN whose up projection has no gated form.
	resActIn backend.Kernel
	// wantGated marks the FFN up projection that takes a gated variant
	// whenever its split allows one, so a retune to an in-group split builds
	// it even where the first split had none.
	wantGated bool
	// redN is how many outputs red sums on a batched twin that splits k: every
	// row of every token column (ragMV).
	redN int
	// act is set on sm_70's tensor-core twin (voltaMV): the kernel that writes
	// the float activation as binary16 into g.f16Buf before kern reads it, over
	// actN threads.
	act  backend.Kernel
	actN int
	// actT says act writes the token-major layout (kernels.ActF16T) the
	// fused operand writers produce (rmsF16For, actF16For). The MMA twin's
	// act writes the m16n8 fragment layout (kernels.ActF16) instead, and a
	// fused writer standing in for it hands the GEMM a scrambled operand.
	actT bool
	// vt is the GemmVolta tile a binary16 twin was built at (stagedGemm), so
	// a block's q, k and v can be put in one grid (qkvGemm); zero elsewhere.
	vt kernels.VoltaTile
}
