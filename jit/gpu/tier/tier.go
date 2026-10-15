// Package tier is jitllm's GPU decode tier: an nn.Device backed by the generated device kernels (jit/gpu).
//
// tier.GPU is a router over one devTier per card, each with its own residency
// map, budget, kernels, staging buffers and lock. It places a model's blocks
// across them fastest device first, in contiguous runs, falling back to the
// host. tier.Open is the one-device path; OpenWith and New are the
// several-device ones. See multi.go for which device, and the type comments in
// this file for which state is per device and which is shared.
//
// nn declares the device interface and this package implements it, so the
// dependency runs from here to nn and never the other way.
package tier

import (
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

type resident struct {
	qs, d, sc backend.Buf
	nrows     int
	k         int
	t         kernels.Quant
	ok        bool // false = declined, do not retry
	// shared says one buffer serves several blocks: a streamed mixture's
	// compact expert bank, live only between a block's suspension and its own
	// FFN. The per-block free walk must skip it and must not refund it; see
	// sharedCompact.
	shared bool
	// imported says these three buffers alias the packed arena's host memory
	// instead of holding a copy (a unified device; see backend.HostImport). It
	// is what makes bytes() zero. pk is that arena entry, which the arena
	// keeps mapped until the buffers are freed (kernels.Arena.Unimport).
	imported bool
	pk       *kernels.Packed
}

// bytes is what this tensor occupies on the device, from its shape rather than
// from the buffers (backend.Buf does not report its own size).
//
// An imported tensor costs nothing here: its bytes are the arena's, counted
// once against host RAM (GPU.HostReserved), and charging them again would
// double-count them and decline blocks that cost nothing.
func (r *resident) bytes() uint64 {
	// A shared bank costs zero per block: sharedCompact charges its real bytes
	// once when it is made and Close refunds them.
	if r == nil || !r.ok || r.imported || r.shared {
		return 0
	}
	qs, d, sc, err := kernels.PackedWords(r.t, r.nrows, r.k)
	if err != nil {
		return 0
	}
	return uint64(qs+d+sc) * 4
}

// resKey identifies an uploaded weight.
//
// A block-keyed entry holds no Go pointer, so the host can free the frame the
// card already has (keying on the weight's address kept every offloaded block
// resident twice). (block, slot) is also the better identity: an address
// conflates an expert bank with its expert 0, and slot separates same-shaped
// tensors such as a gate bank and an up bank.
//
// p survives for the ad-hoc path only, where block is -1: a per-matvec offer
// has no block and no slot, and its weights are host-owned. The address is the
// identity only until the host says the bytes there changed (Forget).
type resKey struct {
	block, slot int
	p           *byte // nil for a block-keyed entry
	nrows, k    int
	t           kernels.Quant
}

// kernKey identifies a compiled matvec. bias is part of the key because
// MatVecShape.Bias adds a parameter: a biased and an unbiased matvec of one
// shape are different kernels, and CUDA does not check launch arity, so a
// mix-up would silently drop the bias.
type kernKey struct {
	t     kernels.Quant
	k     int
	rows  int
	split int
	bias  bool
	// rowt is how many rows one thread carries. It is its own field because
	// split already carries two encodings (the MMA path's negative one and the
	// batched path's), and a third would collide silently.
	rowt int
	// group is the in-group split reduction: the kernel writes the final row
	// instead of a partial plane, a different output contract.
	group bool
	// gated is a Gate matvec (kernels.MatVecShape.Gate) and act its activation:
	// one more parameter and a different output, so its own entry.
	gated bool
	act   kernels.ActKind
	// gate2 is a Gate2 matvec: the gate's weight as a second stream.
	gate2 bool
}

// chooseSplit picks how many threads share a row: the fallback table when the
// measured split (tuneSplit) is off or cannot measure. Fitted to a sweep on one
// card (cmd/mvbench): splitting hurts a shape that already fills the card, and
// the peak sits nearer twice the thread slots than one times, so the rule is
// "reach about 2x the slots, cap at 32".
//
// pin is WithSplit's width. Summing S partials is a different float order from
// summing the blocks in sequence, so pinning it to 1 attributes a token
// divergence to the split rather than to a kernel bug.
func chooseSplit(rows, nb, pin int) int {
	if n := pin; n >= 1 {
		for n > 1 && nb%n != 0 {
			n /= 2
		}
		return n
	}
	const slots = 30720 // a small Ampere card's resident threads: 20 SMs x 1536
	split := 1
	for split < 32 && rows*split < 2*slots && nb%(split*2) == 0 {
		split *= 2
	}
	return split
}

// The tier's state is split into per-device and shared explicitly, because a
// field on the wrong side is a data race or a double-spent budget:
//
//	Config   what the caller decides once for the whole tier (A/B arms,
//	         debugging switches, the staging budget). One struct, shared by
//	         pointer, so `jitllm verify -ab` can flip arms in one process.
//	Stats    one device's counters and timings; GPU.Stats sums them and
//	         GPU.DevStats keeps them apart.
//	devTier  everything the device owns: residency map, budget, staging
//	         buffers, kernels, recordings, lock. Nothing here may be shared.
//
// A devTier is a complete tier over one card and satisfies nn.LayerDevice on
// its own, which is what makes GPU a router rather than a second
// implementation.
var _ nn.LayerDevice = (*devTier)(nil)

// Config is the tier's shared settings. Every field is written by the caller
// (or once at Open) and read by every device under that device's lock.
type Config struct {
	// Verify recomputes every served matvec on the host and reports the first
	// shapes that disagree. Slow; a debugging tool, not a mode to ship in.
	Verify bool
	// NoBatch refuses the batched prefill path; see Layers.
	NoBatch bool
	// NoTail pads every chunk to the full batch width instead of generating a
	// narrower kernel set for a ragged tail; the A/B arm for that.
	NoTail bool
	// NoKT refuses the transposed device K cache, leaving [position][head][dim].
	NoKT bool
	// NoFuse issues each fused kernel's parts one launch at a time
	// (kernels.RoPEQKVPaged's three writers): the per-op reference path, the
	// A/B arm and the bisection switch for a fused kernel.
	NoFuse bool
	// NoActFuse keeps the decode matvecs reading an operand quantized by a
	// launch of its own (Quantize, RMSNormQuantRows) instead of building it in
	// their prologue (kernels.ActIn): the A/B arm and bisection switch for
	// the operand prologue alone. NoFuse turns it off too.
	NoActFuse bool
	// GateUp2 runs a dense gated FFN's gate and up as one Gate2 matvec
	// (kernels.MatVecShape.Gate2): each workgroup sums its rows of both
	// matrices over the one operand and writes act(gate)*up, the bits the
	// gate's launch and up's gated launch write, where the backend's lowering
	// makes that exact (kernels.Gate2Exact). On by default.
	GateUp2 bool
	// ActInPrefetch is how many weight words each thread of an operand-prologue
	// matvec prefetches before its prologue (kernels.ActIn.Prefetch): 0 is the
	// default (8), -1 none.
	ActInPrefetch int
	// ActInSites picks which matvecs take the operand prologue, a bit each:
	// 1 q/k/v, 2 the attention output, 4 gate and up, 8 down; 0 is the
	// default set. The A/B and bisection switch per site.
	ActInSites int
	// NoMMA refuses the warp matrix instruction, leaving the dp4a tile. It is
	// the bisection switch for a wrong answer that only the matrix path shows.
	NoMMA bool
	// NoVolta refuses sm_70's f16 tensor-core matvec (kernels.MatVecMMA70),
	// which a card without the integer matrix instruction -- or NoMMA -- takes
	// before the dp4a tile, and the m16n8 binary16 GEMM (devTier.f16Gemm) that
	// sm_75 and later take before MatVecMMA. The A/B arm and the bisection
	// switch for both.
	//
	// It refuses Metal's kernels.GemmTile too (tileMV, tileMoE): both are the
	// binary16-activation twin of the int8 matvec, and a gate that holds the
	// device to the int8 path's precision (model's hybrid batch gate) asks for
	// that with this one knob on every backend.
	NoVolta bool
	// NoOperandFuse keeps a prompt chunk's norm and gated activation apart
	// from the GEMM operand they feed: RMSNorm then ActF16T, ActMul then
	// ActF16T or Quantize, instead of RMSNormF16TRows, ActMulF16T and
	// QuantizeAct (the same bits in one launch). The A/B arm and the
	// bisection switch.
	NoOperandFuse bool
	// PreferGemmInt8 puts a prompt chunk's matvecs on the staged int8 GEMM
	// (kernels.GemmInt8, sm_80 on) ahead of the m16n8 binary16 GEMM. Off by
	// default: on an L4 and an A100 the binary16 GEMM read Llama-3.2-1B's
	// prompt at 0.75 and 0.91 of llama.cpp where this one read 0.67 and 0.78
	// (gpu-kernels.md "The sm_80+ prompt"). The A/B arm.
	PreferGemmInt8 bool
	// GemmDeep forces the deep m16n8 tiles (f16DeepTiles) on every prompt
	// GEMM shape they divide (1) or off (-1); 0 leaves it to tuneGemm's
	// measurement. The A/B arm and the gate's selection switch.
	GemmDeep int
	// PreferTiles puts the collective-tile prompt kernels (kernels.GemmTile,
	// kernels.FlashPrefillTile) ahead of a backend's own -- the m16n8 GEMMs
	// and FlashPrefill80 on CUDA -- wherever the device lowers the tiles. Off
	// by default: it is the arm that gates and measures the IR path on a card
	// that has a faster special case.
	PreferTiles bool
	// NoTileTail keeps GemmTile to one blocking that divides the whole prompt
	// rather than the widest block plus a tail launch (tileMV): the A/B arm
	// and the bisection switch.
	NoTileTail bool
	// NoExactPrompt runs a prompt chunk in the batched scratch batchFor picks
	// (a power of two) rather than one built at the chunk's own grain width:
	// the A/B arm and the bisection switch.
	NoExactPrompt bool
	// NoPromptPrewarm leaves the exact prompt widths' kernels to the first
	// prompt of each width (prewarmPromptWidths): the violation arm of the
	// gate that a placed model's prompts compile nothing.
	NoPromptPrewarm bool
	// NoVoltaMoE keeps a batched mixture's expert matvecs on the dp4a grouped
	// kernel where sm_70's grouped GemmVolta -- or Metal's grouped GemmTile --
	// would run them (moegroup.go): the A/B arm for that, and its bisection
	// switch.
	NoVoltaMoE bool
	// NoRagGroup keeps a decode step of a few sequences off the decode matvec
	// that carries every token in one thread (groupMV), on the tiled twins it
	// replaced: the A/B arm for it, and its bisection switch.
	NoRagGroup bool
	// NoRagFuse runs a few-sequence step's q/k/v, residual adds and gated
	// activation as separate launches where decode fuses them (groupKern,
	// groupQKV): the A/B arm for those fusions, and the reference their gate
	// compares against.
	NoRagFuse bool
	// NoRagHeadOne projects a ragged step's one wanted row through the head's
	// batched twin over every row of the step, where it would run decode's own
	// head matvec on that row alone: the A/B arm for it, and the reference its
	// gate compares against.
	NoRagHeadOne bool
	// KVReserve is how many positions of history a device reserves up front
	// before it grows a page at a time; 0 is devKVReserve (see initKVCap).
	// devKVPage reproduces the one-page start a growth gate needs.
	KVReserve int
	// Sessions is how many concurrent sessions each placed linear block
	// reserves a recurrent pair for; 0 and 1 are one. See devTier.reserved.
	// Without it the first State fills the card and a second finds no room for
	// its own state, so its blocks run on the host. An attention block reserves
	// nothing per session: its history is paged, and a session takes pages as
	// it writes rows.
	Sessions int
	// FlashAttention enables experimental generated online attention for GPU decode.
	// Prefill, MLA, vision and devices without guaranteed 32-lane subgroups use
	// the staged path. Configure before placing layers; default is disabled.
	FlashAttention bool
	// FlashKV runs that decode attention as kernels.FlashDecodeKV -- one
	// workgroup per KV head and key partition, serving the GQA group -- in
	// place of kernels.FlashAttention's one warp per query head. It implies
	// FlashAttention; FlashSplit is then a floor, raised until every key of
	// MaxSeq has a partition to land in.
	FlashKV bool
	// FlashGroup is FlashKV's query heads per workgroup (kernels.FlashShape.Group);
	// 0 serves a whole GQA group, 1 one head.
	FlashGroup int
	// NoFlashKV keeps decode attention on the staged kernels (scores, softmax,
	// split accumulate, reduce) on CUDA and Metal, where FlashKV is the default.
	NoFlashKV bool
	// FlashWarps is FlashKV's workgroup width in warps
	// (kernels.FlashShape.Warps); 0 is the kernel's default.
	FlashWarps int
	// SharedRec keeps the device's recurrent pools in the shared form even
	// when two halves would fit -- the other arm of that form's gates.
	SharedRec bool
	// NoFlashPrefill keeps batched attention on its three-kernel form (scores,
	// softmax, accumulate) instead of kernels.FlashPrefill70 on sm_70 and
	// kernels.FlashPrefillTile on Metal -- the other arm of those kernels'
	// gates and measurements.
	NoFlashPrefill bool
	// SpillingTiles lets the batched GEMM run a tile whose kernel spills
	// registers to local memory. By default a tile the device's compiler
	// reports spilling (backend.KernelResources.SpillBytes) is passed over
	// for the next one that divides the shape; this is the other arm.
	SpillingTiles bool
	// DecodePlanOff puts pieces of kernels.ChooseDecodePlan back as the tier
	// had them before it, for the plan's A/B: 1 the split count (the square
	// root, the cap rounded down, the default width), 2 the group-per-item
	// merge, 4 no vector V loads; 7 is the whole old rule. It always decodes
	// through FlashDecodeKV.
	DecodePlanOff int
	// StagedDecode runs every paged decode through the staged kernels, the
	// path kernels.ChooseDecodePlan takes only at depth: the gates' way to
	// run it on a short history, and the other arm of the plan's A/B.
	StagedDecode bool
	// PrefillPass is the most keys one staged paged prefill launch attends
	// before the history goes in passes (pagedprefill.go); 0 is
	// prefillPassW. Rounded up to 64, and past 256 to a multiple of 256.
	PrefillPass int
	// KVStreamPages forces how many evicted pages one pass over evicted
	// history uploads (kvevict.go); 0 is every free page the layer has.
	KVStreamPages int
	// StagedPassKeys is the most keys one staged paged attention pass
	// covers; 0 sizes it to a 64 MiB plane (devStagedPlane). The staged plan's score and weight
	// planes are rows x heads x keys, so a plan sized for the context would
	// hold gigabytes of a card that a model's histories need (8 GiB of a
	// V100 for a 256-row chunk of a 131072-position Llama 3.1); a row deeper
	// than this goes in passes over its pages, folded (pagedstream.go).
	StagedPassKeys int
	// FlashSplit sets parallel key partitions for generated attention (0/1: one).
	// Configure before placing layers; each partition needs a small partial buffer.
	FlashSplit int
	// TableSplit runs the fitted chooseSplit width instead of the measured one,
	// which is the other arm of that comparison.
	TableSplit bool

	// NoGroupSplit refuses the in-group split reduction, leaving the global
	// partial buffer and its Reduce launch: the bisection switch for a kernel
	// whose correctness depends on the workgroup size matching the declaration
	// (ir.Shared, ir.Barrier).
	NoGroupSplit bool
	// ForceGroupSplit selects the in-group reduction for every eligible shape
	// without consulting the tuner, the only way to hold the split fixed and
	// vary the reduction (a pinned WithSplit makes tuneSplit return before it
	// records a verdict, so the in-group path would silently not run).
	ForceGroupSplit bool
	// ForceIndexedGroup is ForceGroupSplit for the expert (indexed) matvecs
	// alone, at whatever split they were built with. It is separate so a gate
	// can vary the expert reduction with every dense kernel held identical.
	ForceIndexedGroup bool
	// NoSegFuse keeps MLA's q/kv_a and a shared expert's gate/up as separate
	// launches (fuseSegs): the other arm of that fusion's gate and A/B.
	NoSegFuse bool
	// NoMoEFuse keeps a mixture's expert epilogues -- the gated activation
	// and the expert biases -- as separate launches (fuseMoE).
	NoMoEFuse bool
	// NoWarpNorm keeps the whole-row RMSNorm on its shared-memory tree where
	// the subgroup-shuffle reduction would otherwise be taken (warpNorm).
	NoWarpNorm bool
	// NoQKRms keeps a wide q/k norm (olmoe) on the partitioned part/apply
	// pair instead of one whole-row RMSNorm launch per vector.
	NoQKRms bool
	// DecodeRowt is how many rows one decode thread carries; 0 and 1 mean one.
	// Decode has no token columns to reuse a weight across, so this buys no
	// traffic: it buys independent accumulator chains and more consecutive
	// bytes per lane (the tile's rows are adjacent), while the split puts back
	// the threads the tile divides. Whether it pays is a property of the
	// device, so it is a field `jitllm verify -ab` can alternate.
	DecodeRowt int

	// HeadNormRepeat runs the output norm's apply pass this many extra times
	// before the real one: a measurement instrument that prices a dispatch.
	// normApply reads x, hNorm and part and writes h, so it is idempotent.
	HeadNormRepeat int

	// HeadSplit pins the k-split of the output projection only, 0 meaning the
	// tuner's choice. See prepHead.
	HeadSplit int

	// RopeTableHost uploads the host's rotary table instead of building it on
	// the device. It is both the A/B arm and the fallback: Layers is handed
	// the finished cs/csSWA tables too, so a device that cannot build the
	// kernel computes the same bytes by the other route. See
	// Stats.RopeTableWhy.
	RopeTableHost bool

	// HeadMatvecRepeat runs the output projection this many extra times. Like
	// HeadNormRepeat it is idempotent; it prices the head matvec, which a
	// per-command-buffer clock cannot localise.
	HeadMatvecRepeat int

	// KVF16 stores the V cache as packed binary16 instead of float32, halving
	// both the bytes AttnAcc reads per position and the VRAM the cache holds.
	// It is a field so `jitllm verify -ab` can alternate it in one process
	// against one copy of the device weights.
	//
	// It is V only for now. RoPE writes the K cache and, under the NEOX layout,
	// produces its two outputs half a head apart rather than adjacent, so K
	// cannot be packed without restructuring the rotation.
	KVF16 bool
	// Attention history is paged (tier/pagedattn.go,
	// docs/design/device-kv-paging.md): a sequence grows by appending pages,
	// and a page that does not fit is paged out or refused. KVPage is the
	// positions a page holds, a power of two and a multiple of 64; 0 lets the
	// pool pick from the context (devTier.kvPage).
	KVPage int
	// PerLayerSubmit issues one submission per block instead of one for the
	// whole prefix, which is the other arm of that comparison.
	PerLayerSubmit bool
	// SubmitBudget bounds one submission's run time: a range of blocks is
	// cut so that each piece runs within it at the cost measured per block
	// (subbudget.go). Zero is defaultSubmitBudget; a negative budget never
	// cuts.
	SubmitBudget time.Duration
	// VisionPlaneBudget is the most bytes a vision block's score planes may
	// take before its attention goes in query chunks (visionattn.go); 0 is
	// visionPlaneBudget. A gate sets it to force the chunked path on a
	// picture whose planes would fit.
	VisionPlaneBudget int
	// PromptRows is the widest prompt chunk the device reserves its batched
	// scratch for at placement (scratch.go); 0 is nn.MaxDevicePrefillChunk,
	// the chunk the model hands a device. A reservation that does not fit is
	// halved until it does, and Stats.ReservedPrompt says what was kept.
	PromptRows int
	// StepRows is the most rows one ragged step across sessions carries on
	// this device -- the server's step loop sets model.MaxStepRows. A device
	// that runs such steps reserves their scratch and their per-row logits at
	// placement too; 0 runs none and reserves nothing for them.
	StepRows int
	// NoScratchReserve builds the batched scratch at the first prompt, as the
	// device did before the reservation: the other arm of its gates.
	NoScratchReserve bool
	// StreamExperts places a mixture's blocks without their routed bank: gate,
	// up and down hold NExpertUsed sheets each and are refilled every token
	// from the host, out of the selection the router produced. See streamBank.
	// It is forced rather than chosen by placement, so a gate or an A/B can
	// select the streamed path on a model that would fit.
	StreamExperts bool
	// StreamFixedSel makes a streamed block fill its compact bank with experts
	// 0..k-1 instead of the ones the router chose, skipping the Sync and the
	// readback that make the selection known. It produces wrong tokens on
	// purpose: it is a measurement instrument that prices the mid-block
	// suspension, never a mode to run in.
	StreamFixedSel bool
	// StreamGroups is how many pieces a streamed block's selection is uploaded
	// in, so that group g's PCIe transfer overlaps group g+1's file read. 1 is
	// serial read-then-upload, and is what a host with no PrefetchExperts gets;
	// 0 measures it per device (streamtune.go). See streamBank.fill.
	StreamGroups int
	// StreamGroupsStart is where the group tuner starts (the container's
	// figure, format/jlm streamGroupsFor); it is not a pin.
	StreamGroupsStart int
	// StreamDirectBytes is the plane size from which a streamed sheet is sent
	// straight from its host frame rather than gathered; 0 is directSheet. A
	// field so a gate can force either path on a fixture of any size.
	StreamDirectBytes int
	// StreamNoPin sends direct sheets from their host frames as pageable
	// transfers instead of through page-locked halves: the other arm.
	StreamNoPin bool
	// StreamPinHalf is each page-locked half's size; 0 is pinHalf. A gate
	// sets it to one byte so every piece takes its own half and the
	// pipeline's second stage runs on a fixture's small sheets.
	StreamPinHalf int
	// StreamCacheSlots makes each streamed block's bank an expert cache of
	// this many sheets, kept across tokens: a selected expert already there
	// is neither read nor sent. 0 sizes the cache from the room each card has
	// left after placement (GPU.sizeAutoCaches); negative, or a stated size
	// not above the selection, is no cache; a device without room for a
	// block's stated cache gives that block the plain bank
	// (Stats.StreamCacheShort).
	StreamCacheSlots int
	// StreamPrefetch runs the cross-layer probe (StreamProbe's launches) at
	// each streamed block's suspension and starts reading the next block's
	// predicted experts into host frames behind this block's transfers. Host
	// reads only; see streamBank.prefetch.
	StreamPrefetch bool
	// HybridExperts runs a streamed block's routed experts on the host
	// (nn.LayerWeights.HostExperts) instead of sending their sheets: the
	// block's base, router and shared experts stay on the card, and the
	// experts' input and output vectors are all that cross the bus.
	HybridExperts bool
	// NoHybrid sends an auto-streamed block's expert sheets to the card
	// instead of running its experts on the host, which is that block's
	// default (GPU.AutoStream).
	NoHybrid bool
	// NoAutoStream leaves a mixture block that no device can hold resident
	// on the host, as before GPU.AutoStream, instead of streaming it.
	NoAutoStream bool
	// StreamProbe runs, at each streamed block's suspension, the next block's
	// router over this block's normed row and scores its top-k against the
	// selection that block then makes (Stats.ProbeHits): the measurement of a
	// cross-layer expert prefetch. It costs a router launch and a round trip a
	// block, so it is an instrument, never a mode to run in.
	StreamProbe bool
	// StreamSelLog, when set, is handed every streamed block's selection as it
	// comes home (block index, the k expert ids in rank order). The slice is
	// reused; copy it to keep it.
	StreamSelLog func(block int, sel []uint32)
	// Persist composes each decode block's launches into persistent kernels
	// (persist.go): one cooperative grid per run of launches, phases
	// separated by a grid barrier, the per-op kernels' own bodies. Off by
	// default: on a V100 a grid barrier costs more than the launch boundary
	// it replaces inside a CUDA graph (gpu-kernels.md, "A persistent decode
	// block"). CUDA only; elsewhere it is declined (Stats.PersistWhy).
	Persist bool
	// PersistDeps orders a persistent kernel's tasks with dependency
	// counters (ir.PersistDeps): a task waits only for the tasks it reads or
	// overwrites, a tile of an elementwise task only for the producer tiles
	// whose elements it touches, and a workgroup with nothing to do in a task
	// does not wait for it. A mode of Persist; off by default.
	PersistDeps bool
	// PersistGrid caps a persistent kernel's grid below the co-residency
	// limit; zero takes the limit, or the widest launch's groups when fewer.
	PersistGrid int
	// GateUpFuse runs a dense gated FFN's gate and up as one launch over
	// their shared operand (kernels.MatVecSegments) and has down form
	// act(gate)*up and quantize it in its prologue: three launches where there
	// were four, every row reduced as before. Off by default.
	GateUpFuse bool
	// NoGraph issues the token's launches one driver call at a time instead of
	// replaying a captured graph: the other arm of that comparison, a field so
	// both arms can be live in one process.
	NoGraph bool
	// NoTowerChunks refuses a picture whose whole non-causal set does not fit
	// rather than running it row-chunked (chunkTower, LayersPhase).
	NoTowerChunks bool
	// NoPickMV runs the label pick as the head's whole matvec and a gather
	// (the arm before kernels.MatVec's PickOf), for the A/B.
	NoPickMV bool
	// ScalarSoftmax runs the thread-per-head softmax instead of the warp
	// reduction, the other arm of that comparison. It is ignored on a device
	// that did not pass pickSoftmax's check, where the scalar kernel runs
	// anyway.
	ScalarSoftmax bool
	// ScalarLinearRows builds a linear block's ragged step -- a batch's, a
	// step across sessions, a speculation's -- on the scan's one-thread form
	// (a thread a state row, no subgroup) even where the device guarantees 32
	// lanes: the form a device without the guarantee runs, selected on one
	// that has it so the form is gated on real hardware. Set before the first
	// step; it is read when a step's kernels are built.
	ScalarLinearRows bool
	// KVKeepWindow keeps a windowed layer's pages behind its window instead
	// of releasing them (devTier.windowPages): the other arm of that
	// comparison, which must give the same logits bit for bit.
	KVKeepWindow bool
	// StageLimit bounds the host memory held by packs that are prepared and not
	// yet uploaded. Zero means defaultStageLimit. It is shared because the
	// memory is the host's: two devices under their own copy of the limit would
	// hold twice the bound. See stageRoom.
	StageLimit uint64

	// kb is this tier's measurement and bisection knobs (knobs.go), tiles the
	// batched prefill geometry resolved from them, and center the weight
	// centering its emitters are built with. Per tier, never package state:
	// two models in one process keep their own.
	kb     knobs
	tiles  tiles
	center kernels.Center
	// mvMin is the average weight bytes per crossing at which serving matvecs
	// one at a time breaks even. It is a property of the link and the model,
	// not a card; each device accumulates its own sample against it. See
	// mvPays.
	mvMin uint64
	// failAt and the per-device calls counter are always zero unless built with
	// -tags jitllmfault.
	failAt []int
	// FillFirst places blocks on the first device until it is full, then the
	// next, instead of spreading a model that fits across the devices in
	// proportion to their budgets (GPU.PlanBlocks). The spread leaves each card
	// room for the batched prefill's scratch; fill-first leaves the first
	// card none, which is what a caller packing one card, or a gate needing a
	// full card, asks for.
	FillFirst bool

	// NoUnpack refuses the device unpack, so every tensor is packed on the host
	// and uploaded packed: the A/B arm, a field so both arms live in one
	// process. The device unpack ships by default, unlike Paging, because it is
	// the same bytes reaching the same buffers by a cheaper route
	// (TestUnpackMatchesHostPacker).
	NoUnpack bool

	// arena is the packed arena: every block's weights in the device layout,
	// host-resident, so a block paged back onto a card is a DMA rather than a
	// repack. It is shared by every device because the bytes are one heap. Its
	// retention budget is zero by default; see kernels.NewArena and SetArena.
	arena *kernels.Arena
	// stage is the host memory held by packs prepared and not yet uploaded,
	// across every device (one heap, so one pool; see Pool). Its limit is 0,
	// unbounded, because StageLimit is settable after Open; Prewarm compares
	// Used() against whatever it says now.
	stage *Pool
}

// Stats is one device's reporting, or the sum over a tier's devices. It is
// per device so "which device did the work" survives: a total cannot tell
// "both took half the model" from "the second took nothing".
type Stats struct {
	// Device is the label these counters belong to, or the whole tier's label
	// on a summed Stats.
	Device string
	// The two kinds of decline are counted separately because they mean
	// opposite things: an unsupported format is a missing kernel to build,
	// while a tensor that did not fit is the VRAM budget doing its job.
	// DeclineWhy is the reason that went with Declined, kept apart from
	// LastErr, which the last writer wins (a later prefill refusal would
	// overwrite why a block was declined).
	DeclineWhy string
	NoKernel   int // matvecs refused because the quant format has no GPU kernel
	// Failed counts matvec kernels (and their split reductions) the device
	// would not compile. A shape counts once: the failure is cached, and every
	// later matvec of that shape is refused without another attempt.
	Failed   int
	TooSmall int // matvecs refused because one crossing costs more than they save
	LastErr  string
	Declined int // tensors refused because the VRAM budget was already spent
	// ConvBlocks is convolutional tower blocks run on the device (conv.go).
	ConvBlocks int64
	NoRoom     int // tensors the DEVICE refused: something else is using the card
	// SessionDeclines is blocks a session could not take because its history
	// did not fit beside the sessions already there (see Config.Sessions).
	SessionDeclines int
	KVBytes         uint64 // attention history held: the contiguous caches and the pages sequences own
	// KVPoolBytes is the paged pool's charge: every layer's buffers, the
	// dummy and the free pages included -- what competes with weights for the
	// budget, where KVBytes is what sequences hold of it.
	KVPoolBytes uint64
	// BudgetUsed is what the device's budget has spent: weights, scratch and
	// the pool. SetBudget below it pages something out; at it, nothing grows.
	BudgetUsed uint64
	// ScratchBytes is the part of BudgetUsed the device's own buffers hold:
	// the block scratches and the staging (scratch.go).
	ScratchBytes uint64
	// Allocated is what the device's buffers hold by the backend's own count
	// (backend.AllocCounter), 0 where it keeps none, and RoundingBytes the
	// part of BudgetUsed that is the driver's page rounding of them.
	Allocated, RoundingBytes uint64
	// ReservedPrompt and ReservedStep are the prompt chunk and the ragged
	// step the device reserved its scratch for at placement, 0 where it
	// reserved none. A ReservedPrompt narrower than Config.PromptRows is the
	// reservation halved to fit.
	ReservedPrompt, ReservedStep int
	// TowerRows is the rows the device's tower blocks are built for
	// (visrows.go): a picture's, rounded up, never the tower's largest grid
	// unless a picture asked for it. 0 with no tower block placed.
	TowerRows int
	// ScratchRefused counts batched scratches of another width a prompt or a
	// step did not build because the budget had no room; it ran at the
	// reserved width instead.
	ScratchRefused int
	// PromptSplits counts batched chunks the device could not run whole for
	// want of a scratch: they went in narrower submissions or a row at a time.
	PromptSplits int
	// RecBytes is a linear block's recurrent state, which competes for the same
	// budget and is reported apart because it is constant in the context
	// length, where the KV cache grows.
	RecBytes uint64
	// KVGrows is how many times a KV pool layer's buffers grew (growKVLayer):
	// zero on a session that never outran the pages placement and PrewarmKV
	// gave it, and the way to tell "the pool was big enough" from "growth
	// never ran".
	KVGrows int
	Served  int // matvecs served on the device
	// Forgotten counts lone-matvec copies dropped because the host said the
	// bytes at their address changed (Forget): the selection check that a
	// refilled frame reached the device at all.
	Forgotten int
	// RowtTiles and RowtPlain count decode matvecs built with and without the
	// row tile: the selection check, since "the tile made no difference" and
	// "the tile never ran" are otherwise the same number.
	RowtTiles int
	RowtPlain int
	// GroupedMoE counts mixture blocks a batched chunk ran grouped by expert
	// (moegroup.go) rather than one row at a time.
	GroupedMoE int
	// GroupedFloat counts the grouped expert matvecs a batched mixture ran
	// over a float bank (F32, F16, BF16): the selection check for that form.
	GroupedFloat int
	// MLABatched counts latent-attention blocks a batched chunk ran as one
	// submission (mlabatch.go) rather than one row at a time.
	MLABatched int
	// IdxSelects counts the lightning indexer's selections a device ran
	// (DeepSeek V3.2, indexer.go): one per attention call of an indexer block.
	IdxSelects int
	// MSASelects counts MiniMax Sparse Attention's block selections a device
	// ran (msa.go): one per attention call of a selecting block.
	MSASelects int
	// QSASelects counts QSA block selections a device ran (qsa.go): one per
	// attention call of a Qwen3.8-Flash-Next block.
	QSASelects int
	// AltUpBlocks counts Gemma 3n blocks a device ran its AltUp prediction
	// for (altup.go).
	AltUpBlocks int
	// DS4Blocks counts DeepSeek V4 attention halves a device ran, and DS4HC
	// its hyper-connection mixes (ds4.go).
	DS4Blocks, DS4HC int
	// K3Mixes counts Kimi-K3 residual-attention mixes a device ran, and
	// K3Latent its latent mixtures (k3.go).
	K3Mixes, K3Latent int
	// Q4EMixes counts Qwen3.8-Flash-Next hyper-connection mixes a device
	// ran, and Q4ENGram its n-gram embeddings (q4e.go).
	Q4EMixes, Q4ENGram int
	// LinearBatched counts recurrent (linear-attention) blocks a batched chunk
	// ran through the chunked delta rule (kernels.GatedDeltaScan) rather than
	// one row at a time.
	LinearBatched int
	// LinearFused counts recurrent blocks whose rule ran as ONE launch from the
	// convolution's output (kernels.GatedDeltaFused) -- decode or chunk.
	LinearFused int
	// RecShared counts recurrent pools sized in the shared form (recpool.go):
	// one resident state a slot and the device's next-state scratch.
	RecShared int
	// VoltaMV counts batched matvecs built on sm_70's f16 tensor cores
	// (kernels.MatVecMMA70); a refused shape falls to dp4a with the same answer.
	VoltaMV int
	// RagGroupMV counts a few-sequence decode step's matvecs run as the decode
	// matvec with every token in one thread (groupMV) rather than a tiled twin.
	RagGroupMV int
	// RagQKVFused, RagResFused and RagGateFused count a few-sequence step's
	// launches that carried decode's fusions over every token: q/k/v in one
	// launch, the residual added by the projection, and act(gate)*up written
	// by the up projection.
	// GateUpFused counts dense gated FFNs whose gate and up ran as one
	// launch and whose down formed act(gate)*up (Config.GateUpFuse).
	GateUpFused int
	// GateUpWhy is why the last block's gate and up could not be fused.
	GateUpWhy                              string
	RagQKVFused, RagResFused, RagGateFused int
	// GateUp2Runs counts gated FFNs whose gate and up ran as one Gate2
	// launch (Config.GateUp2).
	GateUp2Runs int
	// PersistLaunches counts persistent kernels launched (Config.Persist),
	// PersistParts the per-op launches they ran, PersistPhases their
	// grid-barrier-separated phases, PersistCompiles the compositions built
	// and PersistFallbacks those that failed and ran per-op; PersistWhy is
	// why the path is declined on this device.
	// PersistDepLaunches counts those launched with dependency counters
	// (Config.PersistDeps), PersistDepTiles their tile-wise edges.
	PersistLaunches, PersistParts, PersistPhases, PersistCompiles, PersistFallbacks int
	PersistDepLaunches, PersistDepTiles                                             int
	PersistWhy                                                                      string
	// RagHeadOne counts ragged steps whose head ran decode's own projection on
	// the one row that wanted logits, rather than a batched twin over every
	// row (a prompt chunk's last row, alone in wanting the head).
	RagHeadOne int
	// SampleReads counts tokens whose sampler ran on the device and read
	// back k candidates instead of the vocabulary (nn.Head.SampleK).
	SampleReads int
	// PickReads counts heads whose label pick ran on the device and read
	// back the picked logits instead of the vocabulary (nn.Head.Pick).
	PickReads int
	// SampleLaunches counts the sampler's launches: two top-k passes a
	// token, and the penalty's when it runs.
	SampleLaunches int
	// VoltaGemm counts those of them that are the shared-memory-staged
	// kernels.GemmVolta rather than MatVecMMA70.
	VoltaGemm int
	// SpillRefused counts GemmVolta tiles compiled and passed over because
	// the compiler reported them spilling (Config.SpillingTiles).
	SpillRefused int
	// GemmF16 counts batched matvecs built as GemmVolta on the m16n8 binary16
	// instruction (sm_75 on, devTier.f16Gemm) rather than MatVecMMA.
	GemmF16 int
	// GemmDeep counts those built on the deep tiles (f16DeepTiles).
	GemmDeep int
	// GemmInt8 counts batched matvecs built as kernels.GemmInt8, the staged
	// int8 GEMM (sm_80 on, devTier.int8Gemm).
	GemmInt8 int
	// NormF16, ActF16Fused and QuantActs count a prompt chunk's fused operand
	// launches: a norm that wrote the binary16 GEMM's operand
	// (RMSNormF16TRows), a gated activation that did (ActMulF16T), and one
	// that quantized for the int8 matvec (QuantizeAct). Each is a conversion
	// launch, and a pass over the float row, that did not run.
	NormF16, ActF16Fused, QuantActs int
	// KVPrewarmPages counts pages a pool grew ahead of its history
	// (PrewarmKV): the growth a State's first prompt no longer pays.
	KVPrewarmPages int
	// GroupedVolta counts a batched mixture's expert matvecs run as grouped
	// GemmVolta on sm_70's tensor cores, or grouped GemmTile on Metal, rather
	// than the dp4a grouped matvec.
	GroupedVolta int
	// MLAScores70 counts batched latent-attention scores run on sm_70's
	// m8n8k4 over the row-major latent cache rather than the FMA tile.
	MLAScores70 int
	// MLAAcc70 counts the same blocks' accumulates on m8n8k4.
	MLAAcc70 int
	// ActF16Launches counts the f16 conversions batched sm_70 matvecs ran,
	// and QuantSkipped the int8 quantizes a block skipped because every
	// matvec reading them was on sm_70's f16 path, which reads the float.
	ActF16Launches, QuantSkipped int
	// ResidualFused counts projections that wrote x + W*a themselves, with
	// the residual in the bias slot, instead of a separate Add launch.
	ResidualFused int
	// ResidScaled counts residual adds that scaled the block output first
	// (Granite's residual_scale, kernels.AddScaled): the selection check that
	// a scaled model's blocks took the scaled add rather than a fused one.
	ResidScaled int
	// RowtSplitTiles counts the ones built with both a tile and a k-split, the
	// combination with its own addressing (the partial-sum row is
	// splitSeg*Rows + baseRow + j), which RowtTiles alone cannot show ran.
	RowtSplitTiles int
	// IndexedGroup counts expert (indexed) matvecs built with the in-group
	// split reduction, by mkkID or retuneIndexed: the selection check for a
	// gate on that kernel.
	IndexedGroup int
	// SegFused counts launches of a fuseSegs pair (MLA's q with kv_a, a shared
	// expert's gate with up) -- once per launch, so a recorded graph counts
	// its recording only.
	SegFused int
	// MoEFused counts expert matvec launches with a fused epilogue (fuseMoE),
	// once per launch, so a recorded graph counts its recording only.
	MoEFused int
	// QKRms counts wide q/k norms run as whole-row RMSNorm launches, once per
	// block launched, so a recorded graph counts its recording only.
	QKRms int
	// RopeTables counts rotary table rows built on the device and
	// RopeTableUploads the rows uploaded from the host, cumulative over the
	// session; a model with two rotary bases charges two rows per position.
	// An upload is charged the whole staged table (the batch width, padding
	// included) and the kernel only the valid rows, so on a padded chunk the
	// arms report different totals for the same work. They are the selection
	// check: the two tables agree to one ulp by design, so no output
	// comparison can say which ran.
	RopeTables       int
	RopeTableUploads int
	// RopeTableWhy is what stopped the device from building its own table, or
	// "" where it builds one. A block is never declined for this (the host's
	// table is a correct answer), so the reason is kept here.
	RopeTableWhy string
	// StreamFills counts blocks whose routed bank was assembled from the
	// router's own selection this token, and StreamBlocks counts placements of
	// such a block, cumulative like Blocks and Captures. They are the selection
	// check: a streamed and a resident block compute the same logits.
	StreamFills int
	// StreamOverlaps is how many group reads were issued while an upload was in
	// flight; zero with StreamGroups at its default.
	StreamOverlaps int
	StreamBlocks   int
	Blocks         int // blocks run entirely on the device
	// TPack and TUpload split PrepLayer's cost into the half that is pure host
	// arithmetic and the half that is a device transfer. They decide whether
	// preparing a block off the main loop is worth anything: only the first can
	// overlap a token, since the device lock is held for a whole token and CUDA
	// forbids allocation inside a Session.
	TPack, TUpload time.Duration
	// TSubStage, TSubLaunch and TSubRead split a submission's HOST time: the per-token
	// staging copies before it, the replay (or emit) that issues it, and the
	// read that waits for its answer -- over Submits of them. What the device does
	// meanwhile is TSubRead's share; the other two leave it idle.
	TSubStage, TSubLaunch, TSubRead time.Duration
	TCapture                        time.Duration // inside TSubLaunch: recording new launch sequences
	// TStreamRead and TStreamPut split the mid-block suspension by wall: the
	// host read that makes the routed sheets valid, and the transfers that put
	// them on the card. TStreamWait is the drain before the selection can be
	// read at all. A CPU profile cannot split this, since the reads overlap
	// across goroutines.
	TStreamWait, TStreamRead, TStreamPut time.Duration
	// TStreamSheet, TStreamCopy and TStreamH2D split TStreamPut: finding each
	// routed sheet in the host pager (a read when the page is not resident),
	// gathering it into the staging buffer, and the host-to-device transfer of
	// StreamBytes.
	TStreamSheet, TStreamCopy, TStreamH2D time.Duration
	StreamBytes                           int64
	// StreamDirect counts sheet planes sent straight from a host frame, and
	// StreamGathered the gathered transfers: the selection check for put.
	// StreamPinned counts the direct planes that went through the page-locked
	// halves (sendPieces).
	StreamDirect, StreamGathered, StreamPinned int
	// StreamCacheHits and StreamCacheMisses count selected experts found in
	// and missing from a block's expert cache; StreamCacheShort counts blocks
	// given the plain bank for want of room.
	StreamCacheHits, StreamCacheMisses, StreamCacheShort int
	// HybridRuns counts streamed blocks whose experts ran on the host
	// (Config.HybridExperts), and THybrid the wall of those host calls.
	HybridRuns int
	// HybridRows counts rows of batched chunks whose experts ran on the host.
	HybridRows int
	THybrid    time.Duration
	// StreamCacheSize is the largest expert cache a block was given, in
	// sheets: the check that an auto-sized cache is the size intended.
	StreamCacheSize int
	// StreamGroupsTuned is the group count the fill tuner holds (0: not
	// tuning, a stated Config.StreamGroups).
	StreamGroupsTuned int
	// StreamPinHalfTuned and StreamDirectTuned are the page-locked half and
	// the direct-send threshold the fill tuner holds, in bytes.
	StreamPinHalfTuned, StreamDirectTuned int
	// AutoMeanBase is the model's mean mixture base GPU.AutoStream was told.
	AutoMeanBase uint64
	// StreamPrefetched counts experts the cross-layer prefetch read, and
	// TStreamPrefetchWait the wall a fill spent joining it.
	StreamPrefetched    int
	TStreamPrefetchWait time.Duration
	// ProbeHits of ProbeExperts routed experts were in the previous block's
	// cross-layer prediction (Config.StreamProbe); ProbeFused counts probes
	// skipped because the route fuses the rank with the weights.
	ProbeHits, ProbeExperts, ProbeFused int
	// TPrewarm is packing done off the main loop, kept apart from TPack
	// because only TPack is on the critical path.
	TPrewarm time.Duration
	// Captures is how many times the launch sequence had to be recorded: one
	// per prompt plus one per scoreGrain tokens is the design, and one per
	// token means the key is moving and the recording is pure overhead.
	Captures int
	// AttnMMA reports that the batched scores kernel runs on the warp matrix
	// instruction, whose operands are binary16 -- so the caller's gate has to be
	// NMSE and token ids rather than bit equality.
	AttnMMA bool
	// FlashLaunches counts encoded fused-attention launches (not graph replays).
	FlashLaunches int
	// FusedLaunches counts encoded launches of a kernel that stands for
	// several (kernels.RoPEQKVPaged), not graph replays: the selection check
	// that the fused form ran rather than its parts.
	FusedLaunches int
	// ActInLaunches counts encoded matvec launches that built their own operand
	// (kernels.ActIn), and ActInSaved the norm or quantize launches they
	// replaced: the selection check that the prologue ran.
	ActInLaunches, ActInSaved int
	// ActInDeclined counts prologue kernels the device would not launch at
	// their width (backend.ErrGroupTooWide): those shapes keep two launches.
	ActInDeclined int
	// ActInTimed and ActInKept count the sites tuneActIn decided on this card
	// (from the cache or a trial) and the ones where the prologue won.
	ActInTimed, ActInKept int
	// PagedLaunches counts encoded paged-attention launches: the check that a
	// paged run read its history through the pool.
	PagedLaunches int
	// PagedPrefillLaunches counts layer attentions of a prefill chunk that ran
	// the paged prefill kernels (pagedprefill.go) rather than FlashDecodeKV.
	PagedPrefillLaunches int
	// KVEvictions counts pages sent home (kvevict.go), one per layer-page.
	KVEvictions int
	// KVStageReads counts page reads the card refused a buffer for, which
	// came home through the pool's reserve (readStaged).
	KVStageReads int
	// KVWindowReleased counts a windowed layer's pages released behind its
	// window (devTier.windowPages), one per layer-page.
	KVWindowReleased int
	// KVStreamPasses counts attention passes over evicted history.
	KVStreamPasses int
	// StagedPasses counts attention passes of streamed calls: those over
	// evicted history and those past Config.StagedPassKeys on the card.
	StagedPasses int
	// PipelinePieces counts the pieces of pipelined prompt chunks this device
	// ran (GPU.pipeline).
	PipelinePieces int
	// SubsBeside counts submissions that started while another was in flight
	// on the device (beginSub): the selection check that sessions stepped at
	// once rather than one after another.
	SubsBeside int
	// RanBeside counts submissions whose device work began while another's
	// was running on the device -- measured inside the backend's session, so
	// a lock anywhere between the tier and the device, the backend's own
	// included, keeps it at zero where SubsBeside, counted at the ticket,
	// would not.
	RanBeside int
	// LanesWaited counts calls that found every lane taken and no room for
	// another, and waited for one (takeLane).
	LanesWaited int
	// SessionRows counts rows this device ran in steps across sessions
	// (gpuSession.LayersSessions).
	SessionRows int
	// SessionLinear counts linear blocks those steps ran, every row's
	// recurrent state in one launch: the selection check that a hybrid's step
	// across sessions did not leave its recurrence to anything else.
	SessionLinear int
	// LinearRowsScalar counts linear blocks ragged steps ran on the scan's
	// one-thread form (prepRagLinear) rather than a lane group a state row:
	// the selection check that a device without a guaranteed subgroup -- or
	// one forced by ScalarLinearRows -- stepped its rows through that form.
	LinearRowsScalar int
	// RecChainRows counts rows that stepped a linear block's recurrent state
	// after another row of their own sequence in the same ragged step -- a
	// prompt chunk's rows past its first, once per linear block: the selection
	// check that a hybrid's chunk rode as a run rather than a token a step.
	RecChainRows int
	// RowsLatent counts latent-attention blocks ragged steps ran (LayersRows,
	// and LayersSessions across sessions), every row reading its own paged
	// latent history: the selection check that an MLA model's step ran on the
	// device rather than one sequence after another.
	RowsLatent int
	// RecAligns counts sessions' recurrent states copied between a pool's
	// halves so a step across sessions reads one half (alignRec): it moves
	// when the step's company changes, not while it keeps stepping together.
	RecAligns int
	// RecCarryBytes counts the bytes of sessions' recurrent state a pool's
	// resize carried into its new buffers (recResize), device to device: the
	// selection check that an arrival or a departure moved the other sessions'
	// states at all.
	RecCarryBytes uint64
	// RecZeroBytes counts the bytes a seat's zeroing copied on the device
	// from the zeros uploaded once (recZeroLayers): device copies that are
	// not a carry.
	RecZeroBytes, RecZeroCopies uint64
	// RecZeroSource counts the bytes of zeros those copies read, uploaded
	// once a zeroing.
	RecZeroSource uint64
	// RecZeroHost counts seat zeroings written from the host because the
	// kept zeros could not be had on the device (recZeroLayers).
	RecZeroHost int
	// PagedStagedLaunches counts decode attentions that ran the staged
	// kernels rather than FlashDecodeKV (kernels.ChooseDecodePlan).
	PagedStagedLaunches int
	// PagedPrefill70 counts those prefill launches on sm_70's m8n8k4 pair
	// (PagedAttnScoresMMA70/PagedAttnAccMMA70): the selection check that a
	// prompt on an sm_70 card took the tensor cores, latent blocks included.
	PagedPrefill70 int
	// PagedAccMMA counts prefill launches whose accumulate ran on the m16n8k8
	// binary16 instruction (kernels.PagedAttnAccMMA, sm_75 on): the selection
	// check that the weighted sum of V left the FMA tiles.
	PagedAccMMA int
	// FlashPrefill80 counts prompt attention launches of
	// kernels.FlashPrefill80, scores, softmax and accumulate in one kernel
	// on m16n8k16 (sm_80 on).
	FlashPrefill80 int
	// PagedPrefillPasses counts the passes those launches ran over a history
	// longer than one launch attends -- the selection check for the fold.
	PagedPrefillPasses int
	// EmbedLaunches counts prompt chunks whose rows the device gathered from
	// the tied head (EmbedRows) -- the selection check that the host lookup
	// was actually skipped.
	EmbedLaunches int
	// HeadFolds counts batched chunks that ran the output projection on
	// their last row inside their own submission.
	HeadFolds int
	// TowerChunked counts non-causal sets built row-chunked (chunkTower), and
	// TowerPhases the phase calls run over them (LayersPhase).
	TowerChunked, TowerPhases int
	// PickRows counts label picks the head ran as the picked rows' own
	// projection (pickMV) instead of the whole head and a gather.
	PickRows int
	// FlashPrefills counts batched attention launches of a one-kernel form
	// (kernels.FlashPrefill70, FlashPrefillTile), paged or not.
	FlashPrefills int
	// VisionAttnChunks counts the query chunks vision blocks ran their
	// attention in (visionattn.go) -- the selection check that a large image's
	// score planes were sized for a chunk.
	VisionAttnChunks int
	// MMAWhy is what the device said when it refused the matrix instruction.
	MMAWhy string
	// TileMV counts batched matvecs built as kernels.GemmTile (binary16 tiles
	// on Metal's simdgroup_matrix), and TileWhy is what the device said if it
	// refused them.
	TileMV  int
	TileWhy string
	// TileTail counts the GemmTile matvecs built as a wide block plus a tail
	// launch at a token offset.
	TileTail int
	// ExactPrompts counts prompt chunks run in a scratch of their own grain
	// width rather than a wider one they pad.
	ExactPrompts int
	// PromptWidthsWarmed counts the prompt grain widths whose kernels were
	// built at the end of a placement (prewarmPromptWidths).
	PromptWidthsWarmed int
	// SoftmaxLanes is 32 when the device passed pickSoftmax's on-device check
	// of the warp reduction, and 1 when it fell back to a thread per head. The
	// two produce slightly different floats (a tree sum against a sequential
	// one), so a token divergence wants to know which ran.
	SoftmaxLanes int
	// SoftmaxWhy says why the scalar kernel is running, because "the check
	// failed" and "not offered on this backend" are different facts.
	SoftmaxWhy string
	// Lanes is the subgroup width this device guarantees for the kernels whose
	// arithmetic needs one: 32, or 1 meaning "no promise, use the scalar
	// twins". LanesWhy is the device's own words for a refusal. It is the
	// promise, where SoftmaxLanes is what the softmax ran after its probe;
	// Lanes 32 with SoftmaxLanes 1 is a driver that failed its own promise.
	Lanes    int
	LanesWhy string
	TLayer   time.Duration
	// TEmit is the CPU time spent encoding launches, as opposed to waiting for
	// the device: the ceiling on what an indirect command buffer (which removes
	// only the encoding) could buy on a backend with no recording.
	TEmit                         time.Duration
	TStage, TLaunch, TRead, TConv time.Duration
	TQuant, TUp                   time.Duration
	Restaged                      int
	// Packs is PackWeights calls this device made outside the arena: every
	// pack when the arena is off and the ones it declined when it is on. Only
	// the sum with ArenaStats.Packs answers "did the pack run once"
	// (TestArenaPacksOncePerPageIn).
	Packs int

	// Prepacked is tensors that were already in the device layout (a converted
	// jlm container), and PrepackedBytes is what those uploads moved. Every
	// tensor upload is a pack, an unpack or a prepack; a converted model must
	// show Packs and Unpacks at zero and this at every tensor, since the ids
	// would be identical if something were still rebuilding the layout.
	Prepacked      int
	PrepackedBytes uint64

	// Unpacks is tensors uploaded raw and unpacked by a device kernel, and
	// UnpackBytes is the GGUF bytes those uploads moved. It is non-zero only
	// when a card really did the extraction; a tier that kept packing on the
	// host would otherwise look identical.
	Unpacks     int
	UnpackBytes uint64
	// NoUnpackKernel is tensors the host packed because their format has no
	// device unpack (or NoUnpack is set), and NoUnpackRoom is tensors it
	// packed because the staging would not hold them (the budget refused the
	// buffer, or the tensor is wider than any block's; see rawCap). One is a
	// kernel to write, the other a budget. UnpackWhy is the reason in the
	// kernel table's own words (kernels.UnpackWhyNot).
	NoUnpackKernel int
	NoUnpackRoom   int
	UnpackWhy      string
	// StageVRAM is what the raw staging buffer costs this device right now,
	// the number the slot arithmetic moves by.
	StageVRAM uint64
	// TUnpack is time inside the unpack launch, as opposed to TUpload's
	// transfer, so a slow page-in says which half it is.
	TUnpack time.Duration

	// Slots is how many blocks' weights this device can hold at once: the
	// paging capacity, not the placement (see page.go). Reported even when
	// nothing pages, since it tells "fits with room" from "pages through a few
	// slots".
	Slots int
	// PageIns is a block's weights uploaded into a slot after the load-time
	// one: a block that was paged out and came back. It is what a does-not-fit
	// gate asserts, since a tier that silently declined instead of paging
	// would otherwise look healthy.
	PageIns int
	// PageOuts is a block's weights freed to make room for another block; a
	// model that fits must read zero.
	PageOuts int
	// PageBytes is the device-resident bytes those page-ins brought back. It is
	// not what crossed the link: a covered format uploads raw GGUF bytes
	// (UnpackBytes) and unpacks on the card.
	PageBytes uint64
	// Imports is packed tensors this device wrapped instead of copying (a
	// unified heap). ImportBytes is what those wrappers would have cost had
	// they been uploaded; it must not be added to KVBytes or a pool, because
	// the arena already holds it.
	Imports     int
	ImportBytes uint64
	// Submits is how many Sessions this device opened for block work: one per
	// contiguous run of resident blocks, so one per token on a model that fits
	// and one per hole in the resident set when paging.
	Submits int
}

// add sums another device's counters into this one. Four kinds do not sum: a
// string takes the last non-empty value (a reason is a fact about one device);
// SoftmaxLanes and Lanes take the smallest (a tier is as weak as its weakest
// device); AttnMMA is an OR (one device on binary16 operands makes bit
// equality the wrong gate for the whole run).
func (s *Stats) add(o Stats) {
	if o.Device != "" {
		if s.Device == "" {
			s.Device = o.Device
		} else {
			s.Device += " + " + o.Device
		}
	}
	s.NoKernel += o.NoKernel
	if s.DeclineWhy == "" {
		s.DeclineWhy = o.DeclineWhy
	}
	s.Failed += o.Failed
	s.TooSmall += o.TooSmall
	s.Forgotten += o.Forgotten
	if o.LastErr != "" {
		s.LastErr = o.LastErr
	}
	s.Declined += o.Declined
	s.ConvBlocks += o.ConvBlocks
	s.NoRoom += o.NoRoom
	s.SessionDeclines += o.SessionDeclines
	s.Prepacked += o.Prepacked
	s.PrepackedBytes += o.PrepackedBytes
	s.KVBytes += o.KVBytes
	s.KVPoolBytes += o.KVPoolBytes
	s.BudgetUsed += o.BudgetUsed
	s.ScratchBytes += o.ScratchBytes
	s.Allocated += o.Allocated
	s.RoundingBytes += o.RoundingBytes
	s.ReservedPrompt = max(s.ReservedPrompt, o.ReservedPrompt)
	s.TowerRows = max(s.TowerRows, o.TowerRows)
	s.ReservedStep = max(s.ReservedStep, o.ReservedStep)
	s.ScratchRefused += o.ScratchRefused
	s.PromptSplits += o.PromptSplits
	s.RecBytes += o.RecBytes
	s.KVGrows += o.KVGrows
	s.Served += o.Served
	s.RowtTiles += o.RowtTiles
	s.RowtPlain += o.RowtPlain
	s.VoltaMV += o.VoltaMV
	s.SpillRefused += o.SpillRefused
	s.RagGroupMV += o.RagGroupMV
	s.GateUpFused += o.GateUpFused
	if o.GateUpWhy != "" {
		s.GateUpWhy = o.GateUpWhy
	}
	s.RagQKVFused += o.RagQKVFused
	s.RagResFused += o.RagResFused
	s.PersistLaunches += o.PersistLaunches
	s.PersistParts += o.PersistParts
	s.PersistPhases += o.PersistPhases
	s.PersistCompiles += o.PersistCompiles
	s.PersistFallbacks += o.PersistFallbacks
	s.PersistDepLaunches += o.PersistDepLaunches
	s.PersistDepTiles += o.PersistDepTiles
	if o.PersistWhy != "" {
		s.PersistWhy = o.PersistWhy
	}
	s.RagGateFused += o.RagGateFused
	s.GateUp2Runs += o.GateUp2Runs
	s.RagHeadOne += o.RagHeadOne
	s.SampleReads += o.SampleReads
	s.PickReads += o.PickReads
	s.SampleLaunches += o.SampleLaunches
	s.GroupedMoE += o.GroupedMoE
	s.GroupedFloat += o.GroupedFloat
	s.MLABatched += o.MLABatched
	s.IdxSelects += o.IdxSelects
	s.MSASelects += o.MSASelects
	s.QSASelects += o.QSASelects
	s.AltUpBlocks += o.AltUpBlocks
	s.DS4Blocks += o.DS4Blocks
	s.DS4HC += o.DS4HC
	s.K3Mixes += o.K3Mixes
	s.K3Latent += o.K3Latent
	s.Q4EMixes += o.Q4EMixes
	s.Q4ENGram += o.Q4ENGram
	s.LinearBatched += o.LinearBatched
	s.LinearFused += o.LinearFused
	s.RecShared += o.RecShared
	s.VoltaGemm += o.VoltaGemm
	s.GemmF16 += o.GemmF16
	s.GemmDeep += o.GemmDeep
	s.GemmInt8 += o.GemmInt8
	s.NormF16 += o.NormF16
	s.ActF16Fused += o.ActF16Fused
	s.QuantActs += o.QuantActs
	s.KVPrewarmPages += o.KVPrewarmPages
	s.GroupedVolta += o.GroupedVolta
	s.MLAScores70 += o.MLAScores70
	s.MLAAcc70 += o.MLAAcc70
	s.ActF16Launches += o.ActF16Launches
	s.QuantSkipped += o.QuantSkipped
	s.ResidualFused += o.ResidualFused
	s.ResidScaled += o.ResidScaled
	s.RowtSplitTiles += o.RowtSplitTiles
	s.IndexedGroup += o.IndexedGroup
	s.SegFused += o.SegFused
	s.MoEFused += o.MoEFused
	s.QKRms += o.QKRms
	s.RopeTables += o.RopeTables
	s.RopeTableUploads += o.RopeTableUploads
	if o.RopeTableWhy != "" {
		s.RopeTableWhy = o.RopeTableWhy
	}
	s.StreamFills += o.StreamFills
	s.StreamOverlaps += o.StreamOverlaps
	s.StreamBlocks += o.StreamBlocks
	s.Blocks += o.Blocks
	s.TPack += o.TPack
	s.TSubStage += o.TSubStage
	s.TSubLaunch += o.TSubLaunch
	s.TSubRead += o.TSubRead
	s.TCapture += o.TCapture
	s.TUpload += o.TUpload
	s.TStreamWait += o.TStreamWait
	s.TStreamRead += o.TStreamRead
	s.TStreamPut += o.TStreamPut
	s.TStreamSheet += o.TStreamSheet
	s.TStreamCopy += o.TStreamCopy
	s.TStreamH2D += o.TStreamH2D
	s.StreamBytes += o.StreamBytes
	s.StreamDirect += o.StreamDirect
	s.StreamGathered += o.StreamGathered
	s.StreamPinned += o.StreamPinned
	s.StreamCacheHits += o.StreamCacheHits
	s.StreamCacheMisses += o.StreamCacheMisses
	s.StreamCacheShort += o.StreamCacheShort
	s.HybridRuns += o.HybridRuns
	s.HybridRows += o.HybridRows
	s.THybrid += o.THybrid
	s.StreamCacheSize = max(s.StreamCacheSize, o.StreamCacheSize)
	s.StreamGroupsTuned = max(s.StreamGroupsTuned, o.StreamGroupsTuned)
	s.StreamPinHalfTuned = max(s.StreamPinHalfTuned, o.StreamPinHalfTuned)
	s.StreamDirectTuned = max(s.StreamDirectTuned, o.StreamDirectTuned)
	s.AutoMeanBase = max(s.AutoMeanBase, o.AutoMeanBase)
	s.StreamPrefetched += o.StreamPrefetched
	s.TStreamPrefetchWait += o.TStreamPrefetchWait
	s.ProbeHits += o.ProbeHits
	s.ProbeExperts += o.ProbeExperts
	s.ProbeFused += o.ProbeFused
	s.TPrewarm += o.TPrewarm
	s.Captures += o.Captures
	s.AttnMMA = s.AttnMMA || o.AttnMMA
	s.FlashLaunches += o.FlashLaunches
	s.FusedLaunches += o.FusedLaunches
	s.ActInLaunches += o.ActInLaunches
	s.ActInSaved += o.ActInSaved
	s.ActInDeclined += o.ActInDeclined
	s.ActInTimed += o.ActInTimed
	s.ActInKept += o.ActInKept
	s.PagedLaunches += o.PagedLaunches
	s.PagedPrefillLaunches += o.PagedPrefillLaunches
	s.PagedStagedLaunches += o.PagedStagedLaunches
	s.KVEvictions += o.KVEvictions
	s.KVStageReads += o.KVStageReads
	s.KVWindowReleased += o.KVWindowReleased
	s.KVStreamPasses += o.KVStreamPasses
	s.StagedPasses += o.StagedPasses
	s.PipelinePieces += o.PipelinePieces
	s.SubsBeside += o.SubsBeside
	s.RanBeside += o.RanBeside
	s.LanesWaited += o.LanesWaited
	s.SessionRows += o.SessionRows
	s.SessionLinear += o.SessionLinear
	s.LinearRowsScalar += o.LinearRowsScalar
	s.RecChainRows += o.RecChainRows
	s.RowsLatent += o.RowsLatent
	s.RecAligns += o.RecAligns
	s.RecCarryBytes += o.RecCarryBytes
	s.RecZeroBytes += o.RecZeroBytes
	s.RecZeroCopies += o.RecZeroCopies
	s.RecZeroSource += o.RecZeroSource
	s.RecZeroHost += o.RecZeroHost
	s.PagedPrefill70 += o.PagedPrefill70
	s.PagedAccMMA += o.PagedAccMMA
	s.FlashPrefill80 += o.FlashPrefill80
	s.PagedPrefillPasses += o.PagedPrefillPasses
	s.EmbedLaunches += o.EmbedLaunches
	s.HeadFolds += o.HeadFolds
	s.TowerChunked += o.TowerChunked
	s.TowerPhases += o.TowerPhases
	s.PickRows += o.PickRows
	s.FlashPrefills += o.FlashPrefills
	s.VisionAttnChunks += o.VisionAttnChunks
	if o.MMAWhy != "" {
		s.MMAWhy = o.MMAWhy
	}
	s.TileMV += o.TileMV
	s.TileTail += o.TileTail
	s.ExactPrompts += o.ExactPrompts
	s.PromptWidthsWarmed += o.PromptWidthsWarmed
	if o.TileWhy != "" {
		s.TileWhy = o.TileWhy
	}
	if o.SoftmaxLanes > 0 && (s.SoftmaxLanes == 0 || o.SoftmaxLanes < s.SoftmaxLanes) {
		s.SoftmaxLanes = o.SoftmaxLanes
	}
	if o.SoftmaxWhy != "" {
		s.SoftmaxWhy = o.SoftmaxWhy
	}
	if o.Lanes > 0 && (s.Lanes == 0 || o.Lanes < s.Lanes) {
		s.Lanes = o.Lanes
	}
	if o.LanesWhy != "" {
		s.LanesWhy = o.LanesWhy
	}
	s.TLayer += o.TLayer
	s.TEmit += o.TEmit
	s.TStage += o.TStage
	s.TLaunch += o.TLaunch
	s.TRead += o.TRead
	s.TConv += o.TConv
	s.TQuant += o.TQuant
	s.TUp += o.TUp
	s.Restaged += o.Restaged
	s.Packs += o.Packs
	// Slots sums: two devices with three slots each hold six blocks between
	// them. DevStats still says where they are.
	s.Slots += o.Slots
	s.Submits += o.Submits
	s.PageIns += o.PageIns
	s.PageOuts += o.PageOuts
	s.PageBytes += o.PageBytes
	s.Imports += o.Imports
	s.ImportBytes += o.ImportBytes
	s.Unpacks += o.Unpacks
	s.UnpackBytes += o.UnpackBytes
	s.NoUnpackKernel += o.NoUnpackKernel
	s.NoUnpackRoom += o.NoUnpackRoom
	// StageVRAM sums: each device holds its own staging buffer.
	s.StageVRAM += o.StageVRAM
	s.TUnpack += o.TUnpack
	if o.UnpackWhy != "" {
		s.UnpackWhy = o.UnpackWhy
	}
}

// Pool is a weight budget that every device spending the same bytes shares.
//
// An integrated GPU's heap is system memory, so a budget of its own beside the
// host's would spend the same bytes twice; a discrete card's VRAM is genuinely
// separate. So a device does not own a budget, it maps to one:
//
//	pool "host"   host RAM   <- the iGPU's weights and KV, and the host's own
//	pool "sm_86"  4 GiB VRAM <- the discrete card's weights and KV
//	(on Apple Silicon there is exactly one pool and everything is in it)
//
// backend.Device.UnifiedMemory() says which pool a device belongs to. The iGPU
// is not refused: its execution units are real, and it is charged to the host
// pool.
//
// A device also keeps its own limit, and both are checked: the pool stops two
// devices double-spending one heap, the per-device limit is what -vram means
// for one card.
type Pool struct {
	mu    sync.Mutex
	name  string
	used  uint64
	limit uint64
	// host marks the pool whose bytes are the host's own memory; its used
	// bytes are what GPU.HostReserved reads, and so what the host's weight
	// budget is reduced by.
	host bool
}

// NewPool is a named budget. limit 0 means unbounded, which is what a caller
// that wants the per-device limits to be the only gate passes.
func NewPool(name string, limit uint64) *Pool { return &Pool{name: name, limit: limit} }

// NewHostPool is the budget for devices whose memory is the host's: an
// integrated GPU, a CPU rasteriser, every device on Apple Silicon. limit is the
// share of the host's own budget those devices may hold between them; the host
// takes off its own only what they hold (see GPU.HostReserved).
func NewHostPool(limit uint64) *Pool {
	return &Pool{name: hostPoolName, limit: limit, host: true}
}

func (p *Pool) Name() string { return p.name }

// Host reports that this pool's bytes come out of host RAM.
func (p *Pool) Host() bool { return p != nil && p.host }

func (p *Pool) Used() uint64 { p.mu.Lock(); defer p.mu.Unlock(); return p.used }

func (p *Pool) Limit() uint64 { return p.limit }

// SetLimit retargets the heap this pool stands for, at runtime. It does not
// evict: a pool is an account, not an owner, and only a device knows which of
// its blocks can be given back. GPU.SetBudget lowers the limits and then asks
// each device to trim.
func (p *Pool) SetLimit(n uint64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.limit = n
}

// room reports whether n more bytes would fit, without taking them. PrepLayer
// prices a whole block before uploading any of it (see blockBytes), so the
// check and the charge are separate operations.
func (p *Pool) room(n uint64) bool {
	if p == nil || p.limit == 0 {
		return true
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.used+n <= p.limit
}

func (p *Pool) take(n uint64) {
	if p == nil {
		return
	}
	p.mu.Lock()
	p.used += n
	p.mu.Unlock()
}

func (p *Pool) give(n uint64) {
	if p == nil {
		return
	}
	p.mu.Lock()
	if p.used >= n {
		p.used -= n
	} else {
		p.used = 0
	}
	p.mu.Unlock()
}

// room reports whether n more bytes of weights would fit: in this device's own
// budget and in the pool its memory comes from (see Pool). Callers hold the
// device lock; the pool takes its own.
func (g *devTier) room(n uint64) bool {
	// A buffer freed since the last charge gave its rounding back to the
	// card, not yet to the budget: settled first, or a reclaim that pages
	// blocks out never sees the room it made.
	g.settleRounding()
	n += g.reserved()
	return g.used+n <= g.limit && g.pool.room(n)
}

// roomSeat is room for a session's history on a block that has live sessions
// already: a free seat of Config.Sessions was reserved for it (see reserved),
// so its own share of the reservation is credited back. Callers hold g.mu.
func (g *devTier) roomSeat(need uint64, live int, hist uint64) bool {
	if g.Sessions > 1 && live < g.Sessions {
		res := g.reserved()
		credit := min(hist, res)
		return g.used+res-credit+need <= g.limit && g.pool.room(res-credit+need)
	}
	return g.room(need)
}

// charge records n bytes of resident weights against both accounts.
func (g *devTier) charge(n uint64) {
	g.used += n
	g.explicit += int64(n)
	g.pool.take(n)
	g.settleRounding()
}

// refund gives n bytes back to both accounts, which ReleaseLayers must do or
// the card leaks budget on every migration.
func (g *devTier) refund(n uint64) {
	if n > 0 {
		g.roomGen.Add(1)
	}
	// A refund past zero is a bookkeeping bug; clamping it keeps the two
	// accounts consistent rather than inventing budget out of one of them.
	if n > g.used {
		n = g.used
	}
	g.used -= n
	g.explicit -= int64(n)
	g.pool.give(n)
	g.settleRounding()
}

// devShared is what one device's tier holds for every session: everything
// that is a device address, an account of device memory, or a cache keyed on
// either -- the blocks and their residency, the compiled kernels, the paged
// attention history and the recurrent pools, the budget. What one session's
// calls write between and during their submissions is its devSess, and the
// scratch they run in is a lane (devsess.go); a devTier is the three seen
// together.
type devShared struct {
	// onDev is how many submissions are inside the backend's session on this
	// device right now, and ranBeside how many began while another was
	// (Stats.RanBeside); see runSub.
	onDev     atomic.Int32
	ranBeside atomic.Int64
	// layerGen counts blocks placed and released, so work that depends only
	// on which blocks are here can tell it is still current (prepBatch).
	layerGen uint64
	// tabPend are page-table writes waiting for flushTabs (kvpool.go).
	tabPend []tabWrite
	// autoStream holds the blocks placed with their routed experts off the
	// card, and where those experts run: GPU.AutoStream's mark (expAuto) or a
	// placement's (GPU.PlaceExperts). Such a block is placed streamed whatever
	// StreamExperts says, and given its expert cache after placement
	// (sizeAutoCaches). It is set on every device, so a block that moves
	// keeps it. Under mu.
	autoStream map[int]expertMode
	// hostFns is each session's host side for its hybrid blocks, by block
	// (hostFor); a session's go when it detaches. Under mu.
	hostFns map[uint64]map[int]hostSide
	// ftune measures the streamed fill's knobs (streamtune.go).
	ftune *fillTune
	// sheetStage is the streamed fill's gather buffer per (matrix, plane);
	// see streamStage.
	sheetStage [9][]byte
	// pin is the streamed fill's two page-locked halves, nil until the first
	// direct sheet or on a device with no page-locked memory; pieces is
	// put's list of direct sheets, kept so a token allocates none. See
	// sendPieces.
	pin     [2][]byte
	noPin   bool
	pieces  []sheetPiece
	packWG  sync.WaitGroup
	copyWG  sync.WaitGroup
	packEnd int
	*Config
	mu sync.Mutex
	// tot is the device's counters (Stats). A view names them through its
	// Stats pointer; see devTier.
	tot Stats
	// sess is every session's state on this device, the zero session's
	// included, made at its first call (sessOf). lane0 is the scratch
	// placement builds and lanes its clones, one more for each call that
	// found every lane taken (takeLane); laneGen moves whenever lane0 changes
	// shape, retiring the clones of the old one (devsess.go). dv is the
	// device's own view: lane0's and the zero session's. Under mu.
	sess    map[uint64]*devSess
	lane0   *lane
	lanes   []*lane
	laneGen uint64
	dv      *devTier
	// noClone says the last clone did not fit, at roomGen noCloneRoom and
	// laneGen noCloneGen: until one of them moves, a call with no free lane
	// waits for one rather than building another (takeLane).
	noClone                 bool
	noCloneRoom, noCloneGen uint64
	// laneNext and laneServe are the line of calls waiting for a lane: the
	// next ticket to give out and the one served next (takeLane).
	laneNext, laneServe uint64
	// lane0Want counts calls waiting to borrow lane0 for a tower's range
	// (borrowLane0).
	lane0Want int
	// clk, tickets, quiet and idle are the submissions in flight outside mu
	// and the waits on them (inflight.go). idle's lock is mu.
	clk     subClock
	tickets []uint64
	quiet   int
	idle    sync.Cond
	// tkv is the k/v pair every non-causal block shares and tkvRefs how many
	// hold it (transientkv.go). Guarded by mu.
	tkv     *kvPair
	tkvRefs int
	// subCost is the per-block run time of a submission by its row count's
	// class (subbudget.go). Guarded by mu.
	subCost map[int]time.Duration
	// visCap is the rows the non-causal set is built for and visMax the
	// largest grid a tower placed here takes (visrows.go). Guarded by mu.
	visCap, visMax int
	// visKeys is the keys the set's k/v pairs and score rows hold when it is
	// row-chunked (LayersPhase): a picture of visKeys rows runs visCap rows a
	// call. 0 when the set takes a picture whole (visKeys is visCap).
	visKeys int
	// visSmall counts the calls in a row the set was more than twice what
	// they needed (ReserveRows), which shrinks it at visShrinkCalls.
	visSmall int
	// scratch is the budget the device's own buffers hold -- the block
	// scratches and the staging -- and explicit the running sum of what
	// charge and refund moved, which a scratch window subtracts so a weight or
	// a page charged inside it is not counted twice. winDepth nests windows
	// (scratch.go).
	scratch  uint64
	explicit int64
	// cnt is dev's allocation count, nil where it keeps none: asserted once
	// here, because an interface assertion on the step's path allocated its
	// type cache inside a warm decode (model.TestDecodeDoesNotAllocate).
	cnt backend.AllocCounter
	// rounding is the driver's page rounding charged so far (settleRounding).
	rounding uint64
	// kvPromise is the positions of history a scheduler has admitted rows
	// for and they have not yet taken (GPU.PromiseKV, kvroom.go).
	kvPromise int
	// kvTight says the driver refused the KV pool a growth the budget
	// allowed: the card holds less than the budget says (other processes on
	// it, or a budget set past what was free), so KVRoom counts growth only
	// up to what the driver reports free, asked each time, and a scheduler
	// holds rows back rather than admitting history the card cannot take.
	// A growth that succeeds, or a new budget, clears it.
	kvTight  bool
	winDepth int
	// geoUsed says a split plan was ever placed here (gemma4.go).
	geoUsed bool
	// reserving is set while reserveScratch probes widths, so prepBatch's
	// refusal of a width over the budget is not counted as a prompt's.
	reserving bool
	// closed guards Close: a second teardown would send on the nil channel
	// Close leaves behind and block forever. See Close.
	closed bool
	// forgets are address ranges Forget could not apply because g.mu was
	// held, possibly by the very call whose page-in asked (forget.go); every
	// lookup by pointer applies them first. forgetPend says there are any,
	// so that lookup costs one atomic load when there are none.
	forgetMu   sync.Mutex
	forgets    [][2]uintptr
	forgetPend atomic.Bool
	// packing is every prewarm pack running outside g.mu, by its source
	// address, so a forget covering it spoils it. Guarded by mu.
	packing []*packing
	// recSeats is each session's run of slots in every linear block's
	// recurrent pool (recpool.go), recSlots the slots every pool holds, and
	// recShared the pools' form. recS and recC are one slot of the delta
	// state and of the convolution's window, in floats.
	recSeats   map[uint64]recSeat
	recSlots   int
	recShared  bool
	recS, recC int
	// recNext is the one "next state" buffer the shared form writes, sized
	// for the widest step that needed it; recCopies holds the CopySlots
	// kernels that move it home, by slot width and sequence count.
	recNext      backend.Buf
	recNextBytes uint64
	recCopies    map[[2]int]backend.Kernel
	// recZeros is the zeros a seat's zeroing copies from (recZeroLayers),
	// uploaded at the widest zeroing and kept while there are pools: a
	// prompt that resets its sequence's state made and freed it each time.
	recZeros      backend.Buf
	recZerosBytes uint64
	// recLs is resetRecRows' list of the blocks it zeroes, kept between
	// calls.
	recLs []*layer
	// headSrc is the first byte of the projection the resident head was
	// uploaded from, and headNormHost the norm it was built with: what a
	// second State's PrepHead is compared against (adoptHead). altNorms are
	// the other norms that head serves (headnorm.go).
	headSrc      *byte
	headNormHost []float32
	altNorms     []altNorm
	// busy is the device's call lock: a session's step takes it shared, so
	// steps of several sessions run at once, and a call that changes what
	// every step reads (placement, a release, a migration) takes it
	// exclusively (gpuSession.enter). See docs/design/device-sessions.md.
	busy sync.RWMutex
	// convTo is the convolutional blocks' launch target (convOnce), the
	// device's and not a session's: their calls are one at a time
	// (GPU.convMu).
	convTo launchTo
	// convCall is the convolutional submission's arguments, and convFn its
	// method value, made once (conv.go).
	convCall convCallArgs
	convFn   func(backend.Session)
	// roomGen moves every time this device may have gained room: a refund, or
	// a budget raised. A caller that gave blocks up reads it to know when
	// asking for them back could succeed. See GPU.RoomGen.
	roomGen atomic.Uint64
	dev     backend.Device
	pool    *Pool
	// ord is this device's place in the tier's order, fastest first, and part
	// of its label.
	ord   int
	kerns map[kernKey]backend.Kernel
	// idKerns is the indexed matvecs, by their whole shape; see idKernel.
	idKerns map[kernels.MatVecShape]backend.Kernel
	res     map[resKey]*resident
	splits  map[splitKey]int
	// gemmDeep is tuneGemm's verdict per prompt GEMM shape: true where the
	// deep m16n8 tiles (f16DeepTiles) measured faster than f16Tiles' pick.
	gemmDeep map[splitKey]bool
	// groupSplit records, per shape, whether tuneSplit found the in-group
	// reduction faster than the partial-buffer one.
	groupSplit map[splitKey]bool
	// actWin is the activation amax window the CPU tier chose; see
	// nn.JIT.SetDevice. Zero means the default 32. It is copied per device
	// rather than kept in Config because it is written after Open and read on
	// the serving path, so it lives under the lock that reads it.
	actWin int
	// recCap caches whether sessions can capture: 0 unknown, 1 yes, -1 no.
	recCap int
	// mvc is matVec's call (mvCall), run on the device's view under mu; mvTo
	// its launch target and mvPart its partial sums. They are matVec's own and
	// not a lane's: a matvec a host block offers runs while a session's
	// submission holds whichever lane it took.
	mvc       mvCall
	mvTo      launchTo
	mvPart    backend.Buf
	mvPartCap int
	used      uint64
	limit     uint64
	// why is where limit came from, in words, for the report.
	why string

	// Per-call staging, grown as needed and reused: a decode matvec is issued
	// ~127 times per token, and allocating device memory each time would cost
	// more than the kernel.
	aBuf, axBuf, outBuf backend.Buf
	aCap, axCap, outCap int
	xfBuf               backend.Buf // a float weight's activation
	xfCap               int
	reduce              map[[2]int]backend.Kernel
	retunedAt           int // len(layers) at the last retuneDecode
	// The background split tuner (WithLazyTune): its queue, whether its
	// goroutine runs, Close's stop, and the goroutine to wait for.
	tuneQ       []tuneJob
	tuneRunning bool
	tuneStop    bool
	tuneWG      sync.WaitGroup
	// lastSub is when the device last started or finished a submission
	// (unix ns): the background tuner waits for tuneQuiet past it.
	lastSub   atomic.Int64
	restrides map[[4]int]backend.Kernel // kernels.Restride by shape; see copyKVAcrossStride
	segKerns  map[string]backend.Kernel // kernels.MatVecSegments by shapes; see fuseQKV
	// actInWin is tuneActIn's verdict per operand-prologue site key: true where
	// the prologue measured faster than the two launches on this card.
	// actInTrial makes fuseActIn build every candidate for that measurement.
	actInWin   map[string]bool
	actInTrial bool
	ragKerns   map[moeKernKey]backend.Kernel // a batched mixture's grouped matvecs (moegroup.go)
	moeVolta   map[moeVoltaKey]moeVoltaMV    // their tensor-core twins; see voltaGroupedMV
	ragK       map[ragKey]backend.Kernel     // a ragged step's matvecs; see ragMV
	kshared    map[string]backend.Kernel     // scratch kernels by content; see compileShared
	pcache     *persistCache                 // composed persistent kernels; see persist.go
	// convK, conv and convN are the convolutional blocks' kernels, their
	// shared scratch and how many hold it (conv.go).
	convK map[convKey]backend.Kernel
	conv  convScratch
	// dh is Clef's decision head on this device (dechead.go), and ClefHeads
	// how many requests it has answered.
	dh        dhState
	ClefHeads int64
	convN     int
	ragSeg    map[segShapeKey]segLaunch // a few-sequence step's q/k/v; see groupQKV
	argmaxK   backend.Kernel            // kernels.Argmax over the head's rows; see PrepHead
	// samplePenK and sampleKs are the device sampler's kernels: the penalty
	// over the head's rows and, per k, the two top-k passes (sample.go).
	samplePenK backend.Kernel
	sampleKs   map[int]sampleKerns
	// pickK is the label pick's gather over the head's rows (pick.go).
	pickK backend.Kernel
	// pickMV projects the picked rows of the head alone (kernels.MatVec's
	// PickOf), in place of the head's whole matvec and pickK's gather; nil
	// where the head's format or a cap needs the whole row.
	pickMV backend.Kernel
	hostA  []uint32
	hostAX []float32

	// Host-side scratch, reused rather than allocated per matvec. It is per
	// device even though it is host memory, because lastX/staged say "the
	// activation already sitting in this device's aBuf".
	hostRaw []byte
	hostOut []float32 // a view over hostRaw, for the readback
	lastX   []float32 // the activation vector already staged, for reuse
	staged  bool

	// mv accumulates the sample that decides whether serving matvecs one at a
	// time pays on this device, per session, so the verdict is a property of
	// the model rather than of how much the tier has seen since it opened. The
	// threshold is shared (Config.mvMin). See mvPays and
	// model.TestTheSameWorkCrossesTheSeamEverySession.
	mv map[uint64]*mvState

	bad map[string]int // Verify's disagreeing shapes

	stage      map[resKey]*staged
	stageBytes uint64
	// calls is the fault-injection counter; always zero unless built with
	// -tags jitllmfault.
	calls int

	// The whole-block path: layers uploaded by PrepLayer, run in the scratch
	// of a lane (devsess.go), which a call holds for its whole run. The map is keyed by the
	// model's block index and holds only this device's blocks, so a run on this
	// card is exactly this card's Layers(lo, hi).
	layers map[int]*layer
	// pinned blocks are never the pager's victim, and a budget below what they
	// charge is refused (GPU.Pin, GPU.SetBudget). Guarded by mu.
	pinned map[int]bool
	// stream holds the blocks whose weights stream through this device's
	// slots rather than staying resident (GPU.Stream): a streamed block that
	// does not fit is admitted by paging other streamed blocks out, and only
	// streamed blocks are the pager's victims when it makes room. A block not
	// in it is resident here or declined to the host. Where a block runs is
	// the placement's choice (model.Place.Stream), not a device-wide switch.
	// Guarded by mu.
	stream map[int]bool
	// name is Slot.Name.
	name string
	// The pager's state. A block's weights live in a slot and may leave; its KV
	// cache, norms and biases may not, having no backing store. See page.go.
	widest uint64 // the widest block's pageable bytes seen on this device
	// imp is this device's zero-copy path, set at New only when the device both
	// offers one and has a unified heap. See importing().
	imp       backend.HostImport
	impAlign  uint64
	pageBytes uint64 // pageable bytes charged right now, their rounding included: the slots in use
	pagesIn   int    // blocks whose weights are resident right now
	// The device unpack's staging: one buffer holding the widest covered
	// tensor's raw GGUF bytes while the kernel reads them, charged like any
	// other non-pageable allocation. See unpack.go.
	rawBuf   backend.Buf
	rawBytes uint64
	// rawCap is the largest staging this device will hold: the widest covered
	// tensor of a block it has been offered. A wider tensor (the output
	// projection, a lone matvec) is packed on the host instead, since it is
	// uploaded once and never paged.
	rawCap uint64
	unks   map[unpackKey]backend.Kernel
	// kvCap is the scratch's position capacity: what every kernel in g.bs
	// bakes as its stride and score-row width. Every session shares the one
	// scratch, so it grows toward the longest context any session asked for
	// (maxSeqAsked); the histories themselves are paged and are each
	// session's own.
	//
	// It starts at one KV page and doubles, so a session costs the positions
	// it reached rather than the full context the caller asked for.
	kvCap int
	// maxSeqAsked is the longest context any session asked for: the ceiling
	// kvCap grows toward and never past.
	maxSeqAsked int
	// paged is set by the first text plan: every attention history is
	// paged (pagedPlan). kvp is the pool
	// every session's history lives in, and kvCap is the whole context,
	// because no kernel bakes a capacity.
	paged bool
	kvp   *kvPool
	// mmaOff records that this device has no integer matrix instruction, so the
	// first refusal is the last: it is a property of the backend, not of a
	// shape.
	mmaOff bool
	// tileOff records that this device refused kernels.GemmTile; see tileMV.
	tileOff bool
	// tileCap is the collective-tile probe (tilesOK): 0 not yet asked, 1 the
	// device lowers the 8x8 f16 tile GemmTile and FlashPrefillTile multiply
	// in, -1 it does not.
	tileCap int8
	// tileTallCap is tileTall's probe, in tileCap's terms.
	tileTallCap int8
	// tileF32Cap is tileF32's probe, in tileCap's terms.
	tileF32Cap int8
	// voltaMoE is the grouped tensor-core mixture's probe (voltaOn): 0 not yet
	// asked, 1 this device runs it, -1 it does not.
	voltaMoE int8
	// f16K is f16GemmK's probe: 0 not yet asked, -1 no m16n8 binary16
	// instruction, else its k.
	f16K int
	// i8Gemm is int8GemmOn's probe: 0 not yet asked, 1 the device lowers
	// GemmInt8, -1 it does not.
	i8Gemm int8
}

// GPU serves decode matvecs and whole blocks from one or more devices. It is a
// router: the arithmetic, the residency and the submissions belong to devTier,
// and what lives here is the block-to-device map, the order devices are asked
// in, and the rule that keeps a device's blocks contiguous. Nothing here knows
// what a transformer is beyond one repeating unit; the seam is still PrepLayer
// declining per block, and paging only makes a decline non-permanent.
type GPU struct {
	*Config
	// mu guards the routing (own, cur, head) and nothing else. A device's own
	// state is under that device's lock, and GPU methods release mu before
	// calling into a device, so a Prewarm is not queued behind an upload.
	mu sync.Mutex
	// place serialises a whole offer sequence, which mu (taken per PrepLayer)
	// cannot. See gpuSession.BeginPlacement.
	place sync.Mutex
	// model is the model whose blocks this tier holds (nn.ModelOwner), and
	// held says the tier has taken one at all. ownMu is held across a whole
	// offer, so two models cannot both find the tier free. See owner.go.
	ownMu sync.Mutex
	model uint64
	held  bool
	// devs is every device, fastest first; see order().
	devs []*devTier
	// spreadAll steers a plan over every device rather than the fewest that
	// hold it: streamed blocks, whose caches take what the bases leave
	// (planShares). Under mu.
	spreadAll bool
	// streamPlanned says the streamed plan was priced, and streamSeen counts
	// streamed blocks offered before it was.
	streamPlanned bool
	streamSeen    int
	// own maps a block index to the device holding it. A block with no entry is
	// on the host.
	own map[int]*devTier
	// staged is a block StageBlock prepared on a device ahead of a move's
	// commit, not yet routed there (move.go).
	staged map[int]*devTier
	// stagedHead is the device StageHead prepared the projection on.
	stagedHead *devTier
	// retire is what a move left on its source -- a block, or a projection
	// (block -1) -- freed after the route switched (RetireMoved), and
	// retiring the frees running; every placement call waits for them
	// (settle), so none re-creates a block a free is about to take.
	retire   []retiree
	retiring sync.WaitGroup
	// geo is a block's scratch set (gemma4.go: Gemma 4's attention geometry
	// and FFN width), so a range is cut where it changes as it is cut where
	// the device does (runsInto). Set with own; a block's never changes.
	geo map[int]geoKey
	// split says a block of a split-geometry model was placed: every run then
	// carries one rotary table (run.tables).
	split bool
	// cur is the index into devs of the device currently taking blocks. It only
	// moves forward while a placement grows, which keeps each device's blocks
	// one contiguous run; ReleaseLayers recomputes it.
	cur int
	// head is the device holding the output projection, normally the one that
	// holds the last block, so the projection rides that submission.
	head *devTier
	// runBuf is layersCall's runs slice between calls; see layersCall.
	runBuf []run
	// convTmp is ConvLayers' activation between two devices' runs, and convMu
	// holds a ConvLayers call whole: the activation stays on each card between
	// its submissions, in the scratch every convolutional block there shares
	// (conv.go), so two sessions' calls go one after the other.
	convTmp [2][]float32
	// convRuns is ConvLayers' runs, reused under convMu.
	convRuns []run
	convMu   sync.Mutex
	// rowsErr is why the last LayersRows failed, naming the device: the
	// devices' LastErr summed by Stats is whichever wrote last, not the one
	// that refused.
	rowsErr string
	// refused is the device whose last ReserveKV said no, or nil. See Refused.
	refused *devTier
	// spillFrom, when set, sends the next PrepLayer of block spillLi to a
	// device after spillFrom, and frees spillFrom's copy once one takes it.
	// See SpillAfter.
	spillFrom *devTier
	spillLi   int
	// planN is how many blocks the caller said it is about to offer
	// (PlanBlocks), and share, once the first block has priced one, is the
	// device index each block is steered to -- nil while there is no plan or
	// the model does not fit. See PlanBlocks.
	planN int
	share []int
	// planExtra is what the placement leaves room for besides the blocks'
	// weights (PlanBlocks), and planBase each device's use when the plan
	// began, so a fixed cost is measured against it and a device already
	// holding another model's weights is counted for what it has left.
	planExtra uint64
	planBase  []uint64
	// pools is every distinct budget, for reporting. Two devices on one heap
	// share one entry.
	pools []*Pool
	// hostPool is the one of those whose bytes are the host's, or nil when no
	// device is on host memory. HostReserved reads it.
	hostPool *Pool
}

// Slot is one device offered to the tier, and what it may spend.
type Slot struct {
	// Dev is opened by the caller; the tier takes ownership and Close closes it.
	Dev backend.Device
	// Bytes caps this device's resident weights. Zero means defaultBudget.
	Bytes uint64
	// Pool is the heap Bytes are spent from. Nil gives the device a pool of its
	// own, sized Bytes -- which is right for a discrete card and wrong for an
	// integrated one; see Pool and planSlots.
	Pool *Pool
	// Why is where Bytes came from, in words, for GPU.Budgets to report.
	Why string
	// Name is what a placement calls this device ("cuda:0", "vulkan:1",
	// "metal", "gpu:0"): the -devices entry that opened it. Empty for a device
	// opened by all or auto, which a placement cannot name.
	Name string
}

// defaultBudget is the budget when neither the caller nor the device says:
// conservative, because overshooting VRAM does not degrade, it fails.
const defaultBudget = 2 << 30

// Open picks one device (choose times a real matvec on each backend) and
// returns the tier. limitBytes caps weight residency; zero asks the device how
// much it has free (budgetFor).
//
// It is OpenWith(WithDevices("auto"), WithBudget(limitBytes)) with the parsing
// skipped.
func Open(limitBytes uint64) (*GPU, error) {
	devs := backend.Open()
	if len(devs) == 0 {
		return nil, fmt.Errorf("tier: no GPU backend on this host")
	}
	return New(planSlots([]backend.Device{choose(devs, knobs{})}, nil, openOpts{budget: limitBytes}))
}

// poolsFor turns a device list into slots with a fixed per-device budget,
// mapping every device whose memory is the host's onto one shared pool: it is
// planSlots with the budget already decided, no probing of free memory.
func poolsFor(devs []backend.Device, bytes uint64) []Slot {
	if bytes == 0 {
		bytes = defaultBudget
	}
	return planSlots(devs, nil, openOpts{budget: bytes})
}

// New builds a tier over the given devices, in the order given, and takes
// ownership of every one of them: the error path closes all of them, including
// the ones it never got to.
func New(slots []Slot, options ...Option) (*GPU, error) {
	// A direct New gets the same option set OpenWith parses; OpenWith passes
	// its own through, so applying twice is applying the same values.
	var o openOpts
	for _, f := range options {
		f(&o)
	}
	o.apply()
	cfg := &Config{
		kb:     o.kb,
		tiles:  tilesFor(o.kb),
		center: o.kern,
		mvMin:  mvMinBytes(o.kb),
		failAt: faultAt(),
		stage:  NewPool("host stage", 0),
		// Always built, budgeted at zero: the structure exists from load, and
		// the budget decides whether anything is retained. At zero (the
		// default) Pack declines immediately.
		arena: kernels.NewArena(o.kb.arena),
	}
	// A jitllmtest build can make the persistent path every tier's default
	// (PersistByDefaultForTest), so the regression gates run through it.
	cfg.Persist = persistDefault
	cfg.PersistDeps = persistDepsDefault
	// Gate2 is the default where it is exact (kernels.Gate2Exact): on a V100
	// it read 1.008-1.028 of the two launches on Llama 1B, 3B and 8B.
	cfg.GateUp2 = true
	for _, f := range o.cfg {
		f(cfg)
	}
	closeAll := func() {
		for _, s := range slots {
			if s.Dev != nil {
				s.Dev.Close()
			}
		}
	}
	if len(slots) == 0 {
		return nil, fmt.Errorf("tier: no devices")
	}
	g := &GPU{Config: cfg, own: map[int]*devTier{}, geo: map[int]geoKey{}}
	for i, s := range slots {
		if s.Dev == nil {
			closeAll()
			return nil, fmt.Errorf("tier: device %d is nil", i)
		}
		limit := s.Bytes
		if limit == 0 {
			limit = defaultBudget
		}
		pool := s.Pool
		if pool == nil {
			pool = NewPool(s.Dev.Name(), limit)
		}
		why := s.Why
		if why == "" {
			why = "the caller set it"
		}
		cnt, _ := s.Dev.(backend.AllocCounter)
		g.devs = append(g.devs, newDevice(&devShared{
			Config:     cfg,
			tot:        Stats{Device: label(s.Dev, i)},
			dev:        s.Dev,
			cnt:        cnt,
			pool:       pool,
			why:        why,
			ord:        i,
			name:       DeviceName(s.Name),
			kerns:      map[kernKey]backend.Kernel{},
			res:        map[resKey]*resident{},
			splits:     map[splitKey]int{},
			groupSplit: map[splitKey]bool{},
			reduce:     map[[2]int]backend.Kernel{},
			limit:      limit,
		}))
		// The zero-copy path is guarded on UnifiedMemory, not on the extension:
		// a discrete card may also advertise VK_EXT_external_memory_host, and
		// importing there would leave the weights behind PCIe for every matvec
		// to re-read. Only a device whose memory is the host's wraps for free.
		if a := backend.ImportAlign(s.Dev); a > 0 && unified(s.Dev) {
			d := g.devs[len(g.devs)-1]
			d.imp, d.impAlign = s.Dev.(backend.HostImport), uint64(a)
		}
		known := false
		for _, p := range g.pools {
			known = known || p == pool
		}
		if !known {
			g.pools = append(g.pools, pool)
		}
		if pool.Host() {
			g.hostPool = pool
		}
	}
	return g, nil
}

// label names a device the way a report should: its ordinal in the tier, the
// hardware and the API that reached it, so two identical cards are
// distinguishable in a counter dump.
func label(d backend.Device, ord int) string {
	return fmt.Sprintf("%d:%s [%s]", ord, d.Name(), d.API())
}

// Devices is how many devices this tier holds.
func (g *GPU) Devices() int { return len(g.devs) }

// Stats sums every device's counters. See Stats.add for the fields that do not
// sum.
func (g *GPU) Stats() Stats {
	var s Stats
	for _, d := range g.devs {
		s.add(d.stats())
	}
	return s
}

// DeviceKernels is one device's compiled kernels and what each cost it.
// Known is false where the backend cannot read its compiler's allocation
// (backend.ResourceReporter), and Kernels is then empty rather than a guess.
type DeviceKernels struct {
	Device  string                    `json:"device"`
	Known   bool                      `json:"known"`
	Kernels []backend.KernelResources `json:"kernels"`
}

// KernelResources is every device's DeviceKernels, fastest device first:
// registers, spill, shared memory and occupancy for each kernel this tier
// has compiled so far.
func (g *GPU) KernelResources() []DeviceKernels {
	out := make([]DeviceKernels, 0, len(g.devs))
	for _, d := range g.devs {
		dk := DeviceKernels{Device: label(d.dev, d.ord)}
		if r, ok := d.dev.(backend.ResourceReporter); ok {
			dk.Known, dk.Kernels = true, r.KernelResources()
		}
		out = append(out, dk)
	}
	return out
}

// DevStats is the same counters kept apart, fastest device first.
func (g *GPU) DevStats() []Stats {
	out := make([]Stats, 0, len(g.devs))
	for _, d := range g.devs {
		out = append(out, d.stats())
	}
	return out
}

func (d *devTier) stats() Stats {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.settleRounding() // see Bytes
	s := d.tot
	s.RanBeside = int(d.ranBeside.Load())
	if d.kvp != nil {
		for _, l := range d.kvp.layers {
			for _, ids := range l.owned {
				s.KVBytes += uint64(len(residentIDs(nil, ids))) * d.kvp.pageBytes(l)
			}
		}
		s.KVPoolBytes = d.kvp.charged()
	}
	s.BudgetUsed = d.used
	s.ScratchBytes, s.ReservedPrompt, s.ReservedStep = d.scratch, d.promptW, d.stepW
	s.TowerRows = d.visCap
	s.Allocated, s.RoundingBytes = uint64(d.allocated()), d.rounding
	return s
}

// Pools reports every distinct budget the tier spends from.
func (g *GPU) Pools() []*Pool { return g.pools }

// mvSample is how many matvec offers mvPays serves before deciding.
//
// mvPays decides, once per model, whether serving matvecs one at a time is
// worth a host/device crossing each. It is all or nothing: the break-even is
// ~3.3 MB of weights per crossing (a lone matvec pays quantize, upload, launch
// and readback), but applying it per tensor regressed, because q, k and v share
// one uploaded activation and declining one of them puts the upload back. So
// the average bytes per crossing over the model decides. See
// docs/engineering-history/placement.md for the measurements.
//
// The sample is taken by serving (a crossing cannot be priced without paying
// for one), about two layers. What reaches this path is whatever placement left
// on the host, so the verdict can differ with the seam.
//
// WithMinMatVecBytes overrides the threshold; zero restores the unpriced
// behaviour. cmd/jitllm fills it from JITLLM_MV_MIN_KB.
const mvSample = 64

// Callers hold g.mu: this is one more test inside the lock MatVec already takes,
// not a second acquisition.
func (g *devTier) mvPays(sid uint64, bytes int) bool {
	// The sample is per session, not per device: latched per device, the
	// verdict depended on how many matvecs the device had been offered since
	// it opened, so two States on one tier computed the same token with
	// different kernels (model.TestHybridSecondSessionMatchesTheFirst). Per
	// session every State samples the same weights in the same order and
	// reaches the same verdict, and keeps it (TestMVPaysDoesNotDrift).
	mv := g.mvFor(sid)
	if mv.verdict == 0 {
		mv.seen += uint64(bytes)
		mv.offers++
		if mv.offers < mvSample {
			return true
		}
		if mv.seen/uint64(mv.offers) >= g.mvMin {
			mv.verdict = 1
		} else {
			mv.verdict = -1
		}
	}
	if mv.verdict < 0 {
		g.TooSmall++
		return false
	}
	return true
}

// mvState is one session's running sample for the crossing verdict.
type mvState struct {
	seen    uint64
	offers  uint64
	verdict int8
}

// mvFor is the sampling state of session sid, created on first use. Callers
// hold g.mu, like everything else mvPays touches.
func (g *devTier) mvFor(sid uint64) *mvState {
	if g.mv == nil {
		g.mv = map[uint64]*mvState{}
	}
	m := g.mv[sid]
	if m == nil {
		m = &mvState{}
		g.mv[sid] = m
	}
	return m
}

// mvMinBytes is that break-even, in bytes: a constant derived from three rates
// measured on one box (host read, device read, seam). A faster link moves it
// down, making the guard conservative; a much dearer seam would make it decline
// too little. Measuring the three at open is the unbuilt alternative.
func mvMinBytes(kb knobs) uint64 {
	if kb.mvMinSet {
		return kb.mvMin
	}
	return 3300 << 10
}

// SetActWindow tells the tier which activation amax window this session uses,
// so its host-side quantize matches the CPU tier's byte for byte.
func (g *devTier) SetActWindow(w int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.actWin = w
}

// Bytes is what the device's budget has spent, the driver's rounding of every
// buffer included. The rounding is settled first: a buffer allocated or freed
// outside a charge moves it, and the last settlement is not the card.
func (g *devTier) Bytes() uint64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.settleRounding()
	return g.used
}

// freeScratch returns every device buffer and scratch-owned kernel a scratch
// holds. It runs on Close and on every KV grow (ensureKVCap rebuilds the
// scratch at the new stride), so a missed buffer leaks a whole set per grow,
// and a later allocation then fails.
//
// It walks the struct by reflection rather than a hand-written list, which
// fell behind every field added since; reflection is affordable once per grow.
// Duplicates are freed once: two fields may alias one buffer, and a double Free
// is a driver error.
func freeScratch(bs *blockScratch) {
	if bs == nil {
		return
	}
	bs.mg.free()  // a pointer, which the field walk below does not follow
	freeFlash(bs) // a slice, likewise
	// Both slices may hold nil where an allocation failed part-way, which is
	// how a card too small for a tower's planes reaches here.
	for _, b := range bs.qoffs {
		if b != nil {
			b.Free()
		}
	}
	bs.qoffs = nil
	for _, b := range bs.vwins { // the windows' per-chunk buffers, likewise
		if b != nil {
			b.Free()
		}
	}
	bs.vwins = nil
	for _, b := range bs.vsegs { // the packed sequences' buffers, likewise
		if b != nil {
			b.Free()
		}
	}
	bs.vsegs = nil
	freePaged(bs.pkv)
	bs.mlab.free()
	bs.ds4.free()
	bs.k3.free()
	bs.q4e.free()
	for n, rk := range bs.ragLin { // a map, which the walk does not follow either
		for _, k := range []backend.Kernel{rk.conv, rk.shift, rk.delta, rk.ngConv} {
			if k != nil {
				closeKern(k)
			}
		}
		delete(bs.ragLin, n)
	}
	seen := map[backend.Buf]bool{}
	seenK := map[backend.Kernel]bool{}
	v := reflect.ValueOf(bs).Elem()
	bufType := reflect.TypeOf((*backend.Buf)(nil)).Elem()
	kernType := reflect.TypeOf((*backend.Kernel)(nil)).Elem()
	for i := 0; i < v.NumField(); i++ {
		f := v.Field(i)
		// Only top-level backend.Kernel fields are closed: those are compiled
		// for this scratch and owned by it. Matvec kernels come from g.kerns,
		// a device-wide cache shared by every scratch, and reach the scratch
		// inside `mv` values, which this type filter skips by construction.
		// Do not turn this into a recursive walk: that would close cached
		// kernels still in use (a use-after-free).
		if f.Type() == kernType && !f.IsNil() {
			k, ok := reflect.NewAt(f.Type(), unsafe.Pointer(f.UnsafeAddr())).
				Elem().Interface().(backend.Kernel)
			if ok && k != nil && !seenK[k] {
				seenK[k] = true
				closeKern(k)
			}
			continue
		}
		if f.Type() != bufType || f.IsNil() {
			continue
		}
		// NewAt because the fields are unexported: reflect panics on
		// Interface() through an unexported name. The address is real and the
		// type is checked above, so re-deriving a value at it is sound.
		b, ok := reflect.NewAt(f.Type(), unsafe.Pointer(f.UnsafeAddr())).
			Elem().Interface().(backend.Buf)
		if !ok || b == nil || seen[b] {
			continue
		}
		seen[b] = true
		b.Free()
	}
	*bs = blockScratch{}
}

// releaseEveryLayer drops every block this device holds, through the same path a
// migration uses. Close calls it; see the comment there.
func (g *devTier) releaseEveryLayer() {
	g.mu.Lock()
	lo, hi, any := 0, 0, false
	for li := range g.layers {
		if !any || li < lo {
			lo = li
		}
		if !any || li >= hi {
			hi = li + 1
		}
		any = true
	}
	g.mu.Unlock()
	if any {
		g.ReleaseLayers(lo, hi)
	}
}

// RecSteps is how many linear blocks advanced their recurrent state during the
// last Layers call on this device. Zero after a failure means nothing moved and
// the caller may restart on the host; non-zero means it may not.
func (g *devTier) RecSteps() int {
	return g.recStepsOf(0)
}

// recStepsOf is RecSteps for session sid: its own last call's count.
func (g *devTier) recStepsOf(sid uint64) int {
	g.mu.Lock()
	defer g.mu.Unlock()
	if ds := g.sess[sid]; ds != nil {
		return ds.recSteps
	}
	return 0
}

// setRecSteps sets session sid's count; see GPU.resetRecSteps.
func (g *devTier) setRecSteps(sid uint64, n int) {
	g.mu.Lock()
	g.sessOf(sid).recSteps = n
	g.mu.Unlock()
}

// setInputs hands session sid's per-layer inputs and token ids to this device
// for its next call (gpuSession.inputsTo).
func (g *devTier) setInputs(sid uint64, ple []float32, ids []int32, ng []float32, marks []int32) {
	g.mu.Lock()
	ds := g.sessOf(sid)
	ds.pleHost, ds.tokIDs, ds.ngHost, ds.qsaMarks = ple, ids, ng, marks
	g.mu.Unlock()
}

func (g *devTier) Close() {
	// The layers go first, through ReleaseLayers (the refunds, freeRaw, the
	// decline tombstones, freeStale) rather than a second copy of that walk.
	// A driver that reclaims the context on close would hide a leak here, but
	// a unified-memory device would not. It takes g.mu, hence before the lock
	// below. The background tuner stops first: it times resident weights.
	g.stopTune()
	g.releaseEveryLayer()

	g.mu.Lock()
	defer g.mu.Unlock()
	// Idempotent: every free below routes through the owner goroutine, whose
	// channel is nil once the device is closed, so a second call would block
	// forever.
	if g.closed {
		return
	}
	g.closed = true
	// Every session's queue, before the device that owns them, once nothing
	// runs on one.
	g.quiesce()
	g.dhClose()
	for _, ds := range g.sess {
		ds.closeQueue()
	}
	if hp, ok := g.dev.(backend.HostPinner); ok {
		for i, b := range g.pin {
			if b != nil {
				hp.UnpinHost(b)
				g.pin[i] = nil
			}
		}
	}
	// nil entries are compile failures, cached so they are not retried; Close
	// on one would panic.
	for _, k := range g.kerns {
		if k != nil {
			closeKern(k)
		}
	}
	for _, k := range g.idKerns {
		closeKern(k)
	}
	for _, r := range g.res {
		if r.ok {
			r.qs.Free()
			r.d.Free()
			if r.sc != nil {
				r.sc.Free()
			}
		}
	}
	for _, k := range g.reduce {
		if k != nil {
			closeKern(k)
		}
	}
	for _, k := range g.restrides {
		closeKern(k)
	}
	// Every layer left the pool with its block; what remains is the pool
	// itself and its charge.
	g.freeKVPool(g.kvp)
	g.kvp = nil
	if g.argmaxK != nil {
		g.argmaxK.Close()
		g.argmaxOut.Free()
	}
	g.closeSample()
	g.lane0.freeSample()
	g.closePick()
	g.lane0.freePick()
	for _, k := range g.recCopies {
		closeKern(k)
	}
	g.freeAltNorms()
	if g.recNext != nil {
		g.recNext.Free()
	}
	for _, k := range g.segKerns {
		if k != nil {
			closeKern(k)
		}
	}
	for _, k := range g.ragKerns {
		if k != nil {
			closeKern(k)
		}
	}
	for _, k := range g.ragK {
		if k != nil {
			closeKern(k)
		}
	}
	for _, k := range g.convK {
		if k != nil {
			closeKern(k)
		}
	}
	for _, k := range g.unks {
		if k != nil {
			closeKern(k)
		}
	}
	g.dropGraph()
	// The block scratches, which nothing else frees (see freeScratch), and
	// the staging, refunded as they go.
	g.dropAllScratch()
	g.closeShared()
	g.closePersistCache()
	if g.rawBuf != nil {
		g.rawBuf.Free()
	}
	g.dev.Close()
}

// dropGraph releases the captured launch sequence. Anything that can move a
// buffer the recording named has to call it: a graph holds device addresses,
// not the Go objects that own them, so a freed-and-reallocated buffer would
// leave a replay reading memory that is no longer its own.
//
// It must not destroy a graph from inside a Session: on CUDA the destroy hops
// to the device-owning goroutine, which is busy running the session, so it
// would deadlock (the same constraint as Alloc). A recording retired from
// inside one goes on the stale list and is freed before the next session.
//
// What it is called for is a change to something every submission may name,
// so it waits for every submission in flight first (quiesce), and its caller,
// holding g.mu, makes the change before any other starts. Callers hold g.mu.
func (g *devTier) dropGraph() {
	g.quiesce()
	// Every recording, not just the current one, in every lane: a moved
	// address invalidates all of them, whichever session made them.
	g.eachLane(func(l *lane) {
		for k, r := range l.recs {
			l.stale = append(l.stale, r)
			delete(l.recs, k)
		}
		freeStaleOf(l)
	})
}

// freeStale destroys this lane's retired recordings. Callers must be outside a
// Session.
func (g *devTier) freeStale() { freeStaleOf(g.lane) }

// quantOf forwards to kernels.QuantOf rather than keeping a second per-format
// list, which went stale when a format was added. It answers for the device
// kernels, not the packer: a format the device cannot decode is false here
// (kernels.DeviceWhyNot says why). A float weight (F32, F16, BF16) is served
// too; the device reads its raw rows (kernels.FloatOf).
func quantOf(t quant.Type) (kernels.Quant, bool) {
	if q, ok := kernels.FloatOf(t); ok {
		return q, true
	}
	q, ok := kernels.QuantOf(t)
	return q, ok && kernels.DeviceWhyNot(q) == ""
}

// decline says why a device would not serve a matvec, which decides whether
// asking the next device is worth anything: noKernel and tooSmall are facts
// about the tensor or model, noRoom and failed about this card. GPU.MatVec is
// the only caller that acts on the difference.
type decline uint8

const (
	served   decline = iota
	noKernel         // the quant format has no GPU kernel: the same on every device
	tooSmall         // the crossing does not pay for this model: the same everywhere
	noRoom           // this card is full, or refused the allocation
	failed           // this card could not compile or could not launch
)

// MatVec implements nn.Device for one device, for the zero session. It
// returns false for anything it cannot serve, and the caller then runs the CPU
// path unchanged. GPU.MatVec is what a caller with several devices uses.
func (g *devTier) MatVec(out []float32, t quant.Type, w []byte, x []float32, nrows, k int) bool {
	return g.matVec(0, out, t, w, x, nrows, k) == served
}

func (g *devTier) matVec(sid uint64, out []float32, t quant.Type, w []byte, x []float32, nrows, k int) decline {
	q, ok := quantOf(t)
	if !ok || k%q.Elems() != 0 || nrows <= 0 || len(w) == 0 {
		g.mu.Lock()
		g.NoKernel++
		g.mu.Unlock()
		return noKernel
	}
	// A matvec served alone pays a whole host/device crossing, and whether that
	// pays is a property of the model (mvPays); unpriced, it made partial
	// placements slower than no device at all.
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.mvPays(sid, len(w)) {
		return tooSmall
	}

	r := g.resident(-1, 0, q, w, nrows, k, nil)
	if r == nil || !r.ok {
		// resident() has already counted this as Declined (the budget) or
		// NoRoom (the card), and either way another card may have the room.
		return noRoom
	}
	split := chooseSplit(nrows, k/32, g.kb.split)
	kern := g.kernel(q, k, nrows, split, 1, false)
	if kern == nil {
		return failed
	}
	var red backend.Kernel
	if split > 1 {
		if red = g.reduceKernel(nrows, split); red == nil {
			split, kern = 1, g.kernel(q, k, nrows, 1, 1, false)
			if kern == nil {
				return failed
			}
		}
	}
	t0 := time.Now()
	// Buffers are sized outside the session: Alloc takes its own ownership hop
	// and cannot nest inside one.
	fresh, ok := g.prepare(x, nrows, k, !kernels.IsFloat(q))
	if !ok || !g.sizeMVPart(nrows*split) || (kernels.IsFloat(q) && !g.sizeXF(k)) {
		return noRoom
	}
	t1 := time.Now()

	// One ownership hop for the whole matvec: two uploads, a launch and a
	// readback batched in one Session.
	g.mvc = mvCall{fresh: fresh, float: kernels.IsFloat(q), r: r, x: x, nrows: nrows, k: k,
		split: split, kern: kern, red: red}
	if g.mvcFn == nil {
		g.mvcFn = g.matVecSession
	}
	g.dev.Session(g.mvcFn)
	err := g.mvc.err
	g.mvc = mvCall{}
	if err != nil {
		return failed
	}
	t3 := time.Now()
	f := g.hostOut[:nrows]
	for i := range f {
		out[i] = f[i]
	}
	g.TStage += t1.Sub(t0)
	g.TLaunch += t3.Sub(t1)
	g.TConv += time.Since(t3)
	if g.Verify {
		g.verify(out, t, w, x, nrows, k)
	}
	g.Served++
	return served
}

// mvCall is one matVec's device side, handed to matVecSession through the
// devTier rather than captured by a closure: the function a Session runs
// escapes, so a closure over the call's locals was a heap object on every
// matvec a host block offered (a mixture's router, every token). g.mu is held
// across the call.
type mvCall struct {
	fresh, float    bool
	r               *resident
	x               []float32
	nrows, k, split int
	kern, red       backend.Kernel
	err             error
}

// matVecSession is matVec's Session: two uploads, a launch and a readback.
func (g *devTier) matVecSession(sess backend.Session) {
	c := &g.mvc
	k, nrows, split := c.k, c.nrows, c.split
	nb := k / 32
	if c.fresh {
		if c.err = sess.Write(g.aBuf, u32b(g.hostA[:k/4])); c.err != nil {
			return
		}
		if c.err = sess.Write(g.axBuf, f32b(g.hostAX[:3*nb])); c.err != nil {
			return
		}
	}
	dst := g.outBuf
	if split > 1 {
		dst = g.mvPart
	}
	// A float weight reads the float activation in the scale plane's slot
	// (kernels.MatVec), written every call: it is k floats.
	d := c.r.d
	if c.float {
		if c.err = sess.Write(g.xfBuf, f32b(c.x[:k])); c.err != nil {
			return
		}
		d = g.xfBuf
	}
	lc := launcher{to: &g.mvTo}
	g.mvTo.s = sess
	defer func() { g.mvTo.s = nil }()
	if c.err = lc.launch(c.kern, (nrows*split+127)/128, 128,
		c.r.qs, d, c.r.sc, g.aBuf, g.axBuf, dst); c.err != nil {
		return
	}
	if split > 1 {
		if c.err = lc.launch(c.red, (nrows+127)/128, 128, g.mvPart, g.outBuf); c.err != nil {
			return
		}
	}
	c.err = sess.Read(g.outBuf, g.hostRaw[:nrows*4])
}

// importing reports that a page-in on this device is a descriptor update rather
// than a transfer: the heap is the host's, the driver will wrap a host pointer,
// and the arena has a budget to hold the packs that get wrapped (without one
// the arena declines every pack and there is nothing to wrap). The device's
// alignment must also be one the arena provides, or the driver would refuse
// every array one call at a time.
func (g *devTier) importing() bool {
	return g.imp != nil && g.impAlign > 0 && g.impAlign <= kernels.HostAlign() &&
		g.arena.Limit() > 0
}

// importPacked wraps an arena entry's three arrays as device buffers that alias
// them, and reports whether all three came back.
//
// The length is the padded one: drivers require the imported size, not just
// the address, to be a multiple of the alignment (len(v)*4 is a refusal on
// Vulkan and a silent nil on Metal). The arena pads every array inside its own
// region. A partial import is unwound rather than kept.
func (g *devTier) importPacked(r *resident, p *kernels.Packed) bool {
	bufs := [3]backend.Buf{}
	ok := true
	for i, v := range [][]uint32{p.QS, p.D, p.SC} {
		if len(v) == 0 {
			// The kernel still takes the parameter. A word of device memory is
			// cheaper than refusing the whole import over an absent array.
			b, err := g.dev.Alloc(4)
			if err != nil {
				ok = false
				break
			}
			bufs[i] = b
			continue
		}
		b, err := g.imp.Import(kernels.Ptr(v), int(kernels.PaddedBytes(len(v))))
		if err != nil {
			g.LastErr = "import: " + err.Error()
			ok = false
			break
		}
		bufs[i] = b
	}
	if !ok {
		for _, b := range bufs {
			if b != nil {
				b.Free()
			}
		}
		return false
	}
	r.qs, r.d, r.sc = bufs[0], bufs[1], bufs[2]
	r.imported, r.ok, r.pk = true, true, p
	g.arena.Import(p)
	return true
}

// resident uploads a tensor once, keyed by resKey.
//
// block is the caller's block index, or -1 for a tensor that is not part of one
// (the output projection, a lone matvec offered by the host tier). It is a tag
// telling the arena which entries to release together, and nothing else.
func (g *devTier) resident(block, slot int, q kernels.Quant, w []byte, nrows, k int, pre *nn.Packed) *resident {
	g.applyForgets()
	key := resKey{block: block, slot: slot, nrows: nrows, k: k, t: q}
	if block < 0 {
		// The ad-hoc path: no block, no slot, and the weight is host-owned.
		key.p = &w[0]
	}
	if r, ok := g.res[key]; ok {
		// The shape is part of the key because an expert bank and its expert 0
		// begin at the same byte: both shapes are legitimate and both may be
		// resident. Sharing one entry would launch a kernel against a
		// differently strided payload (a silent out-of-bounds read), and
		// refusing on a mismatch would stop a bank from ever being placed.
		return r
	}
	r := &resident{nrows: nrows, k: k, t: q}
	g.res[key] = r

	// Price it before packing it: declining after a pack costs a full read and
	// fresh host memory to learn the answer is no. PackedWords is the same
	// arithmetic with no allocation (TestPackedWordsMatchesPack).
	pq, pd, psc, err := kernels.PackedWords(q, nrows, k)
	if err != nil {
		g.LastErr = "PackedWords " + q.String() + ": " + err.Error()
		return r
	}
	need := uint64(pq+pd+psc) * 4

	// A bank in expert pages is assembled on the device, one sheet at a time:
	// there is no host span of the whole bank, and gathering one would hold a
	// layer's every expert in host memory at once.
	if pre != nil && pre.Sheet != nil {
		if !g.room(need) {
			g.Declined++
			g.LastErr = fmt.Sprintf("%s %dx%d needs %d, %d of %d used", q, nrows, k, need, g.used, g.limit)
			return r
		}
		tu := time.Now()
		ok := g.residentSheets(r, q, nrows, k, pre)
		g.TUpload += time.Since(tu)
		if !ok {
			g.NoRoom++
			return r
		}
		g.Prepacked++
		g.PrepackedBytes += need
		g.charge(need)
		return r
	}
	// The arena packs GGUF bytes, so it serves a tensor that has no container
	// form. A container's spans are already the device layout, and read as
	// GGUF they are too short for any quantized format: the arena refused
	// them and every container block was declined on a unified device with an
	// arena budget (TestAContainerWeightIsNotRepackedOnAUnifiedDevice). They
	// are copied below, which also lets the pager reuse the frame they came
	// from; a wrapper over a frame would read the next block's bytes.
	var pk *kernels.Packed
	if pre == nil && g.importing() {
		tp := time.Now()
		p, aerr := g.arena.Pack(block, q, w, nrows, k)
		g.TPack += time.Since(tp)
		if aerr != nil {
			g.LastErr = "arena " + q.String() + ": " + aerr.Error()
			return r
		}
		if p != nil {
			if g.importPacked(r, p) {
				g.Imports++
				g.ImportBytes += need
				return r
			}
			pk = p // the wrap was refused; upload these same bytes instead
		}
	}
	if !g.room(need) {
		g.Declined++
		g.LastErr = fmt.Sprintf("%s %dx%d needs %d, %d of %d used (pool %s: %d of %d)",
			q, nrows, k, need, g.used, g.limit, g.pool.Name(), g.pool.Used(), g.pool.Limit())
		return r // declines from here on; the CPU keeps this tensor
	}
	// A converted container skips every path below: its bytes are already what
	// the kernel reads (packed once, offline), so there is nothing to pack or
	// unpack and no arena entry, and the spans go straight to the driver.
	if pre != nil {
		// Spans shorter than the shape's planes (the container may pad them
		// longer) are none at all -- a host page given back -- or another
		// tensor's, and a kernel would read past them; they are refused,
		// never uploaded.
		if len(pre.QS) < 4*pq || len(pre.D) < 4*pd || len(pre.SC) < 4*psc {
			g.LastErr = fmt.Sprintf("%s %dx%d: planes of %d, %d and %d bytes, short of the shape's %d, %d and %d",
				q, nrows, k, len(pre.QS), len(pre.D), len(pre.SC), 4*pq, 4*pd, 4*psc)
			return r
		}
		tu := time.Now()
		bq, eq := g.allocBytes(pre.QS)
		bd, ed := g.allocBytes(pre.D)
		bs, es := g.allocBytes(pre.SC)
		g.TUpload += time.Since(tu)
		if bq == nil || bd == nil || bs == nil {
			for _, b := range []backend.Buf{bq, bd, bs} {
				if b != nil {
					b.Free()
				}
			}
			g.NoRoom++
			// The driver's own words, so a reader can tell a full card from
			// a failed copy.
			g.LastErr = fmt.Sprintf("%s %dx%d: a plane of %d, %d and %d bytes was not uploaded: %v",
				q, nrows, k, len(pre.QS), len(pre.D), len(pre.SC), errors.Join(eq, ed, es))
			return r
		}
		r.qs, r.d, r.sc, r.ok = bq, bd, bs, true
		g.Prepacked++
		g.PrepackedBytes += need
		g.charge(need)
		return r
	}

	// The device unpack comes before the arena: the upload is roughly the file's
	// bytes and the card extracts far faster than the host packs (see
	// kernels.Unpack and arena.go). It is below the importing() branch because
	// on a unified heap the pack is the device buffer, so the arena wins there.
	// A false is a counted fallback (NoUnpackKernel, NoUnpackRoom), not a
	// decline: the host path below produces the same bytes.
	if g.unpackInto(r, q, w, nrows, k) {
		g.charge(need)
		return r
	}
	// Then the arena: a tensor it holds is a map lookup and a DMA of bytes
	// already in the device layout, and one it does not hold is packed into it
	// when it has budget, so the next page-in is that lookup. nil is not an
	// error: the arena is off (the default) or full.
	var qs, dw, scw []uint32
	p := pk
	if p == nil {
		tp := time.Now()
		var aerr error
		p, aerr = g.arena.Pack(block, q, w, nrows, k)
		g.TPack += time.Since(tp)
		if aerr != nil {
			g.LastErr = "arena " + q.String() + ": " + aerr.Error()
			return r
		}
	}
	if p != nil {
		qs, dw, scw = p.QS, p.D, p.SC
	}
	// Use a background pack if one is waiting. Packing is most of preparing a
	// block and is pure host arithmetic, so it is the part of a migration that
	// can overlap a token (Prewarm). It is below the arena because Prewarm packs
	// into the arena when there is budget; a stage exists only for a tensor the
	// arena declined.
	if qs == nil {
		qs, dw, scw = g.takeStage(key)
	}
	if qs == nil {
		tp := time.Now()
		var err error
		qs, dw, scw, err = kernels.PackWeights(q, w, nrows, k)
		g.TPack += time.Since(tp)
		g.Packs++
		if err != nil {
			g.LastErr = "PackWeights " + q.String() + ": " + err.Error()
			return r
		}
	}
	tu := time.Now()
	defer func() { g.TUpload += time.Since(tu) }()
	alloc := func(v []uint32) backend.Buf {
		if len(v) == 0 {
			// The kernel still takes the parameter, so hand it a word.
			v = []uint32{0}
		}
		b, err := g.dev.Alloc(len(v) * 4)
		if err != nil {
			return nil
		}
		if b.Write(u32b(v)) != nil {
			b.Free()
			return nil
		}
		return b
	}
	bq, bd, bs := alloc(qs), alloc(dw), alloc(scw)
	if bq == nil || bd == nil || bs == nil {
		for _, b := range []backend.Buf{bq, bd, bs} {
			if b != nil {
				b.Free()
			}
		}
		// The device refusing is not the budget declining, and is counted apart:
		// over budget is a choice, out of memory is the card being full (for
		// instance another process holding VRAM).
		g.NoRoom++
		return r
	}
	r.qs, r.d, r.sc, r.ok = bq, bd, bs, true
	g.charge(need)
	return r
}

// residentSheets allocates a bank's three planes and fills them from its
// expert pages: sheet x of each plane at x times one sheet's bytes, which is
// exactly the buffer a contiguous bank would have uploaded.
func (g *devTier) residentSheets(r *resident, q kernels.Quant, nrows, k int, pre *nn.Packed) bool {
	if pre.Sheets <= 0 || nrows%pre.Sheets != 0 {
		g.LastErr = fmt.Sprintf("tier: %d rows do not split into %d sheets", nrows, pre.Sheets)
		return false
	}
	sq, sd, ssc, err := kernels.PackedWords(q, nrows/pre.Sheets, k)
	if err != nil {
		g.LastErr = err.Error()
		return false
	}
	per := [3]int{sq * 4, sd * 4, ssc * 4}
	var bufs [3]backend.Buf
	free := func() {
		for _, b := range bufs {
			if b != nil {
				b.Free()
			}
		}
	}
	for i, n := range per {
		b, err := g.dev.Alloc(max(n*pre.Sheets, 4))
		if err != nil {
			free()
			return false
		}
		bufs[i] = b
	}
	// The sheets are read sheetsAhead ahead of the copy: one at a time, each
	// was a small request at queue depth 1. Each is copied straight from its
	// frame; gathering a window into one buffer first cost 6.7 s of a 10.4 s
	// upload on Qwen3-30B-A3B, more than the copies it saved calls on (the
	// driver already stages a pageable copy).
	rd := newSheetReader(pre)
	defer rd.close()
	for x := 0; x < pre.Sheets; x++ {
		sr := rd.get(x)
		if sr.err != nil {
			sr.release()
			g.LastErr = sr.err.Error()
			free()
			return false
		}
		for i, src := range [3][]byte{sr.p.QS, sr.p.D, sr.p.SC} {
			if per[i] == 0 {
				continue
			}
			if len(src) != per[i] {
				sr.release()
				g.LastErr = fmt.Sprintf("tier: expert %d plane %d is %d bytes, want %d", x, i, len(src), per[i])
				free()
				return false
			}
			if err := bufs[i].WriteAt(x*per[i], src); err != nil {
				sr.release()
				g.LastErr = err.Error()
				free()
				return false
			}
		}
		sr.release()
	}
	r.qs, r.d, r.sc, r.ok = bufs[0], bufs[1], bufs[2], true
	return true
}

// sheetsAhead is how many of a bank's sheets are read ahead of its upload:
// jlm.ReadThreads, the queue depth an NVMe wants.
const sheetsAhead = 16

// sheetRead is one sheet of a bank, read ahead of the upload.
type sheetRead struct {
	p       nn.Packed
	release func()
	err     error
}

// sheetReader reads a bank's sheets in order, sheetsAhead in flight.
type sheetReader struct {
	pre  *nn.Packed
	ch   []chan sheetRead
	next int // the first sheet not yet asked for
}

func newSheetReader(pre *nn.Packed) *sheetReader {
	r := &sheetReader{pre: pre, ch: make([]chan sheetRead, pre.Sheets)}
	for range sheetsAhead {
		r.launch()
	}
	return r
}

func (r *sheetReader) launch() {
	if r.next >= r.pre.Sheets {
		return
	}
	x, c := r.next, make(chan sheetRead, 1)
	r.ch[x] = c
	r.next++
	go func() {
		p, release, err := r.pre.Sheet(x)
		c <- sheetRead{p, release, err}
	}()
}

// get is sheet x, taken in order, and asks for the next one.
func (r *sheetReader) get(x int) sheetRead {
	sr := <-r.ch[x]
	r.ch[x] = nil
	r.launch()
	return sr
}

// close releases every sheet read ahead and never taken, so an upload that
// fails part way leaves no page held.
func (r *sheetReader) close() {
	for x, c := range r.ch {
		if c != nil {
			sr := <-c
			sr.release()
			r.ch[x] = nil
		}
	}
}

// allocBytes uploads one already-packed span. An empty span still needs a
// buffer because the kernel takes the parameter either way (Q4_0 and Q8_0
// have no scale table). A refusal is the driver's error.
func (g *devTier) allocBytes(v []byte) (backend.Buf, error) {
	if len(v) == 0 {
		v = make([]byte, 4)
	}
	b, err := g.dev.Alloc(len(v))
	if err != nil {
		return nil, err
	}
	if err := b.Write(v); err != nil {
		b.Free()
		return nil, err
	}
	return b, nil
}

func (g *devTier) kernel(q kernels.Quant, k, rows, split, rowt int, bias bool) backend.Kernel {
	return g.kernelMode(q, k, rows, split, rowt, bias, false)
}

// kernelMode is kernel with the in-group split reduction selectable. See
// kernels.MatVecShape.GroupSplit: it removes the partial buffer and the Reduce
// launch, which wins on small matvecs and loses on big-k ones, so it is chosen
// per shape by tuneSplit rather than switched on.
func (g *devTier) kernelMode(q kernels.Quant, k, rows, split, rowt int, bias, groupSplit bool) backend.Kernel {
	kk := kernKey{q, k, rows, split, bias, rowt, groupSplit, false, 0, false}
	if c, ok := g.kerns[kk]; ok {
		return c
	}
	ker, err := kernels.MatVec(kernels.MatVecShape{Center: g.center, T: q, K: k, Rows: rows, Split: split,
		Bias: bias, Rowt: rowt, GroupSplit: groupSplit})
	if err != nil {
		g.kerns[kk], g.LastErr = nil, err.Error()
		return nil
	}
	c, err := g.dev.Compile(ker)
	if err != nil {
		// A compile failure is recorded, not swallowed, so a silent fallback
		// to the CPU is visible.
		g.kerns[kk], g.LastErr = nil, err.Error()
		g.Failed++
		return nil
	}
	g.kerns[kk] = c
	return c
}

// idKernel is an indexed matvec -- an expert bank, or MLA's per-head absorb
// bank -- from the device-wide cache, compiling on a miss. Every block of a
// mixture asks for the same few shapes, so they share one kernel, which Close
// closes: a kernel compiled per block and kept by the block was closed by
// nothing, since ReleaseLayers leaves kernels to the caches.
func (g *devTier) idKernel(s kernels.MatVecShape) (backend.Kernel, error) {
	if c, ok := g.idKerns[s]; ok {
		return c, nil
	}
	ker, err := kernels.MatVec(s)
	if err != nil {
		return nil, err
	}
	c, err := g.dev.Compile(ker)
	if err != nil {
		return nil, err
	}
	if g.idKerns == nil {
		g.idKerns = map[kernels.MatVecShape]backend.Kernel{}
	}
	g.idKerns[s] = c
	return c, nil
}

// gatedKernel is kernelMode's Gate variant: the matvec writes act(gate)*row.
// nil where the shape has no final-row form (a partial-writing split).
func (g *devTier) gatedKernel(q kernels.Quant, k, rows, split int, groupSplit bool, act kernels.ActKind) backend.Kernel {
	if split > 1 && !groupSplit {
		return nil
	}
	kk := kernKey{t: q, k: k, rows: rows, split: split, rowt: 1, group: groupSplit, gated: true, act: act}
	if c, ok := g.kerns[kk]; ok {
		return c
	}
	ker, err := kernels.MatVec(kernels.MatVecShape{Center: g.center, T: q, K: k, Rows: rows, Split: split,
		GroupSplit: groupSplit, Gate: true, GateAct: act})
	if err != nil {
		g.kerns[kk], g.LastErr = nil, err.Error()
		return nil
	}
	c, err := g.dev.Compile(ker)
	if err != nil {
		g.kerns[kk], g.LastErr = nil, err.Error()
		g.Failed++
		return nil
	}
	g.kerns[kk] = c
	return c
}

// gate2Kernel is the FFN up projection summing the gate's row beside its own
// and writing act(gate)*up (kernels.MatVecShape.Gate2), or nil where the
// backend cannot make it exact or the split writes partials.
func (g *devTier) gate2Kernel(q kernels.Quant, k, rows, split int, groupSplit bool, act kernels.ActKind) backend.Kernel {
	if split > 1 && !groupSplit || !kernels.Gate2Exact(g.dev.API()) || kernels.IsFloat(q) {
		return nil
	}
	kk := kernKey{t: q, k: k, rows: rows, split: split, rowt: 1, group: groupSplit, gate2: true, act: act}
	if c, ok := g.kerns[kk]; ok {
		return c
	}
	ker, err := kernels.MatVec(kernels.MatVecShape{Center: g.center, T: q, K: k, Rows: rows, Split: split,
		GroupSplit: groupSplit, Gate2: true, GateAct: act})
	if err != nil {
		g.kerns[kk], g.LastErr = nil, err.Error()
		return nil
	}
	c, err := g.dev.Compile(ker)
	if err != nil {
		g.kerns[kk], g.LastErr = nil, err.Error()
		g.Failed++
		return nil
	}
	g.kerns[kk] = c
	return c
}

// reduceKernel compiles the second pass that sums a row's partial results.
func (g *devTier) reduceKernel(rows, split int) backend.Kernel {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	key := [2]int{rows, split}
	if c, ok := g.reduce[key]; ok {
		return c
	}
	ker, err := kernels.Reduce(rows, split)
	if err != nil {
		g.reduce[key] = nil
		return nil
	}
	c, err := g.dev.Compile(ker)
	if err != nil {
		g.reduce[key], g.LastErr = nil, err.Error()
		g.Failed++
		return nil
	}
	g.reduce[key] = c
	return c
}

// sizeF16 grows f16Buf to n bytes, dropping any recording that names the old
// one, under regrow's rules.
func (g *devTier) sizeF16(n int) bool { return g.regrow(&g.f16Buf, &g.f16Cap, n, 1, true, false) }

// sizePart grows the lane's partial-sum buffer. The blocks grow it for a
// shape a recording may not have had: the lane's recordings that name the old
// address are dropped.
func (g *devTier) sizePart(n int) bool { return g.regrow(&g.partBuf, &g.partCap, n, 4, true, true) }

// sizeMVPart grows matVec's own partial sums (devShared.mvPart). No recording
// names them.
func (g *devTier) sizeMVPart(n int) bool {
	return g.regrow(&g.mvPart, &g.mvPartCap, n, 4, false, true)
}

// regrow grows *dst from *have to n elements of unit bytes, dropping the
// captured graph first when drop is set and poisoning the new buffer when
// poison is.
//
// The old buffer is freed before the new one is asked for -- a full card can
// only grow by giving the old space back -- so the capacity goes to zero
// first, or a smaller request would take the fast path against a freed buffer.
// And when the card refuses the new size, the old one is asked for again:
// every launch already built reads this buffer, and the fallback a refusal
// sends a chunk down (narrower submissions, a row at a time) launches them
// against nil -- a panic on Vulkan, address 0 on CUDA. Callers hold g.mu.
func (g *devTier) regrow(dst *backend.Buf, have *int, n, unit int, drop, poison bool) bool {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	if *have >= n {
		return true
	}
	// Staging is the device's own: charged as scratch (scratch.go).
	defer g.scratchWin().close()
	if drop {
		// Every buffer regrow drops for is a lane's: only the lane's
		// recordings name it.
		g.dropLaneGraph()
	}
	old := *have
	if *dst != nil {
		(*dst).Free()
		*dst = nil
	}
	*have = 0
	alloc := func(n int) bool {
		b, err := g.dev.Alloc(n * unit)
		if err != nil {
			g.LastErr = err.Error()
			return false
		}
		if poison {
			g.poisonBuf(b, n*unit)
		}
		*dst, *have = b, n
		return true
	}
	if alloc(n) {
		return true
	}
	if old > 0 {
		alloc(old)
	}
	return false
}

// prepare quantizes the activation vector if it changed and makes sure every
// device buffer is big enough. It performs no device calls except Alloc, so the
// caller can run the transfers inside one session.
//
// It skips everything when x has not changed: q, k and v share one vector, and
// gate and up another, so seven matvecs a layer need four quantizations.
//
// quantize is false for a float weight, whose kernel reads the raw f32
// activation out of the scale plane's slot and never looks at the int8 planes,
// so they are not computed.
func (g *devTier) prepare(x []float32, nrows, k int, quantize bool) (fresh, ok bool) {
	if !g.sizeOut(nrows) {
		return false, false
	}
	if !quantize {
		// The int8 buffers are still sized, because the kernel's signature
		// binds them whatever the weight's format; only the arithmetic is
		// skipped.
		nb := k / 32
		if !g.sizeAct(k/4, 3*nb) {
			return false, false
		}
		return false, true
	}
	if g.staged && len(g.lastX) == len(x) && sameF64(g.lastX, x) {
		return false, true
	}
	g.Restaged++
	nb := k / 32
	if cap(g.hostA) < k/4 {
		g.hostA = make([]uint32, k/4)
		g.hostAX = make([]float32, 3*nb)
	}
	tq := time.Now()
	a, ax := g.hostA[:k/4], g.hostAX[:3*nb]
	// Fanned out across cores when large enough (see quantPar): blocks are
	// independent, so the split needs no synchronisation beyond the wait.
	quantPar(a, ax[:nb], ax[nb:], x, nb, g.actWin)
	g.TQuant += time.Since(tq)

	if !g.sizeAct(len(a), len(ax)) {
		return false, false
	}
	g.lastX = append(g.lastX[:0], x...)
	g.staged = true
	return true, true
}

// quantPar quantizes float32 activations, split across cores. Each 32-element
// block is independent.
func quantPar(a []uint32, as, sum []float32, x []float32, nb, window int) {
	if window < 32 {
		window = 32
	}
	// The split must land on a window boundary: each worker computes the amax
	// of its slice, so cutting a window in half would give its halves
	// different scales from the CPU tier's.
	blocksPerWin := window / 32
	n := runtime.GOMAXPROCS(0)
	if n > 6 {
		n = 6 // decode uses six P-cores; more threads buy variance
	}
	// Fan out only when there is enough work to pay for the goroutines: a
	// typical 2048-float vector is faster serial.
	if nb < 512 {
		n = 1
	}
	if n == 1 {
		kernels.PackActivationsInto(a, as, sum, x, window)
		return
	}
	per := (nb + n - 1) / n
	per = ((per + blocksPerWin - 1) / blocksPerWin) * blocksPerWin
	var wg sync.WaitGroup
	for w := 0; w < n; w++ {
		lo := w * per
		hi := lo + per
		if hi > nb {
			hi = nb
		}
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			kernels.PackActivationsInto(a[lo*8:hi*8], as[lo:hi], sum[lo*2:hi*2], x[lo*32:hi*32], window)
		}(lo, hi)
	}
	wg.Wait()
}

// Reserve sizes every per-call scratch buffer once, for the largest shape the
// model contains (known at load), so nothing grows on the serving path. That
// avoids a free and Alloc inside a token, a dropGraph each time a buffer moves,
// and the grow-failure mode TestGrowFailureDoesNotPoisonTheBuffer guards.
//
// The reservation is a few MiB and is not charged to g.used, which prices
// weights only. The lazy grow paths stay as the fallback for a caller that
// never reserves or a shape past what was reserved.
func (g *devTier) Reserve(maxRows, maxK int) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if maxRows <= 0 || maxK <= 0 {
		return false
	}
	// chooseSplit doubles while rows*split stays under twice the device's
	// thread slots, so a narrow matrix can reach a wider product than the
	// widest matrix does. Bound the partials by both.
	parts := maxRows
	if n := 4 * 30720; n > parts {
		parts = n
	}
	return g.sizeOut(maxRows) && g.sizePart(parts) && g.sizeMVPart(parts) &&
		g.sizeAct(maxK/4, 3*(maxK/32))
}

// sizeAct grows the activation staging buffers. Split out of prepare so that
// Reserve can call it without quantizing anything.
func (g *devTier) sizeAct(na, nax int) bool {
	return g.regrow(&g.aBuf, &g.aCap, na, 4, false, true) && g.regrow(&g.axBuf, &g.axCap, nax, 4, false, false)
}

// sizeXF grows the float activation buffer a float weight's matvec reads.
func (g *devTier) sizeXF(k int) bool { return g.regrow(&g.xfBuf, &g.xfCap, k, 4, false, true) }

// sizeOut grows the readback buffer. nrows is baked into the kernel, so there
// is nothing to publish to the device.
func (g *devTier) sizeOut(nrows int) bool {
	if cap(g.hostRaw) < nrows*4 {
		g.hostRaw = make([]byte, nrows*4)
		g.hostOut = unsafe.Slice((*float32)(unsafe.Pointer(&g.hostRaw[0])), cap(g.hostRaw)/4)
	}
	return g.regrow(&g.outBuf, &g.outCap, nrows, 4, false, false)
}

func sameF64(a, b []float32) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// u32b and f32b view a slice as bytes. An empty slice is empty bytes, not a
// panic: a block with no rotary (a vision tower) legitimately stages an empty
// cos/sin table.
func u32b(v []uint32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}
func f32b(v []float32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}

// verify recomputes the matvec in float64 from the dequantized weights and
// records any shape whose GPU answer disagrees. It compares in situ, which
// catches faults in what the tier does between calls that a kernel tested in
// isolation cannot show.
func (g *devTier) verify(out []float32, t quant.Type, w []byte, x []float32, nrows, k int) {
	if g.bad == nil {
		g.bad = map[string]int{}
	}
	// NMSE over the whole output, not per-row relative error: a long dot
	// product is often near zero by cancellation, and a relative error against
	// a near-zero denominator is unbounded.
	ref := make([]float64, k)
	var sse, sy2 float64
	worst, at := 0.0, -1
	for r := 0; r < nrows; r++ {
		rowBytes := len(w) / nrows
		if err := quant.Dequant(t, w[r*rowBytes:(r+1)*rowBytes], ref); err != nil {
			return
		}
		var want float64
		for i := 0; i < k; i++ {
			want += ref[i] * float64(x[i])
		}
		d := float64(out[r]) - want
		sse += d * d
		sy2 += want * want
		if ad := d * d; ad > worst {
			worst, at = ad, r
		}
	}
	nmse := 0.0
	if sy2 > 0 {
		nmse = sse / sy2
	}
	key := fmt.Sprintf("%s %dx%d", t, nrows, k)
	if g.bad[key] == 0 {
		g.bad[key] = 1
		verdict := "OK"
		if nmse > 1e-10 {
			verdict = "EXCEEDS 1e-10"
		}
		fmt.Printf("  VERIFY: %-18s NMSE %.3e  %s  (worst row %d)\n", key, nmse, verdict, at)
	}
}

// blockBytes is what a whole block would cost on the device, in bytes, counting
// only tensors that are not already resident. A block is all-or-nothing, so it
// is priced whole before any upload; declining on the eighth tensor would
// strand the seven already uploaded.
func (g *devTier) blockBytes(ws []nn.Weight) (uint64, bool) {
	g.applyForgets()
	// copied is what is uploaded whatever the device, and packable what the
	// arena may take instead: a container weight is never the arena's
	// (resident), so it is priced in full even on an importing device.
	var copied, packable uint64
	for _, x := range ws {
		if len(x.Data) == 0 {
			continue
		}
		q, ok := quantOf(x.T)
		if !ok {
			return 0, false
		}
		if _, ok := g.res[resKey{p: &x.Data[0], nrows: x.Rows, k: x.K, t: q}]; ok {
			continue // already paid for at this shape
		}
		// Priced at what the card spends on the three buffers, the driver's
		// page rounding included (scratch.go): admitting at the bytes alone let
		// the rounding push the block's last tensors over the budget.
		pb, err := g.planesFootprint(q, x.Rows, x.K)
		if err != nil {
			return 0, false
		}
		if x.Packed != nil {
			copied += pb
		} else {
			packable += pb
		}
	}
	// On an importing device the arena pays: a tensor it takes is wrapped with
	// no device allocation (see resident.bytes), so only the overflow beyond
	// the arena's headroom is priced against the card.
	if g.importing() {
		packable -= min(packable, g.arena.Headroom())
	}
	return copied + packable, true
}

// staged is a pack that has been done but not uploaded.
type staged struct {
	qs, d, sc []uint32
	t         kernels.Quant
	nrows, k  int
}

// takeStage removes and returns a matching background pack, or nil. The key
// includes the shape, for resident()'s reason: an expert bank and its expert 0
// begin at the same byte.
func (g *devTier) takeStage(key resKey) (qs, d, sc []uint32) {
	if g.stage == nil {
		return nil, nil, nil
	}
	st := g.stage[key]
	if st == nil {
		return nil, nil, nil
	}
	delete(g.stage, key)
	g.stageBytes -= st.bytes()
	g.Config.stage.give(st.bytes())
	return st.qs, st.d, st.sc
}

func (s *staged) bytes() uint64 { return uint64(len(s.qs)+len(s.d)+len(s.sc)) * 4 }

// prewarm packs one weight tensor off the main loop, so that a later PrepLayer
// only has to upload it, tagged with the block the arena groups entries by.
// The pack runs outside g.mu (layersOnce holds it for a whole token), so it can
// overlap decoding; the lock covers only the map operations. It is bounded by
// StageLimit, since a stage is host memory nothing uses yet; over the cap it
// declines and the work happens on the main loop.
func (g *devTier) prewarm(block int, q kernels.Quant, w []byte, nrows, k int) bool {
	if len(w) == 0 {
		return false
	}
	key := resKey{p: &w[0], nrows: nrows, k: k, t: q}
	g.mu.Lock()
	g.applyForgets()
	if g.res[key] != nil || (g.stage != nil && g.stage[key] != nil) {
		g.mu.Unlock()
		return false // already resident, or already staged
	}
	// A format the device unpacks has no host pack to move off the main loop:
	// resident() asks unpackInto before takeStage, so a stage would only hold
	// host memory until DropStage.
	if g.canUnpack(q) {
		g.mu.Unlock()
		return false
	}
	g.mu.Unlock()

	// Into the arena when it has budget: a stage is consumed by the upload,
	// while an arena entry survives it, so the block never packs again.
	if g.arena.Get(q, w, nrows, k) != nil {
		return false // already packed; nothing to do, exactly as "already staged"
	}
	t0 := time.Now()
	if p, err := g.arena.Pack(block, q, w, nrows, k); err == nil && p != nil {
		g.mu.Lock()
		g.TPrewarm += time.Since(t0)
		g.mu.Unlock()
		return true
	}
	g.mu.Lock()
	g.applyForgets()
	if g.res[key] != nil || (g.stage != nil && g.stage[key] != nil) {
		g.mu.Unlock()
		return false
	}
	limit := g.StageLimit
	if limit == 0 {
		limit = defaultStageLimit
	}
	// The tier's total, not this device's: the stage is host memory (see
	// Config.stage).
	if g.Config.stage.Used() >= limit {
		g.mu.Unlock()
		return false
	}
	g.mu.Unlock()

	g.mu.Lock()
	pk := &packing{at: uintptr(unsafe.Pointer(&w[0]))}
	g.packing = append(g.packing, pk)
	g.mu.Unlock()
	t0 = time.Now()
	qs, d, sc, err := kernels.PackWeights(q, w, nrows, k)
	took := time.Since(t0)
	g.mu.Lock()
	defer g.mu.Unlock()
	// A forget the page-in queued while the pack ran is applied first, so it
	// can spoil this one.
	g.applyForgets()
	g.donePacking(pk)
	if err != nil {
		return false
	}
	g.Packs++
	st := &staged{qs: qs, d: d, sc: sc, t: q, nrows: nrows, k: k}
	// Another goroutine, or the main loop, may have taken this tensor while the
	// pack was running. Both outcomes are correct; drop the loser's work.
	if g.res[key] != nil || (g.stage != nil && g.stage[key] != nil) {
		return false
	}
	// The host was told the bytes under the pack changed while it ran: the
	// pack may be of either tensor, so it is not kept.
	if pk.spoiled {
		return false
	}
	if g.stage == nil {
		g.stage = map[resKey]*staged{}
	}
	g.stage[key] = st
	g.stageBytes += st.bytes()
	g.Config.stage.take(st.bytes())
	g.TPrewarm += took
	return true
}

// DropStage frees every pack that has not been uploaded.
func (g *devTier) DropStage() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.Config.stage.give(g.stageBytes)
	g.stage, g.stageBytes = nil, 0
}

// defaultStageLimit is two 30B blocks' worth: enough to keep the main loop fed
// while it adopts one block per token, and small enough that the staging area
// is never the thing that runs the host out of memory.
const defaultStageLimit = 1 << 30

// Err reports the last reason a device operation declined, for diagnostics.
func (g *devTier) Err() string { g.mu.Lock(); defer g.mu.Unlock(); return g.LastErr }

// The packed arena's retention budget is Config.kb.arena, set by WithArena. Unset
// means zero, retain nothing: the arena is a second, wider copy of a block's
// weights in host memory, which a model that does not page should not pay
// for. The pager sets it with SetArena from the placement it actually made.

// maxGraphs bounds the launch-sequence recordings one session holds at once. A
// decode token issues at most two distinct keys (the blocks and, under a
// partial seam, the head), and nCap steps them every scoreGrain positions, so
// the live set is small and the bound only has to stop a long generation
// accumulating. A speculating hybrid's trunk issues many more over a session
// -- every row count a verification, a replay or its commit runs, at both
// recurrent parities -- of which a few run nearly every round, so the bound
// retires the least recently used: retiring all of them re-recorded the hot
// verification each time a rare count came past.
const maxGraphs = 8

// sharedCompact is residentCompact with one buffer per (matrix, format, shape)
// for the whole device instead of one per block.
//
// Block i's compact bank is filled at its suspension and consumed by its own
// FFN a few launches later, so it is dead before the next block needs it, and
// one buffer lets every block of a large streamed mixture fit where per-block
// copies did not. It is safe under the conditions streaming already needs: the
// suspension Syncs before the next fill lands, a streamed range refuses graph
// capture, and a streamed block's submission runs alone on the device
// (exclusiveRange).
//
// A shape that does not match an existing bank gets its own, so a model whose
// blocks differ is correct rather than refused.
func (g *devTier) sharedCompact(which int, q kernels.Quant, rows, k, slots int) (*resident, [3]int, bool) {
	// which is in the key because gate and up have the same shape: without it
	// they would share one buffer and the gate matvec would read the up
	// weights (TestStreamedExpertBankMatchesTheResidentOne).
	key := resKey{block: -(2 + which), slot: slots, nrows: rows, k: k, t: q}
	if r, ok := g.res[key]; ok {
		pq, pd, psc, err := kernels.PackedWords(q, rows, k)
		if err != nil {
			return nil, [3]int{}, false
		}
		return r, [3]int{pq * 4, pd * 4, psc * 4}, r.ok
	}
	r, sh, ok := g.residentCompact(q, rows, k, slots)
	if !ok {
		return nil, sh, false
	}
	// Marked shared so the per-block free walk (dropTensors) leaves it alone;
	// Close frees it once.
	r.shared = true
	g.res[key] = r
	return r, sh, true
}

// residentCompact allocates a routed bank of exactly `slots` expert sheets and
// writes nothing into it: its contents are the router's answer for this token,
// which streamBank.fill supplies. k sheets back to back in one allocation are a
// bank of k experts, because the container packs a bank as independent
// matrices and PackedWords is linear in the row count
// (backend.TestIndexedMatVecMatchesACompactBank).
//
// It is not in g.res: a compact bank is a block's scratch, and sharing it
// through that map would hand a token another block's experts.
func (g *devTier) residentCompact(q kernels.Quant, rows, k, slots int) (*resident, [3]int, bool) {
	var sh [3]int
	pq, pd, psc, err := kernels.PackedWords(q, rows, k)
	if err != nil {
		g.LastErr = "PackedWords " + q.String() + ": " + err.Error()
		return nil, sh, false
	}
	sh = [3]int{pq * 4, pd * 4, psc * 4}
	// nrows is the whole compact bank, so resident.bytes() prices this exactly
	// as it prices the full one and the refund on release matches the charge.
	r := &resident{nrows: rows * slots, k: k, t: q}
	need := uint64(pq+pd+psc) * 4 * uint64(slots)
	if !g.room(need) {
		g.Declined++
		g.LastErr = fmt.Sprintf("%s %dx%d x %d sheets needs %d, %d of %d used",
			q, rows, k, slots, need, g.used, g.limit)
		return nil, sh, false
	}
	bufs := [3]backend.Buf{}
	for i, n := range sh {
		if n == 0 {
			// The kernel still takes the parameter; a format with no SC plane
			// (Q4_0, Q8_0) gets a word rather than a nil argument.
			n = 4
		} else {
			n *= slots
		}
		b, e := g.dev.Alloc(n)
		if e != nil {
			for _, x := range bufs {
				if x != nil {
					x.Free()
				}
			}
			g.NoRoom++
			return nil, sh, false
		}
		bufs[i] = b
	}
	r.qs, r.d, r.sc, r.ok = bufs[0], bufs[1], bufs[2], true
	g.charge(need)
	return r, sh, true
}
