// Package tier is jitllm's GPU decode tier: an nn.LayerDevice backed by the generated device kernels (engine/device/gpu).
//
// gpu.GPU is a router over one devTier per card, each with its own residency
// map, budget, kernels, staging buffers and lock. It places a model's blocks
// across them fastest device first, in contiguous runs, falling back to the
// host. gpu.Open is the one-device path; OpenWith and New are the
// several-device ones. See multi.go for which device, and the type comments in
// this file for which state is per device and which is shared.
//
// nn declares the device interface and this package implements it, so the
// dependency runs from here to nn and never the other way.
package gpu

import (
	"errors"
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/offheap"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/cost"
	hw "github.com/jitllm/jitllm/engine/probe"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

type resident struct {
	qs, d, sc backend.Buf
	nrows     int
	k         int
	t         layout.Quant
	ok        bool // false = declined, do not retry
	// why is the decline's reason, given again to whoever asks for the
	// declined tensor later, so a second offer names it too.
	why string
	// shared says one buffer serves several blocks: a streamed mixture's
	// compact expert bank, live only between a block's suspension and its own
	// FFN. users is the blocks holding it; the per-block free walk drops one,
	// and the last frees it and refunds its bytes (sharedCompact).
	shared bool
	users  int
	// imported says these three buffers alias the packed arena's host memory
	// instead of holding a copy (a unified device; see backend.HostImport). It
	// is what makes bytes() zero. pk is that arena entry, which the arena
	// keeps mapped until the buffers are freed (kernels.Arena.Unimport).
	imported bool
	pk       *Packed
}

// bytes is what this tensor occupies on the device, from its shape rather than
// from the buffers (backend.Buf does not report its own size).
//
// An imported tensor costs nothing here: its bytes are the arena's, counted
// once against host RAM (GPU.HostReserved), and charging them again would
// double-count them and decline blocks that cost nothing.
func (r *resident) bytes() uint64 {
	// A shared bank costs zero per block: sharedCompact charges its real bytes
	// once when it is made, and the last block to let it go refunds them.
	if r == nil || !r.ok || r.imported || r.shared {
		return 0
	}
	return r.planes()
}

// planes is the three buffers' bytes from the shape, whoever pays for them.
func (r *resident) planes() uint64 {
	qs, d, sc, err := layout.PackedWords(r.t, r.nrows, r.k)
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
	// e is the tensor of an entry at no block's slot (adhocKey), p its
	// bytes' address when it is no container entry; both nil for a
	// block-keyed entry.
	e        *jlm.Entry
	p        *byte
	nrows, k int
	t        layout.Quant
}

// adhocKey is the key of a tensor held at no block's slot -- the head, an
// input table, a picture's rows: its container entry, which is the same
// tensor whichever device's memory its bytes were handed in, so two
// instances' copies are one and no later mapping of that memory aliases it;
// the address of its bytes for a weight that is no entry.
func adhocKey(e *jlm.Entry, w []byte, q layout.Quant, nrows, k int) resKey {
	if e != nil {
		return resKey{block: -1, e: e, nrows: nrows, k: k, t: q}
	}
	return resKey{block: -1, p: &w[0], nrows: nrows, k: k, t: q}
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

// Config is the tier's shared settings. Every field is written by the caller
// (or once at Open) and read by every device under that device's lock.
type Config struct {
	// NoBatch has the caller send one prompt row a call (ChunkRows): the
	// way to tell a batching bug from an f32 tie, and the A/B control.
	NoBatch bool
	// NoTail pads every chunk to the full batch width instead of generating a
	// narrower kernel set for a ragged tail; the A/B arm for that.
	NoTail bool
	// ComposedFloat forms the served parts with float activations, no
	// int8 form: the reference a gate holds the int8 forms and the tier's
	// own kernels to.
	ComposedFloat bool
	// Profile is a card's measured row, its launch cost what the composer
	// weighs a fusion against; none keeps the composer's plain rules.
	Profile func(backend.Device) (cost.Profile, bool)
	// ComposedUnsplit forms the served parts with no sum cut into segments
	// (composer.Registry.WithoutSplits): a reference apart from the composer's
	// splits.
	ComposedUnsplit bool
	// ComposedShadow runs the composer's form of each block it would serve
	// beside the tier's own kernels, on a copy of the block's input and into
	// buffers of its own, and compares the two outputs on the device with a
	// reduction the composer forms (Stats.ShadowWorst). The tier's kernels
	// serve: the stream, the history and the logits are theirs whatever the
	// form computes, so the option changes no answer. It needs
	// nn.ComposedBlock, and works on a step issued or recorded alike.
	ComposedShadow bool
	// ComposedLanes, where not zero, is the subgroup the composer forms the
	// served parts for in place of the device's (pickLanes): a narrower one,
	// which a gate forms at to hold each lane count's forms to the tier. A
	// count the device does not hold whole -- not a power of two, or wider
	// than the lanes it guarantees -- is refused.
	ComposedLanes int
	// NoMMA forms no kernel on the device's matrix unit (composedMatrix). The
	// bisection switch for a wrong answer only the matrix unit shows.
	NoMMA bool
	// MatrixTile forces the composer's matrix tile on every prompt
	// projection, the profile's k'th (its own, then composer.Matrix.Tiles); 0
	// leaves it to the composer's model. The gates' selection switch.
	MatrixTile int
	// NoExactPrompt runs a prompt chunk in the batched scratch batchFor picks
	// (a power of two) rather than one built at the chunk's own grain width:
	// the A/B arm and the bisection switch.
	NoExactPrompt bool
	// ComposedWidths runs a prompt the composer serves as the tier's own
	// kernels run one, at its chunk's grain width, a form and a scratch for
	// each, in place of the reserved width with its rows the call's
	// (promptComposed): the violation arm of the gate that a placement
	// composes once.
	ComposedWidths bool
	// NoPromptPrewarm leaves the exact prompt widths' kernels to the first
	// prompt of each width (prewarmPromptWidths): the violation arm of the
	// gate that a placed model's prompts compile nothing.
	NoPromptPrewarm bool
	// Sessions is how many concurrent sessions each placed linear block
	// reserves a recurrent pair for; 0 and 1 are one. See devTier.reserved.
	// Without it the first instance fills the card and a second finds no room for
	// its own state, so its blocks run on the host. An attention block reserves
	// nothing per session: its history is paged, and a session takes pages as
	// it writes rows.
	Sessions int
	// SharedRec keeps the device's recurrent pools in the shared form even
	// when two halves would fit -- the other arm of that form's gates.
	SharedRec bool
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
	// this device -- the server's step loop sets engine.MaxStepRows. A device
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
	// NoAutoStream leaves a mixture block that no device can hold resident
	// on the host, as before GPU.AutoStream, instead of streaming it.
	NoAutoStream bool
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
	// NoGraph issues the token's launches one driver call at a time instead of
	// replaying a captured graph: the other arm of that comparison, a field so
	// both arms can be live in one process.
	NoGraph bool
	// NoTowerChunks refuses a picture whose whole non-causal set does not fit
	// rather than running it row-chunked (chunkTower, LayersPhase).
	NoTowerChunks bool
	// KVKeepWindow keeps a windowed layer's pages behind its window instead
	// of releasing them (devTier.windowPages): the other arm of that
	// comparison, which must give the same logits bit for bit.
	KVKeepWindow bool
	// KVCowFault is the shared-page gate's violation
	// (model.TestDecisionReusesThePrefix): a write into a page the prefix
	// cache names lands in it in place, under every sequence holding it.
	// False in every real run.
	KVCowFault bool
	// StageLimit bounds the host memory held by packs that are prepared and not
	// yet uploaded. Zero means defaultStageLimit. It is shared because the
	// memory is the host's: two devices under their own copy of the limit would
	// hold twice the bound. See stageRoom.
	StageLimit uint64

	// kb is this tier's measurement and bisection knobs (knobs.go), tiles the
	// batched prefill geometry resolved from them, and center the weight
	// centering its emitters are built with. Per tier, never package state:
	// two models in one process keep their own.
	kb    knobs
	tiles tiles
	// failAt and the per-device calls counter are always zero unless built with
	// -tags jitllmfault.
	failAt []int
	// panicAt is failAt for a panic inside a submission.
	panicAt []int
	// fault is the violations a jitllmfault build arms on this tier
	// (WithFaults); nil, none, and in every other build.
	fault *tierFaults
	// FillFirst places blocks on the first device until it is full, then the
	// next, instead of spreading a model that fits across the devices in
	// proportion to their budgets (GPU.PlanBlocks). The spread leaves each card
	// room for the batched prefill's scratch; fill-first leaves the first
	// card none, which is what a caller packing one card, or a gate needing a
	// full card, asks for.
	FillFirst bool

	// arena is the packed arena: every block's weights in the device layout,
	// host-resident, so a block paged back onto a card is a DMA rather than a
	// repack. It is shared by every device because the bytes are one heap. Its
	// retention budget is zero by default; see kernels.NewArena and SetArena.
	arena *Arena
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
	LastErr    string
	Declined   int // tensors refused because the VRAM budget was already spent
	// ConvBlocks is convolutional tower blocks run on the device.
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
	// KVFreeBytes is the part of KVPoolBytes a shrink gives back: the pages
	// beyond those in use and each layer's floor (shrinkKVLayer).
	KVFreeBytes uint64
	// BudgetUsed is what the device's budget has spent: weights, scratch and
	// the pool. SetBudget below it pages something out; at it, nothing grows.
	BudgetUsed uint64
	// ScratchBytes is the part of BudgetUsed the device's own buffers hold:
	// the block scratches and the staging (scratch.go).
	ScratchBytes uint64
	// ComposedBytes is the part of BudgetUsed the composed parts' instances
	// hold: their scratch, a prompt part's at its rows and key bound.
	ComposedBytes uint64
	// SharedBankBytes is the part of BudgetUsed the streamed mixtures' shared
	// compact banks hold, one a shape, while a placed block streams through it.
	SharedBankBytes uint64
	// ComposedArena is the part of ComposedBytes the prompt arenas hold: a
	// scratch's buffers every prompt form at its rows shares, the widest
	// each, which a block's move neither frees nor leaves behind.
	ComposedArena uint64
	// Allocated is what the device's buffers hold by the backend's own count
	// (backend.AllocCounter), 0 where it keeps none, and RoundingBytes the
	// part of BudgetUsed that is the driver's page rounding of them.
	Allocated, RoundingBytes uint64
	// ReservedPrompt and ReservedStep are the prompt chunk and the ragged
	// step the device reserved its scratch for at placement, 0 where it
	// reserved none. A ReservedPrompt narrower than Config.PromptRows is the
	// reservation halved to fit.
	ReservedPrompt, ReservedStep int
	// RowsSteps counts the ragged steps (LayersRows) the device ran and
	// RowsPadded the rows of padding they ran beside their own: a step whose
	// width was refused runs in the next wider scratch built, a 3-row
	// verification in the 512-row prompt scratch.
	RowsSteps, RowsPadded int
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
	// Steps is what the card served, part by part, and the steps it
	// declined, by reason (nn.Steps): the submissions emitted, not those a
	// recording replays (PartsReplayed).
	Steps nn.Steps
	// MatrixLaunches counts the launches of composed kernels formed on the
	// device's matrix unit (composer.Matrix), by part, emitted as Steps is:
	// what of a part ran on the unit, not what was formed for it.
	MatrixLaunches [nn.NumParts]int
	// AttendLaunches counts, as MatrixLaunches, the launches of a part's
	// attention formed on the matrix unit (composer.Flash).
	AttendLaunches [nn.NumParts]int
	// ComposedLanes is the subgroup the composer formed the served parts for
	// (Config.ComposedLanes, else the device's lanes): the selection check of
	// a gate that narrows it.
	ComposedLanes int
	// PartsReplayed counts the steps a recording replayed the composer's
	// parts in, by part: what Steps, counted where a step is emitted,
	// does not see.
	PartsReplayed [nn.NumParts]int
	// UnformedAtStep counts the parts a step asked for that no placement or
	// PrepHead formed, each declined unformed (servedFor): a step composes
	// nothing.
	UnformedAtStep int
	// UnformedAtStepParts is UnformedAtStep by part (FormedPart).
	UnformedAtStepParts [nn.NumParts]int
	// BlockForms counts, by block, the times a device formed the block's
	// served one-row part (servedFor): the composer's keyed form, compiled
	// once per Key and shared by blocks of one structure, with the block's
	// own bindings. A block is formed once on the device that holds it and
	// once more for each relocation; paging its weights out and in forms
	// nothing. Summed over the devices.
	BlockForms []int
	// PartsFormed counts the composer's forms the device compiled (a Key it
	// held no compiled form of), PartKernels their kernels' driver
	// compiles, and ScratchBuilds the batched scratches built: the census
	// of what a placement composes, which no prompt, step or session after
	// it moves.
	PartsFormed, PartKernels, ScratchBuilds int
	// FormAsks counts the forms asked of the device's form cache, a
	// compiled one among them: what a placement, a relocation or a lane
	// forming a part moves, and no warm step does.
	FormAsks int
	// FormErr is the first form or lowering the device did not build.
	FormErr error
	// ComposerRuns is the times the device's composer formed a request
	// rather than found it (composer.Registry.Formed), counted in the
	// composer whatever the caller marked the form as: a step, a prompt or
	// a session that moves it composed.
	ComposerRuns int
	// ReservedRows is the rows of the reserved width a device holds the
	// composer's prompt and ragged forms at, its buffers in the budget
	// (fitReserved): its capacity, a call of more rows running in chunks of
	// it. Over several devices, the narrowest.
	ReservedRows int
	// HiddenHeads counts the calls whose head the composer served and whose
	// normed rows the caller read back (nn.Head.Hidden), HiddenRows those
	// rows: a speculation step's verification is one call of several.
	HiddenHeads, HiddenRows int
	// ShadowWorst is, per block the composer's form ran beside the tier's own
	// kernels (Config.ComposedShadow), the worst drift of the form's output
	// from the tier's over the steps compared: the squared distance between
	// them over the squared distance the tier's block moved its input,
	// capped at graph.DriftCap. ShadowSteps counts the steps each block was
	// compared in, as the device counted them, a replayed recording's
	// included. Both are read from the device, from the blocks it holds when
	// the stats are taken: a block released or moved starts again.
	ShadowWorst []float32
	ShadowSteps []int
	// Forgotten counts lone-matvec copies dropped because the host said the
	// bytes at their address changed (Forget): the selection check that a
	// refilled frame reached the device at all.
	Forgotten int
	// RecShared counts recurrent pools sized in the shared form (recpool.go):
	// one resident state a slot and the device's next-state scratch.
	RecShared int
	// PersistRuns counts the collected runs of a block's launches the
	// persistent path issued (Config.Persist), composed or not;
	// PersistLaunches the persistent kernels launched, PersistParts the
	// per-op launches they ran, PersistPhases their
	// grid-barrier-separated phases, PersistCompiles the compositions built
	// and PersistFallbacks those that failed and ran per-op; PersistWhy is
	// why the path is declined on this device.
	// PersistDepLaunches counts those launched with dependency counters
	// (Config.PersistDeps), PersistDepTiles their tile-wise edges.
	PersistRuns, PersistLaunches, PersistParts, PersistPhases, PersistCompiles, PersistFallbacks int
	PersistDepLaunches, PersistDepTiles                                                          int
	PersistWhy                                                                                   string
	// RagHeadOne counts ragged steps whose head projected only the one row
	// that wanted logits, by decode's own projection or the composer's head at
	// one row, rather than a batched twin over every row (a prompt chunk's
	// last row, alone in wanting the head).
	RagHeadOne int
	// SampleReads counts tokens whose sampler ran on the device and read
	// back k candidates instead of the vocabulary (nn.Head.SampleK).
	SampleReads int
	// PickReads counts heads whose label pick ran on the device and read
	// back the picked logits instead of the vocabulary (nn.Head.Pick).
	PickReads int
	// MatrixTiled counts the composed parts formed at a forced matrix tile
	// (Config.MatrixTile).
	MatrixTiled int
	// KVPrewarmPages counts pages a pool grew ahead of its history
	// (PrewarmKV): the growth a instance's first prompt no longer pays.
	KVPrewarmPages int
	// StreamFills counts blocks whose routed bank was assembled from the
	// router's own selection this token, and StreamBlocks counts placements of
	// such a block, cumulative like Blocks and Captures. They are the selection
	// check: a streamed and a resident block compute the same logits.
	StreamFills int
	// NGramRows counts the n-gram table's rows read in for the card's
	// gathers (rowsRun): the selection check of an n-gram block on a card.
	NGramRows int
	// StreamRowFills counts the fills of a composed call of several rows:
	// every live row's selection filled at once.
	StreamRowFills int
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
	// CachesDropped counts expert caches given back for room (dropCaches),
	// and CacheBytes is what the caches sized after placement hold now.
	CachesDropped int
	CacheBytes    uint64
	// SplitAdmits counts the experts the caches took in.
	SplitAdmits int
	// SplitQueued counts admitted sheets sent on the copy queue beside the
	// launches (backend.Uploader) rather than in their order.
	SplitQueued int
	// TAdmit is the wall of taking experts into the caches.
	TAdmit time.Duration
	// StreamCacheSize is the largest expert cache a block was given, in
	// sheets: the check that an auto-sized cache is the size intended.
	StreamCacheSize int
	// StreamCacheBytes is the bytes the device's expert caches hold now,
	// beside their blocks' plain banks: copies of host pages, given back
	// first when history or a block needs the card (dropCaches).
	StreamCacheBytes uint64
	// StreamGroupsTuned is the group count the fill tuner holds (0: not
	// tuning, a stated Config.StreamGroups).
	StreamGroupsTuned int
	// StreamPinHalfTuned and StreamDirectTuned are the page-locked half and
	// the direct-send threshold the fill tuner holds, in bytes.
	StreamPinHalfTuned, StreamDirectTuned int
	// AutoMeanBase is the model's mean mixture base GPU.AutoStream was told.
	AutoMeanBase uint64
	// TPrewarm is packing done off the main loop, kept apart from TPack
	// because only TPack is on the critical path.
	TPrewarm time.Duration
	// Captures is how many times the launch sequence had to be recorded: one
	// per prompt plus one per scoreGrain tokens is the design, and one per
	// token means the key is moving and the recording is pure overhead.
	Captures int
	// KVEvictions counts pages sent to host memory (kvevict.go), one per layer-page.
	KVEvictions int
	// KVStageReads counts page reads the card refused a buffer for, which
	// was read back through the pool's reserve (readStaged).
	KVStageReads int
	// KVWindowReleased counts a windowed layer's pages released behind its
	// window (devTier.windowPages), one per layer-page.
	KVWindowReleased int
	// KVPagesMapped counts the named pages a prefix cache restore put into a
	// sequence's table on the card, no byte moved (kvshare.go), and
	// KVPagesCopied the named pages a write copied first.
	KVPagesMapped, KVPagesCopied int
	// KVPositionsMigrated counts the positions of history a migration moved
	// between the host and this device (pagedMigrate), each layer's apart.
	KVPositionsMigrated int
	// RecPointsKept and RecPointsLoaded count a linear block's restore
	// points copied into a snapshot on the card and back (recshare.go).
	RecPointsKept, RecPointsLoaded int
	// KVStreamPasses counts attention passes over evicted history.
	KVStreamPasses int
	// ComposedStreamPasses counts the passes of composed blocks over
	// evicted history (streamRun).
	ComposedStreamPasses int
	// PipelinePieces counts the pieces of pipelined prompt chunks this device
	// ran (GPU.pipeline).
	PipelinePieces int
	// StageOverlaps counts rows stages that began on this device while
	// another device ran a stage of another step (GPU.layersRows): the
	// selection check that steps led by different sessions -- the server's
	// decode groups -- kept several cards working at once rather than one
	// after another.
	StageOverlaps int
	// RowsComposedWalks counts the walks GPU.RowsComposed made over every
	// placed block: one a placement, not one an ask.
	RowsComposedWalks int
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
	// EmbedLaunches counts prompt chunks whose rows the device gathered from
	// the tied head (EmbedRows) -- the selection check that the host lookup
	// was actually skipped.
	EmbedLaunches int
	// EmbedReadBacks counts promised rows gathered in a submission of their
	// own and read back (materializeLocked): a head on another card than
	// block 0's hands its rows over this way.
	EmbedReadBacks int
	// HeadFolds counts batched chunks that ran the output projection on
	// their last row inside their own submission.
	HeadFolds int
	// TowerChunked counts non-causal sets built row-chunked (chunkTower), and
	// TowerPhases the phase calls run over them (LayersPhase).
	TowerChunked, TowerPhases int
	// PickRows counts label picks the head ran as the picked rows' own
	// projection (pickMV) instead of the whole head and a gather.
	PickRows int
	// ExactPrompts counts prompt chunks run in a scratch of their own grain
	// width rather than a wider one they pad.
	ExactPrompts int
	// PromptWidthsWarmed counts the prompt grain widths whose kernels were
	// built at the end of a placement (prewarmPromptWidths).
	PromptWidthsWarmed int
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
	TEmit time.Duration
	// Packs is PackWeights calls this device made outside the arena: every
	// pack when the arena is off and the ones it declined when it is on. Only
	// the sum with ArenaStats.Packs answers "did the pack run once"
	// (TestArenaPacksOncePerPageIn).
	Packs int

	// Prepacked is tensors that were already in the device layout (a converted
	// jlm container), and PrepackedBytes is what those uploads moved. Every
	// tensor upload is a pack or a prepack; a converted model must show Packs
	// at zero and this at every tensor, since the ids would be identical if
	// something were still rebuilding the layout.
	Prepacked      int
	PrepackedBytes uint64
	// RopeTableUploads is the rotary table rows taken from the caller rather
	// than computed on the device: a picture's rows, whose table is the
	// caller's of its grid.
	RopeTableUploads int

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
	// PageBytes is the device-resident bytes those page-ins brought back.
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
	if s.DeclineWhy == "" {
		s.DeclineWhy = o.DeclineWhy
	}
	if o.ComposedLanes > 0 && (s.ComposedLanes == 0 || o.ComposedLanes < s.ComposedLanes) {
		s.ComposedLanes = o.ComposedLanes
	}
	s.Steps.Add(&o.Steps)
	for i := range s.MatrixLaunches {
		s.MatrixLaunches[i] += o.MatrixLaunches[i]
		s.AttendLaunches[i] += o.AttendLaunches[i]
		s.PartsReplayed[i] += o.PartsReplayed[i]
	}
	s.UnformedAtStep += o.UnformedAtStep
	for i := range s.UnformedAtStepParts {
		s.UnformedAtStepParts[i] += o.UnformedAtStepParts[i]
	}
	if n := len(o.BlockForms); n > len(s.BlockForms) {
		s.BlockForms = append(s.BlockForms, make([]int, n-len(s.BlockForms))...)
	}
	for i, n := range o.BlockForms {
		s.BlockForms[i] += n
	}
	s.PartsFormed += o.PartsFormed
	s.FormAsks += o.FormAsks
	if s.FormErr == nil {
		s.FormErr = o.FormErr
	}
	s.ComposerRuns += o.ComposerRuns
	s.PartKernels += o.PartKernels
	s.ScratchBuilds += o.ScratchBuilds
	if o.ReservedRows > 0 && (s.ReservedRows == 0 || o.ReservedRows < s.ReservedRows) {
		s.ReservedRows = o.ReservedRows
	}
	s.HiddenHeads += o.HiddenHeads
	s.HiddenRows += o.HiddenRows
	// A block is on one device, so its slot is the only one that is not zero.
	if n := len(o.ShadowWorst); n > len(s.ShadowWorst) {
		s.ShadowWorst = append(s.ShadowWorst, make([]float32, n-len(s.ShadowWorst))...)
		s.ShadowSteps = append(s.ShadowSteps, make([]int, n-len(s.ShadowSteps))...)
	}
	for i, w := range o.ShadowWorst {
		s.ShadowWorst[i] = max(s.ShadowWorst[i], w)
		s.ShadowSteps[i] += o.ShadowSteps[i]
	}
	s.Forgotten += o.Forgotten
	if o.LastErr != "" {
		s.LastErr = o.LastErr
	}
	s.Declined += o.Declined
	s.ConvBlocks += o.ConvBlocks
	s.NoRoom += o.NoRoom
	s.SessionDeclines += o.SessionDeclines
	s.RopeTableUploads += o.RopeTableUploads
	s.Prepacked += o.Prepacked
	s.PrepackedBytes += o.PrepackedBytes
	s.KVBytes += o.KVBytes
	s.KVPoolBytes += o.KVPoolBytes
	s.KVFreeBytes += o.KVFreeBytes
	s.BudgetUsed += o.BudgetUsed
	s.ScratchBytes += o.ScratchBytes
	s.ComposedBytes += o.ComposedBytes
	s.ComposedArena += o.ComposedArena
	s.SharedBankBytes += o.SharedBankBytes
	s.Allocated += o.Allocated
	s.RoundingBytes += o.RoundingBytes
	s.ReservedPrompt = max(s.ReservedPrompt, o.ReservedPrompt)
	s.TowerRows = max(s.TowerRows, o.TowerRows)
	s.ReservedStep = max(s.ReservedStep, o.ReservedStep)
	s.RowsSteps += o.RowsSteps
	s.RowsComposedWalks += o.RowsComposedWalks
	s.RowsPadded += o.RowsPadded
	s.ScratchRefused += o.ScratchRefused
	s.PromptSplits += o.PromptSplits
	s.RecBytes += o.RecBytes
	s.KVGrows += o.KVGrows
	s.PersistRuns += o.PersistRuns
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
	s.RagHeadOne += o.RagHeadOne
	s.SampleReads += o.SampleReads
	s.PickReads += o.PickReads
	s.RecShared += o.RecShared
	s.MatrixTiled += o.MatrixTiled
	s.KVPrewarmPages += o.KVPrewarmPages
	s.StreamFills += o.StreamFills
	s.NGramRows += o.NGramRows
	s.StreamRowFills += o.StreamRowFills
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
	s.CachesDropped += o.CachesDropped
	s.CacheBytes += o.CacheBytes
	s.SplitAdmits += o.SplitAdmits
	s.SplitQueued += o.SplitQueued
	s.TAdmit += o.TAdmit
	s.StreamCacheSize = max(s.StreamCacheSize, o.StreamCacheSize)
	s.StreamCacheBytes += o.StreamCacheBytes
	s.StreamGroupsTuned = max(s.StreamGroupsTuned, o.StreamGroupsTuned)
	s.StreamPinHalfTuned = max(s.StreamPinHalfTuned, o.StreamPinHalfTuned)
	s.StreamDirectTuned = max(s.StreamDirectTuned, o.StreamDirectTuned)
	s.AutoMeanBase = max(s.AutoMeanBase, o.AutoMeanBase)
	s.TPrewarm += o.TPrewarm
	s.Captures += o.Captures
	s.KVEvictions += o.KVEvictions
	s.KVStageReads += o.KVStageReads
	s.KVWindowReleased += o.KVWindowReleased
	s.KVPagesMapped += o.KVPagesMapped
	s.KVPagesCopied += o.KVPagesCopied
	s.KVPositionsMigrated += o.KVPositionsMigrated
	s.RecPointsKept += o.RecPointsKept
	s.RecPointsLoaded += o.RecPointsLoaded
	s.KVStreamPasses += o.KVStreamPasses
	s.ComposedStreamPasses += o.ComposedStreamPasses
	s.PipelinePieces += o.PipelinePieces
	s.StageOverlaps += o.StageOverlaps
	s.SubsBeside += o.SubsBeside
	s.RanBeside += o.RanBeside
	s.LanesWaited += o.LanesWaited
	s.SessionRows += o.SessionRows
	s.SessionLinear += o.SessionLinear
	s.RecChainRows += o.RecChainRows
	s.RowsLatent += o.RowsLatent
	s.RecAligns += o.RecAligns
	s.RecCarryBytes += o.RecCarryBytes
	s.RecZeroBytes += o.RecZeroBytes
	s.RecZeroCopies += o.RecZeroCopies
	s.RecZeroSource += o.RecZeroSource
	s.RecZeroHost += o.RecZeroHost
	s.EmbedLaunches += o.EmbedLaunches
	s.EmbedReadBacks += o.EmbedReadBacks
	s.HeadFolds += o.HeadFolds
	s.TowerChunked += o.TowerChunked
	s.TowerPhases += o.TowerPhases
	s.PickRows += o.PickRows
	s.ExactPrompts += o.ExactPrompts
	s.PromptWidthsWarmed += o.PromptWidthsWarmed
	if o.Lanes > 0 && (s.Lanes == 0 || o.Lanes < s.Lanes) {
		s.Lanes = o.Lanes
	}
	if o.LanesWhy != "" {
		s.LanesWhy = o.LanesWhy
	}
	s.TLayer += o.TLayer
	s.TEmit += o.TEmit
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
	// marks is every sequence's rotary marks on the device (markArena).
	marks markArena
	// frames is the card's own frames over each file a block placed on it
	// is read from (framesOf). Under framesMu, not mu: closing a frames
	// tells the card what it gave back (forget), which takes mu.
	frames   cardFrames
	framesMu sync.Mutex
	// onDev is how many submissions are inside the backend's session on this
	// device right now, and ranBeside how many began while another was
	// (Stats.RanBeside); see runSub.
	onDev     atomic.Int32
	ranBeside atomic.Int64
	// layerGen counts blocks placed and released, so work that depends only
	// on which blocks are here can tell it is still current (prepBatch).
	layerGen uint64
	// placeGen moves with layerGen and whenever the reserved width, a
	// composed prompt instance or the head changes: what GPU.RowsComposed's
	// answer rests on, read without the lock.
	placeGen atomic.Uint64
	// instBytes is what the composed parts' instances charge
	// (Stats.ComposedBytes).
	instBytes uint64
	// arenaBytes is the part of instBytes the prompt arenas hold.
	arenaBytes uint64
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
	pin    [2][]byte
	noPin  bool
	pieces []sheetPiece
	// adm is admissions queued for flushAdmits (their items in admItems),
	// admPub those sent on the copy queue and not yet taken into use, upRing
	// the page-locked ring they go through and ringUsed its fill.
	adm      []admJob
	admPub   []admJob
	admItems []streamItem
	upRing   []byte
	ringUsed int
	packWG   sync.WaitGroup
	copyWG   sync.WaitGroup
	packEnd  int
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
	// subCost is the per-block run time of a submission by its row count's
	// class (subbudget.go). Guarded by mu.
	subCost map[int]time.Duration
	// vis is each non-causal set's sizing and shared k/v pair, by its key
	// (visrows.go): an encoder's, its decision head's and a tower's are each
	// their own. Guarded by mu.
	vis map[geoKey]*visState
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
	// geoUsed says a split plan was ever placed here (geomsets.go).
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
	recSeats map[uint64]recSeat
	// recSnaps are the prefix cache's restore points of the linear blocks,
	// by block and name, and recSnapOrder them oldest first (recshare.go).
	recSnaps map[recSnapKey]*recSnap
	// flushFn and flushErr are flushTabs' session body and its error, kept.
	flushFn      func(backend.Session)
	flushErr     error
	recSnapOrder []recSnapKey
	recSlots     int
	recShared    bool
	recS, recC   int
	// recNext is the one "next state" buffer the shared form writes, sized
	// for the widest step that needed it.
	recNext      backend.Buf
	recNextBytes uint64
	// snapList is the copies snapRec queued for flushSnaps, and snapCount
	// the points they keep: kept, so a snapshot allocates nothing.
	snapList  []backend.CopyRange
	snapCount int
	// zeroList is recZeroLayers' copies, run as one list.
	zeroList []backend.CopyRange
	// recZeros is the zeros a seat's zeroing copies from (recZeroLayers),
	// uploaded at the widest zeroing and kept while there are pools: a
	// prompt that resets its sequence's state made and freed it each time.
	recZeros      backend.Buf
	recZerosBytes uint64
	// recLs is resetRecRows' list of the blocks it zeroes, kept between
	// calls.
	recLs []*layer
	// headEntry is the container entry of the projection the resident head
	// is, headSrc the first byte it was uploaded from where it is no entry,
	// and headNormHost the norm it was built with: what a second instance's
	// PrepHead is compared against (adoptHead). altNorms are the other norms
	// that head serves (headnorm.go).
	headEntry    *jlm.Entry
	headSrc      *byte
	headNormHost []float32
	altNorms     []altNorm
	// busy is the device's call lock: a session's step takes it shared, so
	// steps of several sessions run at once, and a call that changes what
	// every step reads (placement, a release, a migration) takes it
	// exclusively (gpuSession.enter). See docs/design/device-sessions.md.
	busy sync.RWMutex
	// roomGen moves every time this device may have gained room: a refund, or
	// a budget raised. A caller that gave blocks up reads it to know when
	// asking for them back could succeed. See GPU.RoomGen.
	roomGen atomic.Uint64
	// cause is what kind of refusal the last call was (nn.FailCause), written
	// by the code that met it and cleared at the head of every call.
	cause atomic.Uint32
	// notComposedWhy is composedBlock's reason for each block and width it
	// found not the composer's, formatted once.
	notComposedWhy map[[2]int]string
	// stages is the rows stages running here, under the GPU's stageMu, and
	// stageMu is held by the one running (GPU.stageBegin).
	stages  int
	stageMu sync.Mutex
	dev     backend.Device
	pool    *Pool
	// ord is this device's place in the tier's order, fastest first, and part
	// of its label.
	ord int
	res map[resKey]*resident
	// actWin is the activation amax window the CPU tier chose; see
	// nn.JIT.SetDevice. Zero means the default 32. It is copied per device
	// rather than kept in Config because it is written after Open and read on
	// the serving path, so it lives under the lock that reads it.
	actWin int
	// recCap caches whether sessions can capture: 0 unknown, 1 yes, -1 no.
	recCap int
	// servedForms are the forms of the parts the composer serves, by the
	// composer's Key, every lane's (served.go). prewarming says a placement
	// or PrepHead is forming them (Stats.UnformedAtStep).
	servedForms map[composer.Key]*partForm
	prewarming  bool
	// copyGraphs are the copies over x by row width (servedCopy), each one
	// graph so its part keys alike every step.
	copyGraphs map[int]*graph.Graph
	// takeGraphs are the folded head's row takes by width (servedRowTake).
	takeGraphs map[int]*graph.Graph
	// blockForms is Stats.BlockForms, this device's.
	blockForms []int
	// everyRows is the word a served part whose call states no rows reads
	// (everyRow).
	everyRows backend.Buf
	composer  *composer.Registry
	// composerFloat is composer with no int8 form (Config.ComposedFloat),
	// and no split where Config.ComposedUnsplit.
	composerFloat *composer.Registry
	// composerPasses is composer for forms over a history read in passes
	// (composer.Registry.ForPasses).
	composerPasses *composer.Registry
	// halvesTo is where a history page's conversion launches (kvhalves.go),
	// and halvesGraphs its graphs by width and direction.
	halvesTo     launchTo
	halvesGraphs map[halvesKey]*graph.Graph
	// embTo is the composer's gather made real on its own
	// (materializeComposed).
	embTo launchTo
	embRd embRead
	// rows is the picture rows apart from their blocks (picrows.go).
	rows      rowsState
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
	// lastSub is when the device last started or finished a submission
	// (unix ns): the background tuner waits for tuneQuiet past it.
	lastSub atomic.Int64
	// gatherCopies is a page gather's copy list (gatherInto), reused.
	gatherCopies []backend.CopyRange
	pcache       *persistCache // composed persistent kernels; see persist.go

	// Host-side scratch, reused rather than allocated per matvec.
	hostRaw []byte
	hostOut []float32 // a view over hostRaw, for the readback

	stage      map[resKey]*staged
	stageBytes uint64
	// calls is the fault-injection counter; always zero unless built with
	// -tags jitllmfault.
	calls int
	// subCalls counts the submissions for panicAt.
	subCalls int

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
	// the placement's choice (engine.Place.Stream), not a device-wide switch.
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
	// other non-pageable allocation.
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
	// matrixCap is composedMatrix's probe, in tileCap's terms, scaleCap
	// composedScale's.
	matrixCap, scaleCap int8
	// launch is launchWords' answer, asked once where launchSet.
	launch    int
	launchSet bool
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
	// holdsOf counts the instances holding each model on the tier (HoldModel).
	holdsOf map[uint64]int
	// sessions is the last session id Attach gave out; 0 is the zero session.
	sessions atomic.Uint64
	// devs is every device, in the order they were opened (AddDevices
	// appends).
	devs []*devTier
	// counters is what the devices count into; see Counters.
	counters *backend.Counters
	// rowsComp is RowsComposed's last answer and what it rested on: the
	// devices' placement generations, their count and the head's device.
	// Under mu, as is rowsWalks, the walks it made (Stats.RowsComposedWalks).
	rowsComp  rowsComposedAt
	rowsWalks int
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
	// geo is a block's scratch set (geomsets.go: Gemma 4's attention geometry
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
	// rows is the zero session's rows call (rowsCall); a session's is its
	// own, so steps led by different sessions share none of it.
	rows rowsCall
	// stages is how many rows stages are running on every device, each
	// device's own count beside it (devShared.stages), under stageMu: a stage
	// that starts while another device runs one is another call's, and counts
	// as an overlap (Stats.StageOverlaps). rowsCalls is how many rows calls
	// are in flight.
	stageMu   sync.Mutex
	stages    int
	rowsCalls atomic.Int32
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
	cfg := &Config{
		kb:      o.kb,
		tiles:   tilesFor(o.kb),
		failAt:  faultAt(),
		panicAt: faultPanicAt(),
		fault:   o.faults,
		stage:   NewPool("host stage", 0),
		// Always built, budgeted at zero: the structure exists from load, and
		// the budget decides whether anything is retained. At zero (the
		// default) Pack declines immediately.
		arena: NewArena(o.kb.arena),
	}
	cfg.Profile = func(d backend.Device) (cost.Profile, bool) { return cost.Known(hw.GPUName(d), d.API()) }
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
	g := &GPU{Config: cfg, own: map[int]*devTier{}, geo: map[int]geoKey{}, counters: o.counters}
	if g.counters == nil {
		g.counters = slotCounters(slots)
	}
	for i, s := range slots {
		if s.Dev == nil {
			closeAll()
			return nil, fmt.Errorf("tier: device %d is nil", i)
		}
		g.addSlot(i, s)
	}
	return g, nil
}

// addSlot builds the tier's device for slot s at ordinal i and pools it.
func (g *GPU) addSlot(i int, s Slot) {
	cfg := g.Config
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
		Config: cfg,
		tot:    Stats{Device: label(s.Dev, i)},
		dev:    s.Dev,
		cnt:    cnt,
		pool:   pool,
		why:    why,
		ord:    i,
		name:   DeviceName(s.Name),
		res:    map[resKey]*resident{},
		limit:  limit,
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

// label names a device the way a report should: its ordinal in the tier, the
// hardware and the API that reached it, so two identical cards are
// distinguishable in a counter dump.
func label(d backend.Device, ord int) string {
	return fmt.Sprintf("%d:%s [%s]", ord, d.Name(), d.API())
}

// Devices is how many devices this tier holds.
func (g *GPU) Devices() int { return len(g.devs) }

// Counters is what this tier's devices count into: compiles, readbacks to the
// host, refused allocations, the driver objects held, device-clock timing (see
// backend.Counters).
func (g *GPU) Counters() *backend.Counters { return g.counters }

// slotCounters is what devices a caller opened count into: the first device's
// Counters, or new ones when none of them is a backend device.
func slotCounters(slots []Slot) *backend.Counters {
	for _, s := range slots {
		if s.Dev == nil {
			continue
		}
		if c := backend.CountersOf(s.Dev); c != nil {
			return c
		}
	}
	return &backend.Counters{}
}

// Stats sums every device's counters. See Stats.add for the fields that do not
// sum.
func (g *GPU) Stats() Stats {
	var s Stats
	for _, d := range g.devs {
		s.add(d.stats())
	}
	g.mu.Lock()
	s.RowsComposedWalks = g.rowsWalks
	g.mu.Unlock()
	return s
}

// LaneParts is, device by device, how many blocks and heads each lane holds
// an instance of a part of, lane0 first (a head's widths count once). Every lane holds every one the device
// serves: a lane short of one forms it at a step, and one holding more keeps
// a released block's instance charged. The device's own parts (row copies,
// tables, tokens) are not counted.
func (g *GPU) LaneParts() [][]int {
	var out [][]int
	for _, d := range g.devs {
		d.mu.Lock()
		var n []int
		d.eachLane(func(l *lane) {
			seen := map[*graph.Graph]bool{}
			for sk, sv := range l.served {
				if sv != nil && (sk.kind == nn.PartBlock || sk.kind == nn.PartHead) {
					seen[sk.graph] = true
				}
			}
			n = append(n, len(seen))
		})
		d.mu.Unlock()
		out = append(out, n)
	}
	return out
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
	for _, l := range d.layers {
		if l == nil || l.stream == nil || !l.stream.cached() {
			continue
		}
		for m, r := range [3]*resident{l.gate, l.up, l.down} {
			if r != nil && l.stream.plain[m] != nil {
				s.StreamCacheBytes += r.bytes()
			}
		}
	}
	if d.composer != nil {
		s.ComposerRuns = int(d.composer.Formed())
	}
	if d.kvp != nil {
		for _, l := range d.kvp.layers {
			for _, ids := range l.owned {
				s.KVBytes += uint64(len(residentIDs(nil, ids))) * d.kvp.pageBytes(l)
			}
		}
		s.KVPoolBytes = d.kvp.charged()
		for _, l := range d.kvp.layers {
			live := 1
			for _, ids := range l.owned {
				for _, id := range ids {
					if id != 0 {
						live++
					}
				}
			}
			if spare := l.n - max(live, l.floor, 1); spare > 0 {
				s.KVFreeBytes += uint64(spare) * d.kvp.pageBytes(l)
			}
		}
	}
	s.BudgetUsed = d.used
	s.ComposedBytes, s.ComposedArena = d.instBytes, d.arenaBytes
	for _, r := range d.res {
		if r.shared && r.ok {
			s.SharedBankBytes += r.planes()
		}
	}
	for _, l := range d.layers {
		if l != nil && l.stream != nil && l.stream.cached() && l.stream.plain[2] != nil {
			for _, r := range [3]*resident{l.gate, l.up, l.down} {
				if r != nil {
					s.CacheBytes += r.bytes()
				}
			}
		}
	}
	s.ScratchBytes, s.ReservedPrompt, s.ReservedStep = d.scratch, d.promptW, d.stepW
	// Where the composer runs a ragged step in the reserved width, the step's
	// reservation is that width, its per-row logits with it (reserveScratch).
	if bb := d.bbs[d.promptW]; d.stepW == 0 && d.StepRows > 0 && bb != nil && bb.ragLogits != nil {
		s.ReservedStep = d.promptW
	}
	s.TowerRows = d.towerVis().cap
	s.Allocated, s.RoundingBytes = uint64(d.allocated()), d.rounding
	s.BlockForms = append([]int(nil), d.blockForms...)
	if d.shadowing() {
		d.shadowStats(&s)
	}
	return s
}

// Pools reports every distinct budget the tier spends from.
func (g *GPU) Pools() []*Pool { return g.pools }

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
	bs.arena.free()
	bs.arena = nil
	// Both slices may hold nil where an allocation failed part-way, which is
	// how a card too small for a tower's planes reaches here.
	for _, b := range bs.qoffs {
		if b != nil {
			b.Free()
		}
	}
	bs.qoffs = nil
	freePaged(bs.pkv)
	seen := map[backend.Buf]bool{}
	v := reflect.ValueOf(bs).Elem()
	bufType := reflect.TypeOf((*backend.Buf)(nil)).Elem()
	for i := 0; i < v.NumField(); i++ {
		f := v.Field(i)
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

// startCall zeroes session sid's count and cause here at the head of its call
// (GPU.startRows).
func (g *devTier) startCall(sid uint64) {
	g.mu.Lock()
	ds := g.sessOf(sid)
	ds.recSteps = 0
	ds.sessCause.Store(0)
	g.mu.Unlock()
}

// setRecSteps sets session sid's count; see GPU.resetRecSteps.
func (g *devTier) setRecSteps(sid uint64, n int) {
	g.mu.Lock()
	g.sessOf(sid).recSteps = n
	g.mu.Unlock()
}

// setInputs hands session sid's per-layer inputs to this device for its
// next call (gpuSession.inputsTo).
func (g *devTier) setInputs(sid uint64, ple []float32, prompt bool, toks []int32, req []uint32, cut []float32) {
	g.mu.Lock()
	ds := g.sessOf(sid)
	ds.pleHost, ds.prompt, ds.reqHost, ds.cutHost = ple, prompt, req, cut
	ds.stepToks = append(ds.stepToks[:0], toks...)
	g.mu.Unlock()
}

func (g *devTier) Close() {
	// The layers go first, through ReleaseLayers (the refunds, freeRaw, the
	// decline tombstones, freeStale) rather than a second copy of that walk.
	// A driver that reclaims the context on close would hide a leak here, but
	// a unified-memory device would not. It takes g.mu, hence before the lock
	// below.
	g.releaseEveryLayer()
	g.closeFrames()

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
	g.partsClose()
	g.freeMarks()
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
		if g.upRing != nil {
			hp.UnpinHost(g.upRing)
			g.upRing = nil
		}
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
	// Every layer left the pool with its block; what remains is the pool
	// itself and its charge.
	g.freeKVPool(g.kvp)
	g.kvp = nil
	if g.argmaxOut != nil {
		g.argmaxOut.Free()
	}
	g.lane0.freeSample()
	g.lane0.freePick()
	g.closeRows()
	g.freeAltNorms()
	if g.recNext != nil {
		g.recNext.Free()
	}
	g.dropGraph()
	// The block scratches, which nothing else frees (see freeScratch), and
	// the staging, refunded as they go.
	g.dropAllScratch()
	g.closePersistCache()
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

// quantOf forwards to layout.QuantOf rather than keeping a second per-format
// list, which went stale when a format was added. A float weight (F32, F16,
// BF16) is served too; the device reads its raw rows (layout.FloatOf).
func quantOf(t quant.Type) (layout.Quant, bool) {
	if q, ok := layout.FloatOf(t); ok {
		return q, true
	}
	return layout.QuantOf(t)
}

// importing reports that a page-in on this device is a descriptor update rather
// than a transfer: the heap is the host's, the driver will wrap a host pointer,
// and the arena has a budget to hold the packs that get wrapped (without one
// the arena declines every pack and there is nothing to wrap). The device's
// alignment must also be one the arena provides, or the driver would refuse
// every array one call at a time.
func (g *devTier) importing() bool {
	return g.imp != nil && g.impAlign > 0 && g.impAlign <= offheap.HostAlign() &&
		g.arena.Limit() > 0
}

// importPacked wraps an arena entry's three arrays as device buffers that alias
// them, and reports whether all three came back.
//
// The length is the padded one: drivers require the imported size, not just
// the address, to be a multiple of the alignment (len(v)*4 is a refusal on
// Vulkan and a silent nil on Metal). The arena pads every array inside its own
// region. A partial import is unwound rather than kept.
func (g *devTier) importPacked(r *resident, p *Packed) bool {
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
		b, err := g.imp.Import(Ptr(v), int(PaddedBytes(len(v))))
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
// (the output projection, an input table, a picture's rows), keyed on e, its
// container entry (adhocKey). block is also a tag telling the arena which
// entries to release together.
func (g *devTier) resident(block, slot int, e *jlm.Entry, q layout.Quant, w []byte, nrows, k int, pre *nn.Packed) *resident {
	g.applyForgets()
	key := resKey{block: block, slot: slot, nrows: nrows, k: k, t: q}
	if block < 0 {
		key = adhocKey(e, w, q, nrows, k)
	}
	if r, ok := g.res[key]; ok {
		// The shape is part of the key because an expert bank and its expert 0
		// begin at the same byte: both shapes are legitimate and both may be
		// resident. Sharing one entry would launch a kernel against a
		// differently strided payload (a silent out-of-bounds read), and
		// refusing on a mismatch would stop a bank from ever being placed.
		if !r.ok && r.why != "" {
			g.LastErr = r.why
		}
		return r
	}
	r := &resident{nrows: nrows, k: k, t: q}
	g.res[key] = r
	defer func() {
		if !r.ok {
			r.why = g.LastErr
		}
	}()

	// Price it before packing it: declining after a pack costs a full read and
	// fresh host memory to learn the answer is no. PackedWords is the same
	// arithmetic with no allocation (TestPackedWordsMatchesPack).
	pq, pd, psc, err := layout.PackedWords(q, nrows, k)
	if err != nil {
		g.LastErr = "PackedWords " + q.String() + ": " + err.Error()
		return r
	}
	need := uint64(pq+pd+psc) * 4
	// Admitted at what the card spends on the three buffers, the driver's
	// page rounding included, as a block is (blockBytes): a lone matvec
	// admitted at its bytes alone put the card past its budget by the
	// rounding.
	foot := g.planeFootprint(pq) + g.planeFootprint(pd) + g.planeFootprint(psc)

	// A bank in expert pages is assembled on the device, one sheet at a time:
	// there is no host span of the whole bank, and gathering one would hold a
	// layer's every expert in host memory at once.
	if pre != nil && pre.Sheet != nil {
		if !g.room(foot) {
			g.Declined++
			g.LastErr = fmt.Sprintf("%s %dx%d needs %d, %d of %d used", q, nrows, k, foot, g.used, g.limit)
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
	var pk *Packed
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
	if !g.room(foot) {
		g.Declined++
		g.LastErr = fmt.Sprintf("%s %dx%d needs %d, %d of %d used (pool %s: %d of %d)",
			q, nrows, k, foot, g.used, g.limit, g.pool.Name(), g.pool.Used(), g.pool.Limit())
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

	// The arena: a tensor it holds is a map lookup and a DMA of bytes
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
		qs, dw, scw, err = layout.PackWeights(q, w, nrows, k)
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
func (g *devTier) residentSheets(r *resident, q layout.Quant, nrows, k int, pre *nn.Packed) bool {
	if pre.Sheets <= 0 || nrows%pre.Sheets != 0 {
		g.LastErr = fmt.Sprintf("tier: %d rows do not split into %d sheets", nrows, pre.Sheets)
		return false
	}
	sq, sd, ssc, err := layout.PackedWords(q, nrows/pre.Sheets, k)
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

// sizeOut grows the readback buffer. nrows is baked into the kernel, so there
// is nothing to publish to the device.
func (g *devTier) sizeOut(nrows int) bool {
	if cap(g.hostRaw) < nrows*4 {
		g.hostRaw = make([]byte, nrows*4)
		g.hostOut = unsafe.Slice((*float32)(unsafe.Pointer(&g.hostRaw[0])), cap(g.hostRaw)/4)
	}
	return g.regrow(&g.outBuf, &g.outCap, nrows, 4, false, false)
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
	t         layout.Quant
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
func (g *devTier) prewarm(block int, q layout.Quant, w []byte, nrows, k int) bool {
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
	qs, d, sc, err := layout.PackWeights(q, w, nrows, k)
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
func (g *devTier) sharedCompact(which int, q layout.Quant, rows, k, slots int) (*resident, [3]int, bool) {
	// which is in the key because gate and up have the same shape: without it
	// they would share one buffer and the gate matvec would read the up
	// weights (TestStreamedExpertBankMatchesTheResidentOne).
	key := resKey{block: -(2 + which), slot: slots, nrows: rows, k: k, t: q}
	if r, ok := g.res[key]; ok {
		pq, pd, psc, err := layout.PackedWords(q, rows, k)
		if err != nil {
			return nil, [3]int{}, false
		}
		if r.ok {
			r.users++
		}
		return r, [3]int{pq * 4, pd * 4, psc * 4}, r.ok
	}
	r, sh, ok := g.residentCompact(q, rows, k, slots)
	if !ok {
		return nil, sh, false
	}
	// Marked shared so the per-block free walk (dropTensors) only lets go of
	// it: the last block to do so frees it.
	r.shared, r.users = true, 1
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
func (g *devTier) residentCompact(q layout.Quant, rows, k, slots int) (*resident, [3]int, bool) {
	var sh [3]int
	pq, pd, psc, err := layout.PackedWords(q, rows, k)
	if err != nil {
		g.LastErr = "PackedWords " + q.String() + ": " + err.Error()
		return nil, sh, false
	}
	sh = [3]int{pq * 4, pd * 4, psc * 4}
	// nrows is the whole compact bank, so resident.bytes() prices this exactly
	// as it prices the full one and the refund on release matches the charge.
	r := &resident{nrows: rows * slots, k: k, t: q}
	need := uint64(pq+pd+psc) * 4 * uint64(slots)
	// Admitted at what the card spends on the three buffers, as resident is.
	foot := g.planeFootprint(pq*slots) + g.planeFootprint(pd*slots) + g.planeFootprint(psc*slots)
	if !g.room(foot) {
		g.Declined++
		g.LastErr = fmt.Sprintf("%s %dx%d x %d sheets needs %d, %d of %d used",
			q, rows, k, slots, foot, g.used, g.limit)
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
