// Package session is the front-end-neutral state of a jitllm session: the
// transcript, the loaded models and where they are, the hardware report and
// the requests a front end hands the engine. Both front ends, the window and
// the terminal, read and write these types; nothing here draws.
package session

// Role is who produced a turn.
type Role uint8

// The turn roles.
const (
	RoleUser Role = iota
	RoleAssistant
	RoleSystem
	// RoleError is an engine failure rendered in the transcript rather than in
	// a dialog, because a failed generation belongs beside the prompt that
	// caused it.
	RoleError
)

// Turn is one message in the conversation.
type Turn struct {
	Role Role
	Text string
	// Images are the pictures a user turn carries, as paths. They go to the
	// model before Text -- the order every VLM template here was trained with
	// -- and the transcript draws them the same way.
	Images []string
	// Model is which model produced this turn, and Colour indexes the
	// palette it is drawn in; the model can change mid-conversation.
	Model  string
	Colour int

	// Think is the model's reasoning block, already separated from Text so
	// the two can be shown differently.
	Think string
	// Tokens is how many tokens the turn cost. Zero for a user turn.
	Tokens int
	// TokPerSec is the decode rate this turn achieved, for the per-turn
	// footer. Zero means it was not measured.
	TokPerSec float64
}

// MachineReport is the hardware probe, taken once at startup. It is cached
// because backend.Open opens and closes every backend, and re-probing while a
// gpu.GPU is live contends for the same devices.
type MachineReport struct {
	Probed bool
	Err    string

	Arch        string
	PCores      int
	ECores      int
	SMTSiblings int
	DecodeCores int

	MemBudget   uint64
	MemLimit    uint64
	GCBudgetCap uint64
	// MemWall is measured memory read bandwidth, bytes a second; 0 unprobed.
	MemWall float64

	// Kernels is one row per quantization type.
	Kernels []KernelRow
	GPUs    []GPUInfo
}

// KernelRow says which kernel families exist for one quantization type on this
// host. Both columns are shown: on a pre-VNNI host the row-major column reads
// "none" for every k-quant while the packed family, which containers use, is
// fully generated.
type KernelRow struct {
	Format   string
	RowMajor bool
	// Packed is the container path.
	Packed bool
}

// GPUInfo is one device the backend probe found.
type GPUInfo struct {
	Name string
	API  string
	// Slots is how many kernel slots the device reports.
	Slots int
	// Mem is the device-local heap. Unified says whether it is host memory,
	// in which case a scheduler must subtract it from the host budget, never
	// add it.
	Mem     uint64
	Unified bool
}

// PagerStat is the container pager's counters. Reads sits beside BytesRead
// because the pager is priced per request, not per byte.
type PagerStat struct {
	// TurnOuts is the evictions during the last turn. PageOuts only grows, so
	// it cannot say whether the model is paging now; see sessionPagerLine.
	TurnOuts int64

	// Frames is how many block pages are held, and Resident what they cost in
	// bytes; page size varies per model, so the count alone is not a quantity.
	Frames     int
	Resident   uint64
	PageIns    int64
	PageOuts   int64
	BytesRead  uint64
	Reads      int64
	ChunkBytes uint64
}

// DeviceUse is one open device's share of a loaded model, in bytes against a
// capacity rather than a page count, since page size varies per model.
type DeviceUse struct {
	// Name is the tier's own label, "0:<device name> [cuda]".
	Name string
	// Blocks is how many blocks this device holds, not how many it runs.
	Blocks int
	// Used and Limit are the pool's, in bytes.
	Used, Limit uint64
	// Pool is the heap's name; Used is per heap, not per device. Two devices
	// can share one pool (one card under CUDA and Vulkan, or any unified
	// device and the host), so Used must never be summed across rows.
	Pool string
	// Shared marks a device whose memory is host memory (an integrated GPU or
	// Apple Silicon): subtract it from the host, never add it.
	Shared bool
	// KV is the attention history this device holds, which is charged
	// separately from the weights.
	KV uint64
	// KVPool is the paged history pool's charge against the device, free pages
	// included; KV is what sequences hold of it.
	KVPool uint64
	// Weights is the block bytes charged to this device's weight budget.
	Weights uint64
	// Short is the name a device spec uses for it: cuda:0, vulkan:1.
	Short string
}

// Loading is a model being opened or switched to: which, and the stage it is
// at, in words. The zero value is nothing loading.
type Loading struct {
	Path  string
	Stage string
}

// Allocation is where a loaded model actually is.
type Allocation struct {
	Devices []DeviceUse
	// HostUsed is block pages resident in host memory, in bytes, against
	// HostBudget. Zero budget means unlimited.
	HostUsed, HostBudget uint64
	// Dense is the weights that never page -- the embedding and the output
	// projection -- which are read on every token and are not part of the page
	// budget.
	Dense uint64
	// HostBlocks is blocks running on the host, DeviceBlocks those on a
	// device, and the two need not add to the model's depth while a block is
	// in flight.
	HostBlocks, DeviceBlocks, NBlocks int
	// PageBytes is one block page; the file holds NBlocks of them plus Dense.
	PageBytes uint64
	// HostKV is the attention history the host holds for the blocks it runs.
	HostKV uint64
	// Pos is how many positions the session holds, of a MaxSeq window.
	Pos, MaxSeq int
}

// LoadedModel is one open model, as the UI sees it. Several are open and one
// is active; the others keep their weights and pager and give up their
// session, device and worker pool.
type LoadedModel struct {
	Path string
	Name string
	Arch string
	// Blocks is the model's depth, which is what a placement is a fraction of.
	Blocks int
	// Colour indexes the transcript's palette by load order, so colours are
	// stable within a session and start at the beginning of the palette.
	Colour int
	// Grant is the host page budget this model currently holds, and Pinned
	// says the user fixed it rather than the engine dividing it.
	Grant  uint64
	Pinned bool
	Active bool
}

// Problem is something that went wrong, said so a person can act on it: what
// happened, why, and -- where there is one -- the fix as a button. It is a
// banner rather than a dialog because core/dialog holds two lines (ui/UPSTREAM.md
// #13); it stays until dismissed or replaced.
type Problem struct {
	// Seq is the trigger, as DialogReq's is: 0 means there is no problem.
	Seq    uint64
	Title  string
	Detail string
	// Action is a button label, empty for none; Do runs on the UI goroutine.
	Action string
	Do     func()
}
