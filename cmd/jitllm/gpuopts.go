package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// gpuOptions turns the device tier's environment variables into tier.Options.
//
// They are read here and nowhere else: the device packages read no environment,
// so an embedded caller can configure each tier through options.
//
// JITLLM_DEVICE_FAIL_AT is deliberately not here: it is read by
// jit/gpu/tier/fault.go, compiled only under `-tags jitllmfault`, so no release
// binary can be made to pretend its GPU broke.
func gpuOptions() []tier.Option {
	var o []tier.Option
	str := func(k string) string { return os.Getenv(k) }
	num := func(k string) int {
		n, _ := strconv.Atoi(os.Getenv(k))
		return n
	}
	set := func(k string) bool { return os.Getenv(k) != "" }

	// The tuners' answers persist under the user cache directory, keyed by
	// device UUID, driver and executable (tier.WithTuneCache).
	if d, err := os.UserCacheDir(); err == nil {
		// A shape not in it runs the table's split while the tuner measures
		// it in the background (tier.WithLazyTune).
		o = append(o, tier.WithTuneCache(filepath.Join(d, "jitllm")), tier.WithLazyTune(true))
	}
	if v := str("JITLLM_GPU_API"); v != "" {
		o = append(o, tier.WithAPI(v))
	}
	switch str("JITLLM_GPU_TUNE") {
	case "0":
		o = append(o, tier.WithDeviceTune(tier.TuneOff))
	case "1":
		o = append(o, tier.WithDeviceTune(tier.TuneForce))
	}
	// A register cap on every CUDA kernel: the register report's what-if.
	if n := num("JITLLM_CUDA_MAXREGS"); n > 0 {
		o = append(o, tier.WithCUDAMaxRegs(n))
	}
	// The batched GEMM's spilling tiles, refused by default (tier.Config.SpillingTiles).
	if str("JITLLM_GPU_SPILLING_TILES") == "1" {
		o = append(o, tier.WithConfig(func(c *tier.Config) { c.SpillingTiles = true }))
	}
	if set("JITLLM_GPU_VERBOSE") {
		o = append(o, tier.WithVerbose(true))
	}
	if v := rocmDir(); v != "" {
		o = append(o, tier.WithROCm(v))
	}
	if v := str("JITLLM_VK_DEVICE"); v != "" {
		o = append(o, tier.WithVulkanDevice(v))
	}
	if v := str("JITLLM_VK_SUBGROUP"); v != "" {
		o = append(o, tier.WithSubgroup(v))
	}
	if n := num("JITLLM_VK_MAXGROUPS"); n >= 1 {
		o = append(o, tier.WithVulkanMaxGroups(uint32(n)))
	}
	if set("JITLLM_NO_CENTER") || num("JITLLM_CENTER_MINTOK") >= 1 {
		o = append(o, tier.WithCentering(set("JITLLM_NO_CENTER"), num("JITLLM_CENTER_MINTOK")))
	}
	if n := num("JITLLM_GPU_SPLIT"); n >= 1 {
		o = append(o, tier.WithSplit(n))
	}
	if n := num("JITLLM_GPU_ACCSPLIT"); n >= 1 {
		o = append(o, tier.WithAttnAccSplit(n))
	}
	if n := num("JITLLM_ACC_SPLIT"); n >= 1 {
		o = append(o, tier.WithMatVecAccSplit(n))
	}
	if n := num("JITLLM_GPU_LANES"); n >= 1 {
		o = append(o, tier.WithLanes(n))
	}
	if set("JITLLM_GPU_POISON_KV") {
		o = append(o, tier.WithPoisonKV(true))
	}
	if set("JITLLM_GPU_NO_ATTNMMA") {
		o = append(o, tier.WithoutAttnMMA(true))
	}
	var skip []string
	if v := str("JITLLM_GPU_SKIP"); v != "" {
		skip = append(skip, strings.Split(v, ",")...)
	}
	if set("JITLLM_GPU_NOATTN") {
		skip = append(skip, "attn")
	}
	if set("JITLLM_GPU_NOFFN") {
		skip = append(skip, "ffn")
	}
	if len(skip) > 0 {
		o = append(o, tier.WithSkip(skip...))
	}
	b := tier.BatchShape{
		Tok:    num("JITLLM_GPU_TOK"),
		RowT:   num("JITLLM_GPU_ROWT"),
		QTile:  num("JITLLM_GPU_QTILE"),
		KTile:  num("JITLLM_GPU_KTILE"),
		ATile:  num("JITLLM_GPU_ATILE"),
		Parts:  num("JITLLM_GPU_PARTS"),
		AttnNT: num("JITLLM_GPU_ATTNNT"),
		// The ragged decode step's matvec where there is no matrix
		// instruction (jitllm batch); see tier.BatchShape.
		RagTok:   num("JITLLM_GPU_RAGTOK"),
		RagRowT:  num("JITLLM_GPU_RAGROWT"),
		RagSplit: num("JITLLM_GPU_RAGSPLIT"),
		Split:    num("JITLLM_GPU_BSPLIT"),
	}
	if b != (tier.BatchShape{}) {
		o = append(o, tier.WithBatchShape(b))
	}
	if v := str("JITLLM_GPU_MMATILE"); v != "" {
		var mt, nt int
		if _, err := fmt.Sscanf(v, "%dx%d", &mt, &nt); err == nil && mt > 0 && nt > 0 {
			o = append(o, tier.WithMMATile(mt, nt))
		}
	}
	// Zero is a value here -- it restores the unpriced behaviour -- so the
	// presence of the name is what decides, not the number.
	if set("JITLLM_MV_MIN_KB") {
		if n := num("JITLLM_MV_MIN_KB"); n >= 0 {
			o = append(o, tier.WithMinMatVecBytes(uint64(n)<<10))
		}
	}
	// JITLLM_METAL_SPIN_US is how long a Metal Wait polls a command buffer
	// before it blocks (0 blocks at once); JITLLM_METAL_FASTMATH=1 compiles
	// with Metal's fast math; JITLLM_CUDA_INLINE=0 posts CUDA calls to the
	// owner goroutine. All three are A/B arms of a measurement.
	if v := str("JITLLM_METAL_SPIN_US"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			d := time.Duration(n) * time.Microsecond
			if n == 0 {
				d = -1
			}
			o = append(o, tier.WithMetalSpin(d))
		}
	}
	if str("JITLLM_METAL_FASTMATH") == "1" {
		o = append(o, tier.WithMetalFastMath(true))
	}
	// JITLLM_METAL_CONCURRENT=1 opens Metal's compute encoders concurrent,
	// with barriers where dispatches depend: the A/B arm.
	if str("JITLLM_METAL_CONCURRENT") == "1" {
		o = append(o, tier.WithMetalConcurrent(true))
	}
	if str("JITLLM_CUDA_INLINE") == "0" {
		o = append(o, tier.WithCUDAInline(false))
	}
	if v := str("JITLLM_ARENA"); v != "" {
		if n, err := tier.ParseBytes(v); err == nil {
			o = append(o, tier.WithArena(n))
		}
	}
	// JITLLM_GPU_ROWT, the decode row tile, is a number: 1 is one row per thread, and 2/4/8 give
	// a thread that many independent accumulator chains.
	// JITLLM_GPU_FLASH=1 runs decode attention as one generated online-softmax
	// kernel (tier.Config.FlashAttention); JITLLM_GPU_FLASH_SPLIT partitions its
	// keys, merged by a second launch.
	// JITLLM_GPU_FLASH=2 selects the KV-head form (tier.Config.FlashKV), which
	// is the default on CUDA and Metal; =0 keeps decode attention staged (NoFlashKV).
	if str("JITLLM_GPU_FLASH") == "0" {
		o = append(o, tier.WithConfig(func(c *tier.Config) { c.NoFlashKV = true }))
	}
	if f := num("JITLLM_GPU_FLASH"); f > 0 {
		sp := int(num("JITLLM_GPU_FLASH_SPLIT"))
		grp, warps := int(num("JITLLM_GPU_FLASH_GROUP")), int(num("JITLLM_GPU_FLASH_WARPS"))
		o = append(o, tier.WithConfig(func(c *tier.Config) {
			c.FlashAttention, c.FlashKV, c.FlashSplit, c.FlashGroup, c.FlashWarps = f == 1, f == 2, sp, grp, warps
		}))
	}
	if n := num("JITLLM_GPU_ROWT"); n > 1 {
		o = append(o, tier.WithConfig(func(c *tier.Config) { c.DecodeRowt = int(n) }))
	}
	// JITLLM_GPU_KV_PAGE is the positions a KV page holds.
	// JITLLM_GPU_PREFILL_PASS is the keys one paged prefill launch attends
	// before the history goes in passes.
	if w := int(num("JITLLM_GPU_PREFILL_PASS")); w > 0 {
		o = append(o, tier.WithConfig(func(c *tier.Config) { c.PrefillPass = w }))
	}
	// JITLLM_GPU_PERSIST composes each decode block into persistent kernels
	// (tier.WithPersist), JITLLM_GPU_PERSIST_GRID capping their grid;
	// JITLLM_GPU_PERSIST=deps orders them with dependency counters rather
	// than grid barriers (tier.WithPersistDeps).
	if str("JITLLM_GPU_PERSIST") == "deps" {
		o = append(o, tier.WithPersistDeps(true, int(num("JITLLM_GPU_PERSIST_GRID"))))
	} else if set("JITLLM_GPU_PERSIST") {
		o = append(o, tier.WithPersist(true, int(num("JITLLM_GPU_PERSIST_GRID"))))
	}
	if page := int(num("JITLLM_GPU_KV_PAGE")); page > 0 {
		o = append(o, tier.WithConfig(func(c *tier.Config) { c.KVPage = page }))
	}
	// The A/B arms, which are Config fields. Each already carries the
	// measurement that set its default; see tier.WithConfig.
	flags := []struct {
		env string
		on  bool // what the variable's presence means
		set func(*tier.Config)
	}{
		{"JITLLM_GPU_GRAPH", os.Getenv("JITLLM_GPU_GRAPH") == "0", func(c *tier.Config) { c.NoGraph = true }},
		{"JITLLM_GPU_UNPACK", os.Getenv("JITLLM_GPU_UNPACK") == "0", func(c *tier.Config) { c.NoUnpack = true }},
		{"JITLLM_NO_GPU_BATCH", set("JITLLM_NO_GPU_BATCH"), func(c *tier.Config) { c.NoBatch = true }},
		{"JITLLM_GPU_NO_TAIL", set("JITLLM_GPU_NO_TAIL"), func(c *tier.Config) { c.NoTail = true }},
		{"JITLLM_GPU_NO_KT", set("JITLLM_GPU_NO_KT"), func(c *tier.Config) { c.NoKT = true }},
		{"JITLLM_NO_GPU_MMA", set("JITLLM_NO_GPU_MMA"), func(c *tier.Config) { c.NoMMA = true }},
		{"JITLLM_GPU_NO_FUSE", set("JITLLM_GPU_NO_FUSE"), func(c *tier.Config) { c.NoFuse = true }},
		{"JITLLM_GPU_NO_ACTFUSE", set("JITLLM_GPU_NO_ACTFUSE"), func(c *tier.Config) { c.NoActFuse = true }},
		{"JITLLM_GPU_ACTIN_SITES", set("JITLLM_GPU_ACTIN_SITES"),
			func(c *tier.Config) { c.ActInSites = num("JITLLM_GPU_ACTIN_SITES") }},
		{"JITLLM_GPU_ACTIN_PREFETCH", set("JITLLM_GPU_ACTIN_PREFETCH"),
			func(c *tier.Config) { c.ActInPrefetch = num("JITLLM_GPU_ACTIN_PREFETCH") }},
		{"JITLLM_GPU_NO_VOLTA", set("JITLLM_GPU_NO_VOLTA"), func(c *tier.Config) { c.NoVolta = true }},
		{"JITLLM_GPU_NO_FLASH_PREFILL", set("JITLLM_GPU_NO_FLASH_PREFILL"), func(c *tier.Config) { c.NoFlashPrefill = true }},
		{"JITLLM_GPU_NO_TILE_TAIL", set("JITLLM_GPU_NO_TILE_TAIL"), func(c *tier.Config) { c.NoTileTail = true }},
		{"JITLLM_GPU_NO_EXACT_PROMPT", set("JITLLM_GPU_NO_EXACT_PROMPT"), func(c *tier.Config) { c.NoExactPrompt = true }},
		{"JITLLM_GPU_NO_VOLTA_MOE", set("JITLLM_GPU_NO_VOLTA_MOE"), func(c *tier.Config) { c.NoVoltaMoE = true }},
		{"JITLLM_GPU_GEMM_INT8", set("JITLLM_GPU_GEMM_INT8"), func(c *tier.Config) { c.PreferGemmInt8 = true }},
		{"JITLLM_GPU_GEMM_DEEP", set("JITLLM_GPU_GEMM_DEEP"), func(c *tier.Config) { c.GemmDeep = num("JITLLM_GPU_GEMM_DEEP") }},
		{"JITLLM_GPU_PREFER_TILES", set("JITLLM_GPU_PREFER_TILES"), func(c *tier.Config) { c.PreferTiles = true }},
		{"JITLLM_GPU_NO_RAG_GROUP", set("JITLLM_GPU_NO_RAG_GROUP"), func(c *tier.Config) { c.NoRagGroup = true }},
		{"JITLLM_GPU_NO_RAG_FUSE", set("JITLLM_GPU_NO_RAG_FUSE"), func(c *tier.Config) { c.NoRagFuse = true }},
		{"JITLLM_GPU_NO_RAG_HEAD_ONE", set("JITLLM_GPU_NO_RAG_HEAD_ONE"), func(c *tier.Config) { c.NoRagHeadOne = true }},
		{"JITLLM_NO_GPU_GROUP", set("JITLLM_NO_GPU_GROUP"), func(c *tier.Config) { c.NoGroupSplit = true }},
		{"JITLLM_GPU_SCALAR_SOFTMAX", set("JITLLM_GPU_SCALAR_SOFTMAX"), func(c *tier.Config) { c.ScalarSoftmax = true }},
		{"JITLLM_GPU_FORCE_GROUP", set("JITLLM_GPU_FORCE_GROUP"), func(c *tier.Config) { c.ForceGroupSplit = true }},
		{"JITLLM_GPU_STREAM", set("JITLLM_GPU_STREAM"), func(c *tier.Config) { c.StreamExperts = true }},
		{"JITLLM_GPU_STREAM_FIXED", set("JITLLM_GPU_STREAM_FIXED"), func(c *tier.Config) { c.StreamFixedSel = true }},
		{"JITLLM_GPU_PROBE", set("JITLLM_GPU_PROBE"), func(c *tier.Config) { c.StreamProbe = true }},
		{"JITLLM_GPU_STREAM_CACHE", num("JITLLM_GPU_STREAM_CACHE") > 0,
			func(c *tier.Config) { c.StreamCacheSlots = num("JITLLM_GPU_STREAM_CACHE") }},
		{"JITLLM_GPU_STREAM_PREFETCH", set("JITLLM_GPU_STREAM_PREFETCH"), func(c *tier.Config) { c.StreamPrefetch = true }},
		{"JITLLM_GPU_NO_AUTOSTREAM", set("JITLLM_GPU_NO_AUTOSTREAM"), func(c *tier.Config) { c.NoAutoStream = true }},
		{"JITLLM_GPU_HYBRID", os.Getenv("JITLLM_GPU_HYBRID") == "1", func(c *tier.Config) { c.HybridExperts = true }},
		{"JITLLM_GPU_HYBRID", os.Getenv("JITLLM_GPU_HYBRID") == "0", func(c *tier.Config) { c.NoHybrid = true }},
		// =0 gives a hybrid block no expert cache on the card (tier.WithHybridCache).
		{"JITLLM_GPU_HYBRID_CACHE", os.Getenv("JITLLM_GPU_HYBRID_CACHE") == "0", func(c *tier.Config) { c.NoHybridCache = true }},
		{"JITLLM_GPU_SPLIT_ADMIT", num("JITLLM_GPU_SPLIT_ADMIT") > 0,
			func(c *tier.Config) { c.SplitAdmit = num("JITLLM_GPU_SPLIT_ADMIT") }},
		{"JITLLM_GPU_STREAM_NOPIN", set("JITLLM_GPU_STREAM_NOPIN"), func(c *tier.Config) { c.StreamNoPin = true }},
		{"JITLLM_GPU_STREAM_GROUPS", num("JITLLM_GPU_STREAM_GROUPS") > 0,
			func(c *tier.Config) { c.StreamGroups = num("JITLLM_GPU_STREAM_GROUPS") }},
		// =0 uploads the host's rotary table instead of building it on the
		// device; as with JITLLM_GPU_GRAPH, the arm turns a default off.
		{"JITLLM_GPU_ROPETAB", os.Getenv("JITLLM_GPU_ROPETAB") == "0",
			func(c *tier.Config) { c.RopeTableHost = true }},
	}
	for _, f := range flags {
		if f.on {
			o = append(o, tier.WithConfig(f.set))
		}
	}
	// JITLLM_GPU_SELLOG names a file that gets one line per streamed block per
	// token: the block index and its routed expert ids in rank order.
	if path := os.Getenv("JITLLM_GPU_SELLOG"); path != "" {
		if f, err := os.Create(path); err == nil {
			o = append(o, tier.WithConfig(func(c *tier.Config) {
				c.StreamSelLog = func(block int, sel []uint32) {
					fmt.Fprintln(f, block, sel)
				}
			}))
		} else {
			fmt.Fprintln(os.Stderr, "JITLLM_GPU_SELLOG:", err)
		}
	}
	return o
}

// pagerChunk is JITLLM_CHUNK_KIB, the pager's fill granularity, in bytes; 0
// when unset or not a power of two of at least 4 KiB, which keeps the
// container's own.
func pagerChunk() uint64 {
	n, _ := strconv.Atoi(os.Getenv("JITLLM_CHUNK_KIB"))
	if n < 4 || n&(n-1) != 0 {
		return 0
	}
	return uint64(n) << 10
}

// phRead hands a phase meter the tier's cumulative host-read time, so a phase
// row's rate is a bandwidth rather than a floor over its wall. The tier's
// number wins where there is one, because it excludes the parts of the
// suspension that are not I/O; otherwise jlm.File.ReadWait (the wall decode
// spent blocked in EnsureRanges) covers host runs.
func phRead(ph *model.PhaseMeter, dev nn.Device, m *model.Model) {
	if ph == nil {
		return
	}
	if g, ok := dev.(*tier.GPU); ok {
		ph.ReadTime(g.Stats().TStreamRead)
		return
	}
	ph.ReadTime(m.PagerReadWait())
}

// byteCount is a flag holding a byte count: a number with an optional K/M/G/T
// (tier.ParseBytes, the same grammar -devices takes), or 0 for "work it out".
type byteCount uint64

func (b *byteCount) String() string { return strconv.FormatUint(uint64(*b), 10) }

func (b *byteCount) Set(s string) error {
	if strings.TrimSpace(s) == "0" {
		*b = 0
		return nil
	}
	n, err := tier.ParseBytes(s)
	if err != nil {
		return err
	}
	*b = byteCount(n)
	return nil
}

// bytesFlag defines a byte-count flag on fs, defaulting to def, and returns
// where it lands.
func bytesFlag(fs *flag.FlagSet, name string, def uint64, usage string) *uint64 {
	b := byteCount(def)
	fs.Var(&b, name, usage)
	return (*uint64)(&b)
}
