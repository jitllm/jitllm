package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// gpuOptions turns the device tier's environment variables into gpu.Options.
//
// They are read here and nowhere else: the device packages read no environment,
// so an embedded caller can configure each tier through options.
//
// JITLLM_DEVICE_FAIL_AT is deliberately not here: it is read by
// engine/device/gpu/fault.go, compiled only under `-tags jitllmfault`, so no release
// binary can be made to pretend its GPU broke.
func gpuOptions() []gpu.Option {
	var o []gpu.Option
	str := func(k string) string { return os.Getenv(k) }
	num := func(k string) int {
		n, _ := strconv.Atoi(os.Getenv(k))
		return n
	}
	set := func(k string) bool { return os.Getenv(k) != "" }

	if v := str("JITLLM_GPU_API"); v != "" {
		o = append(o, gpu.WithAPI(v))
	}
	switch str("JITLLM_GPU_TUNE") {
	case "0":
		o = append(o, gpu.WithDeviceTune(gpu.TuneOff))
	case "1":
		o = append(o, gpu.WithDeviceTune(gpu.TuneForce))
	}
	// A register cap on every CUDA kernel: the register report's what-if.
	if n := num("JITLLM_CUDA_MAXREGS"); n > 0 {
		o = append(o, gpu.WithCUDAMaxRegs(n))
	}
	if set("JITLLM_GPU_VERBOSE") {
		o = append(o, gpu.WithVerbose(true))
	}
	if v := rocmDir(); v != "" {
		o = append(o, gpu.WithROCm(v))
	}
	if v := str("JITLLM_VK_DEVICE"); v != "" {
		o = append(o, gpu.WithVulkanDevice(v))
	}
	if v := str("JITLLM_VK_SUBGROUP"); v != "" {
		o = append(o, gpu.WithSubgroup(v))
	}
	if n := num("JITLLM_VK_MAXGROUPS"); n >= 1 {
		o = append(o, gpu.WithVulkanMaxGroups(uint32(n)))
	}
	if n := num("JITLLM_GPU_LANES"); n >= 1 {
		o = append(o, gpu.WithLanes(n))
	}
	if set("JITLLM_GPU_POISON_KV") {
		o = append(o, gpu.WithPoisonKV(true))
	}
	b := gpu.BatchShape{
		Tok:    num("JITLLM_GPU_TOK"),
		RowT:   num("JITLLM_GPU_ROWT"),
		QTile:  num("JITLLM_GPU_QTILE"),
		KTile:  num("JITLLM_GPU_KTILE"),
		ATile:  num("JITLLM_GPU_ATILE"),
		Parts:  num("JITLLM_GPU_PARTS"),
		AttnNT: num("JITLLM_GPU_ATTNNT"),
		// The ragged decode step's matvec where there is no matrix
		// instruction (jitllm batch); see gpu.BatchShape.
		RagTok:   num("JITLLM_GPU_RAGTOK"),
		RagRowT:  num("JITLLM_GPU_RAGROWT"),
		RagSplit: num("JITLLM_GPU_RAGSPLIT"),
		Split:    num("JITLLM_GPU_BSPLIT"),
	}
	if b != (gpu.BatchShape{}) {
		o = append(o, gpu.WithBatchShape(b))
	}
	if v := str("JITLLM_GPU_MMATILE"); v != "" {
		var mt, nt int
		if _, err := fmt.Sscanf(v, "%dx%d", &mt, &nt); err == nil && mt > 0 && nt > 0 {
			o = append(o, gpu.WithMMATile(mt, nt))
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
			o = append(o, gpu.WithMetalSpin(d))
		}
	}
	if str("JITLLM_METAL_FASTMATH") == "1" {
		o = append(o, gpu.WithMetalFastMath(true))
	}
	// JITLLM_METAL_CONCURRENT=1 opens Metal's compute encoders concurrent,
	// with barriers where dispatches depend: the A/B arm.
	if str("JITLLM_METAL_CONCURRENT") == "1" {
		o = append(o, gpu.WithMetalConcurrent(true))
	}
	if str("JITLLM_CUDA_INLINE") == "0" {
		o = append(o, gpu.WithCUDAInline(false))
	}
	if v := str("JITLLM_ARENA"); v != "" {
		if n, err := gpu.ParseBytes(v); err == nil {
			o = append(o, gpu.WithArena(n))
		}
	}
	// JITLLM_GPU_KV_PAGE is the positions a KV page holds.
	// JITLLM_GPU_PERSIST composes each decode block into persistent kernels
	// (gpu.WithPersist), JITLLM_GPU_PERSIST_GRID capping their grid;
	// JITLLM_GPU_PERSIST=deps orders them with dependency counters rather
	// than grid barriers (gpu.WithPersistDeps).
	if str("JITLLM_GPU_PERSIST") == "deps" {
		o = append(o, gpu.WithPersistDeps(true, int(num("JITLLM_GPU_PERSIST_GRID"))))
	} else if set("JITLLM_GPU_PERSIST") {
		o = append(o, gpu.WithPersist(true, int(num("JITLLM_GPU_PERSIST_GRID"))))
	}
	if page := int(num("JITLLM_GPU_KV_PAGE")); page > 0 {
		o = append(o, gpu.WithConfig(func(c *gpu.Config) { c.KVPage = page }))
	}
	// The A/B arms, which are Config fields. Each already carries the
	// measurement that set its default; see gpu.WithConfig.
	flags := []struct {
		env string
		on  bool // what the variable's presence means
		set func(*gpu.Config)
	}{
		{"JITLLM_GPU_GRAPH", os.Getenv("JITLLM_GPU_GRAPH") == "0", func(c *gpu.Config) { c.NoGraph = true }},
		{"JITLLM_NO_GPU_BATCH", set("JITLLM_NO_GPU_BATCH"), func(c *gpu.Config) { c.NoBatch = true }},
		{"JITLLM_GPU_NO_TAIL", set("JITLLM_GPU_NO_TAIL"), func(c *gpu.Config) { c.NoTail = true }},
		{"JITLLM_NO_GPU_MMA", set("JITLLM_NO_GPU_MMA"), func(c *gpu.Config) { c.NoMMA = true }},
		{"JITLLM_GPU_NO_EXACT_PROMPT", set("JITLLM_GPU_NO_EXACT_PROMPT"), func(c *gpu.Config) { c.NoExactPrompt = true }},
		{"JITLLM_GPU_STREAM", set("JITLLM_GPU_STREAM"), func(c *gpu.Config) { c.StreamExperts = true }},
		{"JITLLM_GPU_STREAM_CACHE", num("JITLLM_GPU_STREAM_CACHE") > 0,
			func(c *gpu.Config) { c.StreamCacheSlots = num("JITLLM_GPU_STREAM_CACHE") }},
		{"JITLLM_GPU_NO_AUTOSTREAM", set("JITLLM_GPU_NO_AUTOSTREAM"), func(c *gpu.Config) { c.NoAutoStream = true }},
		{"JITLLM_GPU_STREAM_NOPIN", set("JITLLM_GPU_STREAM_NOPIN"), func(c *gpu.Config) { c.StreamNoPin = true }},
		{"JITLLM_GPU_STREAM_GROUPS", num("JITLLM_GPU_STREAM_GROUPS") > 0,
			func(c *gpu.Config) { c.StreamGroups = num("JITLLM_GPU_STREAM_GROUPS") }},
	}
	for _, f := range flags {
		if f.on {
			o = append(o, gpu.WithConfig(f.set))
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
func phRead(ph *engine.PhaseMeter, dev nn.LayerDevice, m *model.Model, fr *jlm.Frames) {
	if ph == nil {
		return
	}
	if g, ok := dev.(*gpu.GPU); ok {
		ph.ReadTime(g.Stats().TStreamRead)
		return
	}
	ph.ReadTime(m.PagerReadWait(fr))
}

// byteCount is a flag holding a byte count: a number with an optional K/M/G/T
// (gpu.ParseBytes, the same grammar -devices takes), or 0 for "work it out".
type byteCount uint64

func (b *byteCount) String() string { return strconv.FormatUint(uint64(*b), 10) }

func (b *byteCount) Set(s string) error {
	if strings.TrimSpace(s) == "0" {
		*b = 0
		return nil
	}
	n, err := gpu.ParseBytes(s)
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
