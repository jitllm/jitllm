package main

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/jit/cpu"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/vulkan"
)

// hardware reports what compute this machine offers jitllm.
//
// It reports rather than probes: every line is read from where the engine reads
// it (sched for topology, jit/cpu for what the emitter generates, backend.Open
// for devices), so it cannot disagree with what a decode does. Nothing is
// timed; `jitllm bench` prints the memory wall.
func hardware() error {
	fmt.Printf("host      %s/%s   %s\n", runtime.GOOS, runtime.GOARCH, runtime.Version())
	reportCPU()
	reportMem()
	reportJIT()
	reportGPU()
	return nil
}

func reportCPU() {
	p, e, smt := sched.PCores(), sched.ECores(), sched.SMTSiblings()
	fmt.Printf("cpu       %s\n", cpuBrand())
	fmt.Printf("          %d logical   %d P-core(s) %s", runtime.NumCPU(), len(p), list(p))
	if len(e) > 0 {
		fmt.Printf("   %d E-core(s) %s", len(e), list(e))
	}
	fmt.Println()
	fmt.Printf("          source    %s\n", sched.CoreSource())
	if t := len(smt) / max(1, len(p)); t > 1 {
		fmt.Printf("          SMT       %d threads per P-core: %s\n", t, list(smt))
	} else {
		fmt.Printf("          SMT       none\n")
	}
	// DecodeCores is the pool jitllm actually builds, JITLLM_CORESET and
	// JITLLM_CORES included, so this reports the shipping configuration rather
	// than the topology it was derived from.
	set := os.Getenv("JITLLM_CORESET")
	if set == "" {
		set = "p"
	}
	d := sched.DecodeCores(schedOptions()...)
	fmt.Printf("          decode    coreset %s, %d core(s) %s", set, len(d), list(d))
	if v := os.Getenv("JITLLM_CORES"); v != "" {
		fmt.Printf(" (JITLLM_CORES=%s)", v)
	}
	fmt.Println()
	fmt.Printf("          caller is worker 0; how many of those drain a region is tuned per shape at run time\n")
	if runtime.GOOS == "darwin" {
		fmt.Printf("          those indices are placeholders: macOS exposes no affinity, so workers run\n")
		fmt.Printf("          unpinned and QoS places them on the performance cluster\n")
	}
}

func reportMem() {
	if b, ok := totalRAM(); ok {
		fmt.Printf("memory    %.1f GiB total\n", float64(b)/(1<<30))
		if nodes := sched.NUMANodes(); len(nodes) > 1 {
			how := "weights interleaved across them"
			if numaNodes() == nil {
				how = "placement left to the kernel (JITLLM_NUMA=off, or a policy was already set)"
			}
			fmt.Printf("          numa %d nodes %v: %s\n", len(nodes), nodes, how)
		}
	} else {
		fmt.Printf("memory    unknown\n")
	}
	fmt.Printf("          read wall not measured here -- `jitllm bench` reports it, as a Go-side\n")
	fmt.Printf("          scalar unpinned probe, which is a LOWER BOUND on the real bandwidth\n")
}

// fastPath is the list of packed formats: a list, never a promise of coverage.
// Asking SupportedNative per type keeps this honest when the emitters differ.
// It is quant.PackedTypes itself, not a copy that could drift.
var fastPath = quant.PackedTypes

func reportJIT() {
	// Show what a run would emit: jit/cpu reads no environment, so apply the
	// same pins a run would.
	num := func(k string) int {
		n, _ := strconv.Atoi(os.Getenv(k))
		return n
	}
	pack, accs := num("JITLLM_PACK"), num("JITLLM_ACCS")
	cpuConfigure()
	// The window is the emitter's answer, not the model's: 256 only where the
	// native k-quant kernel hoists the activation scale out of its sub-block
	// loop, which requires an arm64 target.
	fmt.Printf("jit       %s emitter   activation window %d with Q4_0/Q8_0, %d all-k-quant\n",
		cpu.NativeArch(), cpu.Q8Block, cpu.WideActWindow([]quant.Type{quant.Q4_K}))
	fmt.Printf("          tier      %s\n", cpu.TierReport())
	fmt.Printf("          isa       %s\n", isa())
	var ks []string
	for _, t := range fastPath {
		if !cpu.SupportedNative(t) {
			ks = append(ks, fmt.Sprintf("%s none", t))
			continue
		}
		ks = append(ks, fmt.Sprintf("%s pack%d accs%d", t, cpu.PackWidthNative(pack, t), cpu.AccChains(accs, t)))
	}
	fmt.Printf("          kernels   %s\n", strings.Join(ks, "  "))
	// Two families, two lines. The line above is the GGUF row-major emitters,
	// which a container never decodes through; on a pre-VNNI host it reads
	// "none" while the packed family below runs everything.
	var ps []string
	for _, t := range fastPath {
		switch {
		case !cpu.PackedSupported(t):
			ps = append(ps, fmt.Sprintf("%s none", t))
		case !cpu.SupportedPackedNative(t):
			ps = append(ps, fmt.Sprintf("%s declined", t))
		default:
			ps = append(ps, fmt.Sprintf("%s %s", t, cpu.HostDotLabel()))
		}
	}
	fmt.Printf("          packed    %s   <- the container path\n", strings.Join(ps, "  "))
	fmt.Printf("          pack width is re-tuned on real decode tokens and cached; JITLLM_PACK pins it.\n")
	fmt.Printf("          every op a token runs is generated code; there is no interpreted tier\n")
}

// isa names what the row-major emitter emits, and whether this CPU can run it.
// jit/cpu probes (CPUID leaf 7.1 EAX[4] on amd64, AT_HWCAP or sysctl on arm64)
// On amd64 SupportedNative declines when the feature is absent; arm64 widens
// SDOT instead (jit/cpu/sdotemu.go) and runs every kernel. This line is about
// the GGUF row-major family only; the packed line is what a container runs.
func isa() string {
	switch cpu.NativeArch() {
	case "amd64":
		// Off the AVX2 tier the VNNI line would describe a tier this host
		// cannot run; say which family is absent instead.
		if cpu.HostTier() != cpu.TierAVX2 {
			return "legacy SSE, no VEX: the GGUF row-major quantized kernels do not exist " +
				"on this tier; the packed line below is what a container runs"
		}
		s := "AVX2 + AVX-VNNI, VEX-encoded VPDPBUSD"
		if cpu.SupportedNative(quant.Q4_0) {
			return s + " (probed: present)"
		}
		return s + " (probed: ABSENT -- the GGUF row-major kernels decline; the packed line below is what a container runs)"
	case "arm64":
		s := "NEON + FEAT_DotProd SDOT"
		if !cpu.HasDotProd() {
			s += " (probed: ABSENT -- every SDOT is widened to SMULL/SMLAL/ADDP/SADALP: the same bits, slower)"
		} else {
			s += " (probed: present)"
		}
		return s + "; i8mm unused, decode has one activation vector so SMMLA wastes half its tile"
	}
	return "unknown"
}

func reportGPU() {
	// backend.Open is reached without a tier here, so JITLLM_GPU_VERBOSE has to
	// be handed to it directly: the package reads no environment.
	backend.SetVerbose(os.Getenv("JITLLM_GPU_VERBOSE") != "")
	// The same call tier.Open makes. Opening and closing every backend is
	// covered by backend.TestCloseThenSpawnThreads.
	hipCfg := backend.HIPConfig{Path: rocmDir()}
	devs := backend.OpenWith(backend.Opts{HIP: hipCfg})
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	// No ROCm is not an error: an AMD card then runs through Vulkan. The line
	// says where it was looked for, so a ROCm in an unusual place can be named
	// with JITLLM_ROCM.
	if m := backend.HIPMissing(hipCfg); m != "" {
		fmt.Printf("hip       %s\n", m)
	} else {
		reportHIPList(hipCfg)
	}
	if len(devs) == 0 {
		fmt.Printf("gpu       none available: no CUDA, HIP, Vulkan or Metal device opened here\n")
		fmt.Printf("          (that is a CPU machine, not an error -- `jitllm run` defaults to it)\n")
		return
	}
	label := "gpu"
	for _, d := range devs {
		slots := "unknown"
		if s := d.Slots(); s > 0 {
			slots = strconv.Itoa(s)
		}
		// 49 fits a typical "#0 <vendor> <model> (sm_NN)" device name.
		fmt.Printf("%-9s %-6s %-49s slots %-8s vram %s\n", label, d.API(), d.Name(), slots, vram(d))
		label = ""
		// Whether a page-in here is a transfer or a descriptor update. Two
		// conditions: a device may wrap host pointers without its memory being
		// the host's, in which case wrapping is not free.
		if a := backend.ImportAlign(d); a > 0 {
			why := "but its memory is its own, so a wrapped weight would be read over the bus"
			if u, ok := d.(backend.Unified); ok && u.UnifiedMemory() {
				why = "and its memory IS the host's, so a page-in is a descriptor update"
			}
			fmt.Printf("          %6s wraps host memory at %d-byte alignment, %s\n", "", a, why)
		}
		// The device identity, which says when two lines (ptx and spirv) are
		// one card.
		if id := backend.IdentityOf(d); id.Known() {
			fmt.Printf("          %6s device %s\n", "", id.Key)
		}
	}
	reportSameCard(devs)
	reportVulkanList(devs)
	fmt.Printf("          %s\n", picked(devs))
	// The default budget is derived per device from free memory
	// (tier.budgetFor).
	fmt.Printf("          weight budget defaults to free memory less an eighth of it (at least\n")
	fmt.Printf("          256 MiB) of headroom, per device; -vram BYTES overrides every device and\n")
	fmt.Printf("          -devices cuda:0=3G one of them. A block that does not fit is declined per\n")
	fmt.Printf("          layer, so the rest of the model runs on the CPU -- unless -placement\n")
	fmt.Printf("          streams it (N=DEV~), which SWAPS its weights through the slots the budget\n")
	fmt.Printf("          leaves (the device runs it, and moves its weights per token)\n")
	fmt.Printf("          a device whose memory is SHARED with the host draws from the host's own\n")
	fmt.Printf("          weight budget instead of a budget of its own, and the host loses what it\n")
	fmt.Printf("          takes -- counting both would spend the same bytes twice. On such a device\n")
	fmt.Printf("          the packed arena is not a second copy of the weights, it IS the device\n")
	fmt.Printf("          buffer: the driver is handed its pointer, so a page-in moves nothing and\n")
	fmt.Printf("          is charged nothing (the arena's own budget is what bounds it)\n")
}

// reportHIPList names every HIP device as hip:N, the selector -devices takes.
func reportHIPList(c backend.HIPConfig) {
	for i := 0; i < backend.HIPCount(c); i++ {
		d, err := backend.OpenHIPWith(i, backend.Opts{HIP: c, Enumerating: true})
		if err != nil {
			fmt.Printf("hip:%-5d %v\n", i, err)
			continue
		}
		fmt.Printf("hip:%-5d %-49s vram %s\n", i, d.Name(), vram(d))
		d.Close()
	}
}

// reportVulkanList names every Vulkan physical device with the index that
// selects it. backend.Open opens one device per backend (discrete first), so
// this is the only place the others, such as an integrated GPU, are listed.
func reportVulkanList(open []backend.Device) {
	infos, err := backend.VulkanDevices()
	if err != nil || len(infos) < 2 {
		// One device is already on the line above, and no Vulkan at all is not
		// news here: the device list said so.
		return
	}
	for _, in := range infos {
		note := ""
		if !in.Compute {
			note = "  NO COMPUTE QUEUE -- cannot run a kernel"
		} else if in.Software {
			note = "  software rasteriser: not placed on unless named"
		} else if in.Unified {
			note = "  memory SHARED with the host"
		} else if other := heldBy(open, in, "spirv"); other != "" {
			// Listed, since naming it is legitimate, but marked as not a
			// second card.
			note = "  the SAME device as the " + other + " entry above"
		}
		fmt.Printf("          vulkan:%-2d %-49s%s\n", in.Index, in.Name, note)
		// A device that cannot promise 32 lanes runs the scalar softmax and q/k
		// norm; printed only when it applies.
		if in.Compute && !in.Promises(32) {
			fmt.Printf("          %-11s subgroups %d..%d, no 32-lane guarantee: scalar softmax "+
				"and head norm\n", "", in.Subgroups.Min, in.Subgroups.Max)
		}
	}
}

// reportSameCard says which of the opened devices are one piece of hardware,
// so their memory is not counted twice. The device UUID decides it, not the
// name, which two identical cards share.
func reportSameCard(devs []backend.Device) {
	for i, d := range devs {
		id := backend.IdentityOf(d)
		if !id.Known() {
			continue
		}
		for j := 0; j < i; j++ {
			if !backend.IdentityOf(devs[j]).Same(id) {
				continue
			}
			keep, drop := devs[j], d
			if backend.APIRank(d.API()) > backend.APIRank(devs[j].API()) {
				keep, drop = d, devs[j]
			}
			fmt.Printf("          %6s ★ the %s and %s entries are ONE device (%s): their memory is "+
				"ONE budget,\n", "", devs[j].API(), d.API(), id.Key)
			fmt.Printf("          %6s   and the default set takes %s. %s is still selectable by name.\n",
				"", keep.API(), drop.API())
		}
	}
}

// heldBy names the API of an OPENED device that is the same physical hardware
// as an enumerated one, skipping entries reached through skipAPI itself.
func heldBy(open []backend.Device, in vulkan.Info, skipAPI string) string {
	id := backend.UUIDIdentity(in.UUID[:], in.UUIDOK, "VkPhysicalDeviceIDProperties.deviceUUID")
	if !id.Known() {
		return ""
	}
	for _, d := range open {
		if d.API() != skipAPI && backend.IdentityOf(d).Same(id) {
			return d.API()
		}
	}
	return ""
}

// vram asks the device for its memory through an optional interface, since a
// backend may be unable to say (a Vulkan driver without VK_EXT_memory_budget).
// A device whose memory is the host's says so, because its numbers are claims
// on the same bytes as the host's.
func vram(d backend.Device) string {
	free, total, err := d.Mem()
	if err != nil {
		return "unknown (" + err.Error() + ")"
	}
	s := fmt.Sprintf("%.2f of %.2f GiB free", float64(free)/(1<<30), float64(total)/(1<<30))
	if u, ok := d.(backend.Unified); ok && u.UnifiedMemory() {
		s += " (SHARED with host RAM -- not a separate budget)"
	}
	return s
}

func picked(devs []backend.Device) string {
	if want := os.Getenv("JITLLM_GPU_API"); want != "" {
		return fmt.Sprintf("jitllm would pick %q, pinned by JITLLM_GPU_API", want)
	}
	if len(devs) == 1 {
		return fmt.Sprintf("jitllm would pick %s: it is the only one", devs[0].API())
	}
	// Two views of one device are decided by the backend preference; the
	// matvec probe is for devices that are genuinely different.
	if len(dedupedAPIs(devs)) == 1 {
		return fmt.Sprintf("jitllm would pick %s: these are one device, and CUDA is the preferred "+
			"backend for it (-devices vulkan:N still takes the other)", devs[0].API())
	}
	return "jitllm picks between these by timing one real matvec on each at startup, cached against " +
		"the device set (JITLLM_GPU_API pins it, JITLLM_GPU_TUNE=1 re-probes)"
}

// dedupedAPIs is the set of APIs left once each physical device is counted
// once. A device with no identity counts as itself.
func dedupedAPIs(devs []backend.Device) []string {
	seen := map[string]bool{}
	var out []string
	for _, d := range devs {
		id := backend.IdentityOf(d)
		if id.Known() {
			if seen[id.Key] {
				continue
			}
			seen[id.Key] = true
		}
		out = append(out, d.API())
	}
	return out
}

// cpuBrand is the one fact in this report with no in-tree source: engine/nn/tune.go
// dropped the brand string from its cache key on purpose, because a stepping
// bump invalidates a good answer and the string says nothing about what the
// emitter will do. It is still what a person means by "which CPU is this", so it
// is read here and nowhere else.
func cpuBrand() string {
	if b, err := os.ReadFile("/proc/cpuinfo"); err == nil {
		for _, ln := range strings.Split(string(b), "\n") {
			if k, v, ok := strings.Cut(ln, ":"); ok && strings.TrimSpace(k) == "model name" {
				return strings.TrimSpace(v)
			}
		}
	}
	if s := sysctl("machdep.cpu.brand_string"); s != "" {
		return s
	}
	return "unknown"
}

func totalRAM() (uint64, bool) {
	if b, err := os.ReadFile("/proc/meminfo"); err == nil {
		for _, ln := range strings.Split(string(b), "\n") {
			rest, ok := strings.CutPrefix(ln, "MemTotal:")
			if !ok {
				continue
			}
			f := strings.Fields(rest)
			if len(f) == 0 {
				break
			}
			if n, err := strconv.ParseUint(f[0], 10, 64); err == nil {
				return n * 1024, true // MemTotal is in kB
			}
		}
	}
	if n, err := strconv.ParseUint(sysctl("hw.memsize"), 10, 64); err == nil {
		return n, true
	}
	return 0, false
}

// sysctl shells out because syscall.Sysctl is darwin/BSD-only: reaching it
// directly needs a build tag, and one report is not worth a file per OS. An
// absent sysctl is "unknown", never an error.
func sysctl(name string) string {
	out, err := exec.Command("sysctl", "-n", name).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func list(ns []int) string {
	var b strings.Builder
	for i, n := range ns {
		if i > 0 {
			b.WriteByte(',')
		}
		if i == 12 {
			fmt.Fprintf(&b, "...+%d", len(ns)-12)
			break
		}
		fmt.Fprint(&b, n)
	}
	return b.String()
}

// schedOptions turns the pool's environment variables into sched.Options.
// They are read here and nowhere else: sched reads no environment.
func schedOptions() []sched.Option {
	var o []sched.Option
	if v := os.Getenv("JITLLM_CORESET"); v != "" {
		o = append(o, sched.WithCoreSet(v))
	}
	if v := os.Getenv("JITLLM_CORES"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			o = append(o, sched.WithCores(n))
		}
	}
	// Stage 2 of NUMA: with weight pages interleaved (numaNodes), bind the
	// workers to their nodes so each reads the rows its node holds.
	// JITLLM_NUMA=interleave keeps stage 1 alone, which is the A/B arm.
	if numaNodes() != nil && os.Getenv("JITLLM_NUMA") != "interleave" {
		o = append(o, sched.WithNUMA(true))
	}
	if v := os.Getenv("JITLLM_SPIN_US"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			o = append(o, sched.WithSpin(time.Duration(n)*time.Microsecond))
		}
	}
	return o
}
