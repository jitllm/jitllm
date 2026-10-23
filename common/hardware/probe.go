// Package hardware answers "what can this machine run, and what will the next load
// do with it" -- the desktop equivalent of `jitllm hardware`.
//
// It reports rather than probing on its own account: every field is read from
// where the engine reads it (sched, jit/cpu, backend.Open, tier), so the answer
// cannot disagree with what a decode would do. Nothing is timed; the memory
// wall is a separate, explicit button.
//
// It is the only screen-side package that imports jitllm. What the UI calls
// here is either a pure string parser or a probe run on a worker goroutine.
package hardware

import (
	"fmt"
	"github.com/jitllm/jitllm/dev/bench"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/cpu"
	"github.com/jitllm/jitllm/jit/gpu/tier"

	"github.com/jitllm/jitllm/common/session"
)

// Formats is the list of quantization types the emitter has kernels for.
//
// It is a list, not a promise of coverage, which is why every row asks the
// emitter rather than printing this slice.
var Formats = quant.PackedTypes

// FormatNames is [Formats] as strings, in the same order, so a caller that only
// wants to label a table needs no quant import.
func FormatNames() []string {
	out := make([]string, 0, len(Formats))
	for _, t := range Formats {
		out = append(out, t.String())
	}
	return out
}

// Kernel is what the emitter will do for one quantization type, in both
// families. Native is the GGUF row-major emitter, which a container never
// decodes through; on a pre-VNNI host it reads "none" while the packed family
// runs everything, so both columns are reported.
type Kernel struct {
	Format string
	// Native is the GGUF row-major family: gone from the container path.
	Native     bool
	NativeNote string
	// Packed is the container path. This is the column that matters.
	Packed     bool
	PackedNote string
}

// GPU is one device backend.Open actually opened.
type GPU struct {
	API  string
	Name string
	// Slots is how many threads the device wants resident, or 0 when the
	// runtime will not say.
	Slots int

	Free, Total uint64
	// MemNote carries the reason when Free and Total are not reported, so an
	// unknown is visibly an unknown rather than a zero.
	MemNote string

	// Unified says this device's memory is the host's: subtract it, never add
	// it, or the same bytes are spent twice.
	Unified bool

	// ImportAlign is the alignment at which this device can wrap host memory,
	// or 0 when it cannot. On a unified device that makes a page-in a
	// descriptor update rather than a transfer.
	ImportAlign int
}

// VulkanDev is one entry of the Vulkan enumeration. It is separate from GPUs
// because backend.Open opens one device per backend, so a spec like
// `vulkan:1` names a device the opened list does not contain.
type VulkanDev struct {
	Index      int
	Name       string
	Type       string
	Compute    bool
	Software   bool
	Unified    bool
	Subgroups  [2]int
	Promises32 bool
}

// Report is one probe of this machine.
type Report struct {
	Probed bool
	// Err is set when the probe itself failed or panicked. A driver that dies
	// under us must not take the window with it.
	Err string

	GOOS, GOARCH, GoVersion string
	CPU                     string
	Logical                 int
	PCores, ECores, SMT     []int
	Decode                  []int
	RAMTotal                uint64

	MemBudget, MemLimit   uint64
	GCBudgetCap, GCUsable uint64
	// MemWall is sustained memory read bandwidth across the decode cores, in
	// bytes a second: bench.MemWall, a lower bound on the hardware's peak.
	// Decode reads every active weight once a token, so it bounds the rate.
	MemWall float64

	Arch    string
	ISA     string
	DotKind string
	Kernels []Kernel

	GPUs   []GPU
	Vulkan []VulkanDev
}

var (
	mu     sync.Mutex
	cached Report
)

// Cached returns the last [Probe] and whether there has been one.
func Cached() (Report, bool) {
	mu.Lock()
	defer mu.Unlock()
	return cached, cached.Probed
}

// Probe reads the machine and caches the result.
//
// Call it from a worker, once, and not while a model is loaded: backend.Open
// opens and closes every backend, which contends with a live tier.GPU. Calls
// are serialised here.
func Probe() Report {
	mu.Lock()
	defer mu.Unlock()

	r := Report{
		Probed:    true,
		GOOS:      runtime.GOOS,
		GOARCH:    runtime.GOARCH,
		GoVersion: runtime.Version(),
		CPU:       cpuBrand(),
		Logical:   runtime.NumCPU(),
	}

	// A driver that panics inside dlopen is a bad day, not a crashed desktop.
	// Whatever was filled before the panic is kept: a partial report that says
	// so is worth more than an empty one.
	defer func() {
		if v := recover(); v != nil {
			r.Err = fmt.Sprintf("probe panicked: %v", v)
		}
		cached = r
	}()

	r.PCores, r.ECores, r.SMT = sched.PCores(), sched.ECores(), sched.SMTSiblings()
	// DecodeCores is the pool the engine actually builds, so this is a report
	// of the shipping configuration rather than of the topology it came from.
	r.Decode = sched.DecodeCores()
	r.RAMTotal, _ = totalRAM()
	r.MemWall = bench.MemWall(max(len(r.Decode), 1))

	r.MemBudget, r.MemLimit = sched.MemBudget(), sched.MemLimit()
	r.GCBudgetCap, r.GCUsable = sched.GCBudgetCap(), sched.GCUsable()

	r.Arch, r.ISA, r.DotKind = cpu.NativeArch(), isa(), cpu.HostDotLabel()
	r.Kernels = kernels()

	// backend.Open is reached without a tier here, so verbosity has to be
	// handed to it directly: the package reads no environment.
	backend.SetVerbose(false)
	devs := backend.Open()
	for _, d := range devs {
		r.GPUs = append(r.GPUs, describe(d))
	}
	for _, d := range devs {
		d.Close()
	}

	if infos, err := backend.VulkanDevices(); err == nil {
		for _, in := range infos {
			r.Vulkan = append(r.Vulkan, VulkanDev{
				Index:      in.Index,
				Name:       in.Name,
				Type:       in.Type,
				Compute:    in.Compute,
				Software:   in.Software,
				Unified:    in.Unified,
				Subgroups:  [2]int{int(in.Subgroups.Min), int(in.Subgroups.Max)},
				Promises32: in.Promises(32),
			})
		}
	}
	return r
}

// kernels asks the emitter, per format, in both families.
func kernels() []Kernel {
	out := make([]Kernel, 0, len(Formats))
	for _, t := range Formats {
		k := Kernel{Format: t.String()}

		k.Native = cpu.SupportedNative(t)
		if k.Native {
			k.NativeNote = fmt.Sprintf("pack%d accs%d", cpu.BestPackNative(t), cpu.BestAccs(t))
		} else {
			k.NativeNote = "none"
		}

		switch {
		case !cpu.PackedSupported(t):
			k.PackedNote = "none"
		case !cpu.SupportedPackedNative(t):
			// Generated, and declined on this host: a real capability gap, and
			// a different fact from having no kernel at all.
			k.PackedNote = "declined"
		default:
			k.Packed, k.PackedNote = true, cpu.HostDotLabel()
		}
		out = append(out, k)
	}
	return out
}

// describe asks a device what it has, through the optional interfaces the
// backend exposes -- a driver that cannot say reports its refusal rather than a
// zero.
func describe(d backend.Device) GPU {
	g := GPU{API: d.API(), Name: d.Name(), Slots: d.Slots(), ImportAlign: backend.ImportAlign(d)}

	m, ok := d.(interface {
		Mem() (free, total uint64, err error)
	})
	if !ok {
		g.MemNote = "not reported by this backend"
	} else if free, total, err := m.Mem(); err != nil {
		g.MemNote = "unknown (" + err.Error() + ")"
	} else {
		g.Free, g.Total = free, total
	}

	if u, ok := d.(backend.Unified); ok {
		g.Unified = u.UnifiedMemory()
	}
	return g
}

// isa names what the native emitter emits, and whether this CPU can run it.
//
// jit/cpu probes the feature (CPUID leaf 7.1, AT_HWCAP or sysctl) and declines
// the quantized row-major kernels when it is absent, so this is a measured
// capability.
func isa() string {
	switch cpu.NativeArch() {
	case "amd64":
		// Off the AVX2 tier the VNNI line describes kernels this host cannot
		// run; the tier report says what a run does here instead.
		if cpu.HostTier() != cpu.TierAVX2 {
			return cpu.TierReport()
		}
		s := "AVX2 + AVX-VNNI, VEX-encoded VPDPBUSD"
		if cpu.SupportedNative(quant.Q4_0) {
			return s + " (probed: present)"
		}
		return s + " (probed: ABSENT -- the GGUF row-major kernels decline; the packed column is what a container runs)"
	case "arm64":
		s := "NEON + FEAT_DotProd SDOT"
		if cpu.HasDotProd() {
			return s + " (probed: present)"
		}
		return s + " (probed: ABSENT -- SDOT widened to SMULL/SMLAL/ADDP/SADALP)"
	}
	return "unknown"
}

// Machine is the shell's view of this report.
//
// The conversion lives here and nowhere else, so there is one mapping to the
// contract every screen reads.
func (r Report) Machine() session.MachineReport {
	m := session.MachineReport{
		Probed:      r.Probed,
		Err:         r.Err,
		MemWall:     r.MemWall,
		Arch:        r.Arch,
		PCores:      len(r.PCores),
		ECores:      len(r.ECores),
		SMTSiblings: len(r.SMT),
		DecodeCores: len(r.Decode),
		MemBudget:   r.MemBudget,
		MemLimit:    r.MemLimit,
		GCBudgetCap: r.GCBudgetCap,
	}
	for _, k := range r.Kernels {
		// The two columns stay two columns; see Kernel.
		m.Kernels = append(m.Kernels, session.KernelRow{Format: k.Format, RowMajor: k.Native, Packed: k.Packed})
	}
	for _, g := range r.GPUs {
		m.GPUs = append(m.GPUs, session.GPUInfo{
			Name: g.Name, API: g.API, Slots: g.Slots, Mem: g.Total, Unified: g.Unified,
		})
	}
	return m
}

// ValidateSpec parses a device spec and says what it asks for.
//
// It is tier.ParseDevices, not a parser of this app's own, so the picker
// cannot accept a spec the engine then refuses.
func ValidateSpec(spec string) (string, error) {
	es, err := tier.ParseDevices(spec)
	if err != nil {
		return "", err
	}
	parts := make([]string, 0, len(es))
	device := false
	for _, e := range es {
		device = device || e.Device()
		p := e.API
		if e.HasSel {
			p += ":" + e.Sel
		}
		if e.Bytes > 0 {
			p += fmt.Sprintf(" at %s", session.Bytes(e.Bytes))
		}
		parts = append(parts, p)
	}
	s := strings.Join(parts, ", ")
	if !device {
		return s + " -- no device: the host runs every block", nil
	}
	return s, nil
}

// ValidatePlacement parses a block placement (the CLI's -placement) and checks
// that every device it names is one the device spec opens, so the picker cannot
// accept a map the load then refuses. Empty is valid: the engine places.
//
// It is model.ParsePlacement, not a grammar of this app's own.
func ValidatePlacement(placement, spec string) error {
	if strings.TrimSpace(placement) == "" {
		return nil
	}
	pl, err := model.ParsePlacement(placement, true)
	if err != nil {
		return err
	}
	es, err := tier.ParseDevices(spec)
	if err != nil {
		return err
	}
	opened := map[string]bool{"host": true}
	for _, e := range es {
		if !e.Device() || !e.HasSel {
			continue
		}
		opened[nn.DeviceName(e.API+":"+e.Sel)] = true
	}
	named := []model.Place{}
	for _, b := range pl.Blocks {
		named = append(named, b)
	}
	for _, x := range []*model.Place{pl.Rest, pl.Head} {
		if x != nil {
			named = append(named, *x)
		}
	}
	for _, x := range named {
		if !opened[x.On] {
			return fmt.Errorf("placement names %q, which the device spec %q does not open "+
				"(name each device there: cuda:0,vulkan:0)", x.On, spec)
		}
	}
	return nil
}

// ValidateBudget parses the host weight budget and says what it will mean.
//
// An empty string is not an error: it means the engine's own default, which is
// what fallback carries.
//
// It also warns about the collector: page frames are permanently live, so a
// budget above gcCap (sched.GCBudgetCap) leaves the Go collector running back
// to back. See docs/engineering-history/scheduling-and-measurement.md.
func ValidateBudget(text string, fallback, gcCap uint64) (string, error) {
	if strings.TrimSpace(text) == "" {
		return "default: " + session.Bytes(fallback), nil
	}
	n, err := tier.ParseBytes(text)
	if err != nil {
		return "", err
	}
	s := session.Bytes(n)
	if gcCap > 0 && n > gcCap {
		// A live heap on its goal costs a large share of the CPU in GC.
		s += fmt.Sprintf("  -- above the collector cap (%s): the engine will spend much of its time freeing memory",
			session.Bytes(gcCap))
	}
	return s, nil
}

// cpuBrand is the one fact in this report with no in-tree source: jitllm's own
// tuner drops the brand string from its cache key on purpose, because a stepping
// bump invalidates a good answer and the string says nothing about what the
// emitter will do. It is still what a person means by "which CPU is this".
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
				return n * 1024, true
			}
			break
		}
	}
	if s := sysctl("hw.memsize"); s != "" {
		if n, err := strconv.ParseUint(s, 10, 64); err == nil {
			return n, true
		}
	}
	return 0, false
}

func sysctl(key string) string {
	if runtime.GOOS != "darwin" {
		return ""
	}
	out, err := exec.Command("sysctl", "-n", key).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// Host is this machine, as screen.Machine reads it.
type Host struct{}

// Probe is [Probe].
func (Host) Probe() Report { return Probe() }

// Cached is [Cached].
func (Host) Cached() (Report, bool) { return Cached() }
