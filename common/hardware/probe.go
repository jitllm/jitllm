// Package hardware answers "what can this machine run, and what will the next load
// do with it" -- the desktop equivalent of `jitllm hardware`.
//
// It reports rather than probing on its own account: every field is read from
// where the engine reads it (sched, engine/device/cpu, backend.Open, tier), so the answer
// cannot disagree with what a decode would do. Nothing is timed; the memory
// wall is a separate, explicit button.
//
// It is the only screen-side package that imports jitllm. What the UI calls
// here is either a pure string parser or a probe run on a worker goroutine.
package hardware

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/jitllm/jitllm/dev/bench"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/engine/sched"

	"github.com/jitllm/jitllm/common/session"
)

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

	// Arch is the host's architecture, ISA the instruction sets it executes,
	// and Form how the composer forms its CPU device's kernels from them
	// (model.CPUForm).
	Arch, ISA, Form string

	GPUs   []GPU
	Vulkan []VulkanDev
}

// Probe reads the machine.
//
// Call it from a worker, once, and not while a model is loaded: backend.Open
// opens and closes every backend, which contends with a live gpu.GPU. Callers
// serialise their own calls; [Host] does.
func Probe() (r Report) {
	r = Report{
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
	}()

	r.PCores, r.ECores, r.SMT = sched.PCores(), sched.ECores(), sched.SMTSiblings()
	// DecodeCores is the pool the engine actually builds, so this is a report
	// of the shipping configuration rather than of the topology it came from.
	r.Decode = sched.DecodeCores()
	r.RAMTotal, _ = totalRAM()
	r.MemWall = bench.MemWall(max(len(r.Decode), 1))

	r.MemBudget, r.MemLimit = sched.MemBudget(), sched.MemLimit()
	r.GCBudgetCap, r.GCUsable = sched.GCBudgetCap(), sched.GCUsable()

	r.Arch, r.ISA, r.Form = runtime.GOARCH, cpu.Probe().ISA().String(), engine.CPUForm()

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
	for _, g := range r.GPUs {
		m.GPUs = append(m.GPUs, session.GPUInfo{
			Name: g.Name, API: g.API, Slots: g.Slots, Mem: g.Total, Unified: g.Unified,
		})
	}
	return m
}

// ValidateSpec parses a device spec and says what it asks for.
//
// It is gpu.ParseDevices, not a parser of this app's own, so the picker
// cannot accept a spec the engine then refuses.
func ValidateSpec(spec string) (string, error) {
	es, err := gpu.ParseDevices(spec)
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
	pl, err := engine.ParsePlacement(placement, true)
	if err != nil {
		return err
	}
	es, err := gpu.ParseDevices(spec)
	if err != nil {
		return err
	}
	// The CPU is a device every spec opens.
	opened := map[string]bool{nn.CPU: true}
	for _, e := range es {
		if !e.Device() || !e.HasSel {
			continue
		}
		opened[nn.DeviceName(e.API+":"+e.Sel)] = true
	}
	named := []engine.Place{}
	for _, b := range pl.Blocks {
		named = append(named, b)
	}
	for _, x := range []*engine.Place{pl.Rest, pl.Head} {
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
	n, err := gpu.ParseBytes(text)
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

// Host is this machine, as screen.Machine reads it: it remembers the last
// probe and serialises them.
type Host struct {
	mu     sync.Mutex
	cached Report
}

// Probe reads the machine and caches the result.
func (h *Host) Probe() Report {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.cached = Probe()
	return h.cached
}

// Cached returns the last [Host.Probe] and whether there has been one.
func (h *Host) Cached() (Report, bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.cached, h.cached.Probed
}
