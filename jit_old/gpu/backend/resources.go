package backend

import (
	"errors"
	"sort"
	"sync"
)

// KernelResources is what one compiled kernel costs the device it runs on, as
// that device's toolchain allocated it: the per-kernel facts a register report
// is made of, and a kernel composer would select on.
//
// A field the runtime does not report is -1, never a guess.
type KernelResources struct {
	Name string `json:"name"`
	API  string `json:"api"`
	// Group is the declared workgroup width the occupancy was asked for.
	Group int `json:"group"`
	// Regs is registers per thread.
	Regs int `json:"regs"`
	// SpillBytes is per-thread local memory: on generated code with no local
	// arrays, register spill space. Zero is the bar for a hot kernel.
	SpillBytes int `json:"spill_bytes"`
	// SharedBytes is static shared memory per workgroup.
	SharedBytes int `json:"shared_bytes"`
	// MaxGroup is the widest workgroup this kernel can launch with under its
	// register use; a value below Group would not launch at all.
	MaxGroup int `json:"max_group"`
	// Resident is how many workgroups of width Group fit one multiprocessor
	// (or core) at once; ResidentThreads/MaxThreads is the occupancy.
	Resident        int `json:"resident_groups"`
	ResidentThreads int `json:"resident_threads"`
	MaxThreads      int `json:"max_threads"`
	// Compiles counts how many times this name was compiled on the device
	// (one per distinct shape it was specialised to); the other fields are
	// the worst of those compiles: most registers, most spill, least
	// resident.
	Compiles int `json:"compiles"`
	// Ran is whether any compile of this name was launched: a kernel
	// compiled and passed over (a spilling tile) is reported with Ran false.
	Ran bool `json:"ran"`
}

// Occupancy is ResidentThreads over MaxThreads, or -1 when either is unknown.
func (r KernelResources) Occupancy() float64 {
	if r.ResidentThreads < 0 || r.MaxThreads <= 0 {
		return -1
	}
	return float64(r.ResidentThreads) / float64(r.MaxThreads)
}

// ResourceReporter is a device that can say what each kernel it compiled
// costs. CUDA answers from the driver (cuFuncGetAttribute and
// cuOccupancyMaxActiveBlocksPerMultiprocessor); a backend that cannot read its
// compiler's allocation does not implement it, and the report says so rather
// than printing a guess.
type ResourceReporter interface {
	// KernelResources lists every kernel name compiled on this device so far,
	// worst compile per name, sorted by name.
	KernelResources() []KernelResources
}

// resourced is a compiled kernel that knows what its compile cost.
type resourced interface {
	resources() (KernelResources, bool)
}

// KernelResourcesOf is what compiling k cost its device, for a JIT choosing
// between variants by count; ok is false where the backend cannot say.
func KernelResourcesOf(k Kernel) (r KernelResources, ok bool) {
	if rk, is := k.(resourced); is {
		return rk.resources()
	}
	return r, false
}

// resLedger is the per-device record behind ResourceReporter.
type resLedger struct {
	mu sync.Mutex
	by map[string]KernelResources
}

func (l *resLedger) note(r KernelResources) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.by == nil {
		l.by = map[string]KernelResources{}
	}
	old, ok := l.by[r.Name]
	if !ok {
		r.Compiles = 1
		l.by[r.Name] = r
		return
	}
	old.Compiles++
	if r.Regs > old.Regs {
		old.Regs = r.Regs
	}
	if r.SpillBytes > old.SpillBytes {
		old.SpillBytes = r.SpillBytes
	}
	if r.SharedBytes > old.SharedBytes {
		old.SharedBytes = r.SharedBytes
	}
	if r.Occupancy() < old.Occupancy() {
		old.Group, old.Resident, old.ResidentThreads, old.MaxGroup = r.Group, r.Resident, r.ResidentThreads, r.MaxGroup
	}
	l.by[r.Name] = old
}

func (l *resLedger) ran(name string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if r, ok := l.by[name]; ok {
		r.Ran = true
		l.by[name] = r
	}
}

func (l *resLedger) list() []KernelResources {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make([]KernelResources, 0, len(l.by))
	for _, r := range l.by {
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// ErrGroupTooWide is a compile refused because the kernel's register use
// leaves the device unable to launch its declared workgroup width: the launch
// would fail with too many resources requested, so the compile says so first
// and a caller with another kernel for the job keeps that one.
var ErrGroupTooWide = errors.New("backend: the workgroup is wider than the kernel's registers allow")
