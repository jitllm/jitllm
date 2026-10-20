package tier

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/engine/sched"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/gpu/metal"
	"github.com/jitllm/jitllm/jit/gpu/vulkan"
)

// This file is the opening half of the tier: which devices, what each one may
// spend, and out of whose memory. The router in multi.go and the residency in
// layer.go take those answers as given. The decisions are options, not
// environment variables, so an embedder can make them without mutating the
// process environment; an explicit option wins over the variables.

// closeSlots releases devices the caller handed over. See OpenWith's ownership
// rule: every error path closes what it was given, exactly once.
func closeSlots(slots []Slot) {
	for _, s := range slots {
		if s.Dev != nil {
			s.Dev.Close()
		}
	}
}

// Option configures OpenWith.
type Option func(*openOpts)

type openOpts struct {
	spec    string
	budget  uint64
	host    uint64
	hostSet bool
	slots   []Slot

	// The knobs that were environment variables; see knobs.go.
	kb         knobs
	vk         vulkan.Config
	kern       kernels.Center
	cudaPosted bool
	cudaRegs   int
	metal      metal.Opts
	hip        backend.HIPConfig
	cfg        []func(*Config)
	verbose    bool
	verboseSet bool
	// pools is the registry the tier's discrete cards take their pools from
	// (WithPoolRegistry); nil, each tier its own.
	pools *PoolRegistry
	// forms is the formed kernels shared with other tiers (WithFormCache).
	forms *FormCache
	// op reaches the backends' constructors; the zero value is the real ones.
	op opener
}

// WithDevices names the devices to place blocks on, in the grammar the -devices
// flag takes (see ParseDevices).
func WithDevices(spec string) Option { return func(o *openOpts) { o.spec = spec } }

// WithBudget caps the resident weights on every device that the spec did not
// give a budget of its own (`cuda:0=3G`). Zero, the default, asks each device
// how much memory it has; see budgetFor.
func WithBudget(bytes uint64) Option { return func(o *openOpts) { o.budget = bytes } }

// WithHostBudget states the host's weight budget, out of which the devices whose
// memory is the host's are carved. An integrated GPU reports the host's free
// memory as its own, so the tier caps the unified devices at a share of this
// budget, and HostReserved reports what they actually hold for the host to
// subtract.
// Unset (or zero), it asks sched.MemBudget; zero never means "unlimited".
func WithHostBudget(bytes uint64) Option {
	return func(o *openOpts) { o.host, o.hostSet = bytes, true }
}

// WithSlots hands the tier devices the caller opened itself, with whatever
// budgets and pools it decided, without describing them in a string first.
// It is exclusive with WithDevices.
func WithSlots(slots ...Slot) Option { return func(o *openOpts) { o.slots = slots } }

// OpenWith opens the devices an option set names, budgets them, pools them and
// returns the tier.
//
// OpenWith owns what it is given and what it opens, and closes both on every
// error, so `if err != nil` never leaks; the Closes are idempotent, so a caller
// that also closes defensively is safe.
//
// "auto" means one device here and a bare "gpu" every GPU; a host with no GPU
// is an error from both, and the caller decides whether that means "a CPU
// machine" (auto) or a refusal (gpu).
func OpenWith(options ...Option) (*GPU, error) {
	var o openOpts
	for _, f := range options {
		f(&o)
	}
	o.apply()
	if len(o.slots) > 0 {
		if o.spec != "" {
			// OpenWith owns WithSlots devices from the moment it is called
			// and closes them on this error too, the same rule as New's error
			// path: two errors with opposite ownership rules leaked a device.
			closeSlots(o.slots)
			return nil, fmt.Errorf("tier: WithSlots and WithDevices(%q) both name the devices; "+
				"pass one of them", o.spec)
		}
		return New(o.slots, options...)
	}
	es, err := ParseDevices(o.spec)
	if err != nil {
		return nil, err
	}
	devs, asks, names, err := openEntries(es, o)
	if err != nil {
		return nil, err
	}
	if len(devs) == 0 {
		return nil, fmt.Errorf("tier: -devices %q asks for no device at all", specText(o.spec))
	}
	if err := noDuplicates(devs, o.spec); err != nil {
		for _, d := range devs {
			d.Close()
		}
		return nil, err
	}
	slots := planSlots(devs, asks, o)
	for i := range slots {
		slots[i].Name = names[i]
	}
	return New(slots, options...)
}

// ParseDevices parses the device spec and reports what is wrong with it. The
// grammar, one line per form:
//
//	auto                 one device, chosen by measuring a real matvec on each
//	                     backend -- and a machine with none is a CPU machine
//	all                  every distinct device on the host, fastest first
//	cpu                  no device; the host runs the whole model
//	gpu                  every GPU, as all; a machine with none is an error
//	gpu:N                the Nth device backend.Open reports, counting from 0
//	cuda[=BYTES]         every CUDA device, fastest first
//	cuda:ORD[=BYTES]     one CUDA device by driver ordinal
//	vulkan[=BYTES]       every Vulkan GPU (software rasterisers left out),
//	                     fastest first; one alone when the Vulkan config pins it
//	vulkan:SEL[=BYTES]   one Vulkan device by enumeration index or name substring
//	metal[=BYTES]        the system default Metal device
//	hip[=BYTES]          every AMD device ROCm reports, fastest first
//	hip:ORD[=BYTES]      one AMD device by HIP ordinal (also rocm, amdgcn); an
//	                     ordinal ROCm does not have, or no ROCm, is an error
//
// A bare backend name is every device of that backend, and the router decides
// how many of them a model's blocks use (multi.go offers each block fastest
// first); a selector pins exactly one. A comma-separated list of the last
// three places blocks across all of them, in the order written, fastest first
// inside that order. The names jitllm's own reports print (ptx, spirv, msl)
// are accepted too.
//
// BYTES is a byte count with an optional unit: 3G, 3GiB, 3.5g, 512M, 800000000.
//
// cpu in a list is not a device: whatever no device takes runs on the host
// anyway. Only `-devices cpu` alone changes anything: offer the model to
// nothing.
func ParseDevices(spec string) ([]Entry, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		spec = "auto"
	}
	var out []Entry
	for i, raw := range strings.Split(spec, ",") {
		text := strings.TrimSpace(raw)
		if text == "" {
			return nil, fmt.Errorf("tier: -devices %q: entry %d is empty; want %s",
				spec, i+1, grammar)
		}
		e := Entry{Text: text}
		body := text
		if j := strings.IndexByte(body, '='); j >= 0 {
			n, err := ParseBytes(body[j+1:])
			if err != nil {
				return nil, fmt.Errorf("tier: -devices %q: %w", spec, err)
			}
			e.Bytes, body = n, body[:j]
		}
		if j := strings.IndexByte(body, ':'); j >= 0 {
			e.Sel, e.HasSel, body = body[j+1:], true, body[:j]
		}
		switch strings.ToLower(body) {
		case "cpu", "auto", "all", "gpu":
			e.API = strings.ToLower(body)
		case "cuda", "ptx":
			e.API = "ptx"
		case "vulkan", "spirv":
			e.API = "spirv"
		case "metal", "msl":
			e.API = "msl"
		case "hip", "rocm", "amdgcn":
			e.API = "amdgcn"
		default:
			return nil, fmt.Errorf("tier: -devices %q: %q names no backend; want %s",
				spec, body, grammar)
		}
		if e.HasSel && e.Sel == "" {
			return nil, fmt.Errorf("tier: -devices %q: %q ends in a colon with no device after it; want %s",
				spec, text, grammar)
		}
		switch e.API {
		case "cpu", "auto", "all":
			if e.HasSel {
				return nil, fmt.Errorf("tier: -devices %q: %q takes no device selector, got %q; want %s",
					spec, body, e.Sel, grammar)
			}
		case "msl":
			if e.HasSel {
				return nil, fmt.Errorf("tier: -devices %q: metal opens the system default device "+
					"and takes no selector, got %q; want %s", spec, e.Sel, grammar)
			}
		case "ptx", "gpu", "amdgcn":
			// An ordinal, always: the CUDA driver enumerates by number and
			// gpu:N indexes the list backend.Open reports.
			if e.HasSel {
				if _, err := strconv.Atoi(e.Sel); err != nil {
					return nil, fmt.Errorf("tier: -devices %q: %s takes a device ORDINAL, got %q; "+
						"a name substring selects a Vulkan device (vulkan:%s), not this one",
						spec, body, e.Sel, e.Sel)
				}
			}
		}
		if e.API == "cpu" && e.Bytes > 0 {
			return nil, fmt.Errorf("tier: -devices %q: %q gives the host a DEVICE budget; "+
				"the host's own budget is -maxmem", spec, text)
		}
		out = append(out, e)
	}
	// auto, all and gpu each mean "work the rest out", so a second device beside
	// one of them is two answers to one question. cpu is allowed anywhere.
	for _, e := range out {
		if e.API != "auto" && e.API != "all" && e.API != "gpu" {
			continue
		}
		for _, o := range out {
			if o.API == "cpu" || o.Text == e.Text {
				continue
			}
			return nil, fmt.Errorf("tier: -devices %q: %q already says which devices to use, "+
				"so it cannot be listed beside %q; name every device, or use %q alone",
				spec, e.Text, o.Text, e.Text)
		}
	}
	return out, nil
}

const grammar = "auto | all | cpu | gpu[:N] | (cuda|hip|vulkan|metal)[:DEVICE][=BYTES], comma-separated"

// Entry is one parsed device spec.
type Entry struct {
	// API is "cpu", "auto", "all", "gpu", or one of the backend API names --
	// "ptx", "spirv", "msl" -- whichever spelling was written.
	API string
	// Sel is the device selector: an ordinal for CUDA and gpu:N, an index or a
	// name substring for Vulkan. HasSel separates "no selector" from ":0".
	Sel    string
	HasSel bool
	// Bytes is this device's weight budget from `=BYTES`, or 0 for "not said".
	Bytes uint64
	// Text is the entry as written, so an error and a report can quote what the
	// caller actually typed.
	Text string
}

// name is what a placement calls the device this entry opened, or "" when the
// entry opened several (all, a bare gpu, cuda or vulkan on a host with more
// than one) or chose one itself (auto).
func (e Entry) name(opened int) string {
	if opened != 1 {
		return ""
	}
	switch e.API {
	case "ptx":
		if !e.HasSel {
			return "cuda:0"
		}
		return "cuda:" + e.Sel
	case "spirv":
		if !e.HasSel {
			return "vulkan"
		}
		return "vulkan:" + e.Sel
	case "msl":
		return "metal"
	case "amdgcn":
		if !e.HasSel {
			return "hip:0"
		}
		return "hip:" + e.Sel
	case "gpu":
		if e.HasSel {
			return "gpu:" + e.Sel
		}
	}
	return ""
}

// DeviceName is nn.DeviceName, here because the tier names its devices.
func DeviceName(s string) string { return nn.DeviceName(s) }

// Device reports whether this entry asks for a device at all: "cpu" does not.
func (e Entry) Device() bool { return e.API != "cpu" }

// AdmitsHost reports whether a device spec lets blocks run on the host: "auto",
// or a list that names cpu. Relocation hands a device's blocks to the host
// when its history outgrows the card, so it is on exactly when this is true;
// a spec of devices alone is a caller keeping the model off the CPU.
func AdmitsHost(spec string) (bool, error) {
	es, err := ParseDevices(spec)
	if err != nil {
		return false, err
	}
	for _, e := range es {
		if e.API == "auto" || !e.Device() {
			return true, nil
		}
	}
	return false, nil
}

// ParseBytes reads a byte count with an optional unit: 3G, 3GiB, 3.5g, 512M,
// 800000000. The units are powers of 1024, because every other byte figure
// jitllm prints is.
func ParseBytes(s string) (uint64, error) {
	t := strings.TrimSpace(s)
	if t == "" {
		return 0, fmt.Errorf("%q is not a byte count; want a number with an optional K/M/G/T", s)
	}
	shift := 0
	u := strings.ToUpper(t)
	u = strings.TrimSuffix(u, "B")
	u = strings.TrimSuffix(u, "I")
	if len(u) > 0 {
		switch u[len(u)-1] {
		case 'K':
			shift, u = 10, u[:len(u)-1]
		case 'M':
			shift, u = 20, u[:len(u)-1]
		case 'G':
			shift, u = 30, u[:len(u)-1]
		case 'T':
			shift, u = 40, u[:len(u)-1]
		}
	}
	f, err := strconv.ParseFloat(strings.TrimSpace(u), 64)
	if err != nil || f < 0 {
		return 0, fmt.Errorf("%q is not a byte count; want a number with an optional K/M/G/T", s)
	}
	n := f * float64(uint64(1)<<shift)
	if n < 1 {
		return 0, fmt.Errorf("%q rounds to zero bytes, which is not a budget", s)
	}
	return uint64(n), nil
}

// openEntries opens every device the entries name, in the order written, and
// closes the ones it did open if a later one fails: a half-opened list is a
// leak (a context, a locked OS thread, an owner goroutine), not a partial
// success.
func openEntries(es []Entry, o openOpts) ([]backend.Device, []uint64, []string, error) {
	do := o.devOpts()
	var out []backend.Device
	var asks []uint64
	var names []string
	fail := func(err error) ([]backend.Device, []uint64, []string, error) {
		for _, d := range out {
			d.Close()
		}
		return nil, nil, nil, err
	}
	// took records the entry's own `=BYTES` against every device it opened, so
	// `all=3G` gives three devices 3G each and `cuda:0=3G,vulkan:1=8G` gives one
	// each -- the budget follows the device it was written beside.
	took := func(e Entry, ds ...backend.Device) {
		for _, d := range ds {
			out = append(out, d)
			asks = append(asks, e.Bytes)
			names = append(names, e.name(len(ds)))
		}
	}
	for _, e := range es {
		api := e.API
		if api == "gpu" && !e.HasSel {
			api = "all" // every GPU
		}
		switch api {
		case "cpu":
			continue
		case "auto", "gpu":
			devs := backend.OpenWith(do)
			if len(devs) == 0 {
				return fail(fmt.Errorf("tier: no GPU backend on this host"))
			}
			if e.HasSel {
				n, _ := strconv.Atoi(e.Sel) // ParseDevices checked it
				if n < 0 || n >= len(devs) {
					names := make([]string, len(devs))
					for i, d := range devs {
						names[i] = fmt.Sprintf("%d:%s [%s]", i, d.Name(), d.API())
					}
					for _, d := range devs {
						d.Close()
					}
					return fail(fmt.Errorf("tier: -devices %q: device %d of %d; this host has %s",
						e.Text, n, len(devs), strings.Join(names, ", ")))
				}
				for i, d := range devs {
					if i != n {
						d.Close()
					}
				}
				took(e, devs[n])
				continue
			}
			took(e, choose(devs, o.kb))
		case "all":
			devs, err := o.op.allDevices(do)
			if err != nil {
				return fail(err)
			}
			took(e, order(devs, o.kb.tune)...)
		case "ptx":
			if !e.HasSel {
				devs, err := o.op.everyCUDA(do)
				if err != nil {
					return fail(fmt.Errorf("tier: -devices %q: %w", e.Text, err))
				}
				took(e, order(devs, o.kb.tune)...)
				continue
			}
			ord, _ := strconv.Atoi(e.Sel)
			d, err := o.op.newCUDA(ord, do)
			if err != nil {
				return fail(fmt.Errorf("tier: -devices %q: %w", e.Text, err))
			}
			took(e, d)
		case "spirv":
			if !e.HasSel && do.Vulkan.Device == "" {
				devs, err := o.op.everyVulkan(do)
				if err != nil {
					return fail(fmt.Errorf("tier: -devices %q: %w", e.Text, err))
				}
				took(e, order(devs, o.kb.tune)...)
				continue
			}
			d, err := o.op.newVulkan(e.Sel, do)
			if err != nil {
				return fail(fmt.Errorf("tier: -devices %q: %w", e.Text, err))
			}
			took(e, d)
		case "amdgcn":
			if !e.HasSel {
				devs, err := o.op.everyHIP(do)
				if err != nil {
					return fail(fmt.Errorf("tier: -devices %q: %w", e.Text, err))
				}
				took(e, order(devs, o.kb.tune)...)
				continue
			}
			ord, _ := strconv.Atoi(e.Sel)
			d, err := o.op.newHIP(ord, do)
			if err != nil {
				return fail(fmt.Errorf("tier: -devices %q: %w", e.Text, err))
			}
			took(e, d)
		case "msl":
			d, err := o.op.newMetal(do)
			if err != nil {
				return fail(fmt.Errorf("tier: -devices %q: %w", e.Text, err))
			}
			took(e, d)
		}
	}
	return out, asks, names, nil
}

// noDuplicates refuses a spec that named the same device twice: two contexts
// on one card would each be budgeted the whole card's free memory, which fails
// halfway through a block rather than degrading.
func noDuplicates(devs []backend.Device, spec string) error {
	for i, a := range devs {
		for j := 0; j < i; j++ {
			b := devs[j]
			if a.API() == b.API() && a.Name() == b.Name() && ordinalOf(a) == ordinalOf(b) {
				return fmt.Errorf("tier: -devices %q names %s [%s] twice (entries %d and %d); "+
					"one device is one budget, and two contexts on it would each be given "+
					"the whole card", specText(spec), a.Name(), a.API(), j+1, i+1)
			}
		}
	}
	return nil
}

// ordinalOf is the device's index within its backend, or -1 when the backend
// does not say. It is what tells two identical cards apart, since their names
// are the same string.
func ordinalOf(d backend.Device) int {
	if o, ok := d.(backend.OrdinalDevice); ok {
		return o.Ordinal()
	}
	return -1
}

// opener is how the tier reaches the backends' device constructors: its zero
// value is the real backends, and a gate puts a fake host behind a spec, and
// counts what the spec resolves to, by setting fields through withOpener. It
// rides in openOpts, so one tier's fake host is no other tier's.
type opener struct {
	cudaCount     func() (int, error)
	openCUDA      func(int, backend.Opts) (backend.Device, error)
	vulkanDevices func() ([]vulkan.Info, error)
	openVulkan    func(string, backend.Opts) (backend.Device, error)
	openMetal     func(backend.Opts) (backend.Device, error)
	hipCount      func(backend.HIPConfig) int
	openHIP       func(int, backend.Opts) (backend.Device, error)
}

func (p opener) countCUDA() (int, error) {
	if p.cudaCount != nil {
		return p.cudaCount()
	}
	return backend.CUDACount()
}

func (p opener) newCUDA(ord int, do backend.Opts) (backend.Device, error) {
	if p.openCUDA != nil {
		return p.openCUDA(ord, do)
	}
	return backend.OpenCUDAWith(ord, do)
}

func (p opener) listVulkan() ([]vulkan.Info, error) {
	if p.vulkanDevices != nil {
		return p.vulkanDevices()
	}
	return backend.VulkanDevices()
}

func (p opener) newVulkan(sel string, do backend.Opts) (backend.Device, error) {
	if p.openVulkan != nil {
		return p.openVulkan(sel, do)
	}
	return backend.OpenVulkanWith(sel, do)
}

func (p opener) newMetal(do backend.Opts) (backend.Device, error) {
	if p.openMetal != nil {
		return p.openMetal(do)
	}
	return backend.OpenMetalWith(do)
}

func (p opener) countHIP(c backend.HIPConfig) int {
	if p.hipCount != nil {
		return p.hipCount(c)
	}
	return backend.HIPCount(c)
}

func (p opener) newHIP(ord int, do backend.Opts) (backend.Device, error) {
	if p.openHIP != nil {
		return p.openHIP(ord, do)
	}
	return backend.OpenHIPWith(ord, do)
}

// withOpener replaces the backends' constructors on the tier this option opens.
func withOpener(p opener) Option { return func(o *openOpts) { o.op = p } }

// allDevices opens every distinct device on this host. It enumerates rather
// than calling backend.Open, which returns one device per backend.
//
// Software rasterisers are skipped, as backend.openVulkan skips them: they are
// strictly worse than the CPU tier. Naming one explicitly still works
// (`-devices vulkan:llvmpipe`).
func (p opener) allDevices(do backend.Opts) ([]backend.Device, error) {
	out := append(p.cudaDevices(do), p.hipDevices(do)...)
	out = append(out, p.vulkanGPUs(do)...)
	if d, err := p.newMetal(enumerating(do)); err == nil {
		out = append(out, d)
	}
	out = dropSecondAPI(out)
	if len(out) == 0 {
		return nil, fmt.Errorf("tier: no GPU backend on this host")
	}
	return out, nil
}

// cudaDevices opens every CUDA device that opens, by ordinal.
func (p opener) cudaDevices(do backend.Opts) []backend.Device {
	do.Enumerating = true
	var out []backend.Device
	if n, err := p.countCUDA(); err == nil {
		for i := 0; i < n; i++ {
			if d, err := p.newCUDA(i, do); err == nil {
				out = append(out, d)
			}
		}
	}
	return out
}

// hipDevices opens every HIP device that opens, by ordinal. No ROCm is none.
func (p opener) hipDevices(do backend.Opts) []backend.Device {
	do.Enumerating = true
	var out []backend.Device
	for i := 0; i < p.countHIP(do.HIP); i++ {
		if d, err := p.newHIP(i, do); err == nil {
			out = append(out, d)
		}
	}
	return out
}

// everyHIP is a bare `hip`: every HIP device. With none, the error is the one
// ordinal 0 gives, which names why (no ROCm and where it was looked for, or no
// device).
func (p opener) everyHIP(do backend.Opts) ([]backend.Device, error) {
	if out := p.hipDevices(do); len(out) > 0 {
		return out, nil
	}
	do.Enumerating = true
	d, err := p.newHIP(0, do)
	if err != nil {
		return nil, err
	}
	return []backend.Device{d}, nil
}

// vulkanGPUs opens every Vulkan device that computes and is not a software
// rasteriser.
func (p opener) vulkanGPUs(do backend.Opts) []backend.Device {
	do.Enumerating = true
	var out []backend.Device
	if infos, err := p.listVulkan(); err == nil {
		for _, in := range infos {
			if !in.Compute || in.Software {
				continue
			}
			if d, err := p.newVulkan(strconv.Itoa(in.Index), do); err == nil {
				out = append(out, d)
			}
		}
	}
	return out
}

// everyCUDA is a bare `cuda`: every CUDA device. With none open, the error is
// the one ordinal 0 gives, which says why (no driver, no device).
func (p opener) everyCUDA(do backend.Opts) ([]backend.Device, error) {
	if out := p.cudaDevices(do); len(out) > 0 {
		return out, nil
	}
	do.Enumerating = true
	d, err := p.newCUDA(0, do)
	if err != nil {
		return nil, err
	}
	return []backend.Device{d}, nil
}

// everyVulkan is a bare `vulkan`: every Vulkan GPU. With none, it is the
// device the Vulkan backend picks on its own, or the error saying why there is
// none.
func (p opener) everyVulkan(do backend.Opts) ([]backend.Device, error) {
	if out := p.vulkanGPUs(do); len(out) > 0 {
		return out, nil
	}
	do.Enumerating = true
	d, err := p.newVulkan("", do)
	if err != nil {
		return nil, err
	}
	return []backend.Device{d}, nil
}

// Merge records that one physical device was enumerated by two backends, and
// which of the two the default set kept. It is returned so a gate can assert
// what the dedupe did rather than infer it from a device count.
type Merge struct {
	Kept    string // the device label that stays in the default set
	Dropped string // the device label that does not
	KeptAPI string
	DropAPI string
	ID      string // the identity both of them reported
}

func (m Merge) String() string {
	return fmt.Sprintf("%s [%s] and %s [%s] are one device (%s); keeping %s",
		m.Kept, m.KeptAPI, m.Dropped, m.DropAPI, m.ID, m.KeptAPI)
}

// preferOnePerCard returns the index of one device per piece of physical
// hardware, and the merges it made. It closes nothing; the caller owns every
// device and closes by position (choose keeps one slot and closes the rest),
// which is why it returns indices. A set keyed on backend.Device would also
// panic on a non-comparable implementation.
//
// The key is the device UUID (backend.Identity), never the name: two identical
// cards share a name, and merging them would delete half the machine's VRAM. A
// device that cannot supply an identity stays distinct.
//
// The one kept is the preferred backend (backend.APIRank: CUDA or Metal over
// Vulkan for the same card), not the first enumerated; a tie keeps the earlier
// entry.
func preferOnePerCard(devs []backend.Device) ([]int, []Merge) {
	keep := make([]int, 0, len(devs))
	var merges []Merge
	for i, d := range devs {
		id := backend.IdentityOf(d)
		dup := -1
		if id.Known() {
			for n, j := range keep {
				if backend.IdentityOf(devs[j]).Same(id) {
					dup = n
					break
				}
			}
		}
		if dup < 0 {
			keep = append(keep, i)
			continue
		}
		k := devs[keep[dup]]
		if backend.APIRank(d.API()) > backend.APIRank(k.API()) {
			keep[dup] = i
			merges = append(merges, Merge{Kept: d.Name(), Dropped: k.Name(),
				KeptAPI: d.API(), DropAPI: k.API(), ID: id.Key})
			continue
		}
		merges = append(merges, Merge{Kept: k.Name(), Dropped: d.Name(),
			KeptAPI: k.API(), DropAPI: d.API(), ID: id.Key})
	}
	return keep, merges
}

// dropSecondAPI keeps one device per card and closes the rest. Pooling keeps
// the accounting right (planSlots); this also saves the slower second context
// and its driver overhead.
func dropSecondAPI(devs []backend.Device) []backend.Device {
	keep, merges := preferOnePerCard(devs)
	kept := make([]bool, len(devs))
	out := make([]backend.Device, 0, len(keep))
	for _, i := range keep {
		kept[i] = true
		out = append(out, devs[i])
	}
	for i, d := range devs {
		if !kept[i] {
			d.Close()
		}
	}
	if backend.Verbose() {
		for _, m := range merges {
			fmt.Fprintf(os.Stderr, "jitllm: %s\n", m)
		}
	}
	return out
}

// cardName strips the decoration jitllm's own backends add: cudaDev.Name is
// "#0 <card name> (sm_86)" and vkDev.Name is "<card name>" for the same
// card. It only names a pool; identity decides which devices share one
// (preferOnePerCard).
func cardName(s string) string {
	if strings.HasPrefix(s, "#") {
		if i := strings.IndexByte(s, ' '); i >= 0 {
			s = s[i+1:]
		}
	}
	if i := strings.LastIndex(s, " ("); i >= 0 && strings.HasSuffix(s, ")") {
		s = s[:i]
	}
	return strings.TrimSpace(s)
}

// hostPoolName is what a report calls the pool an integrated GPU spends from.
const hostPoolName = "host RAM"

// minBudget is the floor a derived budget lands on when the device says it has
// almost nothing left. It deliberately holds nothing useful, so the tier
// declines blocks in the ordinary way rather than claiming room the card does
// not have.
const minBudget = 64 << 20

// planSlots gives every device a budget and the pool it spends from. asks[i] is
// device i's own `=BYTES` from the spec, or 0 when it said nothing.
//
// The pool is the heap, not the device (see Pool): an integrated GPU's memory
// is system memory, so its budget and the host's are claims on the same bytes.
// Devices land on the host pool if they answer UnifiedMemory, a shared card
// pool when one card is reached through two APIs, and a pool of their own
// otherwise.
func planSlots(devs []backend.Device, asks []uint64, o openOpts) []Slot {
	host := o.host
	if !o.hostSet {
		host = sched.MemBudget()
	}
	if host == 0 {
		host = unifiedTotal(devs)
	}
	share := hostShare(host, o.budget)
	slots := make([]Slot, len(devs))
	for i, d := range devs {
		ask, src := o.budget, "-vram"
		if i < len(asks) && asks[i] > 0 {
			ask, src = asks[i], "the spec"
		}
		limit, why := budgetFor(d, ask, src)
		if unified(d) && limit > share {
			// The carve-out is the ceiling: the device reports the host's free
			// memory, and a budget derived from that would hand one device most
			// of the machine.
			limit, why = share, whyShare(o)
		}
		slots[i] = Slot{Dev: d, Bytes: limit, Why: why}
	}
	// One pool per heap, in two passes: group the devices that share bytes
	// (every unified device shares the host's; devices reporting the same
	// backend.Identity share a card's), then size each group's pool from the
	// largest budget in it -- the members are claims on the same memory, so
	// the pool holds the largest of them and never their sum. A device with no
	// identity gets a pool of its own.
	group := make([]int, len(devs)) // which pool each device belongs to
	hostIdx := -1
	byID := map[string]int{}
	for i, d := range devs {
		group[i] = -1
		if unified(d) {
			if hostIdx < 0 {
				hostIdx = i
			}
			group[i] = hostIdx
			continue
		}
		if id := backend.IdentityOf(d); id.Known() {
			if j, ok := byID[id.Key]; ok {
				group[i] = j
			} else {
				byID[id.Key], group[i] = i, i
			}
		}
		if group[i] < 0 {
			group[i] = i
		}
	}
	pools := make([]*Pool, len(devs))
	for i := range devs {
		g := group[i]
		if pools[g] == nil {
			if g == hostIdx {
				pools[g] = NewHostPool(share)
			} else {
				// The card's name, or the device's own when it has no card
				// name to strip.
				n := cardName(devs[g].Name())
				if n == "" {
					n = devs[g].Name()
				}
				pools[g] = NewPool(n, slots[g].Bytes)
			}
		}
		// A later member with a bigger budget raises the heap's ceiling.
		if !pools[g].Host() && pools[g].Limit() < slots[i].Bytes {
			pools[g] = NewPool(pools[g].Name(), slots[i].Bytes)
		}
	}
	if r := o.pools; r != nil {
		// One pool per physical card across every tier opened with the
		// registry: what one model gives back, another takes. The card's
		// device limit is the pool's, so the pool is the gate.
		for g, p := range pools {
			if p == nil || p.Host() {
				continue
			}
			id := backend.IdentityOf(devs[g])
			if !id.Known() {
				continue
			}
			pools[g] = r.poolFor(id.Key, p)
		}
		for i := range devs {
			if p := pools[group[i]]; p != nil && !p.Host() && slots[i].Bytes < p.Limit() {
				slots[i].Bytes = p.Limit()
			}
		}
	}
	for i := range devs {
		slots[i].Pool = pools[group[i]]
	}
	return slots
}

// hostShare is how much of the host's weight budget devices whose memory is the
// host's may hold between them: all of it, less the headroom budgetFor leaves
// on a card (an eighth, at least 256 MiB, at most 1 GiB and at most half). The
// device and the host are one pool, so a model that fits the host budget
// belongs on the device. Nothing is counted twice: the host loses what the
// device actually holds (HostReserved), never this ceiling. The headroom is
// what the host side keeps when the device is full -- the pager's frames for
// the blocks the device declined, and its own scratch.
//
// A quarter of the host budget stood here; on a 16 GiB M4 it was 1.92 GiB and
// put the last block and the head of a 2.26 GiB model on the CPU by default.
//
// A per-device budget in the spec (`vulkan:1=8G`) overrides it. An explicit
// -vram (WithBudget) is taken at face value for the pool too, and unified
// devices share that number rather than each getting it.
func hostShare(host, budget uint64) uint64 {
	if budget > 0 {
		return budget
	}
	if host == 0 {
		// sched.MemBudget returns 0 when it cannot read a limit at all, which
		// is "unknown" and not "unlimited".
		return defaultBudget
	}
	return host - min(max(host/8, 256<<20), maxHeadroom, host/2)
}

// unifiedTotal is the memory the first unified device says the GPU may hold,
// or 0 for none: the host's own statement of it (Metal's
// recommendedMaxWorkingSetSize), taken when the host budget is unknown --
// sched.MemBudget is 0 on darwin, where a 2 GiB default would sit below a
// small model's blocks, head and prompt scratch once the scratch is charged.
func unifiedTotal(devs []backend.Device) uint64 {
	for _, d := range devs {
		if !unified(d) {
			continue
		}
		if _, total, err := d.Mem(); err == nil {
			return total
		}
	}
	return 0
}

func whyShare(o openOpts) string {
	if o.budget > 0 {
		return "-vram, shared by every device on the host's memory"
	}
	return "the host budget less its headroom, shared by every device on the host's memory"
}

// budgetFor is what one device may spend on resident weights, and why: free
// memory less headroom, unless the caller asked for a figure.
//
// The headroom is an eighth of free, at least 256 MiB, at most half and at most
// 1 GiB. It covers staging buffers, compiled modules and driver allocations,
// and it is a fraction because `free` is a snapshot of a machine where others
// may allocate next; overshooting VRAM fails rather than degrades. "At most
// half" is for a nearly full card, which then declines blocks per layer. "At
// most 1 GiB" is because an eighth of a big card is a whole block: on a 16 GiB
// card it cost a large mixture two blocks to the host, while the peak beyond
// the charged budget (with the widest batched mixture scratch) was ~0.5 GiB.
func budgetFor(d backend.Device, ask uint64, src string) (uint64, string) {
	if ask > 0 {
		return ask, src
	}
	free, total, err := d.Mem()
	if err != nil {
		return defaultBudget, fmt.Sprintf("the default: %v", err)
	}
	if free == 0 {
		return minBudget, fmt.Sprintf("the floor: the device reports 0 of %.2f GiB free",
			float64(total)/(1<<30))
	}
	head := min(free/8, maxHeadroom)
	if head < 256<<20 {
		head = 256 << 20
	}
	if head > free/2 {
		head = free / 2
	}
	limit := free - head
	if limit < minBudget {
		limit = minBudget
	}
	return limit, fmt.Sprintf("%.2f GiB free less %.2f GiB headroom",
		float64(free)/(1<<30), float64(head)/(1<<30))
}

// maxHeadroom caps budgetFor's headroom; see there.
const maxHeadroom = 1 << 30

func unified(d backend.Device) bool {
	u, ok := d.(backend.Unified)
	return ok && u.UnifiedMemory()
}

func specText(s string) string {
	if strings.TrimSpace(s) == "" {
		return "auto"
	}
	return s
}

// Budget is one device's allowance and where it comes from, so a report can
// say "out of the card" or "out of the host's memory" beside the figure.
type Budget struct {
	Device string // the device label, ordinal included
	Name   string // the name -devices gave it (cuda:0, vulkan:1), as a placement names it
	Limit  uint64 // resident weight bytes this device may hold
	Used   uint64 // what it has charged against Limit
	Why    string // where Limit came from, in words
	Pool   string // the heap Limit is spent from
	// PoolLimit is what that heap allows in total. It is smaller than the sum of
	// its devices' limits exactly when the devices share bytes.
	PoolLimit uint64
	// Host is set when the pool is the host's memory, so these bytes come off
	// the host's own budget.
	Host bool
}

// DeviceFree is what each device's driver reports free, fastest device first,
// 0 where it cannot say. It asks the driver, where Budgets reads only the
// tier's own ledger: a buffer the ledger never charged still shows here.
func (g *GPU) DeviceFree() []uint64 {
	out := make([]uint64, len(g.devs))
	for i, d := range g.devs {
		if free, _, err := d.dev.Mem(); err == nil {
			out[i] = free
		}
	}
	return out
}

// Hardware is the physical device behind one device of the tier, whatever
// spec opened it: `cuda`, `cuda:1`, `gpu` and `vulkan:N` reaching one card all
// give the same Key.
type Hardware struct {
	// Key is the device's identity (backend.Identity: the UUID CUDA and
	// Vulkan both report for one card, so one card under two APIs is one
	// Key), or Ref when the backend cannot say.
	Key string
	// Ref is the backend and the device's ordinal in it, as -devices names it
	// ("cuda:1", "vulkan:0", "metal"); the backend name and the device name
	// when the backend reports no ordinal.
	Ref string
	// Name is what -devices called it (Budget.Name): "cuda:1", "gpu:0", or ""
	// when the entry opened several devices.
	Name string
}

// Hardware names the physical device behind each of the tier's devices,
// fastest first. It is what a caller sharing devices between tiers keys on (a
// server serialising sessions per card): two tiers' entries are equal exactly
// when they are one device.
func (g *GPU) Hardware() []Hardware {
	out := make([]Hardware, 0, len(g.devs))
	for _, d := range g.devs {
		h := Hardware{Ref: DeviceName(d.dev.API()) + ":" + d.dev.Name(), Name: d.name}
		if o := ordinalOf(d.dev); o >= 0 {
			h.Ref = DeviceName(d.dev.API() + ":" + strconv.Itoa(o))
		}
		h.Key = h.Ref
		if id := backend.IdentityOf(d.dev); id.Known() {
			h.Key = id.Key
		}
		out = append(out, h)
	}
	return out
}

// Budgets is every device's allowance, fastest first.
func (g *GPU) Budgets() []Budget {
	out := make([]Budget, 0, len(g.devs))
	for _, d := range g.devs {
		d.mu.Lock()
		b := Budget{Device: d.Stats.Device, Name: d.name, Limit: d.limit, Used: d.used, Why: d.why}
		d.mu.Unlock()
		if d.pool != nil {
			b.Pool, b.PoolLimit, b.Host = d.pool.Name(), d.pool.Limit(), d.pool.Host()
		}
		out = append(out, b)
	}
	return out
}

// HostReserved is how many bytes of the host's memory this tier holds now: what
// the devices on the host's memory have charged to their pool (blocks, KV
// pages, scratch, the driver's rounding), plus the packed arena's limit. The
// host takes it off its own budgets (model.Model.SetPageBudget,
// model.State.SetMemBudget), and it moves with placement: an integrated GPU
// holding nothing costs the host nothing, however large the pool it may grow
// into. Zero when no device is on the host's memory and the arena is off.
//
// The pool's limit caps the devices; it is not a charge on the host. Charging
// it was a quarter of the host budget taken from the pager for an iGPU that
// held no block (TestHostLosesOnlyWhatAUnifiedDeviceHolds).
func (g *GPU) HostReserved() uint64 {
	if g == nil {
		return 0
	}
	if g.hostPool == nil {
		// No device on the host's memory, but the arena is still host memory.
		return g.arenaReserved()
	}
	return g.hostPool.Used() + g.arenaReserved()
}

// arenaReserved is the host memory the packed arena may retain. It is host RAM
// the tier spends, so HostReserved includes it. It is the limit rather than the
// current usage: the arena fills to its limit on exactly the models where it is
// on. A unified device that wraps arena bytes charges nothing to its own budget
// or pool (resident.bytes) because those bytes are counted here, once.
func (g *GPU) arenaReserved() uint64 {
	if g == nil || g.Config == nil {
		return 0
	}
	return g.Config.arena.Limit()
}

// enumerating is do marked as part of a device search.
func enumerating(do backend.Opts) backend.Opts {
	do.Enumerating = true
	return do
}
