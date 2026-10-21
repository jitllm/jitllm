package hip

import (
	"bytes"
	"fmt"
	"runtime"
	"strings"
	"unsafe"
)

// HIP keeps a current device per OS thread (hipSetDevice), as CUDA keeps a
// current context, and goroutines migrate between threads. Every call that
// depends on it runs through Device.On, which pins the goroutine to its thread
// and selects the device for the length of the call.

// Runtime is a loaded HIP runtime and comgr, ready to enumerate devices.
type Runtime struct {
	h *hipAPI
	c *Comgr
}

// Open loads ROCm under a config and initialises HIP. It answers (nil, nil)
// when ROCm is absent, unloadable, or present with no AMD device: that is the
// normal state of most hosts, and AMD hardware then runs through Vulkan. An
// error is something worse, a ROCm that loaded and then failed.
func Open(c Config) (*Runtime, error) {
	l := load(c)
	if l.hipErr != nil || l.comgrErr != nil {
		return nil, nil
	}
	if e := l.hip.init(0); e != 0 {
		// hipInit fails with hipErrorNoDevice, or with an initialisation error
		// when the kernel driver (amdgpu/KFD) is not there: both are "no HIP
		// device here", not a fault in jitllm.
		return nil, nil
	}
	return &Runtime{h: l.hip, c: &Comgr{a: l.comgr}}, nil
}

// Why is the reason a config yields no runtime, or "" when it yields one. It
// is for a report; Open's (nil, nil) is the answer a caller acts on.
func Why(c Config) string {
	l := load(c)
	if l.hipErr != nil {
		return l.hipErr.Error()
	}
	if l.comgrErr != nil {
		return l.comgrErr.Error()
	}
	if e := l.hip.init(0); e != 0 {
		return l.hip.check(e, "hipInit").Error()
	}
	return ""
}

// Count is how many HIP devices the runtime sees.
func (r *Runtime) Count() int {
	var n int32
	if r.h.deviceCount(&n) != 0 {
		return 0
	}
	return int(n)
}

// Comgr is the runtime's code object manager.
func (r *Runtime) Comgr() *Comgr { return r.c }

// Device is one HIP device.
type Device struct {
	r    *Runtime
	ord  int32
	name string
	// Arch is gcnArchName with its target features ("gfx90a:sramecc+:xnack-"),
	// which is exactly what comgr is told to generate for.
	Arch string
	// Wave is the wavefront width the device runs kernels at by default.
	Wave       int
	slots      int
	maxGrid    int
	integrated bool
	// pci is (domain, bus, device), the location identity is derived from.
	pci   [3]int32
	pciOK bool
	uuid  [16]byte
	uuOK  bool
}

// OpenDevice describes HIP device ord. It allocates nothing on the device.
func (r *Runtime) OpenDevice(ord int) (*Device, error) {
	n := r.Count()
	if ord < 0 || ord >= n {
		return nil, fmt.Errorf("hip: device %d was asked for, and the runtime reports %d", ord, n)
	}
	d := &Device{r: r, ord: int32(ord)}
	var err error
	d.On(func() {
		err = d.describe()
	})
	if err != nil {
		return nil, err
	}
	return d, nil
}

func (d *Device) describe() error {
	h := d.r.h
	buf := make([]byte, 256)
	if h.deviceName(&buf[0], int32(len(buf)), d.ord) == 0 {
		if i := bytes.IndexByte(buf, 0); i > 0 {
			d.name = string(buf[:i])
		}
	}
	if d.name == "" {
		d.name = fmt.Sprintf("HIP device %d", d.ord)
	}
	props := make([]byte, propsBuf)
	if err := h.check(h.properties(&props[0], d.ord), "hipGetDevicePropertiesR0600"); err != nil {
		return err
	}
	arch, err := archFromProps(props)
	if err != nil {
		return err
	}
	d.Arch = arch
	attr := func(a int32) (int, bool) {
		var v int32
		if h.attribute(&v, a, d.ord) != 0 {
			return 0, false
		}
		return int(v), true
	}
	d.Wave, _ = attr(attrWarpSize)
	if cu, ok := attr(attrMultiprocessors); ok {
		if tpc, ok := attr(attrMaxThreadsPerCU); ok {
			d.slots = cu * tpc
		}
	}
	d.maxGrid, _ = attr(attrMaxGridDimX)
	in, _ := attr(attrIntegrated)
	d.integrated = in != 0
	dom, ok1 := attr(attrPCIDomainID)
	bus, ok2 := attr(attrPCIBusID)
	dev, ok3 := attr(attrPCIDeviceID)
	d.pci, d.pciOK = [3]int32{int32(dom), int32(bus), int32(dev)}, ok1 && ok2 && ok3
	d.uuOK = h.uuid(&d.uuid[0], d.ord) == 0
	return nil
}

// archFromProps reads gcnArchName out of a hipDeviceProp_tR0600 and refuses
// anything that is not a gfx target, which is what a layout this binding does
// not know would produce.
func archFromProps(props []byte) (string, error) {
	f := props[propsArchOffset : propsArchOffset+archLen]
	i := bytes.IndexByte(f, 0)
	if i < 0 {
		i = len(f)
	}
	arch := string(f[:i])
	if !strings.HasPrefix(arch, "gfx") {
		return "", fmt.Errorf("hip: gcnArchName read as %q, not a gfx target: the runtime's "+
			"hipDeviceProp_t is not the R0600 layout this binding reads", arch)
	}
	return arch, nil
}

// On runs f with this device current on a pinned OS thread.
func (d *Device) On(f func()) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	d.r.h.setDevice(d.ord)
	f()
}

// Name is the marketing name hipDeviceGetName gives.
func (d *Device) Name() string { return d.name }

// Ordinal is the HIP device index.
func (d *Device) Ordinal() int { return int(d.ord) }

// Slots is compute units times resident threads per unit, or 0 when the
// runtime would not say.
func (d *Device) Slots() int { return d.slots }

// MaxGrid is the most workgroups one launch may carry along x, or 0.
func (d *Device) MaxGrid() int { return d.maxGrid }

// Integrated reports an APU: its memory is the host's.
func (d *Device) Integrated() bool { return d.integrated }

// PCI is the device's (domain, bus, device) location, and false when the
// runtime would not give all three.
func (d *Device) PCI() ([3]int32, bool) { return d.pci, d.pciOK }

// UUID is hipDeviceGetUuid's answer. On AMD it is the KFD's unique id
// ("GPU-" and hex), not the Vulkan deviceUUID, so it identifies the card among
// HIP devices but cannot match it across APIs; see VulkanUUID.
func (d *Device) UUID() ([16]byte, bool) { return d.uuid, d.uuOK }

// VulkanUUID is the deviceUUID AMD's Vulkan drivers report for this card: four
// little-endian u32 of PCI domain, bus, device and function (mesa's
// ac_compute_device_uuid, which RADV uses). HIP exposes no PCI function; a GPU
// is function 0. False when the PCI location is unknown, which leaves the two
// views distinct rather than merging on a guess.
func (d *Device) VulkanUUID() ([16]byte, bool) {
	var u [16]byte
	if !d.pciOK {
		return u, false
	}
	for i, v := range [4]uint32{uint32(d.pci[0]), uint32(d.pci[1]), uint32(d.pci[2]), 0} {
		u[4*i], u[4*i+1], u[4*i+2], u[4*i+3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
	}
	return u, true
}

// Mem is the device's free and total memory.
func (d *Device) Mem() (free, total uint64, err error) {
	d.On(func() { err = d.r.h.check(d.r.h.memInfo(&free, &total), "hipMemGetInfo") })
	return free, total, err
}

// Ptr is a device allocation.
type Ptr struct {
	p uintptr
	n int
}

// Size is the allocation's length in bytes.
func (p *Ptr) Size() int { return p.n }

// Arg is the kernel argument for this buffer: its device address.
func (p *Ptr) Arg() uintptr { return p.p }

// The calls below assume the device is current: they run inside On.

// Alloc allocates n bytes of device memory.
func (d *Device) Alloc(n int) (*Ptr, error) {
	if n <= 0 {
		n = 4
	}
	var p uintptr
	if err := d.r.h.check(d.r.h.malloc(&p, uint64(n)), "hipMalloc"); err != nil {
		return nil, err
	}
	return &Ptr{p: p, n: n}, nil
}

// Free releases an allocation.
func (d *Device) Free(p *Ptr) {
	if p.p != 0 {
		d.r.h.free(p.p)
		p.p = 0
	}
}

// Write copies host bytes in at an offset, synchronously.
func (d *Device) Write(p *Ptr, off int, src []byte) error {
	if len(src) == 0 {
		return nil
	}
	return d.r.h.check(d.r.h.htod(p.p+uintptr(off), unsafe.Pointer(&src[0]), uint64(len(src))), "hipMemcpyHtoD")
}

// Read copies the start of an allocation out, synchronously.
func (d *Device) Read(p *Ptr, dst []byte) error {
	if len(dst) == 0 {
		return nil
	}
	return d.r.h.check(d.r.h.dtoh(unsafe.Pointer(&dst[0]), p.p, uint64(len(dst))), "hipMemcpyDtoH")
}

// Copy moves n bytes between allocations on the device.
func (d *Device) Copy(dst *Ptr, dstOff int, src *Ptr, srcOff, n int) error {
	if n == 0 {
		return nil
	}
	return d.r.h.check(d.r.h.dtod(dst.p+uintptr(dstOff), src.p+uintptr(srcOff), uint64(n)), "hipMemcpyDtoD")
}

// Sync waits for every stream on the device.
func (d *Device) Sync() error { return d.r.h.check(d.r.h.sync(), "hipDeviceSynchronize") }

// Stream is a non-blocking HIP stream: a backend.Queue.
type Stream struct{ s uintptr }

// NewStream creates a stream that does not synchronise with the null stream.
func (d *Device) NewStream() (*Stream, error) {
	var s uintptr
	if err := d.r.h.check(d.r.h.streamCreate(&s, streamNonBlock), "hipStreamCreateWithFlags"); err != nil {
		return nil, err
	}
	return &Stream{s: s}, nil
}

// DestroyStream releases a stream.
func (d *Device) DestroyStream(s *Stream) { d.r.h.streamDestroy(s.s) }

// SyncStream waits for one stream.
func (d *Device) SyncStream(s *Stream) error {
	return d.r.h.check(d.r.h.streamSync(s.s), "hipStreamSynchronize")
}

// WriteOn and ReadOn are Write and Read ordered on a stream. hipMemcpyWithStream
// returns once the copy is done, so the host bytes are not held past the call.
func (d *Device) WriteOn(s *Stream, p *Ptr, off int, src []byte) error {
	if len(src) == 0 {
		return nil
	}
	return d.r.h.check(d.r.h.htodStream(p.p+uintptr(off), unsafe.Pointer(&src[0]),
		uint64(len(src)), hipMemcpyDefault, s.s), "hipMemcpyWithStream")
}

func (d *Device) ReadOn(s *Stream, p *Ptr, dst []byte) error {
	if len(dst) == 0 {
		return nil
	}
	return d.r.h.check(d.r.h.dtohStream(unsafe.Pointer(&dst[0]), p.p,
		uint64(len(dst)), hipMemcpyDefault, s.s), "hipMemcpyWithStream")
}

// Module is a loaded code object and its one kernel.
type Module struct {
	m, f uintptr
	// args is the launch's argument storage: the device addresses, and the
	// array of pointers to them hipModuleLaunchKernel reads.
	vals []uintptr
	ptrs []unsafe.Pointer
}

// Load loads a code object and finds its kernel by name.
func (d *Device) Load(co []byte, name string) (*Module, error) {
	if len(co) == 0 {
		return nil, fmt.Errorf("hip: empty code object")
	}
	m := &Module{}
	if err := d.r.h.check(d.r.h.moduleLoad(&m.m, unsafe.Pointer(&co[0])), "hipModuleLoadData"); err != nil {
		return nil, err
	}
	nm := cstr(name)
	if err := d.r.h.check(d.r.h.moduleFunction(&m.f, m.m, &nm[0]), "hipModuleGetFunction "+name); err != nil {
		d.r.h.moduleUnload(m.m)
		return nil, err
	}
	return m, nil
}

// Unload releases a module.
func (d *Device) Unload(m *Module) {
	if m.m != 0 {
		d.r.h.moduleUnload(m.m)
		m.m = 0
	}
}

// Launch runs groups workgroups of width threads on a stream (nil for the null
// stream), with one device address per kernel parameter. It does not wait.
//
// The argument storage is the module's and is not safe for two launches of
// one module at once; a caller on several streams passes its own through
// LaunchArgs.
func (d *Device) Launch(s *Stream, m *Module, groups, width int, args []*Ptr) error {
	if cap(m.vals) < len(args) {
		m.vals = make([]uintptr, len(args))
		m.ptrs = make([]unsafe.Pointer, len(args))
	}
	return d.LaunchArgs(s, m, groups, width, args, m.vals[:len(args)], m.ptrs[:len(args)])
}

// LaunchArgs is Launch with caller-owned argument storage: vals and ptrs are
// each at least len(args) long.
func (d *Device) LaunchArgs(s *Stream, m *Module, groups, width int, args []*Ptr, vals []uintptr, ptrs []unsafe.Pointer) error {
	if len(args) == 0 {
		return fmt.Errorf("hip: a kernel with no arguments")
	}
	for i, a := range args {
		vals[i] = a.p
		ptrs[i] = unsafe.Pointer(&vals[i])
	}
	var st uintptr
	if s != nil {
		st = s.s
	}
	return d.r.h.check(d.r.h.launch(m.f, uint32(groups), 1, 1, uint32(width), 1, 1, 0, st,
		unsafe.Pointer(&ptrs[0]), nil), "hipModuleLaunchKernel")
}

// Loaded reports whether both ROCm libraries load under a config, whether or
// not any device is present.
func Loaded(c Config) bool {
	l := load(c)
	return l.hipErr == nil && l.comgrErr == nil
}
