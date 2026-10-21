package backend

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/gpu/amdgpu"
	"github.com/jitllm/jitllm/jit_old/gpu/hip"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// hipDev is one AMD GPU through ROCm. Its kernels are jitllm's IR lowered to
// LLVM IR (jit_old/gpu/amdgpu) and turned into a code object by comgr in process;
// see docs/design/amd-hip.md.
//
// HIP keeps a current device per OS thread, so every call runs inside
// hip.Device.On, which pins the goroutine and selects the device. A Session
// on the default queue is one such call around the caller's f; its launches go
// to the null stream, and its reads, which are synchronous copies on that
// stream, order after them.
type hipDev struct {
	allocCount
	d    *hip.Device
	tg   amdgpu.Target
	comg *hip.Comgr
	// life is read-held by every call on the device and write-held by Close.
	// run serialises Sessions on the null stream. mu guards closed and queues
	// for the length of a lookup only, so a Buf call made inside a Session
	// does not wait on the Session holding it.
	life   sync.RWMutex
	run    sync.Mutex
	mu     sync.Mutex
	closed bool
	queues map[*hipQueue]bool
}

var errHIPClosed = errors.New("backend: hip: the device is closed")

// HIPConfig is where ROCm is looked for; tier.WithROCm fills it.
type HIPConfig = hip.Config

// hipRuntime is the loaded runtime per config, so enumerating and opening do
// not initialise HIP twice.
func hipRuntime(c hip.Config) (*hip.Runtime, error) { return hip.Open(c) }

// HIPCount is how many HIP devices ROCm reports under a config: zero, with no
// error, when ROCm is absent.
func HIPCount(c hip.Config) int {
	rt, err := hipRuntime(c)
	if err != nil || rt == nil {
		return 0
	}
	return rt.Count()
}

// HIPMissing is why a config has no HIP devices, naming where ROCm was looked
// for, or "" when it has some.
func HIPMissing(c hip.Config) string {
	if HIPCount(c) > 0 {
		return ""
	}
	why := hip.Why(c)
	if hip.Loaded(c) {
		if why == "" {
			why = "no device"
		}
		return "ROCm found and reports no HIP device: " + why
	}
	return fmt.Sprintf("ROCm not found (searched %s)", hip.Searched(c))
}

// OpenHIPWith opens HIP device ord. An ordinal the runtime does not have, or
// no ROCm at all, is an error naming the cause: a caller who asked for a HIP
// device by name gets it or a reason, never another backend.
func OpenHIPWith(ord int, o Opts) (Device, error) {
	if ord >= 0 || o.Enumerating {
		var err error
		if ord, err = admit("amdgcn", ord, o.Enumerating); err != nil {
			return nil, err
		}
	}
	if ord < 0 && o.Enumerating {
		ord = 0 // the default pick
	}
	rt, err := hipRuntime(o.HIP)
	if err != nil {
		return nil, err
	}
	if rt == nil && hip.Loaded(o.HIP) {
		return nil, fmt.Errorf("hip:%d requested; 0 HIP devices (%s)", ord, hip.Why(o.HIP))
	}
	if rt == nil {
		return nil, fmt.Errorf("hip:%d requested but ROCm was not found (searched %s)",
			ord, hip.Searched(o.HIP))
	}
	if n := rt.Count(); ord < 0 || ord >= n {
		return nil, fmt.Errorf("hip:%d requested; %d HIP devices", ord, n)
	}
	d, err := rt.OpenDevice(ord)
	if err != nil {
		return nil, err
	}
	tg, err := amdgpu.TargetFor(d.Arch)
	if err != nil {
		return nil, err
	}
	if d.Wave != 0 && d.Wave != tg.Wave {
		return nil, fmt.Errorf("hip:%d: %s runs %d-lane waves and the lowering targets %d",
			ord, d.Arch, d.Wave, tg.Wave)
	}
	return &hipDev{d: d, tg: tg, comg: rt.Comgr()}, nil
}

// openHIP is what backend.Open uses: the first HIP device, if there is one.
func openHIP(o Opts) (Device, error) {
	if HIPCount(o.HIP) == 0 {
		return nil, fmt.Errorf("backend: hip: %s", HIPMissing(o.HIP))
	}
	return OpenHIPWith(-1, o)
}

// Name is ordinal, card and arch, as CUDA's is ordinal, card and target.
func (h *hipDev) Name() string {
	return fmt.Sprintf("#%d %s (%s)", h.d.Ordinal(), h.d.Name(), h.tg.Base())
}

// API is the lowering's name.
func (h *hipDev) API() string { return "amdgcn" }

// Ordinal is the HIP device index (OrdinalDevice).
func (h *hipDev) Ordinal() int { return h.d.Ordinal() }

// Slots is compute units times resident threads per unit.
func (h *hipDev) Slots() int { return h.d.Slots() }

// UnifiedMemory: an APU's memory is the host's. Subtract, do not add.
func (h *hipDev) UnifiedMemory() bool { return h.d.Integrated() }

// Identity is the deviceUUID AMD's Vulkan driver reports for the same card,
// built from the PCI location (hip.Device.VulkanUUID). hipDeviceGetUuid's own
// answer is the KFD id, which no Vulkan driver reports, so it would never
// match and one card would count twice.
func (h *hipDev) Identity() Identity {
	u, ok := h.d.VulkanUUID()
	return UUIDIdentity(u[:], ok, "PCI location as RADV's deviceUUID (hipDeviceGetAttribute)")
}

// GuaranteedLanes: the IR's shuffle is a 32-lane xor with a mask below 32,
// which ds_bpermute computes exactly on a 32-wide wave and within each
// aligned half of a 64-wide one. So 32 holds on every AMD target, by the
// lowering rather than by observation; nothing else is offered.
func (h *hipDev) GuaranteedLanes(w int) (bool, string) {
	if w == ir.SubgroupLanes {
		return true, ""
	}
	return false, fmt.Sprintf("amdgcn lowers the %d-lane xor shuffle and no other width; %d cannot be asked for",
		ir.SubgroupLanes, w)
}

func (h *hipDev) Mem() (free, total uint64, err error) {
	err = h.outsideNoSync(func() error {
		var e error
		free, total, e = h.d.Mem()
		return e
	})
	return free, total, err
}

// isClosed reads closed under mu.
func (h *hipDev) isClosed() bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.closed
}

// outsideNoSync runs f with the device current and alive, without waiting
// for earlier work.
func (h *hipDev) outsideNoSync(f func() error) error {
	h.life.RLock()
	defer h.life.RUnlock()
	if h.isClosed() {
		return errHIPClosed
	}
	var err error
	h.d.On(func() { err = f() })
	return err
}

func (h *hipDev) Close() {
	h.life.Lock()
	defer h.life.Unlock()
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.closed {
		return
	}
	h.d.On(func() {
		for q := range h.queues {
			h.d.SyncStream(q.st)
			h.d.DestroyStream(q.st)
		}
		h.d.Sync()
	})
	h.queues = nil
	h.closed = true
}

// hipFootprint is what an n-byte hipMalloc costs: the runtime hands out pages
// as CUDA's driver does.
func hipFootprint(n int) int { return cudaFootprint(n) }

func (h *hipDev) Alloc(n int) (Buf, error) {
	var p *hip.Ptr
	err := h.outsideNoSync(func() error {
		var e error
		p, e = h.d.Alloc(n)
		return e
	})
	if err != nil {
		allocRefused.Add(1)
		return nil, err
	}
	return &hipBuf{p: p, n: n, dev: h, acc: h.take(n, hipFootprint(n))}, nil
}

// Compile lowers the kernel for this device's arch, has comgr generate and
// link a code object, and refuses one whose metadata says it was generated at
// a wave size the device does not run or spills to scratch -- a fault the
// offline gate (lowertest.TestAMDGPULowersForEveryTarget) holds every shipped
// kernel to, checked here too because a shape no gate registered reaches it.
func (h *hipDev) Compile(k *ir.Kernel) (Kernel, error) {
	compiles.Add(1)
	src, err := amdgpu.Lower(k, h.tg)
	if err != nil {
		return nil, err
	}
	co, err := h.comg.Compile(src, h.d.Arch)
	if err != nil {
		return nil, fmt.Errorf("backend: hip: %s: %w", k.Name, err)
	}
	if len(co.Kernels) != 1 {
		return nil, fmt.Errorf("backend: hip: %s: the code object holds %d kernels", k.Name, len(co.Kernels))
	}
	if m := co.Kernels[0]; m.Wave != h.tg.Wave || m.Scratch != 0 {
		return nil, fmt.Errorf("backend: hip: %s: generated at wave %d with %d bytes of scratch; "+
			"the device runs wave %d and jitllm's kernels use no private memory", k.Name, m.Wave, m.Scratch, h.tg.Wave)
	}
	var m *hip.Module
	err = h.outsideNoSync(func() error {
		var e error
		m, e = h.d.Load(co.Bytes, k.Name)
		return e
	})
	if err != nil {
		return nil, err
	}
	return &hipKern{m: m, dev: h, name: k.Name, group: k.Group}, nil
}

func (h *hipDev) Copy(dst Buf, dstOff int, src Buf, srcOff, n int) error {
	db, ok := dst.(*hipBuf)
	sb, ok2 := src.(*hipBuf)
	if !ok || !ok2 || db == nil || sb == nil || db.dev != h || sb.dev != h {
		return fmt.Errorf("backend: hip: a copy between buffers this device did not allocate")
	}
	if err := copyRange(db.n, dstOff, sb.n, srcOff, n, db.p == sb.p); err != nil || n == 0 {
		return err
	}
	return h.outside(func() error {
		if err := h.d.Copy(db.p, dstOff, sb.p, srcOff, n); err != nil {
			return err
		}
		return h.d.Sync()
	})
}

// outside runs a call made outside any Session: after everything before it,
// every queue included, which a device-wide synchronise gives.
func (h *hipDev) outside(f func() error) error {
	return h.outsideNoSync(func() error {
		if err := h.d.Sync(); err != nil {
			return err
		}
		return f()
	})
}

// Session runs f on the null stream with the device current.
func (h *hipDev) Session(f func(s Session)) {
	h.run.Lock()
	defer h.run.Unlock()
	h.outsideNoSync(func() error {
		f(&hipSession{h: h})
		return nil
	})
}

type hipBuf struct {
	p   *hip.Ptr
	n   int
	dev *hipDev
	acc *counted
}

func (b *hipBuf) Write(p []byte) error { return b.WriteAt(0, p) }

func (b *hipBuf) WriteAt(off int, p []byte) error {
	if len(p) == 0 {
		return nil
	}
	if off < 0 || off+len(p) > b.n {
		return fmt.Errorf("backend: hip: write of %d bytes at offset %d of a %d-byte buffer", len(p), off, b.n)
	}
	return b.dev.outside(func() error { return b.dev.d.Write(b.p, off, p) })
}

func (b *hipBuf) Read(p []byte) error {
	if len(p) == 0 {
		return nil
	}
	if len(p) > b.n {
		return fmt.Errorf("backend: hip: read of %d bytes from a %d-byte buffer", len(p), b.n)
	}
	countRead(len(p))
	return b.dev.outside(func() error { return b.dev.d.Read(b.p, p) })
}

// Free on a closed device gives back the count only: the device is gone.
func (b *hipBuf) Free() {
	b.dev.outsideNoSync(func() error {
		b.dev.d.Free(b.p)
		return nil
	})
	b.acc.give()
}

type hipKern struct {
	m     *hip.Module
	dev   *hipDev
	name  string
	group [3]int
}

// Launch waits for the kernel, as every backend's Kernel.Launch does.
func (k *hipKern) Launch(groups, width int, bufs ...Buf) error {
	if err := checkWidth("amdgcn", k.name, k.group, width); err != nil {
		return err
	}
	args, err := hipArgs(bufs)
	if err != nil {
		return err
	}
	return k.dev.outside(func() error {
		if err := k.dev.d.Launch(nil, k.m, groups, width, args); err != nil {
			return err
		}
		return k.dev.d.Sync()
	})
}

func (k *hipKern) Close() {
	k.dev.outsideNoSync(func() error {
		k.dev.d.Unload(k.m)
		return nil
	})
}

func hipArgs(bufs []Buf) ([]*hip.Ptr, error) {
	out := make([]*hip.Ptr, len(bufs))
	for i, b := range bufs {
		hb, ok := b.(*hipBuf)
		if !ok || hb == nil {
			return nil, fmt.Errorf("backend: hip: kernel argument %d of %d is not a HIP buffer", i, len(bufs))
		}
		out[i] = hb.p
	}
	return out, nil
}

// hipSession is a Session already inside hip.Device.On. q is nil on the null
// stream.
type hipSession struct {
	h *hipDev
	q *hipQueue
}

func (s *hipSession) stream() *hip.Stream {
	if s.q == nil {
		return nil
	}
	return s.q.st
}

func (s *hipSession) Write(b Buf, p []byte) error { return s.WriteAt(b, 0, p) }

func (s *hipSession) WriteAt(b Buf, off int, p []byte) error {
	hb := b.(*hipBuf)
	if len(p) == 0 {
		return nil
	}
	if off < 0 || off+len(p) > hb.n {
		return fmt.Errorf("backend: hip: write of %d bytes at offset %d of a %d-byte buffer", len(p), off, hb.n)
	}
	if s.q != nil {
		return s.h.d.WriteOn(s.q.st, hb.p, off, p)
	}
	return s.h.d.Write(hb.p, off, p)
}

func (s *hipSession) Read(b Buf, p []byte) error {
	hb := b.(*hipBuf)
	if len(p) == 0 {
		return nil
	}
	countRead(len(p))
	if s.q != nil {
		return s.h.d.ReadOn(s.q.st, hb.p, p)
	}
	return s.h.d.Read(hb.p, p)
}

func (s *hipSession) Launch(k Kernel, groups, width int, bufs ...Buf) error {
	hk := k.(*hipKern)
	if err := checkWidth("amdgcn", hk.name, hk.group, width); err != nil {
		return err
	}
	args, err := hipArgs(bufs)
	if err != nil {
		return err
	}
	if s.q != nil {
		// The module's argument storage is shared by every caller of the
		// kernel and two queues launch it at once: a queue fills its own.
		if len(args) > len(s.q.vals) {
			return fmt.Errorf("backend: hip: %d kernel arguments is too many", len(args))
		}
		return s.h.d.LaunchArgs(s.q.st, hk.m, groups, width, args, s.q.vals[:], s.q.ptrs[:])
	}
	return s.h.d.Launch(nil, hk.m, groups, width, args)
}

func (s *hipSession) Sync() error {
	if s.q != nil {
		return s.h.d.SyncStream(s.q.st)
	}
	return s.h.d.Sync()
}

// hipQueue is a non-blocking stream (backend.Queued).
type hipQueue struct {
	h    *hipDev
	st   *hip.Stream
	mu   sync.Mutex
	vals [24]uintptr
	ptrs [24]unsafe.Pointer
}

func (h *hipDev) NewQueue() (Queue, error) {
	var st *hip.Stream
	err := h.outsideNoSync(func() error {
		var e error
		st, e = h.d.NewStream()
		return e
	})
	if err != nil {
		return nil, err
	}
	q := &hipQueue{h: h, st: st}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.queues == nil {
		h.queues = map[*hipQueue]bool{}
	}
	h.queues[q] = true
	return q, nil
}

func (q *hipQueue) Close() {
	q.mu.Lock()
	defer q.mu.Unlock()
	h := q.h
	h.mu.Lock()
	open := h.queues[q]
	delete(h.queues, q)
	h.mu.Unlock()
	if !open {
		return // the device closed it
	}
	h.outsideNoSync(func() error {
		h.d.SyncStream(q.st)
		h.d.DestroyStream(q.st)
		return nil
	})
}

// SessionOn runs f with q's stream; sessions on two queues run at once.
func (h *hipDev) SessionOn(q Queue, f func(Session)) {
	hq := q.(*hipQueue)
	hq.mu.Lock()
	defer hq.mu.Unlock()
	h.mu.Lock()
	open := !h.closed && h.queues[hq]
	h.mu.Unlock()
	if !open {
		return
	}
	h.outsideNoSync(func() error {
		f(&hipSession{h: h, q: hq})
		return nil
	})
}

var (
	_ Device        = (*hipDev)(nil)
	_ Queued        = (*hipDev)(nil)
	_ Identified    = (*hipDev)(nil)
	_ Unified       = (*hipDev)(nil)
	_ Lanes         = (*hipDev)(nil)
	_ OrdinalDevice = (*hipDev)(nil)
)
