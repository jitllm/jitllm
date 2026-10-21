package backend

import (
	"unsafe"

	"fmt"
	"strconv"
	"sync"

	"github.com/jitllm/jitllm/jit/ir"
	"github.com/jitllm/jitllm/jit/spirv"
	"github.com/jitllm/jitllm/backend/vulkan"
)

type vkDev struct {
	allocCount
	once
	c *vulkan.Ctx
	// free is Session's free list: a session handed to the caller's f
	// escapes, so a fresh one per call was a heap object per token. Nested
	// sessions take one each.
	mu   sync.Mutex
	free []*vkSession
}

var _ BufferLimited = (*vkDev)(nil)

// MaxBuffer is the device's maxStorageBufferRange (BufferLimited).
func (d *vkDev) MaxBuffer() uint64 { return d.c.MaxBuffer() }

// openVulkan opens the Vulkan backend, and drops a software rasteriser.
//
// vulkan.Open will take a CPU device (lavapipe, swiftshader) when the host has
// nothing else, which the package's correctness gates rely on. But tier places
// blocks against what it measures on every Device here, and a software
// rasteriser is strictly worse than the CPU tier, so an unasked-for one is no
// device. Naming it (Opts.Vulkan.Device, or OpenVulkan's argument) is an
// explicit request and is honoured.
func openVulkan(o Opts) (Device, error) { return OpenVulkanWith("", o) }

// OpenVulkan opens one Vulkan device by selector -- an enumeration index or a
// case-insensitive substring of its name -- and "" applies the default rule.
// It is the Vulkan half of OpenCUDA: backend.Open returns only the
// discrete-first pick. See VulkanDevices for the enumeration half.
//
// A selector that names a software rasteriser is honoured: the refusal below is
// for a device nobody asked for.
func OpenVulkan(sel string) (Device, error) { return OpenVulkanWith(sel, Opts{}) }

// OpenVulkanWith is OpenVulkan with this device's open choices.
func OpenVulkanWith(sel string, o Opts) (Device, error) {
	ask := sel
	if ask == "" {
		ask = o.Vulkan.Device
	}
	ord := -1 // the default pick
	if ask != "" {
		ord = -2 // a name, until it parses as an index
		if n, err := strconv.Atoi(ask); err == nil && n >= 0 {
			ord = n
		}
	}
	got, err := admit("spirv", ord, o.Enumerating)
	if err != nil {
		return nil, err
	}
	if ord == -1 && got >= 0 {
		// A restriction resolved the default pick to the device it allows.
		sel = strconv.Itoa(got)
	}
	c, err := vulkan.OpenDeviceWith(sel, o.Vulkan)
	if err != nil {
		return nil, err
	}
	if ord == -2 {
		// A name: the index it matched is known only now, so a refusal here
		// has opened the device for the length of the check.
		if _, err := admit("spirv", c.Index(), o.Enumerating); err != nil {
			c.Close()
			return nil, err
		}
	}
	if c.Software() && ask == "" {
		name := c.Name()
		c.Close()
		// The suggestion is the index, not the name: Name() carries the
		// software suffix by then and would not match itself as a substring.
		return nil, fmt.Errorf("backend: the only Vulkan device is %s; "+
			"declining it so the seam tuner cannot place blocks on the host's own cores "+
			"(a Vulkan device pin of \"0\", or JITLLM_VK_DEVICE=0, takes it anyway)", name)
	}
	ownedDevices.Add(1)
	return &vkDev{c: c}, nil
}

// VulkanDevices enumerates the Vulkan devices on this host without opening one,
// which is what a caller choosing where to put blocks needs before it pays for
// a logical device per candidate. It is the counterpart of CUDACount.
//
// Under a Restriction that holds no Vulkan device it enumerates nothing: the
// NVIDIA driver attaches a graphics context to card 0 for the length of an
// enumeration, which a process confined to another card must not make.
func VulkanDevices() ([]vulkan.Info, error) {
	if _, err := admit("spirv", -1, true); err != nil {
		return nil, err
	}
	return vulkan.List()
}

func (v *vkDev) Name() string { return v.c.Name() }

// Ordinal is the enumeration index this device was opened at, satisfying the
// same optional interface cudaDev does -- it is what a caller passes back to
// OpenVulkan to get this device again, and the one thing that tells two
// identical cards apart.
func (v *vkDev) Ordinal() int { return v.c.Index() }
func (v *vkDev) API() string  { return "spirv" }

// GuaranteedLanes answers from the device's own properties, and where the answer
// is yes because the width will be pinned, Compile pins it. See
// vulkan.Subgroups: the promise is min == max, or
// VK_EXT_subgroup_size_control's required size at pipeline creation, and
// subgroupSize is neither.
func (v *vkDev) GuaranteedLanes(w int) (bool, string) { return v.c.Subgroups().Guarantees(w) }

// Slots is unknown on Vulkan: VkPhysicalDeviceLimits bounds a workgroup but says
// nothing about how many the device holds resident. 0 means "use the caller's
// default".
func (v *vkDev) Slots() int { return 0 }
func (v *vkDev) Close() {
	v.c.Close()
	v.release(&ownedDevices)
}

// Mem satisfies the same optional interface cudaDev does: free and total bytes
// of the device-local heap, from VK_EXT_memory_budget. Vulkan needs none of
// CUDA's owner-goroutine dance -- vkGetPhysicalDeviceMemoryProperties2 is a
// physical-device query with no context to be current on.
func (v *vkDev) Mem() (free, total uint64, err error) { return v.c.Mem() }

// UnifiedMemory says the bytes Mem reports are the host's bytes -- true for an
// integrated GPU and for a CPU device. A caller that budgets host and device
// memory separately must subtract rather than add when this is set; see
// vulkan.Ctx.UnifiedMemory.
func (v *vkDev) UnifiedMemory() bool { return v.c.UnifiedMemory() }

func (v *vkDev) Alloc(n int) (Buf, error) {
	b, err := v.c.Alloc(n)
	if err != nil {
		return nil, err
	}
	return &vkBuf{b: b, acc: v.take(n, vulkanFootprint(n))}, nil
}

func (v *vkDev) Compile(k *ir.Kernel) (Kernel, error) {
	compiles.Add(1)
	// A tile kernel is declined by name on a device that did not enable the
	// features: a driver may run the module anyway and silently compute a
	// wrong answer, so the refusal is the whole safety property.
	if ir.UsesTiles(k) {
		if !v.c.Tiles() {
			return nil, fmt.Errorf("spirv: %s uses cooperative-matrix tiles and %s did not enable them "+
				"(VK_KHR_cooperative_matrix, storageBuffer16BitAccess, storageBuffer8BitAccess, "+
				"shaderFloat16, shaderInt8)", k.Name, v.c.Name())
		}
		if err := v.tileShapes(k); err != nil {
			return nil, err
		}
	}
	mod, err := spirv.Emit(k)
	if err != nil {
		return nil, err
	}
	kern, err := v.c.Compile(mod, k.Name, len(k.Params), k.Lanes)
	if err != nil {
		return nil, err
	}
	vk := &vkKern{k: kern, group: k.Group, name: k.Name, src: k, grid: k.GridSync()}
	if k.GridSync() {
		vk.resident, vk.residentErr = v.GridLimit(vk)
	}
	return vk, nil
}

// ImportAlign and Import are VK_EXT_external_memory_host, which the Vulkan
// context enables when the physical device offers it. The tier uses it only on
// unified devices: an imported weight on a discrete card is read over PCIe by
// every matvec.
func (v *vkDev) ImportAlign() int { return v.c.ImportAlign() }

func (v *vkDev) Import(p unsafe.Pointer, n int) (Buf, error) {
	b, err := v.c.Import(p, n)
	if err != nil {
		return nil, err
	}
	return &vkBuf{b: b}, nil
}

var _ HostImport = (*vkDev)(nil)

// Copy is vkCmdCopyBuffer on the context's one-shot command buffer, never the
// one a Session records into (see vulkan.Ctx.one), submitted and waited for.
func (v *vkDev) Copy(dst Buf, dstOff int, src Buf, srcOff, n int) error {
	db, ok := dst.(*vkBuf)
	sb, ok2 := src.(*vkBuf)
	if !ok || !ok2 || db == nil || sb == nil {
		return fmt.Errorf("backend: spirv: a copy between buffers this backend did not allocate")
	}
	if err := copyRange(db.b.Len(), dstOff, sb.b.Len(), srcOff, n, db.b == sb.b); err != nil || n == 0 {
		return err
	}
	return v.c.Copy(db.b, dstOff, sb.b, srcOff, n)
}

type vkBuf struct {
	b   *vulkan.Buf
	acc *counted
}

func (b *vkBuf) Write(p []byte) error { return b.b.Write(p) }

func (b *vkBuf) WriteAt(off int, p []byte) error { return b.b.WriteAt(off, p) }

func (b *vkBuf) Read(p []byte) error {
	countRead(len(p))
	return b.b.Read(p)
}

func (b *vkBuf) Free() {
	b.b.Free()
	b.acc.give()
}

type vkKern struct {
	k     *vulkan.Kernel
	bound []Buf
	// group and name are the declared workgroup and the kernel's name: Vulkan
	// runs the LocalSize the module declares whatever width the launch says,
	// so a launch at any other width is refused (checkWidth) rather than run
	// on part of its rows.
	group [3]int
	name  string
	// src is the kernel's IR (IRSource); resident and residentErr its
	// co-residency limit when it has a grid barrier.
	src         *ir.Kernel
	resident    int
	grid        bool // src has a grid barrier: decided once at compile, read by every launch
	residentErr error
}

func (k *vkKern) IR() *ir.Kernel { return k.src }

// GridLimit is one workgroup per compute unit, as a vendor extension states
// the count (vulkan/cores.go); a device none states cannot run a grid barrier.
func (v *vkDev) GridLimit(k Kernel) (int, error) {
	vk := k.(*vkKern)
	if !vk.src.GridSync() {
		return 0, fmt.Errorf("backend: spirv: %s has no grid barrier", vk.name)
	}
	n, _ := v.c.Cores()
	if n == 0 {
		return 0, fmt.Errorf("backend: spirv: %s states no compute-unit count (VK_NV_shader_sm_builtins, "+
			"VK_AMD_shader_core_properties), and core Vulkan no co-residency", v.c.Name())
	}
	return n, nil
}

// Launch re-binds only when the buffer set changes. Descriptor updates are
// cheap but they are not free, and a benchmark that re-binds every iteration is
// measuring the binding.
func (k *vkKern) Launch(groups, width int, bufs ...Buf) error {
	if err := checkWidth("spirv", k.name, k.group, width); err != nil {
		return err
	}
	if err := checkGrid("spirv", k.name, k.grid, groups, k.resident, k.residentErr); err != nil {
		return err
	}
	if !same(k.bound, bufs) {
		vb := make([]*vulkan.Buf, len(bufs))
		for i, b := range bufs {
			vb[i] = b.(*vkBuf).b
		}
		if err := k.k.Bind(vb...); err != nil {
			return err
		}
		k.bound = append(k.bound[:0], bufs...)
	}
	return k.k.Launch(groups)
}

func (k *vkKern) Close() { k.k.Close() }

func same(a, b []Buf) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Session records the whole batch into one command buffer, rather than one
// queue drain per dispatch as Kernel.Launch does. See vulkan.Batch for the two
// things Vulkan needs that Metal does not: a barrier between dispatches, and a
// descriptor set per dispatch.
func (d *vkDev) Session(f func(Session)) {
	d.mu.Lock()
	var s *vkSession
	if n := len(d.free); n > 0 {
		s, d.free = d.free[n-1], d.free[:n-1]
	}
	d.mu.Unlock()
	if s == nil {
		s = &vkSession{c: d.c}
	}
	f(s)
	s.Sync()
	d.mu.Lock()
	d.free = append(d.free, s)
	d.mu.Unlock()
}

type vkSession struct {
	c *vulkan.Ctx
	// l is the queue a queued session records on, nil for the context's own.
	l     *vulkan.Queue
	batch *vulkan.Batch
	// vb is Launch's buffer list, reused.
	vb []*vulkan.Buf
}

func (s *vkSession) Write(b Buf, p []byte) error { return s.WriteAt(b, 0, p) }

func (s *vkSession) WriteAt(b Buf, off int, p []byte) error {
	if s.l != nil {
		return s.l.WriteAt(b.(*vkBuf).b, off, p)
	}
	return b.WriteAt(off, p)
}

func (s *vkSession) Read(b Buf, p []byte) error {
	// A read needs everything recorded so far to have run.
	if err := s.Sync(); err != nil {
		return err
	}
	if s.l != nil {
		return s.l.Read(b.(*vkBuf).b, p)
	}
	return b.Read(p)
}

func (s *vkSession) Launch(k Kernel, groups, width int, bufs ...Buf) error {
	vk := k.(*vkKern)
	if err := checkWidth("spirv", vk.name, vk.group, width); err != nil {
		return err
	}
	if err := checkGrid("spirv", vk.name, vk.grid, groups, vk.resident, vk.residentErr); err != nil {
		return err
	}
	if s.batch == nil {
		if s.l != nil {
			s.batch = s.l.NewBatch()
		} else {
			s.batch = s.c.NewBatch()
		}
	}
	s.vb = s.vb[:0]
	for _, b := range bufs {
		s.vb = append(s.vb, b.(*vkBuf).b)
	}
	err := s.batch.Encode(vk.k, groups, s.vb...)
	clear(s.vb)
	return err
}

func (s *vkSession) Sync() error {
	if s.batch == nil {
		return nil
	}
	err := s.batch.Commit()
	s.batch = nil
	return err
}

// tileShapes refuses a kernel whose matrix shape this device did not enumerate.
//
// The driver is not a reliable check: asked for a shape it does not offer, it
// may compute a wrong product or crash (see vulkan.Ctx.SupportsTile). So the
// enumerated list is consulted before the module is built.
func (v *vkDev) tileShapes(k *ir.Kernel) error {
	for _, o := range k.Ops {
		if o.Kind != ir.OpTileMMA {
			continue
		}
		a, b := ir.TileTypeOf(k, o.Args[0]), ir.TileTypeOf(k, o.Args[1])
		if a == nil || b == nil || o.Tile == nil {
			return fmt.Errorf("spirv: %s: tile.mma operands are not tiles", k.Name)
		}
		ca, cb := coopComp(a.Elem), coopComp(b.Elem)
		cc := coopComp(o.Tile.Elem)
		if !v.c.SupportsTile(a.Rows, b.Cols, a.Cols, ca, cb, cc, cc) {
			return fmt.Errorf("spirv: %s wants %dx%dx%d %s*%s+%s->%s at subgroup scope and %s "+
				"enumerates %d combination(s), none of them that one",
				k.Name, a.Rows, b.Cols, a.Cols, ca, cb, cc, cc, v.c.Name(), len(v.c.TileShapes()))
		}
	}
	return nil
}

// coopComp maps an IR tile element onto VkComponentTypeKHR. The integer cases
// are signed because ir.TileElem has no unsigned member: an unsigned matrix is
// a different combination the driver enumerates separately, and silently
// reporting one as the other is the same mistake as omitting the MulAdd's
// signedness mask.
func coopComp(e ir.TileElem) vulkan.ComponentType {
	switch e {
	case ir.TileF16:
		return vulkan.CompF16
	case ir.TileS8:
		return vulkan.CompS8
	case ir.TileI32:
		return vulkan.CompS32
	}
	return vulkan.CompF32
}

// vkQueue is a queue of the context, with the session SessionOn hands out on it.
type vkQueue struct {
	once
	mu sync.Mutex
	l  *vulkan.Queue
	s  vkSession
}

func (q *vkQueue) Close() {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.l.Close()
	q.release(&ownedQueues)
}

// NewQueue is a queue of its own: a command pool, buffers, descriptor pool,
// staging and fence, on one of the device's queues (backend.Queued).
func (d *vkDev) NewQueue() (Queue, error) {
	l, err := d.c.NewQueue()
	if err != nil {
		return nil, err
	}
	ownedQueues.Add(1)
	return &vkQueue{l: l}, nil
}

// SessionOn records f's work on q and submits it there, waiting for
// that queue's fence alone.
func (d *vkDev) SessionOn(q Queue, f func(Session)) {
	vq := q.(*vkQueue)
	vq.mu.Lock()
	defer vq.mu.Unlock()
	vq.s.c, vq.s.l = d.c, vq.l
	f(&vq.s)
	vq.s.Sync()
}
