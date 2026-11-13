package cuda

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"runtime"
	"unsafe"
)

// The exported runtime surface returns errors rather than panicking: a missing
// driver, a busy GPU or an out-of-memory is an error the caller can answer by
// staying on the CPU.

// Device is an initialised CUDA context on one GPU.
type Device struct {
	drv *Driver
	dev CUdevice
	ctx CUctx
	// ord is the CUDA ordinal this context was opened on, and name is what
	// cuDeviceGetName said about it. Target is not an identity (a generation
	// shares one sm_XX), and two identical cards share a name, so both are
	// kept.
	ord  int
	name string
	// Target is the PTX target string for this device's compute capability.
	// It is read, not assumed: ptxas silently accepts a lower target and
	// forgoes newer instructions.
	Target string
	slots  int
	// uuid is the driver's own identity for this device, read once at open.
	// uuidOK false is a refusal, never an all-zero UUID: see UUID.
	uuid   [UUIDSize]byte
	uuidOK bool
	// maxGrid is CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, the most blocks one
	// launch may carry along x (2^31-1 from compute capability 3.0), or 0 when
	// the driver would not say. See MaxGrid.
	maxGrid int
	// sms is CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT and coop whether the
	// device takes a cooperative launch (CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH);
	// see Module.Resident.
	sms int
	// sharedSM is CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR.
	sharedSM int
	coop     bool
	// threadRegs is CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR over
	// its most resident threads, 0 where the driver would not say. See
	// ThreadRegisters.
	threadRegs int
	// l2 is CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, bytes, 0 where the driver
	// would not say.
	l2 int
	// mapHost is whether the device can address registered host memory
	// (CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY and HOST_REGISTER_SUPPORTED),
	// and integrated whether its memory is the host's
	// (CU_DEVICE_ATTRIBUTE_INTEGRATED: a Jetson, not a discrete card).
	mapHost, integrated bool
}

// UUIDSize is the 16 bytes a CUuuid carries, the same 16 bytes
// VkPhysicalDeviceIDProperties.deviceUUID carries for the same physical
// device, which is what makes one card seen through two APIs recognisable.
const UUIDSize = 16

const (
	attrCCMajor = 75
	attrCCMinor = 76
	attrSMCount = 16 // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
	attrMaxTPM  = 39 // CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR
	attrMaxGrid = 5  // CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X
	attrL2      = 38 // CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE
	attrRegsSM  = 82 // CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR
	attrShmSM   = 81 // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR

	attrIntegrated   = 18 // CU_DEVICE_ATTRIBUTE_INTEGRATED
	attrCanMapHost   = 19 // CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY
	attrHostRegister = 99 // CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED
)

// Count is how many CUDA devices this driver can see. It needs cuInit only,
// creates no context and does not lock the caller's OS thread, so it is safe
// for enumeration from any goroutine.
func (drv *Driver) Count() (int, error) {
	if err := drv.call(drv.cuInit(0), "cuInit"); err != nil {
		return 0, err
	}
	return drv.count()
}

func (drv *Driver) count() (int, error) {
	var n int32
	if err := drv.call(drv.cuDeviceGetCount(&n), "cuDeviceGetCount"); err != nil {
		return 0, err
	}
	return int(n), nil
}

// Open initialises the driver and creates a context on device 0; it is
// OpenDevice(0).
func (drv *Driver) Open() (*Device, error) { return drv.OpenDevice(0) }

// OpenDevice initialises the driver and creates a context on CUDA device ord.
// An ordinal past cuDeviceGetCount is an error naming the count, never a
// silent fallback to device 0.
//
// It locks the calling goroutine to its OS thread, and every later call must
// come from that goroutine: a CUDA context is current per OS thread, and a
// migrated goroutine fails with CUDA_ERROR_INVALID_CONTEXT. One owner
// goroutine per device, each holding its own context current, so nothing here
// pushes and pops contexts per call.
func (drv *Driver) OpenDevice(ord int) (*Device, error) {
	runtime.LockOSThread()
	d, err := drv.openDevice(ord)
	if err != nil {
		// The error path must unlock: a goroutine that exits locked has its
		// OS thread destroyed with a raw exit that skips glibc teardown, which
		// later crashes in fakecgo's threadentry (see Close).
		runtime.UnlockOSThread()
		return nil, err
	}
	return d, nil
}

func (drv *Driver) openDevice(ord int) (*Device, error) {
	if err := drv.call(drv.cuInit(0), "cuInit"); err != nil {
		return nil, err
	}
	n, err := drv.count()
	if err != nil {
		return nil, err
	}
	if ord < 0 || ord >= n {
		if n == 0 {
			return nil, fmt.Errorf("cuda: device %d was asked for, and this driver reports no CUDA devices at all", ord)
		}
		return nil, fmt.Errorf("cuda: device %d was asked for, and this driver reports %d device(s), ordinals 0..%d",
			ord, n, n-1)
	}
	d := &Device{drv: drv, ord: ord}
	if err := drv.call(drv.cuDeviceGet(&d.dev, int32(ord)), "cuDeviceGet"); err != nil {
		return nil, err
	}
	if err := drv.call(drv.cuCtxCreate(&d.ctx, 0, d.dev), "cuCtxCreate"); err != nil {
		return nil, err
	}
	d.name = drv.deviceName(d.dev, ord)
	d.uuid, d.uuidOK = drv.deviceUUID(d.dev)
	d.Target = "sm_86"
	if drv.cuDeviceGetAttribute != nil {
		var maj, min int32
		if drv.cuDeviceGetAttribute(&maj, attrCCMajor, d.dev) == 0 &&
			drv.cuDeviceGetAttribute(&min, attrCCMinor, d.dev) == 0 && maj > 0 {
			d.Target = fmt.Sprintf("sm_%d%d", maj, min)
		}
	}
	// Threads the card can hold resident, asked rather than guessed: the
	// attention split (tier.accSplitFor) needs a target thread count, and it
	// varies by an order of magnitude across cards.
	if drv.cuDeviceGetAttribute != nil {
		var sm, tpm int32
		if drv.cuDeviceGetAttribute(&sm, attrSMCount, d.dev) == 0 &&
			drv.cuDeviceGetAttribute(&tpm, attrMaxTPM, d.dev) == 0 && sm > 0 && tpm > 0 {
			d.slots = int(sm) * int(tpm)
			var regs int32
			if drv.cuDeviceGetAttribute(&regs, attrRegsSM, d.dev) == 0 && regs > 0 {
				d.threadRegs = int(regs) / int(tpm)
			}
			var shm int32
			if drv.cuDeviceGetAttribute(&shm, attrShmSM, d.dev) == 0 && shm > 0 {
				d.sharedSM = int(shm)
			}
		}
		d.sms = int(sm)
		var coop int32
		d.coop = drv.cuLaunchCooperativeKernel != nil && drv.cuOccupancy != nil &&
			drv.cuDeviceGetAttribute(&coop, attrCoopLaunch, d.dev) == 0 && coop != 0
		var grid int32
		if drv.cuDeviceGetAttribute(&grid, attrMaxGrid, d.dev) == 0 && grid > 0 {
			d.maxGrid = int(grid)
		}
		var l2 int32
		if drv.cuDeviceGetAttribute(&l2, attrL2, d.dev) == 0 && l2 > 0 {
			d.l2 = int(l2)
		}
		var mapHost, reg, integ int32
		d.mapHost = drv.cuDeviceGetAttribute(&mapHost, attrCanMapHost, d.dev) == 0 && mapHost != 0 &&
			drv.cuDeviceGetAttribute(&reg, attrHostRegister, d.dev) == 0 && reg != 0
		d.integrated = drv.cuDeviceGetAttribute(&integ, attrIntegrated, d.dev) == 0 && integ != 0
	}
	return d, nil
}

// MaxGrid is the most blocks one launch may carry along x, as the driver
// reports it, or 0 when it would not say.
func (d *Device) MaxGrid() int { return d.maxGrid }

// Slots is the number of threads this device can hold resident, or 0 when the
// driver would not say.
func (d *Device) Slots() int { return d.slots }

// Multiprocessors is the device's multiprocessor count, 0 where the driver
// would not say.
func (d *Device) Multiprocessors() int { return d.sms }

// SharedPerMultiprocessor is the workgroup memory one multiprocessor holds,
// in bytes (CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR), 0 where
// the driver did not say.
func (d *Device) SharedPerMultiprocessor() int { return d.sharedSM }

// ThreadRegisters is the 32-bit registers a thread holds when its
// multiprocessor holds the most threads it can, 0 where the driver would not
// say.
func (d *Device) ThreadRegisters() int { return d.threadRegs }

// L2Bytes is the bytes its L2 cache holds, 0 where the driver would not say.
func (d *Device) L2Bytes() int { return d.l2 }

// Ordinal is the CUDA device this context was opened on.
func (d *Device) Ordinal() int { return d.ord }

// Driver is the binding the device was opened through.
func (d *Device) Driver() *Driver { return d.drv }

// Name is the hardware, from cuDeviceGetName (a marketing name such as
// "NVIDIA GeForce RTX ..."), not "sm_86".
func (d *Device) Name() string { return d.name }

// UUID is the driver's identity for this device, and false when the driver
// would not give one.
//
// It is what says two enumerated devices (say CUDA and Vulkan views of one
// card) are one piece of hardware; the name is not, since identical cards
// share it. False is not an all-zero UUID: two unknowns must never compare
// equal and merge.
func (d *Device) UUID() ([UUIDSize]byte, bool) { return d.uuid, d.uuidOK }

// DeviceUUID reads a device's identity without creating a context on it; it
// needs cuInit and nothing else.
func (drv *Driver) DeviceUUID(ord int) ([UUIDSize]byte, bool) {
	var zero [UUIDSize]byte
	if err := drv.call(drv.cuInit(0), "cuInit"); err != nil {
		return zero, false
	}
	var dev CUdevice
	if drv.call(drv.cuDeviceGet(&dev, int32(ord)), "cuDeviceGet") != nil {
		return zero, false
	}
	return drv.deviceUUID(dev)
}

func (drv *Driver) deviceUUID(dev CUdevice) ([UUIDSize]byte, bool) {
	var out [UUIDSize]byte
	for _, c := range drv.uuidCalls() {
		var buf [UUIDSize]byte
		if c.fn(&buf[0], dev) == 0 {
			return buf, true
		}
	}
	return out, false
}

type uuidCall struct {
	name string
	fn   func(*byte, CUdevice) CUresult
}

func (drv *Driver) uuidCalls() []uuidCall {
	var out []uuidCall
	if drv.cuDeviceGetUuidV2 != nil {
		out = append(out, uuidCall{"cuDeviceGetUuid_v2", drv.cuDeviceGetUuidV2})
	}
	if drv.cuDeviceGetUuid != nil {
		out = append(out, uuidCall{"cuDeviceGetUuid", drv.cuDeviceGetUuid})
	}
	return out
}

// UUIDSource names the driver call a UUID would come from on this host, or says
// there is none.
//
// cuDeviceGetUuid_v2 identifies a MIG instance and the v1 call the whole GPU,
// so the report names which one answered.
func (drv *Driver) UUIDSource() string {
	if c := drv.uuidCalls(); len(c) > 0 {
		return c[0].name
	}
	return "this CUDA driver exports neither cuDeviceGetUuid nor cuDeviceGetUuid_v2"
}

// Label is how this device appears in a report: ordinal, name and target,
// because no one of them identifies a card on its own.
func (d *Device) Label() string { return fmt.Sprintf("#%d %s (%s)", d.ord, d.name, d.Target) }

// nameLen is the buffer cuDeviceGetName writes into; the driver truncates to
// fit and NUL-terminates.
const nameLen = 256

// deviceName asks the driver what the card is called, and falls back to
// something distinguishable rather than the empty string.
func (drv *Driver) deviceName(dev CUdevice, ord int) string {
	if drv.cuDeviceGetName != nil {
		buf := make([]byte, nameLen)
		if drv.cuDeviceGetName(&buf[0], int32(len(buf)), dev) == 0 {
			if i := bytes.IndexByte(buf, 0); i > 0 {
				return string(buf[:i])
			}
		}
	}
	return fmt.Sprintf("CUDA device %d", ord)
}

// Close destroys the context. It must run on the goroutine that called Open,
// which still holds that OS thread. Releasing the owner without destroying the
// context lets Go tear the thread down under a live driver context, which
// crashed a later, unrelated thread creation.
func (d *Device) Close() error {
	if d.ctx == 0 {
		return nil
	}
	var err error
	if d.drv.cuCtxDestroy != nil {
		err = d.drv.call(d.drv.cuCtxDestroy(d.ctx), "cuCtxDestroy")
	}
	d.ctx = 0
	// Open called LockOSThread; this is its matching unlock. The lock nests
	// (backend.openCUDA locks too), and a goroutine that exits still locked
	// has its thread destroyed without glibc teardown, crashing the next
	// thread the runtime creates.
	runtime.UnlockOSThread()
	return err
}

// Mem is free and total memory, in bytes, on the device this was opened on.
// cuMemGetInfo takes no device argument and answers for whatever context is
// current, so this binds the device's context first; placement spends bytes
// on this number.
func (d *Device) Mem() (free, total uint64, err error) {
	if err = d.bind(); err != nil {
		return 0, 0, err
	}
	err = d.drv.call(d.drv.cuMemGetInfo(&free, &total), "cuMemGetInfo")
	return
}

// bind makes this device's context current on the calling thread.
//
// Every other call relies on cuCtxCreate having made the context current on
// the owner's locked thread, which stays true for that thread's life; binding
// per launch would add a driver call to the hot path for nothing. Mem is the
// exception because it is cheap and rare.
func (d *Device) bind() error { return d.Bind() }

// Bind makes this device's context current on the calling thread, for a caller
// that runs driver calls on its own thread rather than the owner's (see
// backend's inline sessions). The context may be current on several threads;
// the caller serialises the calls.
func (d *Device) Bind() error {
	if d.ctx == 0 {
		return fmt.Errorf("cuda: device #%d is closed", d.ord)
	}
	return d.drv.call(d.drv.cuCtxSetCurrent(d.ctx), "cuCtxSetCurrent")
}

// Swap makes this device's context current on the calling thread and writes
// the context that was current there (0 for none) to prev, for Restore: a
// thread running one device's work that calls into another hands the first
// its context back, since a context stays current on a thread until another
// is made current. prev is the caller's to hold where the driver's write
// costs nothing, a field of what the call already holds: a local whose
// address the driver takes is moved to the heap, once a call.
func (d *Device) Swap(prev *CUctx) error {
	if err := d.drv.call(d.drv.cuCtxGetCurrent(prev), "cuCtxGetCurrent"); err != nil {
		return err
	}
	if *prev == d.ctx {
		return nil
	}
	return d.Bind()
}

// Restore makes ctx, what Swap found current, current on the calling thread
// again.
func (d *Device) Restore(ctx CUctx) error {
	if ctx == d.ctx {
		return nil
	}
	return d.drv.call(d.drv.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
}

// Buffer is device memory, or host memory the device addresses (Import).
type Buffer struct {
	drv *Driver
	p   CUdevptr
	n   uint64
	// host is the registered host address of an imported buffer, nil for
	// one cuMemAlloc made. Free unregisters it and never frees the pages.
	host unsafe.Pointer
	// view says the memory is another buffer's (View): Free leaves it.
	view bool
}

// Integrated reports whether the device's memory is the host's.
func (d *Device) Integrated() bool { return d.integrated }

// ImportAlign is what Import needs of the address and the length: whole pages,
// because registration page-locks whole pages and a partial one would pin
// bytes the caller does not own. 0 when the device cannot map host memory.
func (d *Device) ImportAlign() int {
	if !d.mapHost || d.drv.cuMemHostRegister == nil {
		return 0
	}
	return os.Getpagesize()
}

// Import page-locks n bytes at p, maps them for every context
// (CU_MEMHOSTREGISTER_PORTABLE|CU_MEMHOSTREGISTER_DEVICEMAP) and returns a
// buffer at their device address. A kernel reading it reads the host's own
// bytes: across the link on a discrete card, in place on an integrated one.
// The caller owns the memory and must keep it alive past Free.
func (d *Device) Import(p unsafe.Pointer, n int) (*Buffer, error) {
	a := d.ImportAlign()
	if a == 0 {
		return nil, fmt.Errorf("cuda: device #%d cannot map host memory", d.ord)
	}
	if n <= 0 || uintptr(p)%uintptr(a) != 0 || n%a != 0 {
		return nil, fmt.Errorf("cuda: importing %d bytes at %p: both must be %d-aligned", n, p, a)
	}
	if err := d.drv.call(d.drv.cuMemHostRegister(p, uint64(n), 1|2), "cuMemHostRegister"); err != nil {
		return nil, err
	}
	b := &Buffer{drv: d.drv, n: uint64(n), host: p}
	if err := d.drv.call(d.drv.cuMemHostGetDevicePointer(&b.p, p, 0), "cuMemHostGetDevicePointer"); err != nil {
		d.drv.cuMemHostUnregister(p)
		return nil, err
	}
	return b, nil
}

func (d *Device) Alloc(n int) (*Buffer, error) {
	b := &Buffer{drv: d.drv, n: uint64(n)}
	if err := d.drv.call(d.drv.cuMemAlloc(&b.p, b.n), "cuMemAlloc"); err != nil {
		return nil, err
	}
	return b, nil
}

// View is n bytes of b from byte off: the same memory at its own device
// address, bound where a launch binds a buffer. b owns the memory; the
// view's Free leaves it.
func (b *Buffer) View(off, n int) (*Buffer, error) {
	if off < 0 || n <= 0 || uint64(off)+uint64(n) > b.n {
		return nil, fmt.Errorf("cuda: a view of %d bytes from %d of a buffer of %d", n, off, b.n)
	}
	return &Buffer{drv: b.drv, p: b.p + CUdevptr(off), n: uint64(n), view: true}, nil
}

func (b *Buffer) Free() {
	if b.view {
		return
	}
	if b.host != nil {
		b.drv.cuMemHostUnregister(b.host)
		return
	}
	b.drv.cuMemFree(b.p)
}

// Arg is the address of the device pointer. cuLaunchKernel takes an array of
// pointers to the arguments, so this is what goes in it for a buffer.
func (b *Buffer) Arg() unsafe.Pointer { return unsafe.Pointer(&b.p) }

// Write copies host memory into the buffer. The caller keeps ownership of p and
// must keep it alive across the call.
func (b *Buffer) Write(p unsafe.Pointer, n int) error { return b.WriteAt(0, p, n) }

// WriteAt copies host memory in at a byte offset, so one device buffer can be
// assembled from several host runs. The driver also refuses an overrun; the
// check here gives a better message and matches the mapped backends, where the
// same overrun is silent.
func (b *Buffer) WriteAt(off int, p unsafe.Pointer, n int) error {
	if off < 0 || uint64(off)+uint64(n) > b.n {
		return fmt.Errorf("cuda: writing %d bytes at offset %d of a %d-byte buffer", n, off, b.n)
	}
	return b.drv.call(b.drv.cuMemcpyHtoD(b.p+CUdevptr(off), p, uint64(n)), "cuMemcpyHtoD")
}

func (b *Buffer) Read(p unsafe.Pointer, n int) error {
	return b.drv.call(b.drv.cuMemcpyDtoH(p, b.p, uint64(n)), "cuMemcpyDtoH")
}

// HostAlloc is n bytes of page-locked host memory, portable to every context.
// A transfer from it runs at the link's rate and needs no staging inside the
// driver; a pageable one is copied through the driver's own pinned bounce
// buffer first. The memory is the driver's, outside the Go heap, and lives
// until FreeHost.
func (drv *Driver) HostAlloc(n int) ([]byte, error) {
	if n <= 0 {
		return nil, fmt.Errorf("cuda: page-locking %d bytes", n)
	}
	var p unsafe.Pointer
	// CU_MEMHOSTALLOC_PORTABLE: pinned for every context, not only the
	// current one, so any card's transfer reads it at full rate.
	if err := drv.call(drv.cuMemHostAlloc(&p, uint64(n), 1), "cuMemHostAlloc"); err != nil {
		return nil, err
	}
	return unsafe.Slice((*byte)(p), n), nil
}

// FreeHost gives back memory HostAlloc returned.
func (drv *Driver) FreeHost(b []byte) {
	if len(b) > 0 {
		drv.cuMemFreeHost(unsafe.Pointer(&b[0]))
	}
}

// Len is the buffer's size in bytes.
func (b *Buffer) Len() int { return int(b.n) }

// CopyDtoD copies n bytes from src at srcOff to dst at dstOff on the device,
// on the legacy stream. It does not wait: a device-to-device copy is
// asynchronous to the host, so a caller that needs the bytes landed follows it
// with Sync. Ranges are the caller's to have checked.
func (drv *Driver) CopyDtoD(dst *Buffer, dstOff int, src *Buffer, srcOff, n int) error {
	return drv.call(drv.cuMemcpyDtoD(dst.p+CUdevptr(dstOff), src.p+CUdevptr(srcOff), uint64(n)), "cuMemcpyDtoD")
}

// Module is loaded PTX.
type Module struct {
	drv *Driver
	m   CUmodule
	f   CUfunc
	// args is the argument array cuLaunchKernel is handed, as wide as the
	// widest launch and reused, so a launch allocates nothing. Per-module
	// mutable state is safe because every driver call runs on the one
	// goroutine backend.cudaDev owns.
	args []unsafe.Pointer
	// maxGrid is the loading device's MaxGrid; see LaunchArgs.
	maxGrid int
	// sms and devCoop are the loading device's; see Resident.
	sms     int
	devCoop bool
	// Coop launches the module with cuLaunchCooperativeKernel, for a kernel
	// with a grid barrier (ir.OpGridBarrier); see Resident.
	Coop bool
	// dynShared is the dynamic shared memory every launch asks for; see
	// SetDynamicShared.
	dynShared int
}

// Load compiles PTX and looks up the entry point.
//
// This is where ptxas runs, the expensive step. The driver caches on disk
// keyed on the PTX text (~/.nv/ComputeCache), so emitted text must be
// byte-identical across runs for the same shape: determinism here is a
// performance property.
func (d *Device) Load(ptx, entry string) (*Module, error) {
	src := append([]byte(ptx), 0)
	m := &Module{drv: d.drv, maxGrid: d.maxGrid, sms: d.sms, devCoop: d.coop}
	if err := d.drv.call(d.drv.cuModuleLoadData(&m.m, unsafe.Pointer(&src[0])), "cuModuleLoadData"); err != nil {
		return nil, err
	}
	// goffi does not marshal Go strings, so the entry name is NUL-terminated
	// here.
	name := append([]byte(entry), 0)
	if err := d.drv.call(d.drv.cuModuleGetFunction(&m.f, m.m, &name[0]), "cuModuleGetFunction"); err != nil {
		d.drv.cuModuleUnload(m.m)
		return nil, err
	}
	return m, nil
}

func (m *Module) Unload() { m.drv.cuModuleUnload(m.m) }

// Args hands back the module's own argument scratch, n entries wide, for a
// caller that fills it with Buffer.Arg and passes it to LaunchArgs; a width
// past the widest so far grows it once. The kernel's parameter list is bound
// by the toolchain's own limit, which Load meets: ptxas refuses an entry
// whose parameters pass CUDA's 4 KB (CUDA C++ Programming Guide, "Function
// Parameters").
func (m *Module) Args(n int) []unsafe.Pointer {
	if n > len(m.args) {
		m.args = make([]unsafe.Pointer, n)
	}
	return m.args[:n]
}

// Launch runs the kernel on the legacy stream.
func (m *Module) Launch(grid, block int, bufs ...*Buffer) error {
	args := m.Args(len(bufs))
	for i := range bufs {
		args[i] = bufs[i].Arg()
	}
	return m.LaunchArgs(0, grid, block, args)
}

// LaunchArgs runs the kernel on st -- the legacy stream when st is zero -- with
// an argument array the caller already built.
//
// The stream is a parameter only for capture: a graph cannot be recorded from
// the legacy stream. Everything else, including every memcpy, stays on the
// legacy stream.
func (m *Module) LaunchArgs(st CUstream, grid, block int, args []unsafe.Pointer) error {
	if len(args) == 0 {
		return fmt.Errorf("cuda: launch with no arguments")
	}
	// cuLaunchKernel takes the grid as an unsigned int, so a grid past 2^32
	// would be truncated into a smaller launch that the driver accepts; and
	// one past the device's limit is refused, but by a bare
	// CUDA_ERROR_INVALID_VALUE. Both are refused here with the numbers rather
	// than split as Vulkan splits a launch past its 65535: a split would put
	// a base parameter on every PTX kernel for a grid of 2^31 blocks, which is
	// 2^31 work items at a one-thread group and, at any wider one, the whole
	// 2^32 reach of the IR's 32-bit flat index or more.
	limit := uint64(math.MaxUint32)
	if m.maxGrid > 0 {
		limit = min(limit, uint64(m.maxGrid))
	}
	if grid < 0 || uint64(grid) > limit {
		return fmt.Errorf("cuda: launch of %d blocks; this device takes at most %d along x", grid, limit)
	}
	if m.Coop {
		// The driver refuses a cooperative grid it cannot hold resident
		// (CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE) rather than launching it.
		return m.drv.call(m.drv.cuLaunchCooperativeKernel(m.f, uint32(grid), 1, 1, uint32(block), 1, 1,
			uint32(m.dynShared), st, unsafe.Pointer(&args[0])), "cuLaunchCooperativeKernel")
	}
	return m.drv.call(m.drv.cuLaunchKernel(m.f, uint32(grid), 1, 1, uint32(block), 1, 1,
		uint32(m.dynShared), st, unsafe.Pointer(&args[0]), nil), "cuLaunchKernel")
}

// Sync blocks until the device is idle.
func (drv *Driver) Sync() error { return drv.call(drv.cuCtxSynchronize(), "cuCtxSynchronize") }

func (drv *Driver) call(r CUresult, what string) error {
	if r == 0 {
		return nil
	}
	if r == cudaErrorOutOfMemory {
		return fmt.Errorf("cuda: %s: %s: %w", what, drv.errStr(r), ErrOutOfMemory)
	}
	return fmt.Errorf("cuda: %s: %s", what, drv.errStr(r))
}

// cudaErrorOutOfMemory is CUDA_ERROR_OUT_OF_MEMORY.
const cudaErrorOutOfMemory CUresult = 2

// ErrOutOfMemory is what a driver call wraps when the device refused memory:
// the card is full, which a caller treats differently from a failure.
const ErrOutOfMemory cudaError = "device out of memory"

type cudaError string

func (e cudaError) Error() string { return string(e) }

// DriverVersion is cuDriverGetVersion's answer (1000*major + 10*minor), false
// when there is no driver to ask.
func (drv *Driver) DriverVersion() (int, bool) {
	var v int32
	if drv.call(drv.cuDriverGetVersion(&v), "cuDriverGetVersion") != nil {
		return 0, false
	}
	return int(v), true
}
