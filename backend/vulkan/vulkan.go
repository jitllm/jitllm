// Package vulkan runs SPIR-V compute kernels through the Vulkan loader.
//
// No cgo: backend/ffi (goffi) dlopens libvulkan.so.1 and the loader exports every core
// entry point, so vkGetInstanceProcAddr trampolining is unnecessary.
//
// Buffers are device-local, with staging when they have to be: host-visible
// memory on a discrete card means every kernel re-reads its weights across
// PCIe. Alloc prefers DEVICE_LOCAL and stages through a host-visible scratch
// buffer with vkCmdCopyBuffer. Where device-local memory is also host-visible
// (integrated GPUs, llvmpipe, resizable BAR) the mapping is written directly.
package vulkan

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/jitllm/jitllm/backend/ffi"
)

type up = unsafe.Pointer

// Loader is one binding of the Vulkan loader through backend/ffi: a Ctx keeps
// the Loader it was opened through, and every call on it, its buffers,
// kernels, queues and batches goes through that.
type Loader struct {
	// The Vulkan entry points, bound through backend/ffi (goffi). The loader
	// exports every core symbol, so no vkGetInstanceProcAddr trampolining.
	vkCreateInstance                         func(up, up, up) Result
	vkEnumeratePhysicalDevices               func(Instance, *uint32, up) Result
	vkGetPhysicalDeviceProperties            func(PhysDevice, up) ffi.None
	vkGetPhysicalDeviceQueueFamilyProperties func(PhysDevice, *uint32, up) ffi.None
	vkGetPhysicalDeviceMemoryProperties      func(PhysDevice, up) ffi.None
	vkGetPhysicalDeviceMemoryProperties2     func(PhysDevice, up) ffi.None
	vkGetPhysicalDeviceProperties2           func(PhysDevice, up) ffi.None
	vkGetPhysicalDeviceFeatures2             func(PhysDevice, up) ffi.None
	vkEnumerateDeviceExtensionProperties     func(PhysDevice, up, *uint32, up) Result
	vkCreateDevice                           func(PhysDevice, up, up, up) Result
	vkDestroyDevice                          func(Device, up) ffi.None
	vkGetDeviceQueue                         func(Device, uint32, uint32, up) ffi.None
	vkCreateBuffer                           func(Device, up, up, up) Result
	vkDestroyBuffer                          func(Device, Buffer, up) ffi.None
	vkGetBufferMemoryRequirements            func(Device, Buffer, up) ffi.None
	vkAllocateMemory                         func(Device, up, up, up) Result
	vkFreeMemory                             func(Device, Memory, up) ffi.None
	vkBindBufferMemory                       func(Device, Buffer, Memory, uint64) Result
	vkMapMemory                              func(Device, Memory, uint64, uint64, uint32, up) Result
	vkCreateShaderModule                     func(Device, up, up, up) Result
	vkDestroyShaderModule                    func(Device, ShaderMod, up) ffi.None
	vkCreateDescriptorSetLayout              func(Device, up, up, up) Result
	vkDestroyDescriptorSetLayout             func(Device, DescLayout, up) ffi.None
	vkCreatePipelineLayout                   func(Device, up, up, up) Result
	vkDestroyPipelineLayout                  func(Device, PipeLayout, up) ffi.None
	vkCreateComputePipelines                 func(Device, uint64, uint32, up, up, up) Result
	vkDestroyPipeline                        func(Device, Pipeline, up) ffi.None
	vkCreateDescriptorPool                   func(Device, up, up, up) Result
	vkDestroyDescriptorPool                  func(Device, DescPool, up) ffi.None
	vkAllocateDescriptorSets                 func(Device, up, up) Result
	vkUpdateDescriptorSets                   func(Device, uint32, up, uint32, up) ffi.None
	vkCreateCommandPool                      func(Device, up, up, up) Result
	vkDestroyCommandPool                     func(Device, CmdPool, up) ffi.None
	vkAllocateCommandBuffers                 func(Device, up, up) Result
	vkBeginCommandBuffer                     func(CmdBuffer, up) Result
	vkEndCommandBuffer                       func(CmdBuffer) Result
	vkResetCommandBuffer                     func(CmdBuffer, uint32) Result
	vkCmdBindPipeline                        func(CmdBuffer, uint32, Pipeline) ffi.None
	vkCmdBindDescriptorSets                  func(CmdBuffer, uint32, PipeLayout, uint32, uint32, up, uint32, up) ffi.None
	vkCmdDispatch                            func(CmdBuffer, uint32, uint32, uint32) ffi.None
	vkCmdPushConstants                       func(CmdBuffer, PipeLayout, uint32, uint32, uint32, up) ffi.None
	vkCmdPipelineBarrier                     func(CmdBuffer, uint32, uint32, uint32, uint32, up, uint32, up, uint32, up) ffi.None
	vkResetDescriptorPool                    func(Device, DescPool, uint32) Result
	vkQueueSubmit                            func(VkQueue, uint32, up, uint64) Result
	vkQueueWaitIdle                          func(VkQueue) Result
	vkCmdCopyBuffer                          func(CmdBuffer, Buffer, Buffer, uint32, up) ffi.None
	// vkGetDeviceProcAddr is how an EXTENSION's entry points are reached: the
	// loader exports the core symbols and this one, and nothing else. See
	// ffi.Fn4At.
	vkGetDeviceProcAddr func(Device, unsafe.Pointer) unsafe.Pointer
	// vkGetInstanceProcAddr is the same door on the INSTANCE side, and it is
	// needed for a physical-device query that the loader does not export --
	// vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR is the first. See
	// coopmat.go.
	vkGetInstanceProcAddr func(Instance, unsafe.Pointer) unsafe.Pointer

	// The fences a rec waits on (queue.go).
	vkCreateFence   func(Device, up, up, up) Result
	vkDestroyFence  func(Device, Fence, up) ffi.None
	vkWaitForFences func(Device, uint32, up, uint32, uint64) Result
	vkResetFences   func(Device, uint32, up) Result
}

// loaderNames is every name the Vulkan loader ships under, in the order they
// are tried. ffi.Open takes the first that loads.
//
// On macOS libvulkan.1.dylib is the loader; libMoltenVK.dylib is the ICD
// itself, which works but skips layer discovery and any second ICD, so it is
// only the fallback for a host without the loader.
func loaderNames() []string {
	return []string{
		"libvulkan.so.1", "libvulkan.so", // linux
		"vulkan-1.dll",                           // windows
		"libvulkan.1.dylib", "libMoltenVK.dylib", // macOS: loader first, ICD as fallback
	}
}

// Load opens the loader and binds every entry point.
func Load() (*Loader, error) {
	lib, err := ffi.Open(loaderNames()...)
	if err != nil {
		return nil, fmt.Errorf("vulkan: %w", err)
	}
	l := &Loader{}
	l.vkCreateInstance = ffi.Fn3[Result, up, up, up](lib, "vkCreateInstance")
	l.vkGetInstanceProcAddr = ffi.Fn2[unsafe.Pointer, Instance, unsafe.Pointer](lib, "vkGetInstanceProcAddr")
	l.vkEnumeratePhysicalDevices = ffi.Fn3[Result, Instance, *uint32, up](lib, "vkEnumeratePhysicalDevices")
	l.vkGetPhysicalDeviceProperties = ffi.Fn2[ffi.None, PhysDevice, up](lib, "vkGetPhysicalDeviceProperties")
	l.vkGetPhysicalDeviceQueueFamilyProperties = ffi.Fn3[ffi.None, PhysDevice, *uint32, up](lib, "vkGetPhysicalDeviceQueueFamilyProperties")
	l.vkGetPhysicalDeviceMemoryProperties = ffi.Fn2[ffi.None, PhysDevice, up](lib, "vkGetPhysicalDeviceMemoryProperties")
	// Core since Vulkan 1.1, so the loader exports it under the un-suffixed
	// name and no KHR fallback is needed at the 1.3 instance this asks for.
	l.vkGetPhysicalDeviceMemoryProperties2 = ffi.Fn2[ffi.None, PhysDevice, up](lib, "vkGetPhysicalDeviceMemoryProperties2")
	// Core since Vulkan 1.1 as well, and the DEVICE has to be 1.1 for the call
	// to be legal -- see readSubgroups, which checks before it asks.
	l.vkGetPhysicalDeviceProperties2 = ffi.Fn2[ffi.None, PhysDevice, up](lib, "vkGetPhysicalDeviceProperties2")
	l.vkGetPhysicalDeviceFeatures2 = ffi.Fn2[ffi.None, PhysDevice, up](lib, "vkGetPhysicalDeviceFeatures2")
	l.vkEnumerateDeviceExtensionProperties = ffi.Fn4[Result, PhysDevice, up, *uint32, up](lib, "vkEnumerateDeviceExtensionProperties")
	l.vkCreateDevice = ffi.Fn4[Result, PhysDevice, up, up, up](lib, "vkCreateDevice")
	l.vkDestroyDevice = ffi.Fn2[ffi.None, Device, up](lib, "vkDestroyDevice")
	l.vkGetDeviceQueue = ffi.Fn4[ffi.None, Device, uint32, uint32, up](lib, "vkGetDeviceQueue")
	l.vkCreateBuffer = ffi.Fn4[Result, Device, up, up, up](lib, "vkCreateBuffer")
	l.vkDestroyBuffer = ffi.Fn3[ffi.None, Device, Buffer, up](lib, "vkDestroyBuffer")
	l.vkGetBufferMemoryRequirements = ffi.Fn3[ffi.None, Device, Buffer, up](lib, "vkGetBufferMemoryRequirements")
	l.vkAllocateMemory = ffi.Fn4[Result, Device, up, up, up](lib, "vkAllocateMemory")
	l.vkFreeMemory = ffi.Fn3[ffi.None, Device, Memory, up](lib, "vkFreeMemory")
	l.vkBindBufferMemory = ffi.Fn4[Result, Device, Buffer, Memory, uint64](lib, "vkBindBufferMemory")
	l.vkMapMemory = ffi.Fn6[Result, Device, Memory, uint64, uint64, uint32, up](lib, "vkMapMemory")
	l.vkCreateShaderModule = ffi.Fn4[Result, Device, up, up, up](lib, "vkCreateShaderModule")
	l.vkDestroyShaderModule = ffi.Fn3[ffi.None, Device, ShaderMod, up](lib, "vkDestroyShaderModule")
	l.vkCreateDescriptorSetLayout = ffi.Fn4[Result, Device, up, up, up](lib, "vkCreateDescriptorSetLayout")
	l.vkDestroyDescriptorSetLayout = ffi.Fn3[ffi.None, Device, DescLayout, up](lib, "vkDestroyDescriptorSetLayout")
	l.vkCreatePipelineLayout = ffi.Fn4[Result, Device, up, up, up](lib, "vkCreatePipelineLayout")
	l.vkDestroyPipelineLayout = ffi.Fn3[ffi.None, Device, PipeLayout, up](lib, "vkDestroyPipelineLayout")
	l.vkCreateComputePipelines = ffi.Fn6[Result, Device, uint64, uint32, up, up, up](lib, "vkCreateComputePipelines")
	l.vkDestroyPipeline = ffi.Fn3[ffi.None, Device, Pipeline, up](lib, "vkDestroyPipeline")
	l.vkCreateDescriptorPool = ffi.Fn4[Result, Device, up, up, up](lib, "vkCreateDescriptorPool")
	l.vkDestroyDescriptorPool = ffi.Fn3[ffi.None, Device, DescPool, up](lib, "vkDestroyDescriptorPool")
	l.vkAllocateDescriptorSets = ffi.Fn3[Result, Device, up, up](lib, "vkAllocateDescriptorSets")
	l.vkUpdateDescriptorSets = ffi.Fn5[ffi.None, Device, uint32, up, uint32, up](lib, "vkUpdateDescriptorSets")
	l.vkCreateCommandPool = ffi.Fn4[Result, Device, up, up, up](lib, "vkCreateCommandPool")
	l.vkDestroyCommandPool = ffi.Fn3[ffi.None, Device, CmdPool, up](lib, "vkDestroyCommandPool")
	l.vkAllocateCommandBuffers = ffi.Fn3[Result, Device, up, up](lib, "vkAllocateCommandBuffers")
	l.vkBeginCommandBuffer = ffi.Fn2[Result, CmdBuffer, up](lib, "vkBeginCommandBuffer")
	l.vkEndCommandBuffer = ffi.Fn1[Result, CmdBuffer](lib, "vkEndCommandBuffer")
	l.vkResetCommandBuffer = ffi.Fn2[Result, CmdBuffer, uint32](lib, "vkResetCommandBuffer")
	l.vkCmdBindPipeline = ffi.Fn3[ffi.None, CmdBuffer, uint32, Pipeline](lib, "vkCmdBindPipeline")
	l.vkCmdBindDescriptorSets = ffi.Fn8[ffi.None, CmdBuffer, uint32, PipeLayout, uint32, uint32, up, uint32, up](lib, "vkCmdBindDescriptorSets")
	l.vkCmdDispatch = ffi.Fn4[ffi.None, CmdBuffer, uint32, uint32, uint32](lib, "vkCmdDispatch")
	l.vkCmdPushConstants = ffi.Fn6[ffi.None, CmdBuffer, PipeLayout, uint32, uint32, uint32, up](lib, "vkCmdPushConstants")
	l.vkCmdPipelineBarrier = ffi.Fn10[ffi.None, CmdBuffer, uint32, uint32, uint32, uint32, up, uint32, up, uint32, up](lib, "vkCmdPipelineBarrier")
	l.vkResetDescriptorPool = ffi.Fn3[Result, Device, DescPool, uint32](lib, "vkResetDescriptorPool")
	l.vkQueueSubmit = ffi.Fn4[Result, VkQueue, uint32, up, uint64](lib, "vkQueueSubmit")
	l.vkQueueWaitIdle = ffi.Fn1[Result, VkQueue](lib, "vkQueueWaitIdle")
	l.loadFences(lib)
	l.vkCmdCopyBuffer = ffi.Fn5[ffi.None, CmdBuffer, Buffer, Buffer, uint32, up](lib, "vkCmdCopyBuffer")
	l.vkGetDeviceProcAddr = ffi.Fn2[unsafe.Pointer, Device, unsafe.Pointer](lib, "vkGetDeviceProcAddr")
	return l, nil
}

func check(r Result, what string) error {
	if r != 0 {
		return fmt.Errorf("vulkan: %s failed: VkResult %d", what, r)
	}
	return nil
}

// Ctx is an open device: instance, logical device, queue and command pool.
type Ctx struct {
	// l is the binding every call on this context goes through.
	l *Loader
	// cores and coresFrom are the compute-unit count and the extension that
	// stated it, threadRegs a lane's registers at the most resident waves
	// (cores.go).
	cores      int
	coresFrom  string
	threadRegs int
	slots      int
	inst       Instance
	phys       PhysDevice
	dev        Device
	family     uint32
	// rec is the context's own line of submissions, for calls made on it
	// directly; queues (NewQueue) are others. hw is the compute family's
	// queues, queues is the count handed out and every queue not closed.
	rec
	hw     []*hwQueue
	queues struct {
		sync.Mutex
		n    int
		live map[*Queue]bool
	}
	mem  memProps
	name string
	// driverVer is VkPhysicalDeviceProperties.driverVersion, vendor-encoded:
	// compared for equality only (DriverVersion).
	driverVer uint32
	devType   uint32 // VkPhysicalDeviceType of the device that was picked
	idx       int    // enumeration index of the device that was picked
	// uuid is this physical device's identity and uuidOK says it could be
	// asked for one; see UUID and readDeviceUUID.
	uuid   [uuidSize]byte
	uuidOK bool
	// maxStorage is the picked device's maxStorageBufferRange; see MaxBuffer.
	maxStorage uint32
	// maxGroups is how many workgroups one dispatch may carry: the device's
	// maxComputeWorkGroupCount[0], or Config.MaxGroups where that is lower.
	// See dispatch.
	maxGroups uint32
	// storageAlign is the device's minStorageBufferOffsetAlignment (Buf.View).
	storageAlign uint64
	// dispatches counts vkCmdDispatch recorded; see Dispatches.
	dispatches atomic.Uint64
	budget     bool // VK_EXT_memory_budget is supported and enabled
	// memq is Mem's budget struct, which the driver fills through a pNext
	// uintptr: on the heap with the Ctx, one caller at a time.
	memq struct {
		sync.Mutex
		b memBudgetEXT
	}
	// tiles reports that VK_KHR_cooperative_matrix is present and every feature
	// the tile kernels' SPIR-V declares was enabled at vkCreateDevice. Both,
	// because a driver may run the kernel either way and only compute the right
	// answer with the features on.
	tiles bool
	// shapes is what the driver enumerated, read once at open. A device that
	// cannot do tiles leaves it nil, which refuses every shape.
	shapes []CoopMatShape
	// sg is what this device promises about subgroup width, read once at open.
	// Compile refuses a kernel whose width it cannot promise, which is what
	// makes the promise a selection rule rather than a hope; see subgroup.go.
	sg Subgroups
	// hostAlign is minImportedHostPointerAlignment when
	// VK_EXT_external_memory_host is supported and was enabled on this logical
	// device, and 0 otherwise; Import refuses on 0.
	hostAlign uint64
	// getHostPtrProps is vkGetMemoryHostPointerPropertiesEXT, reached through
	// the device's dispatch table rather than by name; see OpenDevice.
	getHostPtrProps func(Device, uint32, unsafe.Pointer, up) Result
	pins            []any // keeps Go-allocated C structs reachable for the device's life
	// kerns is every Kernel compiled on this context and not yet closed; see
	// Close.
	kerns struct {
		sync.Mutex
		live map[*Kernel]struct{}
	}
	// orphans is every kernel Close found still open, by entry point.
	orphans []string
}

// Orphans is the entry point of every kernel Close found its owner had not
// closed, in the order found.
func (c *Ctx) Orphans() []string { return append([]string(nil), c.orphans...) }

// onHeap is a new zero T on the heap. A struct the driver reaches through a
// uintptr field (a pNext chain, a create info's array) must not live on a
// goroutine stack: the stack is copied when it grows, at any call before the
// driver reads it, and a uintptr is not moved with it.
//
//go:noinline
func onHeap[T any]() *T { return new(T) }

func cstr(s string) (uintptr, []byte) {
	b := append([]byte(s), 0)
	return uintptr(up(&b[0])), b
}

// cstrPin is cstr for a string the driver reads during the call only, pinned to
// the context anyway because it costs one slice header.
func cstrPin(c *Ctx, s string) unsafe.Pointer {
	b := append([]byte(s), 0)
	c.pins = append(c.pins, b)
	return up(&b[0])
}

// Open picks a physical device and opens a compute queue on it. See device.go
// for the ordering, what happens to a software rasteriser, and Config.Device.
func Open() (*Ctx, error) { return OpenDevice("") }

// OpenDevice opens a NAMED physical device: an enumeration index, or a
// case-insensitive substring of its name. "" applies the default rule, which is
// Config.Device if a caller set one and the discrete-first ordering otherwise.
// It lets a caller placing blocks across several Vulkan devices name each one.
func OpenDevice(sel string) (*Ctx, error) { return OpenDeviceWith(sel, Config{}) }

// OpenDeviceWith is OpenDevice with this device's Config.
func OpenDeviceWith(sel string, cfg Config) (*Ctx, error) {
	l, err := Load()
	if err != nil {
		return nil, err
	}
	inst, err := l.instance()
	if err != nil {
		return nil, err
	}
	c := &Ctx{l: l, inst: inst}

	pick, err := l.pickDevice(c.inst, sel, cfg.Device)
	if err != nil {
		c.Close()
		return nil, err
	}
	c.phys, c.family, c.devType, c.idx = pick.phys, pick.family, pick.devType, pick.idx
	c.name = pick.name
	c.uuid, c.uuidOK = pick.uuid, pick.uuidOK
	c.driverVer = pick.driverVer
	c.maxStorage = pick.maxStorage
	c.maxGroups = pick.maxGroups
	c.storageAlign = pick.storageAlign
	if cfg.MaxGroups > 0 && cfg.MaxGroups < c.maxGroups {
		c.maxGroups = cfg.MaxGroups
	}
	if c.devType == devTypeCPU {
		// Only reachable by pinning it. Everything downstream reads
		// Name(), so the warning travels with the device rather than living at
		// the one call site that knew.
		c.name += softwareSuffix
	}
	l.vkGetPhysicalDeviceMemoryProperties(c.phys, up(&c.mem))

	// VK_EXT_memory_budget is what Mem reads. It is enabled on the logical
	// device as well as queried on the physical one, so heapUsage accounts for
	// this process's own allocations rather than only the driver's view.
	c.budget, err = l.hasExtension(c.phys, extMemoryBudget)
	if err != nil {
		c.Close()
		return nil, err
	}

	// VK_EXT_external_memory_host is read for its alignment here, which
	// decides whether an import is possible at all.
	hasHostMem, err := l.hasExtension(c.phys, extExternalMemoryHost)
	if err != nil {
		c.Close()
		return nil, err
	}
	hostAlign := uint64(0)
	if hasHostMem && pick.apiVer >= apiVersion11 {
		hp := onHeap[extMemHostProps]()
		hp.sType = stExtMemHostProps
		p2 := physProps2{sType: stPhysDevProps2, pNext: uintptr(up(hp))}
		l.vkGetPhysicalDeviceProperties2(c.phys, up(&p2))
		hostAlign = hp.minAlign
		runtime.KeepAlive(hp)
	}

	// The subgroup extension is queried on the physical device and enabled on
	// the logical one. Supported means the range can be read; enabled, with
	// subgroupSizeControl and computeFullSubgroups on, lets a pipeline pin its
	// width.
	hasSizeCtl, err := l.hasExtension(c.phys, extSubgroupSizeControl)
	if err != nil {
		c.Close()
		return nil, err
	}
	c.sg = l.readSubgroups(c.phys, pick.apiVer, hasSizeCtl)
	c.cores, c.coresFrom, c.threadRegs, c.slots = l.readCores(c.phys, pick.apiVer)
	c.sg.Off, c.sg.Forced = subgroupEnv(cfg)

	// The tile features are likewise read before the device is made and
	// enabled on it. See readTileFeats.
	tf := l.readTileFeats(c.phys, pick.apiVer)
	hasCoopMat, err := l.hasExtension(c.phys, extCooperativeMatrix)
	if err != nil {
		c.Close()
		return nil, err
	}
	c.tiles = tf.CoopOK && hasCoopMat
	if c.tiles {
		if sh, err := l.CooperativeMatrixShapes(c.inst, c.phys); err == nil {
			c.shapes = sh
		}
	}

	nq := min(max(pick.queues, 1), maxHWQueues)
	prios := make([]float32, nq)
	for i := range prios {
		prios[i] = 1
	}
	c.pins = append(c.pins, prios)
	qci, df, sc := onHeap[queueCI](), onHeap[dotFeat](), onHeap[sizeCtlFeat]()
	*qci = queueCI{sType: stDeviceQueueCI, family: c.family, count: uint32(nq), prio: uintptr(up(&prios[0]))}
	*df = dotFeat{sType: stDotProdFeat, on: 1}
	*sc = sizeCtlFeat{sType: stSizeCtlFeat, control: 1, full: 1}
	dci := deviceCI{sType: stDeviceCI, pNext: uintptr(up(df)),
		nQueues: 1, pQueues: uintptr(up(qci))}
	var exts []uintptr
	if c.tiles {
		extPtr, extBuf := cstr(extCooperativeMatrix)
		c.pins = append(c.pins, extBuf)
		exts = append(exts, extPtr)
	}
	if c.budget {
		extPtr, extBuf := cstr(extMemoryBudget)
		c.pins = append(c.pins, extBuf)
		exts = append(exts, extPtr)
	}
	if hostAlign > 0 {
		extPtr, extBuf := cstr(extExternalMemoryHost)
		c.pins = append(c.pins, extBuf)
		exts = append(exts, extPtr)
	}
	tail := &df.pNext
	if c.sg.Control && !c.sg.Off && !c.sg.Forced {
		extPtr, extBuf := cstr(extSubgroupSizeControl)
		c.pins = append(c.pins, extBuf)
		exts = append(exts, extPtr)
		df.pNext = uintptr(up(sc))
		tail = &sc.pNext
		c.sg.Enabled = true
	}
	if c.tiles {
		*tail = uintptr(up(&tf.coop)) // coop -> s16 -> s8 -> f16i8, linked in readTileFeats
	}
	if len(exts) > 0 {
		c.pins = append(c.pins, &exts)
		dci.nExts = uint32(len(exts))
		dci.pExts = uintptr(up(&exts[0]))
	}
	if err := check(l.vkCreateDevice(c.phys, up(&dci), nil, up(&c.dev)), "vkCreateDevice"); err != nil {
		c.Close()
		return nil, err
	}
	runtime.KeepAlive(qci)
	runtime.KeepAlive(df)
	runtime.KeepAlive(sc)
	runtime.KeepAlive(tf)
	for i := range nq {
		h := &hwQueue{}
		l.vkGetDeviceQueue(c.dev, c.family, uint32(i), up(&h.q))
		c.hw = append(c.hw, h)
	}
	// The extension's entry point is not exported by the loader, so it is
	// fetched from the device's dispatch table; binding it by name would panic
	// at startup. A null answer leaves hostAlign at 0 ("cannot import").
	if hostAlign > 0 {
		if fn := l.vkGetDeviceProcAddr(c.dev, cstrPin(c, "vkGetMemoryHostPointerPropertiesEXT")); fn != nil {
			c.hostAlign = hostAlign
			c.getHostPtrProps = ffi.Fn4At[Result, Device, uint32, unsafe.Pointer, up](fn,
				"vkGetMemoryHostPointerPropertiesEXT")
		}
	}

	if err := c.initRec(&c.rec, c.hw[0]); err != nil {
		c.Close()
		return nil, err
	}
	return c, nil
}

// proc is the process's one VkInstance, created by the first open or List and
// never destroyed (see instance), and what its enumerations have found: seen
// is the first enumeration's device count, and shrunk the count of the first
// one that came back with fewer (see Vanished). It is the package's one
// variable because the instance is the process's by the drivers' constraint,
// not by a choice here.
var proc struct {
	sync.Once
	inst Instance
	err  error
	seen struct {
		sync.Mutex
		first, shrunk int
	}
}

// instance is the process's VkInstance, which every Ctx and every List shares.
//
// One per process is not an economy, it is what keeps the drivers loadable.
// The loader dlopens each ICD when an instance is created and dlcloses it when
// the last instance is destroyed, and NVIDIA's ICD carries initial-exec TLS
// (libGLX_nvidia, libnvidia-glcore and libnvidia-tls are all DF_STATIC_TLS).
// glibc serves a dlopened module's static TLS from a fixed surplus and does
// not reliably get it back on dlclose, so every instance-and-destroy spent a
// few hundred bytes of it: with glibc 2.39 and a recent NVIDIA driver the tenth
// open/close cycle found the surplus gone,
// libnvidia-tls failed to load ("cannot allocate memory in static TLS block"),
// the loader dropped the ICD without an error, and the process enumerated
// llvmpipe alone from then on. With the instance held the ICD stays loaded, and
// a device open costs a logical device rather than an instance as well.
//
// It is also why the loader's environment -- VK_ICD_FILENAMES, VK_DRIVER_FILES,
// VK_INSTANCE_LAYERS -- is read once, at the first open: a caller that wants a
// different set of drivers wants a different process.
func (l *Loader) instance() (Instance, error) {
	proc.Do(func() {
		namePtr, nameBuf := cstr("jitllm")
		ai := onHeap[appInfo]()
		*ai = appInfo{sType: stAppInfo, pName: namePtr, pEngine: namePtr,
			apiVer: 1<<22 | 3<<12} // 1.3: shaderIntegerDotProduct is core
		ici := instanceCI{sType: stInstanceCI, pApp: uintptr(up(ai))}
		proc.err = check(l.vkCreateInstance(up(&ici), nil, up(&proc.inst)), "vkCreateInstance")
		runtime.KeepAlive(ai)
		runtime.KeepAlive(nameBuf)
	})
	return proc.inst, proc.err
}

// Info is one enumerated physical device, without opening it.
type Info struct {
	Index int    // the enumeration index, which is what OpenDevice takes
	Name  string // as the driver reports it, with no software suffix
	Type  string // "discrete GPU", "integrated GPU", "CPU (software rasteriser)", ...
	// Compute is false for a display-only device, which cannot run a kernel and
	// which OpenDevice refuses by name.
	Compute bool
	// Software is a CPU rasteriser: correct, and not a GPU. See pickDevice.
	Software bool
	// Unified says this device's memory IS the host's, which is what makes it a
	// member of the host's memory pool rather than a budget of its own.
	Unified bool
	// Subgroups is what the device promises about subgroup width, WITHOUT
	// opening it: every field of it is a physical-device query. Enabled is
	// necessarily false here -- there is no logical device yet -- so use
	// CouldPin rather than CanPin when reading this one.
	Subgroups Subgroups
	// UUID is VkPhysicalDeviceIDProperties.deviceUUID -- the sixteen bytes that
	// say whether this is the same physical device another backend also
	// enumerated. UUIDOK is false when the device is below Vulkan 1.1 and
	// cannot be asked, and an unknown identity must be read as DISTINCT: see
	// readDeviceUUID.
	UUID   [uuidSize]byte
	UUIDOK bool
	// MaxGroups is maxComputeWorkGroupCount[0], the device's own limit on the
	// workgroups of one dispatch, before any Config.MaxGroups.
	MaxGroups uint32
}

// CouldPin reports what Subgroups.CanPin will say once the device is OPENED: the
// extension is advertised, the compute stage accepts a required size, and
// nothing turned it off. It exists because Info is the enumeration, where no
// logical device has been created and Enabled cannot be true yet.
func (i Info) CouldPin() bool {
	s := i.Subgroups
	return s.Control && !s.Off && !s.Forced && s.ReqStages&stageCompute != 0
}

// Promises reports the width this device will be able to guarantee once opened,
// which is what a report printed BEFORE a device is chosen can honestly say.
func (i Info) Promises(w int) bool {
	s := i.Subgroups
	s.Enabled = i.CouldPin()
	ok, _ := s.Guarantees(w)
	return ok
}

// List enumerates every Vulkan physical device without opening one, so choosing
// devices does not cost a logical device per candidate (as backend.CUDACount).
// It also shows the devices Open's discrete-first rule passes over.
func List() ([]Info, error) { return ListWith(Config{}) }

// ListWith is List as a device opened with cfg would report its subgroups.
func ListWith(cfg Config) ([]Info, error) {
	l, err := Load()
	if err != nil {
		return nil, err
	}
	inst, err := l.instance()
	if err != nil {
		return nil, err
	}
	cands, err := l.enumerate(inst)
	if err != nil {
		return nil, err
	}
	out := make([]Info, 0, len(cands))
	for _, d := range cands {
		in := Info{
			Index:     d.idx,
			Name:      d.name,
			Type:      devTypeName(d.devType),
			Compute:   d.family != noFamily,
			Software:  d.devType == devTypeCPU,
			Unified:   d.devType == devTypeIntegrated || d.devType == devTypeCPU,
			UUID:      d.uuid,
			UUIDOK:    d.uuidOK,
			MaxGroups: d.maxGroups,
		}
		if in.Compute {
			ctl, err := l.hasExtension(d.phys, extSubgroupSizeControl)
			if err == nil {
				in.Subgroups = l.readSubgroups(d.phys, d.apiVer, ctl)
				in.Subgroups.Off, in.Subgroups.Forced = subgroupEnv(cfg)
			}
		}
		out = append(out, in)
	}
	return out, nil
}

func gostr(b []byte) string {
	for i, x := range b {
		if x == 0 {
			return string(b[:i])
		}
	}
	return string(b)
}

func (c *Ctx) Name() string { return c.name }

// Index is the enumeration index this context opened, which is what OpenDevice
// takes to open the same device again.
func (c *Ctx) Index() int { return c.idx }

// UUID is VkPhysicalDeviceIDProperties.deviceUUID for the device this context
// opened, and false when the device could not be asked.
//
// It is the evidence that a Vulkan device and a CUDA device are one card; both
// specifications promise the same sixteen bytes. The name is not: two identical
// cards report the same string.
func (c *Ctx) UUID() ([uuidSize]byte, bool) { return c.uuid, c.uuidOK }

// DriverVersion is the device's VkPhysicalDeviceProperties.driverVersion, in
// its vendor's encoding: fit to tell one driver from another, not to order them.
func (c *Ctx) DriverVersion() uint32 { return c.driverVer }

// MaxBuffer is the most bytes one storage buffer may be bound with
// (maxStorageBufferRange): a larger buffer binds short.
func (c *Ctx) MaxBuffer() uint64 { return uint64(c.maxStorage) }

// MaxGroups is the most workgroups one vkCmdDispatch carries on this context:
// the device's maxComputeWorkGroupCount[0] (65535 on an Intel integrated GPU,
// 2^31-1 on an NVIDIA card), or Config.MaxGroups where that is lower. A launch of more is
// not refused; dispatch splits it.
func (c *Ctx) MaxGroups() uint32 { return c.maxGroups }

func (c *Ctx) Close() {
	if c.dev != 0 {
		// Every queue idle before anything it may still read goes.
		c.waitAll()
		c.queues.Lock()
		for l := range c.queues.live {
			l.r.freeRec()
		}
		c.queues.live = nil
		c.queues.Unlock()
		// A kernel its owner never closed is a shader module, a pipeline, two
		// layouts and a descriptor pool the device still owns, and destroying
		// it then is invalid usage (VUID-vkDestroyDevice-device-05137). They go
		// here, and their names go to Orphans so a gate can see the owner that
		// forgot them.
		c.kerns.Lock()
		var left []*Kernel
		for k := range c.kerns.live {
			left = append(left, k)
		}
		c.kerns.Unlock()
		for _, k := range left {
			c.orphans = append(c.orphans, kernName(k))
			k.Close()
		}
		// The batch pool and fence are children of the device like the command
		// pool, and destroying a device that still owns one is invalid usage
		// (VUID-vkDestroyDevice-device-05137, which the validation layer
		// reported on every Close).
		c.rec.freeRec()
		c.l.vkDestroyDevice(c.dev, nil)
		c.dev = 0
	}
	// The instance is the process's (see instance) and outlives every Ctx.
	c.inst = 0
}
