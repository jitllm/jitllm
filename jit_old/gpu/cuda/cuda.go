package cuda

// A minimal CUDA Driver API binding: no cgo, the driver is dlopened at runtime
// through jit_old/gpu/ffi.

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/gpu/ffi"
)

type (
	CUresult int32
	CUdevice int32
	CUdevptr uint64
	CUctx    uintptr
	CUmodule uintptr
	CUfunc   uintptr
	CUstream uintptr
	CUevent  uintptr

	// A captured launch sequence and its executable form. Both are opaque
	// driver handles; jitllm never inspects a node.
	CUgraph     uintptr
	CUgraphExec uintptr
)

// The driver entry points, bound through jit_old/gpu/ffi.
var (
	cuInit             func(uint32) CUresult
	cuDriverGetVersion func(*int32) CUresult
	cuDeviceGet        func(*CUdevice, int32) CUresult
	cuDeviceGetCount   func(*int32) CUresult
	cuDeviceGetName    func(*byte, int32, CUdevice) CUresult
	cuCtxCreate        func(*CUctx, uint32, CUdevice) CUresult
	cuCtxSetCurrent    func(CUctx) CUresult
	// cuCtxGetDevice answers which card is current on this thread. Only the
	// ordinal gate uses it, to check that Mem left its own context current.
	cuCtxGetDevice       func(*CUdevice) CUresult
	cuCtxSynchronize     func() CUresult
	cuCtxDestroy         func(CUctx) CUresult
	cuMemAlloc           func(*CUdevptr, uint64) CUresult
	cuMemFree            func(CUdevptr) CUresult
	cuMemcpyHtoD         func(CUdevptr, unsafe.Pointer, uint64) CUresult
	cuMemcpyDtoH         func(unsafe.Pointer, CUdevptr, uint64) CUresult
	cuMemcpyDtoD         func(CUdevptr, CUdevptr, uint64) CUresult
	cuModuleLoadData     func(*CUmodule, unsafe.Pointer) CUresult
	cuModuleUnload       func(CUmodule) CUresult
	cuModuleGetFunction  func(*CUfunc, CUmodule, *byte) CUresult
	cuLaunchKernel       func(CUfunc, uint32, uint32, uint32, uint32, uint32, uint32, uint32, CUstream, unsafe.Pointer, unsafe.Pointer) CUresult
	cuGetErrorString     func(CUresult, *unsafe.Pointer) CUresult
	cuMemGetInfo         func(*uint64, *uint64) CUresult
	cuDeviceGetAttribute func(*int32, int32, CUdevice) CUresult
	cuMemHostAlloc       func(*unsafe.Pointer, uint64, uint32) CUresult
	cuMemFreeHost        func(unsafe.Pointer) CUresult
	// cuMemHostRegister page-locks memory the caller already owns and maps
	// it into the device's address space; cuMemHostGetDevicePointer is the
	// address a kernel reads it at. Together they are the host-pointer
	// import (backend.HostImport).
	cuMemHostRegister         func(unsafe.Pointer, uint64, uint32) CUresult
	cuMemHostUnregister       func(unsafe.Pointer) CUresult
	cuMemHostGetDevicePointer func(*CUdevptr, unsafe.Pointer, uint32) CUresult
)

// The device identity calls, bound only if the driver has them (binding panics
// on a missing symbol). Both nil means the driver will not say which card this
// is, and a caller must treat that as distinct, never a match: see
// Device.UUID. cuDeviceGetUuid_v2 (CUDA 11.4+) is MIG-aware: it identifies the
// instance a context is created on, where v1 identifies the physical GPU.
var (
	cuDeviceGetUuid   func(*byte, CUdevice) CUresult
	cuDeviceGetUuidV2 func(*byte, CUdevice) CUresult
)

func loadUUID(lib *ffi.Lib) {
	if lib.Has("cuDeviceGetUuid") {
		cuDeviceGetUuid = ffi.Fn2[CUresult, *byte, CUdevice](lib, "cuDeviceGetUuid")
	}
	if lib.Has("cuDeviceGetUuid_v2") {
		cuDeviceGetUuidV2 = ffi.Fn2[CUresult, *byte, CUdevice](lib, "cuDeviceGetUuid_v2")
	}
}

// Stream capture and graph replay, bound only if the driver has all of them,
// so a driver too old for graphs degrades to issuing launches one at a time.
var (
	cuStreamCreate       func(*CUstream, uint32) CUresult
	cuStreamDestroy      func(CUstream) CUresult
	cuStreamBeginCapture func(CUstream, int32) CUresult
	cuStreamEndCapture   func(CUstream, *CUgraph) CUresult
	cuGraphInstantiate   func(*CUgraphExec, CUgraph, uint64) CUresult
	cuGraphLaunch        func(CUgraphExec, CUstream) CUresult
	cuGraphDestroy       func(CUgraph) CUresult
	cuGraphExecDestroy   func(CUgraphExec) CUresult
)

// graphSyms is the whole set: it is bound all or nothing, so no call site has
// to test more than GraphsAvailable.
var graphSyms = []string{
	"cuStreamCreate", "cuStreamDestroy_v2", "cuStreamBeginCapture_v2",
	"cuStreamEndCapture", "cuGraphInstantiateWithFlags", "cuGraphLaunch",
	"cuGraphDestroy", "cuGraphExecDestroy",
}

func loadGraphs(lib *ffi.Lib) {
	for _, n := range graphSyms {
		if !lib.Has(n) {
			return
		}
	}
	cuStreamCreate = ffi.Fn2[CUresult, *CUstream, uint32](lib, "cuStreamCreate")
	cuStreamDestroy = ffi.Fn1[CUresult, CUstream](lib, "cuStreamDestroy_v2")
	cuStreamBeginCapture = ffi.Fn2[CUresult, CUstream, int32](lib, "cuStreamBeginCapture_v2")
	cuStreamEndCapture = ffi.Fn2[CUresult, CUstream, *CUgraph](lib, "cuStreamEndCapture")
	// cuGraphInstantiate has changed shape twice across CUDA versions;
	// WithFlags has one signature and has been in the driver since 11.4.
	cuGraphInstantiate = ffi.Fn3[CUresult, *CUgraphExec, CUgraph, uint64](lib, "cuGraphInstantiateWithFlags")
	cuGraphLaunch = ffi.Fn2[CUresult, CUgraphExec, CUstream](lib, "cuGraphLaunch")
	cuGraphDestroy = ffi.Fn1[CUresult, CUgraph](lib, "cuGraphDestroy")
	cuGraphExecDestroy = ffi.Fn1[CUresult, CUgraphExec](lib, "cuGraphExecDestroy")
}

// Events, for timing on the device's own clock; optional like the graph calls.
var (
	cuEventCreate      func(*CUevent, uint32) CUresult
	cuEventRecord      func(CUevent, CUstream) CUresult
	cuEventElapsedTime func(*float32, CUevent, CUevent) CUresult
	cuEventDestroy     func(CUevent) CUresult
	cuEventQuery       func(CUevent) CUresult
	// cuEventRecordWithFlags with CU_EVENT_RECORD_EXTERNAL makes an event
	// recorded during stream capture a timed node; a plain record there is
	// only a dependency edge, which cuEventElapsedTime refuses.
	cuEventRecordWithFlags func(CUevent, CUstream, uint32) CUresult
)

func loadEvents(lib *ffi.Lib) {
	for _, n := range []string{"cuEventCreate", "cuEventRecord", "cuEventElapsedTime", "cuEventDestroy_v2", "cuEventQuery"} {
		if !lib.Has(n) {
			return
		}
	}
	cuEventCreate = ffi.Fn2[CUresult, *CUevent, uint32](lib, "cuEventCreate")
	cuEventRecord = ffi.Fn2[CUresult, CUevent, CUstream](lib, "cuEventRecord")
	cuEventElapsedTime = ffi.Fn3[CUresult, *float32, CUevent, CUevent](lib, "cuEventElapsedTime")
	cuEventDestroy = ffi.Fn1[CUresult, CUevent](lib, "cuEventDestroy_v2")
	cuEventQuery = ffi.Fn1[CUresult, CUevent](lib, "cuEventQuery")
	if lib.Has("cuEventRecordWithFlags") {
		cuEventRecordWithFlags = ffi.Fn3[CUresult, CUevent, CUstream, uint32](lib, "cuEventRecordWithFlags")
	}
}

// EventsAvailable reports whether this driver can time on the device clock.
func EventsAvailable() bool { return cuEventElapsedTime != nil }

// Event is a point in a stream, stamped by the device when the stream reaches
// it. Like every call here, it lives on the context-owning goroutine.
type Event struct{ e CUevent }

// NewEvent makes a timing event.
func NewEvent() (*Event, error) {
	if !EventsAvailable() {
		return nil, fmt.Errorf("cuda: this driver has no events")
	}
	var e CUevent
	if err := call(cuEventCreate(&e, 0), "cuEventCreate"); err != nil {
		return nil, err
	}
	return &Event{e}, nil
}

// Record stamps the event when s reaches this point.
func (e *Event) Record(s *Stream) error { return call(cuEventRecord(e.e, s.Handle()), "cuEventRecord") }

// RecordOn stamps the event when the stream with handle h (0 is the legacy
// stream) reaches this point.
func (e *Event) RecordOn(h CUstream) error { return call(cuEventRecord(e.e, h), "cuEventRecord") }

// RecordNode records the event into the graph h is capturing, as a node that
// stamps the device clock on every replay (CU_EVENT_RECORD_EXTERNAL).
func (e *Event) RecordNode(h CUstream) error {
	if cuEventRecordWithFlags == nil {
		return fmt.Errorf("cuda: this driver cannot record an event node")
	}
	return call(cuEventRecordWithFlags(e.e, h, 1), "cuEventRecordWithFlags")
}

// Done reports whether the device has reached the event.
func (e *Event) Done() bool { return cuEventQuery(e.e) == 0 }

// Elapsed is the device time from a to b, in milliseconds; both must have
// completed.
func Elapsed(a, b *Event) (float32, error) {
	var ms float32
	err := call(cuEventElapsedTime(&ms, a.e, b.e), "cuEventElapsedTime")
	return ms, err
}

// Destroy releases the event.
func (e *Event) Destroy() { cuEventDestroy(e.e) }

// GraphsAvailable reports whether this driver can capture and replay a launch
// sequence.
func GraphsAvailable() bool { return cuGraphLaunch != nil }

// loadCUDA binds the driver once per process. The Once prevents a race: the
// entry points are package variables, and a second device opened on another
// goroutine would rewrite them while a launch reads them.
func loadCUDA() error {
	loadOnce.Do(func() { loadErr = bindCUDA() })
	return loadErr
}

var (
	loadOnce sync.Once
	loadErr  error
)

// driverNames is every name the CUDA driver ships under, in the order they are
// tried; ffi.Open takes the first that loads. nvcuda.dll (Windows, found in
// System32) is required: without it Windows silently reported no CUDA device.
// The WSL path is where a Linux binary under WSL finds its driver. A package
// variable so a gate can read it.
var driverNames = []string{
	"libcuda.so.1", "libcuda.so", "/usr/lib/wsl/lib/libcuda.so.1", // linux, incl. WSL
	"nvcuda.dll", // windows
}

func bindCUDA() error {
	lib, err := ffi.Open(driverNames...)
	if err != nil {
		return err
	}
	cuInit = ffi.Fn1[CUresult, uint32](lib, "cuInit")
	cuDriverGetVersion = ffi.Fn1[CUresult, *int32](lib, "cuDriverGetVersion")
	cuDeviceGet = ffi.Fn2[CUresult, *CUdevice, int32](lib, "cuDeviceGet")
	cuDeviceGetCount = ffi.Fn1[CUresult, *int32](lib, "cuDeviceGetCount")
	cuDeviceGetName = ffi.Fn3[CUresult, *byte, int32, CUdevice](lib, "cuDeviceGetName")
	cuCtxCreate = ffi.Fn3[CUresult, *CUctx, uint32, CUdevice](lib, "cuCtxCreate_v2")
	cuCtxSetCurrent = ffi.Fn1[CUresult, CUctx](lib, "cuCtxSetCurrent")
	cuCtxGetDevice = ffi.Fn1[CUresult, *CUdevice](lib, "cuCtxGetDevice")
	cuCtxSynchronize = ffi.Fn0[CUresult](lib, "cuCtxSynchronize")
	cuCtxDestroy = ffi.Fn1[CUresult, CUctx](lib, "cuCtxDestroy_v2")
	cuMemAlloc = ffi.Fn2[CUresult, *CUdevptr, uint64](lib, "cuMemAlloc_v2")
	cuMemFree = ffi.Fn1[CUresult, CUdevptr](lib, "cuMemFree_v2")
	cuMemcpyHtoD = ffi.Fn3[CUresult, CUdevptr, unsafe.Pointer, uint64](lib, "cuMemcpyHtoD_v2")
	cuMemcpyDtoH = ffi.Fn3[CUresult, unsafe.Pointer, CUdevptr, uint64](lib, "cuMemcpyDtoH_v2")
	cuMemcpyDtoD = ffi.Fn3[CUresult, CUdevptr, CUdevptr, uint64](lib, "cuMemcpyDtoD_v2")
	cuModuleLoadData = ffi.Fn2[CUresult, *CUmodule, unsafe.Pointer](lib, "cuModuleLoadData")
	cuModuleUnload = ffi.Fn1[CUresult, CUmodule](lib, "cuModuleUnload")
	cuModuleGetFunction = ffi.Fn3[CUresult, *CUfunc, CUmodule, *byte](lib, "cuModuleGetFunction")
	cuLaunchKernel = ffi.Fn11[CUresult, CUfunc, uint32, uint32, uint32, uint32, uint32, uint32, uint32, CUstream, unsafe.Pointer, unsafe.Pointer](lib, "cuLaunchKernel")
	cuGetErrorString = ffi.Fn2[CUresult, CUresult, *unsafe.Pointer](lib, "cuGetErrorString")
	cuMemGetInfo = ffi.Fn2[CUresult, *uint64, *uint64](lib, "cuMemGetInfo_v2")
	cuDeviceGetAttribute = ffi.Fn3[CUresult, *int32, int32, CUdevice](lib, "cuDeviceGetAttribute")
	cuMemHostAlloc = ffi.Fn3[CUresult, *unsafe.Pointer, uint64, uint32](lib, "cuMemHostAlloc")
	cuMemFreeHost = ffi.Fn1[CUresult, unsafe.Pointer](lib, "cuMemFreeHost")
	cuMemHostRegister = ffi.Fn3[CUresult, unsafe.Pointer, uint64, uint32](lib, "cuMemHostRegister_v2")
	cuMemHostUnregister = ffi.Fn1[CUresult, unsafe.Pointer](lib, "cuMemHostUnregister")
	cuMemHostGetDevicePointer = ffi.Fn3[CUresult, *CUdevptr, unsafe.Pointer, uint32](lib, "cuMemHostGetDevicePointer_v2")
	loadGraphs(lib)
	loadCoop(lib)
	loadQueues(lib)
	loadEvents(lib)
	loadUUID(lib)
	loadFuncAttrs(lib)
	loadPeer(lib)
	return nil
}

func errStr(r CUresult) string {
	// The driver's pointer is taken as unsafe.Pointer, never uintptr, and
	// walked with unsafe.Add.
	var p unsafe.Pointer
	cuGetErrorString(r, &p)
	if p == nil {
		return fmt.Sprintf("CUresult(%d)", r)
	}
	var b []byte
	for i := 0; ; i++ {
		c := *(*byte)(unsafe.Add(p, i))
		if c == 0 {
			break
		}
		b = append(b, c)
	}
	return fmt.Sprintf("%s (CUresult=%d)", string(b), r)
}

func ck(r CUresult, what string) {
	if r != 0 {
		panic(fmt.Sprintf("%s failed: %s", what, errStr(r)))
	}
}
