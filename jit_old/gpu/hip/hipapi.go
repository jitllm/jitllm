package hip

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/gpu/ffi"
)

// hipError is hipError_t. Zero is hipSuccess.
type hipError int32

// The hip_runtime_api.h values this package uses, read off ROCm 6.4's header
// by a compiled probe rather than by counting enum lines (WarpSize is 87, not
// the 89 a count gives).
const (
	errNoDevice = 100 // hipErrorNoDevice

	attrIntegrated       = 16 // hipDeviceAttributeIntegrated
	attrMaxGridDimX      = 29 // hipDeviceAttributeMaxGridDimX
	attrMaxThreadsPerCU  = 57 // hipDeviceAttributeMaxThreadsPerMultiProcessor
	attrMultiprocessors  = 63 // hipDeviceAttributeMultiprocessorCount
	attrWarpSize         = 87 // hipDeviceAttributeWarpSize
	attrPCIBusID         = 67 // hipDeviceAttributePciBusId
	attrPCIDeviceID      = 68 // hipDeviceAttributePciDeviceId
	attrPCIDomainID      = 69 // hipDeviceAttributePciDomainID
	attrMaxSharedPerBlok = 74 // hipDeviceAttributeMaxSharedMemoryPerBlock

	// propsArchOffset is where gcnArchName sits in hipDeviceProp_tR0600, the
	// layout hipGetDevicePropertiesR0600 fills (offsetof, from a compiled
	// probe against ROCm 6.4's header). propsSize is that struct's size;
	// the buffer handed over is larger, so a longer struct from a later
	// runtime writes into slack rather than past the end.
	propsArchOffset = 1160
	propsSize       = 1472
	propsBuf        = 8192
	archLen         = 256

	hipMemcpyDefault = 4
	streamNonBlock   = 1 // hipStreamNonBlocking
)

type hipAPI struct {
	init           func(uint32) hipError
	driverVersion  func(*int32) hipError
	runtimeVersion func(*int32) hipError
	errorString    func(hipError) unsafe.Pointer
	deviceCount    func(*int32) hipError
	setDevice      func(int32) hipError
	deviceName     func(*byte, int32, int32) hipError
	attribute      func(*int32, int32, int32) hipError
	properties     func(*byte, int32) hipError
	uuid           func(*byte, int32) hipError
	memInfo        func(*uint64, *uint64) hipError
	malloc         func(*uintptr, uint64) hipError
	free           func(uintptr) hipError
	htod           func(uintptr, unsafe.Pointer, uint64) hipError
	dtoh           func(unsafe.Pointer, uintptr, uint64) hipError
	dtod           func(uintptr, uintptr, uint64) hipError
	// hipMemcpyWithStream bound twice, by direction, so a device address is
	// never made into an unsafe.Pointer the collector would inspect.
	htodStream     func(uintptr, unsafe.Pointer, uint64, int32, uintptr) hipError
	dtohStream     func(unsafe.Pointer, uintptr, uint64, int32, uintptr) hipError
	sync           func() hipError
	streamCreate   func(*uintptr, uint32) hipError
	streamDestroy  func(uintptr) hipError
	streamSync     func(uintptr) hipError
	moduleLoad     func(*uintptr, unsafe.Pointer) hipError
	moduleUnload   func(uintptr) hipError
	moduleFunction func(*uintptr, uintptr, *byte) hipError
	launch         func(uintptr, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) hipError
}

func bindHIP(names []string) (*hipAPI, error) {
	l, err := openFirst("libamdhip64", names)
	if err != nil {
		return nil, err
	}
	a := &hipAPI{}
	err = bindSafely("libamdhip64", func() {
		a.init = ffi.Fn1[hipError, uint32](l, "hipInit")
		a.driverVersion = ffi.Fn1[hipError, *int32](l, "hipDriverGetVersion")
		a.runtimeVersion = ffi.Fn1[hipError, *int32](l, "hipRuntimeGetVersion")
		a.errorString = ffi.Fn1[unsafe.Pointer, hipError](l, "hipGetErrorString")
		a.deviceCount = ffi.Fn1[hipError, *int32](l, "hipGetDeviceCount")
		a.setDevice = ffi.Fn1[hipError, int32](l, "hipSetDevice")
		a.deviceName = ffi.Fn3[hipError, *byte, int32, int32](l, "hipDeviceGetName")
		a.attribute = ffi.Fn3[hipError, *int32, int32, int32](l, "hipDeviceGetAttribute")
		// The R0600 entry point is the one whose layout propsArchOffset
		// describes. A runtime without it is older than ROCm 6 and is refused
		// below rather than read at the wrong offset.
		if l.Has("hipGetDevicePropertiesR0600") {
			a.properties = ffi.Fn2[hipError, *byte, int32](l, "hipGetDevicePropertiesR0600")
		}
		a.uuid = ffi.Fn2[hipError, *byte, int32](l, "hipDeviceGetUuid")
		a.memInfo = ffi.Fn2[hipError, *uint64, *uint64](l, "hipMemGetInfo")
		a.malloc = ffi.Fn2[hipError, *uintptr, uint64](l, "hipMalloc")
		a.free = ffi.Fn1[hipError, uintptr](l, "hipFree")
		a.htod = ffi.Fn3[hipError, uintptr, unsafe.Pointer, uint64](l, "hipMemcpyHtoD")
		a.dtoh = ffi.Fn3[hipError, unsafe.Pointer, uintptr, uint64](l, "hipMemcpyDtoH")
		a.dtod = ffi.Fn3[hipError, uintptr, uintptr, uint64](l, "hipMemcpyDtoD")
		a.htodStream = ffi.Fn5[hipError, uintptr, unsafe.Pointer, uint64, int32, uintptr](l, "hipMemcpyWithStream")
		a.dtohStream = ffi.Fn5[hipError, unsafe.Pointer, uintptr, uint64, int32, uintptr](l, "hipMemcpyWithStream")
		a.sync = ffi.Fn0[hipError](l, "hipDeviceSynchronize")
		a.streamCreate = ffi.Fn2[hipError, *uintptr, uint32](l, "hipStreamCreateWithFlags")
		a.streamDestroy = ffi.Fn1[hipError, uintptr](l, "hipStreamDestroy")
		a.streamSync = ffi.Fn1[hipError, uintptr](l, "hipStreamSynchronize")
		a.moduleLoad = ffi.Fn2[hipError, *uintptr, unsafe.Pointer](l, "hipModuleLoadData")
		a.moduleUnload = ffi.Fn1[hipError, uintptr](l, "hipModuleUnload")
		a.moduleFunction = ffi.Fn3[hipError, *uintptr, uintptr, *byte](l, "hipModuleGetFunction")
		a.launch = ffi.Fn11[hipError, uintptr, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer](l, "hipModuleLaunchKernel")
	})
	if err != nil {
		return nil, err
	}
	if a.properties == nil {
		return nil, fmt.Errorf("hip: libamdhip64 has no hipGetDevicePropertiesR0600; ROCm 6 or later is required")
	}
	return a, nil
}

func (a *hipAPI) check(e hipError, what string) error {
	if e == 0 {
		return nil
	}
	return fmt.Errorf("hip: %s: %s (hipError=%d)", what, goString(a.errorString(e)), e)
}
