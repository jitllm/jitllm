//go:build darwin

package metal

import (
	"sync"
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/gpu/ffi"
)

// GPUCores is the GPU's core count as the IORegistry states it: the
// "gpu-core-count" property of the AGXAccelerator service (the number System
// Information reports), or 0 when it cannot be read. Metal states neither the
// count nor any forward-progress guarantee between threadgroups; the grid
// barrier's co-residency limit (backend.GridLimit) is one threadgroup per core,
// a reasoned bound rather than a driver guarantee.
func GPUCores() int {
	coresOnce.Do(func() { cores = readGPUCores() })
	return cores
}

var (
	coresOnce sync.Once
	cores     int
)

func readGPUCores() int {
	iokit, err := ffi.Open("/System/Library/Frameworks/IOKit.framework/IOKit")
	if err != nil {
		return 0
	}
	cf, err := ffi.Open("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
	if err != nil {
		return 0
	}
	matching := ffi.Fn1[unsafe.Pointer, *byte](iokit, "IOServiceMatching")
	getService := ffi.Fn2[uint32, uint32, unsafe.Pointer](iokit, "IOServiceGetMatchingService")
	property := ffi.Fn4[unsafe.Pointer, uint32, unsafe.Pointer, unsafe.Pointer, uint32](iokit, "IORegistryEntryCreateCFProperty")
	release := ffi.Fn1[int32, uint32](iokit, "IOObjectRelease")
	cfString := ffi.Fn3[unsafe.Pointer, unsafe.Pointer, *byte, uint32](cf, "CFStringCreateWithCString")
	cfNumber := ffi.Fn3[uint8, unsafe.Pointer, int32, *int64](cf, "CFNumberGetValue")
	cfRelease := ffi.Fn1[ffi.None, unsafe.Pointer](cf, "CFRelease")

	class := []byte("AGXAccelerator\x00")
	// IOServiceGetMatchingService consumes the matching dictionary.
	svc := getService(0, matching(&class[0]))
	if svc == 0 {
		return 0
	}
	defer release(svc)
	keyName := []byte("gpu-core-count\x00")
	const utf8 = 0x08000100 // kCFStringEncodingUTF8
	key := cfString(nil, &keyName[0], utf8)
	if key == nil {
		return 0
	}
	defer cfRelease(key)
	num := property(svc, key, nil, 0)
	if num == nil {
		return 0
	}
	defer cfRelease(num)
	var n int64
	const sint64 = 4 // kCFNumberSInt64Type
	if cfNumber(num, sint64, &n) == 0 || n <= 0 {
		return 0
	}
	return int(n)
}
