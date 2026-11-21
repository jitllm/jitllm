package cuda

import (
	"fmt"

	"github.com/jitllm/jitllm/backend/ffi"
)

// CUfunction_attribute values (cuda.h).
const (
	funcMaxThreads = 0 // CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
	funcShared     = 1 // CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
	funcLocal      = 3 // CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
	funcNumRegs    = 4 // CU_FUNC_ATTRIBUTE_NUM_REGS
	funcMaxDynamic = 8 // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
)

func (drv *Driver) loadFuncAttrs(lib *ffi.Lib) {
	if lib.Has("cuFuncGetAttribute") {
		drv.cuFuncGetAttribute = ffi.Fn3[CUresult, *int32, int32, CUfunc](lib, "cuFuncGetAttribute")
	}
	if lib.Has("cuFuncSetAttribute") {
		drv.cuFuncSetAttribute = ffi.Fn3[CUresult, CUfunc, int32, int32](lib, "cuFuncSetAttribute")
	}
	if lib.Has("cuOccupancyMaxActiveBlocksPerMultiprocessor") {
		drv.cuOccupancyMaxActiveBlocksPerMultiprocessor = ffi.Fn4[CUresult, *int32, CUfunc, int32, uint64](
			lib, "cuOccupancyMaxActiveBlocksPerMultiprocessor")
	}
}

// Attrs is what the driver says a loaded kernel costs a multiprocessor.
type Attrs struct {
	// Regs is registers per thread; Local is local memory per thread in
	// bytes, which for generated code with no local arrays is spill space.
	Regs, Local int
	// Shared is static shared memory per block in bytes; MaxThreads the
	// largest block this kernel can launch with under its register use.
	Shared, MaxThreads int
	// Blocks is how many blocks of the asked width fit one multiprocessor at
	// once (cuOccupancyMaxActiveBlocksPerMultiprocessor), 0 when the driver
	// would not say.
	Blocks int
}

// Attrs reads m's resources for a launch block threads wide. ok is false on
// a driver that does not export cuFuncGetAttribute. It is a driver call and
// runs on the device's owner thread, as every other one does.
func (m *Module) Attrs(block int) (a Attrs, ok bool) {
	if m.drv.cuFuncGetAttribute == nil {
		return a, false
	}
	get := func(attr int32) int {
		var v int32
		if m.drv.cuFuncGetAttribute(&v, attr, m.f) != 0 {
			return -1
		}
		return int(v)
	}
	a.Regs, a.Local = get(funcNumRegs), get(funcLocal)
	a.Shared, a.MaxThreads = get(funcShared), get(funcMaxThreads)
	if m.drv.cuOccupancyMaxActiveBlocksPerMultiprocessor != nil && block > 0 {
		var n int32
		if m.drv.cuOccupancyMaxActiveBlocksPerMultiprocessor(&n, m.f, int32(block), uint64(m.dynShared)) == 0 {
			a.Blocks = int(n)
		}
	}
	return a, true
}

// MaxThreadsPerSM is CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, 0
// when the driver would not say: the denominator of an occupancy.
func (d *Device) MaxThreadsPerSM() int {
	if d.drv.cuDeviceGetAttribute == nil {
		return 0
	}
	var v int32
	if d.drv.cuDeviceGetAttribute(&v, attrMaxTPM, d.dev) != 0 {
		return 0
	}
	return int(v)
}

// SetDynamicShared opts m into n bytes of dynamic shared memory a launch,
// which every launch of it then asks for: a kernel whose shared arrays are
// past the static limit addresses them there (ptx.DynamicShared). The
// driver refuses n past the device's opt-in limit
// (CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN), and the error
// says so.
func (m *Module) SetDynamicShared(n int) error {
	if m.drv.cuFuncSetAttribute == nil {
		return fmt.Errorf("cuda: this driver does not export cuFuncSetAttribute, so a kernel cannot ask for %d bytes of shared memory", n)
	}
	if err := m.drv.call(m.drv.cuFuncSetAttribute(m.f, funcMaxDynamic, int32(n)), "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)"); err != nil {
		return fmt.Errorf("%w: %d bytes of shared memory a workgroup", err, n)
	}
	m.dynShared = n
	return nil
}
