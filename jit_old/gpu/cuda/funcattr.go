package cuda

import "github.com/jitllm/jitllm/jit_old/gpu/ffi"

// What ptxas allocated for a loaded kernel, read back from the driver: the
// per-kernel facts the register report and its gate are made of.
var (
	cuFuncGetAttribute                          func(*int32, int32, CUfunc) CUresult
	cuOccupancyMaxActiveBlocksPerMultiprocessor func(*int32, CUfunc, int32, uint64) CUresult
)

// CUfunction_attribute values (cuda.h).
const (
	funcMaxThreads = 0 // CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
	funcShared     = 1 // CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
	funcLocal      = 3 // CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
	funcNumRegs    = 4 // CU_FUNC_ATTRIBUTE_NUM_REGS
)

func loadFuncAttrs(lib *ffi.Lib) {
	if lib.Has("cuFuncGetAttribute") {
		cuFuncGetAttribute = ffi.Fn3[CUresult, *int32, int32, CUfunc](lib, "cuFuncGetAttribute")
	}
	if lib.Has("cuOccupancyMaxActiveBlocksPerMultiprocessor") {
		cuOccupancyMaxActiveBlocksPerMultiprocessor = ffi.Fn4[CUresult, *int32, CUfunc, int32, uint64](
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
	if cuFuncGetAttribute == nil {
		return a, false
	}
	get := func(attr int32) int {
		var v int32
		if cuFuncGetAttribute(&v, attr, m.f) != 0 {
			return -1
		}
		return int(v)
	}
	a.Regs, a.Local = get(funcNumRegs), get(funcLocal)
	a.Shared, a.MaxThreads = get(funcShared), get(funcMaxThreads)
	if cuOccupancyMaxActiveBlocksPerMultiprocessor != nil && block > 0 {
		var n int32
		if cuOccupancyMaxActiveBlocksPerMultiprocessor(&n, m.f, int32(block), 0) == 0 {
			a.Blocks = int(n)
		}
	}
	return a, true
}

// MaxThreadsPerSM is CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, 0
// when the driver would not say: the denominator of an occupancy.
func (d *Device) MaxThreadsPerSM() int {
	if cuDeviceGetAttribute == nil {
		return 0
	}
	var v int32
	if cuDeviceGetAttribute(&v, attrMaxTPM, d.dev) != 0 {
		return 0
	}
	return int(v)
}
