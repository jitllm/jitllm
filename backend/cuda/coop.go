package cuda

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/backend/ffi"
)

// The cooperative launch, for a kernel with a grid barrier.
//
// The CUDA Programming Guide's grid synchronisation section requires every
// block of a grid that synchronises to be resident at once, sized with the
// occupancy calculator (cuOccupancyMaxActiveBlocksPerMultiprocessor times the
// multiprocessor count), and launched with cuLaunchCooperativeKernel, which
// guarantees that co-residency or refuses the launch
// (CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE). A plain launch of the same grid
// would usually work and would deadlock whenever another stream's work held a
// multiprocessor the grid was counted against.
var (
	cuLaunchCooperativeKernel func(CUfunc, uint32, uint32, uint32, uint32, uint32, uint32, uint32, CUstream, unsafe.Pointer) CUresult
	cuOccupancy               func(*int32, CUfunc, int32, uint64) CUresult
)

// attrCoopLaunch is CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH.
const attrCoopLaunch = 95

func loadCoop(lib *ffi.Lib) {
	if lib.Has("cuLaunchCooperativeKernel") {
		cuLaunchCooperativeKernel = ffi.Fn10[CUresult, CUfunc, uint32, uint32, uint32, uint32, uint32, uint32, uint32, CUstream, unsafe.Pointer](lib, "cuLaunchCooperativeKernel")
	}
	if lib.Has("cuOccupancyMaxActiveBlocksPerMultiprocessor") {
		cuOccupancy = ffi.Fn4[CUresult, *int32, CUfunc, int32, uint64](lib, "cuOccupancyMaxActiveBlocksPerMultiprocessor")
	}
}

// Resident is how many blocks of block threads this module's kernel can have
// resident on the device at once: the occupancy calculator's blocks per
// multiprocessor (which counts the kernel's registers and shared memory) times
// the multiprocessor count. It is the most a cooperative launch may carry. An
// error says the device cannot launch cooperatively at all.
func (m *Module) Resident(block int) (int, error) {
	if !m.devCoop {
		return 0, fmt.Errorf("cuda: this device does not take a cooperative launch")
	}
	var n int32
	if err := call(cuOccupancy(&n, m.f, int32(block), 0), "cuOccupancyMaxActiveBlocksPerMultiprocessor"); err != nil {
		return 0, err
	}
	return int(n) * m.sms, nil
}
