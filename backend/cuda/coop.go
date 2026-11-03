package cuda

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/backend/ffi"
)

// attrCoopLaunch is CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH.
const attrCoopLaunch = 95

func (drv *Driver) loadCoop(lib *ffi.Lib) {
	if lib.Has("cuLaunchCooperativeKernel") {
		drv.cuLaunchCooperativeKernel = ffi.Fn10[CUresult, CUfunc, uint32, uint32, uint32, uint32, uint32, uint32, uint32, CUstream, unsafe.Pointer](lib, "cuLaunchCooperativeKernel")
	}
	if lib.Has("cuOccupancyMaxActiveBlocksPerMultiprocessor") {
		drv.cuOccupancy = ffi.Fn4[CUresult, *int32, CUfunc, int32, uint64](lib, "cuOccupancyMaxActiveBlocksPerMultiprocessor")
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
	if err := m.drv.call(m.drv.cuOccupancy(&n, m.f, int32(block), uint64(m.dynShared)), "cuOccupancyMaxActiveBlocksPerMultiprocessor"); err != nil {
		return 0, err
	}
	return int(n) * m.sms, nil
}
