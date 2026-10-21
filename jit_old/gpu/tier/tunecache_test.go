package tier

import (
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
)

// idDev is a fake device that names its hardware and driver, as CUDA and
// Vulkan do, so the tuners may cache against it.
type idDev struct {
	*fakeDev
	uuid, drv string
}

func (d idDev) Identity() backend.Identity {
	return backend.Identity{Key: "uuid:" + d.uuid, Source: "test"}
}
func (d idDev) DriverVersion() string { return d.drv }
