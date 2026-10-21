package backend

import (
	"fmt"
	"os"
	"strconv"
)

// The package reads no environment; its test binary does, to point the suite's
// Open at a device the default rule would not pick. Each variable is the
// Opts field of the same meaning:
//
//	JITLLM_VK_DEVICE     Vulkan device pin (Opts.Vulkan.Device), e.g. "1" for
//	                     the second card -- the default rule takes a discrete
//	                     card first, so without it an iGPU is never tested
//	JITLLM_VK_MAXGROUPS  workgroups per dispatch (Opts.Vulkan.MaxGroups), e.g.
//	                     "7", which sends nearly every launch in the suite
//	                     through a split dispatch on every Vulkan device
//	JITLLM_ROCM          the ROCm library directory (Opts.HIP.Path); unset
//	                     searches the default places
func init() {
	openDefaults.HIP.Path = os.Getenv("JITLLM_ROCM")
	if v := os.Getenv("JITLLM_VK_DEVICE"); v != "" {
		openDefaults.Vulkan.Device = v
	}
	if v := os.Getenv("JITLLM_VK_MAXGROUPS"); v != "" {
		n, err := strconv.ParseUint(v, 10, 32)
		if err != nil || n == 0 {
			panic(fmt.Sprintf("JITLLM_VK_MAXGROUPS=%q: want a positive count of workgroups", v))
		}
		openDefaults.Vulkan.MaxGroups = uint32(n)
	}
}

// VulkanGroups reports a Vulkan device's workgroups per dispatch and how many
// dispatches it has recorded, and false for any other device. The count is the
// selection check for a split launch: a split and an unsplit launch leave the
// same buffer behind.
func VulkanGroups(d Device) (limit uint32, dispatches uint64, ok bool) {
	v, ok := d.(*vkDev)
	if !ok {
		return 0, 0, false
	}
	return v.c.MaxGroups(), v.c.Dispatches(), true
}

// MetalKernelWrites overrides the write mask a Metal kernel was compiled
// with: the violation arm of the concurrent encoder's gate.
func MetalKernelWrites(k Kernel, m uint64) { k.(*mtlKern).k.SetWrites(m) }
