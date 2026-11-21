package vulkan

import "runtime"

// The device's compute-unit count, for the grid barrier's co-residency limit
// (ir.OpGridBarrier, backend.GridLimit).
//
// Core Vulkan states no such count and no forward-progress guarantee between
// workgroups. Two vendor extensions state the count: VK_NV_shader_sm_builtins
// (shaderSMCount) and VK_AMD_shader_core_properties (shader engines x arrays
// per engine x compute units per array). One workgroup per unit is the bound
// taken: a kernel that can be dispatched at all fits one of its workgroups on
// one unit, so that many are resident at once whenever nothing else holds the
// units -- a reasoned bound, not a driver guarantee like CUDA's cooperative
// launch, and the reason a persistent kernel stays off by default here.
const (
	extSMBuiltins      = "VK_NV_shader_sm_builtins"
	extAMDCoreProps    = "VK_AMD_shader_core_properties"
	stSMBuiltinsProps  = 1000154001 // ..._SHADER_SM_BUILTINS_PROPERTIES_NV
	stAMDCoreProps     = 1000185000 // ..._SHADER_CORE_PROPERTIES_AMD
	apiVersionForProps = apiVersion11
)

type smBuiltinsProps struct {
	sType     uint32
	_         uint32
	pNext     uintptr
	smCount   uint32
	warpsPerS uint32
}

// amdCoreProps is VkPhysicalDeviceShaderCorePropertiesAMD.
type amdCoreProps struct {
	sType                     uint32
	_                         uint32
	pNext                     uintptr
	shaderEngineCount         uint32
	shaderArraysPerEngine     uint32
	computeUnitsPerShaderArr  uint32
	simdPerComputeUnit        uint32
	wavefrontsPerSimd         uint32
	wavefrontSize             uint32
	sgprsPerSimd              uint32
	minSgprAllocation         uint32
	maxSgprAllocation         uint32
	sgprAllocationGranularity uint32
	vgprsPerSimd              uint32
	minVgprAllocation         uint32
	maxVgprAllocation         uint32
	vgprAllocationGranularity uint32
}

// nvWarp is an NVIDIA warp's threads, which VK_NV_shader_sm_builtins counts
// its residency in (shaderWarpsPerSM; its SubgroupSize builtin is 32).
const nvWarp = 32

// readCores asks the two vendor extensions for the compute-unit count; 0 and
// an empty source when neither answers. VK_AMD_shader_core_properties states
// a SIMD's registers and its most resident waves too, and regs is a lane's
// share of them at that occupancy; VK_NV_shader_sm_builtins states none.
// slots is the threads the device holds resident, each extension's units
// times its most resident threads a unit, 0 where it does not say.
func (l *Loader) readCores(p PhysDevice, apiVer uint32) (cores int, from string, regs, slots int) {
	if apiVer < apiVersionForProps {
		return 0, "", 0, 0
	}
	if ok, err := l.hasExtension(p, extSMBuiltins); err == nil && ok {
		sp := onHeap[smBuiltinsProps]()
		sp.sType = stSMBuiltinsProps
		p2 := physProps2{sType: stPhysDevProps2, pNext: uintptr(up(sp))}
		l.vkGetPhysicalDeviceProperties2(p, up(&p2))
		runtime.KeepAlive(sp)
		if sp.smCount > 0 {
			return int(sp.smCount), extSMBuiltins, 0, int(sp.smCount * sp.warpsPerS * nvWarp)
		}
	}
	if ok, err := l.hasExtension(p, extAMDCoreProps); err == nil && ok {
		ap := onHeap[amdCoreProps]()
		ap.sType = stAMDCoreProps
		p2 := physProps2{sType: stPhysDevProps2, pNext: uintptr(up(ap))}
		l.vkGetPhysicalDeviceProperties2(p, up(&p2))
		runtime.KeepAlive(ap)
		if n := ap.shaderEngineCount * ap.shaderArraysPerEngine * ap.computeUnitsPerShaderArr; n > 0 {
			if ap.wavefrontsPerSimd > 0 {
				regs = int(ap.vgprsPerSimd / ap.wavefrontsPerSimd)
			}
			return int(n), extAMDCoreProps, regs, int(n * ap.simdPerComputeUnit * ap.wavefrontsPerSimd * ap.wavefrontSize)
		}
	}
	return 0, "", 0, 0
}

// Cores is the device's compute-unit count and the extension that stated it,
// or 0 and "" when no extension does.
func (c *Ctx) Cores() (int, string) { return c.cores, c.coresFrom }

// Slots is the threads the device holds resident as a vendor extension
// states them (readCores), 0 where none does.
func (c *Ctx) Slots() int { return c.slots }

// ThreadRegisters is a lane's registers when its SIMD holds the most waves it
// can, as a vendor extension states them, 0 where none does.
func (c *Ctx) ThreadRegisters() int { return c.threadRegs }
