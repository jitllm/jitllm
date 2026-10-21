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

// readCores asks the two vendor extensions for the compute-unit count; 0 and
// an empty source when neither answers.
func readCores(p PhysDevice, apiVer uint32) (int, string) {
	if apiVer < apiVersionForProps {
		return 0, ""
	}
	if ok, err := hasExtension(p, extSMBuiltins); err == nil && ok {
		var sp smBuiltinsProps
		sp.sType = stSMBuiltinsProps
		p2 := physProps2{sType: stPhysDevProps2, pNext: uintptr(up(&sp))}
		vkGetPhysicalDeviceProperties2(p, up(&p2))
		runtime.KeepAlive(sp)
		if sp.smCount > 0 {
			return int(sp.smCount), extSMBuiltins
		}
	}
	if ok, err := hasExtension(p, extAMDCoreProps); err == nil && ok {
		var ap amdCoreProps
		ap.sType = stAMDCoreProps
		p2 := physProps2{sType: stPhysDevProps2, pNext: uintptr(up(&ap))}
		vkGetPhysicalDeviceProperties2(p, up(&p2))
		runtime.KeepAlive(ap)
		if n := ap.shaderEngineCount * ap.shaderArraysPerEngine * ap.computeUnitsPerShaderArr; n > 0 {
			return int(n), extAMDCoreProps
		}
	}
	return 0, ""
}

// Cores is the device's compute-unit count and the extension that stated it,
// or 0 and "" when no extension does.
func (c *Ctx) Cores() (int, string) { return c.cores, c.coresFrom }
