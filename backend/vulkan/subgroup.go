package vulkan

// What this device promises about subgroup width, and how a promise is obtained
// where the default is only a hope.
//
// Some kernels depend on 32 lanes sharing a subgroup (a butterfly ShuffleXor
// reduction, the warp matrix instruction). A device with a width range (an
// Intel integrated GPU: 8..32) chooses a width per shader, and those kernels then produce
// NaN non-deterministically. VkPhysicalDeviceSubgroupProperties.subgroupSize is
// only the default width, so it cannot tell such a device apart.
//
// There are exactly two ways to have a width:
//
//   - min == max. The driver has no range to choose from.
//   - Pin it. VK_EXT_subgroup_size_control (core in 1.3) takes a required size
//     at pipeline creation, and the driver must honour it or fail to create the
//     pipeline.
//
// Everything else is a refusal, and a refusal is a scalar kernel rather than a
// broken one. The three cases, as measured:
//
//	NVIDIA discrete      size 32   min 32  max 32   control, compute  -> 32 pinned
//	Intel integrated     size 32   min  8  max 32   control, compute  -> 32 pinned
//	llvmpipe             size  8   min  8  max  8   control, compute  -> refused

import (
	"fmt"
	"runtime"
)

// Subgroups is what one physical device says about subgroup width.
//
// Size is reported for the report's sake and is deliberately not part of any
// decision: see the file header for the device that reports 32 and runs 16.
type Subgroups struct {
	Size   uint32 // the DEFAULT width, which is not a promise
	Stages uint32 // VkShaderStageFlags the subgroup ops are available in
	Ops    uint32 // VkSubgroupFeatureFlags: shuffles are one bit of this

	// Min and Max are the range the driver may choose from, from
	// VK_EXT_subgroup_size_control. They are 0 when Control is false, which is
	// the case where nothing about the width is known.
	Min, Max uint32
	// ReqStages are the stages that will accept a pinned width.
	ReqStages uint32
	// Control says the physical device advertises VK_EXT_subgroup_size_control,
	// which is what makes Min, Max and ReqStages meaningful. Enabled says it was
	// turned on for this logical device, with subgroupSizeControl and
	// computeFullSubgroups, which is what makes pinning legal. They are separate
	// fields because a refusal has to say which half is missing.
	Control bool
	Enabled bool
	// Off and Forced are Config.Subgroup, and both exist for gates and
	// measurement rather than for running models. See subgroupEnv.
	//
	//	off    never pin. The promise must come from Min == Max alone.
	//	force  never pin and promise anyway: a 32-lane kernel handed to a
	//	       device that chooses its own width. It is how the gate that must
	//	       fail is made to fail.
	Off    bool
	Forced bool
}

// subgroupEnv reads the measurement/gate override once, at device creation: a
// pipeline created pinned stays pinned, so a variable consulted later would
// describe a device that is not the one running.
//
// See Config.Subgroup: "force" produces wrong answers on purpose, and exists
// because a guarantee never seen to be needed is not a guarantee.
func subgroupEnv(c Config) (off, forced bool) {
	switch c.Subgroup {
	case "off":
		return true, false
	case "force":
		return false, true
	}
	return false, false
}

// CanPin reports that this context may pin a compute pipeline's width.
func (s Subgroups) CanPin() bool {
	return s.Control && s.Enabled && s.ReqStages&stageCompute != 0
}

// Guarantees reports whether a compute pipeline on this device can be promised
// exactly w lanes per subgroup, and says why not when it cannot.
//
// It is the whole selection rule, and deliberately not a probe: an
// out-of-subgroup shuffle is undefined, so running the kernel can pass on an
// idle device and fail on a busy one. A kernel that needs w lanes is created
// with w pinned or not at all.
func (s Subgroups) Guarantees(w int) (bool, string) {
	if w <= 0 {
		return true, "" // the kernel's answer does not depend on the width
	}
	if s.Forced {
		return true, "subgroup \"force\": the width is ASSUMED, not promised"
	}
	if s.Ops&subgroupShuffle == 0 || s.Ops&subgroupBasic == 0 {
		return false, fmt.Sprintf("the device does not support subgroup shuffles "+
			"(supportedOperations %#x)", s.Ops)
	}
	if s.Stages&stageCompute == 0 {
		return false, "the device offers no subgroup operations in the compute stage"
	}
	if s.CanPin() && uint32(w) >= s.Min && uint32(w) <= s.Max {
		return true, ""
	}
	switch {
	case !s.Control:
		return false, fmt.Sprintf("no %s, so the width is whatever the driver picks "+
			"(subgroupSize reports %d, which is a default and not a bound)",
			extSubgroupSizeControl, s.Size)
	case s.Min == s.Max && s.Min == uint32(w):
		// Pinning is unavailable or switched off, and unnecessary: there is
		// nothing for the driver to choose.
		return true, ""
	case uint32(w) < s.Min || uint32(w) > s.Max:
		return false, fmt.Sprintf("the device runs subgroups of %d..%d lanes and cannot be "+
			"asked for %d", s.Min, s.Max, w)
	case s.Off:
		return false, fmt.Sprintf("subgroup \"off\" and the device picks its width per "+
			"shader from %d..%d, so %d lanes cannot be promised", s.Min, s.Max, w)
	case !s.Enabled:
		return false, fmt.Sprintf("%s is supported and was not enabled on this device, so its "+
			"%d..%d range cannot be pinned", extSubgroupSizeControl, s.Min, s.Max)
	default:
		return false, fmt.Sprintf("the device picks its subgroup width per shader from %d..%d "+
			"and will not pin the compute stage (requiredSubgroupSizeStages %#x)",
			s.Min, s.Max, s.ReqStages)
	}
}

// String is the one-line report: what the device says, and what it will promise.
func (s Subgroups) String() string {
	pin := "no pinning"
	switch {
	case s.Forced:
		pin = "WIDTH ASSUMED (subgroup \"force\")"
	case s.Off:
		pin = "pinning off (subgroup \"off\")"
	case s.CanPin():
		pin = "pins the compute stage"
	}
	return fmt.Sprintf("subgroupSize %d (default), range %d..%d, ops %#x, %s",
		s.Size, s.Min, s.Max, s.Ops, pin)
}

// Subgroups returns what this open device promises. See Guarantees.
func (c *Ctx) Subgroups() Subgroups { return c.sg }

// readSubgroups queries the physical device. control says the extension is
// supported, which is what makes the size-control struct legal to chain; it does
// not say it was enabled, which is Open's business.
//
// A device below Vulkan 1.1 cannot be asked (the query is core-1.1), so it gets
// Subgroups{}, and Guarantees refuses every width.
func readSubgroups(p PhysDevice, apiVer uint32, control bool) Subgroups {
	if apiVer < apiVersion11 {
		return Subgroups{}
	}
	sg := subgroupProps{sType: stSubgroupProps}
	sc := sizeCtlProps{sType: stSizeCtlProps}
	if control {
		sg.pNext = uintptr(up(&sc))
	}
	p2 := physProps2{sType: stPhysDevProps2, pNext: uintptr(up(&sg))}
	vkGetPhysicalDeviceProperties2(p, up(&p2))

	out := Subgroups{Size: sg.size, Stages: sg.stages, Ops: sg.ops, Control: control}
	if control {
		out.Min, out.Max, out.ReqStages = sc.min, sc.max, sc.reqStages
	}
	// sg and sc are reachable only through uintptr pNext fields, which are not
	// references; Open does the same for appInfo.
	runtime.KeepAlive(sg)
	runtime.KeepAlive(sc)
	return out
}
