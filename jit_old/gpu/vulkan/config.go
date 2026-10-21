package vulkan

// The package reads no environment and keeps no configuration: a Config is an
// argument to OpenDeviceWith, so each device carries its own. tier forwards it
// from its option set; cmd/jitllm fills that from JITLLM_VK_DEVICE,
// JITLLM_VK_SUBGROUP and JITLLM_VK_MAXGROUPS.

// Config is what a caller may say about device selection, the subgroup
// guarantee and the workgroups of one dispatch. Zero is "enumerate and decide".
type Config struct {
	// Device pins one device, by enumeration index or name substring. Empty
	// picks the best of what enumerate reports.
	Device string
	// Subgroup is "" (probe the device), "off" (assume no reliable subgroup
	// width, use the scalar reduction) or "force" (assume the driver's answer
	// even where the probe disagrees).
	//
	// "force" produces wrong answers on purpose, so a gate can show the
	// guarantee is needed (scripts/three-way.sh --violate subgroup-width).
	// Never set it outside a gate.
	Subgroup string

	// MaxGroups lowers the workgroups one dispatch may carry below the
	// device's maxComputeWorkGroupCount[0]; 0, or anything above the device's
	// own, is the device's own. A launch past the limit is split into several
	// dispatches either way (see Ctx.dispatch); this exists so that path runs
	// on a device whose limit no real launch reaches. It never changes a
	// result, only how many dispatches carry it.
	MaxGroups uint32
}
