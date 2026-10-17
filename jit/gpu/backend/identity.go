package backend

import (
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/jit/gpu/cuda"
)

// Device identity: is this enumerated device the same piece of hardware as that
// one? One card reached through two backends (cuda:0 and vulkan:0) must not be
// counted as two cards' worth of memory.
//
// The key is the device UUID, which both specifications promise:
// VkPhysicalDeviceIDProperties.deviceUUID and CUDA's cuDeviceGetUuid are the
// same sixteen bytes for the same physical device. Not the name: two identical
// cards share one, and merging them would delete hardware.
//
// A backend that cannot say leaves its devices distinct: an unknown identity
// never matches anything, including another unknown one. See Same.

// Identity is a stable name for the physical device behind a Device, and where
// that name came from.
type Identity struct {
	// Key is the canonical identity. Empty means this backend could not say,
	// and an empty Key matches nothing -- not even another empty Key.
	Key string
	// Source is the API the Key was read from, or the reason there is none. It
	// is reported rather than inferred because "the backends disagree" and "one
	// backend was never asked" are different facts about a machine.
	Source string
}

// Known reports whether this identity can be compared at all.
func (i Identity) Known() bool { return i.Key != "" }

// Same reports that two enumerated devices are one piece of hardware.
//
// An unknown identity is not equal to an unknown identity, or every
// unidentifiable device on a host would collapse into one.
func (i Identity) Same(o Identity) bool { return i.Known() && i.Key == o.Key }

func (i Identity) String() string {
	if !i.Known() {
		return "unknown (" + i.Source + ")"
	}
	return i.Key + " (" + i.Source + ")"
}

// Identified is the optional interface a backend implements when it can name
// the hardware behind a Device. A backend that does not implement it reports an
// unknown identity, which leaves its devices distinct.
type Identified interface{ Identity() Identity }

// IdentityOf asks a device who it is. A device that will not say is distinct
// from every other device, which is the safe direction: see Same.
func IdentityOf(d Device) Identity {
	if i, ok := d.(Identified); ok {
		return i.Identity()
	}
	return Identity{Source: fmt.Sprintf("%s reports no device identity", d.API())}
}

// UUIDIdentity builds an identity from a device UUID, and is the one place the
// two rules about a UUID live.
//
// An all-zero UUID is not an identity: a driver that returns success and writes
// nothing hands back zeros, which would compare equal between two unrelated
// devices. It is refused here, once, so no backend has to remember.
func UUIDIdentity(uuid []byte, ok bool, source string) Identity {
	if !ok {
		return Identity{Source: source}
	}
	zero := true
	for _, b := range uuid {
		if b != 0 {
			zero = false
			break
		}
	}
	if zero {
		return Identity{Source: source + " returned an all-zero UUID, which is not an identity"}
	}
	return Identity{Key: "uuid:" + hex.EncodeToString(uuid), Source: source}
}

// APIRank says which backend to prefer when one physical device is reachable
// through several. Higher wins.
//
// CUDA over Vulkan is measured on the same card. Metal outranks Vulkan because
// on Apple hardware Vulkan is MoltenVK on top of Metal. This orders backends for one card only; different cards are
// measured by choose().
func APIRank(api string) int {
	switch strings.ToLower(api) {
	case "ptx", "cuda":
		return 4
	case "msl", "metal":
		return 3
	case "amdgcn", "hip":
		// AMD's own compute stack over Vulkan on the same card: a decision
		// until a same-card A/B decides it (docs/design/amd-hip.md).
		return 2
	case "spirv", "vulkan":
		return 1
	}
	return 0
}

// Identity for the three backends. CUDA and Vulkan both answer with the device
// UUID their specifications agree on.
func (c *cudaDev) Identity() Identity {
	u, ok := c.d.UUID()
	return UUIDIdentity(u[:], ok, cuda.UUIDSource())
}

func (v *vkDev) Identity() Identity {
	u, ok := v.c.UUID()
	return UUIDIdentity(u[:], ok,
		"VkPhysicalDeviceIDProperties.deviceUUID")
}

// Metal reports an unknown identity: nothing specifies that MoltenVK's
// deviceUUID derives from MTLDevice's registryID, so a match would be a guess.
// Leaving it distinct double-counts nothing, because Apple hardware is unified
// memory and both entries come off the host budget.
func (m *mtlDev) Identity() Identity {
	return Identity{Source: "Metal reports no UUID that MoltenVK's deviceUUID is specified to match"}
}

var (
	_ Identified = (*cudaDev)(nil)
	_ Identified = (*vkDev)(nil)
	_ Identified = (*mtlDev)(nil)
)

// Driven is the optional interface a backend implements when it can name the
// driver behind a Device. A measurement cached against a device is keyed by its
// identity AND its driver: a driver update recompiles every kernel.
type Driven interface{ DriverVersion() string }

// DriverOf names a device's driver, or "" when the backend cannot say.
func DriverOf(d Device) string {
	if v, ok := d.(Driven); ok {
		return v.DriverVersion()
	}
	return ""
}

func (c *cudaDev) DriverVersion() string {
	if v, ok := cuda.DriverVersion(); ok {
		return fmt.Sprintf("cuda%d", v)
	}
	return ""
}

func (v *vkDev) DriverVersion() string { return fmt.Sprintf("vk%08x", v.c.DriverVersion()) }

// Targeted is the optional interface a backend implements when its code
// generator compiles for a named instruction-set tier of the device (CUDA's
// PTX target, "sm_86"). A verdict that depends on the ISA (graph.Target)
// names it.
type Targeted interface{ ISA() string }

// ISAOf names a device's code-generation tier, or "" when the backend cannot
// say.
func ISAOf(d Device) string {
	if v, ok := d.(Targeted); ok {
		return v.ISA()
	}
	return ""
}

func (c *cudaDev) ISA() string { return c.d.Target }
