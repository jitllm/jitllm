package server

import (
	"runtime"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/sched"
)

// DeviceInfo is one enumerated compute resource, in Go terms.
type DeviceInfo struct {
	ID      string
	Backend string // "cpu", "cuda", "hip", "vulkan", "metal"
	Index   int
	Name    string

	Kind DeviceKind

	TotalMemory uint64
	FreeMemory  uint64

	// CountsTowardHostBudget is true when this device's heap is system RAM
	// (an integrated GPU, Apple Silicon): its bytes subtract from the host
	// budget rather than add to it.
	CountsTowardHostBudget bool

	// PhysicalID identifies the hardware behind this entry: the device UUID
	// Vulkan and CUDA both report for one card. Empty means unknown, and an
	// empty id matches nothing, not even another empty one. See
	// backend.Identity.
	PhysicalID string
	// SameDeviceAs is the ID of another entry that is the same hardware, or
	// empty. The entry stays listed but is left out of SpendableTotal.
	SameDeviceAs string

	Available bool
	Why       string
}

// DeviceKind mirrors the proto enum.
type DeviceKind int

// The device kinds: the host CPU, a discrete card with its own memory, and an
// integrated device whose memory is the host's.
const (
	KindUnspecified DeviceKind = iota
	KindHost
	KindDiscrete
	KindIntegrated
)

// HostInfo is what a scheduler may spend on the host.
type HostInfo struct {
	WeightBudget uint64
	Total        uint64
	Available    uint64
	DecodeCores  int
	PCores       int
	ECores       int
	SMTSiblings  int
}

// Devices returns the hardware. The probe is cached for correctness: running
// backend.Open() while a tier holds a CUDA context would give one driver two
// owners. refresh re-probes, but not while a model is loaded.
func (e *Engine) Devices(refresh bool) ([]DeviceInfo, error) {
	e.devMu.Lock()
	defer e.devMu.Unlock()
	if e.probed && !refresh {
		return e.snapshotDevices(), e.devErr
	}
	e.mu.RLock()
	loaded := len(e.models)
	e.mu.RUnlock()
	if loaded > 0 && e.probed {
		// Re-probing here would open a second CUDA context beside the tier's.
		return e.snapshotDevices(), e.devErr
	}
	probe := e.cfg.Probe
	if probe == nil {
		probe = func() ([]DeviceInfo, error) { return probeDevices(e.cfg.ROCm) }
	}
	e.devices, e.devErr = probe()
	// Dedupe whatever the probe returned, injected probes included.
	// markSameDevice is idempotent.
	markSameDevice(e.devices)
	e.probed = true
	return e.snapshotDevices(), e.devErr
}

// snapshotDevices copies the cached probe. Nothing is overlaid: availability
// stays the probe's answer, and live queue depth reaches the wire through
// pbDevice, which asks the gate.
func (e *Engine) snapshotDevices() []DeviceInfo {
	out := make([]DeviceInfo, len(e.devices))
	copy(out, e.devices)
	return out
}

func probeDevices(rocm string) ([]DeviceInfo, error) {
	out := []DeviceInfo{{
		ID:                     HostGateID,
		Backend:                "cpu",
		Name:                   hostName(),
		Kind:                   KindHost,
		TotalMemory:            hostTotal(),
		FreeMemory:             sched.MemBudget(),
		CountsTowardHostBudget: true,
		Available:              true,
	}}

	// Every Vulkan physical device is listed, not only the one backend.Open
	// picks (discrete-first), so a caller can see and name vulkan:1. The query
	// opens nothing.
	if infos, err := backend.VulkanDevices(); err == nil {
		for _, in := range infos {
			d := DeviceInfo{
				ID:        "vulkan:" + strconv.Itoa(in.Index),
				Backend:   "vulkan",
				Index:     in.Index,
				Name:      in.Name,
				Kind:      KindDiscrete,
				Available: in.Compute,
			}
			if in.Unified {
				d.Kind, d.CountsTowardHostBudget = KindIntegrated, true
			}
			d.PhysicalID = backend.UUIDIdentity(in.UUID[:], in.UUIDOK,
				"VkPhysicalDeviceIDProperties.deviceUUID").Key
			if !in.Compute {
				d.Why = "no compute queue: this device cannot run a kernel"
			} else if in.Software {
				d.Why = "software rasteriser: correct, and not a GPU -- it is not placed on unless named"
			}
			out = append(out, d)
		}
	}

	// Every HIP device is listed as hip:N, as every Vulkan device is above. No
	// ROCm is no entry, not an error: the card is then vulkan:N alone.
	hipCfg := backend.HIPConfig{Path: rocm}
	for i := 0; i < backend.HIPCount(hipCfg); i++ {
		d := DeviceInfo{ID: "hip:" + strconv.Itoa(i), Backend: "hip", Index: i, Kind: KindDiscrete}
		hd, err := backend.OpenHIPWith(i, backend.Opts{HIP: hipCfg, Enumerating: true})
		if err != nil {
			d.Why = err.Error()
			out = append(out, d)
			continue
		}
		d.Name, d.Available = hd.Name(), true
		d.PhysicalID = backend.IdentityOf(hd).Key
		if free, total, ok := deviceMem(hd); ok {
			d.FreeMemory, d.TotalMemory = free, total
		}
		if u, ok := hd.(backend.Unified); ok && u.UnifiedMemory() {
			d.Kind, d.CountsTowardHostBudget = KindIntegrated, true
		}
		hd.Close()
		out = append(out, d)
	}

	devs := backend.OpenWith(backend.Opts{HIP: hipCfg})
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	for _, d := range devs {
		api := apiName(d.API())
		if api == "hip" {
			continue // listed above, every ordinal
		}
		if api == "vulkan" {
			// Already listed above; fill in the memory figures for the one
			// Vulkan actually opened, matched by name.
			free, total, ok := deviceMem(d)
			for i := range out {
				if out[i].Backend == "vulkan" && out[i].Name == d.Name() && ok {
					out[i].FreeMemory, out[i].TotalMemory = free, total
				}
			}
			continue
		}
		info := DeviceInfo{
			ID:         api + ":0",
			Backend:    api,
			Name:       d.Name(),
			Kind:       KindDiscrete,
			Available:  true,
			PhysicalID: backend.IdentityOf(d).Key,
		}
		// CUDA's Name already carries the ordinal, and a CUDA device opened by
		// backend.Open is always ordinal 0.
		if free, total, ok := deviceMem(d); ok {
			info.FreeMemory, info.TotalMemory = free, total
		}
		if u, ok := d.(backend.Unified); ok && u.UnifiedMemory() {
			info.Kind, info.CountsTowardHostBudget = KindIntegrated, true
			info.Why = "its memory IS the host's, so its bytes come OFF the host budget rather than adding to it"
		}
		out = append(out, info)
	}
	markSameDevice(out)
	return out, nil
}

// markSameDevice links every entry that is the same PHYSICAL device as an
// earlier one, and returns how many it linked (so a gate can assert it ran).
//
// It marks rather than removes: a caller may still ask for either backend,
// but only the unmarked entry is counted. The unmarked one is the preferred
// backend by backend.APIRank (CUDA or Metal over Vulkan), not the first
// listed, since Vulkan is enumerated first. See
// docs/engineering-history/placement.md 15u.
func markSameDevice(devs []DeviceInfo) int {
	primary := map[string]int{} // identity -> index of the entry that counts
	n := 0
	for i := range devs {
		devs[i].SameDeviceAs = ""
		id := devs[i].PhysicalID
		if id == "" {
			// An unknown identity matches nothing, including another unknown
			// one. Under-merging places a card on two budgets; over-merging
			// deletes a card. Only the second is unrecoverable.
			continue
		}
		j, seen := primary[id]
		if !seen {
			primary[id] = i
			continue
		}
		n++
		if backend.APIRank(devs[i].Backend) > backend.APIRank(devs[j].Backend) {
			// This entry is the better backend for the card: it takes the
			// count and the earlier one becomes the duplicate. Entries that
			// pointed at the old primary are re-pointed so no duplicate
			// points at another duplicate.
			for k := range devs {
				if devs[k].SameDeviceAs == devs[j].ID {
					devs[k].SameDeviceAs = devs[i].ID
				}
			}
			devs[j].SameDeviceAs = devs[i].ID
			primary[id] = i
			continue
		}
		devs[i].SameDeviceAs = devs[j].ID
	}
	return n
}

func deviceMem(d backend.Device) (free, total uint64, ok bool) {
	f, t, err := d.Mem()
	if err != nil {
		return 0, 0, false
	}
	return f, t, true
}

// apiName maps the code generator's name to the name the -devices grammar
// uses, so one vocabulary reaches the API and the CLI.
func apiName(api string) string {
	switch strings.ToLower(api) {
	case "ptx":
		return "cuda"
	case "spirv":
		return "vulkan"
	case "msl":
		return "metal"
	case "amdgcn":
		return "hip"
	}
	return strings.ToLower(api)
}

// Host reports the host's own memory and cores.
func (e *Engine) Host() HostInfo {
	return HostInfo{
		WeightBudget: sched.MemBudget(),
		Total:        hostTotal(),
		Available:    hostAvailable(),
		DecodeCores:  len(sched.DecodeCores()),
		PCores:       len(sched.PCores()),
		ECores:       len(sched.ECores()),
		SMTSiblings:  len(sched.SMTSiblings()),
	}
}

// SpendableTotal is the host's weight budget plus every device whose memory is
// its own, each physical device counted once. An integrated GPU's heap is
// system RAM, and adding it to the host budget spends the same bytes twice;
// so would counting one card under two backends.
func SpendableTotal(host HostInfo, devs []DeviceInfo) uint64 {
	total := host.WeightBudget
	for _, d := range devs {
		if d.Kind == KindHost || d.CountsTowardHostBudget || !d.Available {
			continue
		}
		// markSameDevice decided which entry of a shared card counts.
		if d.SameDeviceAs != "" {
			continue
		}
		total += d.TotalMemory
	}
	return total
}

func hostName() string {
	return runtime.GOOS + "/" + runtime.GOARCH + " host, " + strconv.Itoa(runtime.NumCPU()) + " logical CPUs"
}
