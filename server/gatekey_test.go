package server

import (
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
)

// serialises reports whether a session of b is recorded on a device gate a
// session of a is on: whether the two would share a card. The host gate,
// which every model with a partial seam is on, is left out.
func serialises(t *testing.T, e *Engine, a, b *LoadedModel) bool {
	t.Helper()
	ga := e.gatesFor(deviceGates(a))
	ga.acquire("a")
	defer ga.release()
	gb := e.gatesFor(deviceGates(b))
	defer gb.release()
	return gb.acquire("b") > 0
}

func deviceGates(lm *LoadedModel) []string {
	var out []string
	for _, id := range lm.gateIDs() {
		if id != HostGateID {
			out = append(out, id)
		}
	}
	return out
}

// TestOverlappingSpecsShareTheCardTheyShare: the gates are keyed on the
// physical device, so a model loaded with `cuda` (every card) and one loaded
// with `cuda:1` serialise on card 1 -- the one scratch set they would both
// use -- while models on disjoint cards run at once. The keys are the ones
// tier.Hardware gives a three-card host (TestHardwareIsOneKeyPerPhysicalDevice).
//
// VIOLATION SIGNATURE. Key gateIDs on the spec text (lm.deviceIDs) and this
// fails with "cuda and cuda:1 ran on card 1 at once".
func TestOverlappingSpecsShareTheCardTheyShare(t *testing.T) {
	e := New(Config{ModelDir: t.TempDir()})
	const a, b, c = "uuid:card-a", "uuid:card-b", "uuid:card-c"
	every := &LoadedModel{deviceIDs: []string{"cuda"}, gateKeys: []string{a, b, c}}
	one := &LoadedModel{deviceIDs: []string{"cuda:1"}, gateKeys: []string{b}}
	zero := &LoadedModel{deviceIDs: []string{"cuda:0"}, gateKeys: []string{a}}
	two := &LoadedModel{deviceIDs: []string{"cuda:2"}, gateKeys: []string{c}}

	if !serialises(t, e, every, one) {
		t.Fatal("cuda and cuda:1 ran on card 1 at once: two sessions on one scratch set")
	}
	if !serialises(t, e, one, every) {
		t.Fatal("cuda:1 then cuda ran on card 1 at once")
	}
	if serialises(t, e, zero, two) {
		t.Fatal("cuda:0 and cuda:2 waited for each other: they share no card")
	}
	if serialises(t, e, one, two) {
		t.Fatal("cuda:1 and cuda:2 waited for each other: they share no card")
	}
	// A model on every card and its own twin in the reverse order cannot
	// deadlock: the gates are taken in one sorted order.
	if !serialises(t, e, every, &LoadedModel{gateKeys: []string{c, b, a}}) {
		t.Fatal("two models on the same three cards ran at once")
	}
}

// TestRealOverlappingSpecsShareAGate is the same gate through LoadModel and a
// real tier: `cuda` and `cuda:0` reach this host's first CUDA card and share
// its gate, the same card under Vulkan shares it too (one physical device
// under two backends is one device), and an integrated GPU is a gate of its
// own. With the gates keyed on the spec text, cuda and cuda:0 run at once.
func TestRealOverlappingSpecsShareAGate(t *testing.T) {
	path := modelPath(t, deviceModel)
	if n, err := backend.CUDACount(); err != nil || n == 0 {
		t.Skipf("NO CUDA DEVICE (%d, %v): this gate proved nothing", n, err)
	}
	e := New(Config{Probe: oneCardProbe, Version: "test", MaxBatchRows: 1})
	t.Cleanup(e.Close)
	load := func(id, spec string) *LoadedModel {
		t.Helper()
		lm, err := e.LoadModel(LoadOptions{Path: path, ModelID: id, DeviceIDs: []string{spec}})
		if err != nil {
			t.Fatalf("load onto -devices %s: %v", spec, err)
		}
		t.Logf("-devices %-9s gates %v", spec, lm.gateIDs())
		return lm
	}
	cuda := load("every", "cuda")
	cuda0 := load("first", "cuda:0")
	if !serialises(t, e, cuda, cuda0) {
		t.Fatalf("cuda (%v) and cuda:0 (%v) ran on the first card at once", cuda.gateIDs(), cuda0.gateIDs())
	}
	if k := e.gateKey("cuda:0"); k != cuda0.gateKeys[0] {
		t.Fatalf("GetDeviceQueue(cuda:0) reads gate %q, and the model on cuda:0 queues on %q", k, cuda0.gateKeys[0])
	}

	infos, err := backend.VulkanDevices()
	if err != nil {
		t.Logf("no Vulkan (%v): the cross-backend and integrated arms did not run", err)
		return
	}
	twin, igpu := -1, -1
	for _, in := range infos {
		id := backend.UUIDIdentity(in.UUID[:], in.UUIDOK, "VkPhysicalDeviceIDProperties.deviceUUID")
		switch {
		case id.Known() && id.Key == cuda0.gateKeys[0]:
			twin = in.Index
		case in.Unified && in.Compute && !in.Software:
			igpu = in.Index
		}
	}
	if twin >= 0 {
		vk := load("twin", "vulkan:"+strconv.Itoa(twin))
		if !serialises(t, e, cuda0, vk) {
			t.Fatalf("cuda:0 and vulkan:%d are one card (%v, %v) and ran at once", twin,
				cuda0.gateIDs(), vk.gateIDs())
		}
	} else {
		t.Logf("the first CUDA card is not listed under Vulkan with its UUID: the cross-backend arm did not run")
	}
	if igpu >= 0 {
		ig := load("igpu", "vulkan:"+strconv.Itoa(igpu))
		if serialises(t, e, cuda0, ig) {
			t.Fatalf("cuda:0 and the integrated vulkan:%d share no device (%v, %v) and waited for each other",
				igpu, cuda0.gateIDs(), ig.gateIDs())
		}
	} else {
		t.Logf("no integrated Vulkan GPU: the disjoint arm did not run")
	}
}
