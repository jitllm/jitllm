package tier

import (
	"fmt"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/backend/vulkan"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit/ir"
)

// realDevices opens every device on this host -- CUDA, Vulkan and Metal --
// one at a time, named.
//
// backend.Open() is the wrong list here: it returns one Vulkan device, the
// discrete-first pick, which hides an integrated GPU. The selection rule must
// hold on every device that can run a block.
func realDevices(t *testing.T) []backend.Device {
	t.Helper()
	var out []backend.Device
	if n, err := backend.CUDACount(); err == nil && n > 0 {
		for i := 0; i < n; i++ {
			if !testdevices.Has(t, "cuda", i) {
				continue
			}
			if d, err := backend.OpenCUDA(i); err == nil {
				out = append(out, d)
			}
		}
	}
	infos, err := backend.VulkanDevices()
	if err != nil {
		t.Logf("no Vulkan here: %v", err)
	}
	for _, in := range infos {
		if !in.Compute || !testdevices.Has(t, "vulkan", in.Index) {
			continue
		}
		d, err := backend.OpenVulkanWith(strconv.Itoa(in.Index), unpinned)
		if err != nil {
			t.Logf("vulkan:%d would not open: %v", in.Index, err)
			continue
		}
		out = append(out, d)
	}
	// Metal is a real device too: without it every gate below skipped on
	// Apple Silicon as "no GPU on this host".
	if d, err := backend.OpenMetalWith(backend.Opts{Enumerating: true}); err == nil {
		out = append(out, d)
	}
	return out
}

// unpinned opens a Vulkan device with the required-subgroup-size path off.
var unpinned = backend.Opts{Vulkan: vulkan.Config{Subgroup: "off"}}

// qkPlan is a block with the per-head q/k norm, the second kernel in the
// subgroup-width class.
func qkPlan() *nn.LayerPlan {
	p := fakePlan()
	p.QKNorm = true
	return p
}

func qkWeights(p *nn.LayerPlan) *nn.LayerWeights {
	w := blockWeights(p)
	w.QNorm = make([]float32, p.HeadDim)
	w.KNorm = make([]float32, p.HeadDim)
	for i := range w.QNorm {
		w.QNorm[i], w.KNorm[i] = 1, 1
	}
	return w
}

// TestSubgroupSelectionIsByGuarantee: on every device, which kernel was chosen
// is checked against what the device promises. It asserts the choice, not the
// answer: a device can compute a correct softmax from an undefined-behaviour
// kernel by luck. Stats.Lanes (which the paged softmaxes take) and the head norm's width
// must all follow from the promise.
func TestSubgroupSelectionIsByGuarantee(t *testing.T) {
	devs := realDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU on this host")
	}
	for _, d := range devs {
		d := d
		t.Run(fmt.Sprintf("%s/%s", d.API(), d.Name()), func(t *testing.T) {
			defer d.Close()
			wide, why := backend.GuaranteedLanes(d, ir.SubgroupLanes)
			want := 1
			if wide {
				want = ir.SubgroupLanes
			}
			t.Logf("promise: %d lanes (%s)", want, why)

			g, err := New([]Slot{{Dev: d, Bytes: 1 << 40}},
				WithDeviceTune(TuneOff), WithSubgroup("off"))
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			defer g.Close()
			one := g.devs[0]
			p := qkPlan()
			if !one.PrepLayer(0, p, qkWeights(p)) {
				t.Fatalf("PrepLayer declined: %s", one.LastErr)
			}

			if one.Lanes != want {
				t.Fatalf("device promises %d lanes and the tier settled on %d (%s)",
					want, one.Lanes, one.LanesWhy)
			}
			if one.bs.qkLanes != want {
				t.Fatalf("the per-head norm runs %d lanes, want %d", one.bs.qkLanes, want)
			}
			// The block must also run: the scalar arm's launch geometry
			// differs, and getting it wrong leaves heads uncomputed rather
			// than erroring.
			x := make([]float32, p.NEmbd)
			cs := make([]float32, p.NRot)
			if !one.Layers(0, 1, 0, 1, x, cs, nil, nil) {
				t.Fatalf("Layers: %s", one.LastErr)
			}
			t.Logf("chose %d-lane kernels, block ran: headnorm=%d", one.Lanes, one.bs.qkLanes)
		})
	}
}

// TestSelectionFollowsTheGuaranteeWhenPinningIsOff is the same gate with the
// required-subgroup-size path disabled. With pinning every device promises
// 32 and the test above cannot fail; without it NVIDIA (32/32 by construction)
// keeps the warp kernels and an Intel integrated GPU (8..32) drops to the scalar twins,
// with the model running on both.
func TestSelectionFollowsTheGuaranteeWhenPinningIsOff(t *testing.T) {
	// The subgroup answer is baked into a device at open, so it is opened
	// unpinned rather than configured afterwards.
	infos, err := backend.VulkanDevices()
	if err != nil {
		t.Skipf("no Vulkan here: %v", err)
	}
	seen := map[int]int{}
	for _, in := range infos {
		if !in.Compute || !testdevices.Has(t, "vulkan", in.Index) {
			continue
		}
		d, err := backend.OpenVulkanWith(strconv.Itoa(in.Index), unpinned)
		if err != nil {
			t.Logf("vulkan:%d would not open: %v", in.Index, err)
			continue
		}
		func() {
			defer d.Close()
			wide, why := backend.GuaranteedLanes(d, ir.SubgroupLanes)
			want := 1
			if wide {
				want = ir.SubgroupLanes
			}
			g, err := New([]Slot{{Dev: d, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			defer g.Close()
			one := g.devs[0]
			p := qkPlan()
			if !one.PrepLayer(0, p, qkWeights(p)) {
				t.Fatalf("%s: PrepLayer declined: %s", d.Name(), one.LastErr)
			}
			if one.Lanes != want || one.bs.qkLanes != want {
				t.Fatalf("%s: unpinned promise is %d lanes; tier=%d headnorm=%d (%s)",
					d.Name(), want, one.Lanes, one.bs.qkLanes, why)
			}
			x := make([]float32, p.NEmbd)
			cs := make([]float32, p.NRot)
			if !one.Layers(0, 1, 0, 1, x, cs, nil, nil) {
				t.Fatalf("%s: Layers: %s", d.Name(), one.LastErr)
			}
			seen[want]++
			t.Logf("%-45s unpinned -> %d lanes (%s)", d.Name(), want, why)
		}()
	}
	if len(seen) == 0 {
		t.Skip("no Vulkan device opened")
	}
	if seen[1] == 0 || seen[ir.SubgroupLanes] == 0 {
		t.Logf("NOTE: with pinning off this box produced %v; the gate needs one device of "+
			"each kind to distinguish a rule from a constant", seen)
	}
}
