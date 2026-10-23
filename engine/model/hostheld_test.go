package model

import (
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// unifiedFake is a device on the host's memory that takes every block offered
// and holds perBlock bytes of the host's memory for each one.
type unifiedFake struct {
	hostHungryDevice
	perBlock uint64
	held     map[int]bool
}

func (d *unifiedFake) HostReserved() uint64 { return uint64(len(d.held)) * d.perBlock }

func (d *unifiedFake) PrepLayer(li int, _ *nn.LayerPlan, _ *nn.LayerWeights) bool {
	d.held[li] = true
	return true
}

// MigrateKV: no history is written in this gate, so moving it is success.
func (d *unifiedFake) MigrateKV(int, []float32, []float32, int, bool) bool { return true }

func (d *unifiedFake) ReleaseLayers(lo, hi int) {
	for li := lo; li < hi; li++ {
		delete(d.held, li)
	}
}

// TestPageBudgetLosesOnlyWhatAUnifiedDeviceHolds: the model's page budget is
// the caller's less what a device on the host's memory HOLDS, and it follows
// every placement move -- attach with nothing placed, the seam up, the seam
// down, a second session on the same device, detach, close. The expert budget
// (HostBudget) follows the same number. With followHost disabled the seam
// moves leave the page budget where attaching put it, and this fails on the
// first one.
func TestPageBudgetLosesOnlyWhatAUnifiedDeviceHolds(t *testing.T) {
	m, err := Open(jlmOf(t, models["stories260K"]))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer m.Close()
	if m.PageSize() == 0 {
		t.Fatal("stories260K opened with no container: there is no page budget to read")
	}
	const budget = 1 << 30
	const per = 32 << 20
	m.SetPageBudget(budget)
	noDevice := m.PageBudget()
	if noDevice != budget {
		t.Fatalf("with no device the page budget is %d, want the %d it was given", noDevice, uint64(budget))
	}

	st := m.NewState(64)
	defer st.Close()
	st.SetMemBudget(budget)
	dev := &unifiedFake{perBlock: per, held: map[int]bool{}}
	check := func(when string, blocks int) {
		t.Helper()
		want := uint64(budget - blocks*per)
		if got := m.PageBudget(); got != want {
			t.Fatalf("%s (%d block(s) on the integrated device, %d bytes each): page budget %d, "+
				"want %d -- the host must lose what the device holds, and only that",
				when, blocks, uint64(per), got, want)
		}
		if got := st.HostBudget(); got != want {
			t.Fatalf("%s: expert budget %d, want %d", when, got, want)
		}
	}

	if err := st.SetDeviceLayers(dev, 0); err != nil {
		t.Fatal(err)
	}
	if len(dev.held) != 0 {
		t.Fatalf("SetDeviceLayers(dev, 0) placed %d block(s)", len(dev.held))
	}
	check("attached, nothing placed", 0)

	if n := st.SetGPULayers(3); n != 3 {
		t.Fatalf("SetGPULayers(3) placed %d", n)
	}
	check("seam up to 3", 3)
	st.SetGPULayers(1)
	check("seam down to 1", 1)

	// A second session on the same device: the device is subtracted once,
	// for what it holds, and it stays subtracted while either session has it.
	st2 := m.NewState(64)
	if err := st2.SetDeviceLayers(dev, 0); err != nil {
		t.Fatal(err)
	}
	check("a second session attached", 1)
	st2.Close()
	check("the second session closed", 1)

	if err := st.SetDeviceLayers(nil, -1); err != nil {
		t.Fatal(err)
	}
	if got := m.PageBudget(); got != noDevice {
		t.Fatalf("detached: page budget %d, want the no-device %d back", got, noDevice)
	}
	t.Logf("page budget %d with no device, %d with 3 blocks of %d on the integrated device, "+
		"%d again once detached", noDevice, uint64(budget-3*per), uint64(per), m.PageBudget())
}

// TestPageBudgetFollowsARealIntegratedGPU is the same gate on hardware: an
// Iris Xe (any unified, computing, non-software Vulkan device) under the real
// tier. It asserts what the tier reports against its own ledger, so it fails
// against a HostReserved that reports the pool's ceiling (a quarter of the
// host budget) instead of what the device holds.
func TestPageBudgetFollowsARealIntegratedGPU(t *testing.T) {
	infos, err := backend.VulkanDevices()
	if err != nil {
		t.Skipf("NO VULKAN (%v): there is no integrated device to reach, and this gate proved nothing", err)
	}
	sel := -1
	for _, in := range infos {
		if in.Unified && in.Compute && !in.Software {
			sel = in.Index
		}
	}
	if sel < 0 {
		t.Skipf("NO INTEGRATED GPU among %d Vulkan device(s): this gate proved nothing", len(infos))
	}
	m, err := Open(jlmOf(t, testmodels.Path("stories15M-q8_0.gguf")))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer m.Close()
	const budget = 8 << 30
	g, err := tier.OpenWith(tier.WithDevices("vulkan:"+strconv.Itoa(sel)),
		tier.WithHostBudget(budget), tier.WithDeviceTune(tier.TuneOff))
	if err != nil {
		t.Fatalf("open the integrated GPU: %v", err)
	}
	defer g.Close()
	m.SetPageBudget(budget)
	st := m.NewState(64)
	defer st.Close()

	// held is what the device's ledger says it holds of the host's memory.
	held := func() (used, ceiling uint64) {
		for _, b := range g.Budgets() {
			if b.Host {
				used, ceiling = used+b.Used, b.PoolLimit
			}
		}
		return used, ceiling
	}
	check := func(when string) uint64 {
		t.Helper()
		used, ceiling := held()
		r := g.HostReserved()
		if r != used {
			t.Fatalf("%s (%d block(s) placed): HostReserved %d, the device holds %d of a %d "+
				"ceiling -- the host is charged for something other than what the device holds",
				when, st.GPULayers(), r, used, ceiling)
		}
		if got := m.PageBudget(); got != budget-used {
			t.Fatalf("%s: page budget %d, want %d less the %d the device holds", when, got,
				uint64(budget), used)
		}
		t.Logf("%-28s %d block(s): the device holds %7.2f MiB of a %.2f GiB ceiling, page budget %.4f GiB",
			when, st.GPULayers(), float64(used)/(1<<20), float64(ceiling)/(1<<30),
			float64(m.PageBudget())/(1<<30))
		return used
	}

	if err := st.SetDeviceLayers(g, 0); err != nil {
		t.Fatal(err)
	}
	zero := check("attached, nothing placed")
	if n := st.SetGPULayers(m.Cfg.NLayer); n == 0 {
		t.Fatalf("the integrated GPU took no block of %d: %s", m.Cfg.NLayer, g.Err())
	}
	full := check("seam at every block")
	if full <= zero {
		t.Fatalf("placing %d block(s) moved what the device holds from %d to %d: the charge "+
			"does not follow placement", st.GPULayers(), zero, full)
	}
	st.SetGPULayers(0)
	if back := check("seam back to 0"); back >= full {
		t.Fatalf("bringing every block home left the device holding %d of %d", back, full)
	}
}
