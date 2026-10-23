package tier

import (
	"fmt"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/backend/vulkan"
)

// fakeHost stands three CUDA cards and a Vulkan list behind the backend
// constructors openEntries calls: the Vulkan list is the first two CUDA cards
// again (one physical device under two APIs), an integrated GPU, and a
// software rasteriser. It returns every device the spec opened, so a gate can
// check that nothing was opened twice or left open.
func fakeHost(t *testing.T) *[]*fakeDev {
	t.Helper()
	var opened []*fakeDev
	cards := []string{card0, card1, "00112233445566778899aabbccddeeff"}
	saved := []any{cudaCount, openCUDA, vulkanDevices, openVulkan, openMetal}
	t.Cleanup(func() {
		cudaCount = saved[0].(func() (int, error))
		openCUDA = saved[1].(func(int, backend.Opts) (backend.Device, error))
		vulkanDevices = saved[2].(func() ([]vulkan.Info, error))
		openVulkan = saved[3].(func(string, backend.Opts) (backend.Device, error))
		openMetal = saved[4].(func(backend.Opts) (backend.Device, error))
	})
	cudaCount = func() (int, error) { return len(cards), nil }
	openCUDA = func(ord int, _ backend.Opts) (backend.Device, error) {
		if ord < 0 || ord >= len(cards) {
			return nil, fmt.Errorf("fake: no CUDA device %d", ord)
		}
		d := &fakeDev{name: fmt.Sprintf("#%d %s (sm_70)", ord, model), api: "ptx", uuid: cards[ord],
			free: boxCardFree, total: boxCardTotal}
		opened = append(opened, d)
		return ordFake{d, ord}, nil
	}
	infos := []vulkan.Info{
		{Index: 0, Name: model, Compute: true},
		{Index: 1, Name: model, Compute: true},
		{Index: 2, Name: "Intel(R) Iris(R) Xe Graphics", Compute: true, Unified: true},
		{Index: 3, Name: "llvmpipe", Compute: true, Software: true},
	}
	vkUUID := []string{cards[0], cards[1], "aaaabbbbccccddddeeeeffff00001111", ""}
	vulkanDevices = func() ([]vulkan.Info, error) { return infos, nil }
	openVulkan = func(sel string, _ backend.Opts) (backend.Device, error) {
		i := 0 // "" picks the first, as the backend picks its best
		if sel != "" {
			n, err := strconv.Atoi(sel)
			if err != nil || n < 0 || n >= len(infos) {
				return nil, fmt.Errorf("fake: no Vulkan device %q", sel)
			}
			i = n
		}
		d := &fakeDev{name: infos[i].Name, api: "spirv", uuid: vkUUID[i], free: boxCardFree, total: boxCardTotal}
		opened = append(opened, d)
		return ordFake{d, i}, nil
	}
	openMetal = func(backend.Opts) (backend.Device, error) { return nil, fmt.Errorf("fake: no Metal") }
	return &opened
}

// TestABareBackendNameIsEveryDeviceOfIt: `cuda` is every CUDA card, `vulkan`
// every Vulkan GPU, and a bare `gpu` every GPU after one card under two APIs
// is merged -- while `cuda:1` and `vulkan:2` each pin exactly one. The count
// is what is asserted: with a bare name resolving to one device, `cuda` opens one card of three and fails here. (`gpu:N` indexes
// backend.Open's list, which this fake host does not stand behind.)
func TestABareBackendNameIsEveryDeviceOfIt(t *testing.T) {
	cases := []struct {
		spec  string
		want  int
		names []string // the placement names, or nil to skip the check
	}{
		{"cuda", 3, []string{"", "", ""}},
		{"cuda:1", 1, []string{"cuda:1"}},
		{"cuda=3G", 3, nil},
		{"vulkan", 3, nil}, // the software rasteriser is left out
		{"vulkan:2", 1, []string{"vulkan:2"}},
		{"gpu", 4, nil}, // three CUDA cards and the iGPU; the Vulkan twins merge
		{"all", 4, nil},
		{"cuda:0,vulkan:2", 2, []string{"cuda:0", "vulkan:2"}},
	}
	for _, tc := range cases {
		t.Run(tc.spec, func(t *testing.T) {
			opened := fakeHost(t)
			es, err := ParseDevices(tc.spec)
			if err != nil {
				t.Fatal(err)
			}
			o := openOpts{spec: tc.spec, kb: knobs{tune: TuneOff}}
			devs, asks, names, err := openEntries(es, o)
			if err != nil {
				t.Fatal(err)
			}
			if len(devs) != tc.want {
				t.Fatalf("-devices %s resolved to %d device(s), want %d: %v", tc.spec, len(devs), tc.want, labels(devs))
			}
			if tc.names != nil {
				for i, n := range tc.names {
					if names[i] != n {
						t.Fatalf("-devices %s: device %d is named %q, want %q (names %q)", tc.spec, i, names[i], n, names)
					}
				}
			}
			if tc.spec == "cuda=3G" {
				for i, a := range asks {
					if a != 3<<30 {
						t.Fatalf("cuda=3G gave device %d a budget of %d, want 3 GiB on every card", i, a)
					}
				}
			}
			if err := noDuplicates(devs, tc.spec); err != nil {
				t.Fatal(err)
			}
			// What the spec did not keep is closed, and what it kept is not.
			kept := map[*fakeDev]bool{}
			for _, d := range devs {
				kept[d.(ordFake).fakeDev] = true
			}
			for _, d := range *opened {
				if kept[d] != (d.closed == 0) {
					t.Fatalf("-devices %s: %s [%s] kept=%v closed %d time(s)", tc.spec, d.name, d.api, kept[d], d.closed)
				}
			}
		})
	}
}

// ordFake is a fake with the ordinal a real CUDA or Vulkan device reports,
// which is what tells two cards of one name apart (noDuplicates).
type ordFake struct {
	*fakeDev
	ord int
}

func (d ordFake) Ordinal() int { return d.ord }

func labels(devs []backend.Device) []string {
	out := make([]string, len(devs))
	for i, d := range devs {
		out[i] = d.Name() + " [" + d.API() + "]"
	}
	return out
}

// TestHardwareIsOneKeyPerPhysicalDevice: whatever spec opened a device, its
// Hardware key is the physical device's, so a caller keying on it (the
// server's device gates) sees `cuda` and `cuda:1` share card 1 and nothing
// else, and card 1 reached through Vulkan as the same card. A device with no
// identity falls back to its backend and ordinal, which still tells it apart.
func TestHardwareIsOneKeyPerPhysicalDevice(t *testing.T) {
	keys := func(spec string) []Hardware {
		t.Helper()
		fakeHost(t)
		g, err := OpenWith(WithDevices(spec), WithHostBudget(boxHost), WithDeviceTune(TuneOff))
		if err != nil {
			t.Fatalf("-devices %s: %v", spec, err)
		}
		defer g.Close()
		return g.Hardware()
	}
	every := keys("cuda")
	one := keys("cuda:1")
	vk := keys("vulkan:1")
	igpu := keys("vulkan:2")
	soft := keys("vulkan:3")
	t.Logf("cuda %v | cuda:1 %v | vulkan:1 %v | vulkan:2 %v | vulkan:3 %v", every, one, vk, igpu, soft)

	if len(every) != 3 || len(one) != 1 {
		t.Fatalf("cuda opened %d device(s) and cuda:1 %d; want 3 and 1", len(every), len(one))
	}
	seen := map[string]bool{}
	for _, h := range every {
		if seen[h.Key] {
			t.Fatalf("two of cuda's three cards share key %q: %v", h.Key, every)
		}
		seen[h.Key] = true
	}
	if !seen[one[0].Key] {
		t.Fatalf("cuda:1's key %q is none of cuda's %v: two specs reaching card 1 would not "+
			"share its gate", one[0].Key, every)
	}
	if one[0].Key != "uuid:"+card1 || one[0].Ref != "cuda:1" || one[0].Name != "cuda:1" {
		t.Fatalf("cuda:1 is %+v, want key uuid:%s, ref cuda:1, name cuda:1", one[0], card1)
	}
	if vk[0].Key != one[0].Key || vk[0].Ref != "vulkan:1" {
		t.Fatalf("card 1 under Vulkan is %+v, and under CUDA %+v: one card is one key", vk[0], one[0])
	}
	if seen[igpu[0].Key] {
		t.Fatalf("the integrated GPU's key %q is one of the CUDA cards'", igpu[0].Key)
	}
	if soft[0].Key != "vulkan:3" {
		t.Fatalf("a device with no identity has key %q, want its backend and ordinal vulkan:3", soft[0].Key)
	}
}
