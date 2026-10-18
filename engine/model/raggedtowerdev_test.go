package model

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// raggedBound is TestRaggedTowerOnDeviceMatchesHost's NMSE bound.
const raggedBound = 1e-4

// TestRaggedTowerOnDeviceMatchesHost runs the openjev test model's tower,
// 144 wide and packed at 160, with every block on each device present and
// with one, against the host's encoding of the same picture. The bound is
// tighter than towerBound: the clean arms read 1e-8 to 1e-7 on the laptop's
// CUDA and Vulkan, and the violation -- plans that state no padded width, so
// the device quantizes its rows at 144 under weights packed at 160 -- reads
// 2e-3, inside towerBound.
func TestRaggedTowerOnDeviceMatchesHost(t *testing.T) {
	text, proj := testmodels.Path("openjev/tinyopenjev-for-testing-Q8_0.gguf"), testmodels.Path("openjev/mmproj-tinyopenjev-for-testing-Q8_0.gguf")
	if !vlmAvailable(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (RULE 11)", text)
	}
	m, err := Open(jlmOfPair(t, text, proj), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	tw := m.Tower()
	body, err := os.ReadFile(filepath.Join("testdata", "decision", "chatpicture.json"))
	if err != nil {
		t.Fatal(err)
	}
	req, err := ParseDecisionRequest(body)
	if err != nil || len(req.Images) == 0 {
		t.Fatalf("the request carries no picture (%v)", err)
	}
	img, err := DecodeImageDataURL(req.Images[0])
	if err != nil {
		t.Fatal(err)
	}
	encode := func(s *State) []float32 {
		px, err := s.PreprocessImage(img)
		if err != nil {
			t.Fatal(err)
		}
		out, err := s.Encode(px)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), out...)
	}
	host := tw.testState()
	want := encode(host)
	host.Close()
	defer func(was int) { tw.opt.towerGPULayers = was }(tw.opt.towerGPULayers)
	ran := 0
	for _, spec := range stepDevices() {
		gpu, err := openTier(t, tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
		if err != nil || gpu == nil {
			t.Logf("%s: not present", spec)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer gpu.Close()
			for _, n := range []int{-1, 1} {
				tw.opt.towerGPULayers = n
				s := tw.testState()
				s.SetDevice(gpu)
				placed := s.GPUBlocks()
				if placed == 0 || n == -1 && placed != tw.Cfg.NLayer {
					s.Close()
					t.Fatalf("%s took %d of %d tower blocks (%s)", spec, placed, tw.Cfg.NLayer, gpu.Err())
				}
				got := encode(s)
				s.Close()
				nm, _ := nmse32(want, got)
				t.Logf("%d of %d blocks on %s: NMSE %.3e against the host", placed, tw.Cfg.NLayer, spec, nm)
				if math.IsNaN(nm) || nm > raggedBound {
					t.Errorf("%d blocks on %s: NMSE %.3e against the host", placed, spec, nm)
				}
			}
			tw.opt.towerGPULayers = -1
			m.kpadViolation = true
			defer func() { m.kpadViolation = false }()
			gv, err := openTier(t, tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
			if err != nil || gv == nil {
				t.Fatalf("%s: a second tier: %v", spec, err)
			}
			defer gv.Close()
			s := tw.testState()
			s.SetDevice(gv)
			if s.GPUBlocks() == 0 {
				s.Close()
				t.Fatalf("the violation placed no block on %s", spec)
			}
			bad, _ := nmse32(want, encode(s))
			s.Close()
			t.Logf("no padded width in the plans: NMSE %.3e", bad)
			if !(bad > raggedBound) {
				t.Errorf("%s: plans with no padded width read %.3e: the gate does not see the padding", spec, bad)
			}
		})
	}
	if ran == 0 {
		t.Skip("no device present")
	}
}
