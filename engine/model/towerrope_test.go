package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestRotaryTowerOnDeviceMatchesHost runs the Qwen-VL towers on each device
// against the host: Qwen2-VL's 2-D rotary ViT and Qwen2.5-VL's RMSNorm,
// gated-MLP, windowed one.
//
// A device that did not rotate a vision block ran Qwen2-VL's tower with NO
// rotary -- the plan carried none, so the tier copied k unrotated -- and
// nothing declined: `jitllm run -devices cuda:0 -image` encoded every picture
// with the positions dropped. The violations are that state, and for
// Qwen2.5-VL a device that ignores the windows.
//
// Every block must be accepted. The comparison after ONE device block is the
// tight one, as TestTowerOnDeviceMatchesHost's is: the tiers quantize
// activations and reduce differently and a tower amplifies that per block. The
// whole tower on the device is held to TestQwenTowerMatchesTransformers' bound.
func TestRotaryTowerOnDeviceMatchesHost(t *testing.T) {
	for _, name := range qvlFixtures {
		t.Run(name, func(t *testing.T) { rotaryTowerOnDevice(t, name) })
	}
}

func rotaryTowerOnDevice(t *testing.T, name string) {
	m, g := openQVL(t, name)
	defer m.Close()
	tw := m.Tower()
	if tw == nil || !tw.Cfg.Rope {
		t.Fatal("the fixture's tower is not a 2-D rotary tower")
	}
	c := tw.Cfg
	img := goldenImage(g.Image.W, g.Image.H, g.Image.RGB)
	// encode runs the tower on s and returns the residual after block 0 and
	// the output.
	encode := func(s *Instance) ([]float32, []float32) {
		var b0 []float32
		s.vis.afterBlock = func(li int, x []float32) {
			if li == 0 && b0 == nil { // block -1 is the embedding
				b0 = append([]float32(nil), x...)
			}
		}
		px, err := s.PreprocessImage(img)
		if err != nil {
			t.Fatal(err)
		}
		out, err := s.Encode(px)
		if err != nil {
			t.Fatal(err)
		}
		return b0, append([]float32(nil), out...)
	}
	host := tw.testState()
	want0, want := encode(host)
	host.Close()
	defer func(was int) { tw.opt.towerGPULayers = was }(tw.opt.towerGPULayers)

	ran := 0
	for _, arm := range towerArms() {
		spec := arm.spec
		// Kimi-VL's tower is the chunked attention's gate. The device's set is
		// sized to the picture, and the fixture's picture's planes fit any
		// budget a card would pick, so the gate takes the budget below one
		// query row's planes: every pass is the smallest chunk, on any card.
		cfg := func(*gpu.Config) {}
		if c.Kind == jlm.ProjKimiVL {
			cfg = func(cf *gpu.Config) { cf.VisionPlaneBudget = 1 }
		}
		open := func() *gpu.GPU {
			gpux, err := gpu.OpenWith(gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff), gpu.WithConfig(cfg))
			if err != nil || gpux == nil {
				return nil
			}
			return gpux
		}
		gpux := open()
		if gpux == nil {
			t.Logf("%s: not present", spec)
			continue
		}
		ran++
		t.Run(arm.String(), func(t *testing.T) {
			defer gpux.Close()
			// The whole tower on the device.
			tw.opt.towerGPULayers = -1
			all := tw.testState()
			all.SetDevice(gpux)
			if all.GPUBlocks() != c.NLayer {
				t.Fatalf("%s took %d of %d tower blocks (%s)", spec, all.GPUBlocks(), c.NLayer, gpux.Err())
			}
			var gotAll []float32
			arm.served(t, gpux, c.NLayer, func() { _, gotAll = encode(all) })
			all.Close()
			whole, _ := nmse32(want, gotAll)
			if math.IsNaN(whole) || whole > towerBound {
				t.Fatalf("%s: every tower block on the device, output NMSE %.3e against the host", spec, whole)
			}
			tw.opt.towerGPULayers = 1
			s := tw.testState()
			s.SetDevice(gpux)
			// The device computes the tower's table from the rows' cells
			// (the grid table): it takes no row of it from the caller.
			before := gpux.Stats()
			tables, _ := before.ServedPart(gpu.ComposedTable)
			var got0 []float32
			arm.served(t, gpux, 1, func() { got0, _ = encode(s) })
			s.Close()
			after := gpux.Stats()
			if n := after.RopeTableUploads - before.RopeTableUploads; n != 0 {
				t.Fatalf("the device took %d rotary table rows from the caller: the tower's grid table did not run on it", n)
			}
			if served, _ := after.ServedPart(gpu.ComposedTable); served == tables {
				t.Fatalf("the device served no rotary table for the tower's block: it did not rotate (%s)", gpux.Err())
			}
			clean, _ := nmse32(want0, got0)
			// The bound is the backends' own rounding against the host's, which
			// a sharp tower attention amplifies: one block of synth-qwen2vl
			// read 6e-8 on Vulkan, 2e-5 on CUDA and 5.7e-4 on Metal at a qkv
			// scale of 0.5 -- and 3.6e-4 on Metal with the rotary replaced by
			// the identity on both tiers, so it is not the rotation. No rotary
			// at all reads 1e-1 and up.
			if math.IsNaN(clean) || !(clean < 1e-3) {
				t.Fatalf("%s: after one device block, NMSE %.3e against the host", spec, clean)
			}

			// The violations, each on a tier of its own: one that already holds
			// the block keeps the plan it prepared it with.
			viol := func(what string, setup func(), undo func(), compare func(*Instance) float64) {
				gv := open()
				defer gv.Close()
				v := tw.testState()
				defer v.Close()
				// The composer forms the block from the graph the front end
				// states of the configuration placed (restateTower).
				setup()
				tw.model.restateTower()
				v.SetDevice(gv)
				undo()
				tw.model.restateTower()
				if v.GPUBlocks() == 0 {
					t.Fatalf("%s: the violation placed no block", what)
				}
				bad := compare(v)
				t.Logf("violation %q: NMSE %.3e", what, bad)
				if !(bad > 100*clean) || !(bad > 1e-3) {
					t.Errorf("%s: %s reads %.3e against a clean %.3e -- the gate cannot see it",
						spec, what, bad, clean)
				}
			}
			viol("no rotary in the block's plan", func() { tw.Cfg.Rope = false }, func() { tw.Cfg.Rope = true },
				func(v *Instance) float64 {
					b0, _ := encode(v)
					nm, _ := nmse32(want0, b0)
					return nm
				})
			if c.Kind == jlm.ProjKimiVL {
				// MoonViT pairs neighbours: a device that pairs a head's halves
				// rotates the wrong elements together.
				viol("NEOX pairing in the block's plan", func() { tw.cfg.RopeNeox = true },
					func() { tw.cfg.RopeNeox = false },
					func(v *Instance) float64 {
						b0, _ := encode(v)
						nm, _ := nmse32(want0, b0)
						return nm
					})
			}
			if c.WinPattern > 0 {
				// Windows as wide as the image: the device attends over every
				// patch where the host attends inside windows.
				was := c.WinSize
				viol("the windows ignored", func() { tw.opt.towerGPULayers = -1 }, func() {},
					func(v *Instance) float64 {
						tw.Cfg.WinSize = 1 << 20
						defer func() { tw.Cfg.WinSize = was }()
						_, out := encode(v)
						nm, _ := nmse32(want, out)
						return nm
					})
			}
			t.Logf("%s: all %d tower blocks, output NMSE %.3e against the host; after one device block %.3e",
				spec, c.NLayer, whole, clean)
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}

// TestTextHeadBesideItsTowerOnTheDevice: with the text model's blocks and
// some of its tower's on one device, and the tower's instance still holding
// them, a prompt runs its head on the device too. The head-only call names
// one past the text model's last block, which is the tower's first; read as
// the run's first block it made the device refuse the head as a non-causal
// run -- silently a host head on a dense model, and a refused prompt on a
// hybrid, whose recurrence cannot restart on the host.
func TestTextHeadBesideItsTowerOnTheDevice(t *testing.T) {
	for _, name := range []string{"synth-qwen2vl", "synth-qwen35vl"} {
		t.Run(name, func(t *testing.T) {
			m, g := openQVL(t, name)
			defer m.Close()
			ids := append(append([]int32(nil), g.Pre...), g.Post...)
			host := m.NewInstance(len(ids) + 1)
			want, err := host.Prefill(ids)
			if err != nil {
				t.Fatal(err)
			}
			want = append([]float32(nil), want...)
			host.Close()
			ran := 0
			for _, spec := range stepDevices() {
				gpux, err := gpu.OpenWith(gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
				if err != nil || gpux == nil {
					t.Logf("%s: not present", spec)
					continue
				}
				ran++
				st := m.NewInstance(len(ids) + 1)
				st.SetDevice(gpux)
				ts := m.Tower().testState()
				ts.SetDevice(gpux)
				if st.GPULayers() != m.Cfg.NLayer || ts.GPUBlocks() == 0 {
					t.Fatalf("%s placed %d of %d text and %d tower blocks: the gate needs both (%s)", spec,
						st.GPULayers(), m.Cfg.NLayer, ts.GPUBlocks(), gpux.Err())
				}
				got, err := st.Prefill(ids)
				if err != nil {
					t.Fatalf("%s: %v", spec, err)
				}
				if e := gpux.Err(); e != "" {
					t.Fatalf("%s: the device refused part of the prompt: %s", spec, e)
				}
				nm, _ := nmse32(want, got)
				t.Logf("%s: %d text and %d tower blocks placed, the prompt's logits %.3e from the host", spec,
					st.GPULayers(), ts.GPUBlocks(), nm)
				if !(nm < 1e-5) {
					t.Errorf("%s: the prompt's logits read %.3e against the host", spec, nm)
				}
				ts.Close()
				st.Close()
				gpux.Close()
			}
			if ran == 0 {
				t.Skip("no device on this host")
			}
		})
	}
}
