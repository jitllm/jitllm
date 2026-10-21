package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// kimiRealBound is how far the real tower may sit from Moonshot's run at the
// engine's arithmetic: 27 blocks of a 1152-wide tower against one simulation
// of Q8_0 matrices and int8 activations, whose rounding is not the engine's
// bit for bit.
const kimiRealBound = 5e-3

// kimiFixtures are the Kimi-VL fixtures scripts/kimivlgold.py writes:
// Moonshot's own KimiVLForConditionalGeneration (transformers ships no class
// for it), converted by llama.cpp's converter. Their goldens have the Qwen-VL
// family's shape, so they run its tower, processor, end-to-end and principle
// gates -- not its M-RoPE ones: Kimi-VL's text model is DeepSeek-V3's and
// turns an image's rows at their sequence positions.
var kimiFixtures = []string{"synth-kimivl"}

// towerFixtures is every fixture the Qwen-VL family's tower and end-to-end
// gates run.
func towerFixtures() []string { return slices.Concat(qvlFixtures, kimiFixtures, hyFixtures) }

// kimiTowerGate is TestQwenTowerMatchesTransformers' half for MoonViT: each of
// its own pieces taken out must move the output far past the bound -- the 2-D
// rotary, its neighbour pairing, which axis turns which pair, the table's
// bicubic resampling, the projector's LayerNorm over each patch, and the merge
// group's order.
func kimiTowerGate(t *testing.T, tw *Tower, g *qvlGolden, clean float64) {
	t.Helper()
	for _, v := range []struct {
		name  string
		mut   func(*State)
		fault towerFault
		neox  bool
	}{
		{"no 2-D rotary", func(s *State) {
			gh, gw := s.patchGrid()
			s.vis.ropeSC = make([]float32, gh*gw*tw.Cfg.HeadDim)
			for i := 0; i < len(s.vis.ropeSC); i += 2 {
				s.vis.ropeSC[i] = 1
			}
			s.vis.ropeAt = [2]int{gh, gw}
		}, towerFaultNone, false},
		{"the rotary NEOX-paired", nil, towerFaultNone, true},
		{"the row on the even pairs", nil, faultKimiAxes, false},
		{"the table resampled bilinear", nil, faultBilinearTable, false},
		{"the table read unresampled", nil, faultNoPosLerp, false},
		{"no LayerNorm over each patch", nil, faultKimiNoPreNorm, false},
		{"the merge group transposed", nil, towerFaultShuffleSwap, false},
	} {
		// The pairing is the segment's (tw.cfg), as the host and the device
		// read it.
		tw.Cfg.fault, tw.cfg.RopeNeox = v.fault, v.neox
		bad, _ := qvlEncode(t, tw, g, v.mut)
		tw.Cfg.fault, tw.cfg.RopeNeox = towerFaultNone, false
		nb, _ := nmse32(g.Image.EmbdA8, bad)
		t.Logf("violation %q: NMSE %.3e", v.name, nb)
		if !(nb > 10*towerBound) || math.IsNaN(nb) {
			t.Errorf("%s reads %.3e against a clean %.3e: the gate cannot see it", v.name, nb, clean)
		}
	}
}

// TestKimiVLTextMatchesTransformers holds the text model -- DeepSeek-V3's
// MLA and mixture -- to Moonshot's on Moonshot's own image rows, on the host
// and with every block and the head on each device. The violation is the
// image's rows in reverse: a single-axis rotary still turns each row by its
// place, so the order must show.
func TestKimiVLTextMatchesTransformers(t *testing.T) {
	for _, name := range kimiFixtures {
		t.Run(name, func(t *testing.T) {
			m, g := openQVL(t, name)
			defer m.Close()
			n := len(g.Pre) + g.grid().Rows() + len(g.Post) + len(g.Cont) + 1
			run := func(st *State, embd []float32) [][]float32 {
				defer st.Close()
				lg, err := st.PrefillMixed(Span{Tokens: g.Pre}, Span{Embd: embd, Grid: g.grid()}, Span{Tokens: g.Post})
				if err != nil {
					t.Fatal(err)
				}
				out := [][]float32{append([]float32(nil), lg...)}
				for _, id := range g.Cont[:len(g.Cont)-1] {
					if lg, err = st.Forward(id); err != nil {
						t.Fatal(err)
					}
					out = append(out, append([]float32(nil), lg...))
				}
				return out
			}
			host, flips := qvlWorst(g, run(m.NewState(n), g.Image.Embd))
			if !(host < c6NMSE) || flips > 0 {
				t.Fatalf("host: worst NMSE %.3e against Moonshot's model (bound %.0e), %d argmax flips", host, c6NMSE, flips)
			}
			E := m.Cfg.NEmbd
			rev := make([]float32, len(g.Image.Embd))
			for r := 0; r < len(rev)/E; r++ {
				copy(rev[r*E:], g.Image.Embd[len(rev)-(r+1)*E:len(rev)-r*E])
			}
			bad, _ := qvlWorst(g, run(m.NewState(n), rev))
			t.Logf("host %.3e; the image's rows reversed (the violation) %.3e", host, bad)
			if !(bad > 1e4*math.Max(host, 1e-12)) {
				t.Fatalf("the rows reversed read %.3e against %.3e: the gate cannot see their order", bad, host)
			}
			for _, spec := range stepDevices() {
				gpu, err := openTier(t, tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
				if err != nil || gpu == nil {
					t.Logf("%s: not present", spec)
					continue
				}
				st := m.NewState(n)
				st.SetDeviceLayers(gpu, -1)
				if got := st.GPULayers(); got != m.Cfg.NLayer {
					t.Fatalf("%s placed %d of %d blocks: %v", spec, got, m.Cfg.NLayer, gpu.Err())
				}
				w, flips := qvlWorst(g, run(st, g.Image.Embd))
				gpu.Close()
				t.Logf("every block and the head on %s: %.3e against Moonshot's model", spec, w)
				if !(w < 1e-5) || flips > 0 {
					t.Errorf("%s: worst NMSE %.3e, %d argmax flips", spec, w, flips)
				}
			}
		})
	}
}

// TestKimiVLChatSpansAreTheProcessors renders a picture and a question through
// the container's own template and holds the tokens around the image's rows to
// the prompt Moonshot's processor built for the golden.
func TestKimiVLChatSpansAreTheProcessors(t *testing.T) {
	m, g := openQVL(t, "synth-kimivl")
	defer m.Close()
	spans, err := m.ChatSpansImages([]ChatMessage{{Role: "user", Content: "What is in the picture?", Images: 1}},
		[]Image{{Embd: g.Image.Embd, Grid: g.grid()}}, true)
	if err != nil {
		t.Fatal(err)
	}
	var before, after []int32
	img := false
	for _, sp := range spans {
		switch {
		case len(sp.Embd) > 0:
			img = true
		case img:
			after = append(after, sp.Tokens...)
		default:
			before = append(before, sp.Tokens...)
		}
	}
	t.Logf("before the image %v, after %v", before, after)
	if !slices.Equal(before, g.Pre) || !slices.Equal(after, g.Post) {
		t.Fatalf("the template wraps the image in %v ... %v, Moonshot's processor in %v ... %v",
			before, after, g.Pre, g.Post)
	}
}

// TestKimiVLKeepsTheFivePrinciples runs Kimi-VL's picture prompt through the
// principles' conditions against Moonshot's own model (mRopeKeepsTheFivePrinciples).
func TestKimiVLKeepsTheFivePrinciples(t *testing.T) {
	for _, name := range kimiFixtures {
		t.Run(name, func(t *testing.T) { mRopeKeepsTheFivePrinciples(t, name) })
	}
}

// TestKimiVLTowerOnDeviceMatchesHost runs MoonViT's blocks on every device
// against the host (rotaryTowerOnDevice), its pairing among the violations.
func TestKimiVLTowerOnDeviceMatchesHost(t *testing.T) {
	for _, name := range kimiFixtures {
		t.Run(name, func(t *testing.T) { rotaryTowerOnDevice(t, name) })
	}
}

// TestKimiVLRealTowerMatchesMoonshot holds the real Kimi-VL-A3B-Instruct's
// MoonViT and projector, converted at F16 by llama.cpp's converter, to
// Moonshot's own code in float32 on a real picture its own processor prepared
// (scripts/kimivlreal.py) -- on the host and with every tower block on each
// device. The whole 16B reference does not fit the test host; the text model is
// DeepSeek-V3's, which Moonlight-16B-A3B already holds to llama.cpp, and its
// file is here only so the tower has a container. The bound is against the
// golden's own simulation of the engine's arithmetic (Q8_0 matrices, int8
// activations); the violation is the projector's per-patch LayerNorm dropped.
func TestKimiVLRealTowerMatchesMoonshot(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "qwenvl", "real-Kimi-VL-A3B-Instruct-tower.json"))
	if err != nil {
		t.Skipf("GOLDEN MISSING: %v -- run scripts/kimivlreal.py on the checkpoint (RULE 11); this gate proved nothing", err)
	}
	g := &qvlGolden{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	text, proj := testmodels.Path("Moonlight-16B-A3B-Instruct-Q4_K_M.gguf"), testmodels.Path("mmproj-Kimi-VL-A3B-Instruct-f16.gguf")
	if !vlmAvailable(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and %s -- scripts/kimivlreal.py writes the mmproj (RULE 11)", text, proj)
	}
	m, err := Open(jlmOfPair(t, text, proj), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	tw := m.Tower()
	encode := func(gpu *tier.GPU) []float32 {
		ts := tw.testState()
		defer ts.Close()
		if gpu != nil {
			ts.SetDevice(gpu)
			if ts.GPUBlocks() != tw.Cfg.NLayer {
				t.Fatalf("the device took %d of %d tower blocks: %v", ts.GPUBlocks(), tw.Cfg.NLayer, gpu.Err())
			}
		}
		px, err := ts.PreprocessImage(goldenImage(g.Image.W, g.Image.H, g.Image.RGB))
		if err != nil {
			t.Fatal(err)
		}
		if ts.Grid() != g.grid() {
			t.Fatalf("the picture's grid is %v, Moonshot's %v", ts.Grid(), g.grid())
		}
		out, err := ts.Encode(px)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), out...)
	}
	host := encode(nil)
	a8, _ := nmse32(g.Image.EmbdA8, host)
	f32, _ := nmse32(g.Image.Embd, host)
	sim, _ := nmse32(g.Image.Embd, g.Image.EmbdA8)
	t.Logf("%d rows: NMSE %.3e against Moonshot's tower at the engine's arithmetic, %.3e against its float32 "+
		"(where that arithmetic alone moves it %.3e)", g.grid().Rows(), a8, f32, sim)
	if math.IsNaN(a8) || a8 > kimiRealBound {
		t.Fatalf("the real tower reads %.3e against Moonshot's (bound %.0e)", a8, kimiRealBound)
	}
	tw.Cfg.fault = faultKimiNoPreNorm
	bad, _ := nmse32(g.Image.EmbdA8, encode(nil))
	tw.Cfg.fault = towerFaultNone
	t.Logf("no per-patch LayerNorm (the violation): %.3e", bad)
	if !(bad > 10*kimiRealBound) {
		t.Fatalf("the violation reads %.3e against %.3e: the gate cannot see it", bad, a8)
	}
	for _, spec := range stepDevices() {
		gpu, err := openTier(t, tier.WithDevices(spec))
		if err != nil || gpu == nil {
			t.Logf("%s: not present", spec)
			continue
		}
		got := encode(gpu)
		gpu.Close()
		dh, _ := nmse32(host, got)
		dm, _ := nmse32(g.Image.EmbdA8, got)
		t.Logf("%s: every tower block placed, %.3e from the host, %.3e from Moonshot's", spec, dh, dm)
		if math.IsNaN(dm) || dm > kimiRealBound {
			t.Errorf("%s: the tower reads %.3e against Moonshot's", spec, dm)
		}
	}
}
