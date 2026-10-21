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

// TestHunyuanVLTextMatchesTransformers holds the text model to transformers'
// on the golden's own image rows -- decode, a batch slot and every device
// (mRopeMatchesTransformers) -- and fails XD-RoPE's two pieces taken out: each
// pair's second element on its first element's axis (llama.cpp's
// ggml_rope_multi, which assigns axes per pair; RULE 7m), and k normed before
// the rotary.
func TestHunyuanVLTextMatchesTransformers(t *testing.T) {
	for _, name := range hyFixtures {
		t.Run(name, func(t *testing.T) {
			mRopeMatchesTransformers(t, name)
			m, g := openQVL(t, name)
			defer m.Close()
			if !m.Cfg.RopeXD || m.ropeB == nil {
				t.Fatal("the container's text model is not an XD-RoPE one")
			}
			run := func() float64 {
				st := m.NewState(len(g.Pre) + g.grid().Rows() + len(g.Post) + len(g.Cont) + 1)
				defer st.Close()
				w, flips := qvlWorst(g, qvlRun(t, st, g, true))
				if flips > 0 {
					w = math.Max(w, 1)
				}
				return w
			}
			clean := run()
			was := m.ropeB.Runs
			m.ropeB.Runs = m.rope.Runs
			perPair := run()
			m.ropeB.Runs = was
			m.faults.xdKNormFirst = true
			kFirst := run()
			m.faults.xdKNormFirst = false
			t.Logf("clean %.3e; axes per pair (llama.cpp) %.3e; k normed before the rotary %.3e", clean, perPair, kFirst)
			for what, bad := range map[string]float64{"axes per pair": perPair, "k normed first": kFirst} {
				if !(bad > 1000*math.Max(clean, 1e-12)) {
					t.Errorf("%s reads %.3e against %.3e: the gate cannot see it", what, bad, clean)
				}
			}
		})
	}
}

// hyFixtures are the HunyuanVL fixtures scripts/hunyuanvlgold.py writes:
// transformers' own HunYuanVLForConditionalGeneration, converted by
// llama.cpp's converter.
var hyFixtures = []string{"synth-hunyuanvl"}

// hyTowerGate is TestQwenTowerMatchesTransformers' half for HunyuanVL's tower:
// each of its own pieces taken out must move the output far past the bound --
// the table resampled with aligned corners, the table unresampled, the merge
// group transposed, and the newline rows.
func hyTowerGate(t *testing.T, tw *Tower, g *qvlGolden, clean float64) {
	t.Helper()
	for _, v := range []struct {
		name  string
		fault towerFault
	}{
		{"the table resampled with aligned corners", faultBilinearTable},
		{"the table read unresampled", faultNoPosLerp},
		{"the merge group transposed", towerFaultShuffleSwap},
		{"no newline rows", faultHyNoNewline},
	} {
		tw.Cfg.fault = v.fault
		bad, _ := qvlEncode(t, tw, g, nil)
		tw.Cfg.fault = towerFaultNone
		nb, _ := nmse32(g.Image.EmbdA8, bad)
		t.Logf("violation %q: NMSE %.3e", v.name, nb)
		if !(nb > 10*towerBound) || math.IsNaN(nb) {
			t.Errorf("%s reads %.3e against a clean %.3e: the gate cannot see it", v.name, nb, clean)
		}
	}
}

// TestHunyuanVLLaysTheImageOutAsTransformersDoes holds xdRope to
// transformers' get_rope_index row for row: the text at its sequence
// position on every axis, the picture's begin and end rows the same, its grid
// rows -- the newline column included -- at (position, column, row, picture
// 0), and the continuation unshifted.
func TestHunyuanVLLaysTheImageOutAsTransformersDoes(t *testing.T) {
	for _, name := range hyFixtures {
		t.Run(name, func(t *testing.T) {
			m, g := openQVL(t, name)
			defer m.Close()
			st := m.NewState(len(g.Pos4) + 1)
			defer st.Close()
			rows, next, err := st.spanRope(0, 0, qvlSpans(g, true))
			if err != nil {
				t.Fatal(err)
			}
			if len(rows)+len(g.Cont) != len(g.Pos4) {
				t.Fatalf("%d prompt rows and %d continuation, transformers lays out %d", len(rows), len(g.Cont),
					len(g.Pos4))
			}
			for i, r := range rows {
				if r != g.Pos4[i] {
					t.Fatalf("row %d at %v, transformers %v", i, r, g.Pos4[i])
				}
			}
			if want := g.Pos4[len(rows)][0]; next != want {
				t.Fatalf("the continuation starts at %d, transformers %d", next, want)
			}
			t.Logf("%d rows on four axes as transformers lays them out; the continuation at %d", len(rows), next)
		})
	}
}

// TestHunyuanVLKeepsTheFivePrinciples runs HunyuanVL's picture prompt through
// the principles' conditions against transformers (mRopeKeepsTheFivePrinciples).
func TestHunyuanVLKeepsTheFivePrinciples(t *testing.T) {
	for _, name := range hyFixtures {
		t.Run(name, func(t *testing.T) { mRopeKeepsTheFivePrinciples(t, name) })
	}
}

// TestHunyuanVLPrefixCacheKeysOnPositions restores a picture prompt from the
// prefix cache onto transformers' continuation, and keys the same rows under
// another grid apart (mRopePrefixCacheKeysOnPositions).
func TestHunyuanVLPrefixCacheKeysOnPositions(t *testing.T) {
	for _, name := range hyFixtures {
		t.Run(name, func(t *testing.T) { mRopePrefixCacheKeysOnPositions(t, name) })
	}
}

// TestHunyuanVLChatSpansAreTheProcessors renders a picture and a question
// through the container's own template and holds the tokens around the
// picture's rows -- its start and end place-holders included -- to the prompt
// transformers' processor built.
func TestHunyuanVLChatSpansAreTheProcessors(t *testing.T) {
	for _, name := range hyFixtures {
		t.Run(name, func(t *testing.T) {
			m, g := openQVL(t, name)
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
			t.Logf("before the picture %v, after %v", before, after)
			if !slices.Equal(before, g.Pre) || !slices.Equal(after, g.Post) {
				t.Fatalf("the template wraps the picture in %v ... %v, the processor in %v ... %v",
					before, after, g.Pre, g.Post)
			}
		})
	}
}

// TestHunyuanVLTowerOnDeviceMatchesHost runs every tower block on each device
// against the host and against transformers.
func TestHunyuanVLTowerOnDeviceMatchesHost(t *testing.T) {
	for _, name := range hyFixtures {
		t.Run(name, func(t *testing.T) {
			m, g := openQVL(t, name)
			defer m.Close()
			tw := m.Tower()
			host, _ := qvlEncode(t, tw, g, nil)
			defer func(was int) { tw.opt.towerGPULayers = was }(tw.opt.towerGPULayers)
			tw.opt.towerGPULayers = -1
			ran := 0
			for _, spec := range stepDevices() {
				gpu, err := openTier(t, tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
				if err != nil || gpu == nil {
					t.Logf("%s: not present", spec)
					continue
				}
				ran++
				var blocks int
				got, _ := qvlEncode(t, tw, g, func(s *State) {
					s.SetDevice(gpu)
					blocks = s.GPUBlocks()
				})
				gpuErr := gpu.Err()
				gpu.Close()
				if blocks != tw.Cfg.NLayer {
					t.Fatalf("%s took %d of %d tower blocks: %v", spec, blocks, tw.Cfg.NLayer, gpuErr)
				}
				vh, _ := nmse32(host, got)
				vt, _ := nmse32(g.Image.EmbdA8, got)
				t.Logf("%s: every tower block placed; NMSE %.3e against the host, %.3e against transformers", spec, vh, vt)
				if math.IsNaN(vh) || vh > towerBound || math.IsNaN(vt) || vt > towerBound {
					t.Errorf("%s: NMSE %.3e against the host, %.3e against transformers (bound %.0e)",
						spec, vh, vt, towerBound)
				}
			}
			if ran == 0 {
				t.Skip("no device on this host")
			}
		})
	}
}

// TestHunyuanVLRealTowerOnDevice runs the real HunyuanOCR's 27 tower blocks
// on each device, against the host and against transformers' rows
// (scripts/hunyuanvlreal.py). The tower's capacity is lowered to 1024^2
// pixels first: at the checkpoint's own 4096^2 (65536 rows) its scratch does
// not fit a 4 GB card, which declines it on budget, and the picture is 512^2.
func TestHunyuanVLRealTowerOnDevice(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "qwenvl", "real-HunyuanOCR.json"))
	if err != nil {
		t.Skipf("GOLDEN MISSING: %v -- run scripts/hunyuanvlreal.py (RULE 11); this gate proved nothing", err)
	}
	g := &qvlGolden{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	text, proj := testmodels.Path("HunyuanOCR-f16.gguf"), testmodels.Path("mmproj-HunyuanOCR-f16.gguf")
	if !vlmAvailable(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj -- HYVL_GGUF=1 scripts/hunyuanvlreal.py writes them (RULE 11)", text)
	}
	m, err := Open(jlmOfPair(t, text, proj), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	tw := m.Tower()
	tw.Cfg.MaxPixels = 1024 * 1024
	encode := func(s *State) []float32 {
		f, err := os.Open(filepath.Join("testdata", "quad-and-disc.png"))
		if err != nil {
			t.Fatal(err)
		}
		px, err := s.Preprocess(f)
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		out, err := s.Encode(px)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), out...)
	}
	hs := tw.testState()
	host := encode(hs)
	hs.Close()
	defer func(was int) { tw.opt.towerGPULayers = was }(tw.opt.towerGPULayers)
	tw.opt.towerGPULayers = -1
	ran := 0
	for _, spec := range stepDevices() {
		gpu, err := openTier(t, tier.WithDevices(spec))
		if err != nil || gpu == nil {
			t.Logf("%s: not present", spec)
			continue
		}
		ran++
		s := tw.testState()
		s.SetDevice(gpu)
		got := encode(s)
		blocks, gpuErr := s.GPUBlocks(), gpu.Err()
		s.Close()
		gpu.Close()
		if blocks != tw.Cfg.NLayer {
			t.Fatalf("%s took %d of %d tower blocks: %v", spec, blocks, tw.Cfg.NLayer, gpuErr)
		}
		vh, _ := nmse32(host, got)
		vt, _ := nmse32(g.Image.Embd, got)
		t.Logf("%s: every tower block placed; NMSE %.3e against the host, %.3e against transformers", spec, vh, vt)
		if math.IsNaN(vh) || vh > 1e-3 || math.IsNaN(vt) || vt > 2e-3 {
			t.Errorf("%s: NMSE %.3e against the host, %.3e against transformers", spec, vh, vt)
		}
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}

// TestHunyuanVLPictureCountsItsNewlineAndEndRows holds State.Picture -- the
// path `jitllm run -image` and the server take -- to the golden's grid: a
// picture is H*(W+1)+2 rows, which the header-sized prompt (GridFor,
// GridRows) and the preprocessed picture must agree on. Counting the merged
// units alone (H*W) undercounts, and the picture was refused.
func TestHunyuanVLPictureCountsItsNewlineAndEndRows(t *testing.T) {
	for _, name := range hyFixtures {
		t.Run(name, func(t *testing.T) {
			m, g := openQVL(t, name)
			defer m.Close()
			st := m.NewState(len(g.Pre) + g.grid().Rows() + len(g.Post) + 1)
			defer st.Close()
			pc, err := st.Picture(goldenImage(g.Image.W, g.Image.H, g.Image.RGB))
			if err != nil {
				t.Fatal(err)
			}
			c := m.Tower().Cfg
			sized, err := c.GridFor(g.Image.W, g.Image.H)
			if err != nil {
				t.Fatal(err)
			}
			if pc.Grid != g.grid() || pc.Rows != g.grid().Rows() || c.GridRows(sized) != pc.Rows {
				t.Fatalf("the picture is %d rows on %+v, the prompt sized it %d, transformers' %d on %+v",
					pc.Rows, pc.Grid, c.GridRows(sized), g.grid().Rows(), g.grid())
			}
			t.Logf("%d rows on %+v", pc.Rows, pc.Grid)
		})
	}
}
