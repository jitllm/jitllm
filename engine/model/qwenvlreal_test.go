package model

import (
	"encoding/json"
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestQwenVLRealMatchesTransformers holds a real Qwen-VL checkpoint, with a
// real picture in its prompt, to transformers on the CPU in float32
// (scripts/qwenvlreal.py): the text model on transformers' own image rows,
// which pins M-RoPE on the real geometry, and the whole model from the pixels
// on. The container is the one llama.cpp's converter writes from the same
// checkpoint at F16, so what separates the two is the F16 weights and the
// tower's Q8_0 and int8.
//
// The violation is the image at sequential positions, as an engine without
// M-RoPE lays it: its logits are the "before" and the clean arm's the "after".
//
// Qwen2.5-VL-3B is held to transformers in bfloat16 (its float32 copy does
// not fit in memory) on the published Q4_K_M and Q8_0
// mmproj, so its bound is the argmax with the quantization's NMSE: it is the
// one checkpoint here whose llama-mtmd-cli answer leaves transformers' at
// the first token, which makes transformers the reference that settles it.
func TestQwenVLRealMatchesTransformers(t *testing.T) {
	for _, c := range []struct {
		name, text, proj string
		// textNMSE bounds the text model on transformers' image rows, whole
		// the model from the pixels on; flips is the argmaxes either may part.
		textNMSE, whole float64
		textFlips       int
		// mrope says the arm is precise enough to see sequential image
		// positions: at F16 against float32 it is, at Q4_K_M against
		// bfloat16 the quantization alone is 1.2e-2 and the violation 1.9e-2.
		mrope bool
	}{
		{"Qwen2-VL-2B-Instruct", "Qwen2-VL-2B-Instruct-f16.gguf", "mmproj-Qwen2-VL-2B-Instruct-f16.gguf", 1e-3, 2e-2, 0, true},
		{"Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf", 2e-2, 2e-2, 1, false},
		// Qwen3-VL-2B and Qwen3.5-0.8B at F16 against float32: the
		// interleaved M-RoPE, the deepstack rows (Qwen3-VL) and the resampled
		// position table on the real geometry.
		{"Qwen3-VL-2B-Instruct", "Qwen3-VL-2B-Instruct-f16.gguf", "mmproj-Qwen3-VL-2B-Instruct-f16.gguf", 1e-3, 2e-2, 0, true},
		{"Qwen3.5-0.8B", "Qwen3.5-0.8B-f16.gguf", "mmproj-Qwen3.5-0.8B-f16.gguf", 1e-3, 2e-2, 0, true},
		// HunyuanOCR at F16 against float32 (scripts/hunyuanvlreal.py): XD-RoPE
		// and the newline merger on the real geometry.
		{"HunyuanOCR", "HunyuanOCR-f16.gguf", "mmproj-HunyuanOCR-f16.gguf", 1e-3, 2e-2, 0, true},
	} {
		name := c.name
		t.Run(name, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "qwenvl", "real-"+name+".json"))
			if err != nil {
				t.Skipf("GOLDEN MISSING: %v -- run scripts/qwenvlreal.py on the checkpoint (RULE 11); this gate proved nothing", err)
			}
			g := &qvlGolden{}
			if err := json.Unmarshal(raw, g); err != nil {
				t.Fatal(err)
			}
			text, proj := testmodels.Path(c.text), testmodels.Path(c.proj)
			if !vlmAvailable(text, proj) {
				testmodels.Missing(t, "MODEL MISSING: %s and its mmproj -- QVL_GGUF=1 scripts/qwenvlreal.py writes them (RULE 11)", text)
			}
			m, err := Open(jlmOfPair(t, text, proj), noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			n := len(g.Pre) + g.grid().Rows() + len(g.Post) + len(g.Cont) + 1
			// state is a text state, on gpu when one is given: placed before the
			// tower, as the CLI places it, so a card too small for both gives
			// the text model its blocks and the tower what is left.
			state := func(gpu *tier.GPU) *State {
				st := m.NewState(n)
				if gpu != nil {
					st.SetDevice(gpu)
					if st.GPULayers() == 0 {
						t.Fatalf("the device took no text block: %v", gpu.Err())
					}
				}
				return st
			}
			runOn := func(st *State, img Span) [][]float32 {
				lg, err := st.PrefillMixed(Span{Tokens: g.Pre}, img, Span{Tokens: g.Post})
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
			run := func(img Span) [][]float32 {
				st := state(nil)
				defer st.Close()
				return runOn(st, img)
			}
			var deep []float32
			if len(g.Image.Deep) > 0 {
				deep = g.Image.Deep
			}
			after, flips := qvlWorst(g, run(Span{Embd: g.Image.Embd, Grid: g.grid(), Deep: deep}))
			before, bflips := qvlWorst(g, run(Span{Embd: g.Image.Embd, Deep: deep}))
			t.Logf("transformers (%s) image rows, %dx%d merged grid, %d rows teacher-forced: worst logit NMSE %.3e, "+
				"%d argmax flips; at sequential positions (before M-RoPE) %.3e, %d flips",
				g.dtypeName(), g.Image.Grid[1], g.Image.Grid[2], len(g.Logits), after, flips, before, bflips)
			if math.IsNaN(after) || after > c.textNMSE || flips > c.textFlips {
				t.Errorf("the text model on transformers' image rows: worst NMSE %.3e, %d flips", after, flips)
			}
			if c.mrope && !(before > 10*after) {
				t.Errorf("sequential positions read %.3e against %.3e: the gate cannot see M-RoPE", before, after)
			}

			// The whole model: the engine's preprocessing and tower, on the host
			// and with the tower and the text model on each device -- the text
			// model on textGPU, placed first, and the tower on gpu. It reports
			// false when the text model took the card and left the tower no
			// room, which the caller answers by running the tower alone.
			whole := func(t *testing.T, textGPU, gpu *tier.GPU) bool {
				st := state(textGPU)
				defer st.Close()
				tw := m.Tower()
				ts := tw.testState()
				defer ts.Close()
				if gpu != nil {
					ts.SetDevice(gpu)
				}
				f, err := os.Open(filepath.Join("testdata", "quad-and-disc.png"))
				if err != nil {
					t.Fatal(err)
				}
				px, err := ts.Preprocess(f)
				f.Close()
				if err != nil {
					t.Fatal(err)
				}
				if ts.Grid() != g.grid() {
					t.Fatalf("the picture's grid is %v, transformers' %v", ts.Grid(), g.grid())
				}
				emb, err := ts.Encode(px)
				if err != nil {
					t.Fatal(err)
				}
				emb = append([]float32(nil), emb...)
				embd, deep := splitDeep(emb, ts.Grid().Rows(), tw.Cfg.ProjDim)
				tower, _ := nmse32(g.Image.Embd, embd)
				if deep != nil {
					d, _ := nmse32(g.Image.Deep, deep)
					t.Logf("the engine's deepstack rows NMSE %.3e against transformers'", d)
				}
				worst, wflips := qvlWorst(g, runOn(st, Span{Embd: embd, Grid: ts.Grid(), Deep: deep}))
				t.Logf("%d of %d tower and %d of %d text blocks placed: the engine's own tower rows NMSE %.3e "+
					"against transformers'; end to end worst logit NMSE %.3e, %d argmax flips",
					ts.GPUBlocks(), tw.Cfg.NLayer, st.GPULayers(), m.Cfg.NLayer, tower, worst, wflips)
				// The device asked for runs the tower: blocks placed, and every
				// placed block's run made there (RULE 10: the count, not the
				// call). Its set is built for this picture's rows, not the
				// tower's largest grid, which on HunyuanOCR (65536 patches) was
				// 5.67 GB of scratch and no tower block on any card here.
				if math.IsNaN(worst) || worst > c.whole || wflips > 1 {
					t.Errorf("end to end: worst NMSE %.3e, %d flips", worst, wflips)
				}
				if gpu != nil {
					placed, text := ts.GPUBlocks(), st.GPULayers()
					st := gpu.Stats()
					t.Logf("tower on the device: %d of %d blocks placed, %d runs there and %d on the host; "+
						"its set built for %d rows (the tower takes %d), %d bytes of scratch",
						placed, tw.Cfg.NLayer, ts.vis.devBlocks, ts.vis.hostBlocks, st.TowerRows,
						tw.Cfg.MaxSeq(), st.ScratchBytes)
					// The text model is placed first, as the CLI places it, and
					// on a 4 GB card it can leave no room for a tower block: the
					// tower is then held to the card on its own, where it must
					// take blocks.
					if placed == 0 && textGPU != nil {
						t.Logf("the text model took %d of %d blocks and left no room (%s): the tower runs alone",
							text, m.Cfg.NLayer, gpu.Err())
						return false
					}
					if placed == 0 || ts.vis.devBlocks != int64(placed) ||
						ts.vis.hostBlocks != int64(tw.Cfg.NLayer-placed) {
						t.Errorf("%d of %d tower blocks placed, %d ran on the device and %d on the host: %s",
							placed, tw.Cfg.NLayer, ts.vis.devBlocks, ts.vis.hostBlocks, gpu.Err())
					}
					// A picture of this size is a few thousand rows at most.
					if st.TowerRows == 0 || st.TowerRows > 4096 {
						t.Errorf("the tower's set is built for %d rows (its largest grid %d) for a %v picture",
							st.TowerRows, tw.Cfg.MaxSeq(), ts.Grid())
					}
				}
				return true
			}
			t.Run("host", func(t *testing.T) { whole(t, nil, nil) })
			for _, spec := range stepDevices() {
				t.Run(spec, func(t *testing.T) {
					gpu, err := openTier(t, tier.WithDevices(spec))
					if err != nil || gpu == nil {
						t.Skipf("%s: not present (%v)", spec, err)
					}
					ok := whole(t, gpu, gpu)
					gpu.Close()
					if ok {
						return
					}
					// A card of its own: the text model's weights are the
					// model's on the tier, and stay there after its State.
					alone, err := openTier(t, tier.WithDevices(spec))
					if err != nil || alone == nil {
						t.Fatalf("%s: reopening: %v", spec, err)
					}
					defer alone.Close()
					whole(t, nil, alone)
				})
			}
		})
	}
}

// TestQwenVLWindowsSurviveQueryChunks runs Qwen2.5-VL-3B's tower on its
// largest picture, whose score planes pass a device's vision budget, so every
// block's attention runs in query chunks (tier.Stats.VisionAttnChunks) and
// each chunk masks with its own rows' windows. The device is held to the host,
// which never chunks; the violation is windows as wide as the image on the
// device.
func TestQwenVLWindowsSurviveQueryChunks(t *testing.T) {
	text := testmodels.Path("Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf")
	proj := testmodels.Path("mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf")
	if !vlmAvailable(text, proj) {
		t.Skipf("MODEL MISSING: %s and its mmproj (set JITLLM_MODELS) -- this gate proved nothing", text)
	}
	m, err := Open(jlmOfPair(t, text, proj), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	tw := m.Tower()
	side := int(math.Sqrt(float64(tw.Cfg.MaxPixels)))
	img := image.NewRGBA(image.Rect(0, 0, side, side))
	for y := 0; y < side; y++ {
		for x := 0; x < side; x++ {
			img.Set(x, y, color.RGBA{uint8(x * 255 / side), uint8(y * 255 / side), uint8((x ^ y) * 7), 255})
		}
	}
	encode := func(gpu *tier.GPU) ([]float32, int) {
		ts := tw.testState()
		defer ts.Close()
		if gpu != nil {
			ts.SetDevice(gpu)
			if ts.GPUBlocks() == 0 {
				t.Fatalf("the device took no tower block: %v", gpu.Err())
			}
		}
		px, err := ts.PreprocessImage(img)
		if err != nil {
			t.Fatal(err)
		}
		e, err := ts.Encode(px)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), e...), ts.GPUBlocks()
	}
	want, _ := encode(nil)
	ran := 0
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			gpu, err := openTier(t, tier.WithDevices(spec))
			if err != nil || gpu == nil {
				t.Skipf("%s: not present (%v)", spec, err)
			}
			defer gpu.Close()
			ran++
			before := gpu.Stats().VisionAttnChunks
			got, placed := encode(gpu)
			chunks := gpu.Stats().VisionAttnChunks - before
			if chunks == 0 {
				t.Fatalf("%d of %d tower blocks placed and no attention ran in query chunks: "+
					"this arm proved nothing", placed, tw.Cfg.NLayer)
			}
			clean, _ := nmse32(want, got)
			was := tw.Cfg.WinSize
			tw.Cfg.WinSize = 1 << 20
			wide, _ := encode(gpu)
			tw.Cfg.WinSize = was
			bad, _ := nmse32(want, wide)
			t.Logf("%dx%d picture, %d of %d tower blocks placed, %d query chunks: NMSE %.3e against the host; "+
				"windows as wide as the image %.3e", side, side, placed, tw.Cfg.NLayer, chunks, clean, bad)
			if math.IsNaN(clean) || clean > 2e-2 {
				t.Errorf("the chunked tower reads %.3e against the host", clean)
			}
			if !(bad > 10*clean) {
				t.Errorf("windows as wide as the image read %.3e against %.3e: the gate cannot see them", bad, clean)
			}
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}
