package model

import (
	"encoding/binary"
	"encoding/json"
	"image"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// The gemma3 and internvl towers held to transformers itself: the processor's
// pixel values, and the tower and projector's embeddings, written by
// scripts/visiongold.py from the checkpoints' own weights in float32.

// visionFamily is one projector's gate inputs.
type visionFamily struct {
	name string
	open func(testing.TB, ...Option) (*Model, *Tower)
}

func visionFamilies() []visionFamily {
	return []visionFamily{{"gemma3", openGemma3}, {"internvl", openInternVL}}
}

// visionGoldDir is where visiongold.py writes a family's goldens.
func visionGoldDir(fam string) string { return testmodels.Path(filepath.Join("vlm", "goldens", fam)) }

// readF32 reads a little-endian float32 file, failing a missing one by name.
func readF32(t testing.TB, path string) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		testmodels.Missing(t, "GOLDEN MISSING: %v -- RULE 11: run scripts/visiongold.py; without it "+
			"this gate proves nothing", err)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

// goldMeta is visiongold.py's per-picture record.
type goldMeta struct {
	Tiles, Size, W, H int
	Processor         string
	ImageToken        int     `json:"image_token"`
	Chat              string  `json:"chat"`
	ChatIDs           []int32 `json:"chat_ids"`
}

func readMeta(t testing.TB, fam, pic string) goldMeta {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(visionGoldDir(fam), pic+".json"))
	if err != nil {
		testmodels.Missing(t, "GOLDEN MISSING: %v -- RULE 11: run scripts/visiongold.py %s", err, fam)
	}
	var m goldMeta
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

// goldPictures are the pictures visiongold.py ran: the repo's synthetic one
// and a photograph, PNG both, so Go decodes the pixels PIL decoded.
func goldPicture(t testing.TB, pic string) image.Image {
	t.Helper()
	path := testImage
	if pic == "photo" {
		path = testmodels.Path(filepath.Join("vlm", "goldens", "photo.png"))
	}
	f, err := os.Open(path)
	if err != nil {
		testmodels.Missing(t, "PICTURE MISSING: %v -- RULE 11: run scripts/visiongold.py", err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

// chw2hwc lays transformers' [tile][channel][y][x] out as Encode's
// [tile][y][x][channel].
func chw2hwc(px []float32, n int) []float32 {
	one := 3 * n * n
	out := make([]float32, len(px))
	for t := 0; t < len(px)/one; t++ {
		for ch := 0; ch < 3; ch++ {
			for i := 0; i < n*n; i++ {
				out[t*one+i*3+ch] = px[t*one+ch*n*n+i]
			}
		}
	}
	return out
}

// pixelDiff counts the elements that differ at all, and the largest gap.
func pixelDiff(got, want []float32) (n int, worst float64) {
	for i := range want {
		if got[i] != want[i] {
			n++
			worst = math.Max(worst, math.Abs(float64(got[i]-want[i])))
		}
	}
	return n, worst
}

// TestPreprocessMatchesTransformers holds gemma3's and internvl's processors to
// transformers' on real pictures, to the bit: the antialiased resize, its
// fixed-point rounding, the tiling and the float32 normalisation. Then the
// violations, each of which must move pixels: the CLIP normalisation constants
// in place of the model's, the plain bilinear resize the other projectors use,
// and the tiles handed over in reverse.
func TestPreprocessMatchesTransformers(t *testing.T) {
	for _, fam := range visionFamilies() {
		t.Run(fam.name, func(t *testing.T) {
			m, tw := fam.open(t)
			defer m.Close()
			s := tw.testState()
			defer s.Close()
			for _, pic := range []string{"quad-and-disc", "photo"} {
				meta := readMeta(t, fam.name, pic)
				want := chw2hwc(readF32(t, filepath.Join(visionGoldDir(fam.name), pic+".pixels.f32")), meta.Size)
				img := goldPicture(t, pic)
				if tiles := tw.Cfg.tilesOf(img.Bounds().Dx(), img.Bounds().Dy()); tiles != meta.Tiles {
					t.Errorf("%s: %d tile(s), the processor made %d", pic, tiles, meta.Tiles)
				}
				got, err := s.PreprocessImage(img)
				if err != nil {
					t.Fatal(err)
				}
				if len(got) != len(want) {
					t.Fatalf("%s: %d values, the processor made %d (%d tiles)", pic, len(got), len(want), meta.Tiles)
				}
				n, worst := pixelDiff(got, want)
				t.Logf("%s %dx%d -> %d tile(s): %d of %d values differ, worst %.3g (%s)",
					pic, meta.W, meta.H, meta.Tiles, n, len(want), worst, meta.Processor)
				if n != 0 {
					t.Errorf("%s: %d of %d pixel values differ from %s (worst %.3g)", pic, n, len(want),
						meta.Processor, worst)
				}

				// The violations.
				vio := func(name string, apply func() (undo func())) {
					undo := apply()
					v, err := s.PreprocessImage(img)
					undo()
					if err != nil {
						t.Fatalf("%s: %v", name, err)
					}
					if len(v) != len(want) {
						t.Logf("violation %q: %d values against %d -- caught by shape", name, len(v), len(want))
						return
					}
					n, worst := pixelDiff(v, want)
					t.Logf("violation %q: %d of %d values differ, worst %.3g", name, n, len(want), worst)
					if n == 0 {
						t.Errorf("violation %q moved no pixel: the gate cannot see it", name)
					}
				}
				vio("CLIP's mean and std", func() func() {
					was := tw.Cfg
					tw.Cfg.Mean = [3]float64{0.48145466, 0.4578275, 0.40821073}
					tw.Cfg.Std = [3]float64{0.26862954, 0.26130258, 0.27577711}
					return func() { tw.Cfg = was }
				})
				vio("the plain bilinear resize", func() func() {
					was := tw.Cfg
					tw.Cfg.Kind = jlm.ProjIdefics3
					return func() { tw.Cfg = was }
				})
				if meta.Tiles > 1 {
					vio("the tiles in reverse", func() func() {
						was := tw.Cfg
						tw.Cfg.fault = towerFaultTilesReversed
						return func() { tw.Cfg = was }
					})
				}
			}
		})
	}
}

// TestVisionMatchesTransformers holds each tower and projector, end to end, to
// transformers' image features: on the raw rainbow (no preprocessing in it)
// and on the processor's own pixels for a photograph -- for internvl that is
// thirteen tiles, so the tile order and the thumbnail are in it. The bound is
// the int8 activation quantization's compounding over the blocks, a decade
// under the violations: the pool skipped, the shuffle transposed, and the class
// token where llama.cpp puts it.
func TestVisionMatchesTransformers(t *testing.T) {
	for _, fam := range visionFamilies() {
		t.Run(fam.name, func(t *testing.T) {
			m, tw := fam.open(t)
			defer m.Close()
			dir := visionGoldDir(fam.name)
			n := tw.Cfg.ImageSz
			run := func(px []float32) []float32 {
				s := tw.testState()
				defer s.Close()
				out, err := s.Encode(px)
				if err != nil {
					t.Fatal(err)
				}
				return append([]float32(nil), out...)
			}
			check := func(name string, got, want []float32, bound float64) float64 {
				if len(got) != len(want) {
					t.Fatalf("%s: %d values, transformers %d", name, len(got), len(want))
				}
				nm, worst := nmseOf(got, want)
				t.Logf("%-28s NMSE %.3e  max|d| %.4f", name, nm, worst)
				if math.IsNaN(nm) || math.IsInf(nm, 0) {
					t.Fatalf("%s: NMSE %v -- a degenerate comparison is a failure", name, nm)
				}
				if bound > 0 && nm > bound {
					t.Errorf("%s: NMSE %.3e against transformers (bound %.0e)", name, nm, bound)
				}
				return nm
			}
			// Clean reads 4.6e-3 (gemma3's 27 blocks) on the rainbow. That is the
			// engine's int8 activations against Q8_0 weights: transformers' own
			// tower with its Linear inputs rounded to q8_0 per 32 and its
			// weights to Q8_0 reads 2.1e-3 against itself, the weights alone
			// 2.7e-4. llama.cpp (f16 weights, f32 activations) reads 1e-6.
			const bound = 2e-2
			rain := rainbow(n)
			want := readF32(t, filepath.Join(dir, "rainbow.features.f32"))
			check("rainbow", run(rain), want, bound)

			meta := readMeta(t, fam.name, "photo")
			px := chw2hwc(readF32(t, filepath.Join(dir, "photo.pixels.f32")), meta.Size)
			photo := readF32(t, filepath.Join(dir, "photo.features.f32"))
			cleanPhoto := check("photo ("+strconv.Itoa(meta.Tiles)+" tiles)", run(px), photo, bound)

			check("grid", run(patchGrid(n, tw.Cfg.PatchSz)), readF32(t, filepath.Join(dir, "grid.features.f32")), bound)

			// The violations, on the photograph. The shuffle's order is not
			// among them: the reference itself moves by NMSE 1.9e-2 to 2.8e-2
			// under a 2x2 transpose, since after 24 blocks neighbouring patches
			// look alike, which is inside this gate's precision; it is held by
			// TestProjectorMatchesTransformers on rows that are no picture.
			vio := func(name string, f towerFault, clsLast bool, px, want []float32, clean float64) {
				was, wasCLS := tw.Cfg.fault, tw.opt.towerCLSLast
				tw.Cfg.fault, tw.opt.towerCLSLast = f, clsLast
				nm := check("violation: "+name, run(px), want, 0)
				tw.Cfg.fault, tw.opt.towerCLSLast = was, wasCLS
				if nm < 10*clean || nm < bound {
					t.Errorf("violation %q reads NMSE %.3e against a clean %.3e: the gate cannot see it",
						name, nm, clean)
				}
			}
			switch fam.name {
			case "gemma3":
				vio("pooling skipped", towerFaultPoolCorner, false, px, photo, cleanPhoto)
			case "internvl":
				vio("class token last (llama.cpp's layout)", towerFaultNone, true, px, photo, cleanPhoto)
			}
		})
	}
}
