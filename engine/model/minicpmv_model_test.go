package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// MiniCPM-V 4.0, end to end: the GGUF pair converted into one container, the
// tower and resampler held to the checkpoint's own code (scripts/minicpmvgold.py
// writes the goldens beside the model), and the model's answer about a picture.

var (
	mcvText = testmodels.Path("minicpmv/MiniCPM-V-4-Q4_K_M.gguf")
	mcvProj = testmodels.Path("minicpmv/mmproj-MiniCPM-V-4-f16.gguf")
	// mcvGold is scripts/minicpmvgold.py's output for the pictures below.
	mcvGold = testmodels.Path("minicpmv/hf/MiniCPM-V-4/gold")
	// mcvPhoto is llama.cpp's tools/mtmd/test-1.jpeg resampled by Pillow to
	// 896x448 and stored as PNG: two whole 448 slices and an overview, a cut
	// on which the processor and llama.cpp's port agree pixel for pixel, and a
	// PNG so neither decoder's JPEG rounding is in the comparison.
	mcvPhoto = testmodels.Path("minicpmv/photo896x448.png")
)

func openMiniCPMV(t testing.TB, opts ...Option) *Model {
	t.Helper()
	if !vlmAvailable(mcvText, mcvProj) {
		t.Skipf("MODEL MISSING: %s, its mmproj and its merged container -- this gate proved nothing", mcvText)
	}
	m, err := Open(jlmOfPair(t, mcvText, mcvProj), opts...)
	if err != nil {
		t.Fatal(err)
	}
	if m.Tower() == nil {
		m.Close()
		t.Fatal("the merged container carries no tower, so this gate proves nothing")
	}
	return m
}

// TestMiniCPMVTowerShape pins what the container says about the tower and the
// resampler, every figure read off the file first.
func TestMiniCPMVTowerShape(t *testing.T) {
	m := openMiniCPMV(t)
	defer m.Close()
	c := m.Tower().Cfg
	for _, tc := range []struct {
		name      string
		got, want int
	}{
		{"NLayer", c.NLayer, 27}, {"NEmbd", c.NEmbd, 1152}, {"NHead", c.NHead, 16},
		// 4304 padded to a whole Q8_0 block at conversion (convert.padFFN).
		{"NFFN", c.NFFN, 4320},
		{"ImageSz", c.ImageSz, 448}, {"PatchSz", c.PatchSz, 14},
		{"PosBuckets", c.PosBuckets, 70}, {"Queries", c.Queries, 64}, {"Tokens", c.Tokens(), 64},
		{"ProjDim", c.ProjDim, 2560}, {"text NEmbd", m.Cfg.NEmbd, 2560},
	} {
		if tc.got != tc.want {
			t.Errorf("%s = %d, want %d", tc.name, tc.got, tc.want)
		}
	}
	if c.Projector != jlm.ProjResampler.String() || c.CLS || c.Rope || c.PreLN {
		t.Errorf("projector %s, class token %v, rope %v, pre-LN %v", c.Projector, c.CLS, c.Rope, c.PreLN)
	}
	if rs := m.Tower().rs; rs == nil || rs.heads != 20 || rs.d != 2560 {
		t.Errorf("the resampler is %+v, want 20 heads of 128 over 2560", rs)
	}
}

// mcvGolden is one picture's manifest from scripts/minicpmvgold.py.
type mcvGolden struct {
	W, H   int
	Grid   []int
	Pieces []struct {
		W, H, GH, GW int
		Pos          []int32
		VitSum       float64 `json:"vit_sum"`
		EmbSum       float64 `json:"emb_sum"`
	}
}

func readF32n(t testing.TB, path string, n int) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		testmodels.Missing(t, "GOLDEN MISSING: %v -- scripts/minicpmvgold.py writes it", err)
	}
	if len(b) != 4*n {
		t.Fatalf("%s is %d bytes, want %d floats", path, len(b), n)
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func readMcvGolden(t testing.TB, name string) mcvGolden {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(mcvGold, name+".json"))
	if err != nil {
		t.Skipf("GOLDEN MISSING: %v -- run scripts/minicpmvgold.py; this gate proved nothing", err)
	}
	var g mcvGolden
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	if len(g.Pieces) == 0 {
		t.Fatal("the golden has no pieces")
	}
	return g
}

// nmseOf is sum (got-want)^2 / sum want^2, refusing a degenerate comparison:
// NaN > bound is false, so a non-finite reading must fail on its own.
func visNMSE(t testing.TB, what string, got, want []float32) float64 {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d floats, want %d", what, len(got), len(want))
	}
	var num, den float64
	for i := range want {
		d := float64(got[i] - want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	n := num / den
	if math.IsNaN(n) || math.IsInf(n, 0) || den == 0 {
		t.Fatalf("%s: NMSE %v over a reference of energy %v -- degenerate, a failure", what, n, den)
	}
	return n
}

// mcvPictures are the goldens the gates read: the photo, cut into an overview
// and two slices, and the repository's 384 square, which is an overview alone.
var mcvPictures = []struct{ name, path string }{
	{"photo896x448", mcvPhoto},
	{"quad-and-disc", testImage},
}

func decodeImage(t testing.TB, path string) image.Image {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("IMAGE MISSING: %v -- this gate proved nothing", err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

// TestMiniCPMVMatchesTransformers holds the whole picture path to the
// checkpoint's own code: Pillow's cut and resize, the tower's output after its
// post-LayerNorm, and the resampler's 64 rows, piece by piece. The tower runs
// on the golden's own pixels, and the resampler is held twice -- after this
// tower, and alone on the reference's own tower output -- so each stage's error
// is its own.
//
// The tower's bound is the int8 activation's, and on this tower it is wide:
// from block 13 on, nine channels of the residual sit at ~2000 against a
// median of ~0.3 (SigLIP-so400m's massive activations), so a 32-wide
// quantization window holding one of them rounds its neighbours away. The
// tower reads ~1.2e-2 against float32 here where llama.cpp's f16 run reads
// 0.5% relative RMS on the dump's samples (TestMiniCPMVTowerMatchesLlamaCpp);
// the answers still match llama.cpp's token for token. The resampler alone
// reads 1.6-3.0e-4, which is what lets its violations be seen.
func TestMiniCPMVMatchesTransformers(t *testing.T) {
	m := openMiniCPMV(t)
	defer m.Close()
	tw := m.Tower()
	c := tw.Cfg
	const vitBound, embBound, rsBound = 3e-2, 3e-2, 5e-4
	for _, pic := range mcvPictures {
		t.Run(pic.name, func(t *testing.T) {
			g := readMcvGolden(t, pic.name)
			img := decodeImage(t, pic.path)
			lay, err := tw.Layout(g.W, g.H)
			if err != nil {
				t.Fatal(err)
			}
			if len(lay.Pieces) != len(g.Pieces) {
				t.Fatalf("%d pieces, the processor cut %d", len(lay.Pieces), len(g.Pieces))
			}
			s := tw.testState()
			defer s.Close()
			px, err := tw.Pieces(img, lay)
			if err != nil {
				t.Fatal(err)
			}
			for i, gp := range g.Pieces {
				b := lay.Pieces[i]
				if b.W != gp.W || b.H != gp.H {
					t.Fatalf("piece %d is %dx%d, the processor's %dx%d", i, b.W, b.H, gp.W, gp.H)
				}
				base := filepath.Join(mcvGold, fmt.Sprintf("%s.p%d", pic.name, i))
				want := readF32n(t, base+".px", gp.W*gp.H*3)
				// One 8-bit step, normalised, is 1/127.5. A downscale's sums sit
				// within f32 noise of a half on a few samples in ten thousand.
				off, worst := 0, float32(0)
				for j := range want {
					d := float32(math.Abs(float64(px[i][j] - want[j])))
					if d > 1e-6 {
						off++
					}
					worst = max(worst, d)
				}
				t.Logf("piece %d %dx%d: %d of %d pixel values off the processor's, worst by %.5f",
					i, gp.W, gp.H, off, len(want), worst)
				if worst > 1.01/127.5 || off*1000 > len(want) {
					t.Errorf("piece %d: the preprocessing is not the processor's", i)
				}

				var vit []float32
				s.vis.beforeProj = func(x []float32) { vit = append(vit[:0], x...) }
				emb, err := s.EncodeGrid(want, gp.GH, gp.GW)
				if err != nil {
					t.Fatal(err)
				}
				s.vis.beforeProj = nil
				wantVit := readF32n(t, base+".vit", gp.GH*gp.GW*c.NEmbd)
				wantEmb := readF32n(t, base+".emb", c.Tokens()*c.ProjDim)
				nv := visNMSE(t, "tower", vit, wantVit)
				ne := visNMSE(t, "end to end", emb, wantEmb)
				alone := func() float64 {
					out := make([]float32, c.Tokens()*c.ProjDim)
					if err := s.resample(out, wantVit, gp.GH, gp.GW); err != nil {
						t.Fatal(err)
					}
					return visNMSE(t, "resampler", out, wantEmb)
				}
				nr := alone()
				t.Logf("piece %d (%dx%d patches): tower NMSE %.3e, end to end %.3e, resampler alone %.3e",
					i, gp.GW, gp.GH, nv, ne, nr)
				if nv > vitBound || ne > embBound || nr > rsBound {
					t.Errorf("piece %d: tower %.3e (bound %.0e), end to end %.3e (bound %.0e), "+
						"resampler alone %.3e (bound %.0e)", i, nv, vitBound, ne, embBound, nr, rsBound)
				}
				// The violations, each a step this tower and projector add.
				for _, f := range []struct {
					name  string
					fault towerFault
					tower bool // measured end to end rather than on the resampler alone
				}{
					{"positions read without buckets", faultNoBuckets, true},
					{"keys without their 2-D position", faultNoKeyPos, false},
					{"queries without their LayerNorm", faultQueryRaw, false},
				} {
					s.vis.fault = f.fault
					s.vis.rs.qReady = false
					var nb float64
					if f.tower {
						bad, err := s.EncodeGrid(want, gp.GH, gp.GW)
						if err != nil {
							t.Fatal(err)
						}
						nb = visNMSE(t, f.name, append([]float32(nil), bad...), wantEmb)
					} else {
						nb = alone()
					}
					s.vis.fault = towerFaultNone
					s.vis.rs.qReady = false
					t.Logf("violation, %s: NMSE %.3e", f.name, nb)
					// A key's position moves the answer little -- the queries
					// attend mostly by content -- so that violation is held to
					// the piece's own clean reading rather than to ten bounds.
					floor := 10 * embBound
					if !f.tower {
						floor = max(rsBound, 3*nr)
					}
					if nb < floor {
						t.Errorf("violation %q reads %.3e, under %.1e: the gate does not see it", f.name, nb, floor)
					}
				}
			}
		})
	}
}

// TestMiniCPMVPieceOrderAndNormalisation is the cut's own violations: the
// slices in the wrong order, and pixels the tower was not trained on. Both
// run the whole picture path against the processor's embeddings.
func TestMiniCPMVPieceOrderAndNormalisation(t *testing.T) {
	m := openMiniCPMV(t)
	defer m.Close()
	tw := m.Tower()
	c := tw.Cfg
	g := readMcvGolden(t, "photo896x448")
	img := decodeImage(t, mcvPhoto)
	lay, err := tw.Layout(g.W, g.H)
	if err != nil {
		t.Fatal(err)
	}
	if lay.Rows*lay.Cols < 2 {
		t.Fatalf("the photo cuts into %dx%d slices; the order gate needs two", lay.Cols, lay.Rows)
	}
	want := make([][]float32, len(g.Pieces))
	for i := range want {
		want[i] = readF32n(t, filepath.Join(mcvGold, fmt.Sprintf("photo896x448.p%d.emb", i)), c.Tokens()*c.ProjDim)
	}
	run := func(f towerFault) []float64 {
		s := tw.testState()
		defer s.Close()
		tw.Cfg.fault = f
		pieces, err := tw.PicturePieces(img, lay)
		tw.Cfg.fault = towerFaultNone
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float64, len(pieces))
		for i, p := range pieces {
			emb, err := s.encodePicture(p)
			if err != nil {
				t.Fatal(err)
			}
			out[i] = visNMSE(t, fmt.Sprint("piece ", i), emb, want[i])
		}
		return out
	}
	clean := run(towerFaultNone)
	t.Logf("clean, from the PNG through every step: NMSE per piece %.3e", clean)
	// TestMiniCPMVMatchesTransformers' end-to-end bound: the same pieces, here
	// reached from the PNG instead of from the processor's pixels.
	for _, n := range clean {
		if n > 3e-2 {
			t.Fatalf("the clean picture path reads %.3e", n)
		}
	}
	for _, f := range []struct {
		name  string
		fault towerFault
	}{{"slices reversed", faultSliceOrder}, {"no mean or std", faultNoNorm}} {
		bad := run(f.fault)
		t.Logf("violation, %s: NMSE per piece %.3e", f.name, bad)
		worst := 0.0
		for _, n := range bad {
			worst = max(worst, n)
		}
		if worst < 0.1 {
			t.Errorf("violation %q: worst piece %.3e, which the gate cannot tell from clean", f.name, worst)
		}
	}
}

// TestMiniCPMVOnDeviceMatchesHost places EVERY tower block on each backend and
// holds the projector's input to the host's, on a grid that is not the
// square one: a device that only ran the square slice would be untested here.
func TestMiniCPMVOnDeviceMatchesHost(t *testing.T) {
	m := openMiniCPMV(t)
	defer m.Close()
	tw := m.Tower()
	c := tw.Cfg
	host := tw.testState()
	px, gh, gw := mcvOverview(t, tw, host) // the overview: 45x23 patches
	encode := func(s *State) ([]float32, []float32) {
		var vit []float32
		s.vis.beforeProj = func(x []float32) { vit = append(vit[:0], x...) }
		emb, err := s.EncodeGrid(px, gh, gw)
		if err != nil {
			t.Fatal(err)
		}
		return vit, append([]float32(nil), emb...)
	}
	hv, he := encode(host)
	host.Close()
	ran := 0
	for _, spec := range stepDevices() {
		gd, err := openTier(t, tier.WithDevices(spec))
		if err != nil || gd == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer gd.Close()
			s := tw.testState()
			defer s.Close()
			s.SetDevice(gd)
			if s.GPUBlocks() != c.NLayer {
				t.Fatalf("%s took %d of %d tower blocks (%s)", spec, s.GPUBlocks(), c.NLayer, gd.Err())
			}
			dv, de := encode(s)
			nv := visNMSE(t, "tower", dv, hv)
			ne := visNMSE(t, "embeddings", de, he)
			t.Logf("%s, %d of %d blocks on the device, a %dx%d grid: tower NMSE %.3e, embeddings %.3e",
				spec, s.GPUBlocks(), c.NLayer, gw, gh, nv, ne)
			// Both tiers quantize activations to int8 and round differently,
			// compounding over 27 blocks (TestTowerOnDeviceMatchesHost bounds
			// one block); the violations below read orders above this.
			if nv > 1e-2 || ne > 1e-2 {
				t.Fatalf("%s: tower %.3e, embeddings %.3e -- the device is not running this tower", spec, nv, ne)
			}
			// And one block alone, before the roundings compound: the bound
			// that sees a wrong graph on this grid.
			after0 := func(s *State) []float32 {
				var x0 []float32
				s.vis.afterBlock = func(li int, x []float32) {
					if li == 0 && x0 == nil {
						x0 = append([]float32(nil), x...)
					}
				}
				if _, err := s.EncodeGrid(px, gh, gw); err != nil {
					t.Fatal(err)
				}
				s.vis.afterBlock = nil
				return x0
			}
			hs := tw.testState()
			h0 := after0(hs)
			hs.Close()
			defer func(was int) { tw.opt.towerGPULayers = was }(tw.opt.towerGPULayers)
			tw.opt.towerGPULayers = 1
			one := tw.testState()
			defer one.Close()
			one.SetDevice(gd)
			n0 := visNMSE(t, "block 0", after0(one), h0)
			t.Logf("%s, one block placed: NMSE %.3e after it", spec, n0)
			if n0 > 5e-5 {
				t.Fatalf("%s: NMSE %.3e after one block -- the device is not running this block's graph", spec, n0)
			}
		})
	}
	if ran == 0 {
		t.Skip("no GPU backend on this host")
	}
}

// TestMiniCPMVDescribesAnImage asks the model about the photo, and about a
// blank picture through the identical path: if both answers name the photo,
// the embeddings are not what produced the answer.
func TestMiniCPMVDescribesAnImage(t *testing.T) {
	m := openMiniCPMV(t)
	defer m.Close()
	tw := m.Tower()
	img := decodeImage(t, mcvPhoto)
	describe := func(img image.Image) string {
		lay, err := tw.Layout(img.Bounds().Dx(), img.Bounds().Dy())
		if err != nil {
			t.Fatal(err)
		}
		pieces, err := tw.PicturePieces(img, lay)
		if err != nil {
			t.Fatal(err)
		}
		pic, err := m.PictureSpans(lay, 0, true, pieces)
		if err != nil {
			t.Fatal(err)
		}
		msgs := []ChatMessage{{Role: "user", Content: "Describe this image in one sentence.", Images: 1}}
		spans, err := m.ChatSpansParts(msgs, [][]Span{pic}, true)
		if err != nil {
			t.Fatal(err)
		}
		st := m.NewState(SpanPositions(spans, m.Cfg.NEmbd) + 48)
		defer st.Close()
		logits, err := st.PrefillMixed(spans...)
		if err != nil {
			t.Fatal(err)
		}
		var out []int32
		for i := 0; i < 40; i++ {
			next := Greedy(logits)
			if m.Vocab.IsEOG(next) {
				break
			}
			out = append(out, next)
			if logits, err = st.Forward(next); err != nil {
				t.Fatal(err)
			}
		}
		return strings.ToLower(m.Vocab.Decode(out))
	}
	got := describe(img)
	t.Logf("jitllm: %q", got)
	blank := image.NewRGBA(img.Bounds())
	none := describe(blank)
	t.Logf("blank:  %q", none)
	if !namesThePhoto(got) {
		t.Errorf("the answer names nothing in the photo (a newspaper, the New York Times, the moon): %q", got)
	}
	if namesThePhoto(none) {
		t.Errorf("a BLANK picture is described as the photo too (%q): the embeddings are not what "+
			"produced the answer", none)
	}
}

// namesThePhoto reports whether an answer is about mcvPhoto: the New York
// Times front page of 21 July 1969, "MEN WALK ON MOON". llama.cpp b10825 on
// MiniCPM-V 4.0 answers "A newspaper headline that says Men Walk on Moon."
func namesThePhoto(s string) bool {
	s = strings.ToLower(s)
	for _, w := range []string{"newspaper", "new york times", "moon", "front page"} {
		if strings.Contains(s, w) {
			return true
		}
	}
	return false
}
