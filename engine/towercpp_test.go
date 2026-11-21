package engine

import (
	"image"
	"math"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestSmolVLMTowerMatchesLlamaCpp checks SmolVLM-256M's stated tower against
// the GGUF oracle, beside its reference (transformers' class, which
// TestSmolVLMVideoMatchesTransformers holds the tower's rows to): the GGUF
// oracle's own graph dump of its projector output on a checkerboard:
//
//	llama-mtmd-debug -m SmolVLM-256M-Instruct-Q8_0.gguf \
//	  --mmproj mmproj-SmolVLM-256M-Instruct-Q8_0.gguf --image cb -n 512 -p encode
//
// The debug tool builds the picture in memory as raw samples, 1 and 0 on every
// channel, and normalises nothing, so the container is converted with its
// normalisation restated to the identity (convert.WithPictureStep) and given
// the checkerboard four squares a side, whose first tile -- a crop of the
// picture every resize of the processor's leaves as it is -- is the 64 rows the
// GGUF oracle encodes. The dump prints three leading and three trailing values
// of rows 0, 1, 2, 61, 62 and 63 and the whole tensor's sum: the samples catch
// a layout bug (the patch order, the shuffle's), the sum sees every element.
// The bar is relative RMS: the int8 activations here against the GGUF oracle's
// f32/f16 compound to ~1.5% over twelve blocks, a permutation to O(1). The
// shipping normalisation, which the dump does not apply, is the violation.
func TestSmolVLMTowerMatchesLlamaCpp(t *testing.T) {
	rows := [6]int{0, 1, 2, 61, 62, 63}
	vals := [6][6]float64{
		{2.9709, -0.2394, 7.9154, -0.0333, 3.9492, -6.2315},
		{2.3362, 1.1205, 4.9934, 1.0887, 3.3559, -5.4497},
		{2.0612, 0.4978, 5.8252, -0.2149, 3.9973, -5.3926},
		{8.0296, 3.1371, 5.3551, 4.4943, 8.4025, -3.6975},
		{10.8150, 1.2275, 8.8473, 2.4137, 12.9995, -7.4657},
		{15.4850, 4.6529, 1.5212, 3.8733, 7.9017, 0.0195},
	}
	const sum = 1605.572876
	text, mm := testmodels.Path("SmolVLM-256M-Instruct-Q8_0.gguf"), testmodels.Path("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf")
	if !sourcePresent(text, mm) {
		testmodels.Missing(t, "MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", text, mm)
	}
	found := false
	identity := convert.WithPictureStep(func(s *jlm.PreStep) {
		if s.Kind == jlm.PreNormalize {
			s.Floats = []float32{0, 0, 0, 1, 1, 1}
			found = true
		}
	})
	raw := filepath.Join(t.TempDir(), "smolvlm-raw"+jlm.Ext)
	if _, err := convert.FromGGUFs(text, mm, raw, jlm.Fingerprint{Host: "test", Writer: BuildIDForTest()}, identity); err != nil {
		t.Fatal(err)
	}
	if !found {
		t.Fatal("SmolVLM's processor states no normalisation to restate")
	}
	// rel is the relative RMS of the tower's projector output on the
	// checkerboard against the dump's samples, and its sum's distance.
	rel := func(path string) (float64, float64) {
		t.Helper()
		m, err := model.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		// Four squares a side: the processor's resizes of the whole picture are
		// the identity there and its first tile is the checkerboard's corner as
		// it is, the picture the GGUF oracle encodes. At the tower's own side
		// the picture would be resized up and down again before its global view
		// is cut.
		side := 4 * int(m.Container.Vision().ImageSize)
		cb := image.NewNRGBA(image.Rect(0, 0, side, side))
		for y := range side {
			for x := range side {
				v := uint8(0)
				if (x+y)%2 == 0 {
					v = 255
				}
				i := cb.PixOffset(x, y)
				cb.Pix[i], cb.Pix[i+1], cb.Pix[i+2], cb.Pix[i+3] = v, v, v, 255
			}
		}
		s := testInstance(t, m, 64)
		defer s.Close()
		// The tile's samples are the checkerboard's two levels, half each,
		// whatever the normalisation makes of them: a resize would leave
		// levels between.
		px, _ := towerPixelsAt(t, s, cb, 0)
		levels := map[float32]int{}
		for _, v := range px {
			levels[v]++
		}
		if len(levels) != 2 || 2*levels[px[0]] != len(px) {
			t.Fatalf("the first tile's samples take %d levels (%v): the picture was resized before it was cut, and is not the GGUF oracle's", len(levels), levels)
		}
		out, err := s.EncodePicture(cb)
		if err != nil {
			t.Fatal(err)
		}
		pd := m.Cfg.NEmbd
		// The first tile's rows follow its markers.
		sq, _, err := m.Tower.Square(side, side, 0)
		if err != nil {
			t.Fatal(err)
		}
		at, k := sq.MarkN, m.Tower.SquareOut(sq)-sq.MarkN-sq.TrailN
		if k != 64 || len(out) < (at+k)*pd {
			t.Fatalf("the first tile is %d rows after %d markers of the picture's %d values", k, at, len(out))
		}
		out = out[at*pd : (at+k)*pd]
		var sse, sy2, total float64
		for i, row := range rows {
			for j := range 6 {
				col := j
				if j >= 3 {
					col = pd - 6 + j
				}
				got, want := float64(out[row*pd+col]), vals[i][j]
				sse += (got - want) * (got - want)
				sy2 += want * want
			}
		}
		for _, v := range out {
			total += float64(v)
		}
		if math.IsNaN(sse) || math.IsNaN(total) {
			t.Fatalf("a non-finite output: sse %g, sum %g", sse, total)
		}
		return math.Sqrt(sse / sy2), math.Abs(total-sum) / math.Abs(sum)
	}
	r, d := rel(raw)
	t.Logf("against llama-mtmd-debug: relative RMS %.4f over 36 sampled values, the sum %.2f%% off", r, d*100)
	if r > 0.05 || d > 0.05 {
		t.Errorf("the projector output differs from the GGUF oracle's: relative RMS %.4f, the sum %.2f%% off (bound 5%%)", r, d*100)
	}
	vr, vd := rel(jlmOfPair(t, text, mm))
	t.Logf("violation, SmolVLM's own normalisation: relative RMS %.4f, the sum %.2f%% off", vr, vd*100)
	if vr <= 0.05 && vd <= 0.05 {
		t.Errorf("VIOLATION: the normalised checkerboard reads %.4f and %.2f%%, inside the bound: the gate cannot see the input", vr, vd*100)
	}
}

// TestTinyOpenJevTowerMatchesLlamaCpp checks tinyopenjev-for-testing's tower
// -- 144 wide, not a whole number of Q8 blocks, so each of its blocks reads
// its rows widened with zeros by matrices stored 160 inputs wide -- against
// the GGUF oracle's graph dump of its projector output on a checkerboard:
//
//	llama-mtmd-debug -m tinyopenjev-for-testing-Q8_0.gguf \
//	  --mmproj mmproj-tinyopenjev-for-testing-Q8_0.gguf --image cb -n 512 -p encode
//
// The debug tool builds the 512x512 picture as raw samples, 1 and 0 on every
// channel, and normalises nothing, so the container is converted with its
// normalisation restated to the identity, as TestSmolVLMTowerMatchesLlamaCpp
// does. The dump prints three leading and three trailing values of rows 0,
// 1, 2, 253, 254 and 255 of the merger's 256 and the whole tensor's sum. The
// bar is SmolVLM's; the shipping normalisation is the violation.
func TestTinyOpenJevTowerMatchesLlamaCpp(t *testing.T) {
	rows := [6]int{0, 1, 2, 253, 254, 255}
	vals := [6][6]float64{
		{-0.0350, -0.0414, 0.0519, -0.0061, 0.0113, 0.0431},
		{-0.0458, -0.0532, 0.0408, -0.0251, 0.0027, 0.0594},
		{-0.0515, -0.0464, 0.0554, -0.0441, -0.0041, 0.0568},
		{0.0510, -0.0092, 0.0505, -0.0405, -0.0399, 0.0505},
		{0.0513, -0.0191, 0.0343, -0.0390, -0.0464, 0.0618},
		{0.0472, -0.0433, 0.0400, -0.0444, -0.0420, 0.0539},
	}
	const sum = 121.839928
	text := testmodels.Path(filepath.Join("openjev", "tinyopenjev-for-testing-Q8_0.gguf"))
	mm := testmodels.Path(filepath.Join("openjev", "mmproj-tinyopenjev-for-testing-Q8_0.gguf"))
	if !sourcePresent(text, mm) {
		testmodels.Missing(t, "MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", text, mm)
	}
	found := false
	identity := convert.WithPictureStep(func(s *jlm.PreStep) {
		if s.Kind == jlm.PreNormalize {
			s.Floats = []float32{0, 0, 0, 1, 1, 1}
			found = true
		}
	})
	raw := filepath.Join(t.TempDir(), "tinyopenjev-raw"+jlm.Ext)
	if _, err := convert.FromGGUFs(text, mm, raw, jlm.Fingerprint{Host: "test", Writer: BuildIDForTest()}, identity); err != nil {
		t.Fatal(err)
	}
	if !found {
		t.Fatal("tinyopenjev's processor states no normalisation to restate")
	}
	rel := func(path string) (float64, float64) {
		t.Helper()
		m, err := model.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		if w := m.Container.Vision().NEmbd; w%32 == 0 {
			t.Fatalf("the tower is %d wide, a whole number of blocks: this gate reads no padded row", w)
		}
		const side = 512
		cb := image.NewNRGBA(image.Rect(0, 0, side, side))
		for y := range side {
			for x := range side {
				l := uint8(0)
				if (x+y)%2 == 0 {
					l = 255
				}
				i := cb.PixOffset(x, y)
				cb.Pix[i], cb.Pix[i+1], cb.Pix[i+2], cb.Pix[i+3] = l, l, l, 255
			}
		}
		s := testInstance(t, m, 64)
		defer s.Close()
		px, _ := towerPixelsAt(t, s, cb, 0)
		levels := map[float32]int{}
		for _, p := range px {
			levels[p]++
		}
		if len(levels) != 2 || 2*levels[px[0]] != len(px) {
			t.Fatalf("the picture's samples take %d levels (%v): it was resized, and is not the GGUF oracle's", len(levels), levels)
		}
		out, err := s.EncodePicture(cb)
		if err != nil {
			t.Fatal(err)
		}
		pd := m.Cfg.NEmbd
		if len(out) != 256*pd {
			t.Fatalf("the picture is %d values, the GGUF oracle's 256 rows of %d", len(out), pd)
		}
		var sse, sy2, total float64
		for i, row := range rows {
			for j := range 6 {
				col := j
				if j >= 3 {
					col = pd - 6 + j
				}
				got, want := float64(out[row*pd+col]), vals[i][j]
				sse += (got - want) * (got - want)
				sy2 += want * want
			}
		}
		for _, x := range out {
			total += float64(x)
		}
		if math.IsNaN(sse) || math.IsNaN(total) {
			t.Fatalf("a non-finite output: sse %g, sum %g", sse, total)
		}
		return math.Sqrt(sse / sy2), math.Abs(total-sum) / math.Abs(sum)
	}
	r, d := rel(raw)
	t.Logf("against llama-mtmd-debug: relative RMS %.4f over 36 sampled values, the sum %.2f%% off", r, d*100)
	if r > 0.05 || d > 0.05 {
		t.Errorf("the projector output differs from the GGUF oracle's: relative RMS %.4f, the sum %.2f%% off (bound 5%%)", r, d*100)
	}
	vr, vd := rel(jlmOfPair(t, text, mm))
	t.Logf("violation, tinyopenjev's own normalisation: relative RMS %.4f, the sum %.2f%% off", vr, vd*100)
	if vr <= 0.05 && vd <= 0.05 {
		t.Errorf("VIOLATION: the normalised checkerboard reads %.4f and %.2f%%, inside the bound: the gate cannot see the input", vr, vd*100)
	}
}

// TestMoondreamTowerMatchesLlamaCpp checks moondream2's stated tower against
// the GGUF oracle's graph dump of its projector output on a checkerboard:
//
//	llama-mtmd-debug -m moondream2-text-model-f16_ct-vicuna.gguf \
//	  --mmproj moondream2-mmproj-f16-20250414.gguf --image cb -n 378 -p encode
//
// The debug tool builds the 378x378 picture as raw samples, 1 and 0 on every
// channel, and normalises nothing, so the container is converted with its
// normalisation restated to the identity, as TestSmolVLMTowerMatchesLlamaCpp
// does; the processor's resize and crop are the identity at the tower's side.
// The dump prints three leading and three trailing values of rows 0, 1, 2,
// 726, 727 and 728 of the 729 and the whole tensor's sum. The file states 28
// blocks and its block 27 is block 26 byte for byte, padding the GGUF oracle
// never runs (convert's dropPaddedBlock): run as a block, the output reads a
// relative RMS of 0.68 and a sum 21% off. Without it the int8 activations
// here against the GGUF oracle's f32/f16 compound over 27 blocks to 0.052
// and 6.0% on amd64 (SmolVLM's twelve read ~1.5%, InternVL's 4.6% on
// amd64 and 6.7% on the M4), so the bar is 8% and 10%, under a tenth of the
// padding's. The shipping normalisation is the violation.
func TestMoondreamTowerMatchesLlamaCpp(t *testing.T) {
	rows := [6]int{0, 1, 2, 726, 727, 728}
	vals := [6][6]float64{
		{0.3823, 1.1614, 1.3537, 1.7729, 0.2392, -0.7965},
		{-5.3397, -10.7389, 6.3170, 1.0830, -2.5277, -2.0037},
		{-5.2632, -9.0741, 5.6166, 0.1460, -2.3256, -1.1393},
		{-3.3217, 0.0770, 5.8415, 8.7523, -2.6976, -8.1462},
		{-0.4842, 1.3990, 1.4011, 1.6893, 0.6980, -4.3346},
		{0.0138, -1.0643, 1.7534, 2.8735, -1.3090, -2.5282},
	}
	const sum = 2974.681641
	text := testmodels.Path("moondream2/moondream2-text-model-f16_ct-vicuna.gguf")
	mm := testmodels.Path("moondream2/moondream2-mmproj-f16-20250414.gguf")
	if !sourcePresent(text, mm) {
		testmodels.Missing(t, "MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", text, mm)
	}
	found := false
	identity := convert.WithPictureStep(func(s *jlm.PreStep) {
		if s.Kind == jlm.PreNormalize {
			s.Floats = []float32{0, 0, 0, 1, 1, 1}
			found = true
		}
	})
	raw := filepath.Join(t.TempDir(), "moondream2-raw"+jlm.Ext)
	if _, err := convert.FromGGUFs(text, mm, raw, jlm.Fingerprint{Host: "test", Writer: BuildIDForTest()}, identity); err != nil {
		t.Fatal(err)
	}
	if !found {
		t.Fatal("moondream2's processor states no normalisation to restate")
	}
	rel := func(path string) (float64, float64) {
		t.Helper()
		m, err := model.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		if n := m.Container.Vision().NLayer; n != 27 {
			t.Fatalf("the tower runs %d blocks; moondream2's has 27 and a padding block", n)
		}
		side := int(m.Container.Vision().ImageSize)
		cb := image.NewNRGBA(image.Rect(0, 0, side, side))
		for y := range side {
			for x := range side {
				v := uint8(0)
				if (x+y)%2 == 0 {
					v = 255
				}
				i := cb.PixOffset(x, y)
				cb.Pix[i], cb.Pix[i+1], cb.Pix[i+2], cb.Pix[i+3] = v, v, v, 255
			}
		}
		s := testInstance(t, m, 64)
		defer s.Close()
		out, err := s.EncodePicture(cb)
		if err != nil {
			t.Fatal(err)
		}
		pd := m.Cfg.NEmbd
		if len(out) != 729*pd {
			t.Fatalf("%d values, not 729 rows of %d", len(out), pd)
		}
		var sse, sy2, total float64
		for i, row := range rows {
			for j := range 6 {
				col := j
				if j >= 3 {
					col = pd - 6 + j
				}
				got, want := float64(out[row*pd+col]), vals[i][j]
				sse += (got - want) * (got - want)
				sy2 += want * want
			}
		}
		for _, v := range out {
			total += float64(v)
		}
		if math.IsNaN(sse) || math.IsNaN(total) {
			t.Fatalf("a non-finite output: sse %g, sum %g", sse, total)
		}
		return math.Sqrt(sse / sy2), math.Abs(total-sum) / math.Abs(sum)
	}
	const rmsBound, sumBound = 0.08, 0.10
	r, d := rel(raw)
	t.Logf("against llama-mtmd-debug: relative RMS %.4f over 36 sampled values, the sum %.2f%% off", r, d*100)
	if r > rmsBound || d > sumBound {
		t.Errorf("the projector output differs from the GGUF oracle's: relative RMS %.4f, the sum %.2f%% off (bounds %.0f%% and %.0f%%)",
			r, d*100, rmsBound*100, sumBound*100)
	}
	vr, vd := rel(jlmOfPair(t, text, mm))
	t.Logf("violation, moondream2's own normalisation: relative RMS %.4f, the sum %.2f%% off", vr, vd*100)
	if vr <= rmsBound && vd <= sumBound {
		t.Errorf("VIOLATION: the normalised checkerboard reads %.4f and %.2f%%, inside the bound: the gate cannot see the input", vr, vd*100)
	}
}
