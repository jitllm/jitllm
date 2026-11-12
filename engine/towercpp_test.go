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
// llama.cpp, an oracle beside its reference (transformers' class, which
// TestSmolVLMVideoMatchesTransformers holds the tower's rows to): llama.cpp's
// own graph dump of its projector output on a checkerboard:
//
//	llama-mtmd-debug -m SmolVLM-256M-Instruct-Q8_0.gguf \
//	  --mmproj mmproj-SmolVLM-256M-Instruct-Q8_0.gguf --image cb -n 512 -p encode
//
// The debug tool builds the picture in memory as raw samples, 1 and 0 on
// every channel, and normalises nothing, so the container is converted with
// its normalisation restated to the identity (convert.WithPictureStep) and
// given the same checkerboard at the tower's size, whose global view -- the
// last of the squares Idefics3's processor cuts it into, the 64 rows llama.cpp
// encodes -- is resized by the identity. The dump prints three leading and three trailing values of rows
// 0, 1, 2, 61, 62 and 63 and the whole tensor's sum: the samples catch a
// layout bug (the patch order, the shuffle's), the sum sees every element.
// The bar is relative RMS: the int8 activations here against llama.cpp's
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
		// The global view is the picture's last square, the checkerboard at
		// its own size: its 64 rows before the one marker that closes it.
		n := len(out) / pd
		if n < 65 || len(out) != n*pd {
			t.Fatalf("the tower writes %d values, not whole rows of %d past the global view's 64 and its marker", len(out), pd)
		}
		out = out[(n-65)*pd : (n-1)*pd]
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
		t.Errorf("the projector output differs from llama.cpp's: relative RMS %.4f, the sum %.2f%% off (bound 5%%)", r, d*100)
	}
	vr, vd := rel(jlmOfPair(t, text, mm))
	t.Logf("violation, SmolVLM's own normalisation: relative RMS %.4f, the sum %.2f%% off", vr, vd*100)
	if vr <= 0.05 && vd <= 0.05 {
		t.Errorf("VIOLATION: the normalised checkerboard reads %.4f and %.2f%%, inside the bound: the gate cannot see the input", vr, vd*100)
	}
}
