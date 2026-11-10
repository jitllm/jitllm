package engine

import (
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestBidirRunIsNeverCut prefills Gemma 3 4B's picture, 256 rows attending
// both ways, behind a prompt that starts it off a chunk boundary, in 64-row
// chunks and in one chunk of 512: the chunk the picture starts in ends where
// the picture does, so the two agree. The violation lets the chunk end
// inside the picture, and the logits part.
func TestBidirRunIsNeverCut(t *testing.T) {
	text := testmodels.Path("gemma-3-4b-it-Q4_K_M.gguf")
	mm := testmodels.Path("vlm/mmproj-gemma-3-4b-it-f16.gguf")
	if !sourcePresent(text, mm) {
		testmodels.Missing(t, "MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", text, mm)
	}
	m, err := model.Open(jlmOfPair(t, text, mm))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	f, err := os.Open(filepath.Join("testdata", "quad-and-disc.png"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	pspan, err := m.PictureSpan(img)
	if err != nil {
		t.Fatal(err)
	}
	ask := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 9) + "Describe this image."
	spans, err := m.ChatSpans([]model.ChatMessage{{Role: "user", Content: ask, Images: 1}}, []model.Span{pspan}, true)
	if err != nil {
		t.Fatal(err)
	}
	at, runs := 0, 0
	for _, sp := range spans {
		if sp.Picture != nil {
			if !sp.Bidir {
				t.Fatal("the picture's span is not a bidirectional run")
			}
			runs++
			break
		}
		at += sp.N(m.Cfg.NEmbd)
	}
	if runs != 1 || at%64 == 0 || at/64 == (at+m.Tower.Out)/64 {
		t.Fatalf("the picture starts at %d (%d runs): no 64-row chunk boundary falls inside it", at, runs)
	}
	logits := func(chunk int, cut bool) []float32 {
		s := testInstance(t, m, model.SpanPositions(spans, m.Cfg.NEmbd)+1, WithPrefillChunk(chunk))
		defer s.Close()
		s.bidirCutFault = cut
		lg, err := s.PrefillMixed(spans...)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), lg...)
	}
	whole := logits(512, false)
	small := logits(64, false)
	nm, worst := nmseOf(small, whole)
	t.Logf("the picture at %d: 64-row chunks against 512, logit NMSE %.3e, max|d| %.4f", at, nm, worst)
	if math.IsNaN(nm) || nm > 1e-4 {
		t.Errorf("64-row chunks read NMSE %.3e against one chunk", nm)
	}
	vn, vw := nmseOf(logits(64, true), whole)
	t.Logf("violation, the picture cut at the chunk: NMSE %.3e, max|d| %.4f", vn, vw)
	if math.IsNaN(vn) || vn < 10*max(nm, 1e-6) {
		t.Errorf("a cut picture reads NMSE %.3e against a clean %.3e: the gate cannot see the cut", vn, nm)
	}
}
