//go:build amd64 && linux && jitllmtest

package engine

import (
	"image"
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/model"
)

// TestEveryTowerRunsSSE encodes a picture through every stated tower on the
// forced SSE tier (cpu:sse, the instance asserted to run there) and holds its
// rows to the AVX2 tier's in the same process, within the CPU device's band:
// the two tiers are two transcriptions, not one (RULE 11d). A grey picture's
// rows on the SSE tier, past ten times the band, are the violation.
func TestEveryTowerRunsSSE(t *testing.T) {
	img := principlePicture()
	grey := image.NewGray(img.Bounds())
	for i := range grey.Pix {
		grey.Pix[i] = 128
	}
	const bound = 1e-3
	for _, src := range towerFixtures(t) {
		t.Run(src.name, func(t *testing.T) {
			path := pairContainer(t, src)
			ssemTook(t, path)
			encode := func(pic image.Image, opts ...Option) []float32 {
				m, err := model.Open(path)
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				s := testInstance(t, m, 64, opts...)
				defer s.Close()
				rows, err := s.EncodePicture(pic)
				if err != nil {
					t.Fatal(err)
				}
				return slices.Clone(rows)
			}
			want := encode(img)
			e, _ := nmseOf(encode(img, sseDevice()), want)
			if bad, _ := nmseOf(encode(grey, sseDevice()), want); !(bad > 10*bound) {
				t.Fatalf("VIOLATION: a grey picture's rows on the SSE tier sit at NMSE %.3e from the picture's, within ten times the bound %.0e", bad, bound)
			}
			if math.IsNaN(e) || e > bound {
				t.Fatalf("the SSE tier's rows sit at NMSE %.3e from the AVX2 tier's (bound %.0e)", e, bound)
			}
			t.Logf("the SSE tier's rows %.3e from the AVX2 tier's", e)
		})
	}
}
