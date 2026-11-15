package engine

import (
	"image"
	"testing"

	"github.com/jitllm/jitllm/model"
)

// TestPrimeSeesThePromptsTail: Prime makes a prompt the history the
// penalties read -- the window holds its last RepeatLastN tokens, a
// negative id none, and whatever the sampler saw before none -- and
// PrimeSpans the token spans' alone, a picture's rows none. A copy primed
// leaves the sampler it was copied from as it was.
func TestPrimeSeesThePromptsTail(t *testing.T) {
	s := Sampler{RepeatPen: 1.3, RepeatLastN: 3}
	s.Observe(9)
	s.Observe(1)
	before := s
	s.Prime([]int32{4, 9, -1, 4, 7, 2})
	got := s.window(12)
	for tok, want := range map[int]float32{4: 1, 7: 1, 2: 1, 9: 0, 1: 0} {
		if got[tok] != want {
			t.Errorf("after Prime, token %d counts %v in the window, want %v", tok, got[tok], want)
		}
	}
	if w := before.window(12); w[9] != 1 || w[1] != 1 {
		t.Errorf("Prime changed the sampler it was copied from: tokens 9 and 1 count %v and %v", w[9], w[1])
	}
	p := Sampler{PresencePen: 0.5, RepeatLastN: 8}
	p.PrimeSpans([]model.Span{{Tokens: []int32{5, 5}}, {Picture: image.NewGray(image.Rect(0, 0, 1, 1)), Tokens: []int32{8}}, {Tokens: []int32{6}}})
	got = p.window(12)
	for tok, want := range map[int]float32{5: 2, 6: 1, 8: 0} {
		if got[tok] != want {
			t.Errorf("after PrimeSpans, token %d counts %v in the window, want %v", tok, got[tok], want)
		}
	}
}
