package model

import "testing"

// mcvLlamaAnswer is llama-mtmd-cli b10825's greedy answer about mcvPhoto:
//
//	llama-mtmd-cli -m MiniCPM-V-4-Q4_K_M.gguf --mmproj mmproj-MiniCPM-V-4-f16.gguf \
//	  --image photo896x448.png -p "Describe this image in one sentence." \
//	  --temp 0 --top-k 1 -dev none -fa off
const mcvLlamaAnswer = "A newspaper headline that says Men Walk on Moon."

// TestMiniCPMVTeacherForcedMatchesLlamaCpp feeds llama.cpp's own prompt --
// BOS, the turn, the overview and two slices in mtmd's markers (no image id),
// the question -- and its answer, token by token, and asks at every position
// whether jitllm's argmax is llama.cpp's token. The picture's pixels are the
// same in both (a 896x448 picture cuts and resamples identically there and
// here), so what is compared is the tower, the resampler and the text model.
func TestMiniCPMVTeacherForcedMatchesLlamaCpp(t *testing.T) {
	m := openMiniCPMV(t)
	defer m.Close()
	tw := m.Tower()
	img := decodeImage(t, mcvPhoto)
	lay, err := tw.Layout(img.Bounds().Dx(), img.Bounds().Dy())
	if err != nil {
		t.Fatal(err)
	}
	pieces, err := tw.PicturePieces(img, lay)
	if err != nil {
		t.Fatal(err)
	}
	pic, err := m.PictureSpans(lay, 0, false, pieces)
	if err != nil {
		t.Fatal(err)
	}
	bos := m.container.Vocab().BOS
	spans := []Span{{Tokens: append([]int32{bos}, m.Vocab.EncodeSpecial("<|im_start|>user\n", false)...)}}
	spans = append(spans, pic...)
	spans = append(spans, Span{Tokens: m.Vocab.EncodeSpecial(
		"Describe this image in one sentence.<|im_end|>\n<|im_start|>assistant\n", false)})
	if n := SpanPositions(spans, m.Cfg.NEmbd); n != 214 {
		t.Errorf("the prompt is %d positions; llama.cpp's was 214", n)
	}
	ref := m.Vocab.Encode(mcvLlamaAnswer, false)
	st := m.NewInstance(SpanPositions(spans, m.Cfg.NEmbd) + len(ref) + 2)
	defer st.Close()
	logits, err := st.PrefillMixed(spans...)
	if err != nil {
		t.Fatal(err)
	}
	agree, ties := 0, 0
	for i, want := range ref {
		got := Greedy(logits)
		switch margin := logits[got] - logits[want]; {
		case got == want:
			agree++
		case margin < 0.5:
			ties++
			t.Logf("position %d: jitllm %q, llama.cpp %q, a %.3f tie", i, m.Vocab.Decode([]int32{got}),
				m.Vocab.Decode([]int32{want}), margin)
		default:
			t.Errorf("position %d: jitllm %q, llama.cpp %q, by %.3f", i, m.Vocab.Decode([]int32{got}),
				m.Vocab.Decode([]int32{want}), margin)
		}
		if logits, err = st.Forward(want); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("%d of %d teacher-forced positions agree with llama.cpp, %d ties", agree, len(ref), ties)
}
