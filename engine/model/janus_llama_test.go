package model

import "testing"

// TestJanusTowerMatchesLlamaCpp holds the tower and aligner node by node to
// llama-mtmd-debug b10825 on its 384 checkerboard:
//
//	llama-mtmd-debug -m Janus-Pro-1B-Q4_K_M.gguf --mmproj mmproj-Janus-Pro-1B-Q8_0.gguf \
//	  --image cb -n 384 -p encode
//	python3 scripts/mtmddump.py DUMP testdata/golden/janus-mtmd-cb384.json \
//	  pos_embed layer_out-0 layer_out-11 layer_out-23 node_765
//
// The mmproj is Q8_0, so both engines quantize the activations here (ggml's
// q8_0 dot), each with its own rounding, compounding over 24 blocks.
func TestJanusTowerMatchesLlamaCpp(t *testing.T) {
	m := openJanus(t)
	defer m.Close()
	tw := m.Tower()
	c := tw.Cfg
	ref := readMtmdNodes(t, "janus-mtmd-cb384.json")
	s := tw.testState()
	defer s.Close()
	got := map[string][]float32{}
	s.vis.afterBlock = func(li int, x []float32) {
		switch li {
		case -1:
			got["pos_embed"] = append([]float32(nil), x...)
		case 0:
			got["layer_out-0"] = append([]float32(nil), x...)
		case 11:
			got["layer_out-11"] = append([]float32(nil), x...)
		case 23:
			got["layer_out-23"] = append([]float32(nil), x...)
		}
	}
	out, err := s.Encode(towerCb(c.ImageSz))
	if err != nil {
		t.Fatal(err)
	}
	got["node_765"] = append([]float32(nil), out...)
	for _, n := range []struct {
		name  string
		width int
		bar   float64
	}{
		{"pos_embed", c.NEmbd, 0.01}, {"layer_out-0", c.NEmbd, 0.03}, {"layer_out-11", c.NEmbd, 0.05},
		{"layer_out-23", c.NEmbd, 0.10}, {"node_765", c.ProjDim, 0.12},
	} {
		x := got[n.name]
		if x == nil {
			t.Fatalf("%s was not captured", n.name)
		}
		rel, sr := compareMtmdNode(t, n.name, x, n.width, ref[n.name])
		t.Logf("%-13s relative RMS %.4f over 36 values, sum off by %.4f of sum|x|", n.name, rel, sr)
		if rel > n.bar || sr > n.bar {
			t.Errorf("%s: relative RMS %.4f, sum %.4f (bar %.2f)", n.name, rel, sr, n.bar)
		}
	}
}

// janusLlamaAnswer is llama-mtmd-cli b10825's greedy answer about mcvPhoto
// (--jinja: the GGUF's template reads typed content), its prompt
// "<|User|>: <__media__>Describe this image in one sentence.\n\n<|Assistant|>:"
// after a BOS, 591 positions.
const janusLlamaAnswer = "This image is a scanned cover of The New York Times from June 20, 1960, " +
	"reporting on astronauts landing on a plain, collecting rocks"

// TestJanusTeacherForcedMatchesLlamaCpp feeds llama.cpp's prompt -- which
// wraps the 576 embeddings in nothing, where the processor puts
// <begin_of_image> and <end_of_image> around them -- and its answer.
func TestJanusTeacherForcedMatchesLlamaCpp(t *testing.T) {
	m := openJanus(t)
	defer m.Close()
	tw := m.Tower()
	img := decodeImage(t, mcvPhoto())
	s := tw.testState()
	px, err := s.PreprocessImage(img)
	if err != nil {
		t.Fatal(err)
	}
	emb, err := s.Encode(px)
	if err != nil {
		t.Fatal(err)
	}
	emb = append([]float32(nil), emb...)
	s.Close()
	bos := m.container.Vocab().BOS
	spans := []Span{
		{Tokens: append([]int32{bos}, m.Vocab.EncodeSpecial("<|User|>: ", false)...)},
		{Embd: emb},
		{Tokens: m.Vocab.EncodeSpecial("Describe this image in one sentence.\n\n<|Assistant|>:", false)},
	}
	if n := SpanPositions(spans, m.Cfg.NEmbd); n != 591 {
		t.Errorf("the prompt is %d positions; llama.cpp's was 591", n)
	}
	ref := m.Vocab.Encode(janusLlamaAnswer, false)
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
