package model

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

var qwen2vlText = testmodels.Path("Qwen2-VL-2B-Instruct-Q4_K_M.gguf")
var qwen2vlProj = testmodels.Path("mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf")

func openQwen2VL(t testing.TB) *Model {
	t.Helper()
	if !vlmAvailable(qwen2vlText, qwen2vlProj) {
		t.Skipf("MODEL MISSING: %s, its mmproj and its merged container -- this gate "+
			"proved nothing", qwen2vlText)
	}
	m, err := Open(jlmOfPair(t, qwen2vlText, qwen2vlProj))
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// TestQwen2VLTowerShape pins what the container says the merger is, because
// every one of these is derived from the file's own arithmetic rather than
// defaulted, and a wrong one reshapes the sequence without faulting.
func TestQwen2VLTowerShape(t *testing.T) {
	m := openQwen2VL(t)
	defer m.Close()
	tw := m.Tower()
	if tw == nil {
		t.Fatal("the container carries no tower")
	}
	c := tw.Cfg
	for _, tc := range []struct {
		name      string
		got, want int
	}{
		{"NLayer", c.NLayer, 32},
		{"NEmbd", c.NEmbd, 1280},
		{"NHead", c.NHead, 16},
		{"HeadDim", c.HeadDim, 80},
		{"NFFN", c.NFFN, 5120},
		{"ImageSz", c.ImageSz, 560},
		{"PatchSz", c.PatchSz, 14},
		{"Patches", c.Patches(), 1600},
		// Scale is derived from mm.0's k (5120 over NEmbd 1280 is 4 patches,
		// 2x2); the file has no scale_factor key.
		{"Scale", c.Scale, 2},
		{"Tokens", c.Tokens(), 400},
		{"ProjDim", c.ProjDim, 1536},
	} {
		if tc.got != tc.want {
			t.Errorf("%s = %d, want %d", tc.name, tc.got, tc.want)
		}
	}
	if !c.Rope {
		t.Error("the tower must take ROTARY positions: Qwen2-VL ships no v.position_embd")
	}
	if c.CLS {
		t.Error("Qwen2-VL's ViT has no class token")
	}
	// The projector's output width must be the text model's input width, or the
	// splice is a wrong-sized buffer rather than a wrong answer.
	if c.ProjDim != int(m.Cfg.NEmbd) {
		t.Errorf("the projector emits %d and the text model takes %d", c.ProjDim, m.Cfg.NEmbd)
	}
	t.Logf("qwen2vl tower: %d blocks, %d wide, %d patches -> %d tokens, merger %d->%d",
		c.NLayer, c.NEmbd, c.Patches(), c.Tokens(), c.NEmbd*c.Scale*c.Scale, c.ProjDim)
}

// TestQwen2VLDescribesAnImage is end-to-end because the cheaper oracles are
// degenerate: llama-mtmd-debug's gray and checkerboard images give identical
// patches (the checkerboard period divides the 14-pixel patch), for which the
// tower's exact output is identical rows, and this ViT's ~1e7 activations make
// llama.cpp's rounding-driven rows incomparable. So the input is a real picture
// and the check is what the model says. The reference is llama.cpp on the
// same file:
//
//	llama-mtmd-cli -m Qwen2-VL-2B-Instruct-Q4_K_M.gguf \
//	  --mmproj mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf \
//	  --image engine/model/testdata/quad-and-disc.png \
//	  -p "Describe this image in one sentence." -dev none
//	-> "A yellow circle is placed in the center of a red, blue, and green background."
//
// A blank image is run through the same path as the violation: if both
// answers name the picture, the embeddings are not what produced the answer.
func TestQwen2VLDescribesAnImage(t *testing.T) {
	m := openQwen2VL(t)
	defer m.Close()
	tw := m.Tower()
	if tw == nil {
		t.Fatal("the container carries no tower")
	}
	f, err := os.Open(filepath.Join("testdata", "quad-and-disc.png"))
	if err != nil {
		t.Skipf("IMAGE MISSING: %v -- this gate proved nothing", err)
	}
	real, sz, err := preprocessSized(tw, f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}

	// Qwen2-VL's own wrapping, from the chat template in the GGUF:
	//   <|im_start|>user\n ... <|vision_start|> EMBEDDINGS <|vision_end|> ...
	// <|image_pad|> is NOT emitted: the embedding span is the padding.
	id := func(name string) []int32 {
		v, ok := m.Vocab.ID(name)
		if !ok {
			t.Fatalf("the vocabulary has no %s, so the image markers cannot be built", name)
		}
		return []int32{v}
	}
	pre := append(m.Vocab.Encode("<|im_start|>user\n", false), id("<|vision_start|>")...)
	post := append(id("<|vision_end|>"),
		m.Vocab.Encode("Describe this image in one sentence.<|im_end|>\n<|im_start|>assistant\n", false)...)

	describe := func(px []float32) string {
		ts := tw.testState()
		defer ts.Close()
		if err := ts.SetImageSize(sz[0], sz[1]); err != nil {
			t.Fatal(err)
		}
		emb, err := ts.Encode(px)
		if err != nil {
			t.Fatal(err)
		}
		st := m.NewInstance(len(pre) + len(emb)/tw.Cfg.ProjDim + len(post) + 40)
		defer st.Close()
		logits, err := st.PrefillMixed(Span{Tokens: pre}, Span{Embd: emb, Grid: ts.Grid()}, Span{Tokens: post})
		if err != nil {
			t.Fatal(err)
		}
		var out []int32
		for i := 0; i < 32; i++ {
			next := Greedy(logits)
			if next == m.Vocab.EOS {
				break
			}
			out = append(out, next)
			if logits, err = st.Forward(next); err != nil {
				t.Fatal(err)
			}
		}
		return strings.ToLower(m.Vocab.Decode(out))
	}

	got := describe(real)
	t.Logf("jitllm:    %q", got)
	t.Logf("llama.cpp: \"a yellow circle is placed in the center of a red, blue, and green background.\"")
	for _, w := range []string{"yellow", "circle"} {
		if !strings.Contains(got, w) {
			t.Errorf("the description does not mention %q -- llama.cpp says "+
				"\"a yellow circle ... red, blue, and green\": %q", w, got)
		}
	}

	// The violation: the same path with no picture in it.
	blank := make([]float32, len(real))
	none := describe(blank)
	t.Logf("blank:     %q", none)
	if strings.Contains(none, "yellow") && strings.Contains(none, "circle") {
		t.Errorf("a BLANK image is described as a yellow circle too (%q), so the "+
			"embeddings are not what produced the answer", none)
	}
}
