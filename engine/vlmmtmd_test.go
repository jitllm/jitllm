package engine

import (
	"encoding/json"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// mtmdOracle is scripts/vlmgold.py's record of one llama-mtmd-cli run: the
// prompt as the chunks it built, its picture and its greedy answer.
type mtmdOracle struct {
	Picture string `json:"picture"`
	Chunks  []struct {
		Text  *string `json:"text"`
		Image bool    `json:"image"`
	} `json:"chunks"`
	Out    string `json:"out"`
	Tokens int    `json:"tokens"`
}

// TestTowerAnswersMatchLlamaMtmd teacher-forces llama-mtmd-cli's greedy
// answer through the stated tower and the text model on the CPU device: the
// same prompt chunks and the same picture, already at the tower's size, so
// neither engine resizes it. The answer reads small print off a newspaper
// page, and the two towers' rows differ by the precision each runs at
// (int8 activations here, an f16 mmproj there), so the opening tokens must
// agree up to a near tie (tieMargin) and most of the rest must agree.
//
// InternVL's class token goes first here, as the reference puts it, and
// last in llama.cpp (docs/divergences.md), which moves every patch's
// position: held to the reference, InternVL agrees with llama.cpp at 27 of
// 32 positions where llama.cpp's layout agreed at 31, and parts from it at
// the fifth (a "vintage" before "newspaper"), so it is held to its lower
// share alone, no opening tokens. A grey picture of the same size is a
// violation for both, and Gemma 3's rows causal, where llama.cpp decodes
// them both ways, for it.
func TestTowerAnswersMatchLlamaMtmd(t *testing.T) {
	for _, c := range []struct {
		name, text, mm string
		share, lead    int
	}{
		{"gemma3", "gemma-3-4b-it-Q4_K_M.gguf", "vlm/mmproj-gemma-3-4b-it-f16.gguf", 85, 8},
		{"internvl", "vlm/InternVL3-1B-Instruct-Q8_0.gguf", "vlm/mmproj-InternVL3-1B-Instruct-Q8_0.gguf", 75, 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			raw, err := os.ReadFile(testmodels.Path(filepath.Join("vlm", "oracle", c.name+"-cli.json")))
			if err != nil {
				testmodels.Missing(t, "ORACLE MISSING: %v -- run scripts/vlmgold.py %s (RULE 11)", err, c.name)
			}
			var o mtmdOracle
			if err := json.Unmarshal(raw, &o); err != nil {
				t.Fatal(err)
			}
			text, mm := testmodels.Path(c.text), testmodels.Path(c.mm)
			if !sourcePresent(text, mm) {
				testmodels.Missing(t, "MODEL MISSING: %s and %s (see docs/testing.md) -- this gate proved nothing", text, mm)
			}
			m, err := model.Open(jlmOfPair(t, text, mm))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			f, err := os.Open(testmodels.Path(filepath.Join("vlm", "goldens", o.Picture)))
			if err != nil {
				testmodels.Missing(t, "PICTURE MISSING: %v -- run scripts/visiongold.py", err)
			}
			img, err := png.Decode(f)
			f.Close()
			if err != nil {
				t.Fatal(err)
			}
			picture, err := m.PictureSpan(img)
			if err != nil {
				t.Fatal(err)
			}
			bidir := m.Container.Vision().Tower.Span.Bidir
			s := testInstance(t, m, 64)
			encode := func(pic image.Image) []float32 {
				rows, err := s.EncodePicture(pic)
				if err != nil {
					t.Fatal(err)
				}
				return slices.Clone(rows)
			}
			rows := encode(img)
			grey := image.NewRGBA(img.Bounds())
			draw.Draw(grey, grey.Bounds(), image.NewUniform(color.Gray{128}), image.Point{}, draw.Src)
			blank := encode(grey)
			s.Close()
			e := m.Cfg.NEmbd

			ref := m.Vocab.Encode(o.Out, false)
			if got := m.Vocab.Decode(ref); got != o.Out {
				t.Fatalf("llama.cpp's answer does not survive re-tokenization:\n %q\n %q", o.Out, got)
			}
			lead := c.lead
			force := func(pic []float32, both bool, log bool) (agree int, worst, excess float32) {
				var spans []model.Span
				for i, ch := range o.Chunks {
					switch {
					case ch.Image:
						spans = append(spans, model.Span{Embd: pic, Grid: picture.Grid, Bidir: both})
					case ch.Text != nil:
						// The first chunk takes the vocabulary's BOS, as mtmd's
						// tokenizer adds it.
						spans = append(spans, model.Span{Tokens: m.Vocab.EncodeSpecial(*ch.Text, i == 0)})
					}
				}
				n := model.SpanPositions(spans, e)
				if o.Tokens > 0 && n != o.Tokens {
					t.Fatalf("the prompt is %d positions here and %d in llama.cpp", n, o.Tokens)
				}
				st := testInstance(t, m, n+len(ref)+1)
				defer st.Close()
				lg, err := st.PrefillMixed(spans...)
				if err != nil {
					t.Fatal(err)
				}
				for i, want := range ref {
					got := int32(argmax(lg))
					if got == want {
						agree++
					} else {
						d := lg[got] - lg[want]
						excess += d
						if i < lead {
							worst = max(worst, d)
						}
						if log {
							t.Logf("  position %d: llama.cpp %q, here %q by %.3f", i,
								m.Vocab.Decode([]int32{want}), m.Vocab.Decode([]int32{got}), d)
						}
					}
					if i+1 < len(ref) {
						if lg, err = st.Forward(want); err != nil {
							t.Fatal(err)
						}
					}
				}
				return agree, worst, excess
			}
			agree, worst, excess := force(rows, bidir, true)
			t.Logf("%s: %d of %d positions agree with llama-mtmd-cli, the disagreements' margins sum to %.3f; llama.cpp said %q",
				o.Picture, agree, len(ref), excess, o.Out)
			if worst > tieMargin {
				t.Errorf("a disagreement in the first %d tokens by %.3f (tie bound %.1f)", lead, worst, tieMargin)
			}
			if agree*100 < c.share*len(ref) {
				t.Errorf("only %d of %d positions agree, under %d%%", agree, len(ref), c.share)
			}
			violations := []struct {
				name string
				pic  []float32
				both bool
			}{{"a grey picture's rows", blank, bidir}}
			if bidir {
				violations = append(violations, struct {
					name string
					pic  []float32
					both bool
				}{"the picture's rows causal", rows, false})
			}
			for _, v := range violations {
				va, _, ve := force(v.pic, v.both, false)
				t.Logf("violation, %s: %d of %d agree, margins sum to %.3f", v.name, va, len(ref), ve)
				if ve <= excess {
					t.Errorf("VIOLATION: %s disagrees with llama.cpp no more than the clean run (%.3f against %.3f)",
						v.name, ve, excess)
				}
			}
		})
	}
}
