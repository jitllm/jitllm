package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// The text half of gemma3 and internvl with a picture in the prompt.

// cliOracle is scripts/vlmgold.py's record of one llama-mtmd-cli run: the
// prompt as the chunks it built and its greedy answer.
type cliOracle struct {
	Picture string `json:"picture"`
	Chunks  []struct {
		Text  *string `json:"text"`
		Image bool    `json:"image"`
	} `json:"chunks"`
	Out    string `json:"out"`
	Tokens int    `json:"tokens"`
}

// TestVisionGreedyMatchesLlamaMtmd teacher-forces llama-mtmd-cli's greedy
// answer through jitllm: the same prompt chunks and the same picture (already
// at the tower's size, so neither engine resizes it).
//
// The bar is not text-only's "agree until a tie": the picture is a newspaper
// page, the answer reads small print off it, and the two towers' embeddings
// differ by the precision each runs at (jitllm's int8 activations against
// llama.cpp's f16 mmproj, ~3% of the projector output). llama.cpp's own CUDA
// tower parts from its CPU tower on gemma3's answer at the token jitllm does.
// So the opening tokens must agree up to ties, and most of the rest must.
//
// gemma3's image rows are bidirectional, as llama.cpp decodes them
// (mtmd_decode_use_non_causal); the causal violation runs beside the clean arm
// and must disagree with llama.cpp by more.
func TestVisionGreedyMatchesLlamaMtmd(t *testing.T) {
	for _, fam := range visionFamilies() {
		t.Run(fam.name, func(t *testing.T) {
			raw, err := os.ReadFile(testmodels.Path(filepath.Join("vlm", "oracle", fam.name+"-cli.json")))
			if err != nil {
				testmodels.Missing(t, "ORACLE MISSING: %v -- RULE 11: run scripts/vlmgold.py %s", err, fam.name)
			}
			var o cliOracle
			if err := json.Unmarshal(raw, &o); err != nil {
				t.Fatal(err)
			}
			m, tw := fam.open(t)
			defer m.Close()
			f, err := os.Open(testmodels.Path(filepath.Join("vlm", "goldens", o.Picture)))
			if err != nil {
				testmodels.Missing(t, "PICTURE MISSING: %v -- run scripts/visiongold.py", err)
			}
			// llama.cpp's class token layout (towerCLSLast): this gate holds
			// the text model and the tower's arithmetic to it, and the tower's
			// layout is the reference's elsewhere (TestVisionMatchesTransformers).
			if os.Getenv("JITLLM_VLM_HFLAYOUT") == "" {
				tw.opt.towerCLSLast = true
				defer func() { tw.opt.towerCLSLast = false }()
			}
			ts := tw.testState()
			px, err := ts.Preprocess(f)
			f.Close()
			if err != nil {
				t.Fatal(err)
			}
			if len(px) != tw.Cfg.ImageSz*tw.Cfg.ImageSz*3 {
				t.Fatalf("%s is not one square at the tower's size: %d values", o.Picture, len(px))
			}
			e, err := ts.Encode(px)
			if err != nil {
				t.Fatal(err)
			}
			emb := append([]float32(nil), e...)
			ts.Close()

			ref := m.Vocab.Encode(o.Out, false)
			if got := m.Vocab.Decode(ref); got != o.Out {
				t.Fatalf("llama.cpp's answer does not survive re-tokenization:\n %q\n %q", o.Out, got)
			}
			// lead is the opening the answer must hold to the tie bound.
			const lead = 8
			force := func(bidir bool) (agree int, worst, excess float32) {
				var spans []Span
				for i, c := range o.Chunks {
					switch {
					case c.Image:
						spans = append(spans, Span{Embd: emb, Bidir: bidir})
					case c.Text != nil:
						// The first chunk takes the vocabulary's BOS, as mtmd's
						// tokenizer adds it: the rendered template's own was
						// stripped there.
						spans = append(spans, Span{Tokens: m.Vocab.EncodeSpecial(*c.Text, i == 0)})
					}
				}
				if n := SpanPositions(spans, m.Cfg.NEmbd); o.Tokens > 0 && n != o.Tokens {
					t.Fatalf("the prompt is %d positions here and %d in llama.cpp", n, o.Tokens)
				}
				st := m.NewInstance(SpanPositions(spans, m.Cfg.NEmbd) + len(ref) + 1)
				defer st.Close()
				logits, err := st.PrefillMixed(spans...)
				if err != nil {
					t.Fatal(err)
				}
				for i, want := range ref {
					got := Greedy(logits)
					if got == want {
						agree++
					} else {
						d := logits[got] - logits[want]
						excess += d
						if i < lead {
							worst = max(worst, d)
						}
						t.Logf("  position %d: llama.cpp %q, jitllm %q by %.3f (after %q)", i,
							m.Vocab.Text(want), m.Vocab.Text(got), d, m.Vocab.Decode(ref[:i]))
					}
					if i+1 < len(ref) {
						if logits, err = st.Forward(want); err != nil {
							t.Fatal(err)
						}
					}
				}
				return agree, worst, excess
			}
			bidir := tw.Cfg.Kind == jlm.ProjGemma3
			agree, worst, excess := force(bidir)
			t.Logf("%s: %d of %d positions agree with llama-mtmd-cli, the disagreements' margins sum "+
				"to %.3f; llama.cpp said %q", o.Picture, agree, len(ref), excess, o.Out)
			if worst > tieMargin {
				t.Errorf("a disagreement in the first %d tokens with a %.3f margin (tie bound %.1f)",
					lead, worst, tieMargin)
			}
			if agree*100 < 85*len(ref) {
				t.Errorf("only %d of %d positions agree", agree, len(ref))
			}
			if bidir {
				va, _, ve := force(false)
				t.Logf("violation, causal image rows: %d of %d agree, margins sum to %.3f", va, len(ref), ve)
				if ve <= excess {
					t.Errorf("causal image rows disagree with llama.cpp no more than bidirectional ones "+
						"(%.3f against %.3f): the gate cannot see the mask", ve, excess)
				}
			}
		})
	}
}

// TestChatSpansMatchTheProcessor holds the whole chat prompt, image markers
// included, to what transformers' processor tokenizes for the same messages:
// every image row stands for one image-token id there.
func TestChatSpansMatchTheProcessor(t *testing.T) {
	for _, fam := range visionFamilies() {
		t.Run(fam.name, func(t *testing.T) {
			meta := readMeta(t, fam.name, "photo")
			m, tw := fam.open(t)
			defer m.Close()
			emb := make([]float32, tw.Cfg.TokensFor(meta.W, meta.H)*tw.Cfg.ProjDim)
			spans, err := m.ChatSpans([]ChatMessage{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Describe this image in one sentence.", Images: 1},
			}, [][]float32{emb}, true)
			if err != nil {
				t.Fatal(err)
			}
			var got []int32
			bidir := false
			for _, sp := range spans {
				if sp.Embd != nil {
					for range len(sp.Embd) / m.Cfg.NEmbd {
						got = append(got, int32(meta.ImageToken))
					}
					bidir = bidir || sp.Bidir
					continue
				}
				got = append(got, sp.Tokens...)
			}
			if !slices.Equal(got, meta.ChatIDs) {
				t.Fatalf("the prompt differs from the processor's:\n got  %v\n want %v\n(the processor rendered %q)",
					trimRun(got, int32(meta.ImageToken)), trimRun(meta.ChatIDs, int32(meta.ImageToken)), meta.Chat)
			}
			if want := tw.Cfg.Kind == jlm.ProjGemma3; bidir != want {
				t.Errorf("the image span is bidirectional: %v, want %v", bidir, want)
			}
			t.Logf("%d ids, %d of them the image's, equal to the processor's", len(got),
				tw.Cfg.TokensFor(meta.W, meta.H))
		})
	}
}

// trimRun shortens every run of id to its length, so a mismatch prints.
func trimRun(ids []int32, id int32) []int32 {
	var out []int32
	for i := 0; i < len(ids); i++ {
		if ids[i] != id {
			out = append(out, ids[i])
			continue
		}
		j := i
		for j < len(ids) && ids[j] == id {
			j++
		}
		out = append(out, id, -int32(j-i))
		i = j - 1
	}
	return out
}
