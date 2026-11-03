package model

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// flatSpans lays spans out as one id sequence with -1 where an image goes, so a
// whole prompt is one comparison.
func flatSpans(t *testing.T, spans []Span) []int32 {
	t.Helper()
	var out []int32
	for _, sp := range spans {
		if sp.Embd != nil {
			out = append(out, -1)
			continue
		}
		out = append(out, sp.Tokens...)
	}
	return out
}

// TestChatSpansFollowTheTemplate pins the image marker sequence, which no
// numerical gate sees: with every marker wrong the picture still arrives
// correctly encoded, unannounced. The markers are mtmd's (tools/mtmd/mtmd.cpp
// at b10825):
//
//	tok_ov_img_start = {"\n\n", "<fake_token_around_image>", "<global-img>"}
//	tok_ov_img_end   = {"<fake_token_around_image>"}
//
// The turn follows the model's own template, which departs from mtmd by one
// token on purpose: SmolVLM's template writes "User:" before an image-led turn,
// where mtmd emits "User: " (a ' ', 216, before the "\n\n").
func TestChatSpansFollowTheTemplate(t *testing.T) {
	m, tw := openTower(t)
	defer m.Close()
	emb := make([]float32, tw.Cfg.Tokens()*tw.Cfg.ProjDim)
	spans, err := m.ChatSpans([]ChatMessage{{Role: "user", Content: "Describe the image.", Images: 1}},
		[][]float32{emb}, true)
	if err != nil {
		t.Fatal(err)
	}
	//                <|im_start|> User  :   \n\n  <fake…> <global-img> IMG <fake…>
	want := []int32{1, 11126, 42, 1116, 49189, 49152, -1, 49189,
		// Describe  the  image  .  <end_of_utterance>  \n  Ass  istant  :
		37964, 260, 2443, 30, 49279, 198, 9519, 9531, 42}
	if got := flatSpans(t, spans); !slices.Equal(got, want) {
		t.Fatalf("spans\n  got  %v\n  want %v", got, want)
	}
	// The embedding span is the caller's slice, not a copy: the CLI sizes a
	// Instance from these spans and encodes into it afterwards.
	for _, sp := range spans {
		if sp.Embd != nil && &sp.Embd[0] != &emb[0] {
			t.Error("the image span copied the embeddings instead of referring to them")
		}
	}
	if got, want := SpanPositions(spans, m.Cfg.NEmbd), len(want)-1+tw.Cfg.Tokens(); got != want {
		t.Errorf("SpanPositions %d, want %d", got, want)
	}

	// The refusals, each of which would otherwise put a picture on the wrong
	// turn or none at all.
	one := []ChatMessage{{Role: "user", Content: "x", Images: 1}}
	if _, err := m.ChatSpans(one, nil, true); err == nil {
		t.Error("one image and no embeddings was accepted")
	}
	lit := []ChatMessage{{Role: "user", Content: "what is <image>?", Images: 1}}
	if _, err := m.ChatSpans(lit, [][]float32{emb}, true); err == nil {
		t.Error("user text containing the placeholder shifted the image count and was accepted")
	}
	if _, err := m.ChatIDs(one, true); err == nil {
		t.Error("ChatIDs tokenized an image placeholder as text")
	}
}

// TestChatSpansDescribeAnImage runs every projector family end to end
// through ChatSpans, the path the CLI and the UI take. A blank image runs beside
// each: a tower wired to nothing still emits fluent text, so if the blank
// answer names the picture too, the picture is not what produced it.
func TestChatSpansDescribeAnImage(t *testing.T) {
	f, err := os.ReadFile(filepath.Join("testdata", "quad-and-disc.png"))
	if err != nil {
		t.Fatalf("IMAGE MISSING: %v -- this gate proved nothing", err)
	}
	for _, c := range []struct {
		name string
		open func(testing.TB) *Model
		want []string // every word must appear for the real image, not all for the blank
		// n is how many tokens to generate: 32 unless the model opens with a
		// preamble (gemma3's "Here's a description of the image in one
		// sentence:" on arm64 leaves the picture's last words past 32).
		n int
	}{
		{"smolvlm", func(t testing.TB) *Model { m, _ := openTower(t); return m }, chatSpansWant()["smolvlm"], 0},
		{"llava", func(t testing.TB) *Model { m, _ := openLlava(t); return m }, chatSpansWant()["llava"], 0},
		{"qwen2vl", openQwen2VL, chatSpansWant()["qwen2vl"], 0},
		{"gemma3", func(t testing.TB) *Model { m, _ := openGemma3(t); return m }, chatSpansWant()["gemma3"], 64},
		{"internvl", func(t testing.TB) *Model { m, _ := openInternVL(t); return m }, chatSpansWant()["internvl"], 0},
		{"internvl2.5", openInternVL25, chatSpansWant()["internvl"], 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			m := c.open(t)
			defer m.Close()
			tw := m.Tower()
			px, sz, err := preprocessSized(tw, strings.NewReader(string(f)))
			if err != nil {
				t.Fatal(err)
			}
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
				spans, err := m.ChatSpansImages([]ChatMessage{{Role: "user",
					Content: "Describe this image in one sentence.", Images: 1}},
					[]Image{{Embd: emb, Grid: ts.Grid()}}, true)
				if err != nil {
					t.Fatal(err)
				}
				n := c.n
				if n == 0 {
					n = 32
				}
				st := m.NewInstance(SpanPositions(spans, m.Cfg.NEmbd) + n + 8)
				defer st.Close()
				logits, err := st.PrefillMixed(spans...)
				if err != nil {
					t.Fatal(err)
				}
				var out []int32
				for range n {
					next := Greedy(logits)
					if m.Vocab.IsEOG(next) {
						break
					}
					out = append(out, next)
					if logits, err = st.Forward(next); err != nil {
						t.Fatal(err)
					}
				}
				return strings.ToLower(m.Vocab.Decode(out))
			}
			got := describe(px)
			none := describe(make([]float32, len(px)))
			t.Logf("image %q", got)
			t.Logf("blank %q", none)
			all := func(s string) bool {
				for _, w := range c.want {
					if !strings.Contains(s, w) {
						return false
					}
				}
				return true
			}
			if !all(got) {
				t.Errorf("the description does not mention all of %v: %q", c.want, got)
			}
			if all(none) {
				t.Errorf("a BLANK image is described with %v too (%q), so the picture is not "+
					"what produced the answer", c.want, none)
			}
		})
	}
}

// chatSpansWant is what each model says about quad-and-disc.png -- a yellow disc
// over red, green and blue quadrants -- set from what it does say, and held to
// the blank-image violation beside it. llava's words are "circle" and "four"
// because with its special tokens matched (its prompt equals llama-tokenize's)
// it describes the sections rather than naming the colours.
func chatSpansWant() map[string][]string {
	return map[string][]string{
		"smolvlm": {"yellow"},
		"llava":   {"circle", "four"},
		"qwen2vl": {"yellow", "circle"},
		// gemma3: "a stylized flag with four colored quadrants - red, green, blue,
		// and white - surrounding a central yellow circle"; internvl: "a colorful,
		// abstract design with a central yellow circle surrounded by red, green,
		// blue, and white squares".
		"gemma3":   {"yellow", "circle"},
		"internvl": {"yellow", "circle"},
	}
}
