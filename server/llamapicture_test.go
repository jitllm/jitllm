package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/model"
)

// TestLlamaPictureMatchesTheEngine: /completion's {prompt_string,
// multimodal_data} reaches the tower: a picture where the text holds
// <__media__> answers as the engine does over the same spans -- the model's
// leading specials, then the text and the picture as Model.PromptSpans lays
// them out -- and two pictures answer differently.
func TestLlamaPictureMatchesTheEngine(t *testing.T) {
	e, lm, c := loadedEngine(t, visionModel, "vlm", LoadOptions{})
	if !lm.m.SeesPictures() {
		t.Fatalf("%s carries no vision tower", visionModel)
	}
	const text = "<|im_start|>User:<__media__>" + askPicture + "<end_of_utterance>\nAssistant:"
	quad := readTestdata(t, "quad-and-disc.png")
	red := testPNG(t, 384, 384, color.NRGBA{R: 200, G: 30, B: 30, A: 255})
	ask := func(pic []byte) string {
		body := fmt.Sprintf(`{"model":"vlm","temperature":0,"n_predict":12,"prompt":{"prompt_string":%q,"multimodal_data":[%q]}}`,
			text, base64.StdEncoding.EncodeToString(pic))
		resp, err := http.Post(c.url+"/completion", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		raw, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		var out struct{ Content string }
		if resp.StatusCode != 200 || json.Unmarshal(raw, &out) != nil {
			t.Fatalf("/completion: %d %s", resp.StatusCode, raw)
		}
		return out.Content
	}
	want := func(pic []byte) string {
		img, _, err := image.Decode(bytes.NewReader(pic))
		if err != nil {
			t.Fatal(err)
		}
		p, err := lm.m.PictureSpan(img)
		if err != nil {
			t.Fatal(err)
		}
		sp, err := lm.m.PromptSpans(text, "<__media__>", []model.Span{p})
		if err != nil {
			t.Fatal(err)
		}
		if lead := lm.m.Vocab.EncodeSpecial("", true); len(lead) > 0 {
			sp = append([]model.Span{{Tokens: lead}}, sp...)
		}
		var b strings.Builder
		err = e.Generate(context.Background(), GenerateOptions{ModelID: lm.id, MaxTokens: 12,
			Prompt: Prompt{Kind: PromptSpans, Spans: sp}, Sampling: &engine.Sampler{}}, func(ev Event) error {
			if ev.Kind == EventToken {
				b.WriteString(ev.Token.Text)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		return b.String()
	}
	gotQuad, gotRed := ask(quad), ask(red)
	t.Logf("quad-and-disc: %q; red: %q", gotQuad, gotRed)
	if w := want(quad); gotQuad != w {
		t.Fatalf("/completion with quad-and-disc: %q, the engine's %q", gotQuad, w)
	}
	if w := want(red); gotRed != w {
		t.Fatalf("/completion with red: %q, the engine's %q", gotRed, w)
	}
	if gotQuad == gotRed {
		t.Fatalf("two pictures gave the same answer %q: the picture does not reach the model", gotQuad)
	}
}
