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

	"github.com/jitllm/jitllm/engine/model"
)

// visionModel is the smallest container with a vision tower in the test
// models: SmolVLM-256M, its idefics3 tower converted in and stated.
const visionModel = "SmolVLM-256M-Instruct-Q8_0-vlm.jlm"

const askPicture = "What is in this picture? Answer in one sentence."

// engineRun is the engine's own picture run, built without either shim: the
// prompt laid out by the model's ChatSpans with each picture a span the
// prefill encodes, and a greedy generate of those spans.
func engineRun(t *testing.T, e *Engine, lm *LoadedModel, pics ...[]byte) (string, []model.Span) {
	t.Helper()
	var sp []model.Span
	for _, pic := range pics {
		img, _, err := image.Decode(bytes.NewReader(pic))
		if err != nil {
			t.Fatal(err)
		}
		p, err := lm.m.PictureSpan(img)
		if err != nil {
			t.Fatal(err)
		}
		sp = append(sp, p)
	}
	spans, err := lm.m.ChatSpans([]model.ChatMessage{{Role: "user", Content: askPicture, Images: len(pics)}}, sp, true)
	if err != nil {
		t.Fatal(err)
	}
	var out strings.Builder
	err = e.Generate(context.Background(), GenerateOptions{ModelID: lm.id, MaxTokens: 12,
		Prompt: Prompt{Kind: PromptSpans, Spans: spans}, Sampling: &model.Sampler{}},
		func(ev Event) error {
			if ev.Kind == EventToken {
				out.WriteString(ev.Token.Text)
			}
			return nil
		})
	if err != nil {
		t.Fatal(err)
	}
	return out.String(), spans
}

// pictureKey is the key naming the first picture of spans.
func pictureKey(t *testing.T, spans []model.Span) model.RowsKey {
	t.Helper()
	for _, sp := range spans {
		if sp.Picture != nil && sp.Key != nil {
			return *sp.Key
		}
	}
	t.Fatal("the spans carry no named picture")
	return model.RowsKey{}
}

// oaImageBody and anImageBody are one user turn of askPicture and pic.
func oaImageBody(pic []byte, mt string, stream bool) string {
	return fmt.Sprintf(`{"model":"vlm","temperature":0,"max_tokens":12,"stream":%v,"messages":[{"role":"user","content":[
		{"type":"image_url","image_url":{"url":"data:%s;base64,%s"}},{"type":"text","text":%q}]}]}`,
		stream, mt, base64.StdEncoding.EncodeToString(pic), askPicture)
}

func anImageBody(pic []byte, mt string, stream bool) string {
	return fmt.Sprintf(`{"model":"vlm","temperature":0,"max_tokens":12,"stream":%v,"messages":[{"role":"user","content":[
		{"type":"image","source":{"type":"base64","media_type":%q,"data":%q}},{"type":"text","text":%q}]}]}`,
		stream, mt, base64.StdEncoding.EncodeToString(pic), askPicture)
}

// replyText is the text of a response from either shim, streamed or not.
func replyText(t *testing.T, url, path, body string) string {
	t.Helper()
	resp, err := http.Post(url+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("%s: %d %s", path, resp.StatusCode, raw)
	}
	var out strings.Builder
	if !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		var r struct {
			Choices []struct {
				Message struct{ Content string } `json:"message"`
			} `json:"choices"`
			Content []struct{ Text string } `json:"content"`
		}
		if err := json.Unmarshal(raw, &r); err != nil {
			t.Fatal(err)
		}
		for _, c := range r.Choices {
			out.WriteString(c.Message.Content)
		}
		for _, c := range r.Content {
			out.WriteString(c.Text)
		}
		return out.String()
	}
	for _, line := range strings.Split(string(raw), "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var f struct {
			Choices []struct {
				Delta struct{ Content string } `json:"delta"`
			} `json:"choices"`
			Delta struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"delta"`
		}
		if err := json.Unmarshal([]byte(data), &f); err != nil {
			t.Fatalf("frame %q: %v", data, err)
		}
		for _, c := range f.Choices {
			out.WriteString(c.Delta.Content)
		}
		if f.Delta.Type == "text_delta" {
			out.WriteString(f.Delta.Text)
		}
	}
	return out.String()
}

// TestImageRequestsMatchTheEngine: a picture sent through either shim,
// streamed or not, gives the tokens the engine's own picture run gives for
// it; and a different picture with the same text is a different prompt and a
// different answer, so the picture reached the model.
func TestImageRequestsMatchTheEngine(t *testing.T) {
	e, lm, c := loadedEngine(t, visionModel, "vlm", LoadOptions{})
	if !lm.m.SeesPictures() {
		t.Fatalf("%s carries no vision tower", visionModel)
	}
	quad := readTestdata(t, "quad-and-disc.png")
	red := testPNG(t, 384, 384, color.NRGBA{R: 200, G: 30, B: 30, A: 255})

	wantQuad, spQuad := engineRun(t, e, lm, quad)
	wantRed, spRed := engineRun(t, e, lm, red)
	t.Logf("quad-and-disc: %q", wantQuad)
	t.Logf("red: %q", wantRed)
	if pictureKey(t, spQuad) == pictureKey(t, spRed) {
		t.Fatal("two different pictures were named alike: the prefix cache would take one for the other")
	}
	if wantQuad == wantRed {
		t.Fatalf("two different pictures gave the same answer %q: the picture does not reach the model", wantQuad)
	}
	for _, x := range []struct {
		name string
		pic  []byte
		want string
	}{{"quad", quad, wantQuad}, {"red", red, wantRed}} {
		for _, stream := range []bool{false, true} {
			if got := replyText(t, c.url, "/v1/chat/completions", oaImageBody(x.pic, "image/png", stream)); got != x.want {
				t.Errorf("OpenAI %s stream=%v: %q, want the engine's %q", x.name, stream, got, x.want)
			}
			if got := replyText(t, c.url, "/v1/messages", anImageBody(x.pic, "image/png", stream)); got != x.want {
				t.Errorf("Anthropic %s stream=%v: %q, want the engine's %q", x.name, stream, got, x.want)
			}
		}
	}
	// Two pictures in one turn keep the client's order: the request matches
	// the engine's run of them in that order, and the swapped order is a
	// different prompt.
	both := func(a, b []byte) string {
		return fmt.Sprintf(`{"model":"vlm","temperature":0,"max_tokens":12,"messages":[{"role":"user","content":[
			{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}},
			{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}},{"type":"text","text":%q}]}]}`,
			base64.StdEncoding.EncodeToString(a), base64.StdEncoding.EncodeToString(b), askPicture)
	}
	wantBoth, _ := engineRun(t, e, lm, quad, red)
	wantSwapped, _ := engineRun(t, e, lm, red, quad)
	t.Logf("quad then red: %q; red then quad: %q", wantBoth, wantSwapped)
	if wantBoth == wantSwapped {
		t.Fatal("two pictures in either order gave the same answer: the order is not a property of the prompt")
	}
	if got := replyText(t, c.url, "/v1/chat/completions", both(quad, red)); got != wantBoth {
		t.Errorf("two pictures: %q, want the engine's %q", got, wantBoth)
	}
	// A text request on the same model still answers: nothing about a
	// picture prompt sticks to the model.
	if got := replyText(t, c.url, "/v1/chat/completions",
		`{"model":"vlm","temperature":0,"max_tokens":4,"messages":[{"role":"user","content":"Hello"}]}`); got == "" {
		t.Fatal("a text request after the picture requests produced nothing")
	}
}

// TestNonVisionModelRefusesImages: a model with no tower answers a picture
// with a 400 naming the model, through both shims.
func TestNonVisionModelRefusesImages(t *testing.T) {
	_, _, c := loadedEngine(t, chatModel, "txt", LoadOptions{})
	pic := testPNG(t, 4, 4, image.White)
	for _, x := range []struct{ path, body string }{
		{"/v1/chat/completions", strings.Replace(oaImageBody(pic, "image/png", false), `"vlm"`, `"txt"`, 1)},
		{"/v1/chat/completions", strings.Replace(oaImageBody(pic, "image/png", true), `"vlm"`, `"txt"`, 1)},
		{"/v1/messages", strings.Replace(anImageBody(pic, "image/png", false), `"vlm"`, `"txt"`, 1)},
		{"/v1/messages", strings.Replace(anImageBody(pic, "image/png", true), `"vlm"`, `"txt"`, 1)},
	} {
		resp, err := http.Post(c.url+x.path, "application/json", strings.NewReader(x.body))
		if err != nil {
			t.Fatal(err)
		}
		raw, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			t.Fatal(err)
		}
		if resp.StatusCode != 400 || !strings.Contains(string(raw), `model \"txt\" does not accept images`) {
			t.Errorf("%s: %d %s, want 400 naming the model", x.path, resp.StatusCode, raw)
		}
	}
}
