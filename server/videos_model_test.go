package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/model"
)

// videoModel is the Qwen2-VL fixture, whose tower reads a video
// (convert.qwen2vlVideo).
const videoModel = "synth-qwen2vl-vlm.jlm"

const askVideo = "What happens in the video?"

// discFrames is n PNG frames of w x h, a disc moving across a gradient.
func discFrames(t *testing.T, n, w, h int) [][]byte {
	t.Helper()
	var out [][]byte
	for f := range n {
		img := image.NewNRGBA(image.Rect(0, 0, w, h))
		for y := range h {
			for x := range w {
				c := color.NRGBA{R: uint8(255 * x / w), G: uint8(255 * y / h), B: 64, A: 255}
				if dx, dy := x-12-10*f, y-h/2; dx*dx+dy*dy < 100 {
					c = color.NRGBA{R: 250, G: 250, B: 250, A: 255}
				}
				img.SetNRGBA(x, y, c)
			}
		}
		var b bytes.Buffer
		if err := png.Encode(&b, img); err != nil {
			t.Fatal(err)
		}
		out = append(out, b.Bytes())
	}
	return out
}

// engineVideoRun is the engine's own answer to a user turn of one video (its
// frames at fps, 0 for the model's default rate) and askVideo, or of askVideo
// alone for no frames: Model.VideoSpans in the turn's video part
// (model.ChatSpansMedia), greedy, twelve tokens.
func engineVideoRun(t *testing.T, e *Engine, lm *LoadedModel, frames [][]byte, fps float64) string {
	t.Helper()
	msg := model.ChatMessage{Role: "user", Content: askVideo}
	var vids [][]model.Span
	if len(frames) > 0 {
		var imgs []image.Image
		for _, f := range frames {
			img, _, err := image.Decode(bytes.NewReader(f))
			if err != nil {
				t.Fatal(err)
			}
			imgs = append(imgs, img)
		}
		sp, err := lm.m.VideoSpans(imgs, fps)
		if err != nil {
			t.Fatal(err)
		}
		vids, msg.Videos = [][]model.Span{sp}, 1
	}
	spans, err := lm.m.ChatSpansMedia([]model.ChatMessage{msg}, nil, vids, nil, model.ToolChoice{}, true)
	if err != nil {
		t.Fatal(err)
	}
	var out strings.Builder
	err = e.Generate(context.Background(), GenerateOptions{ModelID: lm.id, MaxTokens: 12,
		Prompt: Prompt{Kind: PromptSpans, Spans: spans}, Sampling: &engine.Sampler{}},
		func(ev Event) error {
			if ev.Kind == EventToken {
				out.WriteString(ev.Token.Text)
			}
			return nil
		})
	if err != nil {
		t.Fatal(err)
	}
	return out.String()
}

// oaVideoBody is a chat completion of a user turn of a video and askVideo:
// the video as a list of frame URLs, or with fps > 0 as an object of them and
// the rate.
func oaVideoBody(frames [][]byte, fps float64, stream bool) string {
	var urls []string
	for _, f := range frames {
		urls = append(urls, fmt.Sprintf("%q", "data:image/png;base64,"+base64.StdEncoding.EncodeToString(f)))
	}
	video := "[" + strings.Join(urls, ",") + "]"
	if fps > 0 {
		video = fmt.Sprintf(`{"frames":%s,"fps":%g}`, video, fps)
	}
	return fmt.Sprintf(`{"model":"vid","temperature":0,"max_tokens":12,"stream":%v,"messages":[{"role":"user","content":[
		{"type":"video","video":%s},{"type":"text","text":%q}]}]}`, stream, video, askVideo)
}

// TestVideoRequestsMatchTheEngine posts a user turn of a video and text to
// /v1/chat/completions on the Qwen2-VL fixture -- the video a list of frame
// URLs, and an object of them with a rate, which Qwen2-VL reads every frame
// of and places a position a pair whatever it is -- and holds each reply,
// streamed and not, to the engine's own run of the same turn
// (Model.VideoSpans in model.ChatSpansMedia). The frames reversed and the
// turn without its video each answer differently, and the server answers
// the reversed frames as the engine does too: the video reaches the model,
// its order with it. A video part with no frames is refused with a 400.
func TestVideoRequestsMatchTheEngine(t *testing.T) {
	e, lm, c := loadedEngine(t, videoModel, "vid", LoadOptions{})
	if v := lm.m.Container.Vision(); v == nil || v.Tower == nil || v.Tower.Video == nil {
		t.Fatalf("%s states no video for its tower", videoModel)
	}
	frames := discFrames(t, 4, 112, 84)
	rev := slices.Clone(frames)
	slices.Reverse(rev)
	want := engineVideoRun(t, e, lm, frames, 0)
	wantRev := engineVideoRun(t, e, lm, rev, 0)
	wantSlow := engineVideoRun(t, e, lm, frames, 1)
	wantNone := engineVideoRun(t, e, lm, nil, 0)
	t.Logf("the video: %q; reversed: %q; at one frame a second: %q; none: %q", want, wantRev, wantSlow, wantNone)
	if want == wantRev || want == wantNone {
		t.Fatal("VIOLATION: the video reversed or absent answers as the video does: the gate cannot see the video")
	}
	for _, stream := range []bool{false, true} {
		for _, x := range []struct {
			name   string
			frames [][]byte
			fps    float64
			want   string
		}{{"frames", frames, 0, want}, {"reversed", rev, 0, wantRev}, {"with a rate", frames, 1, wantSlow}} {
			if got := replyText(t, c.url, "/v1/chat/completions", oaVideoBody(x.frames, x.fps, stream)); got != x.want {
				t.Errorf("%s stream=%v: %q, want the engine's %q", x.name, stream, got, x.want)
			}
		}
	}
	resp, err := http.Post(c.url+"/v1/chat/completions", "application/json", strings.NewReader(oaVideoBody(nil, 0, false)))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("a video part with no frames: status %d, want 400", resp.StatusCode)
	}
}
