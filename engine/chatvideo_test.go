package engine

import (
	"encoding/json"
	"image"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/model"
)

// chatVideoGolden is scripts/chatvideogold.py's record: a user turn of a
// picture, a video and text, rendered by the fixture's chat template and
// expanded by transformers' Qwen2VLProcessor.
type chatVideoGolden struct {
	Text    string `json:"text"`
	Picture struct {
		H, W int
		RGB  []byte `json:"rgb"`
	} `json:"picture"`
	Video struct {
		N, H, W int
		RGB     []byte `json:"rgb"`
	} `json:"video"`
	IDs []int32 `json:"ids"`
}

// frames is the golden's video, its frames in order.
func (g *chatVideoGolden) frames() []image.Image {
	n := g.Video.H * g.Video.W * 3
	out := make([]image.Image, g.Video.N)
	for i := range out {
		out[i] = rgbPicture(g.Video.RGB[i*n:(i+1)*n], g.Video.W, g.Video.H)
	}
	return out
}

// TestChatVideoFollowsTheProcessor holds a chat turn carrying a picture, a
// video and text (model.ChatSpansMedia, as the front ends build it) to the
// prompt transformers' Qwen2VLProcessor makes of the same turn
// (scripts/chatvideogold.py), id for id: the template's text around them,
// the picture's merged rows at its <|image_pad|> and every pair's at the
// video's <|video_pad|>, each between its markers. The violations: the
// picture and the video swapped, and the turn without its video, each a
// different prompt; a video the turn names and the call does not give, and a
// video given to a turn naming none, are refused.
func TestChatVideoFollowsTheProcessor(t *testing.T) {
	const name = "synth-qwen2vl"
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "qwenvl", name+".chat.json"))
	if err != nil {
		t.Fatalf("GOLDEN MISSING: %v -- run scripts/chatvideogold.py", err)
	}
	g := &chatVideoGolden{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	if len(g.Video.RGB) != g.Video.N*g.Video.H*g.Video.W*3 || len(g.Picture.RGB) != g.Picture.H*g.Picture.W*3 {
		t.Fatalf("the golden's video holds %d bytes for %d frames of %dx%d", len(g.Video.RGB), g.Video.N, g.Video.W, g.Video.H)
	}
	m, _, _ := openQVLTower(t, name)
	defer m.Close()
	pad := func(s string) int32 {
		ids := m.Vocab.EncodeSpecial(s, false)
		if len(ids) != 1 {
			t.Fatalf("%s encodes as %v", s, ids)
		}
		return ids[0]
	}
	imagePad, videoPad := pad("<|image_pad|>"), pad("<|video_pad|>")
	pic := rgbPicture(g.Picture.RGB, g.Picture.W, g.Picture.H)
	frames := g.frames()
	prompt := func(pics []image.Image, vids [][]image.Image, msg model.ChatMessage) ([]int32, error) {
		var ps []model.Span
		for _, p := range pics {
			sp, err := m.PictureSpan(p)
			if err != nil {
				t.Fatal(err)
			}
			ps = append(ps, sp)
		}
		var vs [][]model.Span
		for _, v := range vids {
			sp, err := m.VideoSpans(v, 0)
			if err != nil {
				t.Fatal(err)
			}
			vs = append(vs, sp)
		}
		spans, err := m.ChatSpansMedia([]model.ChatMessage{msg}, ps, vs, nil, model.ToolChoice{}, true)
		if err != nil {
			return nil, err
		}
		var ids []int32
		for _, sp := range spans {
			switch {
			case sp.Picture == nil:
				ids = append(ids, sp.Tokens...)
			case sp.Grid.Frames > 0 || sp.Grid.T > 1:
				for range sp.PicRows {
					ids = append(ids, videoPad)
				}
			default:
				for range sp.PicRows {
					ids = append(ids, imagePad)
				}
			}
		}
		return ids, nil
	}
	turn := model.ChatMessage{Role: "user", Content: "What happens in the video?", Images: 1, Videos: 1}
	got, err := prompt([]image.Image{pic}, [][]image.Image{frames}, turn)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, g.IDs) {
		t.Fatalf("the turn's prompt\n  got  %v\n  want %v", got, g.IDs)
	}
	t.Logf("%d ids, the processor's: %d picture rows, %d video rows", len(got),
		count(got, imagePad), count(got, videoPad))

	swapped, err := prompt([]image.Image{frames[0]}, [][]image.Image{{pic, pic}}, turn)
	if err != nil {
		t.Fatal(err)
	}
	if slices.Equal(swapped, g.IDs) {
		t.Fatal("VIOLATION: the picture and the video swapped lay out as the processor's prompt")
	}
	alone := turn
	alone.Videos = 0
	if bare, err := prompt([]image.Image{pic}, nil, alone); err != nil || slices.Equal(bare, g.IDs) {
		t.Fatalf("VIOLATION: the turn without its video lays out as the processor's prompt (%v)", err)
	}
	if _, err := prompt([]image.Image{pic}, nil, turn); err == nil {
		t.Error("a turn naming a video and given none was accepted")
	}
	if _, err := prompt([]image.Image{pic}, [][]image.Image{frames}, alone); err == nil {
		t.Error("a video given to a turn naming none was accepted")
	}
}

func count(ids []int32, id int32) int {
	n := 0
	for _, x := range ids {
		if x == id {
			n++
		}
	}
	return n
}
