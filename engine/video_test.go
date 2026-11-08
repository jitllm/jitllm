package engine

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// decisionVideo is request name's first video's frames, each a data URL's
// picture decoded.
func decisionVideo(t *testing.T, name string) []image.Image {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "decision", name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var req struct {
		Videos [][]string `json:"videos"`
	}
	if err := json.Unmarshal(b, &req); err != nil {
		t.Fatal(err)
	}
	if len(req.Videos) == 0 || len(req.Videos[0]) == 0 {
		t.Fatalf("request %s holds no video", name)
	}
	var out []image.Image
	for i, u := range req.Videos[0] {
		_, data, ok := strings.Cut(u, ",")
		if !ok {
			t.Fatalf("frame %d is no data URL", i)
		}
		raw, err := base64.StdEncoding.DecodeString(data)
		if err != nil {
			t.Fatal(err)
		}
		img, _, err := image.Decode(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, img)
	}
	return out
}

// TestVideoSpansFollowTheProcessor reads the video request through Clef's
// VL fixture: its spans' tokens, each pair's rows written as the video pads
// they stand for, are a run of the reference prompt token for token (the
// frames sampled, the times and the markers the processor's); each pair
// encodes on the CPU device to finite rows, and a pair whose second frame is
// its first encodes to other rows -- the second frame is read.
func TestVideoSpansFollowTheProcessor(t *testing.T) {
	src, ok := existingModel(testmodels.Path("clef/fixture-vl/clef-fixture-vl.gguf"))
	if !ok {
		testmodels.Missing(t, "MODEL MISSING: %s (scripts/clefgold.py makes it)", testmodels.Path("clef/fixture-vl/clef-fixture-vl.gguf"))
	}
	m, err := model.Open(jlmOfPair(t, src, towerOf(src)))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	spans, err := m.VideoSpans(decisionVideo(t, "video"), 0)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join("testdata", "decision", "clef-fixture-vl.video.reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		IDs []int32 `json:"ids"`
	}
	if err := json.Unmarshal(b, &ref); err != nil {
		t.Fatal(err)
	}
	pad := m.Vocab.EncodeSpecial("<|video_pad|>", false)
	if len(pad) != 1 {
		t.Fatalf("the video pad encodes as %v", pad)
	}
	var flat []int32
	pairs := 0
	for _, sp := range spans {
		if sp.Picture == nil {
			flat = append(flat, sp.Tokens...)
			continue
		}
		pairs++
		for range sp.PicRows {
			flat = append(flat, pad[0])
		}
	}
	found := false
	for i := range ref.IDs {
		if slices.Equal(ref.IDs[i:min(i+len(flat), len(ref.IDs))], flat) {
			found = true
			break
		}
	}
	if pairs != 2 || !found {
		t.Fatalf("%d pairs; the video's tokens %v are no run of the reference prompt %v", pairs, flat, ref.IDs)
	}
	s := testInstance(t, m, 4096)
	defer s.Close()
	enc, err := s.encodePictures(spans, 0)
	if err != nil {
		t.Fatal(err)
	}
	var first []float32
	for _, sp := range enc {
		if sp.Frame == nil {
			continue
		}
		for _, v := range sp.Embd {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatal("a pair encoded to a non-finite row")
			}
		}
		if first == nil {
			first = append([]float32(nil), sp.Embd...)
		}
	}
	still := slices.Clone(spans)
	for i := range still {
		if still[i].Frame != nil {
			still[i].Frame = still[i].Picture
			break
		}
	}
	enc, err = s.encodePictures(still, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, sp := range enc {
		if sp.Frame == nil {
			continue
		}
		var d, n float64
		for i, v := range sp.Embd {
			d += (float64(v) - float64(first[i])) * (float64(v) - float64(first[i]))
			n += float64(first[i]) * float64(first[i])
		}
		if !(d > 1e-4*n) {
			t.Fatalf("a pair read its first frame twice and encoded the same rows (%.3g of %.3g): the second frame is not read", d, n)
		}
		break
	}
}
