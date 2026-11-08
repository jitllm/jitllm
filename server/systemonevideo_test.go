package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/internal/testmodels/fresh"
	"github.com/jitllm/jitllm/model"
)

// TestSystemOneVideoMatchesTheEngine posts the video request (its frames in
// "videos") to Clef's VL fixture and holds the body to what the engine's own
// Decider writes for the same request and video, byte for byte, the video
// given as a list of frames and as an object of them with no rate. The same
// request without its "videos" is a different body: the video reached the
// model. A model whose tower reads no video refuses one by name.
func TestSystemOneVideoMatchesTheEngine(t *testing.T) {
	path := testmodels.Path(filepath.Join("clef", "fixture-vl", "clef-fixture-vl-vlm.jlm"))
	fresh.Must(t, path)
	if _, err := os.Stat(path); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS to the model directory)", path)
	}
	body, err := os.ReadFile(filepath.Join("..", "engine", "testdata", "decision", "video.json"))
	if err != nil {
		t.Fatal(err)
	}
	var req struct {
		Videos [][]string `json:"videos"`
	}
	if err := json.Unmarshal(body, &req); err != nil || len(req.Videos) != 1 {
		t.Fatalf("the request holds %d videos (%v)", len(req.Videos), err)
	}
	var frames []image.Image
	for _, u := range req.Videos[0] {
		_, data, _ := strings.Cut(u, ",")
		raw, err := base64.StdEncoding.DecodeString(data)
		if err != nil {
			t.Fatal(err)
		}
		img, _, err := image.Decode(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		frames = append(frames, img)
	}
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	spans, err := m.VideoSpans(frames, 0)
	if err != nil {
		m.Close()
		t.Fatal(err)
	}
	in, err := engine.NewInstance(m, 4096)
	if err != nil {
		m.Close()
		t.Fatal(err)
	}
	d, err := engine.NewDecider(in)
	if err != nil {
		t.Fatal(err)
	}
	_, state, qs, err := engine.ParseDecisionRequest(body)
	if err != nil {
		t.Fatal(err)
	}
	ans, err := d.DecideMedia(state, qs, nil, [][]model.Span{spans})
	if err != nil {
		t.Fatal(err)
	}
	want := d.DecisionResponse("clef", qs, ans)
	in.Close()
	m.Close()

	cfg := Config{ModelDir: filepath.Dir(path), Probe: noProbe}
	e := New(admitConfig(cfg))
	t.Cleanup(e.Close)
	if _, err := e.LoadModel(LoadOptions{Path: path, ModelID: "clef"}); err != nil {
		t.Fatal(err)
	}
	s := httptest.NewServer(e.Handler())
	t.Cleanup(s.Close)
	post := func(b []byte) (int, []byte) {
		r, err := http.Post(s.URL+"/v1/systemone", "application/json", bytes.NewReader(b))
		if err != nil {
			t.Fatal(err)
		}
		defer r.Body.Close()
		out, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}
		return r.StatusCode, out
	}
	code, got := post(body)
	if code != http.StatusOK {
		t.Fatalf("status %d: %s", code, got)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("the server's body is not the Decider's:\n got %s\nwant %s", got, want)
	}
	var obj map[string]any
	if err := json.Unmarshal(body, &obj); err != nil {
		t.Fatal(err)
	}
	// The object form against the list form, both written from the same map
	// (a map writes the state's keys in its own order, another prompt).
	listed, err := json.Marshal(obj)
	if err != nil {
		t.Fatal(err)
	}
	obj["videos"] = []any{map[string]any{"frames": req.Videos[0]}}
	ob, err := json.Marshal(obj)
	if err != nil {
		t.Fatal(err)
	}
	code, asList := post(listed)
	if code != http.StatusOK {
		t.Fatalf("the video as a list of frames: status %d: %s", code, asList)
	}
	if code, again := post(ob); code != http.StatusOK || !bytes.Equal(again, asList) {
		t.Fatalf("the video as an object of frames is not the list's body: status %d: %s", code, again)
	}
	delete(obj, "videos")
	nb, err := json.Marshal(obj)
	if err != nil {
		t.Fatal(err)
	}
	code, without := post(nb)
	if code != http.StatusOK {
		t.Fatalf("without its video: status %d: %s", code, without)
	}
	if bytes.Equal(without, got) {
		t.Fatal("VIOLATION arm: the body without the video is the body with it")
	}
}
