package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	_ "image/png"
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

// TestSystemOnePicturesMatchTheEngine posts a /v1/systemone request with
// pictures to a decision model that states a joint head and no label readout
// (Clef's VL fixture) and holds the body to what the engine's own Decider
// writes for the same request and pictures, byte for byte. The same request
// without its "images" is a different body: the pictures reached the model.
func TestSystemOnePicturesMatchTheEngine(t *testing.T) {
	path := testmodels.Path(filepath.Join("clef", "fixture-vl", "clef-fixture-vl-vlm.jlm"))
	fresh.Must(t, path)
	if _, err := os.Stat(path); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS to the model directory)", path)
	}
	body, err := os.ReadFile(filepath.Join("..", "engine", "testdata", "decision", "pictures.json"))
	if err != nil {
		t.Fatal(err)
	}
	var req struct {
		Images []string `json:"images"`
	}
	if err := json.Unmarshal(body, &req); err != nil || len(req.Images) == 0 {
		t.Fatalf("the request holds no pictures (%v)", err)
	}

	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	var pics []model.Span
	for i, u := range req.Images {
		_, data, _ := strings.Cut(u, ",")
		raw, err := base64.StdEncoding.DecodeString(data)
		if err != nil {
			t.Fatalf("picture %d: %v", i, err)
		}
		img, _, err := image.Decode(bytes.NewReader(raw))
		if err != nil {
			t.Fatalf("picture %d: %v", i, err)
		}
		sp, err := m.PictureSpan(img)
		if err != nil {
			t.Fatal(err)
		}
		pics = append(pics, sp)
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
	ans, err := d.Decide(state, qs, pics...)
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
	var blind map[string]any
	if err := json.Unmarshal(body, &blind); err != nil {
		t.Fatal(err)
	}
	delete(blind, "images")
	nb, err := json.Marshal(blind)
	if err != nil {
		t.Fatal(err)
	}
	code, without := post(nb)
	if code != http.StatusOK {
		t.Fatalf("without its pictures: status %d: %s", code, without)
	}
	if bytes.Equal(without, got) {
		t.Fatal("VIOLATION arm: the body without the pictures is the body with them")
	}
}
