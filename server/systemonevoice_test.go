package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
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

// TestSystemOneVoiceMatchesTheEngine posts the voice request (its clip in
// "files") to d1-omni and holds the body to what the engine's own Decider
// writes for the same request and clip, byte for byte. The same request
// without its "files" is a different body: the clip reached the model.
func TestSystemOneVoiceMatchesTheEngine(t *testing.T) {
	path := testmodels.Path(filepath.Join("d1-omni", "d1-omni-600M-F16-vlm.jlm"))
	fresh.Must(t, path)
	if _, err := os.Stat(path); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS to the model directory)", path)
	}
	body, err := os.ReadFile(filepath.Join("..", "engine", "testdata", "decision", "voice.json"))
	if err != nil {
		t.Fatal(err)
	}
	var req struct {
		Files []string `json:"files"`
	}
	if err := json.Unmarshal(body, &req); err != nil || len(req.Files) != 1 {
		t.Fatalf("the request holds %d files (%v)", len(req.Files), err)
	}
	_, data, _ := strings.Cut(req.Files[0], ",")
	raw, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		t.Fatal(err)
	}
	clip, err := engine.DecodeWAV(raw)
	if err != nil {
		t.Fatal(err)
	}
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	sp, err := m.ClipSpan(clip)
	if err != nil {
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
	ans, err := d.Decide(state, qs, sp)
	if err != nil {
		t.Fatal(err)
	}
	want := d.DecisionResponse("d1-omni", qs, ans)
	in.Close()
	m.Close()

	cfg := Config{ModelDir: filepath.Dir(path), Probe: noProbe}
	e := New(admitConfig(cfg))
	t.Cleanup(e.Close)
	if _, err := e.LoadModel(LoadOptions{Path: path, ModelID: "d1-omni"}); err != nil {
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
	var deaf map[string]any
	if err := json.Unmarshal(body, &deaf); err != nil {
		t.Fatal(err)
	}
	delete(deaf, "files")
	nb, err := json.Marshal(deaf)
	if err != nil {
		t.Fatal(err)
	}
	code, without := post(nb)
	if code != http.StatusOK {
		t.Fatalf("without its clip: status %d: %s", code, without)
	}
	if bytes.Equal(without, got) {
		t.Fatal("VIOLATION arm: the body without the clip is the body with it")
	}
}
