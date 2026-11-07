package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/internal/testmodels/fresh"
	"github.com/jitllm/jitllm/model"
)

// TestSystemOneMatchesTheEngine posts a /v1/systemone request to a decision
// model the server loaded and holds the body to what the engine's own
// Decider writes for the same request, byte for byte; a text model is
// refused by name. llama.cpp's goldens hold the Decider itself
// (engine.TestDecisionMatchesLlamaCpp).
func TestSystemOneMatchesTheEngine(t *testing.T) {
	path := testmodels.Path(filepath.Join("openjev", "tinyopenjev-for-testing-Q8_0.jlm"))
	fresh.Must(t, path)
	if _, err := os.Stat(path); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS to the model directory)", path)
	}
	body, err := os.ReadFile(filepath.Join("..", "engine", "testdata", "decision", "support.json"))
	if err != nil {
		t.Fatal(err)
	}

	m, err := model.Open(path)
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
	ans, err := d.Decide(state, qs)
	if err != nil {
		t.Fatal(err)
	}
	want := d.DecisionResponse("oj", qs, ans)
	in.Close()
	m.Close()

	cfg := Config{ModelDir: filepath.Dir(path), Probe: noProbe}
	e := New(admitConfig(cfg))
	t.Cleanup(e.Close)
	if _, err := e.LoadModel(LoadOptions{Path: path, ModelID: "oj"}); err != nil {
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
	var parsed map[string]any
	if err := json.Unmarshal(got, &parsed); err != nil {
		t.Fatalf("the body is not JSON: %v", err)
	}
	if code, _ := post([]byte(`{"state": "x"}`)); code != http.StatusUnprocessableEntity {
		t.Errorf("a request with no questions: status %d, want 422", code)
	}
}
