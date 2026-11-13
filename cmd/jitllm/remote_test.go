package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/localserver"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/internal/testmodels/fresh"
)

// fakeService answers as a jitllmd: the capability manifest, and a streamed
// completion that echoes the model it was asked for.
func fakeService(t *testing.T, api int, got *map[string]any) *httptest.Server {
	t.Helper()
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/capabilities":
			json.NewEncoder(w).Encode(map[string]any{"schema": 1, "server_api": api, "build": map[string]any{"version": "v7"}})
		case "/v1/completions", "/v1/chat/completions":
			json.NewDecoder(r.Body).Decode(got)
			(*got)["route"] = r.URL.Path
			w.Header().Set("Content-Type", "text/event-stream")
			for _, p := range []string{"from ", (*got)["model"].(string)} {
				b, _ := json.Marshal(map[string]any{"choices": []map[string]any{{"text": p, "delta": map[string]any{"content": ""}}}})
				w.Write([]byte("data: " + string(b) + "\n\n"))
			}
			w.Write([]byte("data: [DONE]\n\n"))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(s.Close)
	return s
}

// TestRunUsesTheServiceWhenOneAnswers: a model NAME with a service answering
// goes to the service, which is asked by that name, and says so on stderr.
func TestRunUsesTheServiceWhenOneAnswers(t *testing.T) {
	var got map[string]any
	s := fakeService(t, localserver.APIVersion, &got)
	var note bytes.Buffer
	svc, path, err := runTarget(context.Background(), "qwen3", s.URL, "", false, nil, &note)
	if err != nil || svc == nil || path != "" {
		t.Fatalf("runTarget: %v %q %v", svc, path, err)
	}
	if !strings.Contains(note.String(), "using the service at "+s.URL) {
		t.Errorf("stderr: %q", note.String())
	}
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.Float64("temp", 0, "")
	fs.Parse([]string{"-temp", "0.5"})
	var out bytes.Buffer
	r := newRemoteRun(s.URL, "", "qwen3", 5, "", fs, samplingFlags{temp: 0.5})
	if err := r.once(false, "hello", &out); err != nil {
		t.Fatal(err)
	}
	if out.String() != "from qwen3\n" || got["prompt"] != "hello" || got["max_tokens"] != float64(5) ||
		got["temperature"] != 0.5 || got["route"] != "/v1/completions" {
		t.Fatalf("out %q, request %v", out.String(), got)
	}
	if err := r.once(true, "hi", &out); err != nil || got["route"] != "/v1/chat/completions" {
		t.Fatalf("chat: %v %v", err, got)
	}
}

// TestConvertByCatalogNameKeepsTheName: a container converted from a
// catalog name into the models folder answers to that name, so `jitllm run`
// finds it by the name it was fetched by.
func TestConvertByCatalogNameKeepsTheName(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("JITLLM_MODELS", dir)
	dst := filepath.Join(dir, "SmolLM2-360M-Instruct-Q8_0.jlm")
	if err := os.WriteFile(dst, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	aliasCatalogName([]string{"smollm2-360m-instruct"}, dst)
	if p, _, err := localserver.Find([]string{dir}, "smollm2-360m-instruct"); err != nil || p != dst {
		t.Fatalf("Find by the catalog name: %q %v", p, err)
	}
	other := filepath.Join(t.TempDir(), "x.jlm")
	aliasCatalogName([]string{"smollm2-1.7b-instruct"}, other)
	if p, _, _ := localserver.Find([]string{dir}, "smollm2-1.7b-instruct"); p != "" {
		t.Fatalf("a container written elsewhere took an alias in the models folder")
	}
}

// TestRunRefusesAnIncompatibleService: a jitllmd of another API version is
// not silently bypassed with a second engine.
func TestRunRefusesAnIncompatibleService(t *testing.T) {
	var got map[string]any
	s := fakeService(t, localserver.APIVersion+1, &got)
	if _, _, err := runTarget(context.Background(), "qwen3", s.URL, "", false, nil, &bytes.Buffer{}); err == nil {
		t.Fatal("an incompatible service was bypassed")
	}
}

// TestRunFallsBackToTheModelsFolder: with no service, a NAME (here an alias)
// is found in the models folder and run in this process; a path and an
// engine flag run in this process even with a service answering.
func TestRunFallsBackToTheModelsFolder(t *testing.T) {
	fresh.Must(t, testmodels.Path("stories260K.jlm"))
	b, err := os.ReadFile(testmodels.Path("stories260K.jlm"))
	if err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS) -- this gate proved nothing", err)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "stories260K.jlm"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	if ok, err := localserver.AddAlias(dir, "stories260K.jlm", "story"); !ok || err != nil {
		t.Fatal(ok, err)
	}
	t.Setenv("JITLLM_MODELS", dir)
	none := "127.0.0.1:1"
	var note bytes.Buffer
	svc, path, err := runTarget(context.Background(), "story", none, "", false, nil, &note)
	if err != nil || svc != nil || path != filepath.Join(dir, "stories260K.jlm") {
		t.Fatalf("runTarget: %v %q %v", svc, path, err)
	}
	if !strings.Contains(note.String(), "no service answers at "+none) {
		t.Errorf("stderr: %q", note.String())
	}
	if _, _, err := runTarget(context.Background(), "nope", none, "", false, nil, &bytes.Buffer{}); err == nil ||
		!strings.Contains(err.Error(), dir) {
		t.Errorf("an unknown name: %v", err)
	}

	var got map[string]any
	s := fakeService(t, localserver.APIVersion, &got)
	if svc, path, _ := runTarget(context.Background(), filepath.Join(dir, "stories260K.jlm"), s.URL, "", false, nil, &bytes.Buffer{}); svc != nil || path == "" {
		t.Errorf("a path went to the service")
	}
	if svc, _, _ := runTarget(context.Background(), "story", s.URL, "", false, []string{"-devices"}, &bytes.Buffer{}); svc != nil {
		t.Errorf("-devices went to the service")
	}
	if svc, _, _ := runTarget(context.Background(), "story", s.URL, "", true, nil, &bytes.Buffer{}); svc != nil {
		t.Errorf("-embedded went to the service")
	}

	// And the run itself, in this process, by alias.
	if err := runCmd([]string{"-addr", none, "-n", "4", "story", "Once upon a time"}); err != nil {
		t.Fatalf("jitllm run story: %v", err)
	}
}
