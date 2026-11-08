package main

import (
	"bytes"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/localserver"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/server"
)

// TestJitllmRunGoesThroughTheService builds the real `jitllm` and runs it by
// alias against a real engine over a models folder: the service loads the
// model on demand and streams the answer, and jitllm says it used the
// service and starts no engine of its own.
func TestJitllmRunGoesThroughTheService(t *testing.T) {
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
	e := server.New(server.Config{Admit: testdevices.Admit(), ModelDir: dir, Version: "test", Probe: func() ([]server.DeviceInfo, error) {
		return []server.DeviceInfo{{ID: "cpu", Backend: "cpu", Kind: server.KindCPU, Available: true}}, nil
	}, OnDemand: server.LoadOptions{DeviceIDs: []string{"cpu"}}})
	defer e.Close()
	s := httptest.NewServer(e.Handler())
	defer s.Close()

	bin := filepath.Join(t.TempDir(), "jitllm")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	build := exec.Command("go", "build", "-o", bin, "./cmd/jitllm")
	build.Dir = filepath.Join("..", "..", "..")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building jitllm: %v\n%s", err, out)
	}
	var stdout, stderr bytes.Buffer
	run := exec.Command(bin, "run", "-addr", s.URL, "-n", "6", "story", "Once upon a time")
	// An empty models folder: the run must not find the model itself.
	run.Env = append(os.Environ(), "JITLLM_MODELS="+t.TempDir())
	run.Stdout, run.Stderr = &stdout, &stderr
	if err := run.Run(); err != nil {
		t.Fatalf("jitllm run: %v\n%s", err, stderr.String())
	}
	if !strings.Contains(stderr.String(), "using the service at "+s.URL) {
		t.Errorf("stderr does not say the service was used:\n%s", stderr.String())
	}
	if strings.Contains(stderr.String(), "pages ") || strings.Contains(stderr.String(), "memory   budget") {
		t.Errorf("jitllm started an engine of its own:\n%s", stderr.String())
	}
	if strings.TrimSpace(stdout.String()) == "" {
		t.Errorf("no completion on stdout")
	}
	if ms := e.Models(); len(ms) != 1 || ms[0].ID() != "stories260K" {
		t.Fatalf("the service holds %v, want stories260K loaded on demand", ms)
	}
}
