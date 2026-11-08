package main

import (
	"bytes"
	"context"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/server"
)

// TestModelsSetsAliasesAndDisables: `jitllmd models -alias/-disable/-enable`
// reach the server's models folder and `jitllmd models` shows them.
func TestModelsSetsAliasesAndDisables(t *testing.T) {
	b, err := os.ReadFile(testmodels.Path("stories260K.jlm"))
	if err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS) -- this gate proved nothing", err)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "stories260K.jlm"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	e := server.New(server.Config{Admit: testdevices.Admit(), ModelDir: dir, Version: "test", Probe: func() ([]server.DeviceInfo, error) {
		return []server.DeviceInfo{{ID: "cpu", Backend: "cpu", Kind: server.KindCPU, Available: true}}, nil
	}})
	defer e.Close()
	s := httptest.NewServer(e.Handler())
	defer s.Close()

	run := func(args ...string) string {
		var out bytes.Buffer
		if err := cmdModels(context.Background(), cli{out: &out, err: &out}, append([]string{"-addr", s.URL}, args...)); err != nil {
			t.Fatalf("jitllmd models %v: %v\n%s", args, err, out.String())
		}
		return out.String()
	}
	if out := run("-alias", "stories260K=tiny,story"); !strings.Contains(out, "aliases: story, tiny") {
		t.Fatalf("-alias:\n%s", out)
	}
	if out := run("-disable", "tiny"); !strings.Contains(out, "disabled") {
		t.Fatalf("-disable:\n%s", out)
	}
	if out := run(); !strings.Contains(out, "disabled; aliases story, tiny") {
		t.Fatalf("the list does not show the settings:\n%s", out)
	}
	if out := run("-enable", "stories260K", "-unalias", "stories260K=story"); !strings.Contains(out, "enabled  aliases: tiny") {
		t.Fatalf("-enable and -unalias:\n%s", out)
	}
}
