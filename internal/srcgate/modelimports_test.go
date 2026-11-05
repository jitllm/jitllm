package srcgate

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestTheModelImportsNothingOfTheEngine holds docs/design/model-and-engine.md's
// boundary: the model (model/) is weights and the graphs it states, and the
// engine runs it, so nothing the model builds on -- directly or through the
// packages it imports -- is under engine/. The violation is a model file
// importing the engine's CPU device, as one did for its history layout.
func TestTheModelImportsNothingOfTheEngine(t *testing.T) {
	testmodels.SourceTree(t)
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	goTool := filepath.Join(runtime.GOROOT(), "bin", "go")
	if _, err := os.Stat(goTool); err != nil {
		if goTool, err = exec.LookPath("go"); err != nil {
			t.Fatalf("no go tool to list the model's imports with: %v", err)
		}
	}
	deps := func(t *testing.T, overlay string) []string {
		t.Helper()
		args := []string{"list", "-deps", "-f", "{{.ImportPath}}"}
		if overlay != "" {
			args = append(args, "-overlay", overlay)
		}
		cmd := exec.Command(goTool, append(args, "./model")...)
		cmd.Dir = root
		cmd.Env = append(os.Environ(), "CGO_ENABLED=0")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("go list -deps ./model: %v\n%s", err, out)
		}
		return strings.Fields(string(out))
	}
	engine := func(paths []string) []string {
		var in []string
		for _, p := range paths {
			if p == "github.com/jitllm/jitllm/engine" || strings.HasPrefix(p, "github.com/jitllm/jitllm/engine/") {
				in = append(in, p)
			}
		}
		return in
	}
	all := deps(t, "")
	if len(all) == 0 || all[len(all)-1] != "github.com/jitllm/jitllm/model" {
		t.Fatalf("go list -deps ./model listed %d packages, not ending at the model: the gate read nothing", len(all))
	}
	if in := engine(all); len(in) > 0 {
		t.Errorf("the model builds on %d package(s) of the engine: %s", len(in), strings.Join(in, ", "))
	}

	// The violation: one model file importing the CPU device's package.
	tmp := t.TempDir()
	src := filepath.Join(tmp, "violation.go")
	if err := os.WriteFile(src, []byte("package model\n\nimport _ \"github.com/jitllm/jitllm/engine/device/cpu\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	ov := filepath.Join(tmp, "overlay.json")
	j := `{"Replace":{"` + filepath.ToSlash(filepath.Join(root, "model", "zz_violation.go")) + `":"` + filepath.ToSlash(src) + `"}}`
	if err := os.WriteFile(ov, []byte(j), 0o644); err != nil {
		t.Fatal(err)
	}
	if in := engine(deps(t, ov)); len(in) == 0 {
		t.Error("a model file importing engine/device/cpu was not seen: the gate cannot fail")
	}
}
