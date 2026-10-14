package testmodels_test

import (
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestNoProductionCodeImportsTestModels walks every non-test Go file under the
// repository, the nested ui/ and server/ modules included, and fails if one
// imports this package. A shipped binary that found its models through a test
// helper would be reading JITLLM_MODELS for a reason nobody documented.
func TestNoProductionCodeImportsTestModels(t *testing.T) {
	testmodels.SourceTree(t)
	const me = "github.com/jitllm/jitllm/internal/testmodels"
	root := filepath.Join("..", "..")
	fset := token.NewFileSet()
	checked := 0
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			// A dot-directory (.git, a local tool's worktrees) is not this
			// tree's source.
			if p != root && strings.HasPrefix(d.Name(), ".") {
				return filepath.SkipDir
			}
			switch d.Name() {
			case "gen", "vendor", "node_modules", "testdata", "models":
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			return nil
		}
		if strings.Contains(filepath.ToSlash(p), "internal/testmodels/") {
			return nil
		}
		f, err := parser.ParseFile(fset, p, nil, parser.ImportsOnly)
		if err != nil {
			return err
		}
		checked++
		for _, im := range f.Imports {
			if ip := strings.Trim(im.Path.Value, `"`); ip == me || strings.HasPrefix(ip, me+"/") {
				t.Errorf("%s imports %s: it is for tests only", p, me)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if checked < 100 {
		t.Fatalf("walked only %d files from %s -- the walk is not seeing the module", checked, root)
	}
	t.Logf("%d production files, none importing %s", checked, me)
}

// TestDirHonoursTheVariableAndFindsTheRoot pins both arms of Dir: the variable
// wins, and without it the answer is the repository's models/ from any depth.
func TestDirHonoursTheVariableAndFindsTheRoot(t *testing.T) {
	t.Setenv(testmodels.Env, "/somewhere/else")
	if d := testmodels.Dir(); d != "/somewhere/else" {
		t.Fatalf("%s=/somewhere/else: Dir() = %q", testmodels.Env, d)
	}
	if p := testmodels.Resolve("x.jlm"); p != filepath.Join("/somewhere/else", "x.jlm") {
		t.Fatalf("Resolve(bare name) = %q, want it inside the directory", p)
	}
	if p := testmodels.Resolve("./x.jlm"); p != "./x.jlm" {
		t.Fatalf("Resolve(path) = %q, want it untouched", p)
	}

	// The rest finds the repository root, which a shipped binary has none of.
	testmodels.SourceTree(t)
	t.Setenv(testmodels.Env, "")
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(root, "models")
	if d := testmodels.Dir(); d != want {
		t.Fatalf("unset: Dir() = %q, want %q", d, want)
	}
	// From a nested module the first go.mod is the wrong one.
	t.Chdir(filepath.Join(root, "ui"))
	if d := testmodels.Dir(); d != want {
		t.Fatalf("from ui/: Dir() = %q, want %q (the root module's, not ui's)", d, want)
	}
	if _, err := os.Stat(filepath.Join(root, "ui", "go.mod")); err != nil {
		t.Fatalf("ui/ is no longer a nested module, so this case proves nothing: %v", err)
	}
}
