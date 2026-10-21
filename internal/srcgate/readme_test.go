package srcgate

import (
	"encoding/json"
	"fmt"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestTheREADMEGoExamplesBuild compiles every ```go block of the Go library
// guide (docs/embedding-go.md, which the README links for embedding) as the
// main package it claims to be, against the tree it sits in. A renamed or
// re-signatured API the guide still calls fails here rather than on the first
// reader who pastes it.
//
// Each block is laid into the module through `go build -overlay` at a
// directory that does not exist on disk, so the build sees the example as a
// package of this module and nothing is written into the tree.
func TestTheREADMEGoExamplesBuild(t *testing.T) {
	testmodels.SourceTree(t)
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	readme, err := os.ReadFile(filepath.Join(root, "docs", "embedding-go.md"))
	if err != nil {
		t.Fatal(err)
	}
	blocks := goBlocks(string(readme))
	if len(blocks) == 0 {
		t.Fatal("docs/embedding-go.md has no ```go block, so this gate proved nothing")
	}
	goTool := filepath.Join(runtime.GOROOT(), "bin", "go")
	if _, err := os.Stat(goTool); err != nil {
		if goTool, err = exec.LookPath("go"); err != nil {
			t.Fatalf("no go tool to build the README's examples with: %v", err)
		}
	}
	tmp := t.TempDir()
	for i := range blocks {
		src := filepath.Join(tmp, fmt.Sprintf("example%d.go", i))
		if err := os.WriteFile(src, []byte(blocks[i]), 0o644); err != nil {
			t.Fatal(err)
		}
		pkg := filepath.Join(root, "internal", "srcgate", fmt.Sprintf("readmeexample%d", i))
		overlay, err := json.Marshal(map[string]map[string]string{
			"Replace": {filepath.Join(pkg, "main.go"): src},
		})
		if err != nil {
			t.Fatal(err)
		}
		ov := filepath.Join(tmp, fmt.Sprintf("overlay%d.json", i))
		if err := os.WriteFile(ov, overlay, 0o644); err != nil {
			t.Fatal(err)
		}
		cmd := exec.Command(goTool, "build", "-overlay", ov, "-o", filepath.Join(tmp, fmt.Sprintf("example%d", i)), "./"+filepath.ToSlash(mustRel(t, root, pkg)))
		cmd.Dir = root
		cmd.Env = append(os.Environ(), "CGO_ENABLED=0")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Errorf("README.md's go block %d does not build: %v\n%s\n--- the block:\n%s", i+1, err, out, blocks[i])
		}
	}
}

// goBlocks is the body of every fenced ```go block in md.
func goBlocks(md string) []string {
	var out []string
	var cur *strings.Builder
	for line := range strings.Lines(md) {
		trimmed := strings.TrimSpace(line)
		switch {
		case cur == nil && trimmed == "```go":
			cur = &strings.Builder{}
		case cur != nil && trimmed == "```":
			out = append(out, cur.String())
			cur = nil
		case cur != nil:
			cur.WriteString(line)
		}
	}
	return out
}

func mustRel(t *testing.T, base, p string) string {
	t.Helper()
	r, err := filepath.Rel(base, p)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// TestNoCgo holds the README's "no cgo": no Go file in any of the three
// modules (the engine, server/, ui/) imports "C", tests and generated code
// included. CI's CGO_ENABLED=0 cross-build alone does not show it: a cgo
// file is then left out of the build rather than refused, and the build
// passes without whatever that file provided.
func TestNoCgo(t *testing.T) {
	testmodels.SourceTree(t)
	root := filepath.Join("..", "..")
	fset := token.NewFileSet()
	n := 0
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if p != root && (strings.HasPrefix(d.Name(), ".") || d.Name() == "testdata" ||
				d.Name() == "node_modules" || d.Name() == "models") {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, ".go") {
			return nil
		}
		f, err := parser.ParseFile(fset, p, nil, parser.ImportsOnly)
		if err != nil {
			return err
		}
		n++
		for _, im := range f.Imports {
			if path, _ := strconv.Unquote(im.Path.Value); path == "C" {
				t.Errorf("%s imports \"C\": the engine, server and app build with CGO_ENABLED=0 and open "+
					"native libraries at run time through goffi (jit_old/gpu/ffi)", fset.Position(im.Pos()))
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if n < 500 {
		t.Fatalf("parsed %d Go files under %s: the tree was not found, so this gate proved nothing", n, root)
	}
}

// TestTheCoreModuleHasOneDependency holds the README's "the core module's one
// dependency is goffi": the root go.mod requires exactly that module. server/
// and ui/ are modules of their own and may require more.
func TestTheCoreModuleHasOneDependency(t *testing.T) {
	testmodels.SourceTree(t)
	b, err := os.ReadFile(filepath.Join("..", "..", "go.mod"))
	if err != nil {
		t.Fatal(err)
	}
	var reqs []string
	block := false
	for line := range strings.Lines(string(b)) {
		f := strings.Fields(line)
		switch {
		case len(f) == 0 || strings.HasPrefix(f[0], "//"):
		case block && f[0] == ")":
			block = false
		case block:
			reqs = append(reqs, f[0])
		case f[0] == "require" && len(f) >= 2 && f[1] == "(":
			block = true
		case f[0] == "require" && len(f) >= 2:
			reqs = append(reqs, f[1])
		}
	}
	if len(reqs) != 1 || reqs[0] != "github.com/go-webgpu/goffi" {
		t.Fatalf("the core module requires %v; the README says its one dependency is github.com/go-webgpu/goffi", reqs)
	}
}
