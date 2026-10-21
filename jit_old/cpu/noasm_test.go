package cpu

import (
	"io/fs"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestNoAssemblyBesidesTheTrampolines: this project contains no pregenerated .s
// file; every kernel is emitted at run time for the host. The two allowed
// files are the Go-to-generated-code call boundary, which Go can only express
// in its own assembler.
func TestNoAssemblyBesidesTheTrampolines(t *testing.T) {
	testmodels.SourceTree(t)
	allowed := map[string]bool{
		"jit_old/cpu/trampoline_amd64.s": true,
		"jit_old/cpu/trampoline_arm64.s": true,
	}
	root := filepath.Join("..", "..")
	walked, seen := 0, 0
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
			case "node_modules":
				return filepath.SkipDir
			}
			return nil
		}
		walked++
		switch strings.ToLower(filepath.Ext(p)) {
		case ".s", ".asm":
		default:
			return nil
		}
		rel, _ := filepath.Rel(root, p)
		rel = filepath.ToSlash(rel)
		if !allowed[rel] {
			t.Errorf("%s: no pregenerated assembly besides the trampolines -- emit it at run time", rel)
		}
		seen++
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if walked < 500 || seen < len(allowed) {
		t.Fatalf("walked %d files and saw %d assembly files from %s -- the walk is not seeing the module", walked, seen, root)
	}
}
