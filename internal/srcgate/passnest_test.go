package srcgate

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// buildsFromDefinitions is what a pass, and the lowering after the last
// pass, must never do: read a primitive's definition, generate a primitive's
// naive IR, or instance a primitive's expressions. A pass takes IR and
// returns rewritten IR.
var buildsFromDefinitions = regexp.MustCompile(`\bgenerate\(|\.instance\(|\.Prim\b`)

// passFiles are the composer's files outside the decompositions and the
// naive IR's generation.
func passFiles(t *testing.T, dir string) map[string]string {
	t.Helper()
	keep := map[string]bool{"generate.go": true, "compose.go": true, "prim.go": true}
	out := map[string]string{}
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range ents {
		n := e.Name()
		if e.IsDir() || !strings.HasSuffix(n, ".go") || strings.HasSuffix(n, "_test.go") || keep[n] || strings.HasPrefix(n, "decompose") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, n))
		if err != nil {
			t.Fatal(err)
		}
		out[n] = string(b)
	}
	return out
}

// definitionBuilders is every pass file still building kernels from
// primitives' definitions: the rework's list, which only shrinks.
var definitionBuilders = map[string]bool{}

func TestPassesRewriteNests(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	var found []string
	for name, src := range passFiles(t, filepath.Join(root, "jit")) {
		if buildsFromDefinitions.MatchString(src) {
			found = append(found, name)
			if !definitionBuilders[name] {
				t.Errorf("jit/%s builds a kernel from a primitive's definition: a pass rewrites nests", name)
			}
		}
	}
	sort.Strings(found)
	for name := range definitionBuilders {
		if !strings.Contains(strings.Join(found, " "), name) {
			t.Errorf("jit/%s no longer builds from definitions: take it off the list", name)
		}
	}
	t.Logf("passes still building from definitions: %v", found)
}
