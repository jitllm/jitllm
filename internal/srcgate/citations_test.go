package srcgate

import (
	"fmt"
	"go/parser"
	"go/token"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestNoCodeCitesAnotherEngine holds the comments of production Go code to
// no other inference engine's name: a choice is justified by the model's own
// reference, a vendor's documentation or jitllm's measurement, and a comment
// citing another engine reads to the next change as the authority to copy.
// Outside it are what drives another engine on purpose -- the completion
// shim's two files, the oracles (internal/oracle), the scripts and dev tools
// -- tests, which name their oracles, and generated files, whose provenance
// line names their source.
func TestNoCodeCitesAnotherEngine(t *testing.T) {
	testmodels.SourceTree(t)
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	engines := regexp.MustCompile(`(?i)llama\.cpp|llama-(server|cli|bench|completion|eval|mtmd)|\bllama_[a-z]|common_(sampler|params)|\bvllm\b|\bollama\b|sglang|exllama|tensorrt-llm|mistral\.rs`)
	shim := []string{"server/compat_llamacpp.go", "server/llamacpp_engine.go"}
	fset := token.NewFileSet()
	parsed := 0
	var found []string
	for _, f := range trackedFiles(t, root) {
		parts := strings.Split(f, "/")
		if !strings.HasSuffix(f, ".go") || strings.HasSuffix(f, "_test.go") || slices.Contains(parts, "gen") ||
			slices.Contains(parts, "testdata") || slices.Contains(shim, f) ||
			parts[0] == "scripts" || parts[0] == "dev" || strings.HasPrefix(f, "internal/oracle/") {
			continue
		}
		af, err := parser.ParseFile(fset, filepath.Join(root, f), nil, parser.ParseComments|parser.SkipObjectResolution)
		if err != nil {
			t.Errorf("%s: %v", f, err)
			continue
		}
		parsed++
		generated := false
		for _, cg := range af.Comments {
			for _, c := range cg.List {
				if strings.HasPrefix(c.Text, "// Code generated ") && strings.HasSuffix(c.Text, " DO NOT EDIT.") {
					generated = true
				}
			}
		}
		if generated {
			continue
		}
		for _, cg := range af.Comments {
			for _, c := range cg.List {
				if m := engines.FindString(c.Text); m != "" {
					found = append(found, fmt.Sprintf("%s:%d: %q", f, fset.Position(c.Pos()).Line, m))
				}
			}
		}
	}
	if parsed < 500 {
		t.Fatalf("parsed %d Go files under %s: the tree was not found, so this gate proved nothing", parsed, root)
	}
	if len(found) > 0 {
		t.Errorf("%d comments in production code name another inference engine; say what the code computes, or the model's reference, instead:\n%s",
			len(found), strings.Join(found, "\n"))
	}
}
