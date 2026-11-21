package srcgate

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestNoPackageVars holds every tracked Go file, a test's too, to no
// package-level variable: state belongs to the instance that owns it, a
// table or an operand is built where it is used, a sentinel error is a
// constant. The one form allowed is a //go:embed variable, which the
// language accepts at package scope only; buf's generated code (a gen
// directory) is left as buf writes it.
func TestNoPackageVars(t *testing.T) {
	testmodels.SourceTree(t)
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	files := trackedFiles(t, root)
	fset := token.NewFileSet()
	parsed := 0
	var found []string
	for _, f := range files {
		if !strings.HasSuffix(f, ".go") || slices.Contains(strings.Split(f, "/"), "gen") || slices.Contains(strings.Split(f, "/"), "testdata") {
			continue
		}
		af, err := parser.ParseFile(fset, filepath.Join(root, f), nil, parser.ParseComments|parser.SkipObjectResolution)
		if err != nil {
			t.Errorf("%s: %v", f, err)
			continue
		}
		parsed++
		for _, d := range af.Decls {
			g, ok := d.(*ast.GenDecl)
			if !ok || g.Tok != token.VAR {
				continue
			}
			for _, s := range g.Specs {
				vs := s.(*ast.ValueSpec)
				if embeds(g.Doc) || embeds(vs.Doc) {
					continue
				}
				for _, n := range vs.Names {
					found = append(found, fmt.Sprintf("%s:%d: %s", f, fset.Position(n.Pos()).Line, n.Name))
				}
			}
		}
	}
	if parsed < 500 {
		t.Fatalf("parsed %d Go files under %s: the tree was not found, so this gate proved nothing", parsed, root)
	}
	if len(found) > 0 {
		t.Errorf("%d package-level variables in %d Go files:\n%s", len(found), parsed, strings.Join(found, "\n"))
	}
}

// embeds reports a //go:embed directive in a declaration's comments.
func embeds(cg *ast.CommentGroup) bool {
	if cg == nil {
		return false
	}
	for _, c := range cg.List {
		if strings.HasPrefix(c.Text, "//go:embed ") {
			return true
		}
	}
	return false
}
