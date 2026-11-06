package srcgate

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestBlockLocationIsWrittenByItsSetters keeps a block's location in one set
// of hands: Instance.blockRoutes (the route each block runs on) and Instance.gpuLayers
// (the card's prefix) are written only by the setters that keep the two in
// step and hand a block over between its routes (place, placeRange, clearOnDev, shrinkOnDev). A write
// anywhere else -- an assignment, an increment, a composite literal's key --
// is a placement that skips the hand-over.
func TestBlockLocationIsWrittenByItsSetters(t *testing.T) {
	testmodels.SourceTree(t)
	dir := filepath.Join("..", "..", "engine")
	setters := map[string]bool{"place": true, "placeRange": true, "clearOnDev": true, "shrinkOnDev": true}
	fields := map[string]bool{"blockRoutes": true, "gpuLayers": true}
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	checked, writes := 0, 0
	var found []string
	isField := func(e ast.Expr) bool {
		if ix, ok := e.(*ast.IndexExpr); ok {
			e = ix.X
		}
		sel, ok := e.(*ast.SelectorExpr)
		return ok && fields[sel.Sel.Name]
	}
	for _, de := range ents {
		name := de.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, filepath.Join(dir, name), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		checked++
		for _, d := range f.Decls {
			fn, ok := d.(*ast.FuncDecl)
			allowed := ok && fn.Recv != nil && setters[fn.Name.Name]
			ast.Inspect(d, func(n ast.Node) bool {
				var hit ast.Node
				switch n := n.(type) {
				case *ast.AssignStmt:
					for _, l := range n.Lhs {
						if isField(l) {
							hit = l
						}
					}
				case *ast.IncDecStmt:
					if isField(n.X) {
						hit = n.X
					}
				case *ast.KeyValueExpr:
					if k, ok := n.Key.(*ast.Ident); ok && fields[k.Name] {
						hit = n
					}
				}
				if hit == nil {
					return true
				}
				if allowed {
					writes++
				} else {
					found = append(found, fset.Position(hit.Pos()).String())
				}
				return true
			})
		}
	}
	if checked < 50 {
		t.Fatalf("checked %d files under %s: the walk found too little to mean anything", checked, dir)
	}
	if writes == 0 {
		t.Fatalf("the setters write neither field: the gate is matching nothing")
	}
	if len(found) > 0 {
		t.Fatalf("a block's location is written outside its setters (%d sites):\n%s",
			len(found), strings.Join(found, "\n"))
	}
	t.Logf("%d files, %d writes, all in the setters", checked, writes)
}
