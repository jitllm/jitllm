package srcgate

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestNoGoHalfConversionOnTheKVPath keeps binary16 conversion on the engine's
// run-time paths in generated code (RULE 8): no production file under engine/
// or jit_old/gpu/tier may call quant.EncodeHalf or quant.DecodeHalf. The f16 KV
// cache's store and a device's packed-V pages go through nn.NarrowF16 and
// nn.WidenF16; the Go functions stay the reference those kernels are gated
// against (jit_old/cpu/f16narrow_test.go) and the converter's arithmetic, which
// runs once per file, not per token.
func TestNoGoHalfConversionOnTheKVPath(t *testing.T) {
	testmodels.SourceTree(t)
	root := filepath.Join("..", "..")
	fset := token.NewFileSet()
	checked := 0
	var found []string
	for _, dir := range []string{"engine", filepath.Join("jit_old", "gpu", "tier")} {
		err := filepath.WalkDir(filepath.Join(root, dir), func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				if d.Name() == "testdata" {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
				return nil
			}
			f, err := parser.ParseFile(fset, p, nil, 0)
			if err != nil {
				return err
			}
			checked++
			ast.Inspect(f, func(n ast.Node) bool {
				sel, ok := n.(*ast.SelectorExpr)
				if !ok {
					return true
				}
				if x, ok := sel.X.(*ast.Ident); ok && x.Name == "quant" &&
					(sel.Sel.Name == "EncodeHalf" || sel.Sel.Name == "DecodeHalf") {
					found = append(found, fset.Position(sel.Pos()).String()+": quant."+sel.Sel.Name)
				}
				return true
			})
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if checked < 100 {
		t.Fatalf("checked %d files: the tree was not found, so this gate proved nothing", checked)
	}
	for _, f := range found {
		t.Errorf("%s converts binary16 in Go; use nn.NarrowF16 / nn.WidenF16", f)
	}
}
