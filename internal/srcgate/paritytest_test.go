package srcgate

import (
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestNoLiveCodeImportsParitytest keeps main's kernel generators out of the
// engine: paritytest holds them only so tests can hold the composer's
// kernels to them, instruction by instruction. A non-test file outside
// paritytest that imports it is a kernel the engine launches without the
// composer forming it -- a fallback, which an op the composer cannot form
// does not get (it fails to generate, and its block does not place).
func TestNoLiveCodeImportsParitytest(t *testing.T) {
	testmodels.SourceTree(t)
	const pt = "github.com/jitllm/jitllm/paritytest"
	root := filepath.Join("..", "..")
	fset := token.NewFileSet()
	checked := 0
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if p != root && strings.HasPrefix(d.Name(), ".") {
				return filepath.SkipDir
			}
			switch d.Name() {
			case "gen", "vendor", "node_modules", "testdata":
				return filepath.SkipDir
			}
			if filepath.ToSlash(p) == filepath.ToSlash(filepath.Join(root, "paritytest")) {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			return nil
		}
		f, err := parser.ParseFile(fset, p, nil, parser.ImportsOnly)
		if err != nil {
			return err
		}
		checked++
		for _, im := range f.Imports {
			if v := strings.Trim(im.Path.Value, `"`); v == pt || strings.HasPrefix(v, pt+"/") {
				t.Errorf("%s imports %s: the engine launches only kernels the composer forms", p, v)
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
}
