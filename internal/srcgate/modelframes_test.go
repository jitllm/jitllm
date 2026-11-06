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

// TestTheModelHoldsNoFrames holds docs/design/instance-paging.md's shape:
// the model is the weights as a file, and the frames a block's pages are
// read into are the frames of the device running it. No type of model/ has
// a field holding a jlm.Frames -- itself, or inside a pointer, slice, map,
// channel or func type -- and the violation, a struct holding a map of
// them, is seen.
func TestTheModelHoldsNoFrames(t *testing.T) {
	testmodels.SourceTree(t)
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, filepath.Join(root, "model"), func(fi fs.FileInfo) bool {
		return !strings.HasSuffix(fi.Name(), "_test.go")
	}, 0)
	if err != nil {
		t.Fatal(err)
	}
	structs := 0
	for _, p := range pkgs {
		for _, f := range p.Files {
			got, n := framesFields(fset, f)
			structs += n
			for _, at := range got {
				t.Errorf("%s: a model type holds a jlm.Frames", at)
			}
		}
	}
	if structs < 10 {
		t.Fatalf("read %d struct types of model/: the gate read nothing", structs)
	}

	src := "package model\n\nimport \"github.com/jitllm/jitllm/format/jlm\"\n\ntype held struct{ frames map[int]*jlm.Frames }\n"
	f, err := parser.ParseFile(fset, "violation.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := framesFields(fset, f); len(got) == 0 {
		t.Error("a struct holding a map of jlm.Frames was not seen: the gate cannot fail")
	}
}

// framesFields is where f's struct types hold a jlm.Frames, and how many
// struct types it has.
func framesFields(fset *token.FileSet, f *ast.File) ([]string, int) {
	name := ""
	for _, im := range f.Imports {
		if strings.Trim(im.Path.Value, `"`) == "github.com/jitllm/jitllm/format/jlm" {
			name = "jlm"
			if im.Name != nil {
				name = im.Name.Name
			}
		}
	}
	var out []string
	n := 0
	ast.Inspect(f, func(x ast.Node) bool {
		st, ok := x.(*ast.StructType)
		if !ok {
			return true
		}
		n++
		for _, fl := range st.Fields.List {
			ast.Inspect(fl.Type, func(y ast.Node) bool {
				se, ok := y.(*ast.SelectorExpr)
				if !ok {
					return true
				}
				if id, ok := se.X.(*ast.Ident); ok && name != "" && id.Name == name && se.Sel.Name == "Frames" {
					out = append(out, fset.Position(fl.Pos()).String())
				}
				return true
			})
		}
		return true
	})
	return out, n
}
