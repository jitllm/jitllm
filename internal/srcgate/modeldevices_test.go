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

// TestTheModelHoldsNoDevice holds docs/design/instance-paging.md's step 5: the
// model holds nothing of a device. No type of model/ has a field whose type
// names one of nn's interfaces -- the devices' vocabulary -- itself or inside
// a pointer, slice, map, channel or func type, and Model.Close calls nothing
// but its container's Close: what a device keeps of the model goes with the
// last instance holding it there. The violations, a registry of host-memory
// devices and a Close that releases them, are seen.
func TestTheModelHoldsNoDevice(t *testing.T) {
	testmodels.SourceTree(t)
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	notTest := func(fi fs.FileInfo) bool { return !strings.HasSuffix(fi.Name(), "_test.go") }
	nnPkgs, err := parser.ParseDir(fset, filepath.Join(root, "nn"), notTest, 0)
	if err != nil {
		t.Fatal(err)
	}
	ifaces := map[string]bool{}
	for _, p := range nnPkgs {
		for _, f := range p.Files {
			for _, d := range f.Decls {
				gd, ok := d.(*ast.GenDecl)
				if !ok {
					continue
				}
				for _, sp := range gd.Specs {
					if ts, ok := sp.(*ast.TypeSpec); ok {
						if _, ok := ts.Type.(*ast.InterfaceType); ok {
							ifaces[ts.Name.Name] = true
						}
					}
				}
			}
		}
	}
	if len(ifaces) < 10 || !ifaces["LayerDevice"] || !ifaces["HostReserver"] {
		t.Fatalf("read %d of nn's interfaces: the gate read nothing", len(ifaces))
	}

	pkgs, err := parser.ParseDir(fset, filepath.Join(root, "model"), notTest, 0)
	if err != nil {
		t.Fatal(err)
	}
	structs, closes := 0, 0
	for _, p := range pkgs {
		for _, f := range p.Files {
			got, n := deviceFields(fset, f, ifaces)
			structs += n
			for _, at := range got {
				t.Errorf("%s: a model type holds a device", at)
			}
			for _, bad := range closeCalls(fset, f) {
				t.Errorf("%s: Model.Close calls more than its container's Close", bad)
			}
			closes += closeDecls(f)
		}
	}
	if structs < 10 || closes != 1 {
		t.Fatalf("read %d struct types and %d Model.Close of model/: the gate read nothing", structs, closes)
	}

	src := `package model

import "github.com/jitllm/jitllm/nn"

type pageAsk struct{ devs map[nn.HostReserver]int }

type Model struct {
	Container interface{ Close() error }
	owners    []nn.ModelOwner
}

func (m *Model) Close() error {
	for _, o := range m.owners {
		o.ReleaseModel(0)
	}
	return m.Container.Close()
}
`
	f, err := parser.ParseFile(fset, "violation.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := deviceFields(fset, f, ifaces); len(got) != 2 {
		t.Errorf("a registry of host-memory devices and a slice of owners: %d seen, want 2: the gate cannot fail", len(got))
	}
	if len(closeCalls(fset, f)) == 0 {
		t.Error("a Close releasing the devices was not seen: the gate cannot fail")
	}
}

// deviceFields is where f's struct types hold one of nn's interfaces, and how
// many struct types it has.
func deviceFields(fset *token.FileSet, f *ast.File, ifaces map[string]bool) ([]string, int) {
	name := ""
	for _, im := range f.Imports {
		if strings.Trim(im.Path.Value, `"`) == "github.com/jitllm/jitllm/nn" {
			name = "nn"
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
				if id, ok := se.X.(*ast.Ident); ok && name != "" && id.Name == name && ifaces[se.Sel.Name] {
					out = append(out, fset.Position(fl.Pos()).String()+" nn."+se.Sel.Name)
				}
				return true
			})
		}
		return true
	})
	return out, n
}

// closeDecls counts f's methods named Close on *Model.
func closeDecls(f *ast.File) int {
	n := 0
	for _, d := range f.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && isModelClose(fd) {
			n++
		}
	}
	return n
}

// closeCalls is every call in Model.Close but its container's Close.
func closeCalls(fset *token.FileSet, f *ast.File) []string {
	var out []string
	for _, d := range f.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || !isModelClose(fd) || fd.Body == nil {
			continue
		}
		ast.Inspect(fd.Body, func(x ast.Node) bool {
			c, ok := x.(*ast.CallExpr)
			if !ok {
				return true
			}
			if se, ok := c.Fun.(*ast.SelectorExpr); ok && se.Sel.Name == "Close" {
				if in, ok := se.X.(*ast.SelectorExpr); ok && in.Sel.Name == "Container" {
					return true
				}
			}
			out = append(out, fset.Position(c.Pos()).String())
			return true
		})
	}
	return out
}

func isModelClose(fd *ast.FuncDecl) bool {
	if fd.Name.Name != "Close" || fd.Recv == nil || len(fd.Recv.List) != 1 {
		return false
	}
	st, ok := fd.Recv.List[0].Type.(*ast.StarExpr)
	if !ok {
		return false
	}
	id, ok := st.X.(*ast.Ident)
	return ok && id.Name == "Model"
}
