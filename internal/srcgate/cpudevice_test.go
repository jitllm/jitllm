package srcgate

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// deviceKinds are the packages that implement a device: the instance reaches
// each through nn's faces, never their concrete types.
func deviceKinds() []string { return []string{"cpudev", "gpu", "cpu"} }

// TestTheCPUIsADevice holds the instance to reaching the CPU as it reaches a
// card: no field or method of engine.Instance is named for the CPU or the
// host, and no field holds a device package's concrete type -- in its own
// declaration or in a struct of the engine's it is declared as -- where a
// card is held through nn's faces. The violations -- a CPU-named method, a
// field of the CPU device's executor -- are seen.
func TestTheCPUIsADevice(t *testing.T) {
	testmodels.SourceTree(t)
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	types := map[string]*ast.StructType{}
	var inst *ast.StructType
	methods := 0
	ps, err := filepath.Glob(filepath.Join(root, "engine", "*.go"))
	if err != nil {
		t.Fatal(err)
	}
	var files []*ast.File
	for _, p := range ps {
		if strings.HasSuffix(p, "_test.go") {
			continue
		}
		src, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		f, err := parser.ParseFile(fset, p, src, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, f)
		for _, d := range f.Decls {
			g, ok := d.(*ast.GenDecl)
			if !ok {
				continue
			}
			for _, sp := range g.Specs {
				if ts, ok := sp.(*ast.TypeSpec); ok {
					if st, ok := ts.Type.(*ast.StructType); ok {
						types[ts.Name.Name] = st
						if ts.Name.Name == "Instance" {
							inst = st
						}
					}
				}
			}
		}
	}
	if inst == nil {
		t.Fatal("no engine.Instance: the gate read nothing")
	}
	for _, f := range files {
		for _, d := range f.Decls {
			fn, ok := d.(*ast.FuncDecl)
			if !ok || fn.Recv == nil || !onInstance(fn.Recv.List[0].Type) {
				continue
			}
			methods++
			if deviceNamed(fn.Name.Name) {
				t.Errorf("%s: method Instance.%s is named for the CPU or the host", fset.Position(fn.Pos()), fn.Name.Name)
			}
		}
	}
	if methods < 100 {
		t.Fatalf("read %d methods of Instance: the gate read nothing", methods)
	}
	fields := 0
	for _, fl := range inst.Fields.List {
		for _, n := range fl.Names {
			fields++
			if deviceNamed(n.Name) {
				t.Errorf("%s: field Instance.%s is named for the CPU or the host", fset.Position(n.Pos()), n.Name)
			}
			if k := concreteDevice(fl.Type, types, 0); k != "" {
				t.Errorf("%s: field Instance.%s holds %s's concrete type, not a device face", fset.Position(n.Pos()), n.Name, k)
			}
		}
	}
	if fields < 50 {
		t.Fatalf("read %d fields of Instance: the gate read nothing", fields)
	}
}

func onInstance(e ast.Expr) bool {
	if s, ok := e.(*ast.StarExpr); ok {
		e = s.X
	}
	id, ok := e.(*ast.Ident)
	return ok && id.Name == "Instance"
}

// deviceNamed says a name has a word that is the CPU or the host: cpuRun,
// hostChunk, HostBudget, but not occupy or ghost.
func deviceNamed(name string) bool {
	for _, w := range camelWords(name) {
		if w == "cpu" || w == "host" {
			return true
		}
	}
	return false
}

func camelWords(s string) []string {
	var out []string
	cur := []rune{}
	rs := []rune(s)
	for i, r := range rs {
		if i > 0 && unicode.IsUpper(r) && (unicode.IsLower(rs[i-1]) || i+1 < len(rs) && unicode.IsLower(rs[i+1])) {
			out = append(out, strings.ToLower(string(cur)))
			cur = cur[:0]
		}
		cur = append(cur, r)
	}
	return append(out, strings.ToLower(string(cur)))
}

// concreteDevice names the device package whose concrete type e holds, or
// "": a pointer, slice, array or map of one, or a struct of the engine's whose
// fields hold one, one level down.
func concreteDevice(e ast.Expr, types map[string]*ast.StructType, depth int) string {
	switch x := e.(type) {
	case *ast.StarExpr:
		return concreteDevice(x.X, types, depth)
	case *ast.ArrayType:
		return concreteDevice(x.Elt, types, depth)
	case *ast.MapType:
		return concreteDevice(x.Value, types, depth)
	case *ast.SelectorExpr:
		if id, ok := x.X.(*ast.Ident); ok {
			for _, k := range deviceKinds() {
				if id.Name == k {
					return k
				}
			}
		}
	case *ast.Ident:
		if st := types[x.Name]; st != nil && depth == 0 {
			for _, fl := range st.Fields.List {
				if k := concreteDevice(fl.Type, types, depth+1); k != "" {
					return k
				}
			}
		}
	}
	return ""
}
