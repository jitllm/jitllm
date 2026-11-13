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
// card is held through nn's faces. What it embeds of the engine's (a
// Session) is its own: those fields and methods are promoted onto it. The
// violations -- a CPU-named method, a host-named field, a field of the CPU
// device's executor, and an embedded struct's CPU-named field and
// host-named method -- run through the same check and are each seen.
func TestTheCPUIsADevice(t *testing.T) {
	testmodels.SourceTree(t)
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
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
	}
	found, methods, fields := instanceFindings(fset, files)
	if methods < 100 || fields < 50 {
		t.Fatalf("read %d methods and %d fields of Instance: the gate read nothing", methods, fields)
	}
	for _, f := range found {
		t.Error(f)
	}
	t.Logf("%d methods and %d fields of engine.Instance: none named for the CPU or the host, none a device's concrete type",
		methods, fields)

	const bad = `package engine
type Instance struct {
	*Session
	hostRows []float32
	x *cpudev.Exec
}
type Session struct{ cpuPages []byte }
func (s *Instance) cpuChunk() {}
func (s *Session) hostSync() {}
`
	vf, err := parser.ParseFile(fset, "violation.go", bad, parser.SkipObjectResolution)
	if err != nil {
		t.Fatal(err)
	}
	vs, _, _ := instanceFindings(fset, []*ast.File{vf})
	t.Logf("violations: %d seen of 5", len(vs))
	if len(vs) != 5 {
		t.Fatalf("VIOLATION: a CPU-named method, a host-named field, a CPU executor field and an embedded struct's "+
			"CPU-named field and host-named method read as %d findings, want 5: %v", len(vs), vs)
	}
}

// instanceFindings is every finding over files' engine.Instance and the
// engine's structs it embeds -- a field or method named for the CPU or the
// host, a field of a device's concrete type -- and the methods and fields it
// read.
func instanceFindings(fset *token.FileSet, files []*ast.File) (found []string, methods, fields int) {
	types := map[string]*ast.StructType{}
	var inst *ast.StructType
	for _, f := range files {
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
		return found, 0, 0
	}
	// own is Instance and the engine's structs it embeds, whose fields and
	// methods are the instance's.
	own := map[string]*ast.StructType{"Instance": inst}
	for _, fl := range inst.Fields.List {
		if len(fl.Names) == 0 {
			if n := typeName(fl.Type); types[n] != nil {
				own[n] = types[n]
			}
		}
	}
	for _, f := range files {
		for _, d := range f.Decls {
			fn, ok := d.(*ast.FuncDecl)
			if !ok || fn.Recv == nil || own[typeName(fn.Recv.List[0].Type)] == nil {
				continue
			}
			methods++
			if deviceNamed(fn.Name.Name) {
				found = append(found, fset.Position(fn.Pos()).String()+": method "+typeName(fn.Recv.List[0].Type)+"."+fn.Name.Name+
					" (the instance's) is named for the CPU or the host")
			}
		}
	}
	for tn, st := range own {
		for _, fl := range st.Fields.List {
			for _, n := range fl.Names {
				fields++
				if deviceNamed(n.Name) {
					found = append(found, fset.Position(n.Pos()).String()+": field "+tn+"."+n.Name+" (the instance's) is named for the CPU or the host")
				}
				if k := concreteDevice(fl.Type, types, 0); k != "" {
					found = append(found, fset.Position(n.Pos()).String()+": field "+tn+"."+n.Name+" (the instance's) holds "+k+
						"'s concrete type, not a device face")
				}
			}
		}
	}
	return found, methods, fields
}

// typeName is the name of a type or a pointer to one, "" for anything else.
func typeName(e ast.Expr) string {
	if s, ok := e.(*ast.StarExpr); ok {
		e = s.X
	}
	if id, ok := e.(*ast.Ident); ok {
		return id.Name
	}
	return ""
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
