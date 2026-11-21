package srcgate

import (
	"bytes"
	"encoding/json"
	"errors"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// kernelText is what marks a string as device code: PTX's header and entry,
// Metal's library and entry, GLSL's version and group size, CUDA C's entry,
// SPIR-V's assembly and LLVM's AMDGPU entry.
var kernelText = []string{
	".version ", ".target sm_", ".entry ",
	"metal_stdlib", "kernel void",
	"#version ", "local_size_x",
	"__global__",
	"OpCapability", "OpEntryPoint",
	"amdgpu_kernel",
}

// spirvMagic is a SPIR-V module's first word.
const spirvMagic = 0x07230203

// TestNoKernelTextOutsideTheComposer holds RULE 8 where
// TestNoAssemblyBesidesTheTrampolines cannot see: inside Go. Outside
// composer/, no file that ships holds device code in a string or a SPIR-V
// module's first word in a number; a kernel written into a literal is a
// pregenerated kernel whatever its file is called.
func TestNoKernelTextOutsideTheComposer(t *testing.T) {
	testmodels.SourceTree(t)
	for _, c := range []struct {
		name, src string
		holds     bool
	}{
		{"PTX in a string", "package p\nconst k = `.version 7.8\n.target sm_70\n.visible .entry f() { ret; }`\n", true},
		{"Metal in a string", "package p\nvar k = \"#include <metal_stdlib>\\nkernel void f() {}\"\n", true},
		{"a SPIR-V word list", "package p\nvar k = []uint32{0x07230203, 0x00010300}\n", true},
		{"a message naming no kernel", "package p\nvar k = \"compiled the kernel\"\n", false},
	} {
		af, err := parser.ParseFile(token.NewFileSet(), "planted.go", c.src, parser.SkipObjectResolution)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if got := len(kernelLiterals(af)) > 0; got != c.holds {
			t.Fatalf("planted %s: found %v, want %v: the gate cannot tell", c.name, got, c.holds)
		}
	}
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	parsed := 0
	var found []string
	for _, f := range trackedFiles(t, root) {
		parts := strings.Split(f, "/")
		if !strings.HasSuffix(f, ".go") || strings.HasSuffix(f, "_test.go") || parts[0] == "composer" ||
			slices.Contains(parts, "gen") || slices.Contains(parts, "testdata") {
			continue
		}
		af, err := parser.ParseFile(fset, filepath.Join(root, f), nil, parser.SkipObjectResolution)
		if err != nil {
			t.Errorf("%s: %v", f, err)
			continue
		}
		parsed++
		for _, l := range kernelLiterals(af) {
			found = append(found, f+":"+strconv.Itoa(fset.Position(l.Pos()).Line)+": "+l.Value)
		}
	}
	if parsed < 500 {
		t.Fatalf("parsed %d Go files under %s: the tree was not found, so this gate proved nothing", parsed, root)
	}
	if len(found) > 0 {
		t.Errorf("%d literal(s) outside composer/ hold device code; the composer emits it:\n%s", len(found), strings.Join(found, "\n"))
	}
}

// kernelLiterals is af's literals holding device code.
func kernelLiterals(af *ast.File) []*ast.BasicLit {
	var out []*ast.BasicLit
	ast.Inspect(af, func(n ast.Node) bool {
		l, ok := n.(*ast.BasicLit)
		if !ok {
			return true
		}
		switch l.Kind {
		case token.STRING:
			s, err := strconv.Unquote(l.Value)
			if err != nil {
				return true
			}
			for _, m := range kernelText {
				if strings.Contains(s, m) {
					out = append(out, l)
					break
				}
			}
		case token.INT:
			if v, err := strconv.ParseUint(l.Value, 0, 64); err == nil && v == spirvMagic {
				out = append(out, l)
			}
		}
		return true
	})
	return out
}

// rawLoaders is each backend's entry point that takes device code as text or
// a code object rather than as the composer's IR, by receiver type.
var rawLoaders = map[string]string{
	"github.com/jitllm/jitllm/backend/cuda.Device": "Load",
	"github.com/jitllm/jitllm/backend/metal.Ctx":   "Compile",
	"github.com/jitllm/jitllm/backend/vulkan.Ctx":  "Compile",
	"github.com/jitllm/jitllm/backend/hip.Device":  "Load",
	"github.com/jitllm/jitllm/backend/hip.Comgr":   "Compile",
}

// listed is one package as go list states it.
type listed struct {
	ImportPath, Dir, Export string
	GoFiles                 []string
	Standard                bool
	Module                  *struct{ Path string }
	Deps                    []string
}

// TestKernelsReachADeviceOnlyAsIR: a device takes a kernel as the composer's
// IR, through backend.Device.Compile, and nothing else. Each backend's raw
// loader (rawLoaders) takes PTX, MSL, SPIR-V or a code object, so a call of
// one outside the backend's own packages is device code the composer did not
// make. The calls are found by type, in every module of the tree, on the
// files this host builds.
func TestKernelsReachADeviceOnlyAsIR(t *testing.T) {
	testmodels.SourceTree(t)
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	var modules []string
	for _, f := range trackedFiles(t, root) {
		if path.Base(f) == "go.mod" && !strings.Contains(f, "testdata/") {
			modules = append(modules, filepath.Join(root, path.Dir(f)))
		}
	}
	if len(modules) < 2 {
		t.Fatalf("found %d modules under %s: the tree was not found, so this gate proved nothing", len(modules), root)
	}
	exports := map[string]string{}
	var mine []listed
	for _, dir := range modules {
		for _, p := range goList(t, dir) {
			if p.Export != "" {
				exports[p.ImportPath] = p.Export
			}
			if !p.Standard && p.Module != nil && strings.HasPrefix(p.Module.Path, "github.com/jitllm/jitllm") {
				mine = append(mine, p)
			}
		}
	}
	fset := token.NewFileSet()
	imp := importer.ForCompiler(fset, "gc", func(path string) (io.ReadCloser, error) {
		f, ok := exports[path]
		if !ok {
			return nil, errors.New("no export data for " + path)
		}
		return os.Open(f)
	})
	check := func(name string, files []*ast.File) []string {
		info := &types.Info{Selections: map[*ast.SelectorExpr]*types.Selection{}}
		var errs []error
		conf := types.Config{Importer: imp, Error: func(err error) { errs = append(errs, err) }}
		conf.Check(name, fset, files, info)
		if len(errs) > 0 {
			t.Fatalf("%s does not type-check here, so its calls cannot be read: %v", name, errs[0])
		}
		return rawCalls(fset, info)
	}

	planted := func(src string) []string {
		af, err := parser.ParseFile(fset, "planted.go", src, 0)
		if err != nil {
			t.Fatal(err)
		}
		return check("planted", []*ast.File{af})
	}
	if got := planted("package p\nimport \"github.com/jitllm/jitllm/backend/vulkan\"\nfunc f(c *vulkan.Ctx) { c.Compile(nil, \"main\", 1, 32) }\n"); len(got) != 1 {
		t.Fatalf("a planted SPIR-V load reads as %v: the gate cannot tell", got)
	}
	if got := planted("package p\nimport \"github.com/jitllm/jitllm/backend\"\nfunc f(d backend.Device) { d.Compile(nil) }\n"); len(got) != 0 {
		t.Fatalf("a compile of the composer's IR reads as a raw load %v: the gate cannot tell", got)
	}

	seen, checked := 0, 0
	var found []string
	for _, p := range mine {
		if p.ImportPath == "github.com/jitllm/jitllm/backend" || strings.HasPrefix(p.ImportPath, "github.com/jitllm/jitllm/backend/") {
			continue
		}
		seen++
		if !slices.ContainsFunc(p.Deps, isRawBackend) {
			continue
		}
		var files []*ast.File
		for _, g := range p.GoFiles {
			af, err := parser.ParseFile(fset, filepath.Join(p.Dir, g), nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			files = append(files, af)
		}
		found = append(found, check(p.ImportPath, files)...)
		checked++
	}
	if seen < 50 || checked < 3 {
		t.Fatalf("read %d packages and checked %d reaching a backend's raw package: the tree was not listed, so this gate proved nothing", seen, checked)
	}
	sort.Strings(found)
	if len(found) > 0 {
		t.Errorf("%d call(s) outside backend/ hand a device code the composer did not make; Compile the composer's IR:\n%s", len(found), strings.Join(found, "\n"))
	}
	t.Logf("%d packages type-checked of %d", checked, seen)
}

func isRawBackend(dep string) bool {
	for k := range rawLoaders {
		if dep == k[:strings.LastIndex(k, ".")] {
			return true
		}
	}
	return false
}

// rawCalls is every method value info selects whose receiver and name are
// a raw loader's.
func rawCalls(fset *token.FileSet, info *types.Info) []string {
	var out []string
	for sel, s := range info.Selections {
		if s.Kind() != types.MethodVal {
			continue
		}
		recv := s.Recv()
		if p, ok := recv.(*types.Pointer); ok {
			recv = p.Elem()
		}
		n, ok := recv.(*types.Named)
		if !ok || n.Obj().Pkg() == nil {
			continue
		}
		if rawLoaders[n.Obj().Pkg().Path()+"."+n.Obj().Name()] == sel.Sel.Name {
			out = append(out, fset.Position(sel.Pos()).String()+": "+n.Obj().Name()+"."+sel.Sel.Name)
		}
	}
	return out
}

// goList is the packages of the module in dir and every package they import,
// each with its export data built.
func goList(t *testing.T, dir string) []listed {
	t.Helper()
	cmd := exec.Command("go", "list", "-e", "-export", "-deps", "-json=ImportPath,Dir,Export,GoFiles,Standard,Module,Deps", "./...")
	cmd.Dir = dir
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("go list in %s: %v\n%s", dir, err, stderr.String())
	}
	var ps []listed
	for dec := json.NewDecoder(bytes.NewReader(out)); ; {
		var p listed
		if err := dec.Decode(&p); err == io.EOF {
			break
		} else if err != nil {
			t.Fatalf("go list in %s: %v", dir, err)
		}
		ps = append(ps, p)
	}
	return ps
}

// TestTheAssemblersHoldOnlyTheHostReads: composer/x86 and composer/a64 are
// assemblers the composer's lowerings drive, and a whole routine written in
// them is a hand-written kernel. The two they hold are the host's ISA reads
// (composer/x86's CPUIDRead and XCR0Read): they read the hardware's identity
// before anything is formed, there is no op to compose them from, and like
// the trampolines they are the boundary RULE 8 names. A third fails here, and
// so does a listed one that is gone.
func TestTheAssemblersHoldOnlyTheHostReads(t *testing.T) {
	testmodels.SourceTree(t)
	boundary := []string{"composer/x86.CPUIDRead", "composer/x86.XCR0Read"}
	for _, c := range []struct {
		name, src string
		holds     bool
	}{
		{"a routine in a Buf", "package x86\nfunc F() ([]byte, error) { var a Buf; a.RET(); return a.Bytes() }\n", true},
		{"a routine in an A64", "package a64\nfunc F() { a := &A64{}; a.RET() }\n", true},
		{"a method on the assembler", "package x86\nfunc (a *Buf) RET2() { a.RET(); a.RET() }\n", false},
	} {
		af, err := parser.ParseFile(token.NewFileSet(), "planted.go", c.src, parser.SkipObjectResolution)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if got := len(routines(af)) > 0; got != c.holds {
			t.Fatalf("planted %s: found %v, want %v: the gate cannot tell", c.name, got, c.holds)
		}
	}
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	parsed := 0
	var found []string
	for _, f := range trackedFiles(t, root) {
		dir := path.Dir(f)
		if (dir != "composer/x86" && dir != "composer/a64") || !strings.HasSuffix(f, ".go") || strings.HasSuffix(f, "_test.go") {
			continue
		}
		af, err := parser.ParseFile(fset, filepath.Join(root, f), nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		parsed++
		for _, fn := range routines(af) {
			found = append(found, dir+"."+fn)
		}
	}
	if parsed < 10 {
		t.Fatalf("parsed %d assembler files: the tree was not found, so this gate proved nothing", parsed)
	}
	sort.Strings(found)
	for _, f := range found {
		if !slices.Contains(boundary, f) {
			t.Errorf("%s assembles a whole routine outside the composer's lowerings: state it as ops the composer forms", f)
		}
	}
	for _, b := range boundary {
		if !slices.Contains(found, b) {
			t.Errorf("%s is listed as the host's boundary and holds no routine: take it off the list", b)
		}
	}
}

// routines is af's top-level functions that hold an assembler of their own:
// a Buf or an A64 declared, composed or allocated in the function's body.
func routines(af *ast.File) []string {
	isAsm := func(e ast.Expr) bool {
		id, ok := e.(*ast.Ident)
		return ok && (id.Name == "Buf" || id.Name == "A64")
	}
	var out []string
	for _, d := range af.Decls {
		fn, ok := d.(*ast.FuncDecl)
		if !ok || fn.Recv != nil || fn.Body == nil {
			continue
		}
		holds := false
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.ValueSpec:
				holds = holds || (x.Type != nil && isAsm(x.Type))
			case *ast.CompositeLit:
				holds = holds || isAsm(x.Type)
			case *ast.CallExpr:
				if id, ok := x.Fun.(*ast.Ident); ok && id.Name == "new" && len(x.Args) == 1 {
					holds = holds || isAsm(x.Args[0])
				}
			}
			return !holds
		})
		if holds {
			out = append(out, fn.Name.Name)
		}
	}
	return out
}
