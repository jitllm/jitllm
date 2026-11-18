package srcgate

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestOnlyTheComposerEmitsCode holds RULE 8 at its source: machine code and
// device kernels are the composer's. Outside composer/, no file that ships
// uses composer/x86's or composer/a64's assembler (x86.Buf, a64.A64) or
// builds a device kernel by hand (ir.New). Reading their ISA constants, IR
// types or decoders (x86.KernelISA) emits nothing and is not held; a test may
// assemble an instruction or build a kernel to plant a violation in a gate or
// to hold a composed form to the kernel it replaced. openEmitters is the list
// still owed to the composer, and an entry that no longer emits fails too, so
// the list cannot outlive its reason.
func TestOnlyTheComposerEmitsCode(t *testing.T) {
	testmodels.SourceTree(t)
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	var openEmitters []string

	for _, c := range []struct {
		name, src string
		emits     bool
	}{
		{"an x86 buffer", "package p\nimport \"github.com/jitllm/jitllm/composer/x86\"\nfunc f() []byte { var a x86.Buf; a.RET(); b, _ := a.Bytes(); return b }\n", true},
		{"a renamed a64 assembler", "package p\nimport asm \"github.com/jitllm/jitllm/composer/a64\"\nfunc f() { var a asm.A64; a.RET() }\n", true},
		{"the ISA decoder only", "package p\nimport \"github.com/jitllm/jitllm/composer/x86\"\nfunc f(b []byte) bool { _, ok := x86.KernelISA(b); return ok && x86.ISASSE2 != 0 }\n", false},
		{"a device kernel built by hand", "package p\nimport \"github.com/jitllm/jitllm/composer/ir\"\nfunc f() { b := ir.New(\"k\", [3]int{32, 1, 1}); b.Done() }\n", true},
		{"a renamed IR builder", "package p\nimport kir \"github.com/jitllm/jitllm/composer/ir\"\nvar b = kir.New(\"k\", [3]int{32, 1, 1})\n", true},
		{"the IR's types and widths only", "package p\nimport \"github.com/jitllm/jitllm/composer/ir\"\nfunc f() (ir.Type, int) { return ir.F32, ir.SubgroupLanes }\n", false},
	} {
		af, err := parser.ParseFile(token.NewFileSet(), "planted.go", c.src, parser.SkipObjectResolution)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if got := emitsCode(af); got != c.emits {
			t.Fatalf("planted %s: emitsCode = %v, want %v: the gate cannot tell", c.name, got, c.emits)
		}
	}

	fset := token.NewFileSet()
	parsed := 0
	emitting := map[string][]string{}
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
		if emitsCode(af) {
			emitting[path.Dir(f)] = append(emitting[path.Dir(f)], f)
		}
	}
	if parsed < 500 {
		t.Fatalf("parsed %d Go files under %s: the tree was not found, so this gate proved nothing", parsed, root)
	}
	var found []string
	for dir, files := range emitting {
		if !slices.Contains(openEmitters, dir) {
			found = append(found, files...)
		}
	}
	sort.Strings(found)
	if len(found) > 0 {
		t.Errorf("%d file(s) outside composer/ assemble machine code or build a device kernel; the composer makes both:\n%s", len(found), strings.Join(found, "\n"))
	}
	for _, dir := range openEmitters {
		if len(emitting[dir]) == 0 {
			t.Errorf("%s is listed as still emitting and emits nothing: take it off the list", dir)
		}
	}
	t.Logf("%d files read; open: %v", parsed, emitting)
}

// emitsCode reports a file that uses an assembler of composer/x86 or
// composer/a64 or builds a kernel of composer/ir: a selector naming x86.Buf,
// a64.A64 or ir.New through the file's own name for the import, or a dot
// import of any of them.
func emitsCode(af *ast.File) bool {
	asm := map[string]string{}
	for _, im := range af.Imports {
		p, err := strconv.Unquote(im.Path.Value)
		if err != nil {
			continue
		}
		var typ string
		switch p {
		case "github.com/jitllm/jitllm/composer/x86":
			typ = "Buf"
		case "github.com/jitllm/jitllm/composer/a64":
			typ = "A64"
		case "github.com/jitllm/jitllm/composer/ir":
			typ = "New"
		default:
			continue
		}
		name := path.Base(p)
		if im.Name != nil {
			name = im.Name.Name
		}
		if name == "." {
			return true
		}
		asm[name] = typ
	}
	if len(asm) == 0 {
		return false
	}
	found := false
	ast.Inspect(af, func(n ast.Node) bool {
		sel, ok := n.(*ast.SelectorExpr)
		if !ok || found {
			return !found
		}
		if id, ok := sel.X.(*ast.Ident); ok && asm[id.Name] != "" && asm[id.Name] == sel.Sel.Name {
			found = true
		}
		return !found
	})
	return found
}
