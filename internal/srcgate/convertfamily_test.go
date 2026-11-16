package srcgate

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestConverterFamilyCodeOnlyShrinks counts, in each function of convert/,
// what decides on a family by its name: an identifier spelled as an
// architecture constant was (archConstant), and a string literal naming a
// class, a GGUF architecture or a projector the converter lists. The converter is becoming model.json files
// composed of generic primitives (docs/design/model-json.md), so a count may
// only fall: testdata/convertfamily.txt holds each function's count, and a
// function above its line, or one with a count and no line, fails. A fall is
// logged, and the file is lowered when the work lands.
//
// It sees only code keyed on a family's name. A recipe under a neutral name
// is family code too; the assessment of the converter lists those.
func TestConverterFamilyCodeOnlyShrinks(t *testing.T) {
	testmodels.SourceTree(t)
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	lits := familyLiterals(t)
	if len(lits) < 40 {
		t.Fatalf("read %d family names: the gate read nothing", len(lits))
	}
	got, files := convertFamilyCounts(t, root, lits)
	if files < 60 {
		t.Fatalf("read %d files under convert/: the gate read nothing", files)
	}
	want := readFamilyCounts(t, filepath.Join("testdata", "convertfamily.txt"))
	keys := make([]string, 0, len(got))
	for k := range got {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	total, base, fell := 0, 0, 0
	for _, k := range keys {
		n := got[k]
		total += n
		w, listed := want[k]
		switch {
		case !listed:
			t.Errorf("%d %s: family code where the converter had none", n, k)
		case n > w:
			t.Errorf("%d %s: family code grew from %d", n, k, w)
		}
	}
	for k, w := range want {
		base += w
		if got[k] < w {
			fell += w - got[k]
			t.Logf("%s fell from %d to %d: lower its line", k, w, got[k])
		}
	}
	t.Logf("%d references to a family's name in %d functions of convert/ (baseline %d, %d fewer)",
		total, len(got), base, fell)

	viol := `package convert

func hfArchTable() map[string]int { return map[string]int{"LlamaForCausalLM": 1} }

func pick(a Arch) bool { return a == ArchLlama || a == ArchQwen3 }

func plain(n int) int { return n + 1 }
`
	counts := familyCountsIn(t, "viol.go", viol, lits)
	if counts["viol.go:hfArchTable"] != 1 || counts["viol.go:pick"] != 2 || counts["viol.go:plain"] != 0 {
		t.Errorf("the counter read the violation as %v; want a class literal, two constants and nothing", counts)
	}
}

// familyLiterals is every name the converter lists for a family: the
// safetensors classes, the GGUF architectures and the projectors.
func familyLiterals(t *testing.T) map[string]bool {
	t.Helper()
	archs, err := convert.Architectures()
	if err != nil {
		t.Fatal(err)
	}
	m := map[string]bool{}
	for _, l := range [][]string{convert.HFClasses(), archs, convert.Projectors(), convert.AudioProjectors()} {
		for _, n := range l {
			m[n] = true
		}
	}
	return m
}

// convertFamilyCounts walks convert/ (its testdata and model.json files
// aside) and counts each function's family references.
func convertFamilyCounts(t *testing.T, root string, lits map[string]bool) (map[string]int, int) {
	out := map[string]int{}
	files := 0
	err := filepath.Walk(filepath.Join(root, "convert"), func(p string, fi os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if fi.IsDir() && (fi.Name() == "testdata" || strings.HasPrefix(fi.Name(), ".")) {
			return filepath.SkipDir
		}
		if fi.IsDir() || !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			return nil
		}
		src, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(root, p)
		files++
		for k, n := range familyCountsIn(t, filepath.ToSlash(rel), string(src), lits) {
			out[k] += n
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return out, files
}

// familyCountsIn counts, per top-level function of one file (a method as
// Recv.Name), the architecture constants it refers to and the family names
// it spells as string literals. A function with none is not listed.
func familyCountsIn(t *testing.T, rel, src string, lits map[string]bool) map[string]int {
	f, err := parser.ParseFile(token.NewFileSet(), rel, src, 0)
	if err != nil {
		t.Fatalf("%s: %v", rel, err)
	}
	out := map[string]int{}
	for _, d := range f.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Body == nil {
			continue
		}
		name := fd.Name.Name
		if fd.Recv != nil && len(fd.Recv.List) == 1 {
			typ := fd.Recv.List[0].Type
			if s, ok := typ.(*ast.StarExpr); ok {
				typ = s.X
			}
			if id, ok := typ.(*ast.Ident); ok {
				name = id.Name + "." + name
			}
		}
		n := 0
		ast.Inspect(fd.Body, func(x ast.Node) bool {
			switch v := x.(type) {
			case *ast.Ident:
				if archConstant(v.Name) {
					n++
				}
			case *ast.BasicLit:
				if v.Kind == token.STRING {
					if s, err := strconv.Unquote(v.Value); err == nil && lits[s] {
						n++
					}
				}
			}
			return true
		})
		if n > 0 {
			out[rel+":"+name] += n
		}
	}
	return out
}

// readFamilyCounts reads "count file:function" lines; # starts a comment.
func readFamilyCounts(t *testing.T, p string) map[string]int {
	fh, err := os.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer fh.Close()
	out := map[string]int{}
	sc := bufio.NewScanner(fh)
	for line := 1; sc.Scan(); line++ {
		s := strings.TrimSpace(sc.Text())
		if s == "" || strings.HasPrefix(s, "#") {
			continue
		}
		var n int
		var k string
		if _, err := fmt.Sscanf(s, "%d %s", &n, &k); err != nil {
			t.Fatalf("%s:%d: %q is not \"count file:function\"", p, line, s)
		}
		out[k] = n
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return out
}
