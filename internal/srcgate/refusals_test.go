package srcgate

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestNoRefusalButInputsAndLimits holds every error the inference packages
// -- engine, model, nn, composer, graph, format, sched, backend -- make to
// the classes testdata/origins.txt gives them, whatever its wording: an
// error's literal is found by what makes it one (fmt.Errorf, errors.New, a
// panic, an error type's value or conversion, a helper that makes an error
// of its argument, a reason a function returns beside its bool or names an
// Err, a string assigned to a name ending in Err), not by a phrase. A new
// error is unclassified until a line says what it is; one gone leaves the
// table with it. The classes:
//
//   - input: the caller's request or the file's contents fail a check
//   - limit: a device, driver or OS states a bound (memory, a width, an
//     instruction set, co-residency: RULE 13)
//   - fault: an outside call failed and reports it
//   - bug: the engine's own contract broken; reaching it is a defect
//   - context: names where; the cause is the error or reason it carries
//   - skip: an arm not taken, the same answer through another
//   - refusal: a feature the engine does not do, with why: the work list,
//     which the goal's second check drives to empty
//
// The violation -- an error of each kind of origin the table does not
// hold, worded with no phrase a list of phrases would catch -- is seen.
func TestNoRefusalButInputsAndLimits(t *testing.T) {
	testmodels.SourceTree(t)
	table, err := readOrigins(filepath.Join("testdata", "origins.txt"))
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Join("..", "..")
	found, files, err := errorOrigins(root, []string{"engine", "model", "nn", "composer", "graph", "format", "sched", "backend"})
	if err != nil {
		t.Fatal(err)
	}
	if files < 100 || len(found) < 1000 {
		t.Fatalf("read %d files and %d origins: the gate read nothing", files, len(found))
	}
	var missing, stale []string
	for k := range found {
		if _, ok := table[k]; !ok {
			missing = append(missing, fmt.Sprintf("%s\t%s", k.file, strconv.Quote(k.text)))
		}
	}
	for k := range table {
		if !found[k] {
			stale = append(stale, fmt.Sprintf("%s\t%s", k.file, strconv.Quote(k.text)))
		}
	}
	slices.Sort(missing)
	slices.Sort(stale)
	for _, m := range missing {
		t.Errorf("an error testdata/origins.txt does not classify: %s -- say what it is (input, limit, fault, bug, context, skip, or a refusal with why)", m)
	}
	for _, s := range stale {
		t.Errorf("testdata/origins.txt classifies an error the code no longer makes: %s", s)
	}
	counts, open := map[string]int{}, map[string]int{}
	for _, c := range table {
		counts[c.class]++
		if c.class == "refusal" {
			open[c.why]++
		}
	}
	for _, why := range slices.Sorted(maps.Keys(open)) {
		t.Logf("open refusal (%d): %s", open[why], why)
	}
	t.Logf("%d files, %d errors: %v", files, len(found), counts)

	for _, v := range []string{
		`package p; import "fmt"; func f() error { return fmt.Errorf("the tower takes pictures this wide alone") }`,
		`package p; import "errors"; var errWide = errors.New("a window this long goes to the host")`,
		`package p; type tierError string; func (e tierError) Error() string { return string(e) }; const errX tierError = "the head stays home"`,
		`package p; import "fmt"; func fail(f string, a ...any) error { return fmt.Errorf("tier: "+f, a...) }; func g() error { return fail("block %d keeps to the host", 3) }`,
		`package p; import "fmt"; func fits(n int) (bool, string) { return false, fmt.Sprintf("%d rows go one at a time", n) }`,
		`package p; type tier struct{ LastErr string }; func (g *tier) f() { g.LastErr = "the chunk restarts on the host" }`,
	} {
		f, err := parser.ParseFile(token.NewFileSet(), "v.go", v, 0)
		if err != nil {
			t.Fatal(err)
		}
		got := fileOrigins(f, errorHelpers([]*ast.File{f}))
		if len(got) == 0 {
			t.Errorf("the violation %s, an error the table does not hold, was not seen", v)
		}
		for _, s := range got {
			if _, ok := table[originKey{"engine/forward.go", s}]; ok {
				t.Errorf("the violation %q is classified", s)
			}
		}
	}
}

type originKey struct{ file, text string }

type originClass struct{ class, why string }

// readOrigins reads the table: class, file, the literal Go-quoted and, for a
// refusal, why, a line each, tab-separated; a line starting # is a comment.
func readOrigins(path string) (map[originKey]originClass, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	classes := []string{"input", "limit", "fault", "bug", "context", "skip", "refusal"}
	out := map[originKey]originClass{}
	sc := bufio.NewScanner(f)
	sc.Buffer(nil, 1<<20)
	for n := 1; sc.Scan(); n++ {
		line := sc.Text()
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		cols := strings.Split(line, "\t")
		if len(cols) < 3 || len(cols) > 4 {
			return nil, fmt.Errorf("%s:%d: %d columns", path, n, len(cols))
		}
		if !slices.Contains(classes, cols[0]) {
			return nil, fmt.Errorf("%s:%d: class %q is not one of %v", path, n, cols[0], classes)
		}
		text, err := strconv.Unquote(cols[2])
		if err != nil {
			return nil, fmt.Errorf("%s:%d: %v", path, n, err)
		}
		c := originClass{class: cols[0]}
		if len(cols) == 4 {
			c.why = cols[3]
		}
		if c.class == "refusal" && c.why == "" {
			return nil, fmt.Errorf("%s:%d: a refusal says why", path, n)
		}
		k := originKey{cols[1], text}
		if _, ok := out[k]; ok {
			return nil, fmt.Errorf("%s:%d: %s %q twice", path, n, k.file, k.text)
		}
		out[k] = c
	}
	return out, sc.Err()
}

// errorOrigins is every error literal of the production files under dirs of
// root, keyed by file (slash-separated, from root) and text, and how many
// files it read.
func errorOrigins(root string, dirs []string) (map[originKey]bool, int, error) {
	pkgs := map[string][]*ast.File{}
	names := map[*ast.File]string{}
	fset := token.NewFileSet()
	for _, dir := range dirs {
		err := filepath.WalkDir(filepath.Join(root, dir), func(p string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() && d.Name() == "testdata" {
				return filepath.SkipDir
			}
			if d.IsDir() || !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
				return nil
			}
			f, err := parser.ParseFile(fset, p, nil, 0)
			if err != nil {
				return err
			}
			rel, _ := filepath.Rel(root, p)
			rel = filepath.ToSlash(rel)
			pkgs[filepath.Dir(p)] = append(pkgs[filepath.Dir(p)], f)
			names[f] = rel
			return nil
		})
		if err != nil {
			return nil, 0, err
		}
	}
	out := map[originKey]bool{}
	for _, files := range pkgs {
		helpers := errorHelpers(files)
		for _, f := range files {
			for _, s := range fileOrigins(f, helpers) {
				out[originKey{names[f], s}] = true
			}
		}
	}
	return out, len(names), nil
}

// errorHelpers names what makes an error of a string argument in one
// package's files, with the arguments it reads: a function, method or
// closure whose string parameter reaches fmt.Errorf, errors.New, a
// fmt.Sprintf into a name ending in Err, or such a name; and a type with an
// Error method, whose conversion makes one.
func errorHelpers(files []*ast.File) map[string][]int {
	out := map[string][]int{}
	check := func(name string, ft *ast.FuncType, body *ast.BlockStmt) {
		if body == nil || ft.Params == nil {
			return
		}
		i := 0
		for _, p := range ft.Params.List {
			n := max(1, len(p.Names))
			id, ok := p.Type.(*ast.Ident)
			for j := range n {
				if ok && id.Name == "string" && len(p.Names) > 0 && reachesError(body, p.Names[j].Name) {
					out[name] = append(out[name], i)
				}
				i++
			}
		}
	}
	for _, f := range files {
		ast.Inspect(f, func(n ast.Node) bool {
			switch v := n.(type) {
			case *ast.FuncDecl:
				check(v.Name.Name, v.Type, v.Body)
				if v.Name.Name == "Error" && v.Recv != nil && len(v.Recv.List) == 1 {
					t := v.Recv.List[0].Type
					if st, ok := t.(*ast.StarExpr); ok {
						t = st.X
					}
					if id, ok := t.(*ast.Ident); ok {
						out[id.Name] = []int{0}
					}
				}
			case *ast.AssignStmt:
				for i, r := range v.Rhs {
					if fl, ok := r.(*ast.FuncLit); ok && i < len(v.Lhs) {
						if id, ok := v.Lhs[i].(*ast.Ident); ok {
							check(id.Name, fl.Type, fl.Body)
						}
					}
				}
			}
			return true
		})
	}
	return out
}

// reachesError says body makes an error of the parameter named param.
func reachesError(body *ast.BlockStmt, param string) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		switch v := n.(type) {
		case *ast.CallExpr:
			if k := errorCall(v); k == "errorf" || k == "new" {
				for _, a := range v.Args {
					found = found || mentions(a, param)
				}
			}
		case *ast.AssignStmt:
			for i, r := range v.Rhs {
				if i < len(v.Lhs) && errName(v.Lhs[i]) && mentions(r, param) {
					found = true
				}
			}
		}
		return !found
	})
	return found
}

// fileOrigins is every error literal of f, helpers the package's
// (errorHelpers).
func fileOrigins(f *ast.File, helpers map[string][]int) []string {
	var out []string
	add := func(e ast.Expr) {
		if c, ok := e.(*ast.CallExpr); ok && errorCall(c) == "sprintf" && len(c.Args) > 0 {
			e = c.Args[0]
		}
		if s, ok := spelled(e); ok {
			out = append(out, s)
		}
	}
	errType := func(e ast.Expr) bool {
		id, ok := e.(*ast.Ident)
		return ok && slices.Equal(helpers[id.Name], []int{0}) && id.Name != "string"
	}
	ast.Inspect(f, func(n ast.Node) bool {
		switch v := n.(type) {
		case *ast.FuncDecl:
			if v.Body != nil && reasonFunc(v.Name.Name, v.Type) {
				out = append(out, returned(v.Body)...)
			}
		case *ast.FuncLit:
			if reasonFunc("", v.Type) {
				out = append(out, returned(v.Body)...)
			}
		case *ast.ValueSpec:
			if errType(v.Type) {
				for _, val := range v.Values {
					add(val)
				}
			}
		case *ast.CompositeLit:
			if errType(v.Type) {
				for _, e := range v.Elts {
					if kv, ok := e.(*ast.KeyValueExpr); ok {
						e = kv.Value
					}
					add(e)
				}
			}
		case *ast.AssignStmt:
			for i, r := range v.Rhs {
				if i < len(v.Lhs) && errName(v.Lhs[i]) {
					add(r)
				}
			}
		case *ast.CallExpr:
			if len(v.Args) == 0 {
				return true
			}
			switch errorCall(v) {
			case "errorf", "new", "panic":
				add(v.Args[0])
				return true
			case "sprintf":
				return true
			}
			name := ""
			switch fn := v.Fun.(type) {
			case *ast.Ident:
				name = fn.Name
			case *ast.SelectorExpr:
				name = fn.Sel.Name
			}
			for _, i := range helpers[name] {
				if i < len(v.Args) {
					add(v.Args[i])
				}
			}
		}
		return true
	})
	return out
}

// reasonFunc says a function's string result is a reason: it returns a bool
// beside it, it is an error's Error method, or its name says it answers why.
func reasonFunc(name string, ft *ast.FuncType) bool {
	if ft.Results == nil {
		return false
	}
	var str, boolean bool
	for _, r := range ft.Results.List {
		if id, ok := r.Type.(*ast.Ident); ok {
			str = str || id.Name == "string"
			boolean = boolean || id.Name == "bool"
		}
	}
	n := strings.ToLower(name)
	return str && (boolean || name == "Error" || strings.Contains(n, "conflict") || strings.Contains(n, "declin") ||
		strings.Contains(n, "refus") || strings.Contains(n, "why") || strings.Contains(n, "reason") || strings.Contains(n, "unserved"))
}

// returned is every prose literal, or fmt.Sprintf of one, body returns, a
// nested function left to its own turn: a word naming a role or a table is
// not a reason.
func returned(body *ast.BlockStmt) []string {
	var out []string
	ast.Inspect(body, func(n ast.Node) bool {
		switch v := n.(type) {
		case *ast.FuncLit:
			return false
		case *ast.ReturnStmt:
			for _, r := range v.Results {
				if c, ok := r.(*ast.CallExpr); ok && errorCall(c) == "sprintf" && len(c.Args) > 0 {
					r = c.Args[0]
				}
				if s, ok := literal(r); ok && strings.Contains(s, " ") {
					out = append(out, s)
				}
			}
		}
		return true
	})
	return out
}

// errName says e names an error's text: a name ending in Err.
func errName(e ast.Expr) bool {
	switch v := e.(type) {
	case *ast.Ident:
		return strings.HasSuffix(v.Name, "Err")
	case *ast.SelectorExpr:
		return strings.HasSuffix(v.Sel.Name, "Err")
	}
	return false
}

// mentions says e reads the name.
func mentions(e ast.Expr, name string) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && id.Name == name {
			found = true
		}
		return !found
	})
	return found
}

// errorCall names c when it is fmt.Errorf, errors.New, fmt.Sprintf or panic.
func errorCall(c *ast.CallExpr) string {
	switch f := c.Fun.(type) {
	case *ast.Ident:
		if f.Name == "panic" {
			return "panic"
		}
	case *ast.SelectorExpr:
		if x, ok := f.X.(*ast.Ident); ok {
			switch x.Name + "." + f.Sel.Name {
			case "fmt.Errorf":
				return "errorf"
			case "errors.New":
				return "new"
			case "fmt.Sprintf":
				return "sprintf"
			}
		}
	}
	return ""
}

// spelled is what an error's text says: a literal, or a sum of strings with
// %v for each that is not one, prose in some literal of it.
func spelled(e ast.Expr) (string, bool) {
	if s, ok := literal(e); ok {
		return s, s != ""
	}
	var parts []string
	prose := false
	var walk func(ast.Expr) bool
	walk = func(e ast.Expr) bool {
		if b, ok := e.(*ast.BinaryExpr); ok && b.Op == token.ADD {
			return walk(b.X) && walk(b.Y)
		}
		if s, ok := literal(e); ok {
			prose = prose || strings.Contains(s, " ")
			parts = append(parts, s)
			return true
		}
		switch e.(type) {
		case *ast.Ident, *ast.SelectorExpr, *ast.CallExpr, *ast.IndexExpr:
			parts = append(parts, "%v")
			return true
		}
		return false
	}
	if b, ok := e.(*ast.BinaryExpr); !ok || b.Op != token.ADD || !walk(e) || !prose {
		return "", false
	}
	return strings.Join(parts, ""), true
}

// literal is the string a constant expression of string literals spells.
func literal(e ast.Expr) (string, bool) {
	switch v := e.(type) {
	case *ast.BasicLit:
		if v.Kind != token.STRING {
			return "", false
		}
		s, err := strconv.Unquote(v.Value)
		return s, err == nil
	case *ast.BinaryExpr:
		if v.Op != token.ADD {
			return "", false
		}
		a, ok := literal(v.X)
		if !ok {
			return "", false
		}
		b, ok := literal(v.Y)
		return a + b, ok
	case *ast.ParenExpr:
		return literal(v.X)
	}
	return "", false
}
