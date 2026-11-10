package oracle_test

import (
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// goComputeAllowed() is every production function that may do float arithmetic
// on data inside a loop, and why. Anything else fails
// TestNoGoComputeOnTheInferencePath, and so does an entry that no longer
// matches -- a site that was fixed comes off this list rather than leaving a
// hole shaped like it.
func goComputeAllowed() map[string]string {
	return map[string]string{
		// Measurement, not model arithmetic: accounting and the rates measured.
		"engine/hotness.go:ExpertUses":       "expert-use accounting for the report",
		"engine/device/gpu/choose.go:choose": "each device's measured read rate in GB/s, printed when choosing verbosely",
		// Constants built once at load from the config, like the tables a kernel is
		// compiled with.
		"nn/ropetable.go:foldRopeTab": "the rotary table's folded constant planes, once per configuration",
		"nn/ropetable.go:YarnFreqs":   "YaRN's per-dimension frequencies, once per configuration",
		"nn/ropeaxes.go:runFreqs":     "a multi-axis rotary's per-pair frequencies, once per configuration",
		"model/model.go:ropeScale":    "a scaled rotary's per-pair divisors, the shipped factors times the scaling, once at open",
		"graph/sample.go:selection":   "the draw's graph: the binade thresholds it states as constants, once when the graph is built",
		// Geometry from a picture's sizes.
		"model/tower.go:TileGrid": "InternVL's tile grid, the candidate counts walked against the picture's aspect",
		// A clip's resample: libswresample's filter bank, once a pair of rates,
		// as Pillow's taps are once a size (the sound op reads it).
		"composer/swr.go:swrBank": "libswresample's polyphase filter bank, built once a pair of rates as resample.c builds it: a constant table the sound op reads",
		"composer/swr.go:poly":    "av_bessel_i0's rational approximation, evaluated for the bank's Kaiser window once a tap",
		// A request's and a picture's geometry, and a graph's attributes.
		"engine/decide.go:answer":        "a rating question's levels, i/(r-1) for its r answers, once a question: the readout's parameters",
		"model/tower.go:lowerValuesWith": "a tower op's attention scale, 1/sqrt of its head width, once an op: the graph's attribute",
		"model/tower.go:sliceGrid":       "MiniCPM-V's slice grid, the factor pairs' log aspects against the picture's (get_sliced_grid): geometry from its sizes",
		"model/video.go:VideoPlan":       "a video's sampled frames, numpy's linspace over its count, and their times: geometry from its frames and rate",
	}
}

// TestNoGoComputeOnTheInferencePath is AGENTS.md RULE 8 read off the source:
// every op a token runs is generated code, so production Go may not do float
// arithmetic on data in a loop -- a float slice element in +,-,*,/ (or
// op-assigned), a computed float stored into a float element, or a math.* call,
// inside a for or range body -- outside goComputeAllowed(). A scalar computed
// once per call (a softmax scale) is a kernel argument and does not match.
//
// The import guard beside it stops production code borrowing the reference
// arithmetic; this stops it writing its own.
func TestNoGoComputeOnTheInferencePath(t *testing.T) {
	testmodels.SourceTree(t)
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	// t.Chdir restores it: a bare os.Chdir left the import guard beside this
	// walking the parent of the repository.
	t.Chdir(root)
	found := map[string][]int{}
	for _, pkg := range []string{"model", "engine", "nn", "engine/device/gpu", "backend", "sched", "engine/device/cpu", "engine/device/cpudev", "composer", "graph"} {
		fset := token.NewFileSet()
		ents, err := os.ReadDir(filepath.Join(root, pkg))
		if err != nil {
			t.Fatal(err)
		}
		var files []*ast.File
		for _, e := range ents {
			n := e.Name()
			if !strings.HasSuffix(n, ".go") || strings.HasSuffix(n, "_test.go") {
				continue
			}
			f, err := parser.ParseFile(fset, filepath.Join(root, pkg, n), nil, parser.ParseComments)
			if err != nil {
				t.Fatal(err)
			}
			if testOnly(f) {
				continue
			}
			files = append(files, f)
		}
		info := &types.Info{Types: map[ast.Expr]types.TypeAndValue{}, Defs: map[*ast.Ident]types.Object{}, Uses: map[*ast.Ident]types.Object{}}
		conf := types.Config{Importer: importer.ForCompiler(fset, "source", nil), Error: func(error) {}}
		conf.Check(pkg, fset, files, info)
		if len(info.Types) == 0 {
			t.Fatalf("%s: the type checker saw nothing -- this gate would pass by seeing no code", pkg)
		}
		for _, f := range files {
			for _, d := range f.Decls {
				fd, ok := d.(*ast.FuncDecl)
				if !ok || fd.Body == nil {
					continue
				}
				for _, line := range goComputeLines(fset, info, fd.Body) {
					key := pkg + "/" + filepath.Base(fset.Position(f.Pos()).Filename) + ":" + fd.Name.Name
					found[key] = append(found[key], line)
				}
			}
		}
	}
	var keys []string
	for k := range found {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if _, ok := goComputeAllowed()[k]; !ok {
			t.Errorf("%s does float arithmetic on data in a loop at line(s) %v: RULE 8 -- generate a kernel "+
				"(or, if it is measurement or a load-time constant, add it to goComputeAllowed() with the reason)",
				k, found[k])
		}
	}
	for k, why := range goComputeAllowed() {
		if _, ok := found[k]; !ok {
			t.Errorf("goComputeAllowed()[%q] (%s) matches no code any more: take it off the list", k, why)
		}
	}
	t.Logf("%d function(s) with float loop arithmetic, all accounted for", len(found))
}

// testOnly reports a file built only under a test tag.
func testOnly(f *ast.File) bool {
	for _, cg := range f.Comments {
		for _, c := range cg.List {
			if strings.HasPrefix(c.Text, "//go:build") && strings.Contains(c.Text, "jitllmtest") &&
				!strings.Contains(c.Text, "!jitllmtest") {
				return true
			}
		}
		if cg.Pos() > f.Package {
			break
		}
	}
	return false
}

// goComputeLines returns the lines of body's loops that do float arithmetic on
// data (see TestNoGoComputeOnTheInferencePath).
func goComputeLines(fset *token.FileSet, info *types.Info, body *ast.BlockStmt) []int {
	isFloat := func(e ast.Expr) bool {
		tv, ok := info.Types[e]
		if !ok || tv.Value != nil {
			return false
		}
		b, ok := tv.Type.Underlying().(*types.Basic)
		return ok && b.Info()&types.IsFloat != 0
	}
	// tainted holds the float locals a loop assigns from data: an element
	// of a float slice, or another such local. Arithmetic through them is
	// arithmetic on data, as `g := float64(x[i]); s += g * g` is.
	tainted := map[types.Object]bool{}
	objOf := func(id *ast.Ident) types.Object {
		if o := info.Defs[id]; o != nil {
			return o
		}
		return info.Uses[id]
	}
	var floatElem func(n ast.Node) bool
	floatElem = func(n ast.Node) bool {
		hit := false
		ast.Inspect(n, func(z ast.Node) bool {
			switch x := z.(type) {
			case *ast.IndexExpr:
				if isFloat(x) {
					hit = true
				}
			case *ast.Ident:
				if o := objOf(x); o != nil && tainted[o] {
					hit = true
				}
			}
			return !hit
		})
		return hit
	}
	taint := func(loop ast.Node) {
		for changed := true; changed; {
			changed = false
			ast.Inspect(loop, func(m ast.Node) bool {
				// A float a range takes from a slice or array is an element
				// of it, as x[i] is.
				if rs, ok := m.(*ast.RangeStmt); ok {
					if id, ok := rs.Value.(*ast.Ident); ok {
						if o := objOf(id); o != nil && !tainted[o] {
							if b, ok := o.Type().Underlying().(*types.Basic); ok && b.Info()&types.IsFloat != 0 {
								tainted[o], changed = true, true
							}
						}
					}
					return true
				}
				as, ok := m.(*ast.AssignStmt)
				if !ok {
					return true
				}
				for i, l := range as.Lhs {
					id, ok := l.(*ast.Ident)
					if !ok {
						continue
					}
					o := objOf(id)
					if o == nil {
						continue
					}
					if b, ok := o.Type().Underlying().(*types.Basic); !ok || b.Info()&types.IsFloat == 0 {
						continue
					}
					rhs := as.Rhs[0]
					if len(as.Rhs) == len(as.Lhs) {
						rhs = as.Rhs[i]
					}
					if !tainted[o] && floatElem(rhs) {
						tainted[o], changed = true, true
					}
				}
				return true
			})
		}
	}
	mathCall := func(c *ast.CallExpr) bool {
		se, ok := c.Fun.(*ast.SelectorExpr)
		if !ok {
			return false
		}
		id, ok := se.X.(*ast.Ident)
		if !ok || id.Name != "math" {
			return false
		}
		switch n := se.Sel.Name; {
		case strings.HasPrefix(n, "Float"), n == "Inf", n == "NaN", n == "IsNaN", n == "IsInf", n == "Signbit":
			return false // bit casts and constants, not arithmetic
		}
		return true
	}
	seen := map[int]bool{}
	var lines []int
	mark := func(n ast.Node) {
		if l := fset.Position(n.Pos()).Line; !seen[l] {
			seen[l] = true
			lines = append(lines, l)
		}
	}
	ast.Inspect(body, func(n ast.Node) bool {
		var loop *ast.BlockStmt
		switch x := n.(type) {
		case *ast.ForStmt:
			loop = x.Body
		case *ast.RangeStmt:
			loop = x.Body
		}
		if loop == nil {
			return true
		}
		// The loop with its header, where a range's value is declared.
		taint(n)
		ast.Inspect(loop, func(m ast.Node) bool {
			switch x := m.(type) {
			case *ast.CallExpr:
				if mathCall(x) {
					mark(x)
				}
			case *ast.BinaryExpr:
				switch x.Op {
				case token.ADD, token.SUB, token.MUL, token.QUO:
					if isFloat(x) && floatElem(x) {
						mark(x)
					}
				}
			case *ast.AssignStmt:
				switch x.Tok {
				case token.ADD_ASSIGN, token.SUB_ASSIGN, token.MUL_ASSIGN, token.QUO_ASSIGN:
					if isFloat(x.Lhs[0]) && floatElem(x) {
						mark(x)
					}
				case token.ASSIGN:
					for i, l := range x.Lhs {
						ix, ok := l.(*ast.IndexExpr)
						if !ok || !isFloat(ix) || i >= len(x.Rhs) {
							continue
						}
						ast.Inspect(x.Rhs[i], func(z ast.Node) bool {
							if be, ok := z.(*ast.BinaryExpr); ok && isFloat(be) {
								mark(x)
								return false
							}
							return true
						})
					}
				}
			}
			return true
		})
		return false
	})
	return lines
}
