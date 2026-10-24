package oracle_test

import (
	"fmt"
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

// imagePathFiles is the image path in engine/model: preprocessing, the
// position tables built per grid and the projector heads. Every op a pixel or
// a patch passes through there is generated code (AGENTS.md RULE 8), and
// TestImagePathRunsGeneratedCode reads that off the source more strictly than
// TestNoGoComputeOnTheInferencePath does for the engine as a whole.
var imagePathFiles = []string{
	"image.go", "imageproc.go", "resampler.go", "qwen3vit.go", "glmvit.go", "kimivit.go", "phi4vision.go",
	"llama4vision.go", "gemma4vision.go", "slices.go", "qwenvit.go", "pixtral.go", "hunyuanvit.go",
}

// imagePathAllowed is every image-path function the gate finds, and why it
// is not per-element compute. An entry that no longer matches fails the gate,
// as goComputeAllowed's do.
var imagePathAllowed = map[string]string{
	"imageproc.go:aaFloats": "the antialiased resize's filter taps in float64: geometry from the sizes, " +
		"O(output x support) once per picture (and once per grid for Phi-4's position table)",
	"imageproc.go:aaWeights": "the same taps in fixed point and the precision they share: geometry from the sizes, once per picture",
	"image.go:bilinearTaps":  "the bilinear path's resampling matrix: two taps per output from the sizes, once per picture",
	"imageproc.go:normLUT":   "the normalisation's 256 entries per channel from the tower's mean and std; PixLUT reads it per sample",
	"llama4vision.go:bf16NormLUT": "Llama 4's bfloat16 normalisation's 256 entries per channel, the same table read " +
		"the same way",
	"slices.go:slicedGrid": "MiniCPM-V's slice grid: a log of an aspect ratio per candidate grid, from the picture's sizes",
}

// TestImagePathRunsGeneratedCode is the image path's source gate. In the files
// of imagePathFiles it fails on
//
//   - a transcendental math call (Sin, Cos, Pow, Exp, Log ...) anywhere --
//     the resampler's sin/cos table and the position-table weights were these
//     before they were kernels;
//   - float arithmetic on data inside a loop (goComputeLines, the engine-wide
//     gate's test);
//   - a loop that calls, per iteration, an engine/model function doing float
//     arithmetic -- the shape the per-patch cubic weights had, arithmetic
//     moved into a helper the loop-level test cannot see;
//
// outside imagePathAllowed. Rounding and size geometry on scalars (Floor,
// Sqrt, RoundToEven of a picture's sizes) is not flagged.
func TestImagePathRunsGeneratedCode(t *testing.T) {
	testmodels.SourceTree(t)
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(root, "engine", "model")
	fset := token.NewFileSet()
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var files []*ast.File
	for _, e := range ents {
		n := e.Name()
		if !strings.HasSuffix(n, ".go") || strings.HasSuffix(n, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, filepath.Join(dir, n), nil, parser.ParseComments)
		if err != nil {
			t.Fatal(err)
		}
		if !testOnly(f) {
			files = append(files, f)
		}
	}
	found := imagePathFindings(t, fset, files, "engine/model", imagePathFiles)
	var keys []string
	for k := range found {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if _, ok := imagePathAllowed[k]; !ok {
			t.Errorf("%s runs Go arithmetic on the image path: %s -- generate a kernel (engine/nn/image.go), "+
				"or, if it is geometry from a picture's sizes, add it to imagePathAllowed with the reason",
				k, strings.Join(found[k], "; "))
		}
	}
	for k, why := range imagePathAllowed {
		if _, ok := found[k]; !ok {
			t.Errorf("imagePathAllowed[%q] (%s) matches no code any more: take it off the list", k, why)
		}
	}
	t.Logf("%d image-path function(s) with Go arithmetic, all geometry or a constant table", len(found))
}

// TestImagePathGateDiscriminates runs the gate on the shapes it exists to
// catch -- the resampler's table as it was, a cubic weight helper called per
// patch, a per-sample normalisation, a power -- and on legal geometry, which
// it must leave alone.
func TestImagePathGateDiscriminates(t *testing.T) {
	src := `package p

import "math"

type state struct{ sinW []float32 }

func (st *state) axes(n, q int) {
	for p := 0; p < n; p++ {
		for i := 0; i < q; i++ {
			om := float32(1 / math.Pow(10000, float64(float32(i)/float32(q))))
			st.sinW[p*q+i] = float32(math.Sin(float64(float32(p) * om)))
		}
	}
}

func cubicAt(t float32) float32 { return ((1.25*t-2.25)*t)*t + 1 }

func posLerp(w []float32, n int) {
	for i := 0; i < n; i++ {
		w[i] = cubicAt(float32(i) / float32(n))
	}
}

func normalize(dst []float32, px []uint8, m, s float32) {
	for i, v := range px {
		dst[i] = (float32(v) - m) / s
	}
}

func omega(y float64) float64 { return math.Exp(-y * math.Log(10000)) }

func resizeTo(w, h int) (int, int) {
	s := math.Sqrt(float64(w * h))
	return int(math.Floor(float64(w) / s)), int(math.RoundToEven(float64(h) / s))
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "violations.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	found := imagePathFindings(t, fset, []*ast.File{f}, "p", []string{"violations.go"})
	for _, want := range []string{"violations.go:axes", "violations.go:posLerp", "violations.go:normalize", "violations.go:omega"} {
		if len(found[want]) == 0 {
			t.Errorf("%s was not flagged -- the gate cannot see the shape it exists for (found %v)", want, found)
		}
	}
	if len(found["violations.go:resizeTo"]) != 0 || len(found["violations.go:cubicAt"]) != 0 {
		t.Errorf("geometry on scalars was flagged: %v", found)
	}
	t.Logf("flagged %d of the violations' functions, the geometry left alone", len(found))
}

// imagePathFindings type-checks files as package pkg and returns, keyed
// "file:func" for the functions in only, each finding's line and reason.
func imagePathFindings(t *testing.T, fset *token.FileSet, files []*ast.File, pkg string, only []string) map[string][]string {
	t.Helper()
	info := &types.Info{
		Types: map[ast.Expr]types.TypeAndValue{},
		Defs:  map[*ast.Ident]types.Object{},
		Uses:  map[*ast.Ident]types.Object{},
	}
	conf := types.Config{Importer: importer.ForCompiler(fset, "source", nil), Error: func(error) {}}
	conf.Check(pkg, fset, files, info)
	if len(info.Types) == 0 {
		t.Fatalf("%s: the type checker saw nothing -- this gate would pass by seeing no code", pkg)
	}
	isFloat := func(e ast.Expr) bool {
		tv, ok := info.Types[e]
		if !ok || tv.Value != nil {
			return false
		}
		b, ok := tv.Type.Underlying().(*types.Basic)
		return ok && b.Info()&types.IsFloat != 0
	}
	mathName := func(c *ast.CallExpr) string {
		se, ok := c.Fun.(*ast.SelectorExpr)
		if !ok {
			return ""
		}
		if id, ok := se.X.(*ast.Ident); ok && id.Name == "math" {
			return se.Sel.Name
		}
		return ""
	}
	transcendental := map[string]bool{}
	for _, n := range strings.Fields(`Sin Cos Tan Sincos Asin Acos Atan Atan2 Sinh Cosh Tanh Asinh Acosh Atanh
		Pow Pow10 Exp Exp2 Expm1 Log Log2 Log10 Log1p Erf Erfc Erfinv Erfcinv Gamma Lgamma Cbrt Hypot J0 J1 Y0 Y1`) {
		transcendental[n] = true
	}
	// arith reports whether a function body does float arithmetic at all:
	// a non-constant float +, -, * or /, or a math call that is not a bit
	// cast or a constant.
	arith := func(body *ast.BlockStmt) bool {
		hit := false
		ast.Inspect(body, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.BinaryExpr:
				switch x.Op {
				case token.ADD, token.SUB, token.MUL, token.QUO:
					if isFloat(x) {
						hit = true
					}
				}
			case *ast.CallExpr:
				if m := mathName(x); m != "" && !strings.HasPrefix(m, "Float") && m != "Inf" && m != "NaN" &&
					m != "IsNaN" && m != "IsInf" && m != "Signbit" {
					hit = true
				}
			}
			return !hit
		})
		return hit
	}
	decls := map[types.Object]*ast.FuncDecl{}
	for _, f := range files {
		for _, d := range f.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && fd.Body != nil {
				if obj := info.Defs[fd.Name]; obj != nil {
					decls[obj] = fd
				}
			}
		}
	}
	callee := func(c *ast.CallExpr) *ast.FuncDecl {
		var id *ast.Ident
		switch fn := c.Fun.(type) {
		case *ast.Ident:
			id = fn
		case *ast.SelectorExpr:
			id = fn.Sel
		default:
			return nil
		}
		return decls[info.Uses[id]]
	}
	want := map[string]bool{}
	for _, n := range only {
		want[n] = true
	}
	found := map[string][]string{}
	for _, f := range files {
		name := filepath.Base(fset.Position(f.Pos()).Filename)
		if !want[name] {
			continue
		}
		for _, d := range f.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Body == nil {
				continue
			}
			key := name + ":" + fd.Name.Name
			add := func(n ast.Node, why string) {
				found[key] = append(found[key], fmt.Sprintf("line %d: %s", fset.Position(n.Pos()).Line, why))
			}
			ast.Inspect(fd.Body, func(n ast.Node) bool {
				if c, ok := n.(*ast.CallExpr); ok && transcendental[mathName(c)] {
					add(c, "math."+mathName(c))
				}
				return true
			})
			for _, l := range goComputeLines(fset, info, fd.Body) {
				found[key] = append(found[key], fmt.Sprintf("line %d: float arithmetic on data in a loop", l))
			}
			ast.Inspect(fd.Body, func(n ast.Node) bool {
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
				ast.Inspect(loop, func(m ast.Node) bool {
					if c, ok := m.(*ast.CallExpr); ok {
						if g := callee(c); g != nil && g != fd && arith(g.Body) {
							add(c, "calls "+g.Name.Name+", which does float arithmetic, per iteration")
						}
					}
					return true
				})
				return false
			})
		}
	}
	return found
}
