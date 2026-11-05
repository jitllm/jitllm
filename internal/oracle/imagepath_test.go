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

// imagePathAllowed() is every image-path function the gate finds, and why it
// is not per-element compute. An entry that no longer matches fails the gate,
// as goComputeAllowed()'s do.
func imagePathAllowed() map[string]string {
	return map[string]string{
		"model.go:AttnTemp":     "Llama 4's attention temperature at a position, the config's statement its gates read; the composed blocks state it as ops",
		"model.go:configFrom":   "YaRN's attention scale from the config, once at open",
		"hotness.go:ExpertUses": "expert-use accounting for the report",
		"wav.go:DecodeWAV":      "a WAV container's 16-bit PCM read into floats: a format decoder, as image/png's is for a picture",
		"tower.go:TileGrid":     "InternVL's tile grid, the candidate counts walked against the picture's aspect: geometry from the picture's sizes",
		"model.go:ropeScale":    "a scaled rotary's per-pair divisors, the shipped factors times the scaling, once at open",
	}
}

// imagePathCallees() is every model or engine function a loop on the image path
// may call although it does float arithmetic, and why that is no per-element
// compute: measurement and a block's setup at open (RULE 8b), not a pixel's
// or a patch's. An entry that no loop calls any more fails the gate.
func imagePathCallees() map[string]string {
	return map[string]string{
		"seamtune.go:seamStep":                  "the seam tuner's timing statistics, once a step (RULE 8b's measurement)",
		"prefillstream.go:after":                "the prompt streamer's timing of a chunk, once a chunk (RULE 8b's measurement)",
		"delta.go:Delta":                        "a recurrent block's geometry and scale from the config, once a block at open",
		"tower.go:towerEntry":                   "a tower's entry stated as a graph, once at open: its constants are the graph's",
		"tower.go:towerBlock":                   "a tower's block stated as a graph, once a block at open: its constants are the graph's",
		"front.go:blockGraph":                   "a block's graph stated from the config, once a block at open: its constants are the graph's",
		"frontstreams.go:streamCollapseGraphAt": "a block's streams' collapse stated as a graph, once a block at open: its constants are the graph's",
	}
}

// TestImagePathRunsGeneratedCode is the image path's source gate. A picture
// enters the model as the graph's ops (OpPatchify and the tower's blocks),
// so the gate reads every file of model/ and engine/, where any Go of the
// image path lands, and fails on
//
//   - a transcendental math call (Sin, Cos, Pow, Exp, Log ...) anywhere --
//     the resampler's sin/cos table and the position-table weights were these
//     before they were kernels;
//   - float arithmetic on data inside a loop (goComputeLines, the engine-wide
//     gate's test);
//   - a loop that calls, per iteration, a model or engine function doing
//     float arithmetic -- the shape the per-patch cubic weights had,
//     arithmetic moved into a helper the loop-level test cannot see, in its
//     package or the other;
//
// outside imagePathAllowed(). Rounding and size geometry on scalars (Floor,
// Sqrt, RoundToEven of a picture's sizes) is not flagged.
func TestImagePathRunsGeneratedCode(t *testing.T) {
	testmodels.SourceTree(t)
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	pkgs := []imagePkg{
		{"github.com/jitllm/jitllm/model", imagePathFiles(t, fset, filepath.Join(root, "model"))},
		{"github.com/jitllm/jitllm/engine", imagePathFiles(t, fset, filepath.Join(root, "engine"))},
	}
	called := map[string]bool{}
	found := imagePathFindings(t, fset, pkgs, nil, imagePathCallees(), called)
	var keys []string
	for k := range found {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if _, ok := imagePathAllowed()[k]; !ok {
			t.Errorf("%s runs Go arithmetic on the image path: %s -- generate a kernel (engine/nn/image.go), "+
				"or, if it is geometry from a picture's sizes, add it to imagePathAllowed() with the reason",
				k, strings.Join(found[k], "; "))
		}
	}
	for k, why := range imagePathAllowed() {
		if _, ok := found[k]; !ok {
			t.Errorf("imagePathAllowed()[%q] (%s) matches no code any more: take it off the list", k, why)
		}
	}
	for k, why := range imagePathCallees() {
		if !called[k] {
			t.Errorf("imagePathCallees()[%q] (%s) is called by no loop any more: take it off the list", k, why)
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

func Weight(t float32) float32 { return ((1.25*t-2.25)*t)*t + 1 }
`
	// The engine's loop calling the model's arithmetic per patch: a call
	// across the two packages.
	cross := `package q

import "p"

func patches(w []float32, n int) {
	for i := 0; i < n; i++ {
		w[i] = p.Weight(float32(i) / float32(n))
	}
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "violations.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	g, err := parser.ParseFile(fset, "cross.go", cross, 0)
	if err != nil {
		t.Fatal(err)
	}
	found := imagePathFindings(t, fset, []imagePkg{{"p", []*ast.File{f}}, {"q", []*ast.File{g}}},
		[]string{"violations.go", "cross.go"}, nil, map[string]bool{})
	for _, want := range []string{"violations.go:axes", "violations.go:posLerp", "violations.go:normalize", "violations.go:omega", "cross.go:patches"} {
		if len(found[want]) == 0 {
			t.Errorf("%s was not flagged -- the gate cannot see the shape it exists for (found %v)", want, found)
		}
	}
	if len(found["violations.go:resizeTo"]) != 0 || len(found["violations.go:cubicAt"]) != 0 || len(found["violations.go:Weight"]) != 0 {
		t.Errorf("geometry on scalars was flagged: %v", found)
	}
	t.Logf("flagged %d of the violations' functions, the geometry left alone", len(found))
}

// imagePkg is one package the image path's gate reads: the path it is checked
// as and its files.
type imagePkg struct {
	path  string
	files []*ast.File
}

// imagePathFiles is dir's production files, parsed into fset.
func imagePathFiles(t *testing.T, fset *token.FileSet, dir string) []*ast.File {
	t.Helper()
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
	return files
}

// importerFunc is a types.Importer of a func.
type importerFunc func(path string) (*types.Package, error)

func (f importerFunc) Import(path string) (*types.Package, error) { return f(path) }

// imagePathFindings type-checks pkgs in order, a later one importing an
// earlier as checked (so a call across them is the callee's own
// declaration), and returns, keyed "file:func" for the functions of the
// files only names (every file where only is empty), each finding's line and
// reason.
func imagePathFindings(t *testing.T, fset *token.FileSet, pkgs []imagePkg, only []string, callees map[string]string, called map[string]bool) map[string][]string {
	t.Helper()
	src := importer.ForCompiler(fset, "source", nil)
	checked := map[string]*types.Package{}
	infos := make([]*types.Info, len(pkgs))
	for i, p := range pkgs {
		info := &types.Info{
			Types: map[ast.Expr]types.TypeAndValue{},
			Defs:  map[*ast.Ident]types.Object{},
			Uses:  map[*ast.Ident]types.Object{},
		}
		conf := types.Config{Importer: importerFunc(func(path string) (*types.Package, error) {
			if c := checked[path]; c != nil {
				return c, nil
			}
			return src.Import(path)
		}), Error: func(error) {}}
		pkg, _ := conf.Check(p.path, fset, p.files, info)
		if len(info.Types) == 0 {
			t.Fatalf("%s: the type checker saw nothing -- this gate would pass by seeing no code", p.path)
		}
		checked[p.path], infos[i] = pkg, info
	}
	isFloat := func(info *types.Info, e ast.Expr) bool {
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
	arith := func(info *types.Info, body *ast.BlockStmt) bool {
		hit := false
		ast.Inspect(body, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.BinaryExpr:
				switch x.Op {
				case token.ADD, token.SUB, token.MUL, token.QUO:
					if isFloat(info, x) {
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
	declInfo := map[*ast.FuncDecl]*types.Info{}
	for i, p := range pkgs {
		for _, f := range p.files {
			for _, d := range f.Decls {
				if fd, ok := d.(*ast.FuncDecl); ok && fd.Body != nil {
					if obj := infos[i].Defs[fd.Name]; obj != nil {
						decls[obj], declInfo[fd] = fd, infos[i]
					}
				}
			}
		}
	}
	want := map[string]bool{}
	for _, n := range only {
		want[n] = true
	}
	found := map[string][]string{}
	for i, p := range pkgs {
		info := infos[i]
		findingsOf(fset, info, p.files, want, decls, declInfo, arith, mathName, transcendental, callees, called, found)
	}
	return found
}

// findingsOf adds to found the findings in files, checked into info: the
// transcendental calls, the loops' float arithmetic, and a loop's call of a
// function in decls that does float arithmetic (arith) and callees does not
// excuse.
func findingsOf(fset *token.FileSet, info *types.Info, files []*ast.File, want map[string]bool,
	decls map[types.Object]*ast.FuncDecl, declInfo map[*ast.FuncDecl]*types.Info,
	arith func(*types.Info, *ast.BlockStmt) bool, mathName func(*ast.CallExpr) string,
	transcendental map[string]bool, callees map[string]string, called map[string]bool, found map[string][]string) {
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
	for _, f := range files {
		name := filepath.Base(fset.Position(f.Pos()).Filename)
		if len(want) > 0 && !want[name] {
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
						if g := callee(c); g != nil && g != fd && arith(declInfo[g], g.Body) {
							k := filepath.Base(fset.Position(g.Pos()).Filename) + ":" + g.Name.Name
							if _, ok := callees[k]; ok {
								called[k] = true
								return true
							}
							add(c, "calls "+g.Name.Name+", which does float arithmetic, per iteration")
						}
					}
					return true
				})
				return false
			})
		}
	}
}
