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

// goComputeAllowed is every production function that may do float arithmetic
// on data inside a loop, and why. Anything else fails
// TestNoGoComputeOnTheInferencePath, and so does an entry that no longer
// matches -- a site that was fixed comes off this list rather than leaving a
// hole shaped like it.
var goComputeAllowed = map[string]string{
	// Measurement, not model arithmetic: timing statistics and accounting.
	"jit/gpu/tier/split.go:timeSequences": "median/IQR of the split tuner's timings",
	"engine/model/hotness.go:ExpertUses":  "expert-use accounting for the report",
	// Diagnostics that compare the device against a reference, off by default.
	"jit/gpu/tier/tier.go:verify":           "Config verify: a device matvec against the dequantized reference, a debugging knob",
	"jit/gpu/tier/layer.go:softmaxAgreesAt": "a load-time probe: whether the subgroup softmax agrees with the plain one on this card",
	// Constants built once at load from the config, like the tables a kernel is
	// compiled with.
	"engine/nn/ropetable.go:buildRopeTab":      "the rotary table's folded constant planes, once per configuration",
	"engine/nn/ropetable.go:YarnFreqs":         "YaRN's per-dimension frequencies, once per configuration",
	"jit/gpu/tier/layer.go:l4TempTab":          "Llama 4's attention-temperature table, once per plan",
	"engine/model/image.go:bilinearTaps":       "the resize's resampling matrix: geometry from the image's sizes, once per image",
	"engine/model/imageproc.go:aaFloats":       "the antialiased resize's filter taps: geometry from the image's sizes, once per image (and once per grid for Phi-4's position table)",
	"engine/model/imageproc.go:normLUT":        "the normalisation's 256-entry table per channel, from the tower's mean and std",
	"engine/model/llama4vision.go:bf16NormLUT": "the bfloat16 normalisation's 256-entry table per channel, from the tower's mean and std",
	// Geometry of a picture or a grid, from its sizes alone: no pixel and no
	// activation is read.
	"engine/model/slices.go:slicedGrid":       "MiniCPM-V's slice grid: which cols x rows best matches the picture's aspect, from its width and height",
	"engine/nn/ropeaxes.go:runFreqs":          "a multi-axis rotary's per-pair frequencies, once per configuration",
	"engine/model/audio.go:encodeAudio":       "a log-mel feature's normalising scale: one scalar per feature from the generated sums, as a tuner's statistics are; every frame's arithmetic is a kernel",
	"engine/model/audio.go:audioBlocks":       "the relative positions' sinusoids: a table of the clip's row count, geometry as the rotary tables are; no activation is read",
	"engine/model/audio.go:buildAudio":        "the depthwise bias folded into the BatchNorm's shift, once at load: a load-time constant of the weights",
	"engine/model/audiodev.go:audioPE":        "the relative positions' sinusoids for a device's scratch: a table of its sizes, built once",
	"engine/model/audio.go:tables":            "the windowed DFT bases: load-time constant tables of the tower's framing",
	"engine/model/audio.go:slaneyFilterbank":  "the Slaney mel filterbank: a load-time constant table of the tower's framing",
	"engine/model/wav.go:DecodeWAV":           "a WAV container's 16-bit PCM read into floats: a format decoder, as image/png's is for a picture",
	"engine/model/videoproc.go:sampleIndices": "a video's sampled frame indices (np.linspace): geometry from the frame count and rate, once per video; no pixel is read",
}

// TestNoGoComputeOnTheInferencePath is AGENTS.md RULE 8 read off the source:
// every op a token runs is generated code, so production Go may not do float
// arithmetic on data in a loop -- a float slice element in +,-,*,/ (or
// op-assigned), a computed float stored into a float element, or a math.* call,
// inside a for or range body -- outside goComputeAllowed. A scalar computed
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
	for _, pkg := range []string{"engine/model", "engine/nn", "jit/gpu/tier", "jit/gpu/backend", "engine/sched", "jit/cpu"} {
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
		info := &types.Info{Types: map[ast.Expr]types.TypeAndValue{}}
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
		if _, ok := goComputeAllowed[k]; !ok {
			t.Errorf("%s does float arithmetic on data in a loop at line(s) %v: RULE 8 -- generate a kernel "+
				"(or, if it is measurement or a load-time constant, add it to goComputeAllowed with the reason)",
				k, found[k])
		}
	}
	for k, why := range goComputeAllowed {
		if _, ok := found[k]; !ok {
			t.Errorf("goComputeAllowed[%q] (%s) matches no code any more: take it off the list", k, why)
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
	floatElem := func(n ast.Node) bool {
		hit := false
		ast.Inspect(n, func(z ast.Node) bool {
			if ix, ok := z.(*ast.IndexExpr); ok && isFloat(ix) {
				hit = true
			}
			return !hit
		})
		return hit
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
