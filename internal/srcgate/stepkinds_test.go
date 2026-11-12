package srcgate

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestContainerStepsAreGeneric holds the container's picture steps
// (jlm.PreKind) to the generic ones, each with what makes it so. A
// processor's own resize, tiling or canvas is the converter's
// (convert/picstep.go): stated there as the picture's geometry and lowered
// to these. A step kind with one processor behind it is that family's code
// under the format's name, which the architecture rule keeps in the
// converter.
func TestContainerStepsAreGeneric(t *testing.T) {
	testmodels.SourceTree(t)
	allowed := map[string]string{
		"PreResize":        "a resize to a window fixed at conversion, any filter and arithmetic",
		"PreResizeLive":    "a resize to each square's own size, the geometry's",
		"PreCanvas":        "the resized picture laid on a canvas of a fill colour, the geometry's",
		"PreNormalize":     "the mean and std each channel takes",
		"PreNormalizeBF16": "PreNormalize in bfloat16, as torch computes it on a bfloat16 tensor",
		"PreVideoResize":   "a video's frames' resize to one size of the video's own; the next to move to the converter",
	}
	root := filepath.Join("..", "..")
	files, err := filepath.Glob(filepath.Join(root, "format", "jlm", "*.go"))
	if err != nil {
		t.Fatal(err)
	}
	found := map[string]bool{}
	read := 0
	for _, p := range files {
		if strings.HasSuffix(p, "_test.go") {
			continue
		}
		src, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		read++
		for _, k := range stepKinds(t, p, src) {
			if _, ok := allowed[k]; !ok {
				t.Errorf("format/jlm/%s declares the picture step %s: a processor's step is the converter's "+
					"(convert/picstep.go), stated as geometry and lowered to the generic ones", filepath.Base(p), k)
			}
			found[k] = true
		}
	}
	if read < 20 {
		t.Fatalf("read %d files in format/jlm: the gate read nothing", read)
	}
	for k := range allowed {
		if !found[k] {
			t.Errorf("format/jlm no longer declares %s: take it off the list", k)
		}
	}
	viol := "package jlm\n\nconst (\n\tPreResize PreKind = 1\n\tPreSmartResize PreKind = 3\n\tPreTile\n)\n"
	if got := stepKinds(t, "viol.go", []byte(viol)); len(got) != 3 || got[1] != "PreSmartResize" || got[2] != "PreTile" {
		t.Errorf("the violation's step kinds read as %v, want PreResize, PreSmartResize and PreTile", got)
	}
}

// stepKinds is every constant of type PreKind src declares, the ones a
// const block repeats the type of among them.
func stepKinds(t *testing.T, name string, src []byte) []string {
	t.Helper()
	f, err := parser.ParseFile(token.NewFileSet(), name, src, 0)
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, d := range f.Decls {
		g, ok := d.(*ast.GenDecl)
		if !ok || g.Tok != token.CONST {
			continue
		}
		typ := ""
		for _, s := range g.Specs {
			v := s.(*ast.ValueSpec)
			switch id, ok := v.Type.(*ast.Ident); {
			case ok:
				typ = id.Name
			case v.Type == nil && len(v.Values) > 0:
				typ = ""
			}
			if typ != "PreKind" {
				continue
			}
			for _, n := range v.Names {
				out = append(out, n.Name)
			}
		}
	}
	return out
}
