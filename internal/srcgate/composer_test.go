package srcgate

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

const module = "github.com/jitllm/jitllm/"

// composerImports is everything jit/ may import beyond the standard library
// and its own packages.
var composerImports = map[string]bool{
	module + "engine/graph": true,
	module + "format/jlm":   true,
	module + "format/quant": true,
}

// decomposition is the one file that may name a graph op kind.
const decomposition = "jit/decompose.go"

func TestJITIsTheComposerAlone(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range composerViolations(t, root, filepath.Join(root, "jit")) {
		t.Error(v)
	}
}

// TestComposerGateDiscriminates plants each kind of violation in a scratch
// tree and asks that the gate name it.
func TestComposerGateDiscriminates(t *testing.T) {
	for name, src := range map[string]string{
		"old jit":     `package jit; import _ "github.com/jitllm/jitllm/jit_old/gpu/ir"`,
		"paritytest":  `package jit; import _ "github.com/jitllm/jitllm/paritytest"`,
		"engine":      `package jit; import _ "github.com/jitllm/jitllm/engine/nn"`,
		"op kind":     `package jit; import "github.com/jitllm/jitllm/engine/graph"; var _ = graph.OpMatVec`,
		"arch":        `package jit; import "github.com/jitllm/jitllm/format/jlm"; var _ = jlm.ArchLlama`,
		"family name": `package jit; func emitGemmaNorm() {}`,
		"family str":  `package jit; var s = "qwen"`,
	} {
		dir := t.TempDir()
		if err := os.MkdirAll(filepath.Join(dir, "jit"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "jit", "x.go"), []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
		if vs := composerViolations(t, dir, filepath.Join(dir, "jit")); len(vs) == 0 {
			t.Errorf("%s: the gate let it through", name)
		}
	}
}

func composerViolations(t *testing.T, root, dir string) []string {
	var out []string
	fset := token.NewFileSet()
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			return err
		}
		rel, _ := filepath.Rel(root, p)
		rel = filepath.ToSlash(rel)
		f, err := parser.ParseFile(fset, p, nil, parser.SkipObjectResolution)
		if err != nil {
			return err
		}
		pkgOf := map[string]string{}
		for _, im := range f.Imports {
			path, _ := strconv.Unquote(im.Path.Value)
			name := path[strings.LastIndex(path, "/")+1:]
			if im.Name != nil {
				name = im.Name.Name
			}
			pkgOf[name] = path
			std := !strings.Contains(strings.SplitN(path, "/", 2)[0], ".")
			own := strings.HasPrefix(path, module+"jit/")
			if !std && !own && !composerImports[path] {
				out = append(out, rel+": imports "+path)
			}
		}
		named := rel == decomposition
		ast.Inspect(f, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.SelectorExpr:
				id, ok := x.X.(*ast.Ident)
				if !ok {
					break
				}
				switch {
				case pkgOf[id.Name] == module+"engine/graph" && strings.HasPrefix(x.Sel.Name, "Op") && !named:
					out = append(out, rel+": names graph."+x.Sel.Name+" outside the decomposition")
				case pkgOf[id.Name] == module+"format/jlm" && strings.HasPrefix(x.Sel.Name, "Arch"):
					out = append(out, rel+": names jlm."+x.Sel.Name)
				}
			case *ast.Ident:
				if familyNamed(x.Name) {
					out = append(out, rel+": "+x.Name+" names a model family")
				}
			case *ast.BasicLit:
				if x.Kind == token.STRING {
					for _, w := range strings.FieldsFunc(strings.ToLower(x.Value), func(r rune) bool {
						return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9')
					}) {
						if familyToken(w) {
							out = append(out, rel+": a string names "+w)
						}
					}
				}
			}
			return true
		})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}
