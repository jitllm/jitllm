package srcgate

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"unicode"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// familyDirs are the trees that run a model: none of them may know which
// family a model is. The converter is where a family's facts are read and
// stated (RULE 7); past it a model is the graph its container states.
func familyDirs() []string { return []string{"model", "engine", "nn", "composer", "graph", "backend"} }

// familyShort are names this tree gave families that no architecture
// constant spells: Kimi-K3, Qwen3.8-Flash-Next, DeepSeek-V4 and V3.2, and the
// sparse attentions named for MiniMax and Qwen.
func familyShort() []string { return []string{"k3", "q4e", "ds4", "ds32", "msa", "qsa"} }

// familyBare are stems that name a family only with a digit after them:
// alone, phi is an SSA phi node.
func familyBare(s string) bool { return s == "phi" }

// familyGeneric are architecture-name stems that are also ordinary words.
func familyGeneric(s string) bool {
	switch s {
	case "seed", "clip", "dots", "modern", "command", "stable", "nomic", "arcee", "gpt", "lfm", "smol", "star":
		return true
	}
	return false
}

// TestNoFamilyCode holds the engine to knowing no family: the architecture
// is the converter's alone, so nothing under familyDirs imports convert/ or
// reads its architecture type or constants, nothing asks an architecture
// what it means (a method on an Arch value but its name and validity),
// format/jlm declares no architecture (no type, constant, name table or field
// named for one), and no file, type, function, method, field, variable or
// constant under familyDirs is named for a family. The names are the
// architecture constants' own (ArchQwen4Exp is qwen4exp, its stem qwen) and
// familyShort. Comments and strings are not read. The violations -- the
// converter imported in a model, an architecture read and asked there, an
// architecture declared in format/jlm, a family's field, a family's file --
// are seen.
func TestNoFamilyCode(t *testing.T) {
	testmodels.SourceTree(t)
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	fam := familyNames(t, root)
	if len(fam.full) < 40 {
		t.Fatalf("read %d architecture constants: the gate read nothing", len(fam.full))
	}
	files := 0
	for _, d := range familyDirs() {
		err := filepath.Walk(filepath.Join(root, d), func(p string, fi os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			if fi.IsDir() && (fi.Name() == "testdata" || strings.HasPrefix(fi.Name(), ".")) {
				return filepath.SkipDir
			}
			if fi.IsDir() || !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
				return nil
			}
			rel, _ := filepath.Rel(root, p)
			files++
			if fam.fileNamed(fi.Name()) {
				t.Errorf("%s: a file named for a family", rel)
			}
			src, err := os.ReadFile(p)
			if err != nil {
				return err
			}
			for _, bad := range fam.inSource(rel, src) {
				t.Errorf("%s", bad)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if files < 300 {
		t.Fatalf("read %d files under %v: the gate read nothing", files, familyDirs())
	}
	jf, err := filepath.Glob(filepath.Join(root, "format", "jlm", "*.go"))
	if err != nil {
		t.Fatal(err)
	}
	read := 0
	for _, p := range jf {
		if strings.HasSuffix(p, "_test.go") {
			continue
		}
		src, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		rel, _ := filepath.Rel(root, p)
		read++
		for _, b := range declaresArch(rel, src) {
			t.Errorf("%s", b)
		}
	}
	if read < 20 {
		t.Fatalf("read %d files in format/jlm: the gate read nothing", read)
	}

	viol := `package model

import "github.com/jitllm/jitllm/convert"

type layer struct{ q4e bool }

func swa(a convert.Arch) bool { return a == convert.ArchGemma4 }

func isQwen3Block() {}

func opPhi() {}

func rope(c *convert.Config) bool { return c.Arch.XDRope() || c.Arch.Valid() }
`
	got := fam.inSource("model/viol.go", []byte(viol))
	if len(got) != 6 {
		t.Errorf("the violations gave %d findings, want 6 (the converter imported, a field, the architecture type, "+
			"an architecture, a function, an architecture asked): %v", len(got), got)
	}
	jviol := `package jlm

type Arch uint16

const ArchHunyuanVL Arch = 1

func viewArchName(a Arch) (string, bool) { return "hunyuan_vl", a == 1 }

type Config struct{ Arch uint16 }
`
	if bad := declaresArch("format/jlm/viol.go", []byte(jviol)); len(bad) != 4 {
		t.Errorf("an architecture declared in format/jlm gave %v, want 4 findings (a type, a constant, a table, a field)", bad)
	}
	for _, n := range []string{"q4erun.go", "gemma4.go", "k3run.go"} {
		if !fam.fileNamed(n) {
			t.Errorf("the violation %s is not seen as a family's file", n)
		}
	}
	for _, n := range []string{"mrope.go", "philox.go", "llamacpp_engine.go", "phase.go", "mla.go"} {
		if fam.fileNamed(n) {
			t.Errorf("%s is seen as a family's file, and is not one", n)
		}
	}
}

type families struct {
	// arch is every architecture constant's name (ArchQwen4Exp); full their
	// lowercased names (qwen4exp) and stems their leading letters (qwen).
	arch        map[string]bool
	full, stems []string
}

// familyNames reads the architecture constants from the converter.
func familyNames(t *testing.T, root string) families {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, filepath.Join(root, "convert"), func(fi os.FileInfo) bool {
		return !strings.HasSuffix(fi.Name(), "_test.go")
	}, 0)
	if err != nil {
		t.Fatal(err)
	}
	f := families{arch: map[string]bool{}}
	seen := map[string]bool{}
	for _, p := range pkgs {
		for _, file := range p.Files {
			for _, d := range file.Decls {
				gd, ok := d.(*ast.GenDecl)
				if !ok || gd.Tok != token.CONST {
					continue
				}
				for _, sp := range gd.Specs {
					vs := sp.(*ast.ValueSpec)
					for _, n := range vs.Names {
						name := n.Name
						if !strings.HasPrefix(name, "Arch") || name == "ArchNone" || len(name) < 6 || !unicode.IsUpper(rune(name[4])) {
							continue
						}
						f.arch[name] = true
						low := strings.ToLower(name[4:])
						f.full = append(f.full, low)
						stem := low
						if i := strings.IndexFunc(low, unicode.IsDigit); i > 0 {
							stem = low[:i]
						}
						if len(stem) >= 3 && !familyGeneric(stem) && !seen[stem] {
							seen[stem] = true
							f.stems = append(f.stems, stem)
						}
					}
				}
			}
		}
	}
	sort.Strings(f.stems)
	return f
}

// words splits an identifier into its lowercased words: at each upper-case
// letter that starts one and at underscores, a word's digits kept on it
// (Q4eHead is q4e, head; isGemma4 is is, gemma4).
func words(id string) []string {
	id = strings.ReplaceAll(strings.ReplaceAll(id, "LlamaCpp", "_"), "llamacpp", "_")
	var out []string
	cur := []rune{}
	rs := []rune(id)
	for i, r := range rs {
		if r == '_' {
			if len(cur) > 0 {
				out = append(out, strings.ToLower(string(cur)))
			}
			cur = cur[:0]
			continue
		}
		if unicode.IsUpper(r) && len(cur) > 0 {
			prevUpper := unicode.IsUpper(rs[i-1])
			nextLower := i+1 < len(rs) && unicode.IsLower(rs[i+1])
			if !prevUpper || nextLower {
				out = append(out, strings.ToLower(string(cur)))
				cur = cur[:0]
			}
		}
		cur = append(cur, r)
	}
	if len(cur) > 0 {
		out = append(out, strings.ToLower(string(cur)))
	}
	return out
}

// wordNamed is whether one word names a family: a short name or a stem
// alone or followed by digits (phi3, gemma4n), or a full name.
func (f families) wordNamed(w string) bool {
	for _, s := range familyShort() {
		if w == s {
			return true
		}
	}
	for _, s := range f.full {
		if len(s) >= 5 && w == s {
			return true
		}
	}
	for _, s := range f.stems {
		if w == s && !familyBare(s) || (strings.HasPrefix(w, s) && len(w) > len(s) && unicode.IsDigit(rune(w[len(s)]))) {
			return true
		}
	}
	return false
}

// identNamed is whether an identifier names a family by any of its words, or
// holds a full name of five letters or more anywhere (Qwen3Next).
func (f families) identNamed(id string) bool {
	for _, w := range words(id) {
		if f.wordNamed(w) {
			return true
		}
	}
	low := strings.ToLower(strings.ReplaceAll(strings.ReplaceAll(id, "LlamaCpp", "_"), "llamacpp", "_"))
	for _, s := range f.full {
		if len(s) >= 6 && strings.Contains(low, s) {
			return true
		}
	}
	return false
}

// declaresArch is every declaration in format/jlm source that names an
// architecture: a type, constant, function or field called Arch, Arch... or
// ...ArchName.
func declaresArch(rel string, src []byte) []string {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, rel, src, 0)
	if err != nil {
		return []string{rel + ": " + err.Error()}
	}
	named := func(n string) bool {
		return n == "Arch" || strings.HasSuffix(n, "ArchName") ||
			strings.HasPrefix(n, "Arch") && len(n) > 4 && unicode.IsUpper(rune(n[4]))
	}
	var out []string
	at := func(p token.Pos, what, n string) {
		out = append(out, fset.Position(p).String()+": "+what+" "+n+": the architecture is the converter's alone")
	}
	ast.Inspect(file, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.FuncDecl:
			if named(x.Name.Name) {
				at(x.Pos(), "a function", x.Name.Name)
			}
		case *ast.TypeSpec:
			if named(x.Name.Name) {
				at(x.Pos(), "a type", x.Name.Name)
			}
		case *ast.ValueSpec:
			for _, nm := range x.Names {
				if named(nm.Name) {
					at(nm.Pos(), "a value", nm.Name)
				}
			}
		case *ast.Field:
			for _, nm := range x.Names {
				if named(nm.Name) {
					at(nm.Pos(), "a field", nm.Name)
				}
			}
		}
		return true
	})
	return out
}

// fileNamed is whether a file's name starts with a family's name: k3run.go,
// gemma4.go, q4edev.go.
func (f families) fileNamed(name string) bool {
	base := strings.TrimSuffix(name, ".go")
	if strings.HasPrefix(base, "llamacpp") {
		return false
	}
	for _, s := range familyShort() {
		if strings.HasPrefix(base, s) {
			return true
		}
	}
	for _, s := range f.full {
		if len(s) >= 5 && strings.HasPrefix(base, s) {
			return true
		}
	}
	for _, s := range f.stems {
		if strings.HasPrefix(base, s) && len(base) > len(s) && (unicode.IsDigit(rune(base[len(s)])) || base[len(s)] == '_') {
			return true
		}
		if base == s && !familyBare(s) {
			return true
		}
	}
	return false
}

// inSource is every declaration in src named for a family and every
// architecture constant it reads.
func (f families) inSource(rel string, src []byte) []string {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, rel, src, 0)
	if err != nil {
		return []string{rel + ": " + err.Error()}
	}
	var out []string
	at := func(p token.Pos) string { return fset.Position(p).String() }
	ast.Inspect(file, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.ImportSpec:
			if x.Path.Value == `"github.com/jitllm/jitllm/convert"` {
				out = append(out, at(x.Pos())+": imports the converter: the architecture is its alone")
			}
		case *ast.SelectorExpr:
			if id, ok := x.X.(*ast.Ident); ok && id.Name == "convert" && (x.Sel.Name == "Arch" || f.arch[x.Sel.Name]) {
				out = append(out, at(x.Pos())+": reads "+x.Sel.Name+": the container states what the architecture means")
			}
		case *ast.CallExpr:
			if m, ok := x.Fun.(*ast.SelectorExpr); ok && m.Sel.Name != "String" && m.Sel.Name != "Valid" {
				if a, ok := m.X.(*ast.SelectorExpr); ok && a.Sel.Name == "Arch" {
					out = append(out, at(x.Pos())+": asks the architecture "+m.Sel.Name+": the container states what the architecture means")
				}
			}
		case *ast.FuncDecl:
			if f.identNamed(x.Name.Name) {
				out = append(out, at(x.Pos())+": a function named for a family: "+x.Name.Name)
			}
		case *ast.TypeSpec:
			if f.identNamed(x.Name.Name) {
				out = append(out, at(x.Pos())+": a type named for a family: "+x.Name.Name)
			}
		case *ast.ValueSpec:
			for _, nm := range x.Names {
				if f.identNamed(nm.Name) {
					out = append(out, at(nm.Pos())+": a value named for a family: "+nm.Name)
				}
			}
		case *ast.Field:
			for _, nm := range x.Names {
				if f.identNamed(nm.Name) {
					out = append(out, at(nm.Pos())+": a field named for a family: "+nm.Name)
				}
			}
		}
		return true
	})
	return out
}
