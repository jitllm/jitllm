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

// TestNoFamilyCode holds the engine to knowing no family: no architecture
// constant is read outside convert/ and format/jlm, no architecture is asked
// what it means past the converter (a method on an Arch value but its name
// and validity), format/jlm compares an architecture only in its name tables,
// and no file, type, function, method, field, variable or constant under
// familyDirs is named for a family. The names are the architecture
// constants' own (ArchQwen4Exp is qwen4exp, its stem qwen) and familyShort.
// Comments and strings are not read. The violations -- an architecture
// compared in a model, an architecture asked in a model, an architecture
// compared in format/jlm, a family's field, a family's file -- are seen.
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
	tables := 0
	for _, p := range jf {
		if strings.HasSuffix(p, "_test.go") {
			continue
		}
		src, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		rel, _ := filepath.Rel(root, p)
		bad, n := fam.comparedIn(rel, src)
		tables += n
		for _, b := range bad {
			t.Errorf("%s", b)
		}
	}
	if tables < 5 {
		t.Fatalf("read %d architecture name tables in format/jlm: the gate read nothing", tables)
	}

	viol := `package model

import "github.com/jitllm/jitllm/format/jlm"

type layer struct{ q4e bool }

func swa(a jlm.Arch) bool { return a == jlm.ArchGemma4 }

func isQwen3Block() {}

func opPhi() {}

func rope(c *jlm.Config) bool { return c.Arch.XDRope() || c.Arch.Valid() }
`
	got := fam.inSource("model/viol.go", []byte(viol))
	if len(got) != 4 {
		t.Errorf("the violations gave %d findings, want 4 (a field, an architecture, a function, an architecture asked): %v", len(got), got)
	}
	jviol := `package jlm

func (a Arch) XDRope() bool { return a == ArchHunyuanVL }

func viewArchName(a Arch) (string, bool) {
	switch a {
	case ArchHunyuanVL:
		return "hunyuan_vl", true
	}
	return "", false
}
`
	if bad, n := fam.comparedIn("format/jlm/viol.go", []byte(jviol)); len(bad) != 1 || n != 1 {
		t.Errorf("an architecture compared beside a name table gave %v and %d table(s), want one finding and one table", bad, n)
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

// familyNames reads the architecture constants from format/jlm.
func familyNames(t *testing.T, root string) families {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, filepath.Join(root, "format", "jlm"), func(fi os.FileInfo) bool {
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

// comparedIn is every architecture constant format/jlm source compares
// outside a name table (a function named ...ArchName, name or Valid), and how
// many name tables it holds.
func (f families) comparedIn(rel string, src []byte) ([]string, int) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, rel, src, 0)
	if err != nil {
		return []string{rel + ": " + err.Error()}, 0
	}
	var out []string
	tables := 0
	for _, d := range file.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Body == nil {
			continue
		}
		name := fd.Name.Name
		if strings.HasSuffix(name, "ArchName") {
			tables++
			continue
		}
		if name == "name" || name == "Valid" {
			continue
		}
		ast.Inspect(fd.Body, func(n ast.Node) bool {
			var ids []ast.Expr
			switch x := n.(type) {
			case *ast.BinaryExpr:
				if x.Op == token.EQL || x.Op == token.NEQ {
					ids = []ast.Expr{x.X, x.Y}
				}
			case *ast.CaseClause:
				ids = x.List
			}
			for _, e := range ids {
				if id, ok := e.(*ast.Ident); ok && f.arch[id.Name] {
					out = append(out, fset.Position(id.Pos()).String()+": "+name+" compares "+id.Name+
						": what an architecture means is stated by the converter")
				}
			}
			return true
		})
	}
	return out, tables
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
		case *ast.SelectorExpr:
			if id, ok := x.X.(*ast.Ident); ok && id.Name == "jlm" && f.arch[x.Sel.Name] {
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
