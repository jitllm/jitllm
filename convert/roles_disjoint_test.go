package convert

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestRoleNamesAreDisjoint holds the role tables apart. roleOf asks the core
// table and then each group in turn, so a name in two of them would be read
// as the first one's role and the second's entry would never run; the tables
// are switches, which refuse a repeated name inside one of them but not
// across them.
func TestRoleNamesAreDisjoint(t *testing.T) {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, ".", func(fi fs.FileInfo) bool {
		return !strings.HasSuffix(fi.Name(), "_test.go")
	}, 0)
	if err != nil {
		t.Fatal(err)
	}
	tables := map[string]bool{
		"coreRoleOf": true, "deepseek4RoleOf": true, "flagshipRoleOf": true, "gemma3nRoleOf": true,
		"gemma3nvRoleOf": true, "kimiK3RoleOf": true, "qwen4expRoleOf": true, "visionFamRoleOf": true,
	}
	owner := map[string]string{}
	found := map[string]bool{}
	for _, pkg := range pkgs {
		for _, f := range pkg.Files {
			for _, d := range f.Decls {
				fd, ok := d.(*ast.FuncDecl)
				if !ok || !tables[fd.Name.Name] {
					continue
				}
				found[fd.Name.Name] = true
				ast.Inspect(fd.Body, func(n ast.Node) bool {
					cc, ok := n.(*ast.CaseClause)
					if !ok {
						return true
					}
					for _, e := range cc.List {
						lit, ok := e.(*ast.BasicLit)
						if !ok || lit.Kind != token.STRING {
							continue
						}
						name, err := strconv.Unquote(lit.Value)
						if err != nil {
							t.Fatal(err)
						}
						if prev, dup := owner[name]; dup && prev != fd.Name.Name {
							t.Errorf("%q is in both %s and %s", name, prev, fd.Name.Name)
						}
						owner[name] = fd.Name.Name
					}
					return true
				})
			}
		}
	}
	for name := range tables {
		if !found[name] {
			t.Errorf("%s was not found: this gate read nothing of it", name)
		}
	}
	if len(owner) < 300 {
		t.Fatalf("only %d names read from the tables -- this gate proved nothing", len(owner))
	}
	for r := jlm.RoleDecHiddenNorm; r <= jlm.RoleDecFFNDownBias; r++ {
		if !r.Valid() {
			continue
		}
		n := r.String()
		for _, name := range []string{n, n + ".weight"} {
			hits := 0
			for _, group := range roleGroups() {
				if _, ok := group(name); ok {
					hits++
				}
			}
			if _, ok := coreRoleOf(name); ok {
				hits++
			}
			if hits > 1 {
				t.Errorf("%q is named by %d tables", name, hits)
			}
		}
	}
}
