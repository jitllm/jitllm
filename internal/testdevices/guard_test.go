package testdevices_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

const me = "github.com/jitllm/jitllm/internal/testdevices"

// walkGo calls f for every Go file under the repository, the nested modules
// included, skipping what is not this tree's source.
func walkGo(t *testing.T, f func(path string)) {
	t.Helper()
	testmodels.SourceTree(t)
	root := filepath.Join("..", "..")
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if p != root && strings.HasPrefix(d.Name(), ".") {
				return filepath.SkipDir
			}
			switch d.Name() {
			case "gen", "vendor", "node_modules", "testdata", "models":
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(p, ".go") && !strings.Contains(filepath.ToSlash(p), "internal/testdevices/") {
			f(p)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// TestNoProductionCodeImportsTestDevices: a shipped binary reads no
// environment, and this package reads one.
func TestNoProductionCodeImportsTestDevices(t *testing.T) {
	fset := token.NewFileSet()
	checked := 0
	walkGo(t, func(p string) {
		// internal/testlock is test support too: it imports "testing".
		if strings.HasSuffix(p, "_test.go") || strings.Contains(filepath.ToSlash(p), "internal/testlock/") {
			return
		}
		f, err := parser.ParseFile(fset, p, nil, parser.ImportsOnly)
		if err != nil {
			t.Fatal(err)
		}
		checked++
		for _, im := range f.Imports {
			if strings.Trim(im.Path.Value, `"`) == me {
				t.Errorf("%s imports %s: it is for tests only", p, me)
			}
		}
	})
	if checked == 0 {
		t.Fatal("walked no production file: the guard looked at nothing")
	}
}

// opens is every call that reaches a device, as package.name: a backend open
// or listing, a tier open, a server engine and its probe.
func opens() map[string]bool {
	m := map[string]bool{"gpu.OpenWith": true, "server.New": true, "server.probeDevices": true}
	for _, n := range []string{"OpenWith", "OpenCUDAWith", "OpenVulkanWith", "OpenHIPWith", "OpenMetalWith",
		"VulkanDevicesWith", "NewVulkanDriver"} {
		m["backend."+n] = true
	}
	return m
}

// TestEveryDeviceOpenCarriesTheSelection: the selection holds only for an open
// that carries it (Admit: backend.Opts.Admit, gpu.WithAdmit, server's
// Config.Admit), so a test whose open does not is the card a shared box
// loses. Every device open in a test is in a function that names
// testdevices.Admit or calls one of its package's test functions that does,
// and the opens that take no options (backend.Open, OpenCUDA, OpenVulkan,
// VulkanDevices, vulkan.NewDriver) are not called from a test at all.
func TestEveryDeviceOpenCarriesTheSelection(t *testing.T) {
	type fn struct {
		name   string
		admits bool
		calls  map[string]bool
		opens  []string
	}
	byDir := map[string][]*fn{}
	bare := map[string]bool{"backend.Open": true, "backend.OpenCUDA": true, "backend.OpenVulkan": true,
		"backend.VulkanDevices": true, "vulkan.NewDriver": true}
	opener := opens()
	fset := token.NewFileSet()
	count := 0
	walkGo(t, func(p string) {
		if !strings.HasSuffix(p, "_test.go") {
			return
		}
		f, err := parser.ParseFile(fset, p, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		rel, err := filepath.Rel(filepath.Join("..", ".."), filepath.Dir(p))
		if err != nil {
			t.Fatal(err)
		}
		rel = filepath.ToSlash(rel)
		if rel == "backend/vulkan" {
			return // the driver's own tests, below every open
		}
		pkg := f.Name.Name
		for _, d := range f.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Body == nil {
				continue
			}
			e := &fn{name: fd.Name.Name, calls: map[string]bool{}}
			ast.Inspect(fd.Body, func(n ast.Node) bool {
				c, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				q, name := "", ""
				switch f := c.Fun.(type) {
				case *ast.Ident:
					name = f.Name
					switch {
					case rel == "backend" && pkg == "backend":
						q = "backend"
					case rel == "engine/device/gpu" && pkg == "gpu":
						q = "gpu"
					case rel == "server" && pkg == "server" && opener["server."+name]:
						q = "server"
					}
				case *ast.SelectorExpr:
					if x, ok := f.X.(*ast.Ident); ok {
						q, name = x.Name, f.Sel.Name
					}
				}
				e.calls[name] = true
				if q == "testdevices" && name == "Admit" {
					e.admits = true
				}
				at := fset.Position(c.Pos()).String()
				switch {
				case bare[q+"."+name]:
					t.Errorf("%s: %s.%s takes no options, so it cannot carry the selection: "+
						"call its With form with Admit: testdevices.Admit()", at, q, name)
				case opener[q+"."+name]:
					e.opens = append(e.opens, at+": "+q+"."+name)
				}
				return true
			})
			byDir[rel] = append(byDir[rel], e)
		}
	})
	for dir, fns := range byDir {
		carries := map[string]bool{}
		for grew := true; grew; {
			grew = false
			for _, e := range fns {
				if carries[e.name] {
					continue
				}
				ok := e.admits
				for c := range e.calls {
					ok = ok || carries[c]
				}
				if ok {
					carries[e.name], grew = true, true
				}
			}
		}
		for _, e := range fns {
			count += len(e.opens)
			if carries[e.name] {
				continue
			}
			for _, o := range e.opens {
				t.Errorf("%s (in %s, %s): the open carries no selection: pass testdevices.Admit() "+
					"(backend.Opts.Admit, gpu.WithAdmit, Config.Admit), or call a helper of the package's tests that does",
					o, e.name, dir)
			}
		}
	}
	if count < 50 {
		t.Fatalf("found %d device opens in the tests: the guard looked at nothing", count)
	}
}
