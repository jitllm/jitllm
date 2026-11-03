package testdevices_test

import (
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
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

// opensADevice() is a call that reaches a device.
func opensADevice() *regexp.Regexp {
	return regexp.MustCompile(`\b(backend\.(Open|OpenWith|OpenCUDA|OpenCUDAWith|OpenVulkan|OpenVulkanWith|OpenHIPWith|OpenMetalWith)|tier\.(Open|OpenWith|New)|LoadModel)\(`)
}

// TestEveryPackageThatOpensADeviceInstallsTheSelection: the selection holds
// only in a test binary that installed it (testdevices.Install), so a package
// whose tests reach a device and never did would be the card a shared box
// loses. A package with no device in its tests is not asked.
func TestEveryPackageThatOpensADeviceInstallsTheSelection(t *testing.T) {
	opens := map[string]string{}
	installs := map[string]bool{}
	walkGo(t, func(p string) {
		if !strings.HasSuffix(p, "_test.go") {
			return
		}
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		dir := filepath.Dir(p)
		var code []string
		for _, l := range strings.Split(string(b), "\n") {
			if !strings.HasPrefix(strings.TrimSpace(l), "//") {
				code = append(code, l)
			}
		}
		src := strings.Join(code, "\n")
		if strings.Contains(src, "testdevices.Install(") {
			installs[dir] = true
		}
		if opensADevice().MatchString(src) {
			opens[dir] = p
		}
	})
	if len(installs) == 0 {
		t.Fatal("no package installs the selection: the guard looked at nothing")
	}
	for dir, first := range opens {
		if !installs[dir] {
			t.Errorf("%s opens a device (%s) and no test file there calls testdevices.Install: "+
				"add a TestMain, as engine/model/devskip_test.go has", dir, first)
		}
	}
}
