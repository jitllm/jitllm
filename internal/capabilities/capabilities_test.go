package capabilities

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
)

// testFuncs is every Test function declared in the module's *_test.go files
// (the nested modules included: a gate in server/ is evidence too).
func testFuncs(t *testing.T, root string) map[string]bool {
	t.Helper()
	re := regexp.MustCompile(`(?m)^func (Test\w+)\(`)
	have := map[string]bool{}
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && (d.Name() == "node_modules" || d.Name() == ".git" || d.Name() == "website") {
			return filepath.SkipDir
		}
		if d.IsDir() || !strings.HasSuffix(p, "_test.go") {
			return nil
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		for _, m := range re.FindAllSubmatch(b, -1) {
			have[string(m[1])] = true
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return have
}

// checkMaintained is the gate's body, separate so the violation test can hand
// it a broken file.
func checkMaintained(m maintained, root string, tests map[string]bool) []string {
	var bad []string
	evidence := func(where string, names []string) {
		if len(names) == 0 {
			bad = append(bad, where+": no evidence named")
		}
		for _, n := range names {
			if !tests[n] {
				bad = append(bad, where+": evidence "+n+" is no test in the tree")
			}
		}
	}
	for _, f := range m.InputFormats {
		evidence("input format "+f.Name, f.Evidence)
	}
	seen := map[string]bool{}
	for _, b := range m.Backends {
		if seen[b.Name] {
			bad = append(bad, "backend "+b.Name+" listed twice")
		}
		seen[b.Name] = true
		if st, err := os.Stat(filepath.Join(root, filepath.FromSlash(b.Package))); err != nil || !st.IsDir() {
			bad = append(bad, "backend "+b.Name+": package "+b.Package+" is not in the tree")
		}
		switch b.Kind {
		case "device":
			if backend.APIRank(b.Name) == 0 {
				bad = append(bad, "backend "+b.Name+": not a name backend.APIRank knows")
			}
		case "host":
		default:
			bad = append(bad, "backend "+b.Name+": kind "+b.Kind+" is neither host nor device")
		}
		if !b.Implemented && len(b.Verified) > 0 {
			bad = append(bad, "backend "+b.Name+": verified but not implemented")
		}
		if b.Implemented && len(b.Verified) == 0 && b.Note == "" {
			bad = append(bad, "backend "+b.Name+": implemented and verified nowhere, with no note saying so")
		}
		for _, v := range b.Verified {
			if v.Hardware == "" {
				bad = append(bad, "backend "+b.Name+": a verification names no hardware")
			}
			evidence("backend "+b.Name+" on "+v.Hardware, v.Evidence)
		}
	}
	// Every device backend the engine opens is listed: backend.Open's order
	// is CUDA, HIP, Vulkan, Metal.
	for _, api := range []string{"cuda", "hip", "vulkan", "metal"} {
		if !seen[api] {
			bad = append(bad, "device backend "+api+" is not in maintained.json")
		}
	}
	return bad
}

// TestMaintainedEvidenceExists holds the hand-kept half to the tree: every
// gate named as evidence is a test that exists, every backend's package is a
// directory, every device backend is one the engine names, and a backend
// verified nowhere says so in words.
func TestMaintainedEvidenceExists(t *testing.T) {
	testmodels.SourceTree(t)
	m, err := loadMaintained()
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Join("..", "..")
	for _, b := range checkMaintained(m, root, testFuncs(t, root)) {
		t.Error(b)
	}
}

// TestMaintainedGateFires is the gate above run against violations: each
// broken file must be refused.
func TestMaintainedGateFires(t *testing.T) {
	testmodels.SourceTree(t)
	root := filepath.Join("..", "..")
	tests := testFuncs(t, root)
	cases := map[string]func(*maintained){
		"unknown gate":      func(m *maintained) { m.Backends[0].Verified[0].Evidence = []string{"TestNoSuchGate"} },
		"no evidence":       func(m *maintained) { m.InputFormats[0].Evidence = nil },
		"missing package":   func(m *maintained) { m.Backends[0].Package = "jit_old/nowhere" },
		"unknown device":    func(m *maintained) { m.Backends[len(m.Backends)-1].Name = "rocm2" },
		"silent unverified": func(m *maintained) { m.Backends[len(m.Backends)-1].Note = "" },
		"dropped backend":   func(m *maintained) { m.Backends = m.Backends[:len(m.Backends)-1] },
	}
	for name, breakIt := range cases {
		m, err := loadMaintained()
		if err != nil {
			t.Fatal(err)
		}
		if m.Backends[len(m.Backends)-1].Name != "hip" {
			t.Fatal("the cases assume hip is the last backend")
		}
		breakIt(&m)
		if len(checkMaintained(m, root, tests)) == 0 {
			t.Errorf("%s: the gate passed a broken file", name)
		}
	}
}

// TestManifestIsDerived checks the derived half is the code's lists, not a
// copy: the counts match the converter's and quant's own answers.
func TestManifestIsDerived(t *testing.T) {
	m, err := Doc()
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Architectures) != len(convert.Architectures()) || len(m.SafetensorsClasses) != len(convert.HFClasses()) ||
		len(m.Projectors) != len(convert.Projectors()) || len(m.Quantization.Packed) != len(quant.PackedTypes) {
		t.Fatal("the manifest's lists are not the converter's")
	}
	for _, g := range append(append([]Graph{}, m.Architectures...), m.SafetensorsClasses...) {
		if g.Graph == "" || strings.HasPrefix(g.Graph, "arch(") {
			t.Errorf("%s converts to an unnamed graph %q", g.Name, g.Graph)
		}
	}
	for _, q := range m.Quantization.Packed {
		found := false
		for _, d := range m.Quantization.Dequantable {
			found = found || d == q
		}
		if !found {
			t.Errorf("%s is packed and not dequantable", q)
		}
	}
	if len(m.API.Endpoints) == 0 {
		t.Fatalf("api.json lists no endpoint; run: %s", Command)
	}
	if m.Build != nil {
		t.Error("Doc carries a build identity; the file describes the source tree")
	}
	b, err := Build("v9.9.9")
	if err != nil {
		t.Fatal(err)
	}
	if b.Build == nil || b.Build.Version != "v9.9.9" || b.Build.Go == "" {
		t.Errorf("Build's identity is %+v", b.Build)
	}
}
