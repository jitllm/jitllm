package meta_test

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

const modPath = "github.com/jitllm/jitllm"

// pkgGraph maps a package's import path to the jitllm packages it imports
// directly, from non-test files only.
func pkgGraph(t *testing.T, root string) map[string][]string {
	t.Helper()
	g := map[string][]string{}
	err := filepath.Walk(root, func(p string, fi os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if fi.IsDir() {
			// A dot-directory (.git, a local tool's worktrees: full copies of
			// this repo at other commits) is not this tree's source. go build
			// excludes nested modules; a filesystem walk must skip them itself.
			if p != root && strings.HasPrefix(fi.Name(), ".") {
				return filepath.SkipDir
			}
			switch fi.Name() {
			case "models", "testdata", "docs", "scripts":
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			return nil
		}
		rel, _ := filepath.Rel(root, filepath.Dir(p))
		self := modPath
		if rel != "." {
			self += "/" + filepath.ToSlash(rel)
		}
		f, perr := parser.ParseFile(token.NewFileSet(), p, nil, parser.ImportsOnly)
		if perr != nil {
			return perr
		}
		for _, im := range f.Imports {
			ip := strings.Trim(im.Path.Value, `"`)
			if strings.HasPrefix(ip, modPath) && ip != self {
				g[self] = append(g[self], ip)
			}
		}
		if _, ok := g[self]; !ok {
			g[self] = nil
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return g
}

// reaches returns the path from -> ... -> to, or nil.
func reaches(g map[string][]string, from, to string) []string {
	type node struct {
		p    string
		path []string
	}
	seen := map[string]bool{from: true}
	q := []node{{from, []string{from}}}
	for len(q) > 0 {
		n := q[0]
		q = q[1:]
		for _, d := range g[n.p] {
			if d == to {
				return append(append([]string{}, n.path...), d)
			}
			if !seen[d] {
				seen[d] = true
				q = append(q, node{d, append(append([]string{}, n.path...), d)})
			}
		}
	}
	return nil
}

// TestOnlyTheConverterReachesGGUF is the rule "GGUF is the converter's input"
// made structural instead of conventional. It is an import-graph assertion
// because a coupling like this changes no behaviour, so no oracle can see it,
// and it is transitive so a decoupling cannot come back through an
// intermediate package. The failure prints the path it found.
//
// The allowed list is short on purpose. Adding to it is a decision.
func TestOnlyTheConverterReachesGGUF(t *testing.T) {
	testmodels.SourceTree(t)
	const gguf = modPath + "/convert/gguf"
	// The desktop app is a converter front-end: ui/screen's Convert tab calls
	// the converter in process, and the other ui entries reach gguf through
	// it (or ui/catalog's convert.DestFor); ui/mock and ui/stage stand in for
	// and render those screens. They are listed by exact name so
	// the stale-exemption check below catches any that stop reaching gguf.
	// None of the engine packages (jlm, model, tok, nn) is excused.
	const uiConverts = "the app's Convert tab runs the converter in process (ui/screen/models_convert.go)"
	// The front ends' shared logic converts too: common/convertjob runs the
	// converter in process for both the desktop app and the terminal app, and
	// the other common packages and tui reach gguf through it.
	const frontConverts = "the front ends' convert job runs the converter in process (common/convertjob)"
	// The daemon converts too (ModelService's Convert RPC), so server/ and
	// jitllmd reach gguf through the converter.
	const daemonConverts = "ModelService.Convert streams a conversion, so the daemon runs the converter"
	allowed := map[string]string{
		modPath + "/convert":      "the converter: GGUF in, .jlm out, and the only package that knows two formats",
		modPath + "/cmd/jitllm":   "`jitllm info` and `jitllm tokenize` inspect a GGUF; neither runs a model",
		modPath + "/ui":           uiConverts,
		modPath + "/ui/app":       uiConverts,
		modPath + "/ui/engine":    uiConverts,
		modPath + "/ui/screen":    uiConverts,
		modPath + "/ui/cmd/shots": uiConverts,
		modPath + "/ui/mock":      uiConverts,
		// The packager draws the window's icon from ui/app, which links the
		// app whole; it converts nothing itself.
		modPath + "/ui/cmd/pack":       uiConverts,
		modPath + "/ui/stage":          uiConverts,
		modPath + "/common/catalog":    frontConverts,
		modPath + "/common/config":     frontConverts,
		modPath + "/common/convertjob": frontConverts,
		modPath + "/common/discover":   frontConverts,
		modPath + "/common/engine":     frontConverts,
		modPath + "/tui":               frontConverts,
		modPath + "/server":            daemonConverts,
		// The front ends' API serves server.Engine, which carries the daemon's
		// Convert RPC with it.
		modPath + "/common/api":         daemonConverts,
		modPath + "/server/cmd/jitllmd": daemonConverts,
		// docs/models.md is rendered from the converter's lists themselves
		// (convert.Architectures, HFClasses, Projectors); it converts nothing.
		modPath + "/internal/modelsdoc":         "renders docs/models.md from the converter's own lists; converts nothing",
		modPath + "/internal/capabilities":      "renders the capability manifest from the converter's own lists; converts nothing",
		modPath + "/server/cmd/gencapabilities": "writes the capability manifest through internal/capabilities; converts nothing",
		// Tests only (TestNoProductionCodeImportsTestModels covers it): the
		// suite converts, so a container an earlier build left is
		// reconverted from the GGUF beside it.
		modPath + "/internal/testmodels/fresh": "test helper: reconverts a container this build cannot read from its GGUF; tests only",
	}
	g := pkgGraph(t, filepath.Join("..", ".."))
	if len(g) < 10 {
		t.Fatalf("the walk found %d packages -- this gate proved nothing", len(g))
	}
	var pkgs []string
	for p := range g {
		pkgs = append(pkgs, p)
	}
	sort.Strings(pkgs)

	found, offenders := map[string]bool{}, 0
	for _, p := range pkgs {
		if p == gguf {
			continue
		}
		path := reaches(g, p, gguf)
		if path == nil {
			continue
		}
		why, ok := allowed[p]
		if !ok {
			t.Errorf("%s reaches gguf and must not:\n  %s\n"+
				"  GGUF is the converter's input. If this package genuinely converts or\n"+
				"  inspects one, add it to the allowed list here and say why.",
				strings.TrimPrefix(p, modPath+"/"), strings.Join(path, "\n    -> "))
			offenders++
			continue
		}
		found[p] = true
		t.Logf("%-18s reaches gguf -- %s", strings.TrimPrefix(p, modPath+"/"), why)
	}
	// Checked in both directions: an entry that no longer reaches gguf is a
	// stale exemption.
	for p := range allowed {
		if !found[p] {
			t.Errorf("%s is exempt and does not reach gguf -- drop the exemption",
				strings.TrimPrefix(p, modPath+"/"))
		}
	}
	if len(found) == 0 && offenders == 0 {
		t.Fatal("nothing reaches gguf at all -- the walk found nothing, so this gate proved nothing")
	}
}
