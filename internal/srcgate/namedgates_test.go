package srcgate

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestEveryNamedGateExists holds AGENTS.md to the tree: every test it names
// as a gate is a Test function a _test.go file of the module declares. A gate
// renamed, deleted or written over leaves its name behind as a claim of
// coverage nothing gives -- TestStreamRowsServeOnTheCard went with 9757b669a,
// which wrote another gate over its file, and ran nothing for a fan before
// anyone saw. A name followed by * names a prefix, and Name(Suffix) both
// Name and NameSuffix. The violation, a name no file declares, is seen.
func TestEveryNamedGateExists(t *testing.T) {
	testmodels.SourceTree(t)
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	declared, err := declaredTests(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(declared) < 500 {
		t.Fatalf("read %d Test functions under %s: the gate read nothing", len(declared), root)
	}
	text, err := os.ReadFile(filepath.Join(root, "AGENTS.md"))
	if err != nil {
		t.Fatal(err)
	}
	named := namedGates(string(text))
	if len(named) < 30 {
		t.Fatalf("AGENTS.md names %d gates: the gate read nothing", len(named))
	}
	for _, n := range named {
		if !gateDeclared(declared, n) {
			t.Errorf("AGENTS.md names %s, which no _test.go declares", n.String())
		}
	}
	if v := namedGates("the gate `TestNoSuchGateAnywhere` and `TestTwoPagedModels*`"); len(v) != 2 ||
		gateDeclared(declared, v[0]) || !gateDeclared(declared, v[1]) {
		t.Errorf("the violation: a name no file declares was not seen, or a prefix that one does was not found (%v)", v)
	}
}

// gateName is a test AGENTS.md names: a whole name, or a prefix (Name*).
type gateName struct {
	name   string
	prefix bool
}

func (n gateName) String() string {
	if n.prefix {
		return n.name + "*"
	}
	return n.name
}

// namedGates is every test name in text: TestX, TestX* (a prefix) and
// TestX(Y) (TestX and TestXY).
func namedGates(text string) []gateName {
	re := regexp.MustCompile(`\bTest[A-Z][A-Za-z0-9_]*(\*|\([A-Za-z0-9_]+\))?`)
	seen := map[gateName]bool{}
	var out []gateName
	add := func(n gateName) {
		if !seen[n] {
			seen[n] = true
			out = append(out, n)
		}
	}
	for _, m := range re.FindAllStringSubmatch(text, -1) {
		base := strings.TrimSuffix(strings.SplitN(m[0], "(", 2)[0], "*")
		switch {
		case m[1] == "*":
			add(gateName{name: base, prefix: true})
		case strings.HasPrefix(m[1], "("):
			add(gateName{name: base})
			add(gateName{name: base + strings.Trim(m[1], "()")})
		default:
			add(gateName{name: base})
		}
	}
	return out
}

// gateDeclared reports whether a Test function of the module is n.
func gateDeclared(declared map[string]bool, n gateName) bool {
	if !n.prefix {
		return declared[n.name]
	}
	for d := range declared {
		if strings.HasPrefix(d, n.name) {
			return true
		}
	}
	return false
}

// declaredTests is every Test function a _test.go file under root declares.
func declaredTests(root string) (map[string]bool, error) {
	re := regexp.MustCompile(`(?m)^func (Test[A-Za-z0-9_]*)\(`)
	out := map[string]bool{}
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			switch d.Name() {
			case ".git", "node_modules", "models", "testdata":
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, "_test.go") {
			return nil
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		for _, m := range re.FindAllSubmatch(b, -1) {
			out[string(m[1])] = true
		}
		return nil
	})
	return out, err
}
