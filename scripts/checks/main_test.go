package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func repoRoot(t *testing.T) string {
	t.Helper()
	root, err := moduleRoot()
	if err != nil {
		t.Skip("LOUD SKIP -- NOT A PASS: this reads the source tree, and there is none above this binary")
	}
	return root
}

// trackedFiles is every file git tracks.
func trackedFiles(t *testing.T, root string) []string {
	t.Helper()
	files, err := git(root, "ls-files")
	if err != nil {
		t.Skipf("LOUD SKIP -- NOT A PASS: no git to list the tree: %v", err)
	}
	if len(files) < 100 {
		t.Fatalf("git lists %d files: not the repository", len(files))
	}
	return files
}

// uncovered is the coverage gate's body.
func uncovered(m Map, files []string) []string {
	var out []string
	for _, f := range files {
		hit := false
		for _, e := range m.Entries {
			hit = hit || matches(e.Prefix, f)
		}
		if !hit {
			out = append(out, f)
		}
	}
	return out
}

// testsIn is every Test function in a package directory's _test.go files;
// a pattern ending in /... takes the tree under it.
func testsIn(t *testing.T, dir string, recursive bool) []string {
	t.Helper()
	re := regexp.MustCompile(`(?m)^func (Test\w+)\(`)
	var names []string
	walk := func(p string, d os.DirEntry) error {
		if d.IsDir() || !strings.HasSuffix(p, "_test.go") {
			return nil
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		for _, m := range re.FindAllSubmatch(b, -1) {
			names = append(names, string(m[1]))
		}
		return nil
	}
	if !recursive {
		ents, err := os.ReadDir(dir)
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range ents {
			if err := walk(filepath.Join(dir, e.Name()), e); err != nil {
				t.Fatal(err)
			}
		}
		return names
	}
	err := filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		return walk(p, d)
	})
	if err != nil {
		t.Fatal(err)
	}
	return names
}

// badChecks is the map's own consistency: every entry names defined checks
// and an evidence file that exists, every check's packages exist, and every
// alternative of every -run pattern selects a test in its packages.
func badChecks(t *testing.T, m Map, root string) []string {
	var bad []string
	used := map[string]bool{}
	for _, e := range m.Entries {
		if e.Prefix == "" {
			bad = append(bad, "an entry has an empty prefix, which would cover everything and prove nothing")
		}
		if len(e.Checks) == 0 && e.Manual == "" {
			bad = append(bad, e.Prefix+": no check and no manual step")
		}
		for _, c := range e.Checks {
			used[c] = true
			if _, ok := m.Checks[c]; !ok {
				bad = append(bad, e.Prefix+": check "+c+" is not defined")
			}
		}
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(e.Evidence))); err != nil {
			bad = append(bad, e.Prefix+": evidence "+e.Evidence+" does not exist")
		}
	}
	for name, c := range m.Checks {
		if !used[name] {
			bad = append(bad, "check "+name+" is used by no entry")
		}
		if len(c.Command) > 0 {
			continue
		}
		if len(c.Packages) == 0 {
			bad = append(bad, "check "+name+" names no package")
			continue
		}
		var tests []string
		for _, p := range c.Packages {
			rec := strings.HasSuffix(p, "/...")
			dir := filepath.Join(root, filepath.FromSlash(c.Module), filepath.FromSlash(strings.TrimSuffix(p, "/...")))
			if st, err := os.Stat(dir); err != nil || !st.IsDir() {
				bad = append(bad, "check "+name+": package "+p+" is not a directory")
				continue
			}
			tests = append(tests, testsIn(t, dir, rec)...)
		}
		if c.Run == "" {
			if len(tests) == 0 {
				bad = append(bad, "check "+name+": its packages hold no test")
			}
			continue
		}
		for _, alt := range strings.Split(c.Run, "|") {
			re, err := regexp.Compile(alt)
			if err != nil {
				bad = append(bad, "check "+name+": -run alternative "+alt+": "+err.Error())
				continue
			}
			hit := false
			for _, tn := range tests {
				hit = hit || re.MatchString(tn)
			}
			if !hit {
				bad = append(bad, "check "+name+": -run alternative "+alt+" selects no test in its packages")
			}
		}
	}
	return bad
}

// TestMapCoversTheTree: every tracked file is covered by some entry, and the
// map names only checks, packages, gates and evidence that exist.
func TestMapCoversTheTree(t *testing.T) {
	root := repoRoot(t)
	m, err := loadMap()
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range uncovered(m, trackedFiles(t, root)) {
		t.Errorf("%s is covered by no entry in scripts/checks/map.json", f)
	}
	for _, b := range badChecks(t, m, root) {
		t.Error(b)
	}
}

// TestMapGateFires runs the gate above against violations.
func TestMapGateFires(t *testing.T) {
	root := repoRoot(t)
	files := trackedFiles(t, root)
	m, err := loadMap()
	if err != nil {
		t.Fatal(err)
	}
	// A top-level path no entry names.
	if len(uncovered(m, append(files, "newsubsystem/x.go"))) != len(uncovered(m, files))+1 {
		t.Error("a new top-level path passed the coverage gate")
	}
	breakers := map[string]func(*Map){
		"renamed gate": func(m *Map) {
			c := m.Checks["five-principles"]
			c.Run += "|TestNoSuchGateAnywhere"
			m.Checks["five-principles"] = c
		},
		"missing package": func(m *Map) {
			c := m.Checks["sched"]
			c.Packages = []string{"./engine/nowhere"}
			m.Checks["sched"] = c
		},
		"undefined check":  func(m *Map) { m.Entries[0].Checks = append(m.Entries[0].Checks, "nope") },
		"missing evidence": func(m *Map) { m.Entries[0].Evidence = "docs/nowhere.md" },
		"empty prefix":     func(m *Map) { m.Entries[0].Prefix = "" },
	}
	for name, breakIt := range breakers {
		m, err := loadMap()
		if err != nil {
			t.Fatal(err)
		}
		breakIt(&m)
		if len(badChecks(t, m, root)) == 0 {
			t.Errorf("%s: the gate passed a broken map", name)
		}
	}
}

// TestPlanTakesTheUnion: a path under two prefixes gets both entries'
// checks, and an unknown path is reported, not dropped.
func TestPlanTakesTheUnion(t *testing.T) {
	m, err := loadMap()
	if err != nil {
		t.Fatal(err)
	}
	p, err := plan(m, []string{"engine/paging.go", "model/model.go", "nowhere/x.go"})
	if err != nil {
		t.Fatal(err)
	}
	have := map[string]bool{}
	for _, c := range p.Checks {
		have[c.Name] = true
		if c.Check.Command == nil && (c.Argv[0] != "./scripts/cap" || c.Argv[3] != "taskset") {
			t.Errorf("%s does not run under scripts/cap on the P-cores: %v", c.Name, c.Argv)
		}
	}
	if !have["five-principles"] || !have["model-correctness"] {
		t.Errorf("engine/* and model/ plan %v", have)
	}
	// engine/* is the engine package's own files: a subpackage's path does
	// not take its checks.
	sub, err := plan(m, []string{"sched/x.go"})
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range sub.Checks {
		if c.Name == "composed" || c.Name == "relocation" {
			t.Errorf("sched/x.go takes the engine package's %s check", c.Name)
		}
	}
	if len(p.Unmapped) != 1 || p.Unmapped[0] != "nowhere/x.go" {
		t.Errorf("unmapped %v", p.Unmapped)
	}
}

// TestSkipIsNeverAPass feeds summarize recorded go test -json streams: a skip
// for a model or a device, alone or beside passes, is reported as that skip;
// a pattern that ran nothing is no-tests; a build failure is a failure.
func TestSkipIsNeverAPass(t *testing.T) {
	ev := func(action, test, output string) string {
		return `{"Action":"` + action + `","Package":"p","Test":"` + test + `","Output":` + quoteJSON(output) + "}\n"
	}
	cases := []struct {
		name   string
		stream string
		want   Status
	}{
		{"all pass", ev("run", "TestA", "") + ev("pass", "TestA", ""), Pass},
		{"model skip beside a pass", ev("pass", "TestA", "") +
			ev("output", "TestB", "    x_test.go:9: MODEL MISSING: /m/x.gguf (set JITLLM_MODELS to the model directory)\n") +
			ev("skip", "TestB", ""), SkippedModel},
		{"device skip", ev("output", "TestC", "    x_test.go:9: no GPU backend on this host\n") + ev("skip", "TestC", ""), SkippedDevice},
		{"other skip", ev("output", "TestD", "    x_test.go:9: objdump not installed\n") + ev("skip", "TestD", ""), SkippedOther},
		{"nothing ran", `{"Action":"output","Package":"p","Output":"testing: warning: no tests to run\n"}` + "\n" +
			`{"Action":"pass","Package":"p"}` + "\n", NoTests},
		{"a failure", ev("pass", "TestA", "") + ev("fail", "TestE", ""), Fail},
		{"a build failure", `{"Action":"fail","Package":"p"}` + "\n", Fail},
	}
	for _, c := range cases {
		r := summarize(c.name, strings.NewReader(c.stream), nil)
		if r.Status != c.want {
			t.Errorf("%s: %s, want %s (%+v)", c.name, r.Status, c.want, r)
		}
	}
	// A command that failed with no event at all is a failure, not no-tests.
	if r := summarize("exit", strings.NewReader(""), os.ErrNotExist); r.Status != Fail {
		t.Errorf("a failed command with no events reads %s", r.Status)
	}
}

func quoteJSON(s string) string {
	b := strings.Builder{}
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
	return b.String()
}
