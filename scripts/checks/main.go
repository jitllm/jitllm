// Command checks says which checks a change needs, and runs them.
//
//	go run ./scripts/checks <path>...          the checks for these paths
//	go run ./scripts/checks HEAD~3..HEAD       the checks for a git diff range
//	go run ./scripts/checks                    the checks for the working tree's changes
//	go run ./scripts/checks -run <...>         run them, each under scripts/cap
//	go run ./scripts/checks -json <...>        the plan, or the results, as JSON
//
// The map is map.json beside this file: each subsystem, by path prefix, names
// the checks it needs (go test packages with -run patterns, or a command), the
// fixtures and devices each needs, and the evidence file AGENTS.md's routing
// table sends a reader to. TestMapCoversTheTree holds every tracked file to
// some entry and every named gate to a test that exists.
//
// With -run each check reports one of:
//
//	pass                      everything it selected ran and passed
//	fail                      a test failed, or the package did not build
//	skipped-missing-model     tests skipped for a model, fixture or reference not on this box
//	skipped-missing-device    tests skipped for a GPU this box does not have
//	skipped-other             tests skipped for another stated reason
//	no-tests                  the pattern selected nothing: a gate that proved nothing
//
// A check with both passes and skips reports the skip: a skip is not a pass
// (RULE 10), so the exit code is 0 only when every check passed.
package main

import (
	"bufio"
	"bytes"
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

//go:embed map.json
var mapJSON []byte

// Map is map.json.
type Map struct {
	Checks  map[string]Check `json:"checks"`
	Entries []Entry          `json:"entries"`
}

// Check is one thing to run: go test over packages in a module, or a command
// from the repository root.
type Check struct {
	Module   string   `json:"module,omitempty"`
	Packages []string `json:"packages,omitempty"`
	Run      string   `json:"run,omitempty"`
	Cap      string   `json:"cap,omitempty"`
	Command  []string `json:"command,omitempty"`
	Needs    Needs    `json:"needs"`
	Note     string   `json:"note,omitempty"`
}

// Needs is what a check needs from the box beyond the source tree.
type Needs struct {
	// Models: GGUFs, containers or references under JITLLM_MODELS.
	Models bool `json:"models,omitempty"`
	// Device: a GPU any backend opens.
	Device bool `json:"device,omitempty"`
}

// Entry maps a path prefix to its checks.
type Entry struct {
	Prefix   string   `json:"prefix"`
	Evidence string   `json:"evidence"`
	Checks   []string `json:"checks"`
	// Manual is what no automated check covers, said in words.
	Manual string `json:"manual,omitempty"`
}

func loadMap() (Map, error) {
	var m Map
	d := json.NewDecoder(bytes.NewReader(mapJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&m); err != nil {
		return m, fmt.Errorf("checks: map.json: %w", err)
	}
	return m, nil
}

// Plan is the checks a set of paths needs.
type Plan struct {
	Paths []string `json:"paths"`
	// Unmapped are paths no entry covers; TestMapCoversTheTree keeps it empty
	// for tracked files, so a new top-level path shows here first.
	Unmapped []string    `json:"unmapped"`
	Checks   []PlanCheck `json:"checks"`
	Manual   []string    `json:"manual"`
	Evidence []string    `json:"evidence"`
}

// PlanCheck is one check, with the command that runs it.
type PlanCheck struct {
	Name    string   `json:"name"`
	Check   Check    `json:"check"`
	Argv    []string `json:"argv"`
	Dir     string   `json:"dir"`
	Because []string `json:"because"`
}

// matches reports whether an entry's prefix covers path.
func matches(prefix, path string) bool {
	return prefix != "" && strings.HasPrefix(path, prefix)
}

// plan resolves paths to checks: the union over every entry whose prefix
// covers each path.
func plan(m Map, paths []string) (Plan, error) {
	p := Plan{Paths: paths, Unmapped: []string{}, Checks: []PlanCheck{}, Manual: []string{}, Evidence: []string{}}
	because := map[string][]string{}
	manual, evidence := map[string]bool{}, map[string]bool{}
	for _, path := range paths {
		hit := false
		for _, e := range m.Entries {
			if !matches(e.Prefix, path) {
				continue
			}
			hit = true
			for _, c := range e.Checks {
				if _, ok := m.Checks[c]; !ok {
					return p, fmt.Errorf("checks: entry %s names check %q, which map.json does not define", e.Prefix, c)
				}
				if !contains(because[c], e.Prefix) {
					because[c] = append(because[c], e.Prefix)
				}
			}
			if e.Manual != "" && !manual[e.Prefix+": "+e.Manual] {
				manual[e.Prefix+": "+e.Manual] = true
				p.Manual = append(p.Manual, e.Prefix+": "+e.Manual)
			}
			if !evidence[e.Evidence] {
				evidence[e.Evidence] = true
				p.Evidence = append(p.Evidence, e.Evidence)
			}
		}
		if !hit {
			p.Unmapped = append(p.Unmapped, path)
		}
	}
	names := make([]string, 0, len(because))
	for n := range because {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		c := m.Checks[n]
		dir, argv := commandOf(c)
		p.Checks = append(p.Checks, PlanCheck{Name: n, Check: c, Argv: argv, Dir: dir, Because: because[n]})
	}
	return p, nil
}

func contains(ss []string, s string) bool {
	for _, x := range ss {
		if x == s {
			return true
		}
	}
	return false
}

// commandOf is the command line a check runs as, and the directory (from the
// repository root) it runs in. A go test runs under scripts/cap on the six
// P-cores, as RULE 3 and RULE 5 require.
func commandOf(c Check) (string, []string) {
	if len(c.Command) > 0 {
		return ".", c.Command
	}
	capScript := "./scripts/cap"
	if c.Module != "." && c.Module != "" {
		capScript = "../scripts/cap"
	}
	mem := c.Cap
	if mem == "" {
		mem = "8G"
	}
	argv := []string{capScript, mem, "--", "taskset", "-c", "0,2,4,6,8,10", "go", "test"}
	argv = append(argv, c.Packages...)
	argv = append(argv, "-count=1")
	if c.Run != "" {
		argv = append(argv, "-run", c.Run)
	}
	dir := c.Module
	if dir == "" {
		dir = "."
	}
	return dir, argv
}

// quote renders argv as a shell line.
func quote(argv []string) string {
	out := make([]string, len(argv))
	for i, a := range argv {
		if a == "" || strings.ContainsAny(a, " |'\"$*?()") {
			out[i] = "'" + strings.ReplaceAll(a, "'", `'\''`) + "'"
		} else {
			out[i] = a
		}
	}
	return strings.Join(out, " ")
}

// changedPaths turns the arguments into paths: a git range (A..B or A...B)
// becomes its diff, nothing becomes the working tree's changes against HEAD
// plus untracked files, anything else is a path.
func changedPaths(root string, args []string) ([]string, error) {
	if len(args) == 0 {
		out, err := git(root, "diff", "--name-only", "HEAD")
		if err != nil {
			return nil, err
		}
		untracked, err := git(root, "ls-files", "--others", "--exclude-standard")
		if err != nil {
			return nil, err
		}
		return dedupe(append(out, untracked...)), nil
	}
	var paths []string
	for _, a := range args {
		if strings.Contains(a, "..") && !strings.HasPrefix(a, ".") {
			out, err := git(root, "diff", "--name-only", a)
			if err != nil {
				return nil, err
			}
			paths = append(paths, out...)
			continue
		}
		paths = append(paths, filepath.ToSlash(filepath.Clean(a)))
	}
	return dedupe(paths), nil
}

func dedupe(ss []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, s := range ss {
		if s != "" && !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	sort.Strings(out)
	return out
}

func git(root string, args ...string) ([]string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	cmd.Stderr = os.Stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git %s: %w", strings.Join(args, " "), err)
	}
	var lines []string
	for _, l := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if l != "" {
			lines = append(lines, l)
		}
	}
	return lines, nil
}

// moduleRoot walks up to the go.mod that declares the engine module.
func moduleRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		b, err := os.ReadFile(filepath.Join(dir, "go.mod"))
		if err == nil && bytes.HasPrefix(b, []byte("module github.com/jitllm/jitllm\n")) {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("checks: no github.com/jitllm/jitllm go.mod above the working directory")
		}
		dir = parent
	}
}

// Status is one check's outcome.
type Status string

const (
	Pass          Status = "pass"
	Fail          Status = "fail"
	SkippedModel  Status = "skipped-missing-model"
	SkippedDevice Status = "skipped-missing-device"
	SkippedOther  Status = "skipped-other"
	NoTests       Status = "no-tests"
)

// Result is one check run.
type Result struct {
	Name    string         `json:"name"`
	Status  Status         `json:"status"`
	Counts  map[Status]int `json:"counts"`
	Skipped []string       `json:"skipped,omitempty"`
	Failed  []string       `json:"failed,omitempty"`
	Command string         `json:"command"`
}

// The skip messages the tree prints, by what they say is missing. A model
// gate reports through testmodels.Missing ("MODEL MISSING") or names its file;
// a device gate names the device.

// classify names why a test skipped from what it printed.
func classify(output string) Status {
	switch {
	case modelSkip().MatchString(output):
		return SkippedModel
	case deviceSkip().MatchString(output):
		return SkippedDevice
	}
	return SkippedOther
}

// testEvent is one line of `go test -json`.
type testEvent struct {
	Action  string `json:"Action"`
	Package string `json:"Package"`
	Test    string `json:"Test"`
	Output  string `json:"Output"`
}

// summarize reads a go test -json stream into a result. exitErr is the
// command's own failure, which a build error reports without a test event.
func summarize(name string, r io.Reader, exitErr error) Result {
	res := Result{Name: name, Counts: map[Status]int{}}
	out := map[string]*strings.Builder{}
	key := func(e testEvent) string { return e.Package + "." + e.Test }
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	pkgFailed := false
	for sc.Scan() {
		var e testEvent
		if json.Unmarshal(sc.Bytes(), &e) != nil {
			continue
		}
		if e.Test == "" {
			if e.Action == "fail" {
				pkgFailed = true
			}
			continue
		}
		k := key(e)
		switch e.Action {
		case "output":
			if out[k] == nil {
				out[k] = &strings.Builder{}
			}
			out[k].WriteString(e.Output)
		case "pass":
			// A subtest's pass is its parent's; count top-level tests only.
			if !strings.Contains(e.Test, "/") {
				res.Counts[Pass]++
			}
		case "fail":
			res.Counts[Fail]++
			res.Failed = append(res.Failed, k)
		case "skip":
			why := ""
			if out[k] != nil {
				why = out[k].String()
			}
			s := classify(why)
			res.Counts[s]++
			res.Skipped = append(res.Skipped, fmt.Sprintf("%s [%s] %s", k, s, firstSkipLine(why)))
		}
	}
	switch {
	case res.Counts[Fail] > 0 || pkgFailed || exitErr != nil && res.Counts[Pass] == 0 && len(res.Skipped) == 0:
		res.Status = Fail
	case res.Counts[SkippedModel] > 0:
		res.Status = SkippedModel
	case res.Counts[SkippedDevice] > 0:
		res.Status = SkippedDevice
	case res.Counts[SkippedOther] > 0:
		res.Status = SkippedOther
	case res.Counts[Pass] == 0:
		res.Status = NoTests
	default:
		res.Status = Pass
	}
	return res
}

// firstSkipLine is the line a skip printed its reason on.
func firstSkipLine(out string) string {
	for _, l := range strings.Split(out, "\n") {
		l = strings.TrimSpace(l)
		if l != "" && !strings.HasPrefix(l, "=== ") && !strings.HasPrefix(l, "--- ") {
			if len(l) > 160 {
				l = l[:160] + "..."
			}
			return l
		}
	}
	return ""
}

// runCheck runs one planned check from root.
func runCheck(root string, pc PlanCheck) Result {
	argv := pc.Argv
	isTest := len(pc.Check.Command) == 0
	if isTest {
		argv = append(append([]string{}, argv...), "-json")
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Dir = filepath.Join(root, pc.Dir)
	cmd.Stderr = os.Stderr
	if !isTest {
		cmd.Stdout = os.Stderr
		err := cmd.Run()
		res := Result{Name: pc.Name, Status: Pass, Counts: map[Status]int{}, Command: quote(pc.Argv)}
		if err != nil {
			res.Status = Fail
			res.Failed = []string{err.Error()}
		}
		return res
	}
	var buf bytes.Buffer
	cmd.Stdout = &buf
	err := cmd.Run()
	res := summarize(pc.Name, &buf, err)
	res.Command = quote(pc.Argv)
	if res.Status == Fail && res.Counts[Fail] == 0 && err != nil {
		res.Failed = append(res.Failed, "the command failed: "+err.Error()+" (a build error, or no test binary)")
	}
	return res
}

func main() {
	os.Exit(run(os.Args[1:], os.Stdout))
}

func run(args []string, w io.Writer) int {
	set := flag.NewFlagSet("checks", flag.ContinueOnError)
	doRun := set.Bool("run", false, "run the checks, each under scripts/cap, and report each one")
	asJSON := set.Bool("json", false, "print the plan or the results as JSON")
	if err := set.Parse(args); err != nil {
		return 2
	}
	root, err := moduleRoot()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	m, err := loadMap()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	paths, err := changedPaths(root, set.Args())
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	p, err := plan(m, paths)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	if !*doRun {
		if *asJSON {
			enc := json.NewEncoder(w)
			enc.SetIndent("", "  ")
			enc.SetEscapeHTML(false)
			if err := enc.Encode(p); err != nil {
				fmt.Fprintln(os.Stderr, err)
				return 2
			}
			return 0
		}
		printPlan(w, p)
		if len(p.Unmapped) > 0 {
			return 1
		}
		return 0
	}
	var results []Result
	for _, pc := range p.Checks {
		fmt.Fprintf(os.Stderr, "checks: %s: (cd %s && %s)\n", pc.Name, pc.Dir, quote(pc.Argv))
		results = append(results, runCheck(root, pc))
	}
	ok := len(p.Unmapped) == 0
	for _, r := range results {
		ok = ok && r.Status == Pass
	}
	if *asJSON {
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		enc.SetEscapeHTML(false)
		if err := enc.Encode(struct {
			Plan    Plan     `json:"plan"`
			Results []Result `json:"results"`
			OK      bool     `json:"ok"`
		}{p, results, ok}); err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 2
		}
	} else {
		for _, r := range results {
			fmt.Fprintf(w, "%-24s %-24s %s\n", r.Status, r.Name, countsLine(r.Counts))
			for _, f := range r.Failed {
				fmt.Fprintf(w, "    FAIL %s\n", f)
			}
			for _, s := range r.Skipped {
				fmt.Fprintf(w, "    SKIP %s\n", s)
			}
		}
		for _, u := range p.Unmapped {
			fmt.Fprintf(w, "unmapped                 %s: no entry in scripts/checks/map.json\n", u)
		}
		for _, mn := range p.Manual {
			fmt.Fprintf(w, "manual                   %s\n", mn)
		}
		if ok {
			fmt.Fprintln(w, "every check passed")
		} else {
			fmt.Fprintln(w, "INCOMPLETE: a check failed, skipped or selected nothing; a skip is not a pass")
		}
	}
	if !ok {
		return 1
	}
	return 0
}

func countsLine(c map[Status]int) string {
	var parts []string
	for _, s := range []Status{Pass, Fail, SkippedModel, SkippedDevice, SkippedOther} {
		if c[s] > 0 {
			parts = append(parts, fmt.Sprintf("%s %d", s, c[s]))
		}
	}
	return strings.Join(parts, ", ")
}

func printPlan(w io.Writer, p Plan) {
	if len(p.Paths) == 0 {
		fmt.Fprintln(w, "no changed paths")
		return
	}
	for _, pc := range p.Checks {
		var needs []string
		if pc.Check.Needs.Models {
			needs = append(needs, "models in JITLLM_MODELS")
		}
		if pc.Check.Needs.Device {
			needs = append(needs, "a GPU")
		}
		fmt.Fprintf(w, "%s (for %s)\n", pc.Name, strings.Join(pc.Because, ", "))
		fmt.Fprintf(w, "    (cd %s && %s)\n", pc.Dir, quote(pc.Argv))
		if len(needs) > 0 {
			fmt.Fprintf(w, "    needs %s; without them it skips, and a skip is not a pass\n", strings.Join(needs, " and "))
		}
		if pc.Check.Note != "" {
			fmt.Fprintf(w, "    %s\n", pc.Check.Note)
		}
	}
	for _, mn := range p.Manual {
		fmt.Fprintf(w, "manual: %s\n", mn)
	}
	for _, u := range p.Unmapped {
		fmt.Fprintf(w, "UNMAPPED: %s has no entry in scripts/checks/map.json\n", u)
	}
	if len(p.Evidence) > 0 {
		fmt.Fprintf(w, "read first: %s\n", strings.Join(p.Evidence, ", "))
	}
}

func modelSkip() *regexp.Regexp {
	return regexp.MustCompile(`(?i)MODEL MISSING|model not present|not present \(|no \*\.gguf|nor its container|JITLLM_LCPP|JITLLM_MODELS|golden .*missing|reference .*missing`)
}

func deviceSkip() *regexp.Regexp {
	return regexp.MustCompile(`(?i)no (GPU|device|cuda|CUDA|Metal|Vulkan|HIP|AMD)\b|NO AMD DEVICE|CARD TOO SMALL|device took no block|no text block placed on the device|no backend lowers`)
}
