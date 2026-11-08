// Command modelfree runs every test of every module on whatever host CI
// gives it, with no model download and no GPU, and writes what was SKIPPED into
// the job summary. .github/scripts/test-model-free.sh runs it; it is Go rather
// than shell so that one program runs on Linux, macOS and Windows alike.
//
//	go run ./scripts/modelfree
//
// A skip is not a pass. This does not hide skips; it lists each one with the
// reason the test printed, so a reader of a green run can see what it did not
// prove.
//
// JITLLM_MODEL_FREE=1 is the one switch. A test whose model, fixture or
// reference is not on the box reports it through testmodels.Missing, which
// FAILS the test anywhere else (RULE 11: a missing artefact is a task) and
// skips it here with the same message. A device arm skips on a host with no
// GPU whatever the switch says.
//
// The model directory is a fresh one holding only what the repository commits:
// testdata/models/stories260K.gguf and the container this build converts from
// it, so every gate that runs on stories260K runs here, on the CPU. A
// JITLLM_MODELS already in the environment is replaced, not read.
//
// Left out: ./dev/bench, the measurements with throughput floors (RULE 4 puts
// perf floors there), since a shared runner is not a quiet box; ./ui/stage,
// which CI's ui job runs beside the app's build; and ./jit/gpu/backend on
// macOS only (see notRun).
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strings"
)

// notRun names each package left out of the run, and why; the summary repeats
// it, so a reader sees what was never attempted beside what skipped.
func notRun() map[string]string {
	m := map[string]string{
		"github.com/jitllm/jitllm/dev/bench": "perf floors, which want a quiet box: RULE 4",
		"github.com/jitllm/jitllm/ui/stage":  "run by CI's ui job, with the app's build",
	}
	// The macOS runner exposes Apple's paravirtual Metal device, whose
	// pipelines allow 512 threads where the backend's gates force 1024-thread
	// splits (argmaxrows, matvec at split 32 and 64), so Compile refuses them
	// and the gates fail. The Linux and Windows runners, which have no GPU,
	// still run the package.
	if runtime.GOOS == "darwin" {
		m["github.com/jitllm/jitllm/jit/gpu/backend"] = "the runner's paravirtual Metal device allows 512-thread pipelines; the gates force 1024"
	}
	return m
}

func main() {
	if !modelFree() {
		os.Exit(1)
	}
}

// modelFree runs it all and reports whether everything that ran passed; a
// return rather than an exit, so the temporary directories are removed.
func modelFree() bool {
	root, err := moduleRoot()
	if err != nil {
		fatal(err)
	}
	models, err := os.MkdirTemp("", "jitllm-models-")
	if err != nil {
		fatal(err)
	}
	defer os.RemoveAll(models)
	out, err := os.MkdirTemp("", "jitllm-tests-")
	if err != nil {
		fatal(err)
	}
	defer os.RemoveAll(out)
	os.Setenv("JITLLM_MODEL_FREE", "1")
	os.Setenv("JITLLM_MODELS", models)

	failed := false
	group("convert stories260K")
	gguf := filepath.Join(models, "stories260K.gguf")
	if err := copyFile(filepath.Join(root, "testdata", "models", "stories260K.gguf"), gguf); err != nil {
		fatal(err)
	}
	if err := run(root, os.Stdout, "go", "run", "./cmd/jitllm", "convert", gguf, filepath.Join(models, "stories260K.jlm")); err != nil {
		fmt.Println("::error::converting the committed stories260K failed")
		failed = true
	}
	endGroup()

	skip := notRun()
	pkgs, err := list(root)
	if err != nil {
		fatal(err)
	}
	var rootPkgs []string
	for _, p := range pkgs {
		if _, ok := skip[p]; !ok {
			rootPkgs = append(rootPkgs, p)
		}
	}
	type testRun struct {
		name, dir string
		args      []string
	}
	runs := []testRun{{"root", root, rootPkgs}}
	// The jitllmtest tag builds the comparison implementations and the tier
	// forcing hooks, so the SSE tier's kernels run on an AVX2 runner. amd64
	// only: the tier does not exist elsewhere.
	if runtime.GOARCH == "amd64" {
		runs = append(runs, testRun{"sse-tier", root, []string{"-tags", "jitllmtest", "./jit/cpu", "./engine/nn"}})
	}
	runs = append(runs, testRun{"server", filepath.Join(root, "server"), []string{"./..."}})
	runs = append(runs, testRun{"common", filepath.Join(root, "common"), []string{"./..."}})
	runs = append(runs, testRun{"tui", filepath.Join(root, "tui"), []string{"./..."}})
	runs = append(runs, testRun{"jivman", filepath.Join(root, "examples", "jivman"), []string{"./..."}})
	uiPkgs, err := list(filepath.Join(root, "ui"))
	if err != nil {
		fatal(err)
	}
	uiPkgs = slices.DeleteFunc(uiPkgs, func(p string) bool { _, ok := skip[p]; return ok })
	runs = append(runs, testRun{"ui", filepath.Join(root, "ui"), uiPkgs})

	var results []result
	for _, r := range runs {
		res, ok := goTest(r.name, r.dir, out, r.args)
		results = append(results, res)
		if !ok {
			fmt.Printf("::error::go test (%s) failed\n", r.name)
			failed = true
		}
	}
	if err := summarize(results, skip); err != nil {
		fmt.Printf("::error::writing the summary: %v\n", err)
		failed = true
	}
	return !failed
}

// event is one line of `go test -json`.
type event struct {
	Action, Package, Test, Output string
}

type row struct{ pkg, test, why string }

type result struct {
	name                  string
	pass, fail, skip      int
	failed, skipped       []row
	buildFailed, exitFail bool
}

// goTest runs one `go test -json`, printing the package lines and the whole
// output of every failed test as it goes, and counts the top-level tests.
func goTest(name, dir, out string, args []string) (result, bool) {
	group("go test (" + name + ")")
	defer endGroup()
	res := result{name: name}
	log, err := os.Create(filepath.Join(out, name+".json"))
	if err != nil {
		fatal(err)
	}
	defer log.Close()

	cmd := exec.Command("go", append([]string{"test", "-count=1", "-timeout", "45m", "-json"}, args...)...)
	cmd.Dir = dir
	cmd.Stderr = os.Stdout
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		fatal(err)
	}
	if err := cmd.Start(); err != nil {
		fatal(err)
	}
	type key struct{ pkg, test string }
	buf := map[key][]string{}
	sc := bufio.NewScanner(io.TeeReader(stdout, log))
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	for sc.Scan() {
		var e event
		if json.Unmarshal(sc.Bytes(), &e) != nil {
			fmt.Println(sc.Text()) // build errors arrive as plain text
			continue
		}
		k := key{e.Package, e.Test}
		switch {
		case e.Action == "build-output":
			fmt.Print(e.Output)
		case e.Action == "build-fail":
			res.buildFailed = true
		case e.Action == "output" && e.Test != "":
			buf[k] = append(buf[k], e.Output)
		case e.Action == "output":
			if strings.TrimSpace(e.Output) != "PASS" {
				fmt.Print(e.Output)
			}
		case e.Test == "":
		case e.Action == "fail" || e.Action == "skip" || e.Action == "pass":
			if e.Action == "fail" {
				fmt.Print(strings.Join(buf[k], ""))
			}
			if !strings.Contains(e.Test, "/") {
				r := row{strings.TrimPrefix(e.Package, "github.com/jitllm/jitllm"), e.Test, reason(buf[k])}
				if r.pkg == "" {
					r.pkg = "/"
				}
				switch e.Action {
				case "pass":
					res.pass++
				case "fail":
					res.fail++
					res.failed = append(res.failed, r)
				case "skip":
					res.skip++
					res.skipped = append(res.skipped, r)
				}
			}
			delete(buf, k)
		}
	}
	if err := sc.Err(); err != nil {
		fmt.Printf("::error::reading go test (%s): %v\n", name, err)
	}
	res.exitFail = cmd.Wait() != nil
	return res, !res.exitFail && !res.buildFailed && sc.Err() == nil
}

var space = regexp.MustCompile(`\s+`)

// reason is the last line a test printed that is not the harness's own.
func reason(lines []string) string {
	for i := len(lines) - 1; i >= 0; i-- {
		l := strings.TrimSpace(lines[i])
		if l == "" || strings.HasPrefix(l, "=== ") || strings.HasPrefix(l, "--- ") {
			continue
		}
		l = space.ReplaceAllString(l, " ")
		if len(l) > 220 {
			l = l[:220]
		}
		return l
	}
	return "(no reason printed)"
}

// summarize writes the counts per run, then every failure and skip with its
// reason, into the job summary (stdout outside Actions).
func summarize(results []result, skip map[string]string) error {
	w := io.Writer(os.Stdout)
	if p := os.Getenv("GITHUB_STEP_SUMMARY"); p != "" {
		f, err := os.OpenFile(p, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if err != nil {
			return err
		}
		defer f.Close()
		w = f
	}
	var pass, fail, skipped int
	var failed, skips []row
	for _, r := range results {
		pass += r.pass
		fail += r.fail
		skipped += r.skip
		failed = append(failed, r.failed...)
		skips = append(skips, r.skipped...)
	}
	fmt.Fprintf(w, "## Model-free tests, %s/%s\n\n", runtime.GOOS, runtime.GOARCH)
	fmt.Fprintf(w, "%d passed, %d failed, **%d skipped** (top-level tests)\n\n", pass, fail, skipped)
	fmt.Fprint(w, "| run | passed | failed | skipped |\n|---|---|---|---|\n")
	for _, r := range results {
		note := ""
		if r.buildFailed {
			note = " (a package did not build)"
		}
		fmt.Fprintf(w, "| %s%s | %d | %d | %d |\n", r.name, note, r.pass, r.fail, r.skip)
	}
	var left []string
	for p, why := range skip {
		left = append(left, fmt.Sprintf("`%s` (%s)", strings.TrimPrefix(p, "github.com/jitllm/jitllm/"), why))
	}
	slices.Sort(left)
	fmt.Fprintf(w, "\nNot run at all: %s. The only model here is the committed stories260K; a test that needs "+
		"any other skips with MODEL MISSING. No GPU backend is exercised on these runners.\n\n", strings.Join(left, ", "))
	for _, s := range []struct {
		title string
		rows  []row
	}{{"Failed", failed}, {"Skipped -- these proved nothing on this run", skips}} {
		if len(s.rows) == 0 {
			continue
		}
		fmt.Fprintf(w, "<details><summary>%s: %d</summary>\n\n| package | test | reason |\n|---|---|---|\n", s.title, len(s.rows))
		for _, r := range s.rows {
			fmt.Fprintf(w, "| `%s` | `%s` | %s |\n", r.pkg, r.test, strings.ReplaceAll(r.why, "|", "/"))
		}
		fmt.Fprint(w, "\n</details>\n\n")
	}
	return nil
}

// moduleRoot is the engine module's directory, wherever this was started from
// inside it.
func moduleRoot() (string, error) {
	b, err := exec.Command("go", "env", "GOMOD").Output()
	if err != nil {
		return "", fmt.Errorf("go env GOMOD: %w", err)
	}
	mod := strings.TrimSpace(string(b))
	if mod == "" || mod == os.DevNull {
		return "", fmt.Errorf("not inside a module")
	}
	dir := filepath.Dir(mod)
	// From inside server/, common/, ui/ or tui/, GOMOD is the nested module's.
	if base := filepath.Base(dir); base == "server" || base == "common" || base == "ui" || base == "tui" {
		dir = filepath.Dir(dir)
	}
	if _, err := os.Stat(filepath.Join(dir, "testdata", "models", "stories260K.gguf")); err != nil {
		return "", fmt.Errorf("%s is not the jitllm repository: %w", dir, err)
	}
	return dir, nil
}

// list is every package of the module in dir.
func list(dir string) ([]string, error) {
	cmd := exec.Command("go", "list", "./...")
	cmd.Dir = dir
	cmd.Stderr = os.Stderr
	b, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("go list in %s: %w", dir, err)
	}
	return strings.Fields(string(b)), nil
}

func run(dir string, out io.Writer, name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	cmd.Stdout = out
	cmd.Stderr = out
	return cmd.Run()
}

func copyFile(from, to string) error {
	b, err := os.ReadFile(from)
	if err != nil {
		return err
	}
	return os.WriteFile(to, b, 0o644)
}

func group(name string) { fmt.Printf("::group::%s\n", name) }
func endGroup()         { fmt.Println("::endgroup::") }

func fatal(err error) {
	fmt.Printf("::error::%v\n", err)
	os.Exit(1)
}
