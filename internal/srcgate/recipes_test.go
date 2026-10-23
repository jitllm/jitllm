//go:build unix

package srcgate

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/internal/testmodels/fresh"
)

// TestRecipesRun runs every command block of docs/recipes/*.md against the
// jitllm and jitllmd this tree builds, a real server included, and holds each
// block to the output and exit status the recipe says it has. The conventions
// a recipe is written to are in docs/recipes/README.md.
//
// A block that needs a model the box does not have is reported through
// testmodels.Missing (a failure, or a named skip in the model-free run); a
// Python or JavaScript block whose interpreter cannot import the openai SDK is
// a named skip. Nothing is passed by not running.
func TestRecipesRun(t *testing.T) {
	testmodels.SourceTree(t)
	for _, tool := range []string{"bash", "curl"} {
		if _, err := exec.LookPath(tool); err != nil {
			t.Fatalf("the recipes run their blocks with %s, and it is not on PATH: %v", tool, err)
		}
	}
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	files, err := filepath.Glob(filepath.Join(root, "docs", "recipes", "*.md"))
	if err != nil {
		t.Fatal(err)
	}
	var recipes []string
	for _, f := range files {
		if filepath.Base(f) != "README.md" {
			recipes = append(recipes, f)
		}
	}
	if len(recipes) == 0 {
		t.Fatal("docs/recipes has no recipe, so this gate proved nothing")
	}
	bin := buildRecipeBinaries(t, root)
	env := recipeEnv(root, bin)
	only := os.Getenv("JITLLM_RECIPE")
	ran := 0
	for _, f := range recipes {
		name := strings.TrimSuffix(filepath.Base(f), ".md")
		if only != "" && only != name {
			continue
		}
		t.Run(name, func(t *testing.T) {
			src, err := os.ReadFile(f)
			if err != nil {
				t.Fatal(err)
			}
			blocks, err := parseRecipe(string(src))
			if err != nil {
				t.Fatalf("%s: %v", f, err)
			}
			ran += runRecipe(t, blocks, env)
		})
	}
	if only == "" && ran == 0 {
		t.Fatal("no recipe block ran, so this gate proved nothing")
	}
	t.Logf("%d recipe block(s) ran", ran)
}

// TestRecipeRunnerFails holds the runner to its own violations: a block whose
// output misses its expectation, a block that exits with the wrong status,
// and a block a download stands in for that is not a download. A runner that
// passed these would pass a recipe that no longer works.
func TestRecipeRunnerFails(t *testing.T) {
	work := t.TempDir()
	env := os.Environ()
	for _, c := range []struct {
		md, want string
	}{
		{"```sh\necho hello\n```\n<!-- test: expect goodbye -->\n", `does not contain "goodbye"`},
		{"<!-- test: exit 1 -->\n```sh\ntrue\n```\n", "exited 0, want 1"},
		{"```sh\nfalse\n```\n", "exited 1, want 0"},
		{"<!-- test: instead true -->\n```sh\nrm -rf models\n```\n", "only a download"},
		{"<!-- test: frobnicate -->\n```sh\ntrue\n```\n", "unknown directive"},
		{"```sh\ncurl -s localhost:4242\n```\n", "127.0.0.1"},
	} {
		blocks, err := parseRecipe(c.md)
		if err == nil {
			_, err = runBlock(context.Background(), blocks[0], work, env, map[string]string{})
		}
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("recipe %q: err = %v, want one containing %q", c.md, err, c.want)
		}
	}
	// And the control: a block that does what it says passes.
	blocks, err := parseRecipe("<!-- test: exit 3 -->\n```sh\necho 127.0.0.1:4242; exit 3\n```\n<!-- test: expect 127.0.0.1: -->\n")
	if err != nil {
		t.Fatal(err)
	}
	ports := map[string]string{}
	out, err := runBlock(context.Background(), blocks[0], work, env, ports)
	if err != nil {
		t.Fatalf("a block that meets its expectations failed: %v", err)
	}
	if strings.Contains(out, ":4242") || ports["4242"] == "" {
		t.Fatalf("127.0.0.1:4242 was not rewritten to a free port: %q, %v", out, ports)
	}
}

// recipeBlock is one fenced block of a recipe and the directives around it.
type recipeBlock struct {
	line       int
	lang       string
	body       string
	background bool
	exit       int
	instead    string
	skip       string
	needs      []string
	expects    []string
}

var (
	directiveRE = regexp.MustCompile(`^<!-- test: ([a-z]+)(?: (.*?))? -->$`)
	loopbackRE  = regexp.MustCompile(`127\.0\.0\.1:(4242|80\d\d)\b`)
)

// parseRecipe reads md's fenced blocks and their directives.
func parseRecipe(md string) ([]recipeBlock, error) {
	var (
		out     []recipeBlock
		pending recipeBlock
		needs   []string
		cur     *recipeBlock
		body    strings.Builder
	)
	for i, line := range strings.Split(md, "\n") {
		n := i + 1
		if cur != nil {
			if strings.TrimSpace(line) == "```" {
				cur.body = body.String()
				out = append(out, *cur)
				cur = nil
				continue
			}
			body.WriteString(line + "\n")
			continue
		}
		if lang, ok := strings.CutPrefix(line, "```"); ok {
			b := pending
			b.line, b.lang = n, strings.TrimSpace(lang)
			b.needs = append([]string(nil), needs...)
			cur = &b
			body.Reset()
			pending = recipeBlock{}
			continue
		}
		m := directiveRE.FindStringSubmatch(strings.TrimSpace(line))
		if m == nil {
			continue
		}
		arg := m[2]
		switch m[1] {
		case "background":
			pending.background = true
		case "exit":
			v, err := strconv.Atoi(arg)
			if err != nil {
				return nil, fmt.Errorf("line %d: exit %q: %v", n, arg, err)
			}
			pending.exit = v
		case "instead":
			pending.instead = arg
		case "skip":
			if arg == "" {
				return nil, fmt.Errorf("line %d: a skip names its reason", n)
			}
			pending.skip = arg
		case "needs":
			needs = append(needs, arg)
		case "expect":
			if len(out) == 0 {
				return nil, fmt.Errorf("line %d: an expect with no block before it", n)
			}
			out[len(out)-1].expects = append(out[len(out)-1].expects, arg)
		default:
			return nil, fmt.Errorf("line %d: unknown directive %q", n, m[1])
		}
	}
	if cur != nil {
		return nil, fmt.Errorf("line %d: a block that never closes", cur.line)
	}
	return out, nil
}

// runRecipe runs blocks in order in a fresh directory, stopping at the first
// failure, and returns how many ran.
func runRecipe(t *testing.T, blocks []recipeBlock, env []string) int {
	work := t.TempDir()
	ports := map[string]string{}
	var bg []*backgroundBlock
	t.Cleanup(func() {
		for _, b := range bg {
			b.stop()
		}
	})
	ran := 0
	for _, b := range blocks {
		if !runnable(b.lang) {
			continue
		}
		for _, f := range b.needs {
			// A container an earlier build left is reconverted, as every
			// gate's is (fresh).
			if err := fresh.Container(testmodels.Path(f)); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(testmodels.Path(f)); err != nil {
				t.Run("needs "+f, func(t *testing.T) {
					testmodels.Missing(t, "this block and every later one need %s in %s (internal/testmodels/fetch.sh fetches it): %v",
						f, testmodels.Dir(), err)
				})
				return ran
			}
		}
		ok := t.Run(fmt.Sprintf("line%d", b.line), func(t *testing.T) {
			if b.skip != "" {
				t.Skipf("skipped by the recipe: %s", b.skip)
			}
			if reason := interpreterMissing(b.lang, work); reason != "" {
				t.Skip(reason)
			}
			if b.background {
				p, err := startBackground(b, work, env, ports)
				if err != nil {
					t.Fatal(err)
				}
				bg = append(bg, p)
				ran++
				return
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
			defer cancel()
			out, err := runBlock(ctx, b, work, env, ports)
			ran++
			if err != nil {
				for _, p := range bg {
					t.Logf("background block at line %d printed:\n%s", p.line, p.out.String())
				}
				t.Fatalf("block at line %d: %v\n--- the block:\n%s--- it printed:\n%s", b.line, err, b.body, out)
			}
		})
		if !ok {
			t.Fatalf("stopped at the block on line %d; the blocks after it depend on it", b.line)
		}
	}
	return ran
}

func runnable(lang string) bool { return lang == "sh" || lang == "python" || lang == "js" }

// rewrite puts each 127.0.0.1:4242 (the default) and 127.0.0.1:80NN on a free port, the same one for the same
// address throughout the recipe, and refuses any other spelling of a local
// server, which the rewrite would miss.
func rewrite(s string, ports map[string]string) (string, error) {
	rest := loopbackRE.ReplaceAllString(s, "")
	if strings.Contains(rest, "localhost:") || regexp.MustCompile(`[^0-9.]:(4242|80\d\d)\b`).MatchString(rest) {
		return "", errors.New("a recipe addresses its server as 127.0.0.1:4242 or 127.0.0.1:80NN, which the test rewrites to a free port; this block spells it otherwise")
	}
	var err error
	out := loopbackRE.ReplaceAllStringFunc(s, func(m string) string {
		k := m[len("127.0.0.1:"):]
		if ports[k] == "" {
			p, e := freePort()
			if e != nil {
				err = e
			}
			ports[k] = p
		}
		return "127.0.0.1:" + ports[k]
	})
	return out, err
}

func freePort() (string, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", err
	}
	defer l.Close()
	return strconv.Itoa(l.Addr().(*net.TCPAddr).Port), nil
}

// command is the process a block runs as, its source rewritten.
func command(ctx context.Context, b recipeBlock, work string, env []string, ports map[string]string) (*exec.Cmd, error) {
	src := b.body
	if b.instead != "" {
		if !strings.Contains(src, "curl ") && !strings.Contains(src, "convert -o") {
			return nil, fmt.Errorf("an instead stands in for only a download (curl, or a catalog convert -o); this block is not one")
		}
		src = b.instead + "\n"
	}
	src, err := rewrite(src, ports)
	if err != nil {
		return nil, err
	}
	var cmd *exec.Cmd
	switch b.lang {
	case "sh":
		cmd = exec.CommandContext(ctx, "bash", "-euo", "pipefail", "-c", src)
	case "python", "js":
		ext, prog := ".py", pythonInterpreter()
		if b.lang == "js" {
			ext, prog = ".mjs", "node"
		}
		f := filepath.Join(work, fmt.Sprintf("block%d%s", b.line, ext))
		if err := os.WriteFile(f, []byte(src), 0o644); err != nil {
			return nil, err
		}
		cmd = exec.CommandContext(ctx, prog, f)
	default:
		return nil, fmt.Errorf("a %q block is not run", b.lang)
	}
	cmd.Dir, cmd.Env = work, env
	// The block's children (a server a block started) go with it.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error { return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL) }
	cmd.WaitDelay = 5 * time.Second
	return cmd, nil
}

// runBlock runs one block to completion and checks its exit status and
// expectations. The output is returned whether or not it passed.
func runBlock(ctx context.Context, b recipeBlock, work string, env []string, ports map[string]string) (string, error) {
	cmd, err := command(ctx, b, work, env, ports)
	if err != nil {
		return "", err
	}
	var buf bytes.Buffer
	cmd.Stdout, cmd.Stderr = &buf, &buf
	err = cmd.Run()
	out := buf.String()
	code := 0
	if err != nil {
		var ee *exec.ExitError
		if !errors.As(err, &ee) {
			return out, err
		}
		code = ee.ExitCode()
	}
	if ctx.Err() != nil {
		return out, fmt.Errorf("timed out: %v", ctx.Err())
	}
	if code != b.exit {
		return out, fmt.Errorf("exited %d, want %d", code, b.exit)
	}
	for _, e := range b.expects {
		e, err := rewrite(e, ports)
		if err != nil {
			return out, err
		}
		if !strings.Contains(out, e) {
			return out, fmt.Errorf("its output does not contain %q", e)
		}
	}
	return out, nil
}

// backgroundBlock is a block left running, a server, until its recipe ends.
type backgroundBlock struct {
	line int
	cmd  *exec.Cmd
	out  *lockedBuffer
}

func startBackground(b recipeBlock, work string, env []string, ports map[string]string) (*backgroundBlock, error) {
	cmd, err := command(context.Background(), b, work, env, ports)
	if err != nil {
		return nil, err
	}
	out := &lockedBuffer{}
	cmd.Stdout, cmd.Stderr = out, out
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &backgroundBlock{line: b.line, cmd: cmd, out: out}, nil
}

func (b *backgroundBlock) stop() {
	// The whole group: bash and the server it started.
	if err := syscall.Kill(-b.cmd.Process.Pid, syscall.SIGTERM); err == nil {
		done := make(chan struct{})
		go func() { b.cmd.Wait(); close(done) }()
		select {
		case <-done:
			return
		case <-time.After(10 * time.Second):
		}
	}
	syscall.Kill(-b.cmd.Process.Pid, syscall.SIGKILL)
}

type lockedBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (l *lockedBuffer) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.b.Write(p)
}

func (l *lockedBuffer) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.b.String()
}

func pythonInterpreter() string {
	if p := os.Getenv("JITLLM_RECIPE_PYTHON"); p != "" {
		return p
	}
	return "python3"
}

// interpreterMissing is why a Python or JavaScript block cannot run here, or
// "" when it can.
func interpreterMissing(lang, work string) string {
	switch lang {
	case "python":
		py := pythonInterpreter()
		if out, err := exec.Command(py, "-c", "import openai").CombinedOutput(); err != nil {
			return fmt.Sprintf("python block: %s cannot import openai (pip install openai, or set JITLLM_RECIPE_PYTHON to an interpreter that can): %v %s",
				py, err, strings.TrimSpace(string(out)))
		}
	case "js":
		if nm := os.Getenv("JITLLM_RECIPE_NODE_MODULES"); nm != "" {
			link := filepath.Join(work, "node_modules")
			if _, err := os.Lstat(link); err != nil {
				if err := os.Symlink(nm, link); err != nil {
					return fmt.Sprintf("js block: linking JITLLM_RECIPE_NODE_MODULES: %v", err)
				}
			}
		}
		cmd := exec.Command("node", "--input-type=module", "-e", "import 'openai'")
		cmd.Dir = work
		if out, err := cmd.CombinedOutput(); err != nil {
			return fmt.Sprintf("js block: node cannot import openai (npm install openai, and set JITLLM_RECIPE_NODE_MODULES to its node_modules): %v %s",
				err, firstLine(string(out)))
		}
	}
	return ""
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// buildRecipeBinaries builds jitllm and jitllmd from this tree into a
// temporary directory, which the recipes find first on PATH.
func buildRecipeBinaries(t *testing.T, root string) string {
	t.Helper()
	goTool := filepath.Join(runtime.GOROOT(), "bin", "go")
	if _, err := os.Stat(goTool); err != nil {
		if goTool, err = exec.LookPath("go"); err != nil {
			t.Fatalf("no go tool to build jitllm with: %v", err)
		}
	}
	bin := t.TempDir()
	for _, b := range []struct{ dir, pkg, out string }{
		{root, "./cmd/jitllm", "jitllm"},
		{filepath.Join(root, "server"), "./cmd/jitllmd", "jitllmd"},
	} {
		cmd := exec.Command(goTool, "build", "-o", filepath.Join(bin, b.out), b.pkg)
		cmd.Dir = b.dir
		cmd.Env = append(os.Environ(), "CGO_ENABLED=0")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("building %s: %v\n%s", b.out, err, out)
		}
	}
	return bin
}

// recipeEnv is the environment a block runs in: the built binaries first on
// PATH, no JITLLM_MODELS (a reader's shell has none, so the recipes name their
// directories), and the two variables an instead may copy from.
func recipeEnv(root, bin string) []string {
	var env []string
	for _, kv := range os.Environ() {
		k, _, _ := strings.Cut(kv, "=")
		switch k {
		case "PATH", testmodels.Env, "JITLLM_REPO", "JITLLM_TEST_MODELS":
			continue
		}
		env = append(env, kv)
	}
	dir, err := filepath.Abs(testmodels.Dir())
	if err != nil {
		dir = testmodels.Dir()
	}
	return append(env,
		"PATH="+bin+string(os.PathListSeparator)+os.Getenv("PATH"),
		"JITLLM_REPO="+root,
		"JITLLM_TEST_MODELS="+dir,
	)
}
