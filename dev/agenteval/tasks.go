//go:build unix

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"time"
)

// task is one thing an agent is asked to do. Prompt and Reference may name
// {port} (the task's server port) and {dir} (its workspace).
type task struct {
	ID, Title string
	// Prompt is all the agent is told.
	Prompt string
	// Setup prepares the workspace before the agent starts.
	Setup func(*env) error
	// Check is the success criterion, by machine, after the agent exits.
	Check func(*env) []Check
	// Required is the validation the agent must be seen running.
	Required []requirement
	// Reference is a shell solution: -agent reference runs it, which shows
	// the task can be completed and its checks can pass.
	Reference string
}

// greedy is the request every check sends: stories260K's greedy
// continuation of it is the same on every machine.
const greedy = `{"model":"stories","prompt":"Once upon a time","max_tokens":16,"temperature":0}`

// lily is that continuation.
const lily = "there was a little girl named Lily"

func tasks() []task {
	return []task{
		{
			ID:    "cpu-server",
			Title: "start a CPU server from a GGUF",
			Prompt: `This directory holds stories260K.gguf, a tiny story-writing model. The jitllm and
jitllmd programs are on your PATH (documentation: https://github.com/jitllm/jitllm, in
particular docs/recipes/; and each program's -h).

Start a jitllm server that runs this model on the CPU only (no GPU), listening on
127.0.0.1:{port}, serving the model under the id "stories" on its OpenAI-compatible API.
Confirm it answers a completion request, and leave it running when you finish.
`,
			Setup: func(e *env) error { return copyStories(e, "stories260K.gguf") },
			Check: func(e *env) []Check {
				return append(modelListed(e), completionOnCPU(e)...)
			},
			Required: []requirement{
				ran("converted the GGUF to a container", "jitllm", "convert"),
				ran("sent the server a request", "curl", "127.0.0.1:"),
			},
			Reference: `mkdir -p models && jitllm convert stories260K.gguf models/stories.jlm &&
nohup jitllmd serve -addr 127.0.0.1:{port} -models models -load stories.jlm -id stories -devices cpu > server.log 2>&1 < /dev/null &
for i in $(seq 120); do curl -sf http://127.0.0.1:{port}/healthz && break; sleep 0.25; done
curl -s http://127.0.0.1:{port}/v1/completions -H 'Content-Type: application/json' -d '` + greedy + `'`,
		},
		{
			ID:    "connect-client",
			Title: "write a client against a running server",
			Prompt: `A jitllm server is running at http://127.0.0.1:{port}. It serves an OpenAI-compatible
API, and the model id is "stories".

Write a program in this directory -- client.sh, client.py or client.mjs -- that asks that
model for a greedy (temperature 0) completion of the prompt "Once upon a time", at most 16
tokens, and prints only the generated text. Run it to show it works.
`,
			Setup: serveStories(""),
			Check: func(e *env) []Check {
				c := Check{Name: "a client prints the greedy continuation, and only the text"}
				out, which, err := runClient(e)
				switch {
				case err != nil:
					c.Detail = err.Error()
				case !strings.Contains(out, lily):
					c.Detail = fmt.Sprintf("%s printed %q", which, out)
				case strings.Contains(out, `"choices"`):
					c.Detail = fmt.Sprintf("%s printed the whole response, not only the text: %q", which, out)
				default:
					c.Passed, c.Detail = true, which
				}
				return []Check{c}
			},
			Required: []requirement{{
				Name: "ran a client against the server",
				Match: func(line string) bool {
					t, _, _ := strings.Cut(line, "\t")
					return t == "curl" || t == "python3" || t == "node"
				},
			}},
			Reference: `cat > client.sh <<'EOF'
#!/bin/sh
curl -s http://127.0.0.1:{port}/v1/completions -H 'Content-Type: application/json' \
  -d '` + greedy + `' | sed -n 's/.*"text":"\([^"]*\)".*/\1/p'
EOF
chmod +x client.sh && ./client.sh`,
		},
		{
			ID:    "unsupported-model",
			Title: "tell a supported model from an unsupported one",
			Prompt: `Two GGUF files are in this directory, alpha.gguf and beta.gguf. jitllm and jitllmd are on
your PATH (documentation: https://github.com/jitllm/jitllm).

Find out which of them jitllm can run. Write verdict.json in this form:

  {"alpha.gguf": {"supported": true or false, "reason": "..."},
   "beta.gguf":  {"supported": true or false, "reason": "..."}}

where a reason for an unsupported file names what is unsupported. Do not modify either file.
`,
			Setup: func(e *env) error {
				if err := copyStories(e, "alpha.gguf"); err != nil {
					return err
				}
				// stories260K with its architecture renamed to one no
				// converter lists: the header is otherwise intact.
				b, err := os.ReadFile(filepath.Join(e.dir, "alpha.gguf"))
				if err != nil {
					return err
				}
				if bytes.Count(b, []byte("llama")) == 0 {
					return errors.New("stories260K.gguf no longer spells its architecture llama")
				}
				return os.WriteFile(filepath.Join(e.dir, "beta.gguf"), bytes.ReplaceAll(b, []byte("llama"), []byte("ollmo")), 0o644)
			},
			Check: func(e *env) []Check {
				return verdictChecks(e)
			},
			Required: []requirement{ran("ran jitllm on beta.gguf", "jitllm", "beta.gguf")},
			Reference: `jitllm convert alpha.gguf .agenteval/alpha.jlm
b=$(jitllm convert beta.gguf .agenteval/beta.jlm 2>&1 | grep -o 'architecture "[^"]*" is not implemented' | sed 's/"/\\"/g')
cat > verdict.json <<EOF
{"alpha.gguf": {"supported": true, "reason": "llama converts"},
 "beta.gguf": {"supported": false, "reason": "$b"}}
EOF`,
		},
		{
			ID:    "placement",
			Title: "diagnose and fix a memory budget on a live server",
			Prompt: `The jitllm server at http://127.0.0.1:{port} (model id "stories") reads weights from disk
for every token it generates, although this machine has plenty of free memory. jitllmd's
client verbs talk to it with -addr 127.0.0.1:{port} (see jitllmd -h; documentation at
https://github.com/jitllm/jitllm, docs/recipes/).

Find out why, and fix it without restarting the server. Write the cause, in one sentence,
to diagnosis.txt.
`,
			Setup: serveStories("400K"),
			Check: func(e *env) []Check {
				return placementChecks(e)
			},
			Required: []requirement{{
				Name: "read the server's pager or stats",
				Match: func(line string) bool {
					t, args, _ := strings.Cut(line, "\t")
					return t == "jitllmd" && (strings.HasPrefix(args, "place") || strings.HasPrefix(args, "stats") || strings.HasPrefix(args, "models"))
				},
			}},
			Reference: `jitllmd stats -addr 127.0.0.1:{port}
jitllmd place -addr 127.0.0.1:{port} -model stories
jitllmd place -addr 127.0.0.1:{port} -model stories -maxmem 64M
echo "The model's host weight budget (-maxmem 400K) holds fewer blocks than a token reads, so every token pages them in from disk." > diagnosis.txt`,
		},
		{
			ID:    "server-bug",
			Title: "fix a seeded bug in the server",
			Prompt: `./jitllm is a checkout of jitllm. A user reports that their OpenAI client sends
"parallel_tool_calls": false to jitllmd and still gets several tool calls in one reply,
while requests with "parallel_tool_calls": true get at most one.

Find and fix the bug in ./jitllm, and run the server module's tests that cover the code you
changed. ./jitllm/AGENTS.md has the project's rules.
`,
			Setup: func(e *env) error {
				return seedBug(e)
			},
			Check: func(e *env) []Check {
				return bugChecks(e)
			},
			Required: []requirement{ran("ran the server module's tests", "go", "test")},
			Reference: `cd jitllm && sed 's/c.Single = parallel != nil \&\& \*parallel/c.Single = parallel != nil \&\& !*parallel/' server/tools.go > server/tools.go.new &&
mv server/tools.go.new server/tools.go &&
cd server && go test -count=1 -run 'Tool' .`,
		},
	}
}

func storiesSource(e *env) string {
	return filepath.Join(e.cfg.root, "testdata", "models", "stories260K.gguf")
}

func copyStories(e *env, name string) error {
	b, err := os.ReadFile(storiesSource(e))
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(e.dir, name), b, 0o644)
}

// serveStories starts a server on the task's port with stories260K as
// "stories", on the CPU, under the host budget maxmem when one is given.
func serveStories(maxmem string) func(*env) error {
	return func(e *env) error {
		models := filepath.Join(e.dir, ".agenteval", "models")
		if err := os.MkdirAll(models, 0o755); err != nil {
			return err
		}
		if out, err := e.run(e.dir, "jitllm", "convert", storiesSource(e), filepath.Join(models, "stories.jlm")); err != nil {
			return fmt.Errorf("convert: %v\n%s", err, out)
		}
		args := []string{"serve", "-addr", "127.0.0.1:" + e.port, "-models", models, "-load", "stories.jlm", "-id", "stories", "-devices", "cpu"}
		if maxmem != "" {
			args = append(args, "-maxmem", maxmem)
		}
		return e.start(args...)
	}
}

func httpGet(url string) (string, error) {
	c := http.Client{Timeout: 10 * time.Second}
	r, err := c.Get(url)
	if err != nil {
		return "", err
	}
	defer r.Body.Close()
	b, err := io.ReadAll(r.Body)
	if r.StatusCode != http.StatusOK {
		return string(b), fmt.Errorf("%s: %s: %s", url, r.Status, strings.TrimSpace(string(b)))
	}
	return string(b), err
}

func httpPost(url, body string) (string, error) {
	c := http.Client{Timeout: time.Minute}
	r, err := c.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		return "", err
	}
	defer r.Body.Close()
	b, err := io.ReadAll(r.Body)
	if r.StatusCode != http.StatusOK {
		return string(b), fmt.Errorf("%s: %s: %s", url, r.Status, strings.TrimSpace(string(b)))
	}
	return string(b), err
}

func modelListed(e *env) []Check {
	c := Check{Name: `the server lists the model "stories"`}
	out, err := httpGet(e.url("/v1/models"))
	switch {
	case err != nil:
		c.Detail = err.Error()
	case !strings.Contains(out, `"id":"stories"`):
		c.Detail = out
	default:
		c.Passed = true
	}
	return []Check{c}
}

// completionOnCPU asks for the greedy completion and reads where it ran.
func completionOnCPU(e *env) []Check {
	text := Check{Name: "a greedy completion continues the story"}
	cpu := Check{Name: "every block ran on the host"}
	out, err := httpPost(e.url("/v1/completions"), greedy)
	if err != nil {
		text.Detail, cpu.Detail = err.Error(), "no reply"
		return []Check{text, cpu}
	}
	var r struct {
		Choices []struct {
			Text string `json:"text"`
		} `json:"choices"`
		Jitllm struct {
			DeviceBlocks int `json:"device_blocks"`
			CPUBlocks    int `json:"cpu_blocks"`
		} `json:"jitllm"`
	}
	if err := json.Unmarshal([]byte(out), &r); err != nil || len(r.Choices) == 0 {
		text.Detail, cpu.Detail = out, "no reply"
		return []Check{text, cpu}
	}
	text.Passed, text.Detail = strings.Contains(r.Choices[0].Text, lily), r.Choices[0].Text
	cpu.Passed = r.Jitllm.DeviceBlocks == 0 && r.Jitllm.CPUBlocks > 0
	cpu.Detail = fmt.Sprintf("%d device block(s), %d host block(s)", r.Jitllm.DeviceBlocks, r.Jitllm.CPUBlocks)
	return []Check{text, cpu}
}

// runClient runs the client the agent wrote, whichever kind it is.
func runClient(e *env) (string, string, error) {
	for _, c := range []struct{ file, prog string }{
		{"client.sh", "sh"}, {"client.py", "python3"}, {"client.mjs", "node"},
	} {
		p := filepath.Join(e.dir, c.file)
		if _, err := os.Stat(p); err != nil {
			continue
		}
		cmd := exec.Command(c.prog, p)
		cmd.Dir = e.dir
		cmd.Env = e.environ()
		var out, errb bytes.Buffer
		cmd.Stdout, cmd.Stderr = &out, &errb
		if err := cmd.Run(); err != nil {
			return out.String(), c.file, fmt.Errorf("%s: %v: %s", c.file, err, errb.String())
		}
		return out.String(), c.file, nil
	}
	return "", "", errors.New("no client.sh, client.py or client.mjs in the workspace")
}

func verdictChecks(e *env) []Check {
	shape := Check{Name: "verdict.json parses and names both files"}
	alpha := Check{Name: "alpha.gguf is supported"}
	beta := Check{Name: `beta.gguf is unsupported, naming its architecture "ollmo"`}
	intact := Check{Name: "neither file was modified"}
	b, err := os.ReadFile(filepath.Join(e.dir, "verdict.json"))
	var v map[string]struct {
		Supported *bool  `json:"supported"`
		Reason    string `json:"reason"`
	}
	if err == nil {
		err = json.Unmarshal(b, &v)
	}
	switch {
	case err != nil:
		shape.Detail = err.Error()
	case v["alpha.gguf"].Supported == nil || v["beta.gguf"].Supported == nil:
		shape.Detail = string(b)
	default:
		shape.Passed = true
		alpha.Passed = *v["alpha.gguf"].Supported
		alpha.Detail = v["alpha.gguf"].Reason
		beta.Passed = !*v["beta.gguf"].Supported && strings.Contains(v["beta.gguf"].Reason, "ollmo")
		beta.Detail = v["beta.gguf"].Reason
	}
	src, err := os.ReadFile(storiesSource(e))
	if err == nil {
		a, errA := os.ReadFile(filepath.Join(e.dir, "alpha.gguf"))
		bb, errB := os.ReadFile(filepath.Join(e.dir, "beta.gguf"))
		intact.Passed = errA == nil && errB == nil && sha256.Sum256(a) == sha256.Sum256(src) &&
			sha256.Sum256(bb) == sha256.Sum256(bytes.ReplaceAll(src, []byte("llama"), []byte("ollmo")))
	}
	return []Check{shape, alpha, beta, intact}
}

func placementChecks(e *env) []Check {
	fits := Check{Name: "the model now fits its budget: a token reads nothing from disk"}
	same := Check{Name: "the server was not restarted"}
	diag := Check{Name: "diagnosis.txt names the memory budget"}
	text := Check{Name: "the server still answers correctly"}
	out, err := httpPost(e.url("/v1/completions"), greedy)
	text.Passed, text.Detail = err == nil && strings.Contains(out, lily), firstLine(out)
	stats, err := e.run(e.dir, "jitllmd", "stats", "-addr", "127.0.0.1:"+e.port)
	fits.Passed, fits.Detail = err == nil && strings.Contains(stats, "FITS"), lineWith(stats, "resident")
	// Signal 0 delivers nothing and fails once the process is gone.
	same.Passed = e.pid > 0 && syscall.Kill(e.pid, 0) == nil
	same.Detail = fmt.Sprintf("pid %d", e.pid)
	b, err := os.ReadFile(filepath.Join(e.dir, "diagnosis.txt"))
	d := strings.ToLower(string(b))
	diag.Passed = err == nil && slices.ContainsFunc([]string{"budget", "maxmem", "memory limit", "evict", "page"}, func(w string) bool {
		return strings.Contains(d, w)
	})
	diag.Detail = strings.TrimSpace(string(b))
	return []Check{fits, same, diag, text}
}

// The seeded bug: parallel_tool_calls read inverted.
const (
	seedFile  = "server/tools.go"
	seedRight = "c.Single = parallel != nil && !*parallel"
	seedWrong = "c.Single = parallel != nil && *parallel"
)

// seedBug copies the committed tree into ./jitllm and plants the bug.
func seedBug(e *env) error {
	dst := filepath.Join(e.dir, "jitllm")
	if err := os.MkdirAll(dst, 0o755); err != nil {
		return err
	}
	archive := exec.Command("git", "-C", e.cfg.root, "archive", "HEAD")
	untar := exec.Command("tar", "-x", "-C", dst)
	pipe, err := archive.StdoutPipe()
	if err != nil {
		return err
	}
	untar.Stdin = pipe
	if err := untar.Start(); err != nil {
		return err
	}
	if err := archive.Run(); err != nil {
		return fmt.Errorf("git archive: %w", err)
	}
	if err := untar.Wait(); err != nil {
		return fmt.Errorf("tar: %w", err)
	}
	p := filepath.Join(dst, seedFile)
	b, err := os.ReadFile(p)
	if err != nil {
		return err
	}
	if n := strings.Count(string(b), seedRight); n != 1 {
		return fmt.Errorf("%s has %d copies of %q; the seed needs exactly one -- update dev/agenteval's seed", seedFile, n, seedRight)
	}
	return os.WriteFile(p, []byte(strings.Replace(string(b), seedRight, seedWrong, 1)), 0o644)
}

// hiddenTest is the check the agent never sees: it holds oaToolChoice to
// OpenAI's meaning of parallel_tool_calls.
const hiddenTest = `package server

import (
	"encoding/json"
	"testing"
)

func TestAgentEvalParallelToolCalls(t *testing.T) {
	no, yes := false, true
	for _, c := range []struct {
		parallel *bool
		single   bool
	}{{nil, false}, {&no, true}, {&yes, false}} {
		got, err := oaToolChoice(json.RawMessage(` + "`\"auto\"`" + `), c.parallel, nil)
		if err != nil || got.Single != c.single {
			t.Errorf("parallel_tool_calls %v: Single = %v, err %v; want %v", c.parallel, got.Single, err, c.single)
		}
	}
}
`

func bugChecks(e *env) []Check {
	pass := Check{Name: "a hidden test of parallel_tool_calls passes"}
	builds := Check{Name: "the server module builds and vets"}
	server := filepath.Join(e.dir, "jitllm", "server")
	ht := filepath.Join(server, "agenteval_hidden_test.go")
	if err := os.WriteFile(ht, []byte(hiddenTest), 0o644); err != nil {
		pass.Detail = err.Error()
		return []Check{pass, builds}
	}
	defer os.Remove(ht)
	out, err := e.run(server, "go", "vet", ".")
	builds.Passed, builds.Detail = err == nil, lastLines(out, 5)
	out, err = e.run(server, "go", "test", "-count=1", "-run", "TestAgentEvalParallelToolCalls", ".")
	pass.Passed, pass.Detail = err == nil, lastLines(out, 5)
	return []Check{pass, builds}
}

func firstLine(s string) string {
	s, _, _ = strings.Cut(strings.TrimSpace(s), "\n")
	return s
}

func lineWith(s, word string) string {
	for l := range strings.Lines(s) {
		if strings.Contains(l, word) {
			return strings.TrimSpace(l)
		}
	}
	return ""
}

func lastLines(s string, n int) string {
	l := strings.Split(strings.TrimSpace(s), "\n")
	return strings.Join(l[max(0, len(l)-n):], "\n")
}
