package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/model"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// instructModel is a real instruct model: asked a short question, a greedy
// answer ends on its end-of-turn token after a few words. The tiny models the
// other gates load never sample an end-of-generation token (TinyStories is
// trained without one; the random fixtures put no weight on it).
const instructModel = "SmolLM2-360M-Instruct-Q8_0.jlm"

// ignoreEOSMax is the generate length the ignore_eos gates ask for, well past
// where the instruct model's answer ends.
const ignoreEOSMax = 48

const ignoreEOSQuestion = "What is the capital of France?"

// oaEnding is the part of an OpenAI completion the ignore_eos gates read.
type oaEnding struct {
	Choices []struct {
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *oaUsage `json:"usage"`
}

// postOA sends body to path on the server at base and returns the finish
// reason and the completion token count.
func postOA(t *testing.T, base, path, body string) (string, int) {
	t.Helper()
	resp, err := http.Post(base+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST %s: %v", path, err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("POST %s %s: %d %s", path, body, resp.StatusCode, b)
	}
	var out oaEnding
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("POST %s: %v in %s", path, err, b)
	}
	if len(out.Choices) != 1 || out.Choices[0].FinishReason == nil || out.Usage == nil {
		t.Fatalf("POST %s: no finish reason or usage in %s", path, b)
	}
	return *out.Choices[0].FinishReason, out.Usage.CompletionTokens
}

// wantEarlyThenFull is the gate's shape: the request without ignore_eos ends
// on an end-of-generation token before the limit, and the same request with
// it runs to exactly the limit.
func wantEarlyThenFull(t *testing.T, what string, run func(ignore bool) (string, int)) {
	t.Helper()
	reason, n := run(false)
	if reason != "stop" || n >= ignoreEOSMax || n == 0 {
		t.Fatalf("%s without ignore_eos: finish %q after %d of %d tokens; the gate needs a generate that ends "+
			"on an end-of-generation token first, or it cannot tell ignore_eos from max_tokens", what, reason, n, ignoreEOSMax)
	}
	reason2, n2 := run(true)
	if n2 != ignoreEOSMax || reason2 != "length" {
		t.Fatalf("%s with ignore_eos: finish %q after %d tokens, want \"length\" after exactly %d "+
			"(without it the same request stopped after %d)", what, reason2, n2, ignoreEOSMax, n)
	}
	t.Logf("%s: %d tokens without ignore_eos, %d with", what, n, n2)
}

// TestIgnoreEOSRunsToMaxTokens: `ignore_eos` (the vLLM and llama-server
// extension) on /v1/chat/completions, /v1/completions and the Connect
// GenerateRequest keeps a greedy generate going past the end-of-turn token
// it otherwise stops on, to exactly max_tokens.
func TestIgnoreEOSRunsToMaxTokens(t *testing.T) {
	_, lm, c := loadedEngine(t, instructModel, "inst", LoadOptions{})

	wantEarlyThenFull(t, "/v1/chat/completions", func(ignore bool) (string, int) {
		return postOA(t, c.url, "/v1/chat/completions", fmt.Sprintf(
			`{"model":"inst","messages":[{"role":"user","content":%q}],"max_tokens":%d,"temperature":0,"ignore_eos":%v}`,
			ignoreEOSQuestion, ignoreEOSMax, ignore))
	})

	// The legacy endpoint takes the same conversation as token ids, rendered
	// by the model's own template.
	conv, err := lm.m.ChatIDsTools([]model.ChatMessage{{Role: "user", Content: ignoreEOSQuestion}}, nil, true)
	if err != nil {
		t.Fatal(err)
	}
	prompt, err := json.Marshal(conv)
	if err != nil {
		t.Fatal(err)
	}
	wantEarlyThenFull(t, "/v1/completions", func(ignore bool) (string, int) {
		return postOA(t, c.url, "/v1/completions", fmt.Sprintf(
			`{"model":"inst","prompt":%s,"max_tokens":%d,"temperature":0,"ignore_eos":%v}`,
			prompt, ignoreEOSMax, ignore))
	})

	wantEarlyThenFull(t, "Connect Generate", func(ignore bool) (string, int) {
		r := complete(c, &v1.GenerateRequest{ModelId: "inst", Prompt: ids(conv...),
			MaxTokens: ignoreEOSMax, IgnoreEos: ignore})
		if r.err != nil {
			t.Fatal(r.err)
		}
		if int(r.finished.GetCompletionTokens()) != len(r.ids) {
			t.Fatalf("Connect: %d completion tokens reported, %d ids returned",
				r.finished.GetCompletionTokens(), len(r.ids))
		}
		return map[v1.FinishReason]string{
			v1.FinishReason_FINISH_REASON_EOS:        "stop",
			v1.FinishReason_FINISH_REASON_MAX_TOKENS: "length",
		}[r.finished.GetReason()], len(r.ids)
	})
}

// TestIgnoreEOSRunsToMaxTokensBatched is the same gate for a generate that
// runs as a row of its model's step loop (batch.go), which has its own stop
// test: the instruct model on the first GPU.
func TestIgnoreEOSRunsToMaxTokensBatched(t *testing.T) {
	path := modelPath(t, instructModel)
	e := New(admitConfig(Config{Probe: oneCardProbe, Version: "test", DefaultMaxSeq: 512}))
	t.Cleanup(e.Close)
	lm, err := e.LoadModel(LoadOptions{Path: path, ModelID: "inst", DeviceIDs: []string{"gpu:0"}, Sessions: 1})
	if err != nil {
		t.Skipf("NO DEVICE: loading %s onto -devices gpu:0 failed (%v) -- this gate proved nothing", instructModel, err)
	}
	if lm.loop == nil {
		t.Fatal("a model loaded onto a device has no step loop")
	}
	requireWholeOnDevice(t, e, lm)
	c := serveEngine(t, e)
	conv, err := lm.m.ChatIDsTools([]model.ChatMessage{{Role: "user", Content: ignoreEOSQuestion}}, nil, true)
	if err != nil {
		t.Fatal(err)
	}
	wantEarlyThenFull(t, "a batched row", func(ignore bool) (string, int) {
		r := complete(c, &v1.GenerateRequest{ModelId: "inst", Prompt: ids(conv...),
			MaxTokens: ignoreEOSMax, IgnoreEos: ignore})
		if r.err != nil {
			t.Fatal(r.err)
		}
		if !r.started.GetBatched() {
			t.Fatal("the generate did not run as a row of the step loop")
		}
		return map[v1.FinishReason]string{
			v1.FinishReason_FINISH_REASON_EOS:        "stop",
			v1.FinishReason_FINISH_REASON_MAX_TOKENS: "length",
		}[r.finished.GetReason()], len(r.ids)
	})
}
