package server

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// noSystemTemplate is the template Mistral-7B-Instruct-v0.3's GGUF carries: no
// system role, and a raise on anything that does not alternate user/assistant.
const noSystemTemplate = `{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}`

// TestASystemPromptWithNoRoleIsTheCallersToChange: a system message to a model
// whose chat template has no system role is refused, naming the role, by
// every API -- InvalidArgument from ApplyChatTemplate, Complete and the
// Generate stream, a 400 from the OpenAI, Anthropic and /apply-template routes,
// streamed or not -- and the same requests without it are answered. The model
// is stories260K carrying that template. With chatErr passing the refusal
// through unwrapped, each comes back as an internal error, which a client
// retries; without CheckAdmission's refusal a stream answers 200 and an error
// frame.
func TestASystemPromptWithNoRoleIsTheCallersToChange(t *testing.T) {
	src := testmodels.Path("stories260K.gguf")
	if _, err := os.Stat(src); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v -- this gate proved nothing", err)
	}
	dir := t.TempDir()
	tpl := filepath.Join(dir, "nosystem.jinja")
	if err := os.WriteFile(tpl, []byte(noSystemTemplate), 0o644); err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(dir, "nosystem.jlm")
	if _, err := convert.FromGGUF(src, dst, jlm.Fingerprint{Host: "test"}, convert.WithChatTemplate(tpl)); err != nil {
		t.Fatal(err)
	}
	e := New(admitConfig(Config{ModelDir: dir, Probe: oneCardProbe, Version: "test", NoOnDemand: true}))
	t.Cleanup(e.Close)
	if _, err := e.LoadModel(LoadOptions{Path: dst, ModelID: "m", DeviceIDs: []string{"cpu"}}); err != nil {
		t.Fatal(err)
	}
	c := serveEngine(t, e)
	ctx := context.Background()

	sys := []*v1.ChatMessage{{Role: "system", Content: "Be terse."}, {Role: "user", Content: "hi"}}
	apply := func(msgs []*v1.ChatMessage) error {
		_, err := c.model.ApplyChatTemplate(ctx, req(&v1.ApplyChatTemplateRequest{ModelId: "m", Messages: msgs,
			AddGenerationPrompt: true}))
		return err
	}
	complete := func(msgs []*v1.ChatMessage) error {
		_, err := c.inference.Complete(ctx, req(&v1.GenerateRequest{ModelId: "m", MaxTokens: 1,
			Prompt: &v1.PromptInput{Input: &v1.PromptInput_Chat{Chat: &v1.ChatPrompt{Messages: msgs,
				AddGenerationPrompt: true}}}}))
		return err
	}
	stream := func(msgs []*v1.ChatMessage) error {
		st, err := c.inference.Generate(ctx, req(&v1.GenerateRequest{ModelId: "m", MaxTokens: 1,
			Prompt: &v1.PromptInput{Input: &v1.PromptInput_Chat{Chat: &v1.ChatPrompt{Messages: msgs,
				AddGenerationPrompt: true}}}}))
		if err != nil {
			return err
		}
		defer st.Close()
		for st.Receive() {
		}
		return st.Err()
	}
	for _, rpc := range []struct {
		name string
		call func([]*v1.ChatMessage) error
	}{{"ApplyChatTemplate", apply}, {"Complete", complete}, {"Generate", stream}} {
		ce := wantCode(t, rpc.name+" with a system message", rpc.call(sys), connect.CodeInvalidArgument)
		if !strings.Contains(ce.Message(), "no system role") {
			t.Errorf("%s: %q does not name the missing system role", rpc.name, ce.Message())
		}
		if err := rpc.call(sys[1:]); err != nil {
			t.Errorf("%s without the system message: %v", rpc.name, err)
		}
	}

	// A system prompt sent apart is named as the caller sent it: there is no
	// message 0 in the request for it to be.
	terse := "Be terse."
	apart := &v1.ChatPrompt{System: &terse, Messages: sys[1:], AddGenerationPrompt: true}
	_, err := c.inference.Complete(ctx, req(&v1.GenerateRequest{ModelId: "m", MaxTokens: 1,
		Prompt: &v1.PromptInput{Input: &v1.PromptInput_Chat{Chat: apart}}}))
	ce := wantCode(t, "Complete with a system prompt apart", err, connect.CodeInvalidArgument)
	if !strings.Contains(ce.Message(), "system prompt") || strings.Contains(ce.Message(), "message 0") {
		t.Errorf("Complete with a system prompt apart: %q does not name the system prompt", ce.Message())
	}

	user := `{"role":"user","content":"hi"}`
	openai := `{"model":"m","max_tokens":1,%s"messages":[{"role":"system","content":"Be terse."},` + user + `]}`
	anthropic := `{"model":"m","max_tokens":1,%s"system":"Be terse.","messages":[` + user + `]}`
	without := `{"model":"m","max_tokens":1,%s"messages":[` + user + `]}`
	// A streamed request is refused before its stream opens: once the 200 is
	// written the refusal could only be an error frame.
	for _, r := range []struct{ path, with, without string }{
		{"/v1/chat/completions", fmt.Sprintf(openai, ""), fmt.Sprintf(without, "")},
		{"/v1/chat/completions stream", fmt.Sprintf(openai, `"stream":true,`), fmt.Sprintf(without, `"stream":true,`)},
		{"/v1/messages", fmt.Sprintf(anthropic, ""), fmt.Sprintf(without, "")},
		{"/v1/messages stream", fmt.Sprintf(anthropic, `"stream":true,`), fmt.Sprintf(without, `"stream":true,`)},
		{"/apply-template",
			`{"model":"m","messages":[{"role":"system","content":"Be terse."},` + user + `]}`,
			`{"model":"m","messages":[` + user + `]}`},
	} {
		path, _, _ := strings.Cut(r.path, " ")
		status, body := postRaw(t, c.url+path, r.with)
		if status != http.StatusBadRequest || !strings.Contains(body, "no system role") ||
			!strings.Contains(body, "invalid_request_error") {
			t.Errorf("%s with a system message: %d %s, want a 400 naming the missing system role", r.path, status, body)
		}
		if path == "/v1/messages" && strings.Contains(body, "message 0") {
			t.Errorf("%s: %s names message 0 for the top-level system prompt", r.path, body)
		}
		if status, body := postRaw(t, c.url+path, r.without); status != http.StatusOK {
			t.Errorf("%s without one: %d %s", r.path, status, body)
		}
	}
}
