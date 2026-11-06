package engine

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/tok/jinja"
)

// mistralV1 is the template Mistral-7B-Instruct-v0.3's GGUF carries: no system
// role, and a raise on anything that does not alternate user/assistant.
const mistralV1 = `{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}`

const chatML = `{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}`

// A system prompt reaches a template with no system role inside the first user
// message, as llama.cpp sends it, and a template that has one keeps it as its
// own message.
func TestASystemPromptReachesATemplateWithoutASystemRole(t *testing.T) {
	ctx := func() map[string]any {
		return map[string]any{"add_generation_prompt": true, "bos_token": "<s>", "eos_token": "</s>"}
	}
	render := func(src string, msgs []model.ChatMessage) string {
		tpl, err := jinja.Compile(src)
		if err != nil {
			t.Fatal(err)
		}
		c := ctx()
		msgs = model.FoldSystem(tpl, c, false, msgs)
		ms := make([]any, len(msgs))
		for i, m := range msgs {
			ms[i] = map[string]any{"role": m.Role, "content": m.Content}
		}
		c["messages"] = ms
		out, err := tpl.Render(c)
		if err != nil {
			t.Fatalf("render: %v", err)
		}
		return out
	}
	sys := []model.ChatMessage{{Role: "system", Content: "be terse"}, {Role: "user", Content: "hi"}}
	if got, want := render(mistralV1, sys), "<s>[INST] be terse\nhi [/INST]"; got != want {
		t.Errorf("mistral: %q, want %q", got, want)
	}
	if got := render(chatML, sys); !strings.Contains(got, "<|im_start|>system\nbe terse<|im_end|>") {
		t.Errorf("a template with a system role lost it: %q", got)
	}
	// A system prompt with nothing after it has nowhere to go.
	tpl, _ := jinja.Compile(mistralV1)
	if got := model.FoldSystem(tpl, ctx(), false, sys[:1]); len(got) != 0 {
		t.Errorf("a lone system message survived a template without the role: %+v", got)
	}
}

// Every model takes a system prompt, one way or the other. The log
// names the templates that have no system role, which is the census the fold
// was written for.
func TestEveryChatModelTakesASystemPrompt(t *testing.T) {
	checked := 0
	for _, p := range testmodels.Glob("*.jlm") {
		m, err := model.Open(p)
		if err != nil {
			continue // a stale container; the conversion gates own that
		}
		func() {
			defer m.Close()
			if !m.HasChatTemplate() {
				return
			}
			checked++
			name := filepath.Base(p)
			out, err := m.ChatPrompt([]model.ChatMessage{
				{Role: "system", Content: "be terse"},
				{Role: "user", Content: "hello"},
			}, true)
			if err != nil {
				t.Errorf("%s: %v", name, err)
				return
			}
			if !strings.Contains(out, "be terse") || !strings.Contains(out, "hello") {
				t.Errorf("%s: the system prompt or the message is missing:\n%q", name, out)
			}
			src, _ := m.ChatTemplate("")
			tpl, err := jinja.Compile(src)
			if err == nil {
				bos, eos := m.SpecialText()
				if !model.SystemRendered(tpl, map[string]any{"bos_token": bos, "eos_token": eos}, model.WantsContentParts(tpl)) {
					t.Logf("%s: no system role; the system prompt is folded into the first message", name)
				}
			}
		}()
	}
	if checked == 0 {
		testmodels.Missing(t, "no chat model under %s: this gate checked nothing", testmodels.Dir())
	}
	t.Logf("%d chat model(s) checked", checked)
}
