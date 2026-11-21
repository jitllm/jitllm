package model

import (
	"errors"
	"strings"
	"testing"
)

// mistralGGUFTemplate is the template Mistral-7B-Instruct-v0.3's GGUF carries:
// no system role, and a raise on anything that does not alternate
// user/assistant.
const mistralGGUFTemplate = `{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}`

// droppingTemplate renders user and assistant turns and nothing of any other,
// so a system message would vanish from the prompt without an error.
const droppingTemplate = `{% for m in messages %}{% if m.role == 'user' %}[U]{{ m.content }}{% elif m.role == 'assistant' %}[A]{{ m.content }}{% endif %}{% endfor %}{% if add_generation_prompt %}[A]{% endif %}`

const chatMLTemplate = `{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}`

// TestASystemPromptIsRefusedWhereTheTemplateHasNoSystemRole: a system message
// to a template with no system role is refused with a *SystemRoleError naming
// the message, whether the template raises on the role or leaves its text out,
// and wherever in the conversation it stands; the conversation without it
// renders, and a template with the role renders it as its own turn. The
// violation, the refusal removed, renders the dropping template's prompt
// without the system text and gives no error.
func TestASystemPromptIsRefusedWhereTheTemplateHasNoSystemRole(t *testing.T) {
	render := func(src string, msgs []ChatMessage) (string, error) {
		return RenderChatTemplate(src, "<s>", "</s>", nil, msgs, nil, true)
	}
	sys := []ChatMessage{{Role: "system", Content: "be terse"}, {Role: "user", Content: "hi"}}
	late := []ChatMessage{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "hello"},
		{Role: "system", Content: "be terse"}, {Role: "user", Content: "again"}}
	refused := 0
	for _, tc := range []struct{ name, src string }{{"raises", mistralGGUFTemplate}, {"drops", droppingTemplate}} {
		for _, c := range []struct {
			msgs []ChatMessage
			at   int
		}{{sys, 0}, {late, 2}, {sys[:1], 0}} {
			out, err := render(tc.src, c.msgs)
			var sr *SystemRoleError
			if !errors.As(err, &sr) {
				t.Errorf("%s, %d messages: %q (%v), want a *SystemRoleError", tc.name, len(c.msgs), out, err)
				continue
			}
			if sr.Message != c.at || out != "" || !strings.Contains(err.Error(), "no system role") {
				t.Errorf("%s, %d messages: message %d, %q, output %q; want message %d named and no prompt",
					tc.name, len(c.msgs), sr.Message, err, out, c.at)
			}
			refused++
		}
		if out, err := render(tc.src, sys[1:]); err != nil || !strings.Contains(out, "hi") {
			t.Errorf("%s: the conversation without its system message: %q (%v)", tc.name, out, err)
		}
	}
	if refused != 6 {
		t.Fatalf("%d of 6 refused", refused)
	}
	// A template that writes no user message in this context either is not
	// one without the role: its own error stands.
	needsTools := `{% for t in tools %}{{ t }}{% endfor %}{% for m in messages %}{{ m.content }}{% endfor %}`
	if out, err := render(needsTools, sys); err == nil || errors.As(err, new(*SystemRoleError)) {
		t.Errorf("a template that renders nothing without tools: %q (%v), want its own error", out, err)
	}
	out, err := render(chatMLTemplate, sys)
	if err != nil || !strings.Contains(out, "<|im_start|>system\nbe terse<|im_end|>") {
		t.Errorf("a template with a system role: %q (%v)", out, err)
	}
	if out, err := render(chatMLTemplate, late); err != nil || !strings.Contains(out, "<|im_start|>system\nbe terse") {
		t.Errorf("a template with a system role, the message later: %q (%v)", out, err)
	}
}
