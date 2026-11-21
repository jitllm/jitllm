package main

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestSystemIsRefusedWhereTheTemplateHasNoSystemRole: -system to a model whose
// chat template has no system role is refused, naming the flag and the
// missing role, and the same chat without -system answers. The template is
// Mistral-7B-Instruct-v0.3's GGUF's, carried on stories260K.
func TestSystemIsRefusedWhereTheTemplateHasNoSystemRole(t *testing.T) {
	src := testmodels.Path("stories260K.gguf")
	if _, err := os.Stat(src); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v -- this gate proved nothing", err)
	}
	dir := t.TempDir()
	tpl := filepath.Join(dir, "nosystem.jinja")
	const noSystem = `{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}`
	if err := os.WriteFile(tpl, []byte(noSystem), 0o644); err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(dir, "nosystem.jlm")
	if _, err := convert.FromGGUF(src, dst, jlm.Fingerprint{Host: "test"}, convert.WithChatTemplate(tpl)); err != nil {
		t.Fatal(err)
	}
	chat := func(system string) error {
		_, _, err := capture(t, func() error {
			return run(dst, "hi", 2, 0, "cpu", -1, 0, 1<<30, false, false, false, false, &engine.Sampler{},
				mediaSpec{}, nil, &chatSpec{system: system}, "", 0, 0, nil, nil)
		})
		return err
	}
	err := chat("be terse")
	var sr *model.SystemRoleError
	if !errors.As(err, &sr) || !strings.HasPrefix(err.Error(), "-system: ") || !strings.Contains(err.Error(), "no system role") {
		t.Fatalf("-system to a template with no system role: %v, want it refused naming -system and the role", err)
	}
	if err := chat(""); err != nil {
		t.Fatalf("-chat without -system: %v", err)
	}
}
