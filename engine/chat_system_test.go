package engine

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// Every chat model takes a system prompt or refuses it by name: a template with
// the role puts the text in the prompt, and one without it gives a
// *model.SystemRoleError while the conversation without the system message
// renders. The log names the templates that refuse.
func TestEveryChatModelTakesOrRefusesASystemPrompt(t *testing.T) {
	checked, refused := 0, 0
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
			msgs := []model.ChatMessage{{Role: "system", Content: "be terse"}, {Role: "user", Content: "hello"}}
			out, err := m.ChatPrompt(msgs, true)
			var sr *model.SystemRoleError
			switch {
			case errors.As(err, &sr):
				refused++
				t.Logf("%s: no system role; a system prompt is refused", name)
				if out, err := m.ChatPrompt(msgs[1:], true); err != nil || !strings.Contains(out, "hello") {
					t.Errorf("%s: the conversation without its system message: %q (%v)", name, out, err)
				}
			case err != nil:
				t.Errorf("%s: %v", name, err)
			case !strings.Contains(out, "be terse") || !strings.Contains(out, "hello"):
				t.Errorf("%s: the system prompt or the message is missing:\n%q", name, out)
			}
		}()
	}
	if checked == 0 {
		testmodels.Missing(t, "no chat model under %s: this gate checked nothing", testmodels.Dir())
	}
	t.Logf("%d chat model(s) checked, %d refusing a system prompt", checked, refused)
}
