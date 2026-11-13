//go:build linux

package engine

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/internal/testmodels/fresh"
)

// A probe, not a gate: it prints what a multi-turn conversation actually
// reuses so the number can be read instead of assumed.
func TestReuseProbe(t *testing.T) {
	if os.Getenv("JITLLM_UI_REUSE") == "" {
		t.Skip("set JITLLM_UI_REUSE=1 to measure prefix reuse across turns")
	}
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	sh := newTestShell()
	st := sh.Store
	e := New(sh, sh.state())
	defer e.Close()
	instruct := testmodels.Path("SmolLM2-360M-Instruct-Q8_0.jlm")
	fresh.Must(t, instruct)
	if _, err := os.Stat(instruct); err != nil {
		t.Skipf("no instruct model: %v", err)
	}
	e.Load(instruct)
	pump(t, sh, 90*time.Second, "load", func() bool { return st.Loaded.Get() })

	for turn := 1; turn <= 4; turn++ {
		st.AppendTurn(session.Turn{Role: session.RoleUser,
			Text: fmt.Sprintf("Tell me story number %d about a girl named Lily", turn)})
		reply := st.AppendTurn(session.Turn{Role: session.RoleAssistant})
		hist := st.AllTurns()
		msgs := make([]session.ChatMessage, 0, len(hist))
		for _, h := range hist {
			role := "user"
			if h.Role == session.RoleAssistant {
				role = "assistant"
			}
			if h.Text != "" {
				msgs = append(msgs, session.ChatMessage{Role: role, Content: h.Text})
			}
		}
		e.Send(session.ChatRequest{Chat: true, Messages: msgs,
			Prompt: msgs[len(msgs)-1].Content, Reply: reply, MaxTokens: 24})
		pump(t, sh, 90*time.Second, "turn", func() bool {
			return !st.Busy.Get() && st.Turn(reply).Text != ""
		})
		t.Logf("turn %d: prompt %d tok, reused %d, %d failure(s)",
			turn, e.lastPrompt(), e.lastReused(), e.lastFailures())
	}
}
