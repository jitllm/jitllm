//go:build linux

package engine

import (
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestChatCapableAgreesWithChatPrompt: ChatCapable must answer without failing
// on both kinds of model, so a front end with a chat toggle can ask rather than
// render and read the error once per message.
func TestChatCapableAgreesWithChatPrompt(t *testing.T) {
	for _, tc := range []struct {
		name string
		path string
	}{
		{"instruct", testmodels.Path("SmolLM2-360M-Instruct-Q8_0.jlm")},
		{"base", testmodels.Path("stories260K.jlm")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, err := model.Open(tc.path)
			if err != nil {
				t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
			}
			defer m.Close()

			got := m.ChatCapable()
			_, perr := m.ChatPrompt([]model.ChatMessage{{Role: "user", Content: "hi"}}, true)
			// The two must agree: the cheap question and the expensive one.
			if got != (perr == nil) {
				t.Errorf("ChatCapable=%v but ChatPrompt err=%v", got, perr)
			}
			t.Logf("%s: ChatCapable=%v", tc.name, got)
		})
	}
}
