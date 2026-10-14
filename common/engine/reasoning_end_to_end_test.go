//go:build linux

package engine

import (
	"os"
	"testing"
	"time"

	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/internal/testmodels/fresh"
)

// reasoningModel is a container whose model reasons, because that is the one
// thing this gate exists to prove and no fixture can supply it.
var reasoningModel = testmodels.Path("Qwen3-1.7B-Q4_K_M.jlm")

// A reasoning model's reasoning must reach the store.
//
// Every unit in the path (SplitThinking) can be right while the wiring in
// front of it drops the reasoning, so this runs the real chat path end to end.
// It asserts both the streamed half (StreamThink) and the committed one
// (Turn.Think), since they come from different call sites and are what the
// transcript renders.
func TestAReasoningModelsReasoningReachesTheStore(t *testing.T) {
	fresh.Must(t, reasoningModel)
	if _, err := os.Stat(reasoningModel); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	sh := newTestShell()
	st := sh.Store

	e := New(sh, sh.state())
	defer e.Close()

	e.Load(reasoningModel)
	pump(t, sh, 180*time.Second, "the model to load", func() bool { return st.Loaded.Get() })

	// Whether the store saw reasoning while it was arriving, not only after.
	var sawStreamed string
	unsub := st.StreamThink.SubscribeForever(func(v string) {
		if v != "" && sawStreamed == "" {
			sawStreamed = v
		}
	})
	defer unsub()

	reply := st.AppendTurn(session.Turn{Role: session.RoleAssistant})
	e.Send(session.ChatRequest{
		Chat:     true,
		Prompt:   "What is 17 times 23? Think it through.",
		Reply:    reply,
		Messages: []session.ChatMessage{{Role: "user", Content: "What is 17 times 23? Think it through."}},
		// Enough tokens to get inside the block, not the whole answer: an
		// unclosed block is all reasoning, so this does not need the model to finish.
		MaxTokens: 96,
	})
	pump(t, sh, 300*time.Second, "generation to finish", func() bool { return !st.Busy.Get() })

	turn := st.Turn(reply)
	t.Logf("think (%d bytes): %.200q", len(turn.Think), turn.Think)
	t.Logf("text  (%d bytes): %.200q", len(turn.Text), turn.Text)

	if turn.Think == "" {
		t.Errorf("Turn.Think is empty after %d token(s): the finished bubble has no "+
			"reasoning to disclose.\nanswer was %.300q", turn.Tokens, turn.Text)
	}
	// The disclosure must have opened itself on the first reasoning token, or a
	// long thinking phase looks like a frozen window. It folds on the first
	// answer token, once each, so a click in between sticks.
	if !st.ThinkOpen(reply).Get() {
		t.Errorf("the reasoning arrived and the disclosure stayed shut: the " +
			"window shows a collapsed \"thinking\" row and nothing else")
	}
	if sawStreamed == "" {
		t.Errorf("Store.StreamThink was never non-empty: the live bubble's disclosure " +
			"stays hidden for the whole generation, which is what \"it feels stuck\" is")
	}
	// And the reasoning must not also be in the answer.
	if turn.Think != "" && turn.Text != "" && len(turn.Text) >= 24 {
		if head := turn.Think[:min(24, len(turn.Think))]; contains(turn.Text, head) {
			t.Errorf("the answer repeats the reasoning: %.120q", turn.Text)
		}
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
