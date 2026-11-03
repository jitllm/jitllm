//go:build linux

package engine

import (
	"os"
	"testing"
	"time"

	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// Loading a second model leaves the first loaded and usable: several models
// at once is the design, and only Unload removes one.
func TestLoadingASecondModelLeavesTheFirstLoaded(t *testing.T) {
	a, b := modelPath(), testmodels.Path("stories15M-q8_0.jlm")
	for _, p := range []string{a, b} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
		}
	}
	sh := newTestShell()
	st := sh.Store
	e := New(sh, sh.state())
	defer e.Close()

	loaded := func(want ...string) func() bool {
		return func() bool {
			for _, p := range want {
				if session.LoadState(st.Models.Get(), p) != session.Loaded {
					return false
				}
			}
			return !st.Loading()
		}
	}
	e.Load(a)
	pump(t, sh, 90*time.Second, "the first model to load", loaded(a))
	e.Load(b)
	pump(t, sh, 90*time.Second, "the second model to load", loaded(b))
	if got := session.LoadState(st.Models.Get(), a); got != session.Loaded {
		t.Fatalf("loading %s left %s %q", b, a, got)
	}

	// The first is usable as it stands: the chat talks to it without loading it
	// again, and the second stays loaded beside it.
	e.Use(a)
	pump(t, sh, 30*time.Second, "the chat to move to the first", func() bool { return st.Active.Get() == a && !st.Loading() })
	reply := st.AppendTurn(session.Turn{Role: session.RoleAssistant})
	e.Send(session.ChatRequest{Prompt: "Once upon a time", Reply: reply, MaxTokens: 4})
	pump(t, sh, 60*time.Second, "the first model to answer", func() bool {
		return !st.Busy.Get() && st.Turn(reply).Text != ""
	})
	if n := len(st.Models.Get()); n != 2 {
		t.Fatalf("%d models loaded after chatting with the first, want 2", n)
	}

	// Unload is the one way out, and it is explicit.
	e.Unload(b)
	pump(t, sh, 10*time.Second, "the unload", func() bool { return len(st.Models.Get()) == 1 })
	if session.LoadState(st.Models.Get(), a) != session.Loaded {
		t.Error("unloading the second took the first with it")
	}
}
