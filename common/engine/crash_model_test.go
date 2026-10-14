//go:build linux

package engine

import (
	"os"
	"testing"
	"time"

	"github.com/jitllm/jitllm/common/crash"
	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/internal/testmodels/fresh"
)

// A command that panics mid-reply, counted the way Send counts it, leaves
// the app consistent: not busy, not streaming, no session left at whatever
// position the step stopped, nothing shown as loaded -- and choosing the
// model again chats. Against the violation (afterPanic not closing the
// session) the model is still shown loaded on the broken session.
func TestAPanicMidReplyLeavesTheAppConsistent(t *testing.T) {
	fresh.Must(t, modelPath)
	if _, err := os.Stat(modelPath); err != nil {
		t.Fatalf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proves nothing without it", err)
	}
	crash.OnReport(func(crash.Report) {})
	defer crash.OnReport(nil)
	sh := newTestShell()
	st := sh.Store
	e := New(sh, sh.state())
	defer e.Close()

	e.Load(modelPath)
	pump(t, sh, 90*time.Second, "the model to load", func() bool { return st.Loaded.Get() })

	// Send's own bookkeeping around a command that falls over.
	e.cancel.Store(false)
	e.busyBegin()
	st.Streaming.Set(true)
	e.post("send", func() {
		defer e.busyEnd()
		e.st.Streaming.Set(true)
		panic("a step fell over")
	})
	pump(t, sh, 30*time.Second, "the panic to be dealt with", func() bool {
		return !st.Busy.Get() && !st.Streaming.Get() && !st.Loaded.Get()
	})
	gone := make(chan bool, 1)
	e.Run(func() { gone <- e.sess == nil && e.active == nil && e.loadingPath == "" })
	if !<-gone {
		t.Fatal("the worker kept the session the panic interrupted")
	}
	if len(st.Models.Get()) != 1 {
		t.Errorf("%d model(s) listed, want the one still open", len(st.Models.Get()))
	}

	e.Use(modelPath)
	pump(t, sh, 60*time.Second, "the model to be chosen again", func() bool { return st.Loaded.Get() && !st.Busy.Get() })
	r := st.AppendTurn(session.Turn{Role: session.RoleAssistant})
	e.Send(session.ChatRequest{Prompt: "Once upon a time", Reply: r, MaxTokens: 8})
	pump(t, sh, 60*time.Second, "a reply after the panic", func() bool {
		return !st.Busy.Get() && st.Turn(r).Text != ""
	})
}
