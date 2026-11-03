package engine

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/common/crash"
)

// A panic in a command is reported and ends the command, not the worker: the
// next command runs and Close returns. Against the violation (run calling fn
// bare) the test process dies on the panic.
func TestAPanickingCommandIsReportedAndTheWorkerLives(t *testing.T) {
	sh := newTestShell()
	got := make(chan crash.Report, 1)
	sh.Crashes().OnReport(func(r crash.Report) { got <- r })

	e := New(sh, sh.state())
	sh.Store.Streaming.Set(true)
	if !e.Run(func() { panic("a command fell over") }) {
		t.Fatal("the queue refused the command")
	}
	ran := make(chan struct{})
	if !e.Run(func() { close(ran) }) {
		t.Fatal("the queue refused the second command")
	}
	<-ran
	r := <-got
	if !strings.Contains(r.Text, "a command fell over") || !strings.Contains(r.Text, "crash_test.go") {
		t.Errorf("report lacks the panic or its stack:\n%s", r.Text)
	}
	e.Close()
	sh.DrainQueue(0)
	if sh.Store.Streaming.Get() {
		t.Error("a reply cut off by a panic is still streaming")
	}
	e.Close()
}
