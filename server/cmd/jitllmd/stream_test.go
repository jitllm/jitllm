package main

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jitllm/jitllm/server"
)

// TestRunPrintsEachTokenBeforeTheServerProducesTheNext is the main gate for
// this client. It is a rendezvous: the server will not produce token N+1 until
// the client has written token N to stdout (via notifyWriter.Write, not the
// wire), so a buffering client deadlocks, which the gate turns into a named
// failure.
//
// VIOLATION SIGNATURE. Replace the per-token `emit(c.out, t)` in cmdRun with an
// accumulator printed after the loop and this fails with
//
//	the client did not print token N before the server produced token N+1:
//	it accumulated the stream instead of writing it through
func TestRunPrintsEachTokenBeforeTheServerProducesTheNext(t *testing.T) {
	script := []string{"The ", "capital", " is ", "Paris", "."}
	w := newNotifyWriter(len(script))

	var buffered atomic.Bool
	s := &scripted{
		tokens: script,
		reason: server.FinishEOS,
		before: func(i int) {
			if i == 0 || buffered.Load() {
				return
			}
			select {
			case <-w.ch:
				// Token i-1 reached stdout. Produce token i.
			case <-time.After(3 * time.Second):
				// The fallback makes this fail instead of hang; the flag
				// short-circuits the remaining tokens.
				buffered.Store(true)
			}
		},
	}
	addr, pc := scriptServer(t, s)

	var errOut syncBuffer
	err := runVerb(t, cmdRun, cli{out: w, err: &errOut},
		[]string{"-addr", addr, "-model", "m-1", "hello"}, 30*time.Second)
	if err != nil {
		t.Fatalf("run: %v\nstderr:\n%s", err, errOut.String())
	}
	if buffered.Load() {
		t.Fatal("the client did not print token N before the server produced token N+1: " +
			"it accumulated the stream instead of writing it through")
	}
	if got, want := w.String(), strings.Join(script, ""); got != want {
		t.Fatalf("stdout %q, want %q", got, want)
	}

	// The selection check, restated here so a failure names both facts at once.
	// It is asserted on its own below, where the rendezvous is not in play --
	// see TestRunUsesTheStreamingRpcAndNotTheAccumulatingOne for why that
	// separation matters.
	if n := pc.get("InferenceService/Generate"); n != 1 {
		t.Fatalf("Generate was called %d time(s), want 1; the paths hit were %v", n, pc.paths())
	}
	if n := pc.get("InferenceService/Complete"); n != 0 {
		t.Fatalf("the client called the non-streaming Complete %d time(s)", n)
	}
}

// TestRunUsesTheStreamingRpcAndNotTheAccumulatingOne is the selection check.
// Generate and Complete produce the same text; the rendezvous above catches a
// swap only as a timeout, while this reads the invoked procedure off the
// server's request log so the failure names the RPC.
//
// VIOLATION SIGNATURE. Swap cn.Inference.Generate for cn.Inference.Complete in
// cmdRun and this fails with
//
//	the client called the non-streaming Complete 1 time(s); Generate 0.
//	Complete produces byte-identical output and is not a stream
func TestRunUsesTheStreamingRpcAndNotTheAccumulatingOne(t *testing.T) {
	s := &scripted{tokens: []string{"a", "b"}, reason: server.FinishEOS}
	addr, pc := scriptServer(t, s)

	var o, e syncBuffer
	if err := runVerb(t, cmdRun, cli{out: &o, err: &e},
		[]string{"-addr", addr, "-model", "m-1", "hi"}, 20*time.Second); err != nil {
		t.Fatalf("run: %v", err)
	}
	if o.String() != "ab" {
		t.Fatalf("stdout %q, want %q", o.String(), "ab")
	}
	gen, comp := pc.get("InferenceService/Generate"), pc.get("InferenceService/Complete")
	if comp != 0 || gen != 1 {
		t.Fatalf("the client called the non-streaming Complete %d time(s); Generate %d. "+
			"Complete produces byte-identical output and is not a stream (paths: %v)",
			comp, gen, pc.paths())
	}
}

// TestRunReportsTheQueueAndTheBytesBehindTheRate: the queue wait and
// placement, and the rate with its bytes per token, go to stderr so stdout
// stays exactly the completion.
//
// VIOLATION SIGNATURE. Drop the queued line from printStarted and this fails
// with
//
//	stderr does not report the queue wait
//
// Print the rate without the byte count and it fails with
//
//	stderr quotes a rate with no bytes/token beside it
func TestRunReportsTheQueueAndTheBytesBehindTheRate(t *testing.T) {
	s := &scripted{tokens: []string{"Paris"}, reason: server.FinishEOS}
	addr, _ := scriptServer(t, s)

	var o, e syncBuffer
	if err := runVerb(t, cmdRun, cli{out: &o, err: &e},
		[]string{"-addr", addr, "-model", "m-1", "hi"}, 20*time.Second); err != nil {
		t.Fatalf("run: %v", err)
	}
	so, se := o.String(), e.String()

	if so != "Paris" {
		t.Fatalf("stdout is %q, want exactly the completion %q -- the telemetry belongs on stderr "+
			"so stdout stays pipeable", so, "Paris")
	}
	if !strings.Contains(se, "queued") || !strings.Contains(se, "1.5s") {
		t.Fatalf("stderr does not report the queue wait:\n%s", se)
	}
	if !strings.Contains(se, "behind 2 request(s)") {
		t.Fatalf("stderr does not report the queue DEPTH on entry:\n%s", se)
	}
	if !strings.Contains(se, "22 on cuda:0") || !strings.Contains(se, "10 on cpu") {
		t.Fatalf("stderr does not report the cards' and the CPU's block split:\n%s", se)
	}
	if !strings.Contains(se, "33.30 tok/s") {
		t.Fatalf("stderr does not report the rate:\n%s", se)
	}
	if !strings.Contains(se, "816,010,912 B/token") {
		t.Fatalf("stderr quotes a rate with no bytes/token beside it, which is unfalsifiable:\n%s", se)
	}
	// 33.3 tok/s x 816,010,912 B = 27.17 GB/s, printed with both factors.
	if !strings.Contains(se, "27.17 GB/s") {
		t.Fatalf("stderr does not carry the effective GB/s derived from the rate and the bytes:\n%s", se)
	}
	// The backend is quoted with the number.
	if !strings.Contains(se, "[cuda:0 + host]") {
		t.Fatalf("the decode rate is not quoted with where the blocks ran:\n%s", se)
	}
}

// TestRunReportsABytelessRateAsUncheckableRatherThanAsZero: an unreported
// byte count is not "0.00 GB/s".
func TestRunReportsABytelessRateAsUncheckableRatherThanAsZero(t *testing.T) {
	addr, _ := scriptServer(t, &noBytesBackend{})
	var o, e syncBuffer
	if err := runVerb(t, cmdRun, cli{out: &o, err: &e},
		[]string{"-addr", addr, "-model", "m-1", "hi"}, 20*time.Second); err != nil {
		t.Fatalf("run: %v", err)
	}
	se := e.String()
	if strings.Contains(se, "0.00 GB/s") {
		t.Fatalf("a missing byte count was printed as a computed zero:\n%s", se)
	}
	if !strings.Contains(se, "bytes/token not reported") {
		t.Fatalf("stderr does not say the rate is uncheckable:\n%s", se)
	}
}

// noBytesBackend is the scripted backend with bytes_per_token left at zero.
type noBytesBackend struct{ scripted }

func (n *noBytesBackend) Generate(ctx context.Context, o server.GenerateOptions, emit func(server.Event) error) error {
	if err := emit(server.Event{Kind: server.EventStarted, Started: &server.Started{
		ModelID: o.ModelID, SessionID: "s", PromptTokens: 1,
	}}); err != nil {
		return err
	}
	if err := emit(server.Event{Kind: server.EventToken, Token: &server.Token{ID: 1, Text: "x"}}); err != nil {
		return err
	}
	return emit(server.Event{Kind: server.EventFinished, Finished: &server.Finished{
		Reason: server.FinishEOS, CompletionTokens: 1, TokensPerSecond: 12.5, BytesPerToken: 0,
	}})
}

// TestRunSaysSoWhenTheStreamIsCutShort: a truncated stream and a short answer
// look identical on stdout, so a missing finished event is an error.
func TestRunSaysSoWhenTheStreamIsCutShort(t *testing.T) {
	s := &cutShortBackend{}
	addr, _ := scriptServer(t, s)
	var o, e syncBuffer
	err := runVerb(t, cmdRun, cli{out: &o, err: &e},
		[]string{"-addr", addr, "-model", "m-1", "hi"}, 20*time.Second)
	if err == nil {
		t.Fatal("a stream that ended without a finished event was reported as a success")
	}
	if !strings.Contains(err.Error(), "cut short") {
		t.Fatalf("the error does not name the truncation: %v", err)
	}
	if o.String() != "half" {
		t.Fatalf("the tokens that DID arrive were dropped: stdout %q", o.String())
	}
}

type cutShortBackend struct{ scripted }

func (c *cutShortBackend) Generate(ctx context.Context, o server.GenerateOptions, emit func(server.Event) error) error {
	if err := emit(server.Event{Kind: server.EventStarted, Started: &server.Started{ModelID: o.ModelID}}); err != nil {
		return err
	}
	return emit(server.Event{Kind: server.EventToken, Token: &server.Token{ID: 1, Text: "half"}})
}
