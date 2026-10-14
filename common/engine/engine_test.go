//go:build linux

package engine

import (
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/internal/testmodels/fresh"
)

// modelPath is a real container, because the one thing this gate exists to
// prove is that the app can drive the real engine.
var modelPath = testmodels.Path("stories260K.jlm")

// pump drains the UI queue the way gogpu's OnUpdate does, until want is true.
//
// The test has to act as the UI goroutine: the engine writes through
// Shell.Post and nothing lands until somebody drains, so polling signals
// without draining would miss an engine posting into a queue nobody reads.
func pump(t *testing.T, sh *testShell, d time.Duration, what string, want func() bool) {
	t.Helper()
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		sh.DrainQueue(0)
		if want() {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	sh.DrainQueue(0)
	if !want() {
		t.Fatalf("timed out after %v waiting for %s (status %q)", d, what, sh.Store.Status.Get())
	}
}

// The app must actually load a model and generate text through the engine.
//
// It guards the app reaching the engine at all: the window's screen.Engine must be
// attached, or clicking a model says "No engine wired".
func TestTheEngineLoadsAModelAndGenerates(t *testing.T) {
	fresh.Must(t, modelPath)
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	sh := newTestShell()
	st := sh.Store

	e := New(sh, sh.state())
	defer e.Close()

	e.Load(modelPath)
	if l, ok := st.LoadingOf(modelPath); !ok || l.Stage == "" {
		t.Fatalf("a load in progress is not shown: %+v", l)
	}
	pump(t, sh, 90*time.Second, "the model to load", func() bool { return st.Loaded.Get() })
	pump(t, sh, 10*time.Second, "the load indicator to clear", func() bool { return !st.Loading() })

	if got := st.ModelSummary.Get(); got == "" {
		t.Error("loaded with an empty summary: the header was never read")
	} else {
		t.Logf("loaded: %s", got)
	}

	reply := st.AppendTurn(session.Turn{Role: session.RoleAssistant})

	// Count repaints, not only store contents: Store.SetTurn deliberately does
	// not publish, so an engine that forgets Store.Touch leaves the bubble empty.
	// Watch the revision signal rather than the turn count, since a finished
	// reply does not change the count; Revision drives InvalidateData.
	var repaints atomic.Int64
	unsub := st.Revision.SubscribeForever(func(int) { repaints.Add(1) })
	defer unsub()

	// Collect the partials: arriving a token at a time is the feature, and an
	// engine that published the reply once would pass everything else here.
	var partialsMu sync.Mutex
	var partials []string
	unsubStream := st.Stream.SubscribeForever(func(v string) {
		partialsMu.Lock()
		partials = append(partials, v)
		partialsMu.Unlock()
	})
	defer unsubStream()

	e.Send(session.ChatRequest{
		Prompt: "Once upon a time", Reply: reply, MaxTokens: 16,
	})
	pump(t, sh, 90*time.Second, "generation to finish", func() bool {
		return !st.Busy.Get() && st.Turn(reply).Text != ""
	})

	turn := st.Turn(reply)
	if turn.Text == "" {
		t.Fatal("the assistant turn is empty: Send reached no engine, which is " +
			"the state the first build shipped in")
	}
	if turn.Tokens == 0 {
		t.Error("the turn reports 0 tokens: the decode loop never ran")
	}
	if turn.TokPerSec <= 0 {
		t.Error("the turn reports no rate: the decode was never timed")
	}
	if st.DecodeTokS.Get() <= 0 {
		t.Error("Store.DecodeTokS is zero: the stats strip would stay blank")
	}
	// Per token, not once: listview caches a row's measured height, so only a
	// re-measure per token lets a growing bubble grow.
	if repaints.Load() < int64(turn.Tokens/2) {
		t.Errorf("the transcript was marked dirty %d time(s) for %d token(s): "+
			"the bubble is not re-measured as the reply grows, so it keeps the "+
			"height it had when it was empty", repaints.Load(), turn.Tokens)
	}
	partialsMu.Lock()
	seen := append([]string(nil), partials...)
	partialsMu.Unlock()
	early := 0
	for _, p := range seen {
		if p != "" && len(p) < len(turn.Text) && strings.HasPrefix(turn.Text, p) {
			early++
		}
	}
	// The bar is per token, not "at least one", so a throttle would fail it. Half
	// the tokens, because Shell.Post may drop a few frames on a full queue.
	if early < turn.Tokens/2 {
		t.Errorf("only %d partial(s) for %d token(s) (%d Stream write(s)): the "+
			"reply is not arriving a token at a time",
			early, turn.Tokens, len(seen))
	}
	t.Logf("%d partial(s) for %d token(s)", early, turn.Tokens)

	// The live bubble reads Store.Stream, so the finished reply has to stay
	// there -- clearing it blanks the answer the instant it completes.
	if st.Stream.Get() != turn.Text {
		t.Errorf("Store.Stream is %q and the turn is %q: the live bubble would "+
			"show the wrong text, or none", trunc(st.Stream.Get(), 40),
			trunc(turn.Text, 40))
	}
	t.Logf("generated %d token(s) at %.1f tok/s: %q",
		turn.Tokens, turn.TokPerSec, trunc(turn.Text, 60))

	// The text must be decoded, not concatenated raw pieces, or SentencePiece
	// markers ("▁") show literally.
	if strings.ContainsRune(turn.Text, '\u2581') {
		t.Errorf("the turn carries SentencePiece markers: %q -- the engine is "+
			"concatenating Vocab.Text pieces instead of decoding the ids",
			trunc(turn.Text, 60))
	}
	if !utf8.ValidString(turn.Text) {
		t.Error("the turn is not valid UTF-8: a rune was split across tokens")
	}

	// Busy and Streaming must both be released, or the Send button stays
	// disabled for ever and the app is one generation long.
	if st.Busy.Get() || st.Streaming.Get() {
		t.Errorf("Busy=%v Streaming=%v after the turn finished: the UI stays locked",
			st.Busy.Get(), st.Streaming.Get())
	}
}

// Stop must cancel a generation and keep the model loaded.
func TestStopCancelsAndKeepsTheModel(t *testing.T) {
	fresh.Must(t, modelPath)
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	sh := newTestShell()
	st := sh.Store
	e := New(sh, sh.state())
	defer e.Close()

	e.Load(modelPath)
	pump(t, sh, 90*time.Second, "the model to load", func() bool { return st.Loaded.Get() })

	reply := st.AppendTurn(session.Turn{Role: session.RoleAssistant})
	e.Send(session.ChatRequest{Prompt: "Once upon a time", Reply: reply, MaxTokens: 4096})
	// Let it get going, then stop it.
	pump(t, sh, 30*time.Second, "generation to start", func() bool { return st.Busy.Get() })
	e.Stop()
	pump(t, sh, 60*time.Second, "generation to stop", func() bool { return !st.Busy.Get() })

	if n := st.Turn(reply).Tokens; n >= 4096 {
		t.Errorf("Stop did not cancel: %d token(s) of a 4096 cap", n)
	}
	if !st.Loaded.Get() {
		t.Error("Stop unloaded the model: it must cancel the generation only")
	}
	// And a second Send must still work.
	r2 := st.AppendTurn(session.Turn{Role: session.RoleAssistant})
	e.Send(session.ChatRequest{Prompt: "The end", Reply: r2, MaxTokens: 8})
	pump(t, sh, 60*time.Second, "the second generation", func() bool {
		return !st.Busy.Get() && st.Turn(r2).Text != ""
	})
	if st.Turn(r2).Text == "" {
		t.Error("the model could not generate again after Stop")
	}
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// A second turn that shares a prefix must reuse it: a chat turn is its whole
// history, so without reuse the cost grows with the conversation. The
// assertion is the reuse count (KVRestored), not the time, which on a tiny
// model measures the harness.
func TestASecondTurnReusesTheSharedPrefix(t *testing.T) {
	fresh.Must(t, modelPath)
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	// A private cache dir, so the gate cannot pass on a page another run left.
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	sh := newTestShell()
	st := sh.Store
	e := New(sh, sh.state())
	defer e.Close()

	e.Load(modelPath)
	pump(t, sh, 90*time.Second, "the model to load", func() bool { return st.Loaded.Get() })

	send := func(prompt string) int {
		reply := st.AppendTurn(session.Turn{Role: session.RoleAssistant})
		e.Send(session.ChatRequest{Prompt: prompt, Reply: reply, MaxTokens: 4})
		pump(t, sh, 90*time.Second, "generation", func() bool {
			return !st.Busy.Get() && st.Turn(reply).Text != ""
		})
		return reply
	}

	const shared = "Once upon a time there was a little girl named Lily who"
	send(shared)
	first := e.lastReused()

	// The same prefix again, extended. Everything up to the extension is a hit.
	send(shared + " lived in a small house")
	second := e.lastReused()

	t.Logf("reused %d position(s) on the first turn, %d on the second", first, second)
	if second <= 0 {
		t.Errorf("the second turn reused %d position(s): the prefix cache is not "+
			"working, so every chat turn re-prefills the whole conversation",
			second)
	}
	if second <= first {
		t.Errorf("reuse did not grow across turns (%d then %d): the store is "+
			"not being written, or the namespace changes per session",
			first, second)
	}
	if f := e.lastFailures(); f > 0 {
		t.Errorf("%d store write failure(s): the cache is accepting nothing", f)
	}
}
