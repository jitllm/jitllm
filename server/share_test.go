package server

import (
	"context"
	"testing"

	"github.com/jitllm/jitllm/engine"
)

// Two models loaded into one engine divide its host budget: neither may
// believe it holds the machine. A pin comes off the top, priority gives the
// favoured model all but an eighth, and an unload hands its share back.
func TestLoadedModelsDivideTheHostBudget(t *testing.T) {
	e, a, _ := loadedEngine(t, smallModel, "a", LoadOptions{})
	b, err := e.LoadModel(LoadOptions{Path: a.path, ModelID: "b"})
	if err != nil {
		t.Fatal(err)
	}
	total := e.hostTotal()
	want := func(what string, wa, wb uint64) {
		t.Helper()
		if a.Budget() != wa || b.Budget() != wb {
			t.Fatalf("%s: a=%d b=%d, want %d and %d of %d", what, a.Budget(), b.Budget(), wa, wb, total)
		}
	}
	want("two models", total/2, total/2)

	if err := e.Pin("a", total/4); err != nil {
		t.Fatal(err)
	}
	want("a pinned", total/4, total-total/4)
	if !a.Pinned() || b.Pinned() {
		t.Fatal("the pin is on the wrong model")
	}
	if err := e.Pin("a", 0); err != nil {
		t.Fatal(err)
	}
	want("pin released", total/2, total/2)

	e.Favor("b")
	e.SetPriority(true)
	bg := total / backgroundShare
	want("b favoured", bg, total-bg)

	if _, err := e.UnloadModel("b", true); err != nil {
		t.Fatal(err)
	}
	if a.Budget() != total {
		t.Fatalf("after b unloaded a holds %d, want all %d", a.Budget(), total)
	}
}

// A session with a prompt store resumes a prompt it has seen: the second
// generate of the same prompt restores its prefix instead of prefilling it.
func TestASessionWithAStoreReusesItsPrompt(t *testing.T) {
	e, _, _ := loadedEngine(t, smallModel, "small", LoadOptions{})
	s, err := e.CreateSession(SessionOptions{ModelID: "small", KVStore: engine.NewMemStore(), CacheKey: "test"})
	if err != nil {
		t.Fatal(err)
	}
	run := func() int {
		o := GenerateOptions{SessionID: s.ID(), Prompt: Prompt{Kind: PromptText, Text: story + story + story}, MaxTokens: 2}
		if err := e.Generate(context.Background(), o, func(Event) error { return nil }); err != nil {
			t.Fatal(err)
		}
		var n int
		s.Inspect(func(st *engine.Instance) { n = st.KVRestored() })
		return n
	}
	if n := run(); n != 0 {
		t.Fatalf("the first generate restored %d positions from an empty store", n)
	}
	if n := run(); n == 0 {
		t.Fatal("the second generate of the same prompt restored nothing: the store was not used")
	}
}

// TestASessionDoesNotStarveThePager: a session's history is committed as it
// grows, so creating one must not charge the host budget for its whole
// context at once. Llama-3.2-1B's context is 131072 positions, whose f32
// history over every block is far more than the 3 GiB this model is pinned to,
// while its weights fit several times over: charged up front, the pager was
// left one byte and every token read its weights from disk.
func TestASessionDoesNotStarveThePager(t *testing.T) {
	const pin = 3 << 30
	e, lm, _ := loadedEngine(t, "Llama-3.2-1B-Instruct-Q4_K_M.jlm", "m", LoadOptions{PageBudgetBytes: pin})
	if !lm.m.PagesFit(lm.m.Frames) {
		t.Fatalf("the model does not fit %d bytes before any session: budget %d", pin, lm.m.PageBudget(lm.m.Frames))
	}
	s, err := e.CreateSession(SessionOptions{ModelID: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if !lm.m.PagesFit(lm.m.Frames) {
		t.Fatalf("creating one session of %d positions left the pager %d bytes: the budget was charged "+
			"for the whole context's history up front", s.ms.Limit(), lm.m.PageBudget(lm.m.Frames))
	}
	o := GenerateOptions{SessionID: s.ID(), Prompt: Prompt{Kind: PromptText, Text: "Once upon a time"}, MaxTokens: 4}
	if err := e.Generate(context.Background(), o, func(Event) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if !lm.m.PagesFit(lm.m.Frames) {
		t.Fatalf("after a generate the pager holds %d bytes and the weights do not fit", lm.m.PageBudget(lm.m.Frames))
	}
}

// TestEveryLoadRederivesTheOffHeapBudget: the collector's limit is
// process-wide, so the engine tells Config.OffHeap the page budgets every
// loaded model may grow to, on every load and unload: a second model's load
// keeps the sum at the host total the two divide, and the last unload hands
// back zero, so jitllmd's goheap.OffHeap returns the limit to goheap.Cap's.
//
// VIOLATION SIGNATURE. Without the call in rebudget this fails with
//
//	after loading b: OffHeap was told nothing
func TestEveryLoadRederivesTheOffHeapBudget(t *testing.T) {
	e, a, _ := loadedEngine(t, smallModel, "a", LoadOptions{})
	var told []uint64
	e.cfg.OffHeap = func(b, _ uint64) { told = append(told, b) }
	// Every load re-reads the host (rereadHostLocked); held still here, so
	// the sum is the division's and not the moment's free memory.
	e.hostAvail = func() uint64 { return 6 << 30 }
	e.mu.Lock()
	e.rereadHostLocked()
	total := e.hostTotal()
	e.mu.Unlock()
	last := func(what string, want uint64) {
		t.Helper()
		if len(told) == 0 {
			t.Fatalf("%s: OffHeap was told nothing", what)
		}
		if got := told[len(told)-1]; got != want {
			t.Fatalf("%s: OffHeap told %d, want %d", what, got, want)
		}
		told = told[:0]
	}
	if _, err := e.LoadModel(LoadOptions{Path: a.path, ModelID: "b"}); err != nil {
		t.Fatal(err)
	}
	last("after loading b", total)
	if _, err := e.UnloadModel("b", true); err != nil {
		t.Fatal(err)
	}
	last("after unloading b", total)
	if _, err := e.UnloadModel("a", true); err != nil {
		t.Fatal(err)
	}
	last("after the last unload", 0)
}
