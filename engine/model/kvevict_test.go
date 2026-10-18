package model

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// evictRun is one arm of the eviction gate: the model placed whole on the
// card, a short prompt, then -- squeezed -- the budget held at what the card
// already spends plus a few pages, so the history can only grow by sending its oldest pages
// home, and a teacher-forced decode far past what the pool holds.
type evictRun struct {
	logits   [][]float32
	st       tier.Stats
	resident int
	after    int
	// turnPasses is how many streamed passes the second prompt took, and
	// turnAfter the blocks on the card after it.
	turnPasses, turnAfter int
}

// evictGate is the shared setup: the model, the prompt and the host's token
// ids that every arm is forced through.
type evictGate struct {
	m      *Model
	prompt []int32
	ids    []int32
	// turn is a second prompt after the decode, over a history whose oldest
	// pages are at home: one chunk that streams them. hostTurn holds the
	// host's logits after it and after evictTurnSteps decode steps.
	turn     []int32
	hostTurn [][]float32
	host     [][]float32
	// extra is added to every tier the gate opens: a violation gate arms its
	// fault here, on the gate instance.
	extra []tier.Option
}

const evictDecode = 320

const evictTurnSteps = 4

// evictNMSE bounds the evicting arm's logits against the unconstrained card's
// and the host's: the order-only band is ~2e-3 on this model.
const evictNMSE = 1e-2

func newEvictGate(t *testing.T) *evictGate {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := Open(jlmOf(t, p), noTune)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	eg := &evictGate{m: m}
	eg.prompt = m.Vocab.Encode(strings.Repeat("The clerk counted barrels of salt on the upper floor. ", 3), true)[:24]
	eg.turn = m.Vocab.Encode(strings.Repeat("The river rose every spring until the bridges were islands. ", 10), false)[:100]
	st := m.NewState(eg.seq())
	defer st.Close()
	lg, err := st.Prefill(eg.prompt)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < evictDecode; i++ {
		id := Greedy(lg)
		eg.ids = append(eg.ids, id)
		if lg, err = st.Forward(id); err != nil {
			t.Fatal(err)
		}
		eg.host = append(eg.host, append([]float32(nil), lg...))
	}
	eg.hostTurn = eg.second(t, st)
	return eg
}

// seq is the context every arm opens: the prompt, the decode, the second
// prompt and its steps.
func (eg *evictGate) seq() int {
	return len(eg.prompt) + evictDecode + len(eg.turn) + evictTurnSteps + 1
}

// second runs the second prompt and evictTurnSteps greedy steps after it.
func (eg *evictGate) second(t *testing.T, st *State) [][]float32 {
	lg, err := st.Prefill(eg.turn)
	if err != nil {
		t.Fatalf("the second prompt: %v", err)
	}
	out := [][]float32{append([]float32(nil), lg...)}
	for range evictTurnSteps {
		if lg, err = st.Forward(Greedy(lg)); err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	return out
}

// open is an arm's device and State, the model wholly placed and the prompt
// in; squeeze then holds the budget at a few pages of slack.
func (eg *evictGate) open(t *testing.T, squeeze bool) (*tier.GPU, *State, int) {
	g, err := openTier(t, append([]tier.Option{tier.WithDevices(testdevices.Like(t, "cuda:0")), tier.WithDeviceTune(tier.TuneOff),
		tier.WithConfig(func(c *tier.Config) {
			c.KVPage = 64 // the smallest page: several over a short history
			c.StagedDecode = true
		})}, eg.extra...)...)
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	t.Cleanup(g.Close)
	st := eg.m.NewState(eg.seq())
	t.Cleanup(func() { st.Close() })
	if err := st.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	resident := st.devCount()
	if resident != eg.m.Cfg.NLayer {
		t.Skipf("CARD TOO SMALL: %d of %d blocks placed (%s) -- this gate proved nothing here",
			resident, eg.m.Cfg.NLayer, g.Err())
	}
	if _, err := st.Prefill(eg.prompt); err != nil {
		t.Fatal(err)
	}
	if squeeze {
		// A few pages of slack: the pool fills them, and past them the
		// history can grow only by sending pages home. Evicting needs room to
		// stream through; a pool of the dummy and one page refuses instead.
		if _, err := g.SetBudget(g.Stats().BudgetUsed + 8<<20); err != nil {
			t.Fatal(err)
		}
	}
	return g, st, resident
}

func (eg *evictGate) run(t *testing.T, squeeze bool) evictRun {
	g, st, resident := eg.open(t, squeeze)
	r := evictRun{resident: resident}
	for _, id := range eg.ids {
		lg, err := st.Forward(id)
		if err != nil {
			t.Fatalf("step %d: %v (%s)", len(r.logits), err, g.Err())
		}
		r.logits = append(r.logits, append([]float32(nil), lg...))
	}
	r.after = st.devCount()
	before := g.Stats().KVStreamPasses
	r.logits = append(r.logits, eg.second(t, st)...)
	r.st, r.turnAfter = g.Stats(), st.devCount()
	r.turnPasses = r.st.KVStreamPasses - before
	return r
}

// worstStep is the largest per-step logit NMSE of got against want.
func worstStep(got, want [][]float32) (float64, int) {
	w, at := 0.0, 0
	for i := range want {
		if e := logitNMSE(got[i], want[i]); !(e <= w) {
			w, at = e, i
		}
	}
	return w, at
}

// TestKVEvictionRunsPastTheBudget is slice 4's gate: a history larger than
// the device's KV budget runs to completion with placement unchanged, its
// oldest pages at home and streamed back through the pool's free ids for every
// token's attention, and its logits stay with an unconstrained device arm's
// and the host's. Stats.KVEvictions and KVStreamPasses are the selection
// check: a budget that did not bind would compare a run against itself.
func TestKVEvictionRunsPastTheBudget(t *testing.T) {
	eg := newEvictGate(t)
	free := eg.run(t, false)
	got := eg.run(t, true)
	if got.after != got.resident {
		t.Fatalf("%d of %d blocks on the card after the decode: eviction must not move blocks home",
			got.after, got.resident)
	}
	if got.st.KVEvictions == 0 || got.st.KVStreamPasses == 0 {
		t.Fatalf("%d evictions and %d streamed passes: the budget never bound (pool %d B)",
			got.st.KVEvictions, got.st.KVStreamPasses, got.st.KVPoolBytes)
	}
	if free.st.KVEvictions != 0 {
		t.Fatalf("the unconstrained arm evicted %d page(s)", free.st.KVEvictions)
	}
	// The second prompt streamed as a chunk: a row at a time pays at least one
	// upload pass per layer for every row.
	if rows := len(eg.turn) * eg.m.Cfg.NLayer; got.turnPasses == 0 || got.turnPasses >= rows {
		t.Fatalf("the second prompt took %d streamed passes (a row at a time takes at least %d): it did not stream as a chunk",
			got.turnPasses, rows)
	}
	dev, at := worstStep(got.logits, free.logits)
	all := append(append([][]float32(nil), eg.host...), eg.hostTurn...)
	host, hat := worstStep(got.logits, all)
	ctl, _ := worstStep(free.logits, all)
	t.Logf("%d tokens past a %d-token prompt, pool %.2f MiB: %d page(s) of every layer sent home, %d streamed passes; "+
		"worst logit NMSE %.3e against the unconstrained card (step %d), %.3e against the host (step %d; "+
		"the unconstrained card reads %.3e)", evictDecode, len(eg.prompt), float64(got.st.KVPoolBytes)/(1<<20),
		got.st.KVEvictions, got.st.KVStreamPasses, dev, at, host, hat, ctl)
	// The chunk's attention scratch is far wider than a decode row's, and on a
	// card squeezed to a few pages of slack the tier may make room for it by
	// moving a block home: reported, not refused, since the host block still
	// computes and the logits are held to the same bound.
	t.Logf("the %d-token second prompt over the evicted history: %d streamed passes, %d of %d blocks on the card after it",
		len(eg.turn), got.turnPasses, got.turnAfter, got.resident)
	// Passes fold the history in another order than one launch does, and the
	// difference compounds through the layers: the bound is the device's own
	// band against the host, which a stale or missing page is far outside.
	if !(dev < evictNMSE) {
		t.Fatalf("worst logit NMSE %.3e against the unconstrained card at step %d", dev, at)
	}
	if !(host < evictNMSE) {
		t.Fatalf("worst logit NMSE %.3e against the host at step %d", host, hat)
	}
}

// TestKVEvictionStreamsWithoutAllocating holds a decode token whose attention
// streams evicted pages back through the pool to zero engine allocations: the
// path a picture's long prompt takes on a card that holds part of a model, and one
// TestDecodeDoesNotAllocate's short, unconstrained window never reaches.
func TestKVEvictionStreamsWithoutAllocating(t *testing.T) {
	eg := newEvictGate(t)
	g, st, _ := eg.open(t, true)
	// The window inside one 64-position page: sending a page home is paid
	// once a page of growth, as committing one is, not once a token.
	const n = 16
	cut := len(eg.ids) - n
	for _, id := range eg.ids[:cut] {
		if _, err := st.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	if st.pos/64 != (st.pos+n)/64 {
		t.Fatalf("the window %d..%d crosses a page boundary", st.pos, st.pos+n)
	}
	p0, r0 := g.Stats().KVStreamPasses, eg.m.container.Reads()
	if p0 == 0 {
		t.Fatal("no pass streamed before the window: the budget never bound")
	}
	w := countAllocs(func() {
		for _, id := range eg.ids[cut:] {
			if _, err := st.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
	})
	passes := g.Stats().KVStreamPasses - p0
	if passes == 0 {
		t.Fatal("no pass streamed inside the window: it measured a resident decode")
	}
	allocVerdict(t, fmt.Sprintf("%d streamed pass(es)", passes), w, n, eg.m.container.Reads()-r0, 0)
}

// TestKVEvictionRewindsIntoAnEvictedPage: a history whose oldest pages went
// home is taken back to a position inside one of them -- a speculative
// rollback, a pooled State's next prompt -- and decoded forward through
// another text. The page the position lies inside comes back to the card with
// its home copy, whose rows below the position are still the history, and the
// pages past it are released for the new rows; every step is held to a host
// State that took the same ids. Before the tier learned of rewinds, the new
// rows landed on the dummy page under an evicted page's id and the stream
// uploaded the old copy over them (NMSE 0.92 at the seventh step), and with a
// layer joining later streamDecode indexed past its home copies: the jitllmd
// crash at 64 concurrent requests. A rewind that does not bring the page's
// home copy back reads 0.20.
func TestKVEvictionRewindsIntoAnEvictedPage(t *testing.T) {
	eg := newEvictGate(t)
	g, st, _ := eg.open(t, true)
	for _, id := range eg.ids {
		if _, err := st.Forward(id); err != nil {
			t.Fatalf("%v (%s)", err, g.Err())
		}
	}
	ev := g.Stats().KVEvictions
	if ev < 2 {
		t.Fatalf("%d page(s) sent home: the rewind below would not land in an evicted page", ev)
	}
	// Inside page 1 of 64 positions: at home in every layer. The rows after
	// it are another text, so a page read from its old copy reads wrong.
	const p = 64 + 17
	next := append(append([]int32(nil), eg.turn...), eg.turn...)
	host := eg.m.NewState(eg.seq())
	defer host.Close()
	if _, err := host.Prefill(eg.prompt); err != nil {
		t.Fatal(err)
	}
	for _, id := range eg.ids[:p-len(eg.prompt)] {
		if _, err := host.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	st.rewind(p)
	worst, at := 0.0, 0
	for i, id := range next {
		want, err := host.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		lg, err := st.Forward(id)
		if err != nil {
			t.Fatalf("step %d after the rewind: %v (%s)", i, err, g.Err())
		}
		if e := logitNMSE(lg, want); !(e <= worst) {
			worst, at = e, i
		}
	}
	t.Logf("rewound to %d past %d page(s) at home, %d more sent home after it; worst logit NMSE %.3e against "+
		"the host (step %d)", p, ev, g.Stats().KVEvictions-ev, worst, at)
	if !(worst < evictNMSE) {
		t.Fatalf("worst logit NMSE %.3e against the host at step %d: the rows after the rewind were lost or "+
			"read stale", worst, at)
	}
}
