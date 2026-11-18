package engine

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// evictRun is one arm of the eviction gate: the model placed whole on the
// card, a short prompt and half the decode, then -- squeezed -- the budget held
// at what the card spends there, so the history can only grow by sending its
// oldest pages home, and the rest of a teacher-forced decode far past what the
// pool holds.
type evictRun struct {
	logits   [][]float32
	st       gpu.Stats
	resident int
	after    int
	// turnPasses is how many streamed passes the second prompt took, and
	// turnAfter the blocks on the card after it.
	turnPasses, turnAfter int
	// squeeze is what an unconstrained arm measured for the squeezed arms of
	// its configuration.
	squeeze evictSqueeze
}

// evictSqueeze is what an unconstrained arm measures from the decode's middle
// step on: rest, the most it allocated besides its history -- scratch and
// tables a longer history grows -- which a squeezed arm's budget leaves room
// for, so it binds on history alone; and the history it held there and at the
// end of the decode, which shows how far the history outgrew the squeeze.
type evictSqueeze struct {
	rest, mid, end uint64
}

// evictGate is the shared setup: the model, the prompt and the host's token
// ids that every arm is forced through.
type evictGate struct {
	m      *model.Model
	prompt []int32
	ids    []int32
	// turn is a second prompt after the decode, over a history whose oldest
	// pages are at home: one chunk that streams them. hostTurn holds the
	// host's logits after it and after evictTurnSteps decode steps.
	turn     []int32
	hostTurn [][]float32
	host     [][]float32
}

const evictDecode = 320

const evictTurnSteps = 4

// evictNMSE bounds the evicting arm's logits against the unconstrained card's
// and the host's: the order-only band is ~2e-3 on this model.
const evictNMSE = 1e-2

// evictFloatNMSE bounds the float forms' squeezed card against their
// unconstrained one.
const evictFloatNMSE = 1e-4

func newEvictGate(t *testing.T) *evictGate { return newEvictGateOf(t, evictModels()[0]) }

// evictModels are the checkpoints the eviction gate runs: a dense model, a
// mixture, whose blocks fold their history in passes through forms of their
// own, a selecting model, whose lightning indexer ranks its whole history's
// positions in a sweep of passes before its attention's, one selecting
// by pooled block keys (MiniMax-M3's), whose sweep sums each block's keys,
// and two whose blocks alternate a window and the whole history (cohere2,
// exaone4): the squeeze binds on the whole-history layers while the windowed
// ones hold their window and the positions behind it a rewind may come back
// to; and DeepSeek V4's, whose blocks keep entries beside their rows --
// compressed and selected among (synth-deepseek4) or heavily compressed
// (-hca) -- both sent home and read in each pass.
func evictModels() []string {
	return []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Qwen3-MOE-4x0.6B-Q4_K_M.gguf", "synth-deepseek32.gguf", "synth-minimaxm3.gguf",
		"synth-cohere2.gguf", "synth-exaone4.gguf", "synth-deepseek4.gguf", "synth-deepseek4-hca.gguf", readsTwoFixture}
}

func newEvictGateOf(t *testing.T, name string) *evictGate {
	p := testmodels.Path(name)
	if _, ok := existingModel(p); !ok {
		testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS to the model directory) -- this gate proved nothing", p)
	}
	m, err := model.Open(jlmOf(t, p))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return evictGateOn(t, m)
}

// evictGateOn is newEvictGateOf on the opened model m.
func evictGateOn(t *testing.T, m *model.Model) *evictGate {
	eg := &evictGate{m: m}
	eg.prompt = m.Vocab.Encode(strings.Repeat("The clerk counted barrels of salt on the upper floor. ", 3), true)[:24]
	eg.turn = m.Vocab.Encode(strings.Repeat("The river rose every spring until the bridges were islands. ", 10), false)[:100]
	st := testInstance(t, m, eg.seq())
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
func (eg *evictGate) second(t *testing.T, st *Instance) [][]float32 {
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

// mid is the decode step the squeezed arms are squeezed at.
func (eg *evictGate) mid() int { return len(eg.ids) / 2 }

// open is an arm's device and instance, the model wholly placed, the prompt in
// and the decode taken to its middle step (each step's logits); sq, where set,
// then holds the budget at what the card spends there and what sq measured.
// cfg, where set, changes the arm's device configuration.
func (eg *evictGate) open(t *testing.T, sq *evictSqueeze, cfg func(*gpu.Config), extra ...gpu.Option) (*gpu.GPU, *Instance, int, [][]float32) {
	g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(testdevices.Like(t, "cuda:0")), gpu.WithDeviceTune(gpu.TuneOff),
		gpu.WithConfig(func(c *gpu.Config) {
			c.KVPage = 64 // the smallest page: several over a short history
			if cfg != nil {
				cfg(c)
			}
		})}, extra...)...)
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	t.Cleanup(g.Close)
	st := testInstance(t, eg.m, eg.seq())
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
	// Half the decode before the budget is cut, so the decode's own scratch
	// is built and charged first, and every layer holding its whole history
	// holds pages enough to send home: the page a growth takes and the one it
	// streams through.
	var logits [][]float32
	for _, id := range eg.ids[:eg.mid()] {
		lg, err := st.Forward(id)
		if err != nil {
			t.Fatalf("step %d: %v (%s)", len(logits), err, g.Err())
		}
		logits = append(logits, append([]float32(nil), lg...))
	}
	if sq != nil {
		// The pages the history holds and what the unconstrained arm
		// allocated besides it: past them the history can grow only by
		// sending pages home. No room for growth: a layer's growth is
		// charged its buffers' rounding too, which would take the room before
		// the pages did. A budget of what the card holds gives the pool's
		// free pages back first (SetBudget), so the room is counted from what
		// is left, past the slack the card keeps free beside every
		// allocation (Stats.Slack).
		s := g.Stats()
		if _, err := g.SetBudget(s.BudgetUsed); err != nil {
			t.Fatal(err)
		}
		s = g.Stats()
		if _, err := g.SetBudget(s.BudgetUsed + s.Slack + sq.rest); err != nil {
			t.Fatal(err)
		}
	}
	return g, st, resident, logits
}

// evictRest is what the card's budget holds besides the history's pool.
func evictRest(s gpu.Stats) uint64 { return s.BudgetUsed - min(s.BudgetUsed, s.KVPoolBytes) }

// run is one arm through the decode and the second prompt: squeezed by sq, or
// unconstrained where sq is nil, measuring the squeeze for its configuration.
func (eg *evictGate) run(t *testing.T, sq *evictSqueeze, cfg func(*gpu.Config), extra ...gpu.Option) evictRun {
	g, st, resident, logits := eg.open(t, sq, cfg, extra...)
	r := evictRun{resident: resident, logits: logits}
	s0 := g.Stats()
	r.squeeze.mid = s0.KVBytes
	for _, id := range eg.ids[eg.mid():] {
		lg, err := st.Forward(id)
		if err != nil {
			t.Fatalf("step %d: %v (%s)", len(r.logits), err, g.Err())
		}
		r.logits = append(r.logits, append([]float32(nil), lg...))
		if sq == nil {
			s := g.Stats()
			r.squeeze.rest = max(r.squeeze.rest, evictRest(s)-min(evictRest(s), evictRest(s0)))
		}
	}
	r.squeeze.end = g.Stats().KVBytes
	r.after = st.devCount()
	before := g.Stats().KVStreamPasses
	r.logits = append(r.logits, eg.second(t, st)...)
	r.st, r.turnAfter = g.Stats(), st.devCount()
	if sq == nil {
		r.squeeze.rest = max(r.squeeze.rest, evictRest(r.st)-min(evictRest(r.st), evictRest(s0)))
	}
	r.turnPasses = r.st.KVStreamPasses - before
	return r
}

// stepSpread is the 90th percentile and the largest of got's per-step logit
// NMSE against want from step from on, and the largest's step.
func stepSpread(got, want [][]float32, from int) (p90, worst float64, at int) {
	var per []float64
	for i := from; i < len(want); i++ {
		e := logitNMSE(got[i], want[i])
		if !(e <= worst) {
			worst, at = e, i
		}
		per = append(per, e)
	}
	slices.Sort(per)
	return per[len(per)*9/10], worst, at
}

// evictHeld is whether per-step distances whose bulk is p90 and largest worst
// keep to bound: on a mixture the largest may reach bandWorst of it. Its
// routing turns an order's rounding into a large move at a lone step, as it
// turns the card's own order into one against the host; a stale or missing
// page moves every step after it.
func evictHeld(m *model.Model, p90, worst, bound float64) bool {
	if m.Cfg.NExpert == 0 {
		return worst < bound
	}
	return p90 < bound && worst < bandWorst*bound
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
//
// The arms run the composer's blocks, the squeezed one its forms over passes
// (Stats.ComposedStreamPasses, with no block step declined); main's tier
// squeezed and the composer's float reference squeezed are held to the same
// bound against it.
func TestKVEvictionRunsPastTheBudget(t *testing.T) {
	for _, name := range evictModels() {
		t.Run(name, func(t *testing.T) { evictionRunsPastTheBudget(t, newEvictGateOf(t, name)) })
	}
}

func evictionRunsPastTheBudget(t *testing.T, eg *evictGate) {
	free := eg.run(t, nil, nil)
	got := eg.run(t, &free.squeeze, nil)
	float := func(c *gpu.Config) { c.ComposedFloat = true }
	freeRef := eg.run(t, nil, float)
	ref := eg.run(t, &freeRef.squeeze, float)
	for name, r := range map[string]evictRun{"composed": got, "float reference": ref} {
		for i, n := range r.st.BlockDeclines() {
			if n > 0 {
				t.Errorf("the %s arm declined %d block step(s): %s", name, n, gpu.DeclineName(i))
			}
		}
		if r.st.ComposedStreamPasses == 0 {
			t.Errorf("the %s arm ran no composed pass over the evicted history", name)
		}
	}
	if got.after != got.resident {
		t.Fatalf("%d of %d blocks on the card after the decode: eviction must not move blocks home (%s)",
			got.after, got.resident, got.st.LastErr)
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
	t.Logf("%d tokens past a %d-token prompt, squeezed at step %d to %.3f MiB of history (%.3f MiB at the end "+
		"unconstrained) and %.3f MiB besides it, pool %.2f MiB: %d layer page(s) sent home, %d streamed passes; "+
		"worst logit NMSE %.3e against the unconstrained card (step %d), %.3e against the host (step %d; "+
		"the unconstrained card reads %.3e)", evictDecode, len(eg.prompt), eg.mid(), float64(free.squeeze.mid)/(1<<20),
		float64(free.squeeze.end)/(1<<20), float64(free.squeeze.rest)/(1<<20), float64(got.st.KVPoolBytes)/(1<<20),
		got.st.KVEvictions, got.st.KVStreamPasses, dev, at, host, hat, ctl)
	// The chunk's attention scratch is far wider than a decode row's, and on a
	// card squeezed to its history's pages the tier may make room for it by
	// moving a block home: reported, not refused, since the host block still
	// computes and the logits are held to the same bound.
	t.Logf("the %d-token second prompt over the evicted history: %d streamed passes, %d of %d blocks on the card after it",
		len(eg.turn), got.turnPasses, got.turnAfter, got.resident)
	// Passes fold the history in another order than one launch does, and the
	// difference compounds through the layers: the bound is the device's own
	// band against the host, which a stale or missing page is far outside --
	// evictNMSE, or where the unconstrained card is already farther from the
	// host (a mixture's routing turns the order's differences into larger
	// moves), that distance, measured in this run (the user's decision).
	bound := max(evictNMSE, ctl)
	if p90, _, _ := stepSpread(got.logits, free.logits, eg.mid()); !evictHeld(eg.m, p90, dev, bound) {
		t.Fatalf("logit NMSE %.3e at the 90th percentile of the squeezed steps and %.3e at worst (step %d) against the unconstrained card, past %.3e (%d composed passes, %d streamed)",
			p90, dev, at, bound, got.st.ComposedStreamPasses, got.st.KVStreamPasses)
	}
	// The shipping arms' int8 activations move a mixture's routing, so the
	// evicting arm's distance from the host is bounded by the two it is
	// measured to be from the unconstrained card and that card from the
	// host. What a stale or missing page moves, the float arms below see.
	hostBound := max(bound, math.Pow(math.Sqrt(ctl)+math.Sqrt(dev), 2))
	if !(host < hostBound) {
		t.Fatalf("worst logit NMSE %.3e against the host at step %d, past %.3e", host, hat, hostBound)
	}
	// On the float forms the squeezed and unconstrained cards part by the
	// streamed passes' folding order alone.
	fw, fat := worstStep(ref.logits, freeRef.logits)
	t.Logf("the float forms, squeezed against unconstrained: worst NMSE %.3e (step %d)", fw, fat)
	if !(fw < evictFloatNMSE) {
		t.Fatalf("on the float forms the squeezed card parts from the unconstrained one by %.3e at step %d, past %.0e: the evicted history is not the one it sent home",
			fw, fat, evictFloatNMSE)
	}
	rp, rw, rat := stepSpread(got.logits, ref.logits, 0)
	t.Logf("composed passes: %d, against the float reference squeezed %.3e (step %d), %.3e at the 90th percentile", got.st.ComposedStreamPasses, rw, rat, rp)
	if !evictHeld(eg.m, rp, rw, bound) {
		t.Fatalf("logit NMSE %.3e against the float reference squeezed at step %d, %.3e at the 90th percentile, past %.3e", rw, rat, rp, bound)
	}
}

// TestKVEvictionStreamsWithoutAllocating holds a decode token whose attention
// streams evicted pages back through the pool to zero engine allocations: the
// path a picture's long prompt takes on a card that holds part of a model, and one
// TestDecodeDoesNotAllocate's short, unconstrained window never reaches.
func TestKVEvictionStreamsWithoutAllocating(t *testing.T) {
	eg := newEvictGate(t)
	free := eg.run(t, nil, nil)
	g, st, _, _ := eg.open(t, &free.squeeze, nil)
	// The window inside one 64-position page: sending a page home is paid
	// once a page of growth, as committing one is, not once a token.
	const n = 16
	cut := len(eg.ids) - n
	for _, id := range eg.ids[eg.mid():cut] {
		if _, err := st.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	if st.pos/64 != (st.pos+n)/64 {
		t.Fatalf("the window %d..%d crosses a page boundary", st.pos, st.pos+n)
	}
	p0, r0 := g.Stats().KVStreamPasses, eg.m.Container.Reads()
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
	allocVerdict(t, fmt.Sprintf("%d streamed pass(es)", passes), w, n, eg.m.Container.Reads()-r0, 0)
}

// TestKVEvictionRewindsIntoAnEvictedPage: a history whose oldest pages went
// home is taken back to a position inside one of them -- a speculative
// rollback, a pooled instance's next prompt -- and decoded forward through
// another text. The page the position lies inside comes back to the card with
// its home copy, whose rows below the position are still the history, and the
// pages past it are released for the new rows; every step is held to a host
// Instance that took the same ids. Before the tier learned of rewinds, the new
// rows landed on the dummy page under an evicted page's id and the stream
// uploaded the old copy over them (NMSE 0.92 at the seventh step), and with a
// layer joining later streamDecode indexed past its home copies: the jitllmd
// crash at 64 concurrent requests. A rewind that does not bring the page's
// home copy back reads 0.20.
func TestKVEvictionRewindsIntoAnEvictedPage(t *testing.T) {
	eg := newEvictGate(t)
	free := eg.run(t, nil, nil)
	g, st, _, _ := eg.open(t, &free.squeeze, nil)
	for _, id := range eg.ids[eg.mid():] {
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
	host := testInstance(t, eg.m, eg.seq())
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
	if d, r := st.DeviceDemotions(), st.Relocations(); d != 0 || r != 0 {
		t.Fatalf("%d demotion(s) and %d relocation(s): the steps after the rewind did not all run on the device (%s)", d, r, g.Err())
	}
	if !(worst < evictNMSE) {
		t.Fatalf("worst logit NMSE %.3e against the host at step %d: the rows after the rewind were lost or "+
			"read stale", worst, at)
	}
}
