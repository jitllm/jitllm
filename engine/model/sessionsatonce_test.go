package model

import (
	"fmt"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// Two States on one tier step on the device at the same time: each is its
// own session (tier.GPU.Attach), runs in a lane of its own and submits on a
// queue of its own (jit/gpu/tier/devsess.go), and the device's shared state
// waits only where it must (jit/gpu/tier/inflight.go). These gates hold the
// sessions to the answers each gives alone, and check the configuration was
// selected: their steps overlapped, and the device saw a submission start
// while another's was running on the device (tier.Stats.RanBeside).

// atOnceGen is how many tokens each session decodes in a gate.
const atOnceGen = 48

// atOnceRun is one session's decode: its greedy ids, each step's logits, and
// each step's span on the wall clock.
type atOnceRun struct {
	ids    []int32
	logits [][]float32
	spans  [][2]time.Time
}

// atOncePrompts are the sessions' prompts, one each.
func atOncePrompts(m *Model) [2][]int32 {
	return [2][]int32{
		m.Vocab.Encode("Once upon a time, there was a little girl named", true),
		m.Vocab.Encode("The big dog ran to the park and", true),
	}
}

// atOnceTier opens spec with the tuners off, so every arm runs the same
// kernels in the same reduction order.
func atOnceTier(t *testing.T, spec string, cfg func(*tier.Config)) *tier.GPU {
	t.Helper()
	opts := []tier.Option{tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff)}
	if cfg != nil {
		opts = append(opts, tier.WithConfig(cfg))
	}
	g, err := tier.OpenWith(opts...)
	if err != nil || g == nil {
		noDevice(t, spec, err)
	}
	t.Cleanup(g.Close)
	return g
}

// atOnceState places a State wholly on g and prefills prompt, returning the
// first greedy token.
func atOnceState(t *testing.T, m *Model, g *tier.GPU, seq int, prompt []int32) (*State, int32) {
	t.Helper()
	st := m.NewState(seq)
	t.Cleanup(func() { st.Close() })
	if err := st.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if st.devCount() != m.Cfg.NLayer || !st.HeadOnDevice() {
		t.Fatalf("%d of %d blocks on the device, head there %v: %s", st.devCount(), m.Cfg.NLayer,
			st.HeadOnDevice(), g.Err())
	}
	lg, err := st.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	return st, Greedy(lg)
}

// decode runs n greedy steps from next into r.
func (r *atOnceRun) decode(st *State, next int32, n int) error {
	for range n {
		t0 := time.Now()
		lg, err := st.Forward(next)
		r.spans = append(r.spans, [2]time.Time{t0, time.Now()})
		if err != nil {
			return err
		}
		next = Greedy(lg)
		r.ids = append(r.ids, next)
		r.logits = append(r.logits, slices.Clone(lg))
	}
	return nil
}

// atOnce decodes every state from its first token, one goroutine each,
// started together.
func atOnce(t *testing.T, sts []*State, next []int32, n int) []atOnceRun {
	t.Helper()
	runs := make([]atOnceRun, len(sts))
	errs := make([]error, len(sts))
	var ready, wg sync.WaitGroup
	start := make(chan struct{})
	for k := range sts {
		ready.Add(1)
		wg.Add(1)
		go func() {
			defer wg.Done()
			ready.Done()
			<-start
			errs[k] = runs[k].decode(sts[k], next[k], n)
		}()
	}
	ready.Wait()
	close(start)
	wg.Wait()
	for k, err := range errs {
		if err != nil {
			t.Fatalf("session %d: %v", k, err)
		}
	}
	return runs
}

// sameRun fails unless got is want to the bit: the same ids and the same
// logits at every step.
func sameRun(t *testing.T, what string, got, want atOnceRun) {
	t.Helper()
	if len(got.ids) != len(want.ids) {
		t.Fatalf("%s: %d steps, alone %d", what, len(got.ids), len(want.ids))
	}
	for i := range want.ids {
		if got.ids[i] != want.ids[i] {
			t.Fatalf("%s: step %d is token %d, alone %d\n  at once %v\n  alone   %v", what, i,
				got.ids[i], want.ids[i], got.ids, want.ids)
		}
		for j := range want.logits[i] {
			if got.logits[i][j] != want.logits[i][j] {
				t.Fatalf("%s: step %d logit %d is %v at once, %v alone", what, i, j,
					got.logits[i][j], want.logits[i][j])
			}
		}
	}
}

// overlaps is how many of a's steps overlap one of b's on the wall clock.
func overlaps(a, b atOnceRun) int {
	n := 0
	for _, x := range a.spans {
		for _, y := range b.spans {
			if x[0].Before(y[1]) && y[0].Before(x[1]) {
				n++
				break
			}
		}
	}
	return n
}

// TestSessionsStepAtOnce is the gate for sessions stepping at once: two
// States of stories15M, every block and the head on one device, decode from
// two goroutines together, and each produces exactly the ids and logits it
// produces alone on the same tier. Their steps overlap on the wall clock and
// the device ran submissions beside one another (RanBeside): run with a
// lock held across the submission, the ids still agree and RanBeside stays
// at zero, so this fails.
//
// NVIDIA's Vulkan driver runs one submission's dispatches at a time across
// queues (backend.TestQueuesRunSessionsAtOnce), so there the gate holds the
// answers and the host-side overlap; how much the device itself overlaps is
// TestSessionsOverlapOnCUDA's to measure.
func TestSessionsStepAtOnce(t *testing.T) {
	m, err := Open(jlmOf(t, testmodels.Path("stories15M-q8_0.gguf")), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for _, spec := range testdevices.Specs("cuda:0", "vulkan:0", "vulkan:1", "metal") {
		t.Run(spec, func(t *testing.T) {
			stepAtOnce(t, m, atOncePrompts(m), spec, m.Cfg.NLayer, true, 0)
		})
	}
}

// TestSessionsStepAtOnceComposed is TestSessionsStepAtOnce with the composer
// serving the blocks and the head: each lane holds its own instance of the
// device's forms, so the sessions' submissions run beside each other and
// each still produces exactly what it produces alone.
func TestSessionsStepAtOnceComposed(t *testing.T) {
	m, err := Open(jlmOf(t, testmodels.Path("stories15M-q8_0.gguf")), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for _, spec := range testdevices.Specs("cuda:0", "vulkan:0", "metal") {
		t.Run(spec, func(t *testing.T) {
			stepAtOnce(t, m, atOncePrompts(m), spec, m.Cfg.NLayer, true, tier.ComposedBlock|tier.ComposedHead)
		})
	}
}

// TestSessionsStepAtOnceHybrid is TestSessionsStepAtOnce on hybrids -- the
// gated delta rule beside attention, with and without the routed mixture --
// placed wholly and with a seam that leaves half the blocks on the host, so
// two sessions' recurrent states (each its own seat in the device's pools)
// and the host blocks' single matvecs (each session's own crossing verdict)
// are held to the answers each gives alone. A submission holding a linear
// block runs alone by design (tier exclusiveRange), so the device is not
// asked to run two side by side; the steps still overlap on the wall clock.
func TestSessionsStepAtOnceHybrid(t *testing.T) {
	for _, o := range hybridAtOnce {
		m := hybridModelOpt(t, o.opt)
		defer m.Close()
		nl := m.Cfg.NLayer
		for _, blocks := range []int{nl, nl / 2} {
			for _, spec := range testdevices.Specs("cuda:0", "vulkan:0", "vulkan:1", "metal") {
				t.Run(fmt.Sprintf("%s/%d-of-%d/%s", o.name, blocks, nl, spec), func(t *testing.T) {
					stepAtOnce(t, m, hybridPrompts, spec, blocks, false, 0)
				})
			}
		}
	}
}

// hybridAtOnce are the hybrids the session gates run: the gated delta rule
// beside attention, and the same with qwen3next's routed mixture and shared
// expert.
var hybridAtOnce = []struct {
	name string
	opt  hyOpt
}{{"delta", hyOpt{}}, {"delta+moe", hyOpt{moe: true}}}

// hybridPrompts are the sessions' prompts on the synthetic hybrid, whose
// vocabulary is hyVocab ids with no text behind them.
var hybridPrompts = [2][]int32{{1, 2, 3, 4, 5, 6, 7, 8}, {9, 3, 14, 1, 5, 9, 2, 6}}

// atOncePlace places a State's first blocks on g -- every block and the head
// through atOnceState -- and prefills prompt, returning the first greedy
// token.
func atOncePlace(t *testing.T, m *Model, g *tier.GPU, seq int, prompt []int32, blocks int) (*State, int32) {
	t.Helper()
	if blocks == m.Cfg.NLayer {
		return atOnceState(t, m, g, seq, prompt)
	}
	st := m.NewState(seq)
	t.Cleanup(func() { st.Close() })
	if err := st.SetDeviceLayers(g, blocks); err != nil {
		t.Fatal(err)
	}
	if st.devCount() != blocks {
		t.Fatalf("%d of %d blocks on the device, asked %d: %s", st.devCount(), m.Cfg.NLayer, blocks, g.Err())
	}
	lg, err := st.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	return st, Greedy(lg)
}

// stepAtOnce runs two sessions of m on blocks of a fresh tier over spec, each
// alone and then both at once from two goroutines, and holds the run at once
// to the runs alone. beside also demands that the device ran a submission
// beside another's.
func stepAtOnce(t *testing.T, m *Model, prompts [2][]int32, spec string, blocks int, beside bool, parts tier.Parts) {
	t.Helper()
	const seq = 256
	var cfg func(*tier.Config)
	if parts != 0 {
		cfg = func(c *tier.Config) { c.Composed = parts }
	}
	g := atOnceTier(t, spec, cfg)
	var alone [2]atOnceRun
	for k := range prompts {
		st, next := atOncePlace(t, m, g, seq, prompts[k], blocks)
		if err := alone[k].decode(st, next, atOnceGen); err != nil {
			t.Fatal(err)
		}
		st.Close()
	}
	before := g.Stats()
	var sts []*State
	var next []int32
	for k := range prompts {
		st, n := atOncePlace(t, m, g, seq, prompts[k], blocks)
		sts, next = append(sts, st), append(next, n)
	}
	runs := atOnce(t, sts, next, atOnceGen)
	after := g.Stats()
	for k := range runs {
		sameRun(t, []string{"session 0", "session 1"}[k], runs[k], alone[k])
	}
	ran := after.RanBeside - before.RanBeside
	ov := overlaps(runs[0], runs[1])
	t.Logf("%s, %d of %d blocks: %d steps each, ids and logits identical to each alone; %d of session 0's "+
		"steps overlapped session 1's, %d submissions ran on the device beside another's; the second "+
		"session's lane is %d bytes of scratch (%d with it, %d without)",
		g.Name(), blocks, m.Cfg.NLayer, atOnceGen, ov, ran, int64(after.ScratchBytes)-int64(before.ScratchBytes),
		after.ScratchBytes, before.ScratchBytes)
	if ov == 0 {
		t.Fatal("no step of one session overlapped the other's: they ran one after another")
	}
	if beside && ran == 0 {
		t.Fatal("no submission started while another was in flight: the sessions' device calls " +
			"were serialised")
	}
	if parts != 0 {
		if n, d, _ := after.ServedPart(parts); n == 0 || d != 0 {
			t.Fatalf("the composer served %d part steps and declined %d", n, d)
		}
		if f := after.FormedAtStep - before.FormedAtStep; f != 0 {
			t.Fatalf("%d composed part(s) formed at a step: a lane did not hold its instance", f)
		}
		if err := lanesHold(g, blocks, parts, 2); err != nil {
			t.Fatal(err)
		}
	}
}

// lanesHold is nil when g has at least lanes lanes and each holds an
// instance of the composed part of each of blocks placed blocks, and of the
// head where parts names it.
func lanesHold(g *tier.GPU, blocks int, parts tier.Parts, lanes int) error {
	want := blocks
	if parts&tier.ComposedHead != 0 {
		want++
	}
	for _, d := range g.LaneParts() {
		if len(d) < lanes {
			return fmt.Errorf("%d lane(s), %v parts each: the sessions did not run in lanes of their own", len(d), d)
		}
		for _, n := range d {
			if n != want {
				return fmt.Errorf("the lanes hold %v composed parts, %d each placed: a lane is short of an "+
					"instance or keeps a released block's", d, want)
			}
		}
	}
	return nil
}

// TestSessionsPageAtOnce: two sessions stepping at once on a device that
// pages -- every block streamed through the fewest slots the budget leaves,
// so blocks page out and in under the other session's submissions -- produce
// exactly the ids and logits each produces alone with every block resident.
// PageIns and PageOuts during the run at once are the selection check that the
// budget bound, and RanBeside that the sessions stepped together.
func TestSessionsPageAtOnce(t *testing.T) {
	m, err := Open(jlmOf(t, testmodels.Path("stories15M-q8_0.gguf")), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompts := atOncePrompts(m)
	const seq, gen = 256, 24
	for _, spec := range testdevices.Specs("cuda:0", "vulkan:0", "metal") {
		t.Run(spec, func(t *testing.T) {
			g := atOnceTier(t, spec, nil)
			for li := range m.Cfg.NLayer {
				g.Stream(li, true)
			}
			place := func(prompt []int32) (*State, int32) {
				st := m.NewState(seq)
				t.Cleanup(func() { st.Close() })
				if err := st.SetDeviceLayers(g, m.Cfg.NLayer); err != nil {
					t.Fatal(err)
				}
				// The head may stay on the host: under paging a slot is
				// worth more than the projection (tier PrepHead).
				if st.devCount() != m.Cfg.NLayer {
					t.Fatalf("%d of %d blocks on the device: %s", st.devCount(), m.Cfg.NLayer, g.Err())
				}
				lg, err := st.Prefill(prompt)
				if err != nil {
					t.Fatal(err)
				}
				return st, Greedy(lg)
			}
			// Alone, each with every block resident: paging must not change
			// a bit of it.
			var alone [2]atOnceRun
			for k := range prompts {
				st, next := place(prompts[k])
				if err := alone[k].decode(st, next, gen); err != nil {
					t.Fatal(err)
				}
				st.Close()
			}
			// At once: both placed and prefilled, one step together so the
			// second's lane is built and charged, then the tightest budget
			// that still leaves two slots: every block pages out and back in
			// every token, under the other session's submissions.
			a, na := place(prompts[0])
			b, nb := place(prompts[1])
			first := atOnce(t, []*State{a, b}, []int32{na, nb}, 1)
			hi := g.Stats().BudgetUsed
			if _, err := g.SetBudget(hi); err != nil || g.Stats().Slots < 2 {
				t.Fatalf("the budget the sessions charged (%d bytes) leaves %d slot(s): %v %s", hi,
					g.Stats().Slots, err, g.Err())
			}
			lo := uint64(0)
			for hi-lo > 4<<10 {
				mid := lo + (hi-lo)/2
				if _, err := g.SetBudget(mid); err == nil && g.Stats().Slots >= 2 {
					hi = mid
				} else {
					lo = mid
				}
			}
			if _, err := g.SetBudget(hi); err != nil {
				t.Fatal(err)
			}
			if s := g.Stats().Slots; s < 2 || s >= m.Cfg.NLayer {
				t.Fatalf("%d slot(s) for %d blocks: the budget does not page", s, m.Cfg.NLayer)
			}
			before := g.Stats()
			rest := atOnce(t, []*State{a, b}, []int32{first[0].ids[0], first[1].ids[0]}, gen-1)
			after := g.Stats()
			for k := range rest {
				got := first[k]
				got.ids = append(got.ids, rest[k].ids...)
				got.logits = append(got.logits, rest[k].logits...)
				sameRun(t, []string{"session 0", "session 1"}[k], got, alone[k])
			}
			ins, outs := after.PageIns-before.PageIns, after.PageOuts-before.PageOuts
			beside := after.RanBeside - before.RanBeside
			t.Logf("%s: %d slot(s) for %d blocks, %d page-in(s) and %d page-out(s) while the sessions "+
				"stepped at once, %d submissions beside another; ids and logits identical to each alone "+
				"with every block resident", g.Name(), after.Slots, m.Cfg.NLayer, ins, outs, beside)
			if ins == 0 || outs == 0 {
				t.Fatalf("%d page-ins and %d page-outs: nothing paged while the sessions ran", ins, outs)
			}
			if a.devCount() != m.Cfg.NLayer || b.devCount() != m.Cfg.NLayer {
				t.Fatalf("%d and %d of %d blocks still the device's: the budget sent blocks home, which "+
					"is not paging", a.devCount(), b.devCount(), m.Cfg.NLayer)
			}
		})
	}
}

// TestSessionsEvictAtOnce: two sessions stepping at once whose histories
// outgrow the KV budget, so their oldest pages go home (kvevict.go) while the
// other session steps, and come back through the pool's free pages for every
// attention. Each is teacher-forced through the ids it chose in a run alone
// with no budget, and its logits held to that run's within the eviction gate's
// band -- as are the same two sessions taking turns a step at a time on one
// goroutine, what two sessions on one device did before they stepped at once.
// Eviction under another session's step must neither lose a page nor hand one
// to the wrong sequence. KVEvictions and KVStreamPasses during the run at once
// are the selection check. The greedy ids that differ are logged, not held:
// a streamed pass's order flips a near tie (one in two hundred on CUDA, where
// one session evicting alone already reads 8e-3).
func TestSessionsEvictAtOnce(t *testing.T) {
	m, err := Open(jlmOf(t, testmodels.Path("stories15M-q8_0.gguf")), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for _, spec := range testdevices.Specs("cuda:0", "vulkan:0", "metal") {
		t.Run(spec, func(t *testing.T) {
			turns := evictAtOnce(t, m, spec, false)
			both := evictAtOnce(t, m, spec, true)
			t.Logf("%s: taking turns %d page(s) home, %d streamed passes, worst NMSE %.3e, %d id(s) "+
				"different; at once %d, %d, %.3e, %d (against each alone with no budget)", spec,
				turns.ev, turns.passes, turns.worst, turns.flips, both.ev, both.passes, both.worst, both.flips)
			if both.ev == 0 || both.passes == 0 {
				t.Fatalf("%d evictions and %d streamed passes at once: the budget never bound", both.ev, both.passes)
			}
			// The eviction gate's own band (TestKVEvictionRunsPastTheBudget):
			// a streamed pass sums its keys in another order, by how many
			// pages are home and when they went, which is the schedule's --
			// taking turns and at once each send different pages home at
			// different steps. A page lost or read as another sequence's is
			// tenths, not thousandths.
			for _, a := range []evictArm{turns, both} {
				if !(a.worst < evictNMSE) {
					t.Fatalf("worst logit NMSE %.3e against the runs alone, bound %.0e: a page was lost or "+
						"read as another's", a.worst, evictNMSE)
				}
			}
		})
	}
}

// evictArm is one arm of TestSessionsEvictAtOnce: the pages sent home, the
// streamed passes, and the logits' worst NMSE and greedy ids against the runs
// alone with no budget.
type evictArm struct {
	ev, passes, flips int
	worst             float64
}

// evictAtOnce runs two sessions of m on a fresh tier over spec: each alone
// with no budget, then both teacher-forced through those runs' ids under a
// budget that makes their histories evict -- at once, or taking turns a step
// at a time.
func evictAtOnce(t *testing.T, m *Model, spec string, together bool) evictArm {
	t.Helper()
	prompts := atOncePrompts(m)
	const seq, gen = 128, 100
	g := atOnceTier(t, spec, func(c *tier.Config) {
		c.KVPage = 64 // the smallest page: a short history is two
		c.StagedDecode = true
	})
	var alone [2]atOnceRun
	var first [2]int32
	for k := range prompts {
		st, next := atOnceState(t, m, g, seq, prompts[k])
		first[k] = next
		if err := alone[k].decode(st, next, gen); err != nil {
			t.Fatal(err)
		}
		st.Close()
	}
	sts := make([]*State, len(prompts))
	for k := range prompts {
		sts[k], _ = atOnceState(t, m, g, seq, prompts[k])
	}
	got := make([][][]float32, len(sts))
	errs := make([]error, len(sts))
	// step forces session k through its step i.
	step := func(k, i int) {
		if errs[k] != nil {
			return
		}
		in := first[k]
		if i > 0 {
			in = alone[k].ids[i-1]
		}
		lg, err := sts[k].Forward(in)
		if err != nil {
			errs[k] = err
			return
		}
		got[k] = append(got[k], slices.Clone(lg))
	}
	forced := func(from, to int) {
		if together {
			var wg sync.WaitGroup
			for k := range sts {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for i := from; i < to; i++ {
						step(k, i)
					}
				}()
			}
			wg.Wait()
		} else {
			for i := from; i < to; i++ {
				for k := range sts {
					step(k, i)
				}
			}
		}
		for k, err := range errs {
			if err != nil {
				t.Fatalf("session %d: %v (%s)", k, err, g.Err())
			}
		}
	}
	// One step before the budget is cut, so a lane the second session runs
	// in at once is built and charged first; then a little slack past what
	// the histories hold: they can grow only by sending their oldest pages
	// home.
	forced(0, 1)
	if _, err := g.SetBudget(g.Stats().BudgetUsed + 256<<10); err != nil {
		t.Fatal(err)
	}
	before := g.Stats()
	forced(1, gen)
	after := g.Stats()
	for k, st := range sts {
		if st.devCount() != m.Cfg.NLayer {
			t.Fatalf("session %d holds %d of %d blocks on the device after the run: eviction must not "+
				"move blocks home (%s)", k, st.devCount(), m.Cfg.NLayer, g.Err())
		}
	}
	r := evictArm{ev: after.KVEvictions - before.KVEvictions, passes: after.KVStreamPasses - before.KVStreamPasses}
	for k := range got {
		for i, lg := range got[k] {
			if e := logitNMSE(lg, alone[k].logits[i]); !(e <= r.worst) {
				r.worst = e
			}
			if Greedy(lg) != alone[k].ids[i] {
				r.flips++
			}
		}
	}
	return r
}

// TestSessionsRelocateAtOnce: one session relocates while another decodes
// beside it on the same device -- its seam moves half its blocks home and
// back, then the whole sequence hops to a second tier with its history --
// and both produce exactly the ids and logits each produces alone, the
// relocating one with the same moves at the same steps. A move is a
// structural change of what the other session's submissions read beside it
// (blocks freed and rebuilt, KV pages migrated), so it is held to the same bar
// as a step: RanBeside says the two did run at once.
func TestSessionsRelocateAtOnce(t *testing.T) {
	defer fusedRunsCheck(t, false)()
	m, err := Open(jlmOf(t, testmodels.Path("stories15M-q8_0.gguf")), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	// half puts the session beside the relocating one on the first half of
	// the blocks only, so the relocating session's seam move frees the second
	// half outright and its way back prepares them again -- allocations and
	// uploads beside the other's submissions, not only its own history going.
	for _, half := range []bool{false, true} {
		for _, spec := range testdevices.Specs("cuda:0", "vulkan:0", "metal") {
			t.Run(fmt.Sprintf("%s/half=%v", spec, half), func(t *testing.T) {
				relocateAtOnce(t, m, atOncePrompts(m), spec, half, !half, 0)
			})
		}
	}
}

// TestSessionsRelocateAtOnceComposed is TestSessionsRelocateAtOnce with the
// composer serving the blocks and the head: the relocating session's moves
// free its blocks' instances in every lane and form them again at the
// placement, beside the other session's composed submissions, and the hop
// forms them on the second tier. No form is built at a step on either tier.
func TestSessionsRelocateAtOnceComposed(t *testing.T) {
	m, err := Open(jlmOf(t, testmodels.Path("stories15M-q8_0.gguf")), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for _, half := range []bool{false, true} {
		for _, spec := range testdevices.Specs("cuda:0", "vulkan:0", "metal") {
			t.Run(fmt.Sprintf("%s/half=%v", spec, half), func(t *testing.T) {
				relocateAtOnce(t, m, atOncePrompts(m), spec, half, !half, tier.ComposedBlock|tier.ComposedHead)
			})
		}
	}
}

// TestSessionsRelocateAtOnceHybrid is TestSessionsRelocateAtOnce on the
// hybrids of TestSessionsStepAtOnceHybrid: the relocating session's seam
// move carries its recurrent state home and back and its hop carries it to
// a second tier, while the other session's state stays in its own seat
// beside it, with the other session on every block and on half of them. A
// submission holding a linear block runs alone by design, so RanBeside is
// logged and not held.
func TestSessionsRelocateAtOnceHybrid(t *testing.T) {
	for _, o := range hybridAtOnce {
		m := hybridModelOpt(t, o.opt)
		defer m.Close()
		for _, half := range []bool{false, true} {
			for _, spec := range testdevices.Specs("cuda:0", "vulkan:0", "metal") {
				t.Run(fmt.Sprintf("%s/%s/half=%v", o.name, spec, half), func(t *testing.T) {
					relocateAtOnce(t, m, hybridPrompts, spec, half, false, 0)
				})
			}
		}
	}
}

// relocateAtOnce is TestSessionsRelocateAtOnce's arm for m on spec. beside
// demands that the device ran a submission beside another's; parts names
// what the composer serves.
func relocateAtOnce(t *testing.T, m *Model, prompts [2][]int32, spec string, half, beside bool, parts tier.Parts) {
	t.Helper()
	const seq = 256
	nl := m.Cfg.NLayer
	var cfg func(*tier.Config)
	if parts != 0 {
		cfg = func(c *tier.Config) { c.Composed = parts }
	}
	g := atOnceTier(t, spec, cfg)
	g2 := atOnceTier(t, spec, cfg)
	place := func(prompt []int32, blocks int) (*State, int32) {
		return atOncePlace(t, m, g, seq, prompt, blocks)
	}
	bBlocks := nl
	if half {
		bBlocks = nl / 2
	}
	// moves is the relocating session's run: at step 8 half its blocks
	// go home, at 16 they come back, at 24 it hops to g2.
	moves := func(st *State, next int32, r *atOnceRun) error {
		for i := range atOnceGen {
			switch i {
			case 8:
				if got := st.SetGPULayers(nl / 2); got != nl/2 {
					return fmt.Errorf("the seam moved to %d blocks, asked %d", got, nl/2)
				}
			case 16:
				if got := st.SetGPULayers(nl); got != nl {
					return fmt.Errorf("the seam came back to %d blocks, asked %d", got, nl)
				}
				if parts != 0 {
					if err := lanesHold(g, nl, parts, 1); err != nil {
						return err
					}
				}
			case 24:
				if err := st.SetDevice(g2); err != nil {
					return err
				}
				if st.GPULayers() != nl {
					return fmt.Errorf("%d of %d blocks on the second tier: %s", st.GPULayers(), nl, g2.Err())
				}
			}
			if err := r.decode(st, next, 1); err != nil {
				return err
			}
			next = r.ids[len(r.ids)-1]
		}
		return nil
	}
	var alone [2]atOnceRun
	st, next := atOnceState(t, m, g, seq, prompts[0])
	if err := moves(st, next, &alone[0]); err != nil {
		t.Fatal(err)
	}
	st.Close()
	st, next = place(prompts[1], bBlocks)
	if err := alone[1].decode(st, next, atOnceGen); err != nil {
		t.Fatal(err)
	}
	st.Close()

	before, before2 := g.Stats(), g2.Stats()
	a, na := atOnceState(t, m, g, seq, prompts[0])
	b, nb := place(prompts[1], bBlocks)
	var runs [2]atOnceRun
	var errs [2]error
	var wg sync.WaitGroup
	start := make(chan struct{})
	wg.Add(2)
	go func() { defer wg.Done(); <-start; errs[0] = moves(a, na, &runs[0]) }()
	go func() { defer wg.Done(); <-start; errs[1] = runs[1].decode(b, nb, atOnceGen) }()
	close(start)
	wg.Wait()
	for k, err := range errs {
		if err != nil {
			t.Fatalf("session %d: %v", k, err)
		}
	}
	after := g.Stats()
	sameRun(t, "the relocating session", runs[0], alone[0])
	sameRun(t, "the session beside it", runs[1], alone[1])
	ran := after.RanBeside - before.RanBeside
	t.Logf("%s, the other session on %d of %d blocks: %d steps each with the seam moved twice and a "+
		"hop to a second tier mid-run, ids and logits identical to each alone; %d submissions started "+
		"beside another in flight", g.Name(), bBlocks, nl, atOnceGen, ran)
	// With the other session on half the blocks, the two have only
	// the relocating one's device steps on the first tier to overlap,
	// and sometimes do not: that arm is held to the answers, the full
	// one to running beside each other as well.
	if beside && ran == 0 {
		t.Fatal("no submission ran on the device while another's did: the sessions ran one after another")
	}
	if parts != 0 {
		if err := lanesHold(g, bBlocks, parts, 2); err != nil {
			t.Fatal(err)
		}
		if err := lanesHold(g2, nl, parts, 1); err != nil {
			t.Fatal(err)
		}
		after2 := g2.Stats()
		for k, s := range [2][2]tier.Stats{{before, after}, {before2, after2}} {
			n, d, _ := s[1].ServedPart(parts)
			n0, d0, _ := s[0].ServedPart(parts)
			if n-n0 == 0 || d-d0 != 0 {
				t.Fatalf("tier %d: the composer served %d part steps and declined %d", k+1, n-n0, d-d0)
			}
			if f := s[1].FormedAtStep - s[0].FormedAtStep; f != 0 {
				t.Fatalf("tier %d: %d composed part(s) formed at a step: a move left a lane without its instance", k+1, f)
			}
		}
	}
}

// TestModelsAndSessionsAtOnce: two models on one device, each through a tier
// of its own as the app and the server load them, and two sessions of each --
// four sequences decoding at once from four goroutines -- produce exactly
// the ids and logits each produces alone. Different models share no weights,
// pages or scratch, only the card, its memory and its queues; sessions of one
// model share its tier. RanBeside on both tiers says each model's sessions
// ran beside each other, and the spans that the two models' did too.
func TestModelsAndSessionsAtOnce(t *testing.T) {
	names := []string{"stories15M-q8_0.gguf", "SmolLM2-360M-Instruct-Q8_0.gguf"}
	var ms []*Model
	for _, n := range names {
		m, err := Open(jlmOf(t, testmodels.Path(n)), noTune)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		ms = append(ms, m)
	}
	const seq = 256
	for _, spec := range testdevices.Specs("cuda:0", "vulkan:0", "metal") {
		t.Run(spec, func(t *testing.T) {
			var gs []*tier.GPU
			for range ms {
				gs = append(gs, atOnceTier(t, spec, nil))
			}
			// The four sequences: model k's prompt p.
			type seqOf struct{ m, p int }
			var all []seqOf
			for mi := range ms {
				for p := range 2 {
					all = append(all, seqOf{mi, p})
				}
			}
			prompt := func(s seqOf) []int32 { return atOncePrompts(ms[s.m])[s.p] }
			alone := make([]atOnceRun, len(all))
			for i, s := range all {
				st, next := atOnceState(t, ms[s.m], gs[s.m], seq, prompt(s))
				if err := alone[i].decode(st, next, atOnceGen); err != nil {
					t.Fatal(err)
				}
				st.Close()
			}
			var before []tier.Stats
			for _, g := range gs {
				before = append(before, g.Stats())
			}
			var sts []*State
			var next []int32
			for _, s := range all {
				st, n := atOnceState(t, ms[s.m], gs[s.m], seq, prompt(s))
				sts, next = append(sts, st), append(next, n)
			}
			runs := atOnce(t, sts, next, atOnceGen)
			for i, s := range all {
				sameRun(t, fmt.Sprintf("%s session %d", names[s.m], s.p), runs[i], alone[i])
			}
			for mi, g := range gs {
				st := g.Stats()
				beside := st.RanBeside - before[mi].RanBeside
				t.Logf("%s on %s: %d submissions ran on the device beside another session's, %d call(s) "+
					"waited for a lane, scratch %d bytes, budget used %d: %s", names[mi], g.Name(), beside,
					st.LanesWaited-before[mi].LanesWaited, st.ScratchBytes, st.BudgetUsed, g.Err())
				// With no room on the card for a second lane, a model's sessions
				// take turns by design (tier takeLane), which LanesWaited counts;
				// never beside each other with no wait is the sessions serialised.
				if beside == 0 && st.LanesWaited == before[mi].LanesWaited {
					t.Fatalf("%s's two sessions never ran beside each other, and no call waited for a lane", names[mi])
				}
			}
			// The two models' sequences overlap on the wall clock: one model's
			// tier did not wait for the other's.
			if ov := overlaps(runs[0], runs[2]); ov == 0 {
				t.Fatal("no step of one model overlapped the other's: the models ran one after another")
			} else {
				t.Logf("%d of %s's steps overlapped %s's", ov, names[0], names[1])
			}
		})
	}
}
