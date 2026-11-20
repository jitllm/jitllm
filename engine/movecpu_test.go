package engine

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/nn"
)

// TestBlocksMoveToTheCPUAndBack: the CPU is a device like the cards, and one
// MoveBlocks reaches it. Mid-sequence every block goes from one card to the
// CPU, from the CPU to the other card by name, half of them to the CPU again,
// and all of them to the first card -- the CPU half placed there, the card
// half moved card to card -- each with its history, teacher-forced on a
// CPU control's tokens. Every block is where it was sent (WhereIs and
// Locations name it), and every token stays in the band of the control.
//
// VIOLATION SIGNATURE. Drop the history from moveOffCards (no migrateKV) and the
// tokens after the first move leave the band.
func TestBlocksMoveToTheCPUAndBack(t *testing.T) {
	names := append([]string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf"}, relocModels()...)
	for _, name := range names {
		t.Run(name, func(t *testing.T) { blocksMoveToTheCPUAndBack(t, jlmOf(t, testmodels.Path(name)), false) })
	}
	// The CPU's blocks on what the composer forms for it: each block composed
	// for the CPU at its placement and again at each move back, and every
	// block step the CPU runs between the moves served.
	for _, name := range cpuServedModels() {
		t.Run(name+"/composed-cpu", func(t *testing.T) { blocksMoveToTheCPUAndBack(t, jlmOf(t, testmodels.Path(name)), true) })
	}
}

// composedMoves holds st's composed CPU census to the moves: on the control
// the prompt's and every decode step of every block served; on the moved
// run each block
// composed at the instance's placement, again at the move to the CPU and once
// more for the upper half, and the block steps between the moves served.
func composedMoves(t *testing.T, st *Instance, n, half int, devices bool, ntok, toCPU, toB, halfCPU, toA int) {
	t.Helper()
	served, declined := st.Account(nn.CPU).Steps.Of(nn.ComposedBlock)
	for i, k := range declined {
		if k > 0 {
			t.Errorf("%d block steps declined on the CPU: %s (%v)", k, nn.DeclineName(i), st.Account(nn.CPU).FormErr)
		}
	}
	if !devices {
		// The prompt's chunks, every block of each, and the decode's steps.
		if served < (ntok+1)*n || served%n != 0 {
			t.Errorf("the composed CPU served %d block steps, where the CPU ran the prompt and %d", served, ntok*n)
		}
		return
	}
	if want := (toB-toCPU)*n + (toA-halfCPU)*(n-half); served != want {
		t.Errorf("the composed CPU served %d block steps, where the CPU ran %d", served, want)
	}
	for li, k := range st.Account(nn.CPU).Forms {
		w := 2
		if li >= half {
			w = 3
		}
		if k != w {
			t.Errorf("block %d was composed for the CPU %d times, where it was placed there %d times", li, k, w)
		}
	}
}

func blocksMoveToTheCPUAndBack(t *testing.T, path string, composed bool) {
	a, b := moveDevices(t)
	g, err := openGPU(t, gpu.WithDevices(a+","+b), gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || g == nil {
		noDevice(t, a+","+b, err)
	}
	defer g.Close()
	distinctCards(t, g)
	opt := WithKVF16(false)
	if composed {
		// The composed CPU keeps its history binary16 (composer.Halves).
		opt = WithKVF16(true)
	}
	instOpts := []Option{opt}
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	n := m.Cfg.NLayer
	const ntok, toCPU, toB, halfCPU, toA = 12, 2, 5, 8, 10
	prompt := m.Vocab.Encode("The capital of France is a city that", true)

	// where fails unless every block of [lo, hi) is on dev, by both reads.
	where := func(st *Instance, lo, hi int, dev string) {
		t.Helper()
		locs := st.Locations()
		for li := lo; li < hi; li++ {
			if got := st.WhereIs(li); got != nn.DeviceName(dev) || locs[li] != got {
				t.Fatalf("block %d is on %q (Locations %q), want %q", li, got, locs[li], dev)
			}
		}
	}
	move := func(st *Instance, lo, hi int, to string) BlockMove {
		t.Helper()
		mv, err := st.MoveBlocks(lo, hi, to)
		if err != nil {
			t.Fatalf("blocks %d..%d to %s: %v (%s)", lo, hi-1, to, err, g.Err())
		}
		if len(mv.Declined) > 0 {
			t.Fatalf("%s declined blocks %v: %s", to, mv.Declined, g.Err())
		}
		where(st, lo, hi, to)
		return mv
	}
	half := n / 2
	run := func(ids []int32, devices, moves bool) [][]float32 {
		st := testInstance(t, m, len(prompt)+ntok+1, instOpts...)
		defer st.Close()
		if composed && devices == moves {
			defer func() { composedMoves(t, st, n, half, devices, ntok, toCPU, toB, halfCPU, toA) }()
		}
		if devices {
			if err := st.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			move(st, 0, n, a)
		} else {
			where(st, 0, n, nn.CPU)
		}
		logits, err := st.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		var lg [][]float32
		for i := 0; i < ntok; i++ {
			lg = append(lg, append([]float32(nil), logits...))
			next := Greedy(logits)
			if ids != nil {
				next = ids[i]
			}
			if moves {
				switch i {
				case toCPU:
					if mv := move(st, 0, n, nn.CPU); len(mv.Moved) != n || st.devCount() != 0 {
						t.Fatalf("%d blocks moved to the CPU, %d left on a card", len(mv.Moved), st.devCount())
					}
				case toB:
					if mv := move(st, 0, n, b); len(mv.Moved) != n {
						t.Fatalf("%d of %d blocks moved from the CPU to %s", len(mv.Moved), n, b)
					}
				case halfCPU:
					// A KV-sharing group the cut splits comes to the CPU
					// whole (fixKVGroups), so the cut is where none crosses.
					for li := 0; li < n; li++ {
						if glo, ghi := st.kvSpan(li); glo < half && ghi > half {
							half, li = glo, -1
						}
					}
					move(st, half, n, nn.CPU)
					where(st, 0, half, b)
				case toA:
					if mv := move(st, 0, n, a); len(mv.Moved) != n {
						t.Fatalf("%d of %d blocks moved from the CPU and %s to %s", len(mv.Moved), n, b, a)
					}
				}
			}
			if logits, err = st.Forward(next); err != nil {
				t.Fatal(err)
			}
		}
		return lg
	}
	ctl := run(nil, false, false)
	var ids []int32
	for _, l := range ctl {
		ids = append(ids, Greedy(l))
	}
	moved, still := run(ids, true, true), run(ids, true, false)
	// The card parts from the CPU in the reduction-order band, which an
	// absolute logit difference cannot state: it grows with the logits'
	// scale (mamba-130m's reach 43, and a coherent prefill reads 3.18 apart).
	// The band is the card census's; a history a move dropped or misplaced
	// reads in tenths at every token after it. A recurrent state carries the
	// prefill card's arithmetic forward and a token can amplify it:
	// mamba-130m's sixth token reads 5.6e-2 from the CPU with its prompt on
	// CUDA and 0.17 on Vulkan, to four places whichever card the blocks move
	// to next. A token is held to the band, or to twice what the card reads
	// there never moved: a dropped state reads in tenths where the unmoved
	// card agrees with the CPU. A mixture's router reads the card's own
	// arithmetic too and can cross a near-tie at a token
	// (synth-qwen35moe-hybrid-mtp-q8_0 reads 0.117 at one); the unmoved card
	// crosses it at the same token, so a mixture is held the same way.
	const moveBand = 5e-2
	var worstAll, nmseAll, stillAll float64
	var es []float64
	for i := range ctl {
		var worst float64
		for j, v := range ctl[i] {
			worst = max(worst, math.Abs(float64(v-moved[i][j])))
		}
		e, _ := nmseOf(moved[i], ctl[i])
		s, _ := nmseOf(still[i], ctl[i])
		band := max(moveBand, 2*s)
		es = append(es, e)
		switch {
		case math.IsNaN(worst) || math.IsNaN(e):
			t.Fatalf("token %d: a non-finite logit", i)
		case !(e < band):
			t.Fatalf("token %d: %.3e NMSE from the CPU control (worst logit %.4g), outside the band %.3g (the unmoved card %.3e)", i, e, worst, band, s)
		}
		worstAll, nmseAll, stillAll = max(worstAll, worst), max(nmseAll, e), max(stillAll, s)
	}
	t.Logf("%d tokens, %d blocks moved %s -> cpu -> %s -> half cpu -> %s; worst %.3e NMSE (logit %.4g), median %.3e, from the CPU control; unmoved %.3e",
		ntok, n, a, b, a, nmseAll, worstAll, median(es), stillAll)
}

// TestTheProjectionComesBackFromTheCPU: a projection on a card stays there
// while blocks move to the CPU and back. One the engine placed beside the
// blocks on the first card comes to the CPU with the last of them, its card
// giving its copy back, and goes back with the next move to a card; one the
// placement named on the second card, which holds no block, stays there
// through every move of blocks and moves only when a move names it
// (MoveHead): to the CPU, the card giving its copy back, and to its card
// again. Teacher-forced on a CPU control's tokens, each token is counted on
// the card the tier ran its projection on (Stats.Heads), and on none where
// the CPU projects; the card's ledger holds the projection's bytes while it
// is there and none once it went (Stats.HeadBytes, BudgetUsed); every move
// reports whether the projection moved; and the logits are held to the
// control's inside three times what the same placement reads never moved.
//
// VIOLATION SIGNATURE. Take the projection off its card at every move to the
// CPU (moveOffCards, as it once did) and the token after the first move counts
// no projection on the card; let a named projection follow the blocks (as
// the move once did) and the apart arm's move of the last block reports it
// moved; keep the card's copy when it comes home (as the engine once did) and
// the card still holds its bytes.
func TestTheProjectionComesBackFromTheCPU(t *testing.T) {
	m, err := model.Open(jlmOf(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a, b := moveDevices(t)
	t.Run("beside", func(t *testing.T) { projectionComesBack(t, m, a, b, "") })
	t.Run("apart", func(t *testing.T) { projectionComesBack(t, m, a, b, b) })
}

// projectionComesBack is TestTheProjectionComesBackFromTheCPU's arm with
// every block on a and the projection named on named, or placed by the
// engine where named is "".
func projectionComesBack(t *testing.T, m *model.Model, a, b, named string) {
	n := m.Cfg.NLayer
	const ntok = 12
	prompt := m.Vocab.Encode("The capital of France is a city that", true)
	io := []Option{WithKVF16(false)}
	cpu := testInstance(t, m, len(prompt)+ntok+1, io...)
	lg, err := cpu.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float32{append([]float32(nil), lg...)}
	var ids []int32
	for range ntok {
		ids = append(ids, Greedy(lg))
		if lg, err = cpu.Forward(ids[len(ids)-1]); err != nil {
			t.Fatal(err)
		}
		want = append(want, append([]float32(nil), lg...))
	}
	cpu.Close()

	g, err := openGPU(t, gpu.WithDevices(a+","+b), gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || g == nil {
		noDevice(t, a+","+b, err)
	}
	defer g.Close()
	distinctCards(t, g)
	cards := g.Hardware()
	cardOf := func(name string) int {
		for d, c := range cards {
			if c.Name == name {
				return d
			}
		}
		t.Fatalf("no card is named %s", name)
		return -1
	}
	type step struct {
		lo, hi int
		to     string
		head   bool
	}
	moves := map[int]step{2: {n - 1, n, nn.CPU, false}, 4: {n - 1, n, a, false}, 6: {0, n, nn.CPU, false}, 8: {0, n, a, false}}
	pl := Placement{Rest: &Place{On: a}, Strict: true}
	head := a
	if named != "" {
		pl.Head, head = &Place{On: named}, named
		moves[7], moves[9] = step{to: nn.CPU, head: true}, step{to: named, head: true}
	}
	run := func(moved bool) []float64 {
		st := testInstance(t, m, len(prompt)+ntok+1, append(append([]Option(nil), io...), WithPlacement(pl))...)
		defer st.Close()
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		for li := range n {
			if at, _ := st.DeviceOf(li); at != a {
				t.Fatalf("block %d is on %q, placed on %s", li, at, a)
			}
		}
		headOn(t, st, head)
		lg, err := st.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		e, _ := nmseOf(lg, want[0])
		got := []float64{e}
		on := head
		for i, id := range ids {
			if s, ok := moves[i]; ok && moved {
				was, from := on, -1
				var held, used uint64
				if on != nn.CPU {
					from = cardOf(on)
					held, used = g.DevStats()[from].HeadBytes, g.DevStats()[from].BudgetUsed
				}
				var mv BlockMove
				if s.head {
					if mv, err = st.MoveHead(s.to); err != nil {
						t.Fatalf("token %d: the projection to %s: %v (%s)", i, s.to, err, g.Err())
					}
					on = s.to
				} else {
					if mv, err = st.MoveBlocks(s.lo, s.hi, s.to); err != nil {
						t.Fatalf("token %d: blocks %d..%d to %s: %v (%s)", i, s.lo, s.hi-1, s.to, err, g.Err())
					}
					for li := s.lo; li < s.hi; li++ {
						if at := st.WhereIs(li); at != nn.DeviceName(s.to) {
							t.Fatalf("token %d: block %d is on %q, moved to %s", i, li, at, s.to)
						}
					}
					if named == "" {
						// Placed by the engine beside the blocks: it leaves
						// its card with the last of them, and comes back
						// with them.
						switch {
						case s.to == nn.CPU && s.lo == 0:
							on = nn.CPU
						case s.to != nn.CPU:
							on = s.to
						}
					}
				}
				if mv.Head != (on != was) {
					t.Errorf("token %d: the move reports the projection moved %v, from %s to %s", i, mv.Head, was, on)
				}
				if on != was && from >= 0 {
					ds := g.DevStats()[from]
					if held == 0 || ds.HeadBytes != 0 || used < ds.BudgetUsed+held {
						t.Fatalf("token %d: the projection left %s, which held %d bytes of it and holds %d; its ledger %d -> %d",
							i, was, held, ds.HeadBytes, used, ds.BudgetUsed)
					}
					t.Logf("token %d: the projection left %s: %d bytes of it, the ledger %d -> %d", i, was, held, used, ds.BudgetUsed)
				}
				if on != nn.CPU && g.DevStats()[cardOf(on)].HeadBytes == 0 {
					t.Fatalf("token %d: the projection is on %s, which holds none of its bytes", i, on)
				}
			}
			before := g.DevStats()
			if lg, err = st.Forward(id); err != nil {
				t.Fatalf("token %d: %v", i, err)
			}
			for d, ds := range g.DevStats() {
				w := 0
				if cards[d].Name == on {
					w = 1
				}
				if ran := ds.Heads - before[d].Heads; ran != w {
					t.Fatalf("token %d: %s ran the projection %d time(s), want %d: the projection is on %s", i, cards[d].Name, ran, w, on)
				}
			}
			if on == nn.CPU {
				if st.HeadOnDevice() {
					t.Fatalf("token %d: the instance projects on a card, where its projection came to the CPU", i)
				}
			} else {
				headOn(t, st, on)
			}
			e, _ := nmseOf(lg, want[i+1])
			if math.IsNaN(e) {
				t.Fatalf("token %d: a non-finite logit", i)
			}
			got = append(got, e)
		}
		return got
	}
	still := run(false)
	got := run(true)
	for i := range got {
		if bound := max(3*still[i], 1e-6); got[i] > bound {
			t.Errorf("step %d: NMSE %.3e from the CPU's, above %.3e (unmoved %.3e)", i, got[i], bound, still[i])
		}
	}
	t.Logf("projection on %s: NMSE from the CPU's, moved %.2e; unmoved %.2e", head, got, still)
}

// TestAReleasedProjectionLeavesItsCard: a projection that leaves its card
// leaves its bytes with it, the per-row logits it was admitted with among
// them. Every block and the projection on one card, a move naming the CPU
// for the projection (MoveHead) gives the card's copy back -- the card's
// ledger holds none of it (Stats.HeadBytes) and none of the per-row logits
// (Stats.RowLogitsBytes), and spends at least what both held less
// (BudgetUsed) -- a move naming the card places both there again, and a seam
// shrinking off the card gives them back as well (SetGPULayers). The stepped
// arm reserves a ragged step (Config.StepRows), whose reservation holds its
// width's per-row logits for every projection to come: they stay through
// each release, and the projection's own bytes go. Teacher-forced on a CPU
// control's tokens, each token is counted on the card when the projection is
// there and on none otherwise (Stats.Heads), and its logits are held to the
// control's inside three times what the placement reads never moved.
//
// VIOLATION SIGNATURE. Make the tier's release keep the copy on a card that
// still holds blocks (devTier.dropHead's guard, as it once was) and the card
// holds the projection's bytes after the move to the CPU; keep the per-row
// logits (no dropRowsHead) and it holds those.
func TestAReleasedProjectionLeavesItsCard(t *testing.T) {
	m, err := model.Open(jlmOf(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	card := testdevices.Take(t, 1, "cuda:0")[0]
	t.Run("alone", func(t *testing.T) { releasedProjection(t, m, card, false) })
	t.Run("stepped", func(t *testing.T) { releasedProjection(t, m, card, true) })
}

// releasedProjection is TestAReleasedProjectionLeavesItsCard's arm on card,
// a ragged step reserved there where stepped.
func releasedProjection(t *testing.T, m *model.Model, card string, stepped bool) {
	n := m.Cfg.NLayer
	const ntok = 8
	prompt := m.Vocab.Encode("The capital of France is a city that", true)
	io := []Option{WithKVF16(false)}
	cpu := testInstance(t, m, len(prompt)+ntok+1, io...)
	lg, err := cpu.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	var want [][]float32
	var ids []int32
	for range ntok {
		ids = append(ids, Greedy(lg))
		if lg, err = cpu.Forward(ids[len(ids)-1]); err != nil {
			t.Fatal(err)
		}
		want = append(want, append([]float32(nil), lg...))
	}
	cpu.Close()

	gopts := []gpu.Option{gpu.WithDevices(card), gpu.WithDeviceTune(gpu.TuneOff)}
	if stepped {
		gopts = append(gopts, gpu.WithConfig(func(c *gpu.Config) { c.StepRows = MaxStepRows }))
	}
	g, err := openGPU(t, gopts...)
	if err != nil || g == nil {
		noDevice(t, card, err)
	}
	defer g.Close()
	const (
		toCPU = iota + 1
		toCard
		shrink
	)
	acts := map[int]int{1: toCPU, 3: toCard, 5: shrink}
	run := func(moved bool) []float64 {
		st := testInstance(t, m, len(prompt)+ntok+1,
			append(append([]Option(nil), io...), WithPlacement(Placement{Rest: &Place{On: card}, Head: &Place{On: card}, Strict: true}))...)
		defer st.Close()
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		headOn(t, st, card)
		if _, err := st.Prefill(prompt); err != nil {
			t.Fatal(err)
		}
		on := true
		var got []float64
		for i, id := range ids {
			if act := acts[i]; moved && act != 0 {
				ds := g.DevStats()[0]
				held, rows, used := ds.HeadBytes, ds.RowLogitsBytes, ds.BudgetUsed
				if on && (held == 0 || rows == 0) {
					t.Fatalf("token %d: %s holds the projection, %d bytes of it and %d of its per-row logits", i, card, held, rows)
				}
				if stepped && ds.ReservedStep == 0 {
					t.Fatalf("token %d: %s reserved no ragged step", i, card)
				}
				switch act {
				case toCPU:
					if mv, err := st.MoveHead(nn.CPU); err != nil || !mv.Head {
						t.Fatalf("token %d: the projection to the CPU: moved %v, %v", i, mv.Head, err)
					}
				case toCard:
					if mv, err := st.MoveHead(card); err != nil || !mv.Head {
						t.Fatalf("token %d: the projection to %s: moved %v, %v: %s", i, card, mv.Head, err, g.Err())
					}
				case shrink:
					if got := st.SetGPULayers(n - 1); got != n-1 {
						t.Fatalf("token %d: the seam to %d took %d: %s", i, n-1, got, g.Err())
					}
				}
				on = act == toCard
				ds = g.DevStats()[0]
				// The per-row logits left on the card: none, or the step's
				// reservation's where one holds them.
				kept := ds.RowLogitsBytes
				switch {
				case on && (ds.HeadBytes == 0 || kept == 0):
					t.Fatalf("token %d: the projection is back on %s, which holds %d bytes of it and %d of its per-row logits",
						i, card, ds.HeadBytes, kept)
				case !on && (ds.HeadBytes != 0 || stepped != (kept > 0) || kept > rows || used < ds.BudgetUsed+held+rows-kept):
					t.Fatalf("token %d: the projection left %s, which held %d bytes of it and holds %d, and %d of per-row logits and holds %d (a step reserved %v); its ledger %d -> %d",
						i, card, held, ds.HeadBytes, rows, kept, stepped, used, ds.BudgetUsed)
				case !on:
					t.Logf("token %d: the projection left %s: %d bytes of it and %d of per-row logits given back, %d kept; the ledger %d -> %d",
						i, card, held, rows-kept, kept, used, ds.BudgetUsed)
				}
			}
			before := g.DevStats()[0].Heads
			if lg, err = st.Forward(id); err != nil {
				t.Fatalf("token %d: %v", i, err)
			}
			w := 0
			if on {
				w = 1
			}
			if ran := g.DevStats()[0].Heads - before; ran != w {
				t.Fatalf("token %d: %s ran the projection %d time(s), want %d", i, card, ran, w)
			}
			if st.HeadOnDevice() != on {
				t.Fatalf("token %d: the instance projects on the card %v, want %v", i, st.HeadOnDevice(), on)
			}
			e, _ := nmseOf(lg, want[i])
			if math.IsNaN(e) {
				t.Fatalf("token %d: a non-finite logit", i)
			}
			got = append(got, e)
		}
		return got
	}
	still := run(false)
	got := run(true)
	for i := range got {
		if bound := max(3*still[i], 1e-6); got[i] > bound {
			t.Errorf("step %d: NMSE %.3e from the CPU's, above %.3e (unmoved %.3e)", i, got[i], bound, still[i])
		}
	}
	t.Logf("NMSE from the CPU's, moved %.2e; unmoved %.2e", got, still)
}
