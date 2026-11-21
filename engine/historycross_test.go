package engine

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// TestAHistoryReadAcrossTheCPU: a block reading another block's history
// (Gemma 3n's and Gemma 4's shared-KV layers) runs on a card while the block
// it reads runs on the CPU device, and the reverse. The side a reader runs on
// holds a copy of the history, appended from the other side's rows as the
// source writes them: the card's copy from the CPU's pages, the CPU device's
// from the card's.
//
// The blocks up to the first source read from above it start on the CPU
// device and the rest on the card; the prompt and every token after it are
// held to the CPU's logits, inside three times the band one card holding
// every block stands at, with the CPU's greedy pick. Mid-sequence the
// source moves to the card beside its readers, back to the CPU, the readers
// follow it there, the blocks below them go to the card (the readers now
// read the CPU device's copy), the readers go back up and the blocks below
// them home again: each block where the move put it, no group pulled whole,
// and after every move the copies held are the reads across devices -- one
// a reader's device and source on a card, one a source on the card read on
// the CPU device -- and the CPU device holds no page of a card's block no
// block of it reads. Each move is held to the logits of the token after it.
//
// The reverse placement, the readers on the CPU device from the start, runs
// the prompt and the tokens without moves, and again with the CPU device's
// copy taking no rows once made (crossStale): the logits must leave the
// band, or the gate cannot see a copy that is not appended.
//
// Two instances split so on one card step their rows together (StepRuns)
// and a call each in turn, both ways (historyStepAcrossTheCPU); and two
// sessions sharing a prompt's prefix through the prefix cache run the split
// placement both ways: the second restores the prefix, both are held to the
// CPU's logits for their prompts, and the prefix is held once on each
// device (historySharedAcrossTheCPU). synth-gemma3n restated to keep each
// block's history past its rows (pastTheRows) parts its groups so too, the
// copies its entries, and so does a block reading two others' histories
// past its own (readsTwoFixture). Two blocks on the card reading two
// histories of one block on the CPU device, the lower the history past its
// rows alone (openKeepsTwo), read each from the card's copy, the copy
// widened to the second as its reader is placed. A history past a block's
// rows is named for the prefix cache where its block runs, the card's pages
// of it on the card, so the restated fixtures run the shared arms too.
func TestAHistoryReadAcrossTheCPU(t *testing.T) {
	for _, f := range []struct {
		name string
		open func(*testing.T, string) (*model.Model, *c6Golden, []Option)
	}{{"synth-gemma4-e", openGemma4}, {"synth-gemma3n", openGemma3n},
		{"synth-gemma3n-past-the-rows", openPastTheRows(openGemma3n, "synth-gemma3n", 1)}, {"synth-seedoss-readstwo", openReadsTwo},
		{"synth-seedoss-keepstwo", openKeepsTwo}} {
		t.Run(f.name, func(t *testing.T) {
			m, _, io := f.open(t, f.name)
			defer m.Close()
			for _, dev := range testdevices.Specs("cuda:0", "vulkan:0") {
				t.Run(dev, func(t *testing.T) { historyAcrossTheCPU(t, m, io, dev, false, nil) })
				t.Run(dev+"/readers-on-the-cpu", func(t *testing.T) { historyAcrossTheCPU(t, m, io, dev, true, nil) })
				t.Run(dev+"/cpu-copy-stale", func(t *testing.T) {
					got, band := historyAcrossTheCPU(t, m, io, dev, true, func(st *Instance) { st.crossStale = true })
					crossOutside(t, got, band, "the CPU device's copies left as they were made")
				})
				for _, readersUp := range []bool{true, false} {
					t.Run(fmt.Sprintf("%s/step-readers-on-the-card-%t", dev, readersUp), func(t *testing.T) {
						historyStepAcrossTheCPU(t, m, io, dev, readersUp)
					})
					t.Run(fmt.Sprintf("%s/shared-readers-on-the-card-%t", dev, readersUp), func(t *testing.T) {
						historySharedAcrossTheCPU(t, m, io, dev, readersUp, 0)
					})
					t.Run(fmt.Sprintf("%s/shared-apart-readers-on-the-card-%t", dev, readersUp), func(t *testing.T) {
						historySharedAcrossTheCPU(t, m, io, dev, readersUp, 32)
					})
				}
			}
		})
	}
}

// openReadsTwo opens readsTwoFixture: its last block reads blocks 0's and
// 1's histories past its own.
func openReadsTwo(t *testing.T, _ string) (*model.Model, *c6Golden, []Option) {
	t.Helper()
	m, err := model.Open(jlmOf(t, testmodels.Path("synth-seedoss.gguf")+readsTwoSuffix))
	if err != nil {
		t.Fatal(err)
	}
	if li := m.Cfg.NLayer - 1; len(m.Composing().BlockGraph(li).ReadsMore()) < 2 {
		m.Close()
		t.Fatalf("block %d reads no two histories past its own: the restatement selected nothing", li)
	}
	return m, nil, []Option{WithKVF16(false)}
}

// crossOutside fails t unless a step of got stands outside the band.
func crossOutside(t *testing.T, got, band []float64, what string) {
	t.Helper()
	out := 0
	for i := range got {
		if got[i] > max(3*band[i], 1e-6) {
			out++
		}
	}
	t.Logf("NMSE from the CPU's with %s: %.2e; one card: %.2e", what, got, band)
	if out == 0 {
		t.Fatalf("%s stood inside the band: the gate cannot see a copy that is not appended", what)
	}
}

// crossPrompt is the gate's prompt of n tokens.
func crossPrompt(m *model.Model, n, salt int) []int32 {
	p := make([]int32, n)
	for i := range p {
		p[i] = int32((i*7919 + 13 + salt*104729) % m.Cfg.NVocab)
	}
	return p
}

// crossWant is the CPU's logits after prompt and after each of ntok tokens
// it picks, and the tokens.
func crossWant(t *testing.T, m *model.Model, io []Option, prompt []int32, ntok, maxSeq int) ([][]float32, []int32) {
	t.Helper()
	cpu := testInstance(t, m, maxSeq, io...)
	defer cpu.Close()
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
	return want, ids
}

// crossBand is each step's NMSE from want with every block on card dev.
func crossBand(t *testing.T, m *model.Model, io []Option, dev string, prompt []int32, ids []int32, want [][]float32, maxSeq int) []float64 {
	t.Helper()
	g, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || g == nil {
		noDevice(t, dev, err)
	}
	defer g.Close()
	one := testInstance(t, m, maxSeq, io...)
	defer one.Close()
	if err := one.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if one.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("%s took %d of %d blocks: %s", dev, one.GPULayers(), m.Cfg.NLayer, g.Err())
	}
	return crossRunSteps(t, one, prompt, ids, want, false, nil)
}

// crossRunSteps runs prompt and ids on st, at(i) before token i, and is each
// step's NMSE from want; with greedy each step's pick must be the CPU's.
func crossRunSteps(t *testing.T, st *Instance, prompt []int32, ids []int32, want [][]float32, greedy bool, at func(i int)) []float64 {
	t.Helper()
	lg, err := st.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	got := []float64{0}
	got[0], _ = nmseOf(lg, want[0])
	for i, id := range ids {
		if at != nil {
			at(i)
		}
		if lg, err = st.Forward(id); err != nil {
			t.Fatalf("token %d: %v", i, err)
		}
		e, _ := nmseOf(lg, want[i+1])
		if math.IsNaN(e) {
			t.Fatalf("token %d: a non-finite logit", i)
		}
		if greedy && Greedy(lg) != Greedy(want[i+1]) {
			t.Errorf("token %d: greedy %d, the CPU's %d (NMSE %.3e)", i, Greedy(lg), Greedy(want[i+1]), e)
		}
		got = append(got, e)
	}
	return got
}

// crossPlace is a strict placement of blocks [0, cut) on lower and the rest
// on upper (nn.CPU for the CPU device), the projection the engine's.
func crossPlace(nl, cut int, lower, upper string) Option {
	up := map[int]Place{}
	for li := cut; li < nl; li++ {
		up[li] = Place{On: upper}
	}
	return WithPlacement(Placement{Blocks: up, Rest: &Place{On: lower}, Strict: true})
}

// crossWhere is st's placement, block by block.
func crossWhere(st *Instance, nl int) string {
	var w []string
	for li := range nl {
		at, ok := st.DeviceOf(li)
		if !ok {
			at = nn.CPU
		}
		w = append(w, fmt.Sprintf("%d:%s", li, at))
	}
	return strings.Join(w, " ")
}

// crossCopies fails t unless the copies g's cards and st's CPU device hold
// are the reads across devices st's placement makes, on the cards for each
// of alike instances placed as st is, and the CPU device holds no page of a
// block on a card that no block of it reads.
func crossCopies(t *testing.T, g *gpu.GPU, st *Instance, m *model.Model, alike int, what string) {
	t.Helper()
	if held, need := historyCopiesHeld(g), alike*crossCardReads(st, m); held != need {
		t.Fatalf("%s: the cards hold %d history copies where %d are read (%s)", what, held, need, crossWhere(st, m.Cfg.NLayer))
	}
	held, need := 0, 0
	for li := st.lo; li < st.hi; li++ {
		if !st.devAt(li) {
			continue
		}
		for h := 0; h <= st.kv.nHist(li); h++ {
			pg := st.histPages(li, h)
			if pg == nil {
				continue
			}
			if st.lanesRead(li, h) {
				need++
			}
			if pg.copy {
				held++
			}
			if pg.copy && st.lanesRead(li, h) {
				continue
			}
			for n := range pg.mem.Len() {
				if pg.mem.K(n) != nil {
					t.Fatalf("%s: the CPU device holds page %d of block %d's history %d, on a card, which no block on the CPU device reads (%s)", what, n, li, h, crossWhere(st, m.Cfg.NLayer))
				}
			}
		}
	}
	if held != need {
		t.Fatalf("%s: the CPU device holds %d copies of a card's history where %d are read (%s)", what, held, need, crossWhere(st, m.Cfg.NLayer))
	}
}

// historyAcrossTheCPU runs m split across the CPU device and card dev
// against the CPU: the readers on the card (or, readersDown, on the CPU
// device), the source on the other side. arm, when set, arms a violation
// once the placement is made, on the card tier opens; the moves are left
// out and the bounds are the caller's.
func historyAcrossTheCPU(t *testing.T, m *model.Model, io []Option, dev string, readersDown bool, arm func(*Instance), tier ...gpu.Option) ([]float64, []float64) {
	nl := m.Cfg.NLayer
	src, cut := historyCut(t, m)
	const n, ntok = 600, 14
	prompt := crossPrompt(m, n, 0)
	maxSeq := n + ntok + 4
	want, ids := crossWant(t, m, io, prompt, ntok, maxSeq)
	band := crossBand(t, m, io, dev, prompt, ids, want, maxSeq)

	g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff)}, tier...)...)
	if err != nil || g == nil {
		noDevice(t, dev, err)
	}
	defer g.Close()
	lower, upper := nn.CPU, dev
	if readersDown {
		lower, upper = dev, nn.CPU
	}
	st := testInstance(t, m, maxSeq, append(append([]Option(nil), io...), crossPlace(nl, cut, lower, upper))...)
	defer st.Close()
	if err := st.SetDevice(g); err != nil {
		t.Fatalf("blocks [0,%d) on %s and [%d,%d) on %s: %v (%s)", cut, lower, cut, nl, upper, err, g.Err())
	}
	place := func(lo, hi int, to string) {
		t.Helper()
		want := to
		if want == nn.CPU {
			want = ""
		}
		for li := lo; li < hi; li++ {
			if at, _ := st.DeviceOf(li); at != want {
				t.Fatalf("block %d is on %q where it was put on %s: %s", li, at, to, crossWhere(st, nl))
			}
		}
	}
	place(0, cut, lower)
	place(cut, nl, upper)
	crossCopies(t, g, st, m, 1, "placed")
	if arm != nil {
		arm(st)
	}
	type move struct {
		at, lo, hi int
		to         string
	}
	moves := []move{
		{2, src, src + 1, dev},
		{4, src, src + 1, nn.CPU},
		{6, cut, nl, nn.CPU},
		{8, 0, cut, dev},
		{10, cut, nl, dev},
		{12, 0, cut, nn.CPU},
	}
	if arm != nil || readersDown {
		moves = nil
	}
	h0, r0 := g.Stats().HostCopyPages, g.Stats().HostReadPages
	got := crossRunSteps(t, st, prompt, ids, want, arm == nil, func(i int) {
		for _, mv := range moves {
			if mv.at != i {
				continue
			}
			before := map[int]string{}
			for li := range nl {
				before[li], _ = st.DeviceOf(li)
			}
			bm, err := st.MoveBlocks(mv.lo, mv.hi, mv.to)
			if err != nil {
				t.Fatalf("token %d: blocks [%d,%d) to %s: %v", i, mv.lo, mv.hi, mv.to, err)
			}
			place(mv.lo, mv.hi, mv.to)
			for li := range nl {
				if li >= mv.lo && li < mv.hi {
					continue
				}
				if at, _ := st.DeviceOf(li); at != before[li] {
					t.Fatalf("token %d: block %d went from %q to %q with a move of [%d,%d) to %s: %s", i, li, before[li], at, mv.lo, mv.hi, mv.to, crossWhere(st, nl))
				}
			}
			crossCopies(t, g, st, m, 1, fmt.Sprintf("token %d, after [%d,%d) went to %s", i, mv.lo, mv.hi, mv.to))
			t.Logf("token %d: [%d,%d) to %s: moved %v, %d KV bytes; %s", i, mv.lo, mv.hi, mv.to, bm.Moved, bm.KVBytes, crossWhere(st, nl))
		}
	})
	s := g.Stats()
	if arm == nil {
		switch {
		case readersDown && s.HostReadPages == r0:
			t.Fatal("no page came from the card to the CPU device's copy: the gate read no history across")
		case !readersDown && s.HostCopyPages == h0:
			t.Fatal("no page reached the card's copy from the CPU device: the gate read no history across")
		case !readersDown && s.HostReadPages == r0:
			t.Fatal("the moves put no copy on the CPU device: no page came from the card to it")
		}
		for i := range got {
			if bound := max(3*band[i], 1e-6); got[i] > bound {
				t.Errorf("step %d: NMSE %.3e from the CPU's, above %.3e (one card's %.3e)", i, got[i], bound, band[i])
			}
		}
		formedAtPlacement(t, g)
		t.Logf("NMSE from the CPU's, split: %.2e; one card: %.2e; %d pages to the card's copies, %d to the CPU's", got, band, s.HostCopyPages-h0, s.HostReadPages-r0)
	}
	return got, band
}

// historyStepAcrossTheCPU is two instances split across the CPU device and
// card dev as the gate splits them (the readers on the card where
// readersUp), their rows stepped together and a call each in turn: each
// row's copy of a history is its own session's, appended from its own
// session's rows. Every token's logits are held to the CPU's for that
// prompt alone, teacher-forced.
func historyStepAcrossTheCPU(t *testing.T, m *model.Model, io []Option, dev string, readersUp bool) {
	nl := m.Cfg.NLayer
	_, cut := historyCut(t, m)
	const gen, maxSeq = 8, 96
	prompts := [][]int32{crossPrompt(m, 40, 1), crossPrompt(m, 57, 2)}
	want := make([][][]float32, len(prompts))
	ids := make([][]int32, len(prompts))
	for i, p := range prompts {
		want[i], ids[i] = crossWant(t, m, io, p, gen, maxSeq)
	}
	g, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || g == nil {
		noDevice(t, dev, err)
	}
	defer g.Close()
	lower, upper := dev, nn.CPU
	if readersUp {
		lower, upper = nn.CPU, dev
	}
	insts := make([]*Instance, len(prompts))
	last := make([][]float32, len(prompts))
	for i, p := range prompts {
		st := testInstance(t, m, maxSeq, append(append([]Option(nil), io...), crossPlace(nl, cut, lower, upper))...)
		defer st.Close()
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		lg, err := st.Prefill(p)
		if err != nil {
			t.Fatal(err)
		}
		insts[i], last[i] = st, lg
	}
	for i, st := range insts {
		crossCopies(t, g, st, m, len(insts), fmt.Sprintf("instance %d", i))
	}
	s0 := g.Stats()
	runs := make([]Run, len(prompts))
	stepped := 0
	var worst float64
	for k := range gen {
		for i := range insts {
			e, _ := nmseOf(last[i], want[i][k])
			worst = max(worst, e)
			if !(e <= 1e-4) {
				t.Errorf("instance %d, token %d: NMSE %.3e from the CPU's alone", i, k, e)
			}
		}
		if k == gen-1 {
			break
		}
		if k%2 == 1 {
			for i, st := range insts {
				if last[i], err = st.Forward(ids[i][k]); err != nil {
					t.Fatal(err)
				}
			}
			continue
		}
		for i, st := range insts {
			runs[i] = Run{Instance: st, Tokens: ids[i][k : k+1], Logits: true}
		}
		lg, err := StepRuns(runs)
		if err != nil {
			t.Fatal(err)
		}
		copy(last, lg)
		stepped += len(prompts)
	}
	s := g.Stats()
	if readersUp && s.HostCopyPages == s0.HostCopyPages || !readersUp && s.HostReadPages == s0.HostReadPages {
		t.Fatal("no page crossed to a copy: the steps read no history across the CPU and the card")
	}
	for i, st := range insts {
		if d := st.DeviceDemotions(); d > 0 {
			t.Fatalf("instance %d left the card %d time(s)", i, d)
		}
	}
	formedAtPlacement(t, g)
	t.Logf("%d instances over %d tokens: worst NMSE %.3e from the CPU's alone; %d rows stepped together; %d pages to the card's copies, %d to the CPU's",
		len(prompts), gen, worst, stepped, s.HostCopyPages-s0.HostCopyPages, s.HostReadPages-s0.HostReadPages)
}

// historySharedAcrossTheCPU: two sessions of one model, each its own
// instance placed split across the CPU device and card dev (the readers on
// the card where readersUp, else on the CPU device), share a prompt's
// prefix through the prefix cache. Each is held to the CPU's logits for its
// own prompt and the tokens after it, and the prefix is held once on each
// device: no page a session holds beside the store's on the host
// (kvDuplicates), the second session's CPU copy of a card's block reading
// the first's pages through the store (KVSharedBytes), the second restores
// the prefix (KVRestored), and the card maps the pages the first's named
// (KVPagesMapped): with the readers on the card, its copies of the CPU's
// blocks; with them on the CPU device, the blocks there. A windowed source
// on the card, which pages it apart from the CPU device (64 positions
// against a window of 4 keys' 32), restores through the CPU device's copy
// of it, its pages named at the CPU's geometry.
//
// With hostPage set the CPU device pages every block at hostPage positions
// and the card at 64, so every block on the card pages apart from the CPU
// device, those that see their whole history among them, and the shared
// prefix of five host pages ends inside a card's page: the card restores its
// blocks from its own pages, the one the prefix ends in by the name of the
// positions it shares.
func historySharedAcrossTheCPU(t *testing.T, m *model.Model, io []Option, dev string, readersUp bool, hostPage int) {
	nl := m.Cfg.NLayer
	_, cut := historyCut(t, m)
	const page, ntok, seq = 64, 6, 1024
	hp, shared := page, 4*page
	if hostPage > 0 {
		hp, shared = hostPage, 5*hostPage
	}
	g, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff),
		gpu.WithConfig(func(c *gpu.Config) { c.KVPage = page }))
	if err != nil || g == nil {
		noDevice(t, dev, err)
	}
	defer g.Close()
	lower, upper := dev, nn.CPU
	if readersUp {
		lower, upper = nn.CPU, dev
	}
	prompts := make([][]int32, 2)
	for i := range prompts {
		p := crossPrompt(m, shared, 0)
		prompts[i] = append(p, crossPrompt(m, shareSuffix, i+1)...)
	}
	// One CPU device for both sessions: a shared page is a slot of its arena.
	cpu := sharedCPU(t)
	store := NewSharedStore()
	sess := make([]*Instance, len(prompts))
	firsts := make([][]float32, len(prompts))
	m0 := g.Stats().KVPagesMapped
	for i, p := range prompts {
		st := testInstance(t, m, seq, append(append([]Option(nil), io...), crossPlace(nl, cut, lower, upper), withKVPage(hp), cpu)...)
		defer st.Close()
		if err := st.SetDevice(g); err != nil {
			t.Fatalf("session %d: blocks [0,%d) on %s and [%d,%d) on %s: %v (%s)", i, cut, lower, cut, nl, upper, err, g.Err())
		}
		mustKey(t, st, "cross-share")
		if err := st.ShareKV(store); err != nil {
			t.Fatal(err)
		}
		lg, err := st.PrefillCached(p)
		if err != nil {
			t.Fatalf("session %d: %v", i, err)
		}
		sess[i], firsts[i] = st, lg
	}
	if sess[1].KVRestored() < shared {
		t.Fatalf("the second session restored %d positions of the shared %d", sess[1].KVRestored(), shared)
	}
	for i, st := range sess {
		want, ids := crossWant(t, m, io, prompts[i], ntok, seq)
		e, _ := nmseOf(firsts[i], want[0])
		if !(e <= 1e-4) {
			t.Errorf("session %d's prompt: NMSE %.3e from the CPU's", i, e)
		}
		for k, id := range ids {
			lg, err := st.Forward(id)
			if err != nil {
				t.Fatalf("session %d, token %d: %v", i, k, err)
			}
			if e, _ := nmseOf(lg, want[k+1]); !(e <= 1e-4) {
				t.Errorf("session %d, token %d: NMSE %.3e from the CPU's", i, k, e)
			}
		}
		if d := kvDuplicates(t, st); d != 0 {
			kc := st.kv
			for li := kc.lo; li < len(kc.layers); li++ {
				pg := &kc.layers[li]
				for n := 0; pg.p > 0 && n < pg.mem.Len(); n++ {
					if pg.mem.K(n) != nil {
						t.Logf("block %d page %d: shared %v, on a card %v, copy %v, copied %v, genuine %d, lastSealed %d, in the store %v", li, n, pg.isShared(n), st.devAt(li), pg.copy, pg.copied, pg.genuine, pg.lastSealed, kvStoreHas(kc, kc.pageKey(li, n), li, n, pg.pp))
					}
				}
			}
			t.Errorf("session %d holds %d page(s) of its history twice on the host", i, d)
		}
		crossCopies(t, g, st, m, len(sess), fmt.Sprintf("session %d", i))
	}
	if g.Stats().KVPagesMapped == m0 {
		t.Error("the second session mapped none of the first's pages on the card: the prefix is held twice there")
	}
	if !readersUp && sess[1].KVSharedBytes() == 0 {
		t.Error("the second session's CPU copies read none of the first's pages through the store: the prefix is held twice on the host")
	}
	t.Logf("restored %d; pages mapped on the card %d; the second session reads %d bytes through the store", sess[1].KVRestored(), g.Stats().KVPagesMapped-m0, sess[1].KVSharedBytes())
}
