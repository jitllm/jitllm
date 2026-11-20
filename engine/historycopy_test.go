package engine

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// TestAHistoryReadAcrossCards: a block reading another block's history
// (Gemma 3n's and Gemma 4's shared-KV layers) runs on another card than the
// block it reads. A strict placement puts the blocks up to the first source
// read from above it on one card and the rest on the other, so a reader runs
// a card away from its source: the reader's card holds a copy of the
// source's history, appended as the source appends. The prompt and every
// token after it are held to the CPU's logits, inside three times the band
// one card holding every block stands at, with the CPU's greedy pick.
//
// Mid-sequence the source moves beside its readers (the copy becomes its own
// pages, and no card holds a copy), back to the first card (the readers'
// card keeps its pages as a copy again), to the CPU (its readers stay, and
// their card's copy is written from the CPU's rows: histcross.go) and back,
// and then the readers move to the second card alone, their copy made there
// from the source's whole history. Each move is held to the logits of the
// token after it.
//
// The projection is named on the second card, beside the readers, and is
// there after every move: on synth-gemma4-e that card holds no block of the
// model's home geometry set.
// The step arms split two instances so on one tier and step their rows
// together (historyStepAcrossCards), the projection on either card, and
// again split at block 2, where no block reads a history across the cards.
//
// Each pair of cards of one backend runs, and a CUDA card with a Vulkan card
// of another card, whose copies go through the host.
func TestAHistoryReadAcrossCards(t *testing.T) {
	for _, f := range []struct {
		name string
		open func(*testing.T, string) (*model.Model, *c6Golden, []Option)
	}{{"synth-gemma4-e", openGemma4}, {"synth-gemma3n", openGemma3n}} {
		t.Run(f.name, func(t *testing.T) {
			m, _, io := f.open(t, f.name)
			defer m.Close()
			pairs := historyCopyPairs(t)
			if len(pairs) == 0 {
				noDevice(t, "two cards", nil)
			}
			for _, p := range pairs {
				t.Run(p[0]+","+p[1], func(t *testing.T) { historyReadAcrossCards(t, m, io, p[0], p[1], nil) })
				for _, head := range p {
					t.Run(p[0]+","+p[1]+"/step-head-"+head, func(t *testing.T) { historyStepAcrossCards(t, m, io, p[0], p[1], head, -1) })
				}
				t.Run(p[0]+","+p[1]+"/step-unsplit", func(t *testing.T) { historyStepAcrossCards(t, m, io, p[0], p[1], p[0], 2) })
			}
		})
	}
}

// historyStepAcrossCards is two instances split as the gate splits them on
// one tier (or below block at, where at is not negative), the projection on
// card head, their rows stepped together (StepRuns, the tier's rows path) and
// a call each in turn: each row's copy of a history is its own session's,
// appended from its own session's rows. Every token's logits are held to the
// CPU's for that prompt alone, teacher-forced, and every card counts the
// step's rows.
func historyStepAcrossCards(t *testing.T, m *model.Model, io []Option, a, b, head string, at int) {
	nl := m.Cfg.NLayer
	_, cut := historyCut(t, m)
	split := at < 0
	if !split {
		cut = at
	}
	prompts := [][]int32{make([]int32, 40), make([]int32, 57)}
	for i, p := range prompts {
		for j := range p {
			p[j] = int32((j*(7919+i*104) + 13 + i) % m.Cfg.NVocab)
		}
	}
	const gen, maxSeq = 8, 96
	want := make([][][]float32, len(prompts))
	ids := make([][]int32, len(prompts))
	for i, p := range prompts {
		c := testInstance(t, m, maxSeq, io...)
		lg, err := c.Prefill(p)
		if err != nil {
			t.Fatal(err)
		}
		for range gen {
			want[i] = append(want[i], append([]float32(nil), lg...))
			ids[i] = append(ids[i], Greedy(lg))
			if lg, err = c.Forward(ids[i][len(ids[i])-1]); err != nil {
				t.Fatal(err)
			}
		}
		c.Close()
	}
	g, err := openGPU(t, gpu.WithDevices(a+","+b), gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || g == nil {
		noDevice(t, a+","+b, err)
	}
	defer g.Close()
	up := map[int]Place{}
	for li := cut; li < nl; li++ {
		up[li] = Place{On: b}
	}
	opts := append(append([]Option(nil), io...),
		WithPlacement(Placement{Blocks: up, Rest: &Place{On: a}, Head: &Place{On: head}, Strict: true}))
	insts := make([]*Instance, len(prompts))
	last := make([][]float32, len(prompts))
	for i, p := range prompts {
		st := testInstance(t, m, maxSeq, opts...)
		defer st.Close()
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		lg, err := st.Prefill(p)
		if err != nil {
			t.Fatal(err)
		}
		insts[i], last[i] = st, lg
		headOn(t, st, head)
	}
	if held, need := historyCopiesHeld(g), len(prompts)*crossCardReads(insts[0], m); split && need == 0 || held != need {
		t.Fatalf("%d instances hold %d history copies where %d are read", len(prompts), held, need)
	}
	d0, pages := g.DevStats(), historyAppends(g)
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
	for i, d := range g.DevStats() {
		if got := d.SessionRows - d0[i].SessionRows; got != stepped {
			t.Fatalf("device %d stepped %d row(s) across sessions, want %d: the step did not run on the cards (refusals %v, %v)", i, got, stepped, insts[0].StepRefusal(), insts[1].StepRefusal())
		}
	}
	if split && historyAppends(g) == pages {
		t.Fatal("no row reached a history copy: the step read no history across the cards")
	}
	for i, st := range insts {
		if d := st.DeviceDemotions(); d > 0 {
			t.Fatalf("instance %d left the cards %d time(s)", i, d)
		}
		headOn(t, st, head)
	}
	formedAtPlacement(t, g)
	t.Logf("%d instances over %d tokens: worst NMSE %.3e from the CPU's alone; %d rows stepped together, %d history pages copied",
		len(prompts), gen, worst, stepped, historyAppends(g)-pages)
}

// TestAHistoryCopyRepages: a block reading another block's history on
// another card reads a copy of it there although the two cards' pools page
// apart -- the readers' card was given its pool at the longest page by an
// instance of a long context placed there first, the source's card at a
// shorter one by this instance's -- the copy re-paged as rows on the way
// (copyRepaged). The prompt and every token after it are held to the CPU's
// logits for the prompt alone, and the copies held are the reads across the
// cards. synth-gemma3n restated to keep each block's history past its rows
// (pastTheRows) has its copies re-paged as entries.
func TestAHistoryCopyRepages(t *testing.T) {
	for _, f := range []struct {
		name string
		open func(*testing.T, string) (*model.Model, *c6Golden, []Option)
	}{{"synth-gemma4-e", openGemma4}, {"synth-gemma3n", openGemma3n}, {"synth-gemma3n-past-the-rows", openPastTheRows(openGemma3n, "synth-gemma3n", 1)}} {
		t.Run(f.name, func(t *testing.T) {
			m, _, io := f.open(t, f.name)
			defer m.Close()
			pairs := historyCopyPairs(t)
			if len(pairs) == 0 {
				noDevice(t, "two cards", nil)
			}
			for _, p := range pairs {
				t.Run(p[0]+","+p[1], func(t *testing.T) { historyCopyRepages(t, m, io, p[0], p[1]) })
			}
		})
	}
}

// openPastTheRows is open of fixture name with every block's rows' history
// kept as history h (pastTheRows): a block reading another's reads that
// block's history h, one past its rows.
func openPastTheRows(open func(*testing.T, string) (*model.Model, *c6Golden, []Option), name string, h uint8) func(*testing.T, string) (*model.Model, *c6Golden, []Option) {
	return func(t *testing.T, _ string) (*model.Model, *c6Golden, []Option) {
		m, gold, io := open(t, name)
		f := m.FrontGraphs()
		for li := range f.Blocks {
			f.Blocks[li] = pastTheRows(f.Blocks[li], h)
		}
		if src, _ := historyCut(t, m); len(m.Composing().BlockGraph(src).KeptHistories()) == 0 {
			t.Fatalf("block %d keeps no history past its rows: the restatement selected nothing", src)
		}
		return m, gold, io
	}
}

// historyCopyRepages is TestAHistoryCopyRepages over cards a, the source's,
// and b, the readers'.
func historyCopyRepages(t *testing.T, m *model.Model, io []Option, a, b string) {
	nl := m.Cfg.NLayer
	src, cut := historyCut(t, m)
	const n, ntok = 100, 14
	maxSeq := n + ntok + 4
	prompt := crossPrompt(m, n, 0)
	want, ids := crossWant(t, m, io, prompt, ntok, maxSeq)
	g, err := openGPU(t, gpu.WithDevices(a+","+b), gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || g == nil {
		noDevice(t, a+","+b, err)
	}
	defer g.Close()
	distinctCards(t, g)
	long := testInstance(t, m, 4*maxSeq, append(append([]Option(nil), io...),
		WithPlacement(Placement{Rest: &Place{On: b}, Strict: true}))...)
	defer long.Close()
	if err := long.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if _, err := long.Prefill(prompt[:8]); err != nil {
		t.Fatal(err)
	}
	st := testInstance(t, m, maxSeq, append(append([]Option(nil), io...), crossPlace(nl, cut, a, b))...)
	defer st.Close()
	if err := st.SetDevice(g); err != nil {
		t.Fatalf("blocks [0,%d) on %s and [%d,%d) on %s: %v (%s)", cut, a, cut, nl, b, err, g.Err())
	}
	pa, pb := st.ld.(nn.KVShareDevice).SharePage(src), long.ld.(nn.KVShareDevice).SharePage(src)
	if pa == pb || pa == 0 || pb == 0 {
		t.Fatalf("block %d's history pages %d positions on %s and %d on %s: the gate selected no pools paging apart", src, pa, a, pb, b)
	}
	if held, need := historyCopiesHeld(g), crossCardReads(st, m); need == 0 || held != need {
		t.Fatalf("the cards hold %d history copies where %d are read (%s)", held, need, crossWhere(st, nl))
	}
	pages := historyAppends(g)
	got := crossRunSteps(t, st, prompt, ids, want, true, nil)
	for i, e := range got {
		if !(e <= 1e-4) {
			t.Errorf("step %d: NMSE %.3e from the CPU's", i, e)
		}
	}
	if historyAppends(g) == pages {
		t.Fatal("no page reached a history copy: the gate read no history across the cards")
	}
	t.Logf("pools paging %d and %d positions: NMSE from the CPU's %.2e; %d history pages copied", pa, pb, got, historyAppends(g)-pages)
}

// historyCopyPairs is the pairs of selected devices the gate splits a model
// over: the first two of each backend, and the first CUDA device with a
// Vulkan device of another card.
func historyCopyPairs(t *testing.T) [][2]string {
	t.Helper()
	by := map[string][]string{}
	var apis []string
	for _, s := range testdevices.Specs("cuda:0", "cuda:1", "vulkan:0", "vulkan:1") {
		api := strings.SplitN(s, ":", 2)[0]
		if _, ok := by[api]; !ok {
			apis = append(apis, api)
		}
		by[api] = append(by[api], s)
	}
	var out [][2]string
	for _, api := range apis {
		if d := by[api]; len(d) >= 2 {
			out = append(out, [2]string{d[0], d[1]})
		}
	}
	if c, v := by["cuda"], by["vulkan"]; len(c) > 0 {
		for _, vs := range v {
			g, err := openGPU(t, gpu.WithDevices(c[0]+","+vs), gpu.WithDeviceTune(gpu.TuneOff))
			if err != nil || g == nil {
				continue
			}
			clash := cardClash(g.Hardware())
			g.Close()
			if clash == "" {
				out = append(out, [2]string{c[0], vs})
				break
			}
		}
	}
	return out
}

// historyCut is the first block m's graph reads another block's history of,
// and the cut just above it: blocks below the cut on one card, the rest on
// another, splits that source from a reader of it.
func historyCut(t *testing.T, m *model.Model) (src, cut int) {
	t.Helper()
	for li := range m.Cfg.NLayer {
		g := m.Composing().BlockGraph(li)
		if n := g.ReadsHistoryOf(); n >= 0 && n < li {
			return n, n + 1
		}
		for _, r := range g.ReadsMore() {
			if r.Block < li {
				return r.Block, r.Block + 1
			}
		}
	}
	t.Fatal("no block reads another's history: the gate selected nothing")
	return 0, 0
}

// historyStep is one step of the split run: the token after a move, and the
// move.
type historyStep struct {
	at     int
	lo, hi int
	to     string
}

// crossCardReads is the history copies st's placement needs on its cards:
// each block whose history a block on another card, or on the CPU device,
// reads (histcross.go), once a card it is read on.
func crossCardReads(st *Instance, m *model.Model) int {
	type pair struct {
		src int
		on  string
	}
	need := map[pair]bool{}
	for li := range m.Cfg.NLayer {
		on, ok := st.DeviceOf(li)
		if !ok {
			continue
		}
		g := m.Composing().BlockGraph(li)
		var srcs []int
		n := g.ReadsHistoryOf()
		if n >= 0 && n != li {
			srcs = append(srcs, n)
		}
		for _, r := range g.ReadsMore() {
			srcs = append(srcs, r.Block)
		}
		for _, s := range srcs {
			if at, ok := st.DeviceOf(s); !ok || at != on {
				need[pair{s, on}] = true
			}
		}
	}
	return len(need)
}

// historyReadAcrossCards runs m split over cards a and b against the CPU.
// arm, when set, arms a violation once the placement is made, on the split
// tier tier opens; the moves are left out and the bounds are the caller's.
func historyReadAcrossCards(t *testing.T, m *model.Model, io []Option, a, b string, arm func(), tier ...gpu.Option) ([]float64, []float64) {
	nl := m.Cfg.NLayer
	src, cut := historyCut(t, m)
	const n, ntok = 600, 14
	prompt := make([]int32, n)
	for i := range prompt {
		prompt[i] = int32((i*7919 + 13) % m.Cfg.NVocab)
	}
	maxSeq := n + ntok + 4

	// The CPU's logits after the prompt and after each token it picks.
	cpu := testInstance(t, m, maxSeq, io...)
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

	run := func(st *Instance, at func(i int)) []float64 {
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
			if Greedy(lg) != Greedy(want[i+1]) && arm == nil {
				t.Errorf("token %d: greedy %d, the CPU's %d (NMSE %.3e)", i, Greedy(lg), Greedy(want[i+1]), e)
			}
			got = append(got, e)
		}
		return got
	}

	// The band: every block on card a.
	g1, err := openGPU(t, gpu.WithDevices(a), gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || g1 == nil {
		noDevice(t, a, err)
	}
	one := testInstance(t, m, maxSeq, io...)
	if err := one.SetDevice(g1); err != nil {
		t.Fatal(err)
	}
	if one.GPULayers() != nl {
		t.Fatalf("%s took %d of %d blocks: %s", a, one.GPULayers(), nl, g1.Err())
	}
	band := run(one, nil)
	one.Close()
	g1.Close()

	g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(a + "," + b), gpu.WithDeviceTune(gpu.TuneOff)}, tier...)...)
	if err != nil || g == nil {
		noDevice(t, a+","+b, err)
	}
	defer g.Close()
	distinctCards(t, g)
	up := map[int]Place{}
	for li := cut; li < nl; li++ {
		up[li] = Place{On: b}
	}
	opts := append(append([]Option(nil), io...),
		WithPlacement(Placement{Blocks: up, Rest: &Place{On: a}, Head: &Place{On: b}, Strict: true}))
	st := testInstance(t, m, maxSeq, opts...)
	defer st.Close()
	if err := st.SetDevice(g); err != nil {
		var why []string
		for _, ds := range g.DevStats() {
			why = append(why, ds.LastErr)
		}
		t.Fatalf("blocks [0,%d) on %s and [%d,%d) on %s: %v (the cards: %q)", cut, a, cut, nl, b, err, why)
	}
	where := func() string {
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
	headOn(t, st, b)
	if held, need := historyCopiesHeld(g), crossCardReads(st, m); need == 0 || held != need {
		t.Fatalf("block %d's readers run on %s and it on %s: the cards hold %d history copies where %d are read (%s)", src, b, a, held, need, where())
	}
	if arm != nil {
		arm()
	}
	steps := []historyStep{
		{at: 2, lo: src, hi: src + 1, to: b},
		{at: 4, lo: src, hi: src + 1, to: a},
		{at: 6, lo: src, hi: src + 1, to: nn.CPU},
		{at: 8, lo: src, hi: src + 1, to: a},
		{at: 10, lo: cut, hi: nl, to: b},
	}
	if arm != nil {
		steps = nil
	}
	appended := historyAppends(g)
	got := run(st, func(i int) {
		for _, s := range steps {
			if s.at != i {
				continue
			}
			mv, err := st.MoveBlocks(s.lo, s.hi, s.to)
			if err != nil {
				t.Fatalf("token %d: blocks [%d,%d) to %s: %v", i, s.lo, s.hi, s.to, err)
			}
			want := s.to
			if want == nn.CPU {
				want = ""
			}
			for li := s.lo; li < s.hi; li++ {
				if at, _ := st.DeviceOf(li); at != want {
					t.Fatalf("token %d: block %d is on %q after a move of [%d,%d) to %s: %s", i, li, at, s.lo, s.hi, s.to, where())
				}
			}
			if held, need := historyCopiesHeld(g), crossCardReads(st, m); held != need {
				t.Fatalf("token %d: after blocks [%d,%d) went to %s the cards hold %d history copies where %d are read (%s)", i, s.lo, s.hi, s.to, held, need, where())
			}
			headOn(t, st, b)
			t.Logf("token %d: [%d,%d) to %s: moved %v, %d KV bytes by %q; %s", i, s.lo, s.hi, s.to, mv.Moved, mv.KVBytes, mv.Path, where())
		}
	})
	if arm == nil {
		headOn(t, st, b)
	}
	if arm == nil && historyAppends(g) == appended {
		t.Fatal("no source's rows reached a copy on another card: the gate ran no copy")
	}
	if arm == nil {
		for i := range got {
			if bound := max(3*band[i], 1e-6); got[i] > bound {
				t.Errorf("step %d: NMSE %.3e from the CPU's, above %.3e (one card's %.3e)", i, got[i], bound, band[i])
			}
		}
		formedAtPlacement(t, g)
		t.Logf("NMSE from the CPU's, split: %.2e; one card: %.2e; %d history pages copied", got, band, historyAppends(g))
	}
	return got, band
}

// formedAtPlacement fails t where a card met a part at a step that its
// placement did not form.
func formedAtPlacement(t *testing.T, g *gpu.GPU) {
	t.Helper()
	if s := g.Stats(); s.UnformedAtStep != 0 {
		t.Errorf("the cards met %d part(s) at a step that their placement did not form (%v): %s", s.UnformedAtStep, s.UnformedAtStepParts, g.Err())
	}
}

// headOn fails t unless st's projection runs on card on.
func headOn(t *testing.T, st *Instance, on string) {
	t.Helper()
	n, ok := st.ld.(nn.NamedDevice).HeadDeviceName()
	if !st.HeadOnDevice() || !ok || n != on {
		t.Fatalf("the projection is on %q (on a card: %v), named on %s", n, st.HeadOnDevice(), on)
	}
}

// historyCopiesHeld is the history copies g's cards hold.
func historyCopiesHeld(g *gpu.GPU) int { return g.Stats().HistoryCopiesHeld }

// historyAppends is the pages g copied into history copies.
func historyAppends(g *gpu.GPU) int { return g.Stats().HistoryCopyPages }
