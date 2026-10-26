package model

import (
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestPlacementComposesOnce holds the composer's census on Llama-3.2-1B and
// the hybrid Qwen3.5-0.8B (composeOnceModels) with every part composed and a reserved prompt width of 512 rows
// (JITLLM_COMPOSE_ONCE_ROWS another): once a State's blocks are placed,
// prompts of 1, 5, 33, 100, 511, 512, 513 and 2000 tokens, each followed by
// decode steps, a second session's prompt and decode, and a ragged batch of
// four sequences whose prompts hold more rows than the reserved width compose
// nothing, compile no kernel and build no scratch -- Stats.PartsFormed,
// ScratchBuilds, DeclineBuilt, FormedAtStep and backend.Compiles stay where
// the placement left them, and the composer serves every chunk's every block.
// A prompt longer than the reserved width runs in chunks of it, and the
// scheduler admits no step wider than it (RowsWidth). Each block is composed
// once (Stats.BlockForms); a relocation composes its blocks on the
// destination, once each, and nothing after it composes. The two-card arm
// places the upper half of the blocks and the head on a second card, so the
// head is on another card than block 0, holds the same census and moves the
// lower half across.
//
// JITLLM_VIOLATE_COMPOSE_ONCE=1 forms a prompt at its chunk's grain width
// (Config.ComposedWidths), a form keyed by rows, which must fail.
func TestPlacementComposesOnce(t *testing.T) {
	violate := os.Getenv("JITLLM_VIOLATE_COMPOSE_ONCE") != ""
	// JITLLM_COMPOSE_ONCE_ROWS reserves another prompt width, for a card
	// whose budget does not hold a 512-row composed block beside the model.
	width := 512
	if v := os.Getenv("JITLLM_COMPOSE_ONCE_ROWS"); v != "" {
		w, err := strconv.Atoi(v)
		if err != nil || w < 32 {
			t.Fatalf("JITLLM_COMPOSE_ONCE_ROWS=%q: a width of at least 32 rows", v)
		}
		width = w
	}
	open := func(t *testing.T, devices string) *tier.GPU {
		opts := []tier.Option{tier.WithDeviceTune(tier.TuneOff), tier.WithConfig(func(c *tier.Config) {
			c.Composed, c.PromptRows, c.ComposedWidths = tier.ComposedStep|tier.ComposedPrompt, width, violate
		})}
		if devices != "" {
			opts = append(opts, tier.WithDevices(devices))
		}
		g, err := tier.OpenWith(opts...)
		if err != nil || g == nil {
			t.Fatalf("no GPU device: %v", err)
		}
		t.Cleanup(g.Close)
		return g
	}
	for _, name := range composeOnceModels {
		t.Run(name, func(t *testing.T) {
			p := testmodels.Path(name)
			if _, err := os.Stat(p); err != nil {
				if _, err := os.Stat(strings.TrimSuffix(p, ".gguf") + ".jlm"); err != nil {
					testmodels.Missing(t, "MODEL MISSING: %s", name)
				}
			}
			path := jlmOf(t, p)
			t.Run("one-card", func(t *testing.T) {
				m, err := Open(path, noTune)
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				composesOnce(t, m, open(t, testdevices.Take(t, 1, "cuda:0")[0]), width, "")
			})
			t.Run("two-cards", func(t *testing.T) {
				a, b := moveDevices(t)
				g := open(t, a+","+b)
				distinctCards(t, g)
				up := map[int]Place{}
				m0, err := Open(path)
				if err != nil {
					t.Fatal(err)
				}
				n := m0.Cfg.NLayer
				m0.Close()
				for li := n / 2; li < n; li++ {
					up[li] = Place{On: b}
				}
				m, err := Open(path, noTune, WithPlacement(Placement{Blocks: up, Rest: &Place{On: a}, Head: &Place{On: b}, Strict: true}))
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				composesOnce(t, m, g, width, b)
			})
		})
	}
}

// composeOnceModels are the census's models: Llama-3.2-1B, and Qwen3.5-0.8B,
// a hybrid whose linear blocks step their recurrent state in the composer's
// scans, a prompt's chunk a run and a ragged step's sequences a run each.
var composeOnceModels = []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "qwen35/Qwen3.5-0.8B-Q4_K_M.gguf"}

// composesOnce is TestPlacementComposesOnce's census of m on g. With to set,
// the relocation moves the blocks on the first one's device to it; without,
// the upper half goes to the CPU and back.
func composesOnce(t *testing.T, m *Model, g *tier.GPU, width int, to string) {
	lengths := []int{1, 5, 33, 100, 511, 512, 513, 2000}
	const decode = 2
	total := 0
	for _, n := range lengths {
		total += n + decode
	}
	text := strings.Repeat("The river rose every spring until the bridges were islands, "+
		"and the clerk counted barrels of salt on the upper floor. ", 400)
	ids := m.Vocab.Encode(text, true)
	if len(ids) < total {
		t.Fatalf("%d prompt tokens, %d wanted", len(ids), total)
	}
	nl := m.Cfg.NLayer
	s := m.NewState(total + 128)
	defer s.Close()
	if err := s.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if s.GPULayers() != nl || !s.HeadOnDevice() {
		t.Fatalf("the devices took %d of %d blocks, the head %v: %s", s.GPULayers(), nl, s.HeadOnDevice(), g.Err())
	}
	st0, c0 := g.Stats(), backend.Compiles()
	// What compiles after the placement, named.
	var mu sync.Mutex
	var compiled []string
	backend.CompileLog = func(n string) {
		mu.Lock()
		compiled = append(compiled, n)
		mu.Unlock()
	}
	defer func() { backend.CompileLog = nil }()
	t.Logf("at placement: %d forms (%d kernels) composed, %d scratches built, a reserved width of %d rows, block forms %v",
		st0.PartsFormed, st0.PartKernels, st0.ScratchBuilds, st0.ReservedRows, st0.BlockForms)
	if st0.PartsFormed == 0 {
		t.Fatalf("the placement composed nothing: %s", g.Err())
	}
	if st0.ReservedRows < 2 || st0.ReservedRows > width {
		t.Fatalf("the devices reserved %d rows, asked for %d: %s", st0.ReservedRows, width, g.Err())
	}
	forms := func(what string, want func(li int) int) {
		t.Helper()
		bf := g.Stats().BlockForms
		for li := range nl {
			got := 0
			if li < len(bf) {
				got = bf[li]
			}
			if w := want(li); got != w {
				t.Fatalf("%s: block %d composed %d time(s), want %d (block forms %v)", what, li, got, w, bf)
			}
		}
	}
	forms("the placement", func(int) int { return 1 })
	held := func(what string) {
		t.Helper()
		t.Logf("%s: %s", what, g.Err())
		st := g.Stats()
		if f, k, sb, c := st.PartsFormed-st0.PartsFormed, st.PartKernels-st0.PartKernels, st.ScratchBuilds-st0.ScratchBuilds,
			backend.Compiles()-c0; f != 0 || k != 0 || sb != 0 || c != 0 {
			mu.Lock()
			defer mu.Unlock()
			t.Fatalf("%s composed %d form(s) of %d kernel(s), built %d scratch(es) and compiled %d kernel(s) after the placement %v",
				what, f, k, sb, c, compiled)
		}
		// A prompt block that declines at a chunk compiles the tier's own
		// kernels there (declineTwins), and a part formed at a step names
		// itself: neither runs after the placement.
		if d, a := sumCounts(st.DeclineBuilt[:])-sumCounts(st0.DeclineBuilt[:]), st.FormedAtStep-st0.FormedAtStep; d != 0 || a != 0 {
			t.Fatalf("%s: %d prompt block(s) declined and built the tier's kernels, %d part(s) formed at a step", what, d, a)
		}
	}
	prompts := func(s *State, off int, lengths []int) int {
		t.Helper()
		served0, _, _ := g.Stats().ServedPart(tier.ComposedPrompt)
		steps := 0
		for _, n := range lengths {
			l, err := s.Prefill(ids[off : off+n])
			if err != nil {
				t.Fatalf("a prompt of %d: %v", n, err)
			}
			off += n
			for range decode {
				if l, err = s.Forward(argmaxID(l)); err != nil {
					t.Fatal(err)
				}
				off++
			}
			held("a prompt of " + strconv.Itoa(n) + " tokens and its decode")
			if n > 1 {
				steps += (n + width - 1) / width * nl
			}
		}
		np, dp, _ := g.Stats().ServedPart(tier.ComposedPrompt)
		if np-served0 < steps || dp != 0 {
			t.Fatalf("the composer served %d prompt block steps of the %d the chunks hold and declined %d: %s",
				np-served0, steps, dp, g.Err())
		}
		t.Logf("%d prompt block steps served", np-served0)
		return off
	}
	prompts(s, 0, lengths)

	s2 := m.NewState(512)
	closed := false
	defer func() {
		if !closed {
			s2.Close()
		}
	}()
	if err := s2.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	held("a second session's placement")
	l, err := s2.Prefill(ids[:100])
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s2.Forward(argmaxID(l)); err != nil {
		t.Fatal(err)
	}
	held("a second session's prompt and decode")
	if to != "" {
		groupedCensus(t, m, g, ids)
		held("decode groups stepping at once on the two cards")
	}

	// Four prompts of more rows than the reserved width: the scheduler admits
	// at most the reserved width a step.
	seqLen := width + 64
	sc := NewScheduler(m, 4, seqLen)
	defer func() {
		if !closed {
			sc.Close()
		}
	}()
	if err := sc.st.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	half := width / 2
	seqs := []*Seq{{Prompt: ids[:7], MaxTokens: 4}, {Prompt: ids[50:83], MaxTokens: 6},
		{Prompt: ids[200 : 200+half+40], MaxTokens: 5}, {Prompt: ids[1000 : 1000+half], MaxTokens: 3}}
	rows := 0
	for _, q := range seqs {
		sc.Submit(q)
		rows += len(q.Prompt)
	}
	if rows <= width {
		t.Fatalf("the batch's %d prompt rows fit the reserved width %d", rows, width)
	}
	for done, n := 0, 0; done < len(seqs); n++ {
		if n > 200 {
			t.Fatalf("the batch made no progress: %d of %d retired", done, len(seqs))
		}
		d, err := sc.Step()
		if err != nil {
			t.Fatal(err)
		}
		done += len(d)
	}
	for i, q := range seqs {
		if q.Err != nil {
			t.Fatalf("sequence %d: %v", i, q.Err)
		}
	}
	held("a ragged batch of " + strconv.Itoa(rows) + " prompt rows")
	forms("the sessions and the batch", func(int) int { return 1 })

	// The relocation: each block moved is composed on its destination, once,
	// and nothing composes after it. The other sessions go first: a block
	// another session holds stays on its card for it.
	sc.Close()
	s2.Close()
	closed = true
	lo, hi, dest := nl/2, nl, to
	if to != "" {
		lo, hi = 0, nl/2
	} else {
		at, _ := s.DeviceOf(lo)
		if at == "" || onCPU(at) {
			t.Fatalf("block %d is on %q, not a named card", lo, at)
		}
		if _, err := s.MoveBlocks(lo, hi, "cpu"); err != nil {
			t.Fatal(err)
		}
		dest = at
	}
	mv, err := s.MoveBlocks(lo, hi, dest)
	if err != nil {
		t.Fatal(err)
	}
	for li := lo; li < hi; li++ {
		if at, _ := s.DeviceOf(li); at != dest {
			t.Fatalf("block %d is on %q after a move to %s (declined %v)", li, at, dest, mv.Declined)
		}
	}
	forms("the relocation", func(li int) int {
		if li >= lo && li < hi {
			return 2
		}
		return 1
	})
	st := g.Stats()
	t.Logf("the relocation of blocks %d..%d to %s: %d forms (%d kernels), %d compiles", lo, hi-1, dest,
		st.PartsFormed-st0.PartsFormed, st.PartKernels-st0.PartKernels, backend.Compiles()-c0)
	st0, c0 = st, backend.Compiles()
	prompts(s, 0, []int{5, 100})
	forms("after the relocation", func(li int) int {
		if li >= lo && li < hi {
			return 2
		}
		return 1
	})
	if testing.Verbose() {
		st := g.Stats()
		t.Logf("after every call: %d forms, %d scratches, %d compiles since the relocation", st.PartsFormed-st0.PartsFormed,
			st.ScratchBuilds-st0.ScratchBuilds, backend.Compiles()-c0)
	}
}

func sumCounts(a []int) int {
	n := 0
	for _, v := range a {
		n += v
	}
	return n
}
