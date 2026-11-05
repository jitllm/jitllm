package engine

import (
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestBlocksPageOnTheListedCard: the CPU is a device the caller lists or does
// not (WithCPU), never a default where blocks land. With one card listed and a
// budget three fifths of the model, every block runs on that card, paged
// through its slots, at every step of a teacher-forced generation (WhereIs over every
// block after the prompt and after each token), with the output projection on
// the card and the same greedy token at every step as the card holding the
// whole model resident. Each block is composed once on the card however often
// it pages, and once more when a caller moves it to the CPU and back
// (gpu.Stats.BlockForms). A budget below the slots a swap needs fails
// SetDevice by name and leaves nothing placed.
//
// VIOLATION SIGNATURE. The same card and budget with the CPU listed spills the
// blocks the card does not hold to the CPU, and the WhereIs check, run on it,
// must name them; a gate whose check never saw a block on the CPU proves
// nothing.
func TestBlocksPageOnTheListedCard(t *testing.T) {
	path := jlmOf(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
	for _, spec := range testdevices.Specs("cuda:0") {
		t.Run(spec, func(t *testing.T) { blocksPageOnTheListedCard(t, path, spec) })
	}
}

func blocksPageOnTheListedCard(t *testing.T, path, spec string) {
	probe, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	// Three fifths of the model: the output projection and a few block slots
	// beside a backend's scratch, not every block resident.
	budget := probe.WeightBytes() * 3 / 5
	probe.Close()
	card := nn.DeviceName(spec)
	const ntok = 8

	// attach opens the card under budget (0: none) and an instance of a model
	// with the CPU listed or not.
	attach := func(t *testing.T, budget uint64, cpu bool) (*model.Model, *Instance, *gpu.GPU, error) {
		t.Helper()
		ds := spec
		if budget > 0 {
			ds += "=" + strconv.FormatUint(budget, 10)
		}
		g, err := openGPU(t, gpu.WithDevices(ds), gpu.WithDeviceTune(gpu.TuneOff))
		if err != nil || g == nil {
			noDevice(t, ds, err)
		}
		devs := []string{spec}
		if cpu {
			devs = append(devs, "cpu")
		}
		instOpts := []Option{WithKVF16(false), WithDevices(devs...)}
		m, err := model.Open(path)
		if err != nil {
			g.Close()
			t.Fatal(err)
		}
		st := testInstance(t, m, 64, instOpts...)
		err = st.SetDevice(g)
		t.Cleanup(func() { st.Close(); g.Close(); m.Close() })
		return m, st, g, err
	}
	// offCard is every block not on the card.
	offCard := func(st *Instance) []string {
		var off []string
		locs := st.Locations()
		for li := st.lo; li < st.hi; li++ {
			if at := st.WhereIs(li); at != card || locs[li] != at {
				off = append(off, strconv.Itoa(li)+"@"+at)
			}
		}
		return off
	}
	// run generates ntok tokens, teacher-forced on ids when given, checking
	// where every block is after each step.
	run := func(t *testing.T, m *model.Model, st *Instance, ids []int32) ([]int32, [][]float32) {
		t.Helper()
		check := func(step int) {
			if off := offCard(st); len(off) > 0 {
				t.Fatalf("step %d: blocks off %s: %v", step, card, off)
			}
			if !st.HeadOnDevice() {
				t.Fatalf("step %d: the output projection is not on %s", step, card)
			}
		}
		logits, err := st.Prefill(m.Vocab.Encode("The capital of France is a city that", true))
		if err != nil {
			t.Fatal(err)
		}
		var toks []int32
		var lg [][]float32
		for i := 0; i < ntok; i++ {
			check(i)
			lg = append(lg, append([]float32(nil), logits...))
			next := Greedy(logits)
			toks = append(toks, next)
			if ids != nil {
				next = ids[i]
			}
			if logits, err = st.Forward(next); err != nil {
				t.Fatal(err)
			}
		}
		check(ntok)
		return toks, lg
	}

	m, ctl, cg, err := attach(t, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	nl := m.Cfg.NLayer
	if ctl.devCount() != nl {
		t.Fatalf("the unbudgeted card holds %d of %d blocks: %s", ctl.devCount(), nl, cg.Err())
	}
	want, wantLg := run(t, m, ctl, nil)

	t.Run("pages", func(t *testing.T) {
		m, st, g, err := attach(t, budget, false)
		if err != nil {
			t.Fatal(err)
		}
		if !st.HeadOnDevice() {
			t.Fatalf("the output projection is not on %s: %s", card, g.Err())
		}
		got, lg := run(t, m, st, want)
		worst := 0.0
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("step %d: token %d paged, %d resident", i, got[i], want[i])
			}
			for j := range lg[i] {
				worst = math.Max(worst, math.Abs(float64(lg[i][j]-wantLg[i][j])))
			}
		}
		ps := g.Stats()
		if ps.PageIns <= nl || ps.PageOuts == 0 {
			t.Errorf("%d page-in(s), %d page-out(s) over %d blocks: the blocks did not page", ps.PageIns, ps.PageOuts, nl)
		}
		t.Logf("%s at %d MiB: %d of %d blocks on the card, %d page-ins, %d page-outs, worst logit diff %.3g",
			card, budget>>20, st.devCount(), nl, ps.PageIns, ps.PageOuts, worst)
		if composedByDefault {
			for i, n := range ps.BlockDeclines() {
				if n > 0 {
					t.Errorf("the composer declined %d paged block step(s): %s", n, gpu.DeclineName(i))
				}
			}
			if n := composedBlockSteps(g); n < ntok*nl {
				t.Errorf("the composer served %d block steps, where %d tokens over %d paged blocks take at least %d", n, ntok, nl, ntok*nl)
			}
		}

		// Composed once, then served: each block's one-row part formed once
		// on the card however often its weights paged, and once more for a
		// relocation (Stats.BlockForms). In a build whose composer serves no
		// block, nothing forms.
		once := 0
		if composedByDefault {
			once = 1
		}
		formsOf := func(li int) int {
			if f := g.Stats().BlockForms; li < len(f) {
				return f[li]
			}
			return 0
		}
		for li := 0; li < nl; li++ {
			if n := formsOf(li); n != once {
				t.Errorf("block %d formed %d time(s) on %s over %d page-ins, want %d", li, n, card, ps.PageIns, once)
			}
		}
		// A caller's move of the last block to the CPU and back: the check
		// names the block while it is on the CPU, and coming back forms it
		// once more.
		last := nl - 1
		if mv, err := st.MoveBlocks(last, nl, nn.CPU); err != nil || len(mv.Moved) != 1 {
			t.Fatalf("block %d to the CPU: %v (%+v)", last, err, mv)
		}
		if off := offCard(st); len(off) != 1 || off[0] != strconv.Itoa(last)+"@"+nn.CPU {
			t.Fatalf("with block %d moved to the CPU the check names %v", last, off)
		}
		if _, err := st.Forward(want[ntok-1]); err != nil {
			t.Fatal(err)
		}
		if mv, err := st.MoveBlocks(last, nl, card); err != nil || len(mv.Moved) != 1 {
			t.Fatalf("block %d back to %s: %v (%+v)", last, card, err, mv)
		}
		if _, err := st.Forward(want[0]); err != nil {
			t.Fatal(err)
		}
		if off := offCard(st); len(off) != 0 {
			t.Fatalf("blocks off %s after the move back: %v", card, off)
		}
		if n := formsOf(last); n != 2*once {
			t.Errorf("block %d formed %d time(s) after one relocation, want %d", last, n, 2*once)
		}
	})

	t.Run("too-small-fails-by-name", func(t *testing.T) {
		_, st, _, err := attach(t, 1<<20, false)
		if err == nil || !strings.Contains(err.Error(), "run on no listed card even paged") ||
			!strings.Contains(err.Error(), "CPU is not one of the devices listed") {
			t.Fatalf("a 1 MiB card with the CPU unlisted: err %v", err)
		}
		if st.devCount() != 0 {
			t.Fatalf("%d blocks left placed after the refusal", st.devCount())
		}
		t.Logf("refused: %v", err)
	})

	t.Run("violation-cpu-listed-spills", func(t *testing.T) {
		_, st, _, err := attach(t, budget, true)
		if err != nil {
			t.Fatal(err)
		}
		off := offCard(st)
		if len(off) == 0 {
			t.Fatalf("with the CPU listed every block is on %s at three fifths of the model: "+
				"the budget does not bite, so the paged arm proves nothing", card)
		}
		t.Logf("the CPU listed: %d block(s) off %s, which the check names: %v", len(off), card, off)
	})
}
