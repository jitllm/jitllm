package model

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestSchedulerMixesPromptsIntoDecodeSteps is the gate on chunked admission: a
// prompt rides in the decoding rows' steps as a chunk of consecutive positions
// of one sequence, several rows in one step, its K and V written beside the
// other rows' and read by its own later rows in the same step.
//
// The budget is four prompt tokens a step, so every prompt here spans steps --
// a chunk's first row is the last chunk's successor -- and there are more
// prompts than rows, so later ones are admitted while the others decode. Each
// sequence must give the tokens it gives alone: exactly on the host, and on a
// device up to a first difference that is a tie on the host (tieMargin, the
// device's own band; see TestSchedulerOnDeviceMatchesEachSequenceAlone).
//
// It selects what it gates -- prompt rows rode beside decoding rows, more than
// one of a prompt in such a step, and a prompt left part-fed between steps --
// and it runs the violation: the same schedule with each chunk's rows laid out
// at mirrored positions must part from the sequences alone by more than a tie.
//
// It runs a dense model and a hybrid. A hybrid's chunk steps its sequence's
// one recurrent state a row at a time in position order, which on a device is
// the run forms of the linear kernels (tier.Stats.RecChainRows, the selection
// check there: a chunk's rows past its first chained through its state).
// JITLLM_TEST_DEVICES names the devices; the default is
// every one this project has.
func TestSchedulerMixesPromptsIntoDecodeSteps(t *testing.T) {
	for _, c := range []struct{ name, path string }{
		{"dense", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"},
		{"hybrid", "qwen35/Qwen3.5-0.8B-Q4_K_M.gguf"},
	} {
		t.Run(c.name, func(t *testing.T) { mixedModel(t, testmodels.Path(c.path)) })
	}
}

func mixedModel(t *testing.T, p string) {
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	// noGEMM: the host arm compares exactly, and rows batched beside others
	// would otherwise take the weight-stationary GEMM where the sequence alone
	// takes the matvec (see TestBatchMatchesForward).
	m, err := Open(jlmOf(t, p), noTune, noGEMM)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompts := []string{
		"The capital of France is",
		"Once upon a time, in a small village by the sea, there lived",
		"1, 2, 3,",
		"The quick brown fox jumps over the lazy dog. The quick brown fox",
		"Water boils at",
	}
	gens := []int{12, 20, 8, 16, 10}
	ids := make([][]int32, len(prompts))
	for i, pr := range prompts {
		ids[i] = m.Vocab.Encode(pr, true)
	}
	t.Run("host", func(t *testing.T) { mixedArm(t, m, ids, gens, "") })
	// Every backend this host has: a step's matvecs and attention are chosen
	// per device, and a gate that asks for one covers none of the others.
	specs := testdevices.Specs("cuda:0", "vulkan:0", "vulkan:1", "metal")
	for _, spec := range specs {
		t.Run(spec, func(t *testing.T) { mixedArm(t, m, ids, gens, spec) })
	}
}

// mixedArm runs the gate on one placement: the host for spec "", else every
// block and the head on the device spec names.
func mixedArm(t *testing.T, m *Model, ids [][]int32, gens []int, spec string) {
	c := newMixedCase(t, m, ids, gens, spec)
	got, st, split, chained := c.run(nil)
	t.Logf("%d steps, %d prompt rows, %d of them beside a decoding row, at most %d of one prompt in a step",
		st.Steps, st.PromptRows, st.MixedRows, st.MaxChunk)
	// The configuration under test must be the one that ran.
	switch {
	case st.MixedRows == 0:
		t.Fatal("no prompt row shared a step with a decoding row: admission still stalled the batch")
	case st.MaxChunk < 2:
		t.Fatal("no step carried two rows of one prompt beside a decoding row: prompts went a token per step")
	case !split:
		t.Fatal("no prompt was left part-fed between steps: no chunk ever continued another")
	case spec != "" && m.Cfg.Hybrid() && chained == 0:
		t.Fatal("no row of a prompt chunk chained through a device linear block's state after another: " +
			"the hybrid's chunks did not ride as runs")
	}
	if m.Cfg.Hybrid() && spec != "" {
		t.Logf("%d row-steps of linear blocks chained a chunk's state on the device", chained)
	}
	if w, i, j := c.worst(got); i >= 0 && w > c.tie {
		t.Fatalf("row %d token %d: scheduled %d, alone %d, %.4f apart on the host (bound %.2f)\ngot  %v\nwant %v",
			i, j, got[i][j], c.want[i][j], w, c.tie, got[i], c.want[i])
	} else if i >= 0 {
		t.Logf("row %d token %d: a tie (%.4f apart on the host), not compared past it", i, j, w)
	}
	for i := range got {
		t.Logf("row %d: %q", i, m.Vocab.Decode(got[i]))
	}

	// The violation: every chunk laid out back to front, its first token at
	// its last position. Every row still names a position the sequence owns,
	// so nothing refuses it; only the tokens can tell.
	bad, _, _, _ := c.run(func(base, j, k int) int { return base + k - 1 - j })
	w, i, j := c.worst(bad)
	if i < 0 || w <= c.tie {
		t.Fatalf("chunks at mirrored positions still matched each sequence alone (widest first difference %.4f): "+
			"this gate cannot see a chunk's positions", w)
	}
	t.Logf("violation: row %d parts at token %d, %.4f apart on the host", i, j, w)
}

// mixedCase is the Scheduler gate on one placement: the sequences' tokens
// alone, and how a scheduled run is compared against them.
type mixedCase struct {
	t    *testing.T
	m    *Model
	ids  [][]int32
	gens []int
	spec string
	want [][]int32
	// extra is added to every tier open() makes: a violation gate arms its
	// fault here, after want was taken clean.
	extra []tier.Option
	// tie is how far apart on the host the first difference may be: none on
	// the host, whose batched rows compute exactly what one sequence does.
	tie float64
}

const mixedSeq, mixedRows, mixedBudget = 256, 3, 4

// newMixedCase runs each sequence alone on spec's placement: Prefill and
// Forward on the host, the device's own greedy decode on a device.
func newMixedCase(t *testing.T, m *Model, ids [][]int32, gens []int, spec string) *mixedCase {
	t.Helper()
	c := &mixedCase{t: t, m: m, ids: ids, gens: gens, spec: spec, want: make([][]int32, len(ids))}
	if spec != "" {
		c.tie = tieMargin
	}
	for i := range ids {
		g := c.open()
		st := m.NewState(mixedSeq)
		c.place(st, g)
		var next int32
		if g == nil {
			lg, err := st.Prefill(ids[i])
			if err != nil {
				t.Fatal(err)
			}
			next = Greedy(lg)
		} else {
			for _, id := range ids[i] {
				var err error
				if next, err = st.ForwardGreedy(id); err != nil {
					t.Fatal(err)
				}
			}
		}
		for len(c.want[i]) < gens[i] {
			c.want[i] = append(c.want[i], next)
			if g == nil {
				lg, err := st.Forward(next)
				if err != nil {
					t.Fatal(err)
				}
				next = Greedy(lg)
				continue
			}
			var err error
			if next, err = st.ForwardGreedy(next); err != nil {
				t.Fatal(err)
			}
		}
		st.Close()
		if g != nil {
			g.Close()
		}
	}
	return c
}

func (c *mixedCase) open() *tier.GPU {
	if c.spec == "" {
		return nil
	}
	g, err := openTier(c.t, append(append([]tier.Option{tier.WithDevices(c.spec), tier.WithDeviceTune(tier.TuneOff)},
		testTierOpts(c.t)...), c.extra...)...)
	if err != nil || g == nil {
		noDevice(c.t, c.spec, err)
	}
	c.t.Logf("%s is %s", c.spec, g.Name())
	return g
}

func (c *mixedCase) place(st *State, g *tier.GPU) {
	if g == nil {
		return
	}
	st.SetDevice(g)
	if st.GPULayers() != c.m.Cfg.NLayer || !st.HeadOnDevice() {
		c.t.Fatalf("placed %d of %d blocks, head on the device %v: %q", st.GPULayers(), c.m.Cfg.NLayer,
			st.HeadOnDevice(), g.Err())
	}
}

// run schedules every sequence through one Scheduler, chunks at chunkPos's
// positions where it is set, and returns each sequence's tokens, the
// Scheduler's stats, whether a prompt was ever left part-fed between steps,
// and the rows that chained a device linear block's state after another row of
// their own sequence (tier.Stats.RecChainRows).
func (c *mixedCase) run(chunkPos func(base, j, k int) int) ([][]int32, SchedulerStats, bool, int) {
	t := c.t
	g := c.open()
	if g != nil {
		defer g.Close()
	}
	sc := NewScheduler(c.m, mixedRows, mixedSeq, WithPromptBudget(mixedBudget))
	defer sc.Close()
	sc.chunkPos = chunkPos
	c.place(sc.st, g)
	seqs := make([]*Seq, len(c.ids))
	for i := range c.ids {
		seqs[i] = &Seq{Prompt: c.ids[i], MaxTokens: c.gens[i]}
		sc.Submit(seqs[i])
	}
	split := false
	for retired, steps := 0, 0; retired < len(seqs); steps++ {
		if steps > 400 {
			t.Fatalf("no progress: %d of %d retired", retired, len(seqs))
		}
		done, err := sc.Step()
		if err != nil {
			t.Fatal(err)
		}
		retired += len(done)
		for _, s := range sc.live {
			if s != nil && s.fed > 0 && s.fed < len(s.Prompt) {
				split = true
			}
		}
	}
	// A latent model's steps must have run its latent blocks as ragged rows
	// on the device, each reading its own sequence's pages.
	if g != nil && latentBlocks(c.m) > 0 && g.Stats().RowsLatent == 0 {
		t.Fatalf("no latent block ran in a ragged step: the schedule went somewhere else (%s)", g.Err())
	}
	out := make([][]int32, len(seqs))
	for i, s := range seqs {
		if s.Err != nil {
			t.Fatalf("row %d: %v", i, s.Err)
		}
		if len(s.Out) != c.gens[i] {
			t.Fatalf("row %d: %d tokens, want %d", i, len(s.Out), c.gens[i])
		}
		out[i] = s.Out
	}
	chained := 0
	if g != nil {
		chained = g.Stats().RecChainRows
		checkLinearRowsForm(t, c.m, g)
	}
	return out, sc.Stats(), split, chained
}

// worst is the widest first difference over the sequences: 0 where every
// token matches, and past a first difference nothing is compared.
func (c *mixedCase) worst(got [][]int32) (float64, int, int) {
	w, wi, wj := 0.0, -1, -1
	for i := range got {
		for j := range got[i] {
			if got[i][j] == c.want[i][j] {
				continue
			}
			if gap := hostGap(c.t, c.m, c.ids[i], c.want[i][:j], c.want[i][j], got[i][j]); gap >= w {
				w, wi, wj = gap, i, j
			}
			break
		}
	}
	return w, wi, wj
}
