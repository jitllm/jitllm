package model

import (
	"os"
	"slices"
	"testing"
	"time"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// composedRowsPrompts are prompts of different lengths, so a ragged step's
// rows sit at different positions every step.
var composedRowsPrompts = []string{"Once upon a time, in a small village by the sea, there lived", "1, 2, 3,", "Water boils at"}

// TestComposedRowsServe decodes composedRowsPrompts together on each device
// with the composer asked for the blocks and the head: every step is a
// ragged step, its rows of three sessions at their own positions, each
// reading its own history through its descriptor, and the head projects
// every row. The composer served every block step and every head and
// declined none, and each row's tokens are the host's greedy decode to a
// tie (batchTie), as TestForwardBatchGreedyMatchesEachSequenceAlone judges
// a row.
func TestComposedRowsServe(t *testing.T) {
	m := composedRowsModel(t)
	defer m.Close()
	ids := make([][]int32, len(composedRowsPrompts))
	for i, p := range composedRowsPrompts {
		ids[i] = m.Vocab.Encode(p, true)
	}
	const gen = 12
	want := make([][]int32, len(ids))
	for i := range ids {
		want[i] = hostGreedy(t, m, ids[i], gen)
	}
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			got, _ := batchGreedy(t, m, ids, gen, 256, []tier.Option{tier.WithDevices(dev)})
			composedRowsServed(t, m, lastBatchStats)
			if i, j, gap := composedRowsPart(t, m, ids, want, got); i >= 0 {
				t.Fatalf("row %d token %d: batched %v, the host %v, %.4f apart on the host", i, j, got[i], want[i], gap)
			}
		})
	}
}

// TestComposedRowsTokens decodes composedRowsPrompts together on each
// device with every part asked of the composer (tier.ComposedStep), every
// row's logits read back: each ragged step's greedy tokens are the
// composer's sample over its rows, served at every step and declined at
// none, and each row's token is the host's Greedy over that row's logits,
// capped where the model has a final softcap (gemma-2). With the token left
// to the tier's per-row argmax the logits are the same bits and so are the
// tokens; over the logits negated (a wrong pick) the tokens part from the
// host's.
func TestComposedRowsTokens(t *testing.T) {
	ran, capped := 0, 0
	for _, file := range []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "gemma-2-2b-it-Q4_K_M.gguf"} {
		t.Run(file, func(t *testing.T) {
			p := testmodels.Path(file)
			if _, err := os.Stat(p); err != nil {
				t.Fatalf("MODEL MISSING: %v (set JITLLM_MODELS)", err)
			}
			m, err := Open(jlmOf(t, p), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			composedRowsTokens(t, m)
			ran++
			if m.Cfg.FinalSoftcap != 0 {
				capped++
			}
		})
	}
	if ran != 2 || capped == 0 {
		t.Fatalf("%d of 2 models ran, %d with a final softcap", ran, capped)
	}
}

func composedRowsTokens(t *testing.T, m *Model) {
	t.Helper()
	ids := make([][]int32, len(composedRowsPrompts))
	for i, p := range composedRowsPrompts {
		ids[i] = m.Vocab.Encode(p, true)
	}
	const gen = 12
	nv := m.Cfg.NVocab
	parted := func(toks [][]int32, logits [][]float32) (bad, rows int) {
		for s := range toks {
			for r, tok := range toks[s] {
				rows++
				if tok != Greedy(logits[s][r*nv:(r+1)*nv]) {
					bad++
				}
			}
		}
		return bad, rows
	}
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			toks, logits, steps, st := rowsTokens(t, m, ids, gen, tier.WithDevices(dev))
			ns, ds := st.ServedPart(tier.ComposedSample)
			ns += st.ReplayedPart(tier.ComposedSample)
			if ns != steps || ds != 0 || st.StepDeclines[jit.DeclineNoSample] != 0 || st.FormedPart(tier.ComposedSample) != 0 {
				t.Fatalf("the composer selected the tokens of %d of %d ragged steps, declined %d, formed %d at a step (%s; last error %q)",
					ns, steps, ds, st.FormedPart(tier.ComposedSample), declinesOf(st), st.LastErr)
			}
			if bad, rows := parted(toks, logits); bad != 0 {
				t.Fatalf("%d of %d rows' composed tokens are not the host's Greedy over their logits", bad, rows)
			}
			// The violation: the token over the logits negated.
			set := m.frontGraphs()
			orig := set.cpuSample
			m.front.mu.Lock()
			set.cpuSample = negatedSample(orig)
			m.front.mu.Unlock()
			vt, vl, _, vst := rowsTokens(t, m, ids, gen, tier.WithDevices(dev),
				tier.WithConfig(func(c *tier.Config) { c.NoGraph = true }))
			m.front.mu.Lock()
			set.cpuSample = orig
			m.front.mu.Unlock()
			vn, _ := vst.ServedPart(tier.ComposedSample)
			bad, rows := parted(vt, vl)
			if vn == 0 || bad == 0 {
				t.Fatalf("over the logits negated the composer selected %d steps' tokens and %d of %d rows part: the gate cannot see a wrong pick", vn, bad, rows)
			}
			t.Logf("%d ragged steps' tokens (%d rows, softcap %g) selected by the composer, the tier's argmax's bit for bit; over the logits negated %d of %d part",
				steps, rows, m.Cfg.FinalSoftcap, bad, rows)
		})
	}
}

// rowsTokens decodes ids together for gen tokens a row on a fresh tier of
// opts, every row's logits read back each step: each step's rows' device
// tokens and logits, how many steps ran, and the tier's stats.
func rowsTokens(t *testing.T, m *Model, ids [][]int32, gen int, opts ...tier.Option) (toks [][]int32, logits [][]float32, steps int, st tier.Stats) {
	t.Helper()
	g, err := tier.OpenWith(append([]tier.Option{tier.WithDeviceTune(tier.TuneOff)}, opts...)...)
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()
	s := m.NewBatch(len(ids), 256)
	defer s.Close()
	s.SetDevice(g)
	if s.GPULayers() != m.Cfg.NLayer || !s.HeadOnDevice() {
		t.Fatalf("batch: placed %d of %d blocks: %q", s.GPULayers(), m.Cfg.NLayer, g.Err())
	}
	feed, got := make([]int32, len(ids)), make([]int, len(ids))
	var took []time.Duration
	for step := 0; ; step++ {
		done := true
		for i := range ids {
			if step < len(ids[i]) {
				feed[i] = ids[i][step]
			}
			done = done && got[i] >= gen
		}
		if done {
			slices.Sort(took[len(took)/2:])
			t.Logf("%s: the median of the last %d steps %v", g.Name(), len(took)-len(took)/2, took[len(took)/2+(len(took)-len(took)/2)/2])
			return toks, logits, steps, g.Stats()
		}
		t0 := time.Now()
		lg, err := s.ForwardBatch(feed)
		if err != nil {
			t.Fatal(err)
		}
		took = append(took, time.Since(t0))
		tk := slices.Clone(s.head.Tokens[:len(ids)])
		toks, logits, steps = append(toks, tk), append(logits, slices.Clone(lg)), steps+1
		for i := range ids {
			if step >= len(ids[i])-1 {
				feed[i] = tk[i]
				got[i]++
			}
		}
	}
}

// composedRowsModel is the model the rows gates decode.
func composedRowsModel(t *testing.T) *Model {
	t.Helper()
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("MODEL MISSING: %v (set JITLLM_MODELS)", err)
	}
	m, err := Open(jlmOf(t, p), noTune)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// composedRowsServed holds a batch's stats to the composer having served
// block steps of every block and heads, and declined none.
func composedRowsServed(t *testing.T, m *Model, st tier.Stats) {
	t.Helper()
	nb, db := st.ServedPart(tier.ComposedBlock)
	nh, dh := st.ServedPart(tier.ComposedHead)
	nb += st.ReplayedPart(tier.ComposedBlock)
	if nb == 0 || nb%m.Cfg.NLayer != 0 || db != 0 || nh == 0 || dh != 0 {
		t.Fatalf("the composer served %d block steps of %d blocks and %d heads, and declined %d and %d (%s; last error %q)",
			nb, m.Cfg.NLayer, nh, db, dh, declinesOf(st), st.LastErr)
	}
	t.Logf("the composer served %d block steps and %d heads of the ragged steps, %d formed at a step", nb, nh, st.FormedAtStep)
}

// composedRowsPart is the first row and token where got parts from want past
// a tie on the host, and the gap there; -1 where every row holds.
func composedRowsPart(t *testing.T, m *Model, ids, want, got [][]int32) (int, int, float64) {
	t.Helper()
	for i := range ids {
		for j := range want[i] {
			if got[i][j] == want[i][j] {
				continue
			}
			if gap := hostGap(t, m, ids[i], want[i][:j], want[i][j], got[i][j]); gap > batchTie {
				return i, j, gap
			}
			t.Logf("row %d token %d: a tie, not compared past it", i, j)
			break
		}
	}
	return -1, 0, 0
}
