package model

import (
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestForwardBatchGreedyHybridMatchesEachSequenceAlone is the batched-decode
// gate on a hybrid: five sequences decoded together on the device, their
// linear blocks stepping five independent recurrent states in one launch
// (kernels' run forms, a run of one row each, tier.growSeat), against each sequence decoded
// alone. It catches sequences sharing a history (every row but the first
// fluent and wrong) and a captured graph replayed at the wrong recurrent
// parity. JITLLM_HYBRID_BATCH_MODEL names another hybrid container.
func TestForwardBatchGreedyHybridMatchesEachSequenceAlone(t *testing.T) {
	batchAloneGate(t, "JITLLM_HYBRID_BATCH_MODEL", "qwen35/Qwen3.5-0.8B-Q4_K_M.jlm",
		"a hybrid", func(c Config) bool { return c.Hybrid() }, false)
}

// TestForwardBatchGreedyMixtureMatchesEachSequenceAlone is the same gate on a
// mixture: a batched step routes each row to its own experts, and the rows'
// selections must not bleed into one another. JITLLM_MOE_BATCH_MODEL names
// another mixture container.
//
// It compares against each sequence batched alone, bit for bit: batched
// arithmetic is not decode's, and routing turns a few ulps into another expert,
// so a solo decode is outside any tie band. A batched column does not depend on
// the other columns, so what is left to differ is routing, grouping and the
// combine per row.
func TestForwardBatchGreedyMixtureMatchesEachSequenceAlone(t *testing.T) {
	batchAloneGate(t, "JITLLM_MOE_BATCH_MODEL", "Qwen3-MOE-4x0.6B-Q4_K_M.jlm",
		"a mixture", func(c Config) bool { return c.NExpert > 1 }, true)
}

// TestForwardBatchGreedyWindowedMatchesEachSequenceAlone is the gate on a
// sliding-window model: a batched step attends each row over its own last W
// keys (LayersRows' ragBaseW/ragNW). synth-gptoss's window is 4, so every
// sequence crosses it within its prompt. JITLLM_SWA_BATCH_MODEL names another.
func TestForwardBatchGreedyWindowedMatchesEachSequenceAlone(t *testing.T) {
	batchAloneGate(t, "JITLLM_SWA_BATCH_MODEL", "synth-gptoss.gguf",
		"a sliding-window model", func(c Config) bool { return c.SWAWindow > 0 }, false)
}

// batchAloneGate decodes five sequences batched on the device against each
// decoded alone on the device -- or, exact, against each batched alone, bit for
// bit; is says the model is the kind the gate is for.
func batchAloneGate(t *testing.T, env, def, kind string, is func(Config) bool, exact bool) {
	name := os.Getenv(env)
	if name == "" {
		name = def
	}
	p := testmodels.Path(name)
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	if strings.HasSuffix(p, ".gguf") {
		p = jlmOf(t, p)
	}
	m, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !is(*m.Cfg) {
		t.Fatalf("%s is not %s: this gate would test nothing it exists for", name, kind)
	}
	prompts := []string{
		"The capital of France is",
		"Once upon a time, in a small village by the sea, there lived",
		"1, 2, 3,",
		"The quick brown fox jumps over the lazy dog. The quick brown fox",
		"Water boils at",
	}
	const gen, seq = 16, 256
	want := make([][]int32, len(prompts))
	ids := make([][]int32, len(prompts))
	for i, pr := range prompts {
		ids[i] = m.Vocab.Encode(pr, true)
		if exact {
			one, _, _ := batchGreedy(t, m, ids[i:i+1], gen, seq, nil)
			want[i] = one[0]
			continue
		}
		g := swaTier(t)
		st := m.NewState(seq)
		st.SetDevice(g)
		if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
			t.Fatalf("solo: placed %d of %d blocks: %q", st.GPULayers(), m.Cfg.NLayer, g.Err())
		}
		var next int32
		for _, id := range ids[i] {
			if next, err = st.ForwardGreedy(id); err != nil {
				t.Fatal(err)
			}
		}
		for len(want[i]) < gen {
			want[i] = append(want[i], next)
			if next, err = st.ForwardGreedy(next); err != nil {
				t.Fatal(err)
			}
		}
		st.Close()
		g.Close()
	}
	got, after, _ := batchGreedy(t, m, ids, gen, seq, nil)
	for i := range prompts {
		if exact && !slices.Equal(got[i][:gen], want[i]) {
			t.Fatalf("row %d: batched with four others %v, batched alone %v", i, got[i][:gen], want[i])
		}
	}
	for j := 0; j < len(after) && !exact; j++ {
		if after[j] != want[0][j] {
			t.Fatalf("a single State after the batch, token %d: %d, alone %d\ngot  %v\nwant %v",
				j, after[j], want[0][j], after, want[0])
		}
	}
	for i := range prompts {
		for j := 0; j < gen; j++ {
			if got[i][j] == want[i][j] {
				continue
			}
			// The device's own band, as TestSchedulerOnDeviceMatchesEachSequenceAlone
			// bounds it: synth-gptoss's random weights flip a 0.19 tie at row
			// 4 token 10, on the tree before and after, where a row
			// reading another's history parts by whole logits.
			if gap := hostGap(t, m, ids[i], want[i][:j], want[i][j], got[i][j]); gap > tieMargin {
				t.Fatalf("row %d token %d: batched %d, alone %d, %.4f apart on the host\ngot  %v\nwant %v",
					i, j, got[i][j], want[i][j], gap, got[i][:gen], want[i])
			} else {
				t.Logf("row %d token %d: a tie (%.4f apart on the host), not compared past it", i, j, gap)
			}
			break
		}
		t.Logf("row %d: %q", i, m.Vocab.Decode(got[i]))
	}
}

// TestSharedRecurrentStateMatchesTheDoubledOne: a batch whose recurrent states
// are grown in the shared form (tier.Config.SharedRec -- one resident state a
// block, the device's next-state scratch and a copy back) decodes the same
// tokens, bit for bit, as the double-buffered form. It is the form a wide batch
// falls to when two halves do not fit (Qwen3.5-4B at 128 sequences on a 16 GB card),
// and the copy is exact, so equality is the bar. Stats.RecShared asserts it
// was the shared form that ran; without the copy back every step after the
// first reads the state the batch started with, and the rows part at once.
func TestSharedRecurrentStateMatchesTheDoubledOne(t *testing.T) {
	name := os.Getenv("JITLLM_HYBRID_BATCH_MODEL")
	if name == "" {
		name = "qwen35/Qwen3.5-0.8B-Q4_K_M.jlm"
	}
	p := testmodels.Path(name)
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v -- this gate proved nothing", err)
	}
	m, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !m.Cfg.Hybrid() {
		t.Fatalf("%s is not a hybrid", name)
	}
	var ids [][]int32
	for _, pr := range []string{"The capital of France is", "1, 2, 3,", "Water boils at"} {
		ids = append(ids, m.Vocab.Encode(pr, true))
	}
	const gen, seq = 12, 64
	want, _, _ := batchGreedy(t, m, ids, gen, seq, nil)
	if lastBatchStats.RecShared != 0 {
		t.Fatalf("the doubled arm grew %d shared state(s)", lastBatchStats.RecShared)
	}
	got, _, _ := batchGreedy(t, m, ids, gen, seq, []tier.Option{
		tier.WithConfig(func(c *tier.Config) { c.SharedRec = true }),
	})
	shared := lastBatchStats.RecShared
	if shared == 0 {
		t.Fatal("no recurrent state was grown in the shared form: the gate compared the doubled form with itself")
	}
	for i := range ids {
		if !slices.Equal(got[i][:gen], want[i][:gen]) {
			t.Fatalf("row %d: shared %v, doubled %v", i, got[i][:gen], want[i][:gen])
		}
	}
	t.Logf("%d shared state(s); %d rows identical over %d tokens", shared, len(ids), gen)
}
