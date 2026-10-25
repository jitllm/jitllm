package model

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
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
			got, _, _ := batchGreedy(t, m, ids, gen, 256, []tier.Option{tier.WithDevices(dev), composedOpt(composedParts)})
			composedRowsServed(t, m, lastBatchStats)
			if i, j, gap := composedRowsPart(t, m, ids, want, got); i >= 0 {
				t.Fatalf("row %d token %d: batched %v, the host %v, %.4f apart on the host", i, j, got[i], want[i], gap)
			}
		})
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
	nb, db, _ := st.ServedPart(tier.ComposedBlock)
	nh, dh, _ := st.ServedPart(tier.ComposedHead)
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
