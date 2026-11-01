package model

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestForwardBatchGreedyMatchesEachSequenceAlone decodes several prompts of
// different lengths together through ForwardBatchGreedy -- so the rows sit at
// different positions every step -- and each one alone through ForwardGreedy on
// the same device, and demands the same tokens per row. A row that read another
// row's history, or its own at the wrong base, parts from its solo run.
func TestForwardBatchGreedyMatchesEachSequenceAlone(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.jlm")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := Open(p)
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
	const gen, seq = 16, 256
	want := make([][]int32, len(prompts))
	ids := make([][]int32, len(prompts))
	for i, pr := range prompts {
		ids[i] = m.Vocab.Encode(pr, true)
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

	// Three arms: the matrix-instruction matvec where the card has one, and
	// the two a card without one takes -- forced here.
	for _, arm := range []struct {
		name  string
		opt   []tier.Option
		volta bool
	}{{"default", nil, false},
		// no-mma takes a binary16 tensor-core matvec (sm_70's m8n8k4, the
		// m16n8 GEMM from sm_75); no-volta the dp4a tile.
		{"no-mma", []tier.Option{tier.WithConfig(func(c *tier.Config) { c.NoMMA = true })}, true},
		{"no-mma-no-volta", []tier.Option{tier.WithConfig(func(c *tier.Config) { c.NoMMA, c.NoVolta = true, true })}, false}} {
		got, after, volta := batchGreedy(t, m, ids, gen, seq, arm.opt)
		// The arm that names the Volta kernel must have built it: a refused
		// shape falls to dp4a and would pass this gate on the other kernel.
		// Only PTX lowers the m8n8k4 shape; elsewhere (Metal, SPIR-V) the
		// no-mma arm is the dp4a tile too, and must not claim otherwise.
		if lowers := strings.Contains(lastBatchName, "[ptx]"); arm.name != "default" &&
			(volta > 0) != (arm.volta && lowers) {
			t.Fatalf("%s: %d matvecs on sm_70's f16 tensor cores", arm.name, volta)
		}
		// And a shorter State on the same card afterwards: the batch grew the
		// device's capacity, and a later State's cache must be that wide or
		// RoPERowsT writes past it (tier.capped).
		for j := range after {
			if after[j] != want[0][j] {
				t.Fatalf("%s: a single State after the batch, token %d: %d, alone %d\ngot  %v\nwant %v",
					arm.name, j, after[j], want[0][j], after, want[0])
			}
		}
		for i := range prompts {
			for j := 0; j < gen; j++ {
				if got[i][j] == want[i][j] {
					continue
				}
				// A tie is not a wrong row: the runs reduce in different orders
				// and near-equal logits flip either way. The host says whether
				// this was one; past it the row has diverged.
				if gap := hostGap(t, m, ids[i], want[i][:j], want[i][j], got[i][j]); gap > batchTie {
					t.Fatalf("%s: row %d token %d: batched %d, alone %d, %.4f apart on the host\ngot  %v\nwant %v",
						arm.name, i, j, got[i][j], want[i][j], gap, got[i][:gen], want[i])
				} else {
					t.Logf("%s: row %d token %d: a tie (%.4f apart on the host), not compared past it", arm.name, i, j, gap)
				}
				break
			}
		}
	}
	for i := range prompts {
		t.Logf("row %d: %d tokens identical on both arms: %q", i, gen, m.Vocab.Decode(want[i]))
	}
}

// batchTie is how close two logits may be for a flip between the batched and
// solo device runs to count as reduction order rather than a wrong row: the
// two arms run different matvec kernels, and teacher-forced on these five
// prompts they part by up to 0.55-0.90 of a logit per row on CUDA
// (0.47 a step on average), so two candidates closer than that on the host may
// come out either way. At 0.05 the gate failed on a 0.157 flip that a change
// to the activation quantizer's last bit moved onto row 0, with the two arms
// still 0.47 apart a step on average either side of it. A row reading another
// row's history (tier.SetPagedFault "alias") parts by far more.
const batchTie = 1.0

// hostGap is |logit(a) - logit(b)| on the host after the prompt and prefix.
func hostGap(t *testing.T, m *Model, prompt, prefix []int32, a, b int32) float64 {
	t.Helper()
	st := m.NewState(len(prompt) + len(prefix) + 1)
	defer st.Close()
	var lg []float32
	var err error
	for _, id := range append(append([]int32{}, prompt...), prefix...) {
		if lg, err = st.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	return math.Abs(float64(lg[a] - lg[b]))
}

// lastBatchStats is the tier's Stats as batchGreedy's last run left them, for a
// gate that has to assert which path ran.
var lastBatchStats tier.Stats

// lastBatchName is the device batchGreedy's last run stepped on, as
// tier.GPU.Name prints it: which backend took the batch.
var lastBatchName string

// batchGreedy decodes every prompt together, a token per step, and returns
// each row's first gen generated tokens, the first prompt decoded afterwards by
// a single State on the same card, and the count of sm_70 tensor-core matvecs.
func batchGreedy(t *testing.T, m *Model, ids [][]int32, gen, seq int, opt []tier.Option) (got [][]int32, after []int32, volta int) {
	t.Helper()
	g, err := tier.OpenWith(append([]tier.Option{tier.WithDeviceTune(tier.TuneOff)}, opt...)...)
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()
	st := m.NewBatch(len(ids), seq)
	st.SetDevice(g)
	if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
		t.Fatalf("batch: placed %d of %d blocks: %q", st.GPULayers(), m.Cfg.NLayer, g.Err())
	}
	got = make([][]int32, len(ids))
	feed := make([]int32, len(ids))
	for step := 0; ; step++ {
		done := true
		for i := range ids {
			if step < len(ids[i]) {
				feed[i] = ids[i][step]
			}
			done = done && len(got[i]) >= gen
		}
		if done {
			st.Close()
			one := m.NewState(len(ids[0]) + gen + 1)
			defer one.Close()
			one.SetDevice(g)
			var next int32
			for _, id := range ids[0] {
				if next, err = one.ForwardGreedy(id); err != nil {
					t.Fatal(err)
				}
			}
			for len(after) < gen {
				after = append(after, next)
				if next, err = one.ForwardGreedy(next); err != nil {
					t.Fatal(err)
				}
			}
			lastBatchStats, lastBatchName = g.Stats(), g.Name()
			return got, after, lastBatchStats.VoltaMV + lastBatchStats.GemmF16
		}
		out, err := st.ForwardBatchGreedy(feed)
		if err != nil {
			t.Fatal(err)
		}
		for i := range ids {
			if step >= len(ids[i])-1 {
				got[i] = append(got[i], out[i])
				feed[i] = out[i]
			}
		}
	}
}
