package engine

import (
	"math"
	"os"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestForwardBatchGreedyMatchesEachSequenceAlone decodes several prompts of
// different lengths together through ForwardBatchGreedy -- so the rows sit at
// different positions every step -- and each one alone through ForwardGreedy on
// the same device, and demands the same tokens per row. A row that read another
// row's history, or its own at the wrong base, parts from its solo run.
func TestForwardBatchGreedyMatchesEachSequenceAlone(t *testing.T) {
	p := jlmOf(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.jlm"))
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := model.Open(p)
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
		st := testInstance(t, m, seq)
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

	for _, arm := range []struct {
		name string
		opt  []gpu.Option
	}{{"default", nil}} {
		got, after := batchGreedy(t, m, ids, gen, seq, arm.opt)
		// And a shorter instance on the same card afterwards: the batch grew the
		// device's capacity, and a later session's cache must be that wide or
		// RoPERowsT writes past it (tier.capped).
		for j := range after {
			if after[j] != want[0][j] {
				t.Fatalf("%s: a single instance after the batch, token %d: %d, alone %d\ngot  %v\nwant %v",
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
// row's history (gpu.Faults.SetPagedFault "alias") parts by far more.
const batchTie = 1.0

// hostGap is |logit(a) - logit(b)| on the host after the prompt and prefix.
func hostGap(t *testing.T, m *model.Model, prompt, prefix []int32, a, b int32, io ...Option) float64 {
	t.Helper()
	st := testInstance(t, m, len(prompt)+len(prefix)+1, io...)
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

// batchGreedy decodes every prompt together, a token per step, and returns
// each row's first gen generated tokens, the first prompt decoded afterwards by
// a single instance on the same card, and the count of sm_70 tensor-core matvecs.
func batchGreedy(t *testing.T, m *model.Model, ids [][]int32, gen, seq int, opt []gpu.Option, io ...Option) (got [][]int32, after []int32) {
	got, after, _ = batchGreedyStats(t, m, ids, gen, seq, opt, io...)
	return got, after
}

// batchGreedyStats is batchGreedy and the tier's Stats as the run left them,
// for a gate that has to assert which path ran.
func batchGreedyStats(t *testing.T, m *model.Model, ids [][]int32, gen, seq int, opt []gpu.Option, io ...Option) (got [][]int32, after []int32, stats gpu.Stats) {
	t.Helper()
	g, err := openGPU(t, append([]gpu.Option{gpu.WithDeviceTune(gpu.TuneOff)}, opt...)...)
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()
	st := newSessBatch(m, len(ids), seq, io...)
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
			one := testInstance(t, m, len(ids[0])+gen+1, io...)
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
			return got, after, g.Stats()
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
