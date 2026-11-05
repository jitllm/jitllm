package engine

import (
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// pagedKVPage is the page the gates run at: the smallest the kernels take, so
// a short prompt crosses several page edges and a window starts mid-page.
const pagedKVPage = 64

// pagedOpts is the tier configuration of a paged arm at page size page.
func pagedOpts(page int) []gpu.Option {
	return []gpu.Option{gpu.WithConfig(func(c *gpu.Config) { c.KVPage = page })}
}

// logitNMSE is |a-b|^2 / |b|^2 over one logit row.
func logitNMSE(a, b []float32) float64 {
	var num, den float64
	for i := range b {
		d := float64(a[i] - b[i])
		num += d * d
		den += float64(b[i]) * float64(b[i])
	}
	return num / den
}

// TestPagedKVBatchMatchesHost runs three sequences of different lengths as one
// batch with every row's history in its own pages, and holds each row's
// greedy tokens to the host's up to a tie -- the check that rows of one
// session never read each other's pages.
func TestPagedKVBatchMatchesHost(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := model.Open(jlmOf(t, p))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompts := []string{
		strings.Repeat("Once upon a time, in a small village by the sea, there lived ", 3),
		"1, 2, 3,",
		"Water boils at",
	}
	ids := make([][]int32, len(prompts))
	for i, s := range prompts {
		ids[i] = m.Vocab.Encode(s, true)
	}
	const gen, seq = 12, 256
	got, after, stats := batchGreedyStats(t, m, ids, gen, seq, append([]gpu.Option{gpu.WithDevices("gpu:0")}, pagedOpts(pagedKVPage)...))
	if n, d := stats.ServedPart(nn.ComposedBlock); n == 0 || d != 0 {
		t.Fatalf("the composer served %d of the batch's block steps and declined %d", n, d)
	}
	for i := range ids {
		want := hostGreedy(t, m, ids[i], gen)
		for j := range want {
			if got[i][j] == want[j] {
				continue
			}
			if gap := hostGap(t, m, ids[i], want[:j], want[j], got[i][j]); gap > batchTie {
				t.Fatalf("row %d token %d: %d paged in a batch, %d on the host, %.4f apart", i, j, got[i][j], want[j], gap)
			}
			break
		}
	}
	want0 := hostGreedy(t, m, ids[0], gen)
	for j := range want0 {
		if after[j] != want0[j] {
			if gap := hostGap(t, m, ids[0], want0[:j], want0[j], after[j]); gap > batchTie {
				t.Fatalf("the single instance after the batch, token %d: %d, host %d, %.4f apart", j, after[j], want0[j], gap)
			}
			break
		}
	}
	t.Logf("%d rows, %d tokens each", len(ids), gen)
}

// firstModel is the first of |-separated file names present in the model
// directory, or the first name's path when none is (the caller's Stat reports
// it): hosts name the same checkpoint differently.
func firstModel(names string) string {
	for _, n := range strings.Split(names, "|") {
		if p := testmodels.Path(n); fileExists(p) {
			return p
		}
	}
	return testmodels.Path(strings.Split(names, "|")[0])
}

// hostRun is pagedRun on the host: the prompt one token at a time, then the
// forced ids, every step's logits.
func hostRun(t *testing.T, m *model.Model, seq int, prompt, forced []int32, io ...Option) [][]float32 {
	t.Helper()
	st := testInstance(t, m, seq, io...)
	defer st.Close()
	var lg []float32
	var err error
	for _, id := range prompt {
		if lg, err = st.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	out := [][]float32{slices.Clone(lg)}
	for _, id := range forced {
		if lg, err = st.Forward(id); err != nil {
			t.Fatal(err)
		}
		out = append(out, slices.Clone(lg))
	}
	return out
}

// hostGreedy is prompt's first gen greedy tokens on the host.
func hostGreedy(t *testing.T, m *model.Model, prompt []int32, gen int, io ...Option) []int32 {
	t.Helper()
	st := testInstance(t, m, len(prompt)+gen+1, io...)
	defer st.Close()
	var next int32
	var err error
	for _, id := range prompt {
		if next, err = st.ForwardGreedy(id); err != nil {
			t.Fatal(err)
		}
	}
	out := make([]int32, 0, gen)
	for len(out) < gen {
		out = append(out, next)
		if next, err = st.ForwardGreedy(next); err != nil {
			t.Fatal(err)
		}
	}
	return out
}
