package model

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestSchedulerOnDeviceMatchesEachSequenceAlone runs the Scheduler with every
// block and the head on one device: each step one ragged step of the decoding
// rows and a budget of prompt rows (the wanted rows' logits read back and
// sampled on the host). There are more prompts than
// rows, so later ones are admitted into rows an earlier sequence retired from,
// while the other rows are mid-decode. Each sequence must give the tokens it
// gives alone through ForwardGreedy on the same device.
func TestSchedulerOnDeviceMatchesEachSequenceAlone(t *testing.T) {
	schedulerOnDevice(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
}

// TestSchedulerOnDeviceHybrid is the same gate on a hybrid, whose prompt
// chunks step their sequence's recurrent state as runs on the device, into
// rows whose state Retire zeroed.
func TestSchedulerOnDeviceHybrid(t *testing.T) {
	schedulerOnDevice(t, testmodels.Path("qwen35/Qwen3.5-0.8B-Q4_K_M.gguf"))
}

func schedulerOnDevice(t *testing.T, p string) {
	t.Helper()
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := Open(jlmOf(t, p))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	schedulerOnDeviceModel(t, m, swaTier)
}

// schedulerOnDeviceModel is the gate on an open model, every arm on a tier
// open makes. A latent model's blocks must also have run as ragged rows
// (Stats.RowsLatent).
func schedulerOnDeviceModel(t *testing.T, m *Model, open func(*testing.T) *tier.GPU) {
	t.Helper()
	var err error
	prompts := []string{
		"The capital of France is",
		"Once upon a time, in a small village by the sea, there lived",
		"1, 2, 3,",
		"The quick brown fox jumps over the lazy dog. The quick brown fox",
		"Water boils at",
	}
	// Different budgets, so rows retire at different steps and admission
	// lands mid-decode rather than only at the start.
	gens := []int{12, 20, 8, 16, 10}
	const seq, rows = 256, 3

	want := make([][]int32, len(prompts))
	ids := make([][]int32, len(prompts))
	for i, pr := range prompts {
		ids[i] = m.Vocab.Encode(pr, true)
		g := open(t)
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
		for len(want[i]) < gens[i] {
			want[i] = append(want[i], next)
			if next, err = st.ForwardGreedy(next); err != nil {
				t.Fatal(err)
			}
		}
		st.Close()
		g.Close()
	}

	g := open(t)
	defer g.Close()
	sc := NewScheduler(m, rows, seq)
	defer sc.Close()
	sc.st.SetDevice(g)
	if sc.st.GPULayers() != m.Cfg.NLayer || !sc.st.HeadOnDevice() {
		t.Fatalf("batch: placed %d of %d blocks: %q", sc.st.GPULayers(), m.Cfg.NLayer, g.Err())
	}
	seqs := make([]*Seq, len(prompts))
	for i := range prompts {
		seqs[i] = &Seq{Prompt: ids[i], MaxTokens: gens[i]}
		sc.Submit(seqs[i])
	}
	retired, admittedLate := 0, false
	for steps := 0; retired < len(seqs); steps++ {
		if steps > 200 {
			t.Fatalf("no progress: %d of %d retired", retired, len(seqs))
		}
		if sc.Pending() > 0 && sc.Live() > 0 && steps > 0 {
			admittedLate = true
		}
		done, err := sc.Step()
		if err != nil {
			t.Fatal(err)
		}
		retired += len(done)
	}
	if !admittedLate {
		t.Fatal("every sequence was admitted at the start: the gate never admitted into a row mid-decode")
	}
	if n := latentBlocks(m); n > 0 && g.Stats().RowsLatent == 0 {
		t.Fatalf("%d latent block(s) and none ran in a ragged step: the schedule went somewhere else (%s)",
			n, g.Err())
	} else if n > 0 {
		t.Logf("%d latent block-steps ran as ragged rows", g.Stats().RowsLatent)
	}
	for i, s := range seqs {
		if s.Err != nil {
			t.Fatalf("row %d: %v", i, s.Err)
		}
		if len(s.Out) != gens[i] {
			t.Fatalf("row %d: %d tokens, want %d", i, len(s.Out), gens[i])
		}
		for j := range s.Out {
			if s.Out[j] == want[i][j] {
				continue
			}
			// A tie is not a wrong row. The bound is the device's own band, not
			// batchTie: a joint step fed decode's tokens differs from decode
			// alone by up to 0.53-0.70 of a logit on an arm64 host, on the tiled twins
			// and the few-sequence matvec alike, and one 0.069 flip there failed
			// this gate at batchTie's 0.05. A row reading another row's history
			// diverges by whole logits, not by a tie.
			if gap := hostGap(t, m, ids[i], want[i][:j], want[i][j], s.Out[j]); gap > tieMargin {
				t.Fatalf("row %d token %d: scheduled %d, alone %d, %.4f apart on the host\ngot  %v\nwant %v",
					i, j, s.Out[j], want[i][j], gap, s.Out, want[i])
			} else {
				t.Logf("row %d token %d: a tie (%.4f apart on the host), not compared past it", i, j, gap)
			}
			break
		}
		t.Logf("row %d: %q", i, m.Vocab.Decode(s.Out))
	}
}
