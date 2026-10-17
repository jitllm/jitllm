package model

import (
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
)

func graphModel(t *testing.T, name string) string {
	p := testmodels.Path(name + ".jlm")
	if src := testmodels.Path(name + ".gguf"); fileExists(src) {
		p = jlmOf(t, src)
	}
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	return p
}

// graphDecode greedily decodes n tokens after the prompt and returns every
// position's logits, the State's graph block count and its warm allocations.
func graphDecode(t *testing.T, m *Model, n, frames int) ([][]float32, int, float64) {
	if frames != 0 {
		m.SetPageBudget(uint64(frames) * m.PageSize())
	}
	s := m.NewState(64)
	defer s.Close()
	var out [][]float32
	tok := m.Vocab.Encode("The capital of France is", true)
	var next int32
	for i := 0; i < len(tok)+n; i++ {
		in := next
		if i < len(tok) {
			in = tok[i]
		}
		lg, err := s.Forward(in)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
		next = Greedy(lg)
	}
	allocs := testing.AllocsPerRun(4, func() {
		if _, err := s.Forward(next); err != nil {
			t.Fatal(err)
		}
	})
	return out, s.GraphBlocks(), allocs
}

// TestBlockGraphMatchesTheHandPath: llama's block and qwen3's (per-head q/k
// norm) run through the block graph's lowering give the hand-written body's
// logits bit for bit at every position, the graph path is the one that ran
// (every block, every token), a warm token allocates nothing, a page budget
// below the model gives the same answer, and the comparison can tell two
// different computations apart (an f16 cache against f32).
func TestBlockGraphMatchesTheHandPath(t *testing.T) {
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M", "Qwen3-0.6B-Q8_0"} {
		t.Run(name, func(t *testing.T) {
			p := graphModel(t, name)
			ctl, err := Open(p, noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer ctl.Close()
			gm, err := Open(p, noTune, WithKVF16(false), WithBlockGraph(true))
			if err != nil {
				t.Fatal(err)
			}
			defer gm.Close()
			if g := gm.BlockGraph(0); g == nil {
				t.Fatal("no graph for block 0")
			} else {
				t.Logf("block 0:\n%s", g)
			}
			const n = 16
			nl := gm.Cfg.NLayer
			var want [][]float32
			for _, budget := range []int{0, nl / 3} {
				ref, ran, refAllocs := graphDecode(t, ctl, n, budget)
				if ran != 0 {
					t.Fatalf("the control ran %d graph blocks", ran)
				}
				if budget == 0 {
					want = ref
				}
				got, ran, allocs := graphDecode(t, gm, n, budget)
				t.Logf("budget %d frames: warm allocations, graph %.0f, hand-written %.0f", budget, allocs, refAllocs)
				if want := nl * (len(got) + 5); ran != want { // AllocsPerRun runs once more than asked
					t.Fatalf("budget %d: %d blocks ran through the graph, want %d", budget, ran, want)
				}
				// The graph adds no allocation: zero where the hand-written path
				// makes none, and no more than it where a budget below the
				// model pages.
				if allocs > refAllocs {
					t.Errorf("budget %d: a warm graph token allocates %.1f times, the hand-written %.1f", budget, allocs, refAllocs)
				}
				if budget == 0 && allocs != 0 {
					t.Errorf("a warm resident graph token allocates %.1f times", allocs)
				}
				for i := range want {
					for j := range want[i] {
						if got[i][j] != want[i][j] {
							t.Fatalf("budget %d: position %d logit %d: graph %v, hand-written %v", budget, i, j, got[i][j], want[i][j])
						}
					}
				}
			}
			// The violation: the same comparison over a different computation
			// must fire.
			f16, err := Open(p, noTune, WithKVF16(true), WithBlockGraph(true))
			if err != nil {
				t.Fatal(err)
			}
			defer f16.Close()
			other, _, _ := graphDecode(t, f16, n, 0)
			differ := false
			for i := range want {
				for j := range want[i] {
					differ = differ || other[i][j] != want[i][j]
				}
			}
			if !differ {
				t.Fatal("an f16 cache gave the f32 cache's logits bit for bit: the comparison cannot see a change")
			}
		})
	}
}

// TestBlockGraphRefusesWhatItDoesNotCover: a near miss (gemma-3's block has
// post-norms, a sliding window and GELU) is refused at Open, naming why,
// rather than run through a lowering that would drop them.
func TestBlockGraphRefusesWhatItDoesNotCover(t *testing.T) {
	p := graphModel(t, "gemma-3-1b-it-Q4_K_M")
	m, err := Open(p, noTune, WithBlockGraph(true))
	if err == nil {
		m.Close()
		t.Fatal("gemma-3 opened under WithBlockGraph")
	}
	if !strings.Contains(err.Error(), "does not cover gemma3") {
		t.Fatalf("refused for the wrong reason: %v", err)
	}
}

// TestBlockGraphOfTheRealContainersVerifies: the facts are read from the
// container, and a graph without its q/k norm is refused against them.
func TestBlockGraphOfTheRealContainersVerifies(t *testing.T) {
	m, err := Open(graphModel(t, "Qwen3-0.6B-Q8_0"), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	f, err := m.blockFacts(5)
	if err != nil {
		t.Fatal(err)
	}
	if err := graph.Verify(f.Config.Build(), f); err != nil {
		t.Fatal(err)
	}
	d := f.Config
	d.QKNorm = false
	if err := graph.Verify(d.Build(), f); err == nil || !strings.Contains(err.Error(), "attn_q_norm") {
		t.Fatalf("a graph without qwen3's q/k norm: %v", err)
	}
}

// graphPrompt prefills a prompt of n tokens on a fresh State over a page
// budget of frames (0 keeps the model's), decodes after it, and returns the
// logits after the prompt and after each decoded token, the block-chunks the
// graph ran and the allocations of a warm second prompt on another slot.
func graphPrompt(t *testing.T, m *Model, n, frames int) ([][]float32, int) {
	t.Helper()
	if frames != 0 {
		m.SetPageBudget(uint64(frames) * m.PageSize())
	}
	s := m.NewState(n + 16)
	defer s.Close()
	ids := make([]int32, n)
	for i := range ids {
		ids[i] = int32(100 + (i*7919)%3000)
	}
	lg, err := s.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	out := [][]float32{append([]float32(nil), lg...)}
	for i := 0; i < 4; i++ {
		if lg, err = s.Forward(Greedy(lg)); err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	return out, s.GraphChunks()
}

// TestBlockGraphPromptMatchesTheHandPath is Slice B's host half: a prompt's
// chunks run through the block graph's row lowering (graphRows) give the
// hand-written chunk body's logits bit for bit after the prompt and at the
// decoded positions after it, every block of every chunk ran through the
// graph, a page budget below the model gives the same answer, and the
// comparison fires on a different computation (an f16 cache).
func TestBlockGraphPromptMatchesTheHandPath(t *testing.T) {
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M", "Qwen3-0.6B-Q8_0"} {
		t.Run(name, func(t *testing.T) {
			p := graphModel(t, name)
			ctl, err := Open(p, noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer ctl.Close()
			gm, err := Open(p, noTune, WithKVF16(false), WithBlockGraph(true))
			if err != nil {
				t.Fatal(err)
			}
			defer gm.Close()
			// Several chunks: past the widest chunk the host tuner offers.
			const n = 700
			nl := gm.Cfg.NLayer
			var want [][]float32
			for _, budget := range []int{0, nl / 3} {
				ref, ran := graphPrompt(t, ctl, n, budget)
				if ran != 0 {
					t.Fatalf("the control ran %d graph chunks", ran)
				}
				if budget == 0 {
					want = ref
				}
				got, ran := graphPrompt(t, gm, n, budget)
				if ran == 0 || ran%nl != 0 || ran/nl < 2 {
					t.Fatalf("budget %d: %d block-chunks ran through the graph: want every block of two or more chunks", budget, ran)
				}
				t.Logf("budget %d frames: %d chunks of %d blocks through the graph", budget, ran/nl, nl)
				for i := range want {
					for j := range want[i] {
						if got[i][j] != want[i][j] {
							t.Fatalf("budget %d: position %d logit %d: graph %v, hand-written %v", budget, i, j, got[i][j], want[i][j])
						}
					}
				}
			}
			f16, err := Open(p, noTune, WithKVF16(true), WithBlockGraph(true))
			if err != nil {
				t.Fatal(err)
			}
			defer f16.Close()
			other, _ := graphPrompt(t, f16, n, 0)
			differ := false
			for i := range want {
				for j := range want[i] {
					differ = differ || other[i][j] != want[i][j]
				}
			}
			if !differ {
				t.Fatal("an f16 cache gave the f32 cache's logits bit for bit: the comparison cannot see a change")
			}
		})
	}
}
