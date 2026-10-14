package model

import (
	"fmt"
	"os"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/dev/bench"
	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestBlockFusionAB is fused host decode (sched.Pool.Fuse) against the per-op
// path, in one process, interleaved (RULE 2): an A/A of the fused arm against
// itself first, then the comparison twice. Both arms are one State, its fusion
// switched before each arm's run of tokens (SetBlockFusion), so they share a
// pool, a cache and a history, and the depth grows alike under both; a second
// State beside it would be another user of the cores and close fusion to both.
// Each arm is checked to have run the path it names. Ratios are per-op/fused
// time: above 1, fused decode is faster.
//
//	JITLLM_FUSE_AB=1 JITLLM_MODEL=Llama-3.2-1B-Instruct-Q4_K_M.gguf \
//	  [JITLLM_AB_DEPTH=64] [JITLLM_AB_ROUNDS=20]
func TestBlockFusionAB(t *testing.T) {
	if os.Getenv("JITLLM_FUSE_AB") == "" {
		t.Skip("set JITLLM_FUSE_AB=1")
	}
	p := testmodels.Resolve(os.Getenv("JITLLM_MODEL"))
	if p == "" {
		t.Skip("set JITLLM_MODEL")
	}
	num := func(k string, d int) int {
		if n, err := strconv.Atoi(os.Getenv(k)); err == nil && n > 0 {
			return n
		}
		return d
	}
	depth, rounds := num("JITLLM_AB_DEPTH", 64), num("JITLLM_AB_ROUNDS", 20)
	const run = 16 // decoded tokens an arm's turn
	m, err := Open(jlmOf(t, p), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	// Six comparisons of rounds+1 turns of two arms each, run tokens a turn.
	s := m.NewState(depth + 6*2*(rounds+1)*run + 8)
	defer s.Close()
	prompt := make([]int32, depth)
	for i := range prompt {
		prompt[i] = int32(1 + (i*7919)%1000)
	}
	if _, err := s.Prefill(prompt); err != nil {
		t.Fatal(err)
	}
	arm := func(name string, fused bool) (bench.Case, *int64) {
		var runs int64
		return bench.Case{Name: name, Fn: func(iters int) {
			s.SetBlockFusion(fused)
			r0 := s.FusedRuns
			for i := range iters {
				if _, err := s.ForwardGreedy(int32(1 + i%100)); err != nil {
					t.Fatal(err)
				}
			}
			runs += s.FusedRuns - r0
		}}, &runs
	}
	cores := len(sched.DecodeCores())
	wall := bench.MemWall(cores)
	bpt := float64(m.BytesPerToken())
	rate := func(r bench.Result) string {
		tps := float64(run) / r.BMedian.Seconds()
		return fmt.Sprintf("fused %.2f tok/s = %.1f GB/s = %.0f%% of the %.1f GB/s %d-thread wall measured here",
			tps, tps*bpt/1e9, 100*tps*bpt/wall, wall/1e9, cores)
	}
	fa, fb := "fused-A", "fused-B"
	a, aRuns := arm(fa, true)
	b, bRuns := arm(fb, true)
	aa := bench.AB(a, b, rounds, run)
	t.Logf("A/A, fused against fused, depth %d+, %d tokens a turn: %s", depth, run, aa)
	if *aRuns == 0 || *bRuns == 0 {
		t.Fatalf("the A/A's arms opened %d and %d fused runs: it controlled nothing", *aRuns, *bRuns)
	}
	for pass := 1; pass <= 2; pass++ {
		per, perRuns := arm("per-op", false)
		fu, fuRuns := arm("fused", true)
		r := bench.AB(per, fu, rounds, run)
		t.Logf("pass %d, depth %d+, %d tokens a turn: %s -- %s", pass, depth, run, r, rate(r))
		if *perRuns != 0 || *fuRuns == 0 {
			t.Fatalf("the per-op arm opened %d fused runs and the fused arm %d: the arms are not the paths they name",
				*perRuns, *fuRuns)
		}
	}
}
