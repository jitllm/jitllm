package model

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/dev/bench"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestPagedKVAB is the attention's end-to-end A/B: the decode plan
// (kernels.ChooseDecodePlan) against pieces of the rule before it, in one
// process, interleaved (RULE 2), a two-tier A/A control first and the
// comparison twice. Each arm is its own tier on the same card, with its own
// State, so the two never share a pool or a scratch. It is what measured the
// paged history against the contiguous cache it replaced
// (docs/engineering-history/gpu-kernels.md, "The decode plan through the tier").
//
//	JITLLM_PAGED_AB=1 JITLLM_MODEL=Llama-3.2-1B-Instruct-Q4_K_M.gguf \
//	  [JITLLM_DEV=gpu] [JITLLM_AB_DEPTH=512] [JITLLM_AB_ROUNDS=20] [JITLLM_WALL=GB/s]
//	  [JITLLM_AB_BUDGET=MiB per arm] [JITLLM_AB_VS=planoff[:N]]
//
// Decode: both States prefilled to the depth, then each round decodes a run of
// tokens, so the depth grows alike in both arms. Prefill: each round a fresh
// State prefills the depth's worth of prompt. Ratios are other/plan time:
// above 1, the plan is faster.
func TestPagedKVAB(t *testing.T) {
	if os.Getenv("JITLLM_PAGED_AB") == "" {
		t.Skip("set JITLLM_PAGED_AB=1")
	}
	p := testmodels.Resolve(os.Getenv("JITLLM_MODEL"))
	if p == "" {
		t.Skip("set JITLLM_MODEL")
	}
	spec := os.Getenv("JITLLM_DEV")
	if spec == "" {
		spec = "gpu:0"
	}
	num := func(k string, d int) int {
		if n, err := strconv.Atoi(os.Getenv(k)); err == nil && n > 0 {
			return n
		}
		return d
	}
	depth, rounds := num("JITLLM_AB_DEPTH", 512), num("JITLLM_AB_ROUNDS", 20)
	const run = 16 // decoded tokens a round
	wall := 0.0
	fmt.Sscanf(os.Getenv("JITLLM_WALL"), "%g", &wall)

	m, err := Open(jlmOf(t, p), noTune)
	if err != nil {
		t.Skipf("cannot load: %v", err)
	}
	defer m.Close()
	seq := depth + run*(4*rounds+8) + 8
	prompt := make([]int32, depth)
	for i := range prompt {
		prompt[i] = int32(1 + (i*7919)%1000)
	}

	type arm struct {
		name string
		g    *tier.GPU
		dec  *State
	}
	closeArm := func(a *arm) {
		a.dec.Close()
		a.g.Close()
	}
	// JITLLM_AB_VS=planoff:N puts back the pieces of kernels.ChooseDecodePlan
	// that N names in the other arm (Config.DecodePlanOff); 7, the whole old
	// rule, by default.
	planOff := 7
	if vs, ok := strings.CutPrefix(os.Getenv("JITLLM_AB_VS"), "planoff:"); ok {
		if n, err := strconv.Atoi(vs); err == nil && n > 0 {
			planOff = n
		}
	}
	open := func(name string, other bool) *arm {
		opts := []tier.Option{tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff),
			tier.WithConfig(func(c *tier.Config) {
				if other {
					c.DecodePlanOff = planOff
				}
			})}
		// JITLLM_AB_BUDGET (MiB) splits a small card evenly: the arm opened
		// first otherwise takes what it can and the second gets the rest.
		if mib := num("JITLLM_AB_BUDGET", 0); mib > 0 {
			opts = append(opts, tier.WithBudget(uint64(mib)<<20))
		}
		g, err := openTier(t, opts...)
		if err != nil || g == nil {
			noDevice(t, spec, err)
		}
		s := m.NewState(seq)
		if err := s.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if s.devCount() != m.Cfg.NLayer || !s.HeadOnDevice() {
			t.Fatalf("%s: placed %d of %d blocks: %s", name, s.devCount(), m.Cfg.NLayer, g.Err())
		}
		if _, err := s.Prefill(prompt); err != nil {
			t.Fatal(err)
		}
		return &arm{name: name, g: g, dec: s}
	}
	decode := func(a *arm) bench.Case {
		return bench.Case{Name: a.name, Fn: func(iters int) {
			for i := 0; i < iters; i++ {
				if _, err := a.dec.ForwardGreedy(int32(1 + i%100)); err != nil {
					t.Fatal(err)
				}
			}
		}}
	}
	prefill := func(a *arm) bench.Case {
		return bench.Case{Name: a.name, Fn: func(iters int) {
			for i := 0; i < iters; i++ {
				s := m.NewState(depth + 1)
				if err := s.SetDevice(a.g); err != nil {
					t.Fatal(err)
				}
				if _, err := s.Prefill(prompt); err != nil {
					t.Fatal(err)
				}
				s.Close()
			}
		}}
	}
	gbs := func(r bench.Result, tokens int, kvPos int) string {
		// A decoded token reads the weights and every position's K and V.
		bpt := float64(m.BytesPerToken()) + float64(kvPos)*float64(m.Cfg.NLayer*m.Cfg.NKVHead*m.Cfg.HeadDim)*8
		rate := float64(tokens) / r.BMedian.Seconds()
		s := fmt.Sprintf("paged %.2f tok/s = %.1f GB/s", rate, rate*bpt/1e9)
		if wall > 0 {
			s += fmt.Sprintf(" = %.0f%% of the %.1f GB/s wall", 100*rate*bpt/1e9/wall, wall)
		}
		return s
	}

	// The two arms are two tiers, and the one opened FIRST decodes slightly
	// faster whatever it holds: two tiers of the SAME configuration
	// differ, which a one-tier A/A cannot see. So the A/A is two tiers of
	// the paged configuration, and each pass opens its arms fresh in the
	// order the other pass did not -- pass 1 the other arm first, pass 2 the
	// paged one -- so the order bias lands on each side once.
	a1, a2 := open("paged-A", false), open("paged-B", false)
	aa := bench.AB(decode(a1), decode(a2), rounds, run)
	aaP := bench.AB(prefill(a1), prefill(a2), max(4, rounds/2), 1)
	t.Logf("decode A/A, two paged tiers, depth %d: %s", depth, aa)
	t.Logf("prefill A/A, two paged tiers, %d tokens: %s", depth, aaP)
	closeArm(a1)
	closeArm(a2)
	for pass := 1; pass <= 2; pass++ {
		var other, paged *arm
		if pass == 1 {
			other = open("old rule", true)
			paged = open("paged", false)
		} else {
			paged = open("paged", false)
			other = open("old rule", true)
		}
		first := map[int]string{1: "the other arm", 2: "paged"}[pass]
		r := bench.AB(decode(other), decode(paged), rounds, run)
		t.Logf("decode pass %d (%s opened first), depth %d+: %s -- %s", pass, first, depth, r, gbs(r, run, depth))
		rp := bench.AB(prefill(other), prefill(paged), max(4, rounds/2), 1)
		t.Logf("prefill pass %d (%s opened first), %d tokens: %s -- paged %.1f tok/s", pass, first, depth, rp,
			float64(depth)/rp.BMedian.Seconds())
		closeArm(other)
		closeArm(paged)
	}
}
