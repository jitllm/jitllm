package model

import (
	"cmp"
	"maps"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/metal"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestRaggedStepKernelSplit is the measurement docs/design/ragged-prefill.md
// starts from: the same prompts as rows of one batch session's steps
// (Scheduler, a ragged step) and each prompt alone (Prefill), every block on
// the device, with each kernel's device time taken by the backend's own
// instrument (backend.SetCUDAKernelTiming, metal.SetKernelTiming). It
// measures and asserts nothing (RULE 4), and runs only when JITLLM_SPLIT_MODEL
// names a model.
//
//	JITLLM_SPLIT_MODEL    the model (.gguf or .jlm)
//	JITLLM_TEST_DEVICES   the device (default gpu)
//	JITLLM_SPLIT_PROMPTS  prompts (default 4), JITLLM_SPLIT_TOKENS each (default 900)
//
// Per-kernel timing puts each Metal dispatch in a command buffer of its own,
// so the totals are each kernel alone, not the step's wall time; both arms
// pay it alike. The wall times are printed from an untimed run of each arm.
func TestRaggedStepKernelSplit(t *testing.T) {
	path := os.Getenv("JITLLM_SPLIT_MODEL")
	if path == "" {
		t.Skip("a measurement harness: JITLLM_SPLIT_MODEL names its model")
	}
	spec := testdevices.One("gpu")
	np, ntok := envInt("JITLLM_SPLIT_PROMPTS", 4), envInt("JITLLM_SPLIT_TOKENS", 900)
	m, err := Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	// No recordings: a replayed graph's kernels are timed one replay late,
	// and a ragged step's shapes rarely repeat, so a recorded arm's split
	// would miss most of its launches. Both arms issue every launch.
	g, err := openTier(t, tier.WithDevices(spec), tier.WithConfig(func(c *tier.Config) { c.NoGraph = true }))
	if err != nil || g == nil {
		t.Fatalf("%s: %v", spec, err)
	}
	defer g.Close()
	vocab := min(m.Cfg.NVocab, 30000)
	prompts := make([][]int32, np)
	for i := range prompts {
		prompts[i] = linPrompt(ntok+7*i, vocab)[i:]
	}
	seq := ntok + 8*np + 8

	st := m.NewState(seq)
	defer st.Close()
	if err := st.SetDeviceLayers(g, -1); err != nil || st.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("alone: %v, %d of %d blocks: %s", err, st.GPULayers(), m.Cfg.NLayer, g.Err())
	}
	sc := NewScheduler(m, np, seq)
	defer sc.Close()
	if err := sc.st.SetDeviceLayers(g, -1); err != nil || sc.st.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("batch: %v, %d of %d blocks: %s", err, sc.st.GPULayers(), m.Cfg.NLayer, g.Err())
	}
	alone := func() {
		for _, p := range prompts {
			st.Reset()
			if _, err := st.Prefill(p); err != nil {
				t.Fatal(err)
			}
		}
	}
	seqs := make([]Seq, np)
	batch := func() {
		for i := range seqs {
			seqs[i] = Seq{Prompt: prompts[i], MaxTokens: 1, Out: seqs[i].Out[:0]}
			sc.Submit(&seqs[i])
		}
		for sc.Pending() > 0 || sc.Live() > 0 {
			done, err := sc.Step()
			if err != nil {
				t.Fatal(err)
			}
			for _, s := range done {
				if s.Err != nil {
					t.Fatal(s.Err)
				}
			}
		}
	}
	// Warm both (kernels compiled, tables built), then the walls.
	alone()
	batch()
	wall := func(f func()) time.Duration {
		best := time.Duration(1 << 62)
		for range 3 {
			t0 := time.Now()
			f()
			best = min(best, time.Since(t0))
		}
		return best
	}
	wa, wb := wall(alone), wall(batch)
	t.Logf("RESULT walls, best of 3: %d prompts of ~%d tokens alone %v, as one batch %v (%.2fx)",
		np, ntok, wa, wb, wb.Seconds()/wa.Seconds())

	timed := func(f func()) (map[string]float64, map[string]int) {
		backend.SetCUDAKernelTiming(true)
		metal.SetKernelTiming(true)
		backend.CUDAKernelTimes()
		metal.KernelTimes()
		f()
		us, n := backend.CUDAKernelTimes()
		if mu, mn := metal.KernelTimes(); len(mu) > 0 {
			us, n = mu, mn
		}
		backend.SetCUDAKernelTiming(false)
		metal.SetKernelTiming(false)
		return us, n
	}
	ua, na := timed(alone)
	ub, nb := timed(batch)
	report := func(arm string, us map[string]float64, n map[string]int) {
		tot := 0.0
		fam := map[string]float64{}
		for k, v := range us {
			tot += v
			fam[family(k)] += v
		}
		t.Logf("RESULT %s: %.1f ms of kernels in %d launches", arm, tot/1e3, sumInts(n))
		for _, k := range slices.SortedFunc(maps.Keys(fam), func(a, b string) int { return cmp.Compare(fam[b], fam[a]) }) {
			t.Logf("RESULT %s family %-14s %9.1f ms %5.1f%%", arm, k, fam[k]/1e3, 100*fam[k]/tot)
		}
		names := slices.SortedFunc(maps.Keys(us), func(a, b string) int { return cmp.Compare(us[b], us[a]) })
		for _, k := range names[:min(15, len(names))] {
			t.Logf("RESULT %s %9.1f ms %6d  %s", arm, us[k]/1e3, n[k], k)
		}
	}
	report("alone", ua, na)
	report("batch", ub, nb)
}

// family is a kernel's group for the split: attention, matvec/GEMM, the
// recurrence, the rotary, norms and the rest, by name.
func family(name string) string {
	l := strings.ToLower(name)
	switch {
	case strings.Contains(l, "attn") || strings.Contains(l, "flash") || strings.Contains(l, "paged") ||
		strings.Contains(l, "softmax") || strings.Contains(l, "score"):
		return "attention"
	case strings.Contains(l, "delta") || strings.Contains(l, "conv") || strings.Contains(l, "scan"):
		return "recurrence"
	case strings.Contains(l, "rope"):
		return "rotary"
	case strings.Contains(l, "gemm") || strings.Contains(l, "matvec") || strings.Contains(l, "mv") ||
		strings.Contains(l, "tile") || strings.Contains(l, "mma") || strings.Contains(l, "dp4a"):
		return "matmul"
	case strings.Contains(l, "norm"):
		return "norm"
	}
	return "other"
}

func sumInts(n map[string]int) int {
	s := 0
	for _, v := range n {
		s += v
	}
	return s
}
