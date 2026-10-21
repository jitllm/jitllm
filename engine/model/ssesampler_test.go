//go:build amd64 && jitllmtest

package model

import (
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestSamplerOnTheForcedSSETierMatchesAVX2 runs the whole sampler -- the
// penalty scatter, the segmented ordering, the two cuts and the walk -- on both
// x86 tiers in one process and requires the same token ids. Unlike a model's
// logits, nothing in the sampler depends on lane width: the ordering selects by
// comparison and the draw is a sequential chain, so the tiers must agree
// exactly. cpu.MappedByTier is the selection check.
func TestSamplerOnTheForcedSSETierMatchesAVX2(t *testing.T) {
	if cpu.HostTier() != cpu.TierAVX2 {
		t.Skipf("host tier %v: this gate compares the two x86 tiers", cpu.HostTier())
	}
	const nv = 32000
	logits := make([]float32, nv)
	rng := rand.New(rand.NewSource(4242))
	for i := range logits {
		logits[i] = float32(rng.NormFloat64()) * 2
	}
	// Ties the ordering has to break by id, at several depths.
	for _, id := range []int32{11, 12, 900, 901, 31998, 31999} {
		logits[id] = 6
	}

	cfgs := []Sampler{
		{Temp: 1, TopK: 40, Seed: 1},
		{Temp: 0.7, TopK: 8, TopP: 0.9, Seed: 2},
		{Temp: 1.3, MinP: 0.05, Seed: 3},
		{Temp: 1, TopP: 0.95, Seed: 4},
		{Temp: 0.9, TopK: 64, TopP: 0.85, MinP: 0.01, RepeatPen: 1.4, RepeatLastN: 24, Seed: 5},
		{Temp: 1, Seed: 6},
	}
	run := func(c Sampler) []int32 {
		s := c
		out := make([]int32, 40)
		for i := range out {
			out[i] = s.Sample(logits)
			s.Observe(out[i])
		}
		return out
	}

	want := make([][]int32, len(cfgs))
	for i, c := range cfgs {
		want[i] = run(c)
	}

	old := cpu.ForceTierForTest(cpu.TierSSE)
	defer cpu.ForceTierForTest(old)
	if cpu.HostTier() != cpu.TierSSE {
		t.Fatalf("the force did not reach the probe: host tier %v", cpu.HostTier())
	}
	before := cpu.MappedByTier()
	for i, c := range cfgs {
		got := run(c)
		for j := range got {
			if got[j] != want[i][j] {
				t.Fatalf("config %d: the SSE tier drew %d at step %d where AVX2 drew %d",
					i, got[j], j, want[i][j])
			}
		}
	}
	after := cpu.MappedByTier()
	if d := after[cpu.TierAVX2] - before[cpu.TierAVX2]; d != 0 {
		t.Errorf("%d AVX2 kernel(s) were mapped under the forced SSE tier", d)
	}
	if after[cpu.TierSSE] == before[cpu.TierSSE] {
		t.Error("no SSE-tier kernel was mapped -- this arm did not run the tier it names")
	}
}
