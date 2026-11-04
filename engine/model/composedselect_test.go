package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// selectingBlocks is the blocks of m whose graph selects the positions its
// attention reads (graph.OpSelect).
func selectingBlocks(m *Model) int {
	n := 0
	for li := range m.Cfg.NLayer {
		g := m.Composing().BlockGraph(li)
		if g == nil {
			continue
		}
		for _, o := range g.Ops {
			if o.Kind == graph.OpSelect {
				n++
				break
			}
		}
	}
	return n
}

// TestComposedCPUSelectsKeys runs synth-minimaxm3 on the CPU device, every
// block composed for it, teacher-forced over 24 steps against the CPU's float
// forms over the same history (withCPUFloat): its blocks past the dense lead
// select two of up to six blocks of four positions per kv group
// (graph.OpSelect), within 1e-5 of the float forms' logits at every step
// (float weights: the two part by their reduction order alone; llama.cpp's
// spread is wider than a violation here), and each piece of the selection
// taken out of the composed arm alone past 100 times its worst and past 1e-4.
// DeepSeek V3.2's indexer
// runs in latent blocks, which the CPU device declines by name (a latent
// attention's history); TestDeepseek32OnEveryDevice serves it composed.
func TestComposedCPUSelectsKeys(t *testing.T) {
	const steps, page, bound = 24, 32, 1e-5
	path := jlmOf(t, testmodels.Path("synth-minimaxm3.gguf"))
	ref, err := Open(path, withCPUFloat())
	if err != nil {
		t.Fatal(err)
	}
	defer ref.Close()
	ref.setKVPageForTest(page)
	m, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	m.setKVPageForTest(page)
	if c := coveredBlocks(m); c != m.Cfg.NLayer {
		t.Fatalf("the front end states %d of %d blocks: %s", c, m.Cfg.NLayer, m.Uncovered(m.Cfg.NDenseLead))
	}
	if n, want := selectingBlocks(m), m.Cfg.NLayer-m.Cfg.NDenseLead; n != want || n == 0 {
		t.Fatalf("%d blocks select their keys, want %d", n, want)
	}
	if blocks := (steps + m.Cfg.IdxBlock - 1) / m.Cfg.IdxBlock; blocks <= m.Cfg.IdxTopK {
		t.Fatalf("%d steps are %d blocks against a top-%d: the selection keeps everything", steps, blocks, m.Cfg.IdxTopK)
	}
	toks := cpuTokens(t, ref, steps)
	want := cpuLogits(t, ref, toks, cpuFaultNone, true)
	worstOf := func(got [][]float32) float64 {
		w := 0.0
		for i := range want {
			e := logitNMSE(got[i], want[i])
			if math.IsNaN(e) {
				return math.Inf(1)
			}
			w = max(w, e)
		}
		return w
	}
	worst := worstOf(cpuLogits(t, m, toks, cpuFaultNone, true))
	t.Logf("the composed CPU's worst NMSE from the float forms over %d steps: %.3g", steps, worst)
	if !(worst < bound) {
		t.Fatalf("the composed CPU is %.3g from the float forms, bound %g", worst, bound)
	}
	for _, v := range []c6Violation{
		{"attention over every block", m3Violations()[0].mut},
		{"a top-k one short", m3DevViolations()[1].mut},
		{"two blocks forced in", m3DevViolations()[2].mut},
		{"blocks of eight", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.IdxBlock = 8 }) }},
	} {
		undo := v.mut(m)
		bad := worstOf(cpuLogits(t, m, toks, cpuFaultNone, true))
		undo()
		t.Logf("violation %q: worst NMSE %.3g", v.name, bad)
		if !(bad > 100*worst) || !(bad > 1e-4) {
			t.Errorf("%s: the composed CPU cannot see it -- worst NMSE %.3g against a clean %.3g", v.name, bad, worst)
		}
	}
}
