package model

import (
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
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
// block composed for it, teacher-forced over 24 steps against the CPU's
// float reference within llama.cpp's spread from it (TestComposedCPUBlocksServe's
// band): its blocks past the dense lead select two of up to six blocks of
// four positions per kv group (graph.OpSelect), and each piece of the
// selection taken out of the composed arm leaves the band. DeepSeek V3.2's indexer
// runs in latent blocks, which the CPU device declines by name (a latent
// attention's history); TestDeepseek32OnEveryDevice serves it composed.
func TestComposedCPUSelectsKeys(t *testing.T) {
	const steps, page = 24, 32
	path := jlmOf(t, testmodels.Path("synth-minimaxm3.gguf"))
	m, err := Open(path, noTune)
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
	toks := cpuTokens(t, m, steps)
	want, band := cpuReference(t, path, fixtureSource(testmodels.Path("synth-minimaxm3.gguf")), page, toks)
	med, worst, widest := cpuBandTo(t, want, cpuLogits(t, m, toks, cpuFaultNone, true), band)
	t.Logf("over %d steps the composed CPU is %.3f of the band at the median, worst NMSE %.3g where the band's widest is %.3g", steps, med, worst, widest)
	if !(med <= bandRatio) || !(worst <= bandWorst*widest) {
		t.Fatalf("the composed CPU leaves the band: median ratio %.3f (bound %.1f), worst %.3g (bound %.1f x %.3g)", med, bandRatio, worst, bandWorst, widest)
	}
	for _, v := range []c6Violation{
		{"attention over every block", m3Violations[0].mut},
		{"a top-k one short", m3DevViolations[1].mut},
		{"two blocks forced in", m3DevViolations[2].mut},
		{"blocks of eight", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.IdxBlock = 8 }) }},
	} {
		undo := v.mut(m)
		bm, bw, _ := cpuBandTo(t, want, cpuLogits(t, m, toks, cpuFaultNone, true), band)
		undo()
		t.Logf("violation %q: median ratio %.3f, worst %.3g", v.name, bm, bw)
		if bm <= bandRatio && bw <= bandWorst*widest {
			t.Errorf("%s stays in the band (median ratio %.3f, worst %.3g): the gate cannot see it", v.name, bm, bw)
		}
	}
}
