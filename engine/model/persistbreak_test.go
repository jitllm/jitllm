//go:build jitllmtest

package model

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/ptx"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestPersistentDecodeSeesABrokenBarrier is TestPersistentDecodeMatchesPerOp's
// violation (RULE 10): with every grid barrier letting its waiters through
// (ptx.BreakGridBarrierForTest), a persistent decode reads rows before they
// are written, and its logits must leave the per-op arm's. If they do not,
// the bit-identity the gate asserts says nothing about the barrier.
func TestPersistentDecodeSeesABrokenBarrier(t *testing.T) {
	persistBroken(t, false, func() func() {
		old := ptx.BreakGridBarrierForTest(true)
		return func() { ptx.BreakGridBarrierForTest(old) }
	})
}

// TestDepDecodeSeesABrokenWait is the same violation for the dependency
// counters (Config.PersistDeps): with every wait returning at once
// (ptx.BreakDepWaitForTest) the decode must leave the per-op logits, or the
// dep arm's bit-identity in TestPersistentDecodeMatchesPerOp says nothing
// about the waits.
func TestDepDecodeSeesABrokenWait(t *testing.T) {
	persistBroken(t, true, func() func() {
		old := ptx.BreakDepWaitForTest(true)
		return func() { ptx.BreakDepWaitForTest(old) }
	})
}

// persistBroken decodes per-op and then persistent with break applied, and
// requires the logits to differ.
func persistBroken(t *testing.T, deps bool, brk func() func()) {
	name := "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
	p := testmodels.Path(name)
	if _, err := os.Stat(p); err != nil {
		if _, err := os.Stat(strings.TrimSuffix(p, ".gguf") + ".jlm"); err != nil {
			testmodels.Missing(t, "MODEL MISSING: %s", name)
		}
	}
	m, err := Open(jlmOf(t, p), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode("The river rose every spring until the bridges were islands, and", true)
	const steps = 8
	run := func(persist bool) ([][]float32, tier.Stats) {
		mode := tier.WithPersist(persist, 0)
		if deps {
			mode = tier.WithPersistDeps(persist, 0)
		}
		g, err := openTier(t, tier.WithDeviceTune(tier.TuneOff), mode,
			tier.WithConfig(func(c *tier.Config) { c.NoGraph = true }))
		if err != nil || g == nil {
			t.Skipf("no GPU: %v", err)
		}
		defer g.Close()
		s := m.NewState(len(ids) + steps + 4)
		defer s.Close()
		if err := s.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		l, err := s.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		var out [][]float32
		tok := int32(1)
		for i := 0; i < steps; i++ {
			out = append(out, append([]float32(nil), l...))
			if l, err = s.Forward(tok); err != nil {
				t.Fatal(err)
			}
		}
		s.SetGPULayers(0)
		return out, g.Stats()
	}
	ref, _ := run(false)
	defer brk()()
	got, st := run(true)
	if deps && st.PersistDepLaunches == 0 {
		t.Fatalf("no dependency-counter kernel ran (%s): the violation reached nothing", st.PersistWhy)
	}
	if st.PersistLaunches == 0 {
		t.Fatalf("no persistent kernel ran (%s): the violation reached nothing", st.PersistWhy)
	}
	diff := 0
	for i := range ref {
		for j := range ref[i] {
			if math.Float32bits(ref[i][j]) != math.Float32bits(got[i][j]) {
				diff++
			}
		}
	}
	if diff == 0 {
		t.Fatalf("%d persistent launches whose waits wait for nothing gave the per-op logits bit for bit", st.PersistLaunches)
	}
	t.Logf("seen: %d of %d logits differ with the barrier broken", diff, steps*len(ref[0]))
}
