//go:build jitllmbench

// This file is a measurement instrument, not a gate, so it is opt-in by build
// tag rather than by a skip in every default run:
//
//	go test -tags jitllmbench -run <Name> ./<pkg>
//
// JITLLM_* variables select its parameters.

package model

import (
	"os"
	"testing"
	"time"

	"github.com/jitllm/jitllm/internal/testmodels"
)

func TestPrefillRegionProbe(t *testing.T) {
	p := testmodels.Resolve(os.Getenv("JITLLM_MODEL"))
	if p == "" {
		p = testmodels.Path("tinyllama-1.1b-q3_K_M.gguf")
	}
	m, err := Open(jlmOf(t, p))
	if err != nil {
		t.Skipf("cannot load: %v", err)
	}
	defer m.Close()
	s := m.NewInstance(1024)
	defer s.Close()
	ids := make([]int32, 512)
	for i := range ids {
		ids[i] = int32(1 + i%64)
	}
	if _, err := s.Prefill(ids[:8]); err != nil { // warm
		t.Fatal(err)
	}
	s2 := m.NewInstance(1024)
	defer s2.Close()
	s2.jit.ResetRegions()
	start := time.Now()
	if _, err := s2.Prefill(ids); err != nil {
		t.Fatal(err)
	}
	el := time.Since(start)
	par, ser := s2.jit.Regions()
	t.Logf("%d layers, %d prompt tokens: %d parallel regions (%d/layer, %.1f per layer per token), "+
		"%d inline, %v (%.1f tok/s)",
		m.Cfg.NLayer, len(ids), par, par/int64(m.Cfg.NLayer),
		float64(par)/float64(m.Cfg.NLayer)/float64(len(ids)), ser, el,
		float64(len(ids))/el.Seconds())
}
