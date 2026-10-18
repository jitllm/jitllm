//go:build jitllmfault && (amd64 || arm64)

package model

import (
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestDeepDecodeArmFailsOnAViolation runs the decode-deep window of
// TestDecodeDoesNotAllocate on Qwen3.8-Flash-Next's fixture twice: clean,
// where it must read no engine allocation, and under the decodealloc fault
// (every device call allocates a buffer), where it must read some. An arm that
// reads zero under the violation cannot fail, and its green line proves
// nothing (RULE 10).
func TestDeepDecodeArmFailsOnAViolation(t *testing.T) {
	m := openSynth(t, "synth-qwen4exp")
	defer m.Close()
	ran := 0
	for _, spec := range stepDevices() {
		g, err := openTier(t, append([]tier.Option{tier.WithDevices(spec)}, testTierOpts(t)...)...)
		if err != nil || g == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer g.Close()
			// The first window warms the kernels and the pool, as the real arm's
			// decode arm before it does.
			deepDecodeWindow(t, m, g)
			clean, _, n, _, _ := deepDecodeWindow(t, m, g)
			g.SetFault("rows", "decodealloc")
			bad, _, _, _, _ := deepDecodeWindow(t, m, g)
			g.SetFault("rows", "")
			t.Logf("%s: %d engine allocation(s) over %d clean, %d under the violation", spec, clean.engine, n, bad.engine)
			if clean.engine != 0 {
				t.Errorf("the clean window reads %d engine allocation(s)", clean.engine)
			}
			if bad.engine < int64(n) {
				t.Errorf("the violation (an allocation per device call) reads %d over %d tokens: the arm cannot see it", bad.engine, n)
			}
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}
