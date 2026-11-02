//go:build amd64 && jitllmtest

package cpu

import (
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// TestSSEPackedSignedViolationIsCaught runs the SSE Q8_0 kernels against the
// mistake a mechanical port makes -- keep the VNNI kernel's +128 centring and
// swap in PMADDUBSW -- armed through the same switch as the AVX2 tier's
// (SetNaiveSignedQ8ForTest), and shows the equality gate failing. The
// violation is wrong only where a pair saturates int16 (2*255*127), so a bit
// comparison against a kernel that cannot saturate sees it at any clipping
// rate.
func TestSSEPackedSignedViolationIsCaught(t *testing.T) {
	const g = quant.Q8_0
	tile, tail, fused := ssePackedKernels(t, g)
	p := newSSEPackFix(t, g, ssePackRows, ssePackK, 1, true)

	restore := SetNaiveSignedQ8ForTest(true)
	nTile, nTail, nFused := ssePackedKernels(t, g)
	SetNaiveSignedQ8ForTest(restore)

	cT, cF := runAllSSE(t, p, tile, tail, fused)
	nT, nF := runAllSSE(t, p, nTile, nTail, nFused)
	for _, c := range []struct {
		name         string
		clean, naive []float32
	}{{"tile+tail", cT, nT}, {"fused", cF, nF}} {
		differ := 0
		for r := range c.clean {
			if c.clean[r] != c.naive[r] {
				differ++
			}
		}
		if differ == 0 {
			t.Fatalf("%s: the armed naive sequence changed nothing -- the hook does not reach "+
				"the SSE kernel, and this gate would pass a saturating port", c.name)
		}
		t.Logf("%s: the naive sequence moves %d of %d rows; NMSE %.2e against the oracle",
			c.name, differ, len(c.clean), nmseOf(t, c.naive, p.oracle(t)))
	}

	// On a VNNI host the AVX2 arm is VPDPBUSD, which the hook does not touch, so
	// the shipping equality gate itself must report the violation -- and pass
	// the clean kernels.
	requireAVX2Twin(t)
	if msg := firstExactMismatch(t, p, tile, tail, fused); msg != "" {
		t.Fatalf("the clean kernels already disagree: %s", msg)
	}
	if HostDotKind() != DotVNNI {
		t.Log("no AVX-VNNI here: the AVX2 arm would carry the same armed hook, so the " +
			"SSE-against-SSE comparison above is the proof on this host")
		return
	}
	msg := firstExactMismatch(t, p, nTile, nTail, nFused)
	if msg == "" {
		t.Fatal("TestSSEPackedMatchesAVX2BitForBit's comparison did NOT fire on the saturating kernel")
	}
	t.Logf("the equality gate fires: %s", msg)
}
