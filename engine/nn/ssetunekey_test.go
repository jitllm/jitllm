//go:build amd64 && jitllmtest

package nn

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestTuneKeySeparatesTheSSETier: the tuner's cache key must differ between
// an AVX2 run and a forced-SSE run on the same box, or the SSE process reads
// pack-width and participant answers measured against AVX2 kernels it never
// runs -- hostSig fingerprints EmitNative's quantized bytes, which are AVX2
// code and host-pure, so without the tier suffix the two keys are identical.
// And the AVX2 key must NOT carry a suffix, so every key already on disk
// stays valid.
func TestTuneKeySeparatesTheSSETier(t *testing.T) {
	if cpu.HostTier() != cpu.TierAVX2 {
		t.Skipf("needs an AVX2 host to compare against (host tier %v)", cpu.HostTier())
	}
	avx := hostSig()
	if strings.Contains(avx, "tier=") {
		t.Errorf("the AVX2 key carries a tier suffix, which invalidates every cached key: %q", avx)
	}
	old := cpu.ForceTierForTest(cpu.TierSSE)
	defer cpu.ForceTierForTest(old)
	sse := hostSig()
	if sse == avx || !strings.HasSuffix(sse, ";tier=sse") {
		t.Errorf("the forced-SSE key is %q against the AVX2 key %q: the SSE tier would read the AVX2 tier's tuned answers", sse, avx)
	}
}
