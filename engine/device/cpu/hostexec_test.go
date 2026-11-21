package cpu

import (
	"errors"
	"testing"

	"github.com/jitllm/jitllm/composer/x86"
)

// requireVNNI skips a gate that executes VPDPBUSD directly.
func requireVNNI(t testing.TB) {
	t.Helper()
	if Probe().AVXVNNI {
		return
	}
	t.Skip("this host has no AVX-VNNI, so VPDPBUSD cannot be executed here -- " +
		"TestExecuteVEXDotMatchesVPDPBUSD is what covers this host's sequence")
}

// mustMap is Map for a gate that is about to call the code. It skips on Map's
// own decline, the one predicate that cannot drift from what the emitters
// produce. Every other Map error is fatal, so an absent capability and a
// broken emitter do not look alike.
func mustMap(t testing.TB, code []byte) *Code {
	t.Helper()
	c, err := New().Map(code)
	if err == nil {
		return c
	}
	if errors.Is(err, ErrNoVNNI) || errors.Is(err, ErrISA) {
		// ErrISA is the same decline one tier down: an AVX2-tier kernel on an
		// SSE host, refused by name rather than executed into a SIGILL.
		t.Skipf("NOT RUNNABLE ON THIS HOST: %v -- the emitter produced the kernel "+
			"and the disassembly gates beside this one still cover it; only "+
			"calling it is unsafe here.", err)
	}
	t.Fatalf("Map: %v", err)
	return nil
}

// asmBytes is a's code, the test failed on any error emitting it met.
func asmBytes(t testing.TB, a *x86.Buf) []byte {
	t.Helper()
	b, err := a.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	return b
}
