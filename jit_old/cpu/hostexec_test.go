package cpu

import (
	"errors"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// requireRowMajorExec skips a gate that emits and then runs a row-major kernel
// on a CPU that cannot execute one. The emitters are host-pure (so cross-arch
// disassembly gates can exist), so emitter success is not permission to call
// the code. SupportedNative is the question nn asks, and keying on it rather
// than GOARCH keeps the gate live wherever the path is.
func requireRowMajorExec(t testing.TB, qt quant.Type) {
	t.Helper()
	if SupportedNative(qt) {
		return
	}
	t.Skipf("ROW-MAJOR %v CANNOT BE EXECUTED ON THIS HOST (no AVX-VNNI) -- this "+
		"gate covered nothing here. The emitter still produces the kernel, which "+
		"is why the disassembly gates beside this one still run; only calling it "+
		"is unsafe. The row-major family has no pre-VNNI form by decision "+
		"(native_amd64.go); prevnni.go carries the packed family's AVX2 sequence.", qt)
}

// requireVNNIHost skips a gate that executes VPDPBUSD directly.
func requireVNNIHost(t testing.TB) {
	t.Helper()
	if CPU().AVXVNNI {
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
	c, err := Map(code)
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
