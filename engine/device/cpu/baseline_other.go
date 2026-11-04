//go:build !amd64 && !arm64

package cpu

const errNoBaseline jitError = "jit: no host emitter for this architecture"

// baselineOf always fails off amd64/arm64: there is no emitter, so there is
// nothing to run a model on.
func baselineOf(Features) error { return errNoBaseline }

// gate refuses everything: there is no emitter here.
func (h *Host) gate([]byte) error { return h.baseline }
