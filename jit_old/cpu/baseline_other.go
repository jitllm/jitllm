//go:build !amd64 && !arm64

package cpu

import "errors"

var errNoBaseline = errors.New("jit: no host emitter for this architecture")

// baseline always fails off amd64/arm64: there is no emitter, so there is
// nothing to run a model on.
func baseline() error { return errNoBaseline }

const baselineForceable = false

// mapGate refuses everything: there is no emitter here.
func mapGate([]byte) error { return Baseline() }
