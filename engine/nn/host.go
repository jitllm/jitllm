package nn

import "github.com/jitllm/jitllm/jit_old/cpu"

// HostRefusal is why this host cannot run the engine, or nil: cpu.Baseline,
// asked through the package model.Open already depends on.
//
// It is asked once, at Open, so a host that cannot run a token is refused with
// a sentence naming every missing extension or pending SSE-tier op, rather than
// a panic naming whichever kernel happened to be built first.
func HostRefusal() error { return cpu.Baseline() }
