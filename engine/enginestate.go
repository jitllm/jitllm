package engine

import (
	"sync"
	"sync/atomic"

	"github.com/jitllm/jitllm/engine/device/cpudev"
)

// hostShared is the scratch every instance of a model on one CPU device
// hands the next: a closed session's prompt buffers, for the next instance's
// first prompt (promptBuf), and the host step scratch across sessions
// (stepBuf). The CPU device keeps it for the model (cpudev.CPU.Attached).
type hostShared struct {
	mu    sync.Mutex
	spare *promptBuf
	step  *stepBuf
}

// fuseCount counts fused runs: runs those opened, tries those asked for,
// opened or not. A gate whose subtests all skipped (no model on a model-free
// runner) asked for none, and has nothing to say about fusion either way.
type fuseCount struct {
	runs, tries atomic.Int64
}

// engineState is what an instance shares with the instances it makes itself
// (its tower run, its drafts): the CPU device they run on, their step
// buffers, and the counters and faults of their steps.
type engineState struct {
	// fuse counts the instances' fused runs (fuseCount).
	fuse *fuseCount
	// opNanos are the per-op nanosecond counters WithProfile arms (OpProfile).
	opNanos [nOps]atomic.Int64
	// host is the scratch the model's instances on the CPU device hand each
	// other (hostShared).
	host *hostShared
	// cpu is the CPU device the instances run on: its target, its crews, the
	// model's forms, arenas and history pool on it, and the instances' seats.
	// ownCPU is that the instance that made it closes it (no WithCPU).
	cpu    *cpudev.CPU
	ownCPU bool
	// hostStepRows is the rows StepRuns has run as host steps across
	// sessions (stepHost); HostStepRows reports it.
	hostStepRows int64
}
