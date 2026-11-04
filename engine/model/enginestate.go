package model

import (
	"sync"
	"sync/atomic"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/engine/nn"
)

// engineState is what the engine holds for a model's instances, apart from
// the model's own weights, config and front: the instances it serves, their
// history pools and step buffers, the CPU device's forms, the devices
// holding the model's weights, and the counters and faults of their steps.
// It moves to the engine with the instance (docs/design/model-and-engine.md).
type engineState struct {
	// staleKeyFault is the decision prefix gate's violation
	// (TestDecisionReusesThePrefix): the key a restore names its pages by, in
	// place of the prompt's. Nil in every real run.
	staleKeyFault func(key []int32) []int32
	// stepShareFault, when set, gives every row of a host step the first run's
	// instance as its history. False but in a gate's violation: it is how the
	// gate shows that rows reading one session's history fail it.
	stepShareFault bool
	// stepPosFault, when set, is where a run's row j goes, given the
	// instance's next position and the run's length. Nil but in a gate's
	// violation: it is how the gate shows that a chunk laid out at the wrong
	// positions fails it.
	stepPosFault func(base, j, k int) int
	// fusedRuns counts the model's fused runs, and fuseTries the runs asked
	// for, opened or not: a gate whose subtests all skipped (no model on a
	// model-free runner) asked for none, and has nothing to say about fusion
	// either way.
	fusedRuns, fuseTries atomic.Int64
	// opNanos are the per-op nanosecond counters WithProfile arms (OpProfile).
	opNanos [nOps]atomic.Int64
	// cacheSeq numbers the default cache names of the model's sessions.
	cacheSeq atomic.Int64
	// insts is the model's pool: the instances it serves (Serve).
	insts Pool
	// cpuUse[li] counts the live instances that run block li on the CPU, under
	// bind. An instance placing a block gives its host page back only when the
	// count is zero: placement is per instance and the binding per Model, so
	// another instance running the block on the host would otherwise find it
	// unbound between its pageIn and its matvecs.
	cpuUse []int32
	// spare is a closed session's prompt buffers, for the next instance's first
	// prompt (promptBuf).
	spareMu sync.Mutex
	spare   *promptBuf
	// step is the host step scratch across sessions (stepBuf).
	step *stepBuf
	// kvPool is the host KV pages closed instances gave back (kvPagePool).
	kvPool kvPagePool
	// kvArenas is every instance's composed CPU history, a layer and page size
	// each (kvArenaFor).
	kvArenas cpudev.Arenas
	// cpuForms is the composed CPU blocks' forms (cpuserve.go), and
	// cpuComposes the times the composer was asked for one (CPUComposes).
	cpuForms    cpudev.Forms
	cpuComposes atomic.Int64
	// hostStepRows is the rows StepRuns has run as host steps across
	// sessions (stepHost); HostStepRows reports it.
	hostStepRows int64
	// pool is what WithPool collected, handed to every instance's CPU device.
	pool []cpu.PoolOption
	// live is the open instances, which a prefill asks whether another decodes.
	live live
	// forgetters are the devices an instance attached that copy weights by
	// address: the pager tells them when a frame's bytes change (forgetCopies).
	// owners are the devices holding this model's blocks, released at Close.
	// Both guarded by devMu, since a page-in on any session reads forgetters.
	devMu      sync.Mutex
	forgetters []nn.CopyForgetter
	owners     []nn.ModelOwner
}
