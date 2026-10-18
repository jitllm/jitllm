package model

import "github.com/jitllm/jitllm/engine/graph"

// faults are the violations the gates run their checks against (RULE 10): a
// step done wrong on purpose, to show the gate beside it can see it. Every
// field is zero in a real run, and they live on the Model under test so two
// models in one process cannot disturb each other. The tower's are
// towerFault values on its configuration (vision.go).
type faults struct {
	// stepShare gives every row of a host step the first run's State as its
	// history: how a gate shows rows reading one session's history fail it.
	stepShare bool
	// stepPos is where a run's row j goes, given the State's next position and
	// the run's length: how a gate shows a chunk laid out at the wrong
	// positions fails it.
	stepPos func(base, j, k int) int
	// staleKey is the key a restore names its pages by, in place of the
	// prompt's (TestDecisionReusesThePrefix).
	staleKey func(key []int32) []int32
	// namePos leaves an image row's rotary position out of its name.
	namePos bool
	// tailPruneIgnoresDemand prunes although a row tap wants every row.
	tailPruneIgnoresDemand bool
	// xdKNormFirst norms k before XD-RoPE's rotary, as every other q/k-normed
	// model does: the rotation does not keep a head's RMS, so HunyuanVL's
	// post-rotary norm is load-bearing.
	xdKNormFirst bool
	// noLowering makes the build see no host lowering for an op kind
	// (TestEveryGraphOpHasALowering).
	noLowering *graph.OpKind
}
