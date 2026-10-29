package model

import "github.com/jitllm/jitllm/engine/nn"

// composedStructures is the record of op structures (graph.Graph.Structure)
// that passed their serving gate and ran at or above main's rate, decode and
// prompt, on each backend (docs/perf/composed-readiness.md).
var composedStructures = map[uint64]nn.Backends{}
