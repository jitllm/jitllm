package model

import (
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
)

// composedStructuresOf is the op structures (graph.Graph.Structure) of every
// graph the front end states for the model, and false where it states none
// for a block, the head, the input rows or a tower.
func (m *Model) composedStructuresOf() (map[uint64]bool, bool) {
	f := m.frontGraphs()
	if len(f.blocks) == 0 {
		return nil, false
	}
	for _, why := range [][]Uncovered{f.why, f.towerWhy, f.headsWhy, {f.headWhy, f.embedWhy, f.poolWhy, f.towerEntryWhy, f.towerProjWhy}} {
		for _, u := range why {
			if u != Covered {
				return nil, false
			}
		}
	}
	graphs := append(append(append([]*graph.Graph{f.head, f.embed, f.pool}, f.blocks...), f.tower...), f.heads...)
	set := map[uint64]bool{}
	for _, g := range graphs {
		if g != nil {
			set[g.Structure()] = true
		}
	}
	return set, true
}

// composedReady is the backends where the composer serves this model by
// default: those on which the front end states everything as ops and every op
// structure the model's graphs use is one measured at or above main's rate
// there (composedStructures). It reads the container's own graphs, so no
// model is named.
func (m *Model) composedReady() nn.Backends {
	set, ok := m.composedStructuresOf()
	if !ok {
		return 0
	}
	record := composedStructures
	if m.opt.structures != nil {
		record = m.opt.structures
	}
	ready := nn.BackendCPU | nn.BackendPTX | nn.BackendSPIRV
	for s := range set {
		ready &= record[s]
	}
	return ready
}
