package engine

import (
	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// towerForm is graph g composed on the instance's CPU execution at rows
// rows, over's formats in place of the container's for the roles it names:
// a tower gate's own part, run beside the tower's.
func (s *Instance) towerForm(g *graph.Graph, rows int, over map[jlm.Role]jlm.Type) (*cpudev.Form, error) {
	return s.exec.x.FormGraph(s.partPlan(g, nn.DeclineNoGraph, rows, over))
}
