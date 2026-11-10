package engine

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/model"
)

// TestUnstatedPartsAreRefused holds Open's refusal of a model the front end
// cannot state (Model.Unstated): with every part stated it names none, and
// with each block's inventory carrying a tensor its graph does not bind --
// the violation -- it names block 0 and the reason.
func TestUnstatedPartsAreRefused(t *testing.T) {
	m, err := model.Open(jlmOf(t, models()["stories260K"]))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if err := m.Unstated(); err != nil {
		t.Fatalf("a model whose every part is stated: %v", err)
	}
	// A per-layer input's gate: read only where the model has per-layer
	// inputs, which stories260K has not. A sub-norm would not do, the block
	// norms by it where the container stores one.
	undo := restate(m, func(inv map[jlm.Role]graph.Shape) {
		inv[jlm.RolePLEGate] = graph.Shape{K: m.Cfg.NEmbd, Rows: 8}
	})
	defer undo()
	err = m.Unstated()
	if err == nil || !strings.Contains(err.Error(), "block 0") ||
		!strings.Contains(err.Error(), model.UncoveredUnbound.String()) {
		t.Fatalf("a block storing a tensor its graph does not bind: %v", err)
	}
	t.Logf("refused: %v", err)
}
