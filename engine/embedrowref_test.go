package engine

import (
	"fmt"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/internal/oracle"
)

// embedRowRef is token's row of the embedding as the reference reads the
// container's layout (internal/oracle): the packed device layout where the
// model packs it, row-major otherwise. No scale, no position row.
func embedRowRef(m *model.Model, dst []float32, token int) error {
	e := &m.Embd
	if token < 0 || token >= e.Rows {
		return fmt.Errorf("model: token %d is outside a vocabulary of %d", token, e.Rows)
	}
	if e.Packed != nil && len(e.Packed.QS) != 0 {
		q, ok := layout.QuantOf(e.Typ)
		if !ok {
			return fmt.Errorf("model: the embedding is packed as %v, which has no layout", e.Typ)
		}
		return oracle.RowPacked(dst, q, e.Packed.QS, e.Packed.D, e.Packed.SC, token, e.Rows, e.K)
	}
	return oracle.Row(dst, e.Typ, e.Data, token, e.K)
}
