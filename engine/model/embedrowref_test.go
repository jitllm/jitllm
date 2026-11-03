package model

import (
	"fmt"

	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/internal/oracle"
)

// embedRowRef is token's row of the embedding as the reference reads the
// container's layout (internal/oracle): the packed device layout where the
// model packs it, row-major otherwise. No scale, no position row.
func (m *Model) embedRowRef(dst []float32, token int) error {
	e := &m.embd
	if token < 0 || token >= e.rows {
		return fmt.Errorf("model: token %d is outside a vocabulary of %d", token, e.rows)
	}
	if e.packed != nil && len(e.packed.QS) != 0 {
		q, ok := layout.QuantOf(e.typ)
		if !ok {
			return fmt.Errorf("model: the embedding is packed as %v, which has no layout", e.typ)
		}
		return oracle.RowPacked(dst, q, e.packed.QS, e.packed.D, e.packed.SC, token, e.rows, e.k)
	}
	return oracle.Row(dst, e.typ, e.data, token, e.k)
}
