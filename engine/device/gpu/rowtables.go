package gpu

import (
	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
)

// bindRowTables binds the rows' positions into t where a block's graph
// reads them.
func (g *devTier) bindRowTables(t map[composer.TableKey]backend.Buf, bs *blockScratch) {
	if bs != nil && bs.rowPos != nil {
		t[tableKey(composer.TablePos)] = bs.rowPos
	}
}
