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
	if bs != nil && bs.req != nil {
		t[tableKey(composer.TableRequest)] = bs.req
	}
	// A block may read its rows' tokens (a mixture routed by them).
	if bs != nil && g.embedBufs(bs) {
		t[tableKey(composer.TableTokens)] = bs.embTok
	}
}
