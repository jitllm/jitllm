//go:build jitllmfault || jitllmtest

package tier

import (
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
)

// TestSelectionHistoryViolationIsSeen arms evictselection, the violation
// TestSelectionHistoryStaysOnTheCard guards against, and demands the pool's
// verdict flip for every selecting block while the plain block, the control,
// reads the same either way. keepsHistory is what the pool acts on at both
// sites: noEvict, and the context charged with the block.
func TestSelectionHistoryViolationIsSeen(t *testing.T) {
	plain := &nn.LayerPlan{MaxSeq: 4096}
	qsa := &nn.LayerPlan{MaxSeq: 4096, Q4E: &nn.Q4EPlan{QSARatio: 4}}
	msa := &nn.LayerPlan{MaxSeq: 4096, IdxHeads: 4, IdxBlock: 8}
	idx := &nn.LayerPlan{MaxSeq: 4096, IdxHeads: 4}
	// Two bare Configs: the violation armed on one leaves the other alone.
	cfg, armed := &Config{}, &Config{}
	armed.hook.setFault("rows", "evictselection")
	for name, c := range map[string]struct {
		p    *nn.LayerPlan
		want bool
	}{"plain": {plain, false}, "qsa": {qsa, true}, "msa": {msa, true}, "indexer": {idx, true}} {
		if got := cfg.keepsHistory(c.p); got != c.want {
			t.Fatalf("%s: keepsHistory = %v, want %v", name, got, c.want)
		}
		if got := armed.keepsHistory(c.p); got {
			t.Fatalf("%s: the violation left keepsHistory true: the gate cannot see it", name)
		}
	}
}
