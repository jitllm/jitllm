package paritytest

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// stagedShapes are the paged shapes the staged attention is held at: one
// and several rows, splits, query heads a KV head, head widths.
var stagedShapes = []mainkernels.FlashShape{
	{Rows: 1, Heads: 4, KVHeads: 2, Dim: 64, Page: 256, Splits: 1, Chunk: 64, Scale: 0.125},
	{Rows: 1, Heads: 8, KVHeads: 2, Dim: 128, Page: 256, Splits: 2, Chunk: 128, Scale: 0.088},
	{Rows: 4, Heads: 4, KVHeads: 4, Dim: 64, Page: 256, Splits: 4, Chunk: 256, Scale: 0.125},
	{Rows: 32, Heads: 2, KVHeads: 1, Dim: 96, Page: 512, Splits: 1, Chunk: 32, Scale: 0.1},
}

func pagedOf(s mainkernels.FlashShape) compose.Paged {
	return compose.Paged{Rows: s.Rows, Heads: s.Heads, KVHeads: s.KVHeads, Dim: s.Dim, Page: s.Page, Splits: s.Splits, Chunk: s.Chunk}
}

// emitAt is g's kernel under s.
func emitAt(t *testing.T, g *compose.Row, s compose.Schedule) []*ir.Kernel {
	t.Helper()
	k, err := g.Emit(s)
	if err != nil {
		t.Fatal(err)
	}
	return []*ir.Kernel{k}
}

// TestPagedSoftmaxIsMains holds the composer's staged softmax, a thread an
// item, to main's PagedStagedSoftmax at lanes 1.
func TestPagedSoftmaxIsMains(t *testing.T) {
	for _, s := range stagedShapes {
		name := fmt.Sprintf("r%d_h%d_kv%d_d%d_s%d_c%d", s.Rows, s.Heads, s.KVHeads, s.Dim, s.Splits, s.Chunk)
		t.Run(name, func(t *testing.T) {
			mk, err := mainkernels.PagedStagedSoftmax(s, 1)
			if err != nil {
				t.Fatal(err)
			}
			g := &compose.Row{Width: s.Rows * s.Heads * s.Splits, Rows: 1}
			scores := g.Buf(compose.BufRow)
			g.Buf(compose.BufU32)
			out, part := g.Buf(compose.BufOut), g.Buf(compose.BufOut)
			g.Buf(compose.BufU32)
			desc := g.Buf(compose.BufU32)
			g.PagedSoftmax(scores, out, part, desc, pagedOf(s))
			same(t, name, []*ir.Kernel{mk}, emitAt(t, g, compose.Schedule{Group: 64, Flat: true, Target: sm70}))
		})
	}
}
