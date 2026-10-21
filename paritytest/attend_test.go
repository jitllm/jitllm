package paritytest

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/compose"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// stagedShapes are the paged shapes the staged attention is held at: one
// and several rows, splits, query heads a KV head, head widths.
var stagedShapes = []mainkernels.FlashShape{
	{Rows: 1, Heads: 4, KVHeads: 2, Dim: 64, Page: 256, Splits: 1, Chunk: 64, Scale: 0.125},
	{Rows: 1, Heads: 8, KVHeads: 2, Dim: 128, Page: 256, Splits: 2, Chunk: 128, Scale: 0.088},
	{Rows: 4, Heads: 4, KVHeads: 4, Dim: 64, Page: 256, Splits: 4, Chunk: 256, Scale: 0.125},
	{Rows: 32, Heads: 2, KVHeads: 1, Dim: 96, Page: 512, Splits: 1, Chunk: 32, Scale: 0.1},
	{Rows: 2, Heads: 4, KVHeads: 2, Dim: 64, Page: 256, Splits: 2, Chunk: 64, Scale: 0.125, Softcap: 50},
	{Rows: 1, Heads: 4, KVHeads: 1, Dim: 96, MLA: 64, Page: 256, Splits: 2, Chunk: 64, Scale: 0.1},
	{Rows: 1, Heads: 8, KVHeads: 1, Dim: 576, MLA: 512, Page: 256, Splits: 1, Chunk: 64, Scale: 0.07},
}

func pagedOf(s mainkernels.FlashShape) compose.Paged {
	return compose.Paged{Rows: s.Rows, Heads: s.Heads, KVHeads: s.KVHeads, Dim: s.Dim, Page: s.Page, Splits: s.Splits, Chunk: s.Chunk, MLA: s.MLA}
}

// emitPlanned is the kernel of a planned group.
func emitPlanned(t *testing.T, p compose.Planned) []*ir.Kernel {
	t.Helper()
	k, err := p.G.Emit(p.S)
	if err != nil {
		t.Fatal(err)
	}
	return []*ir.Kernel{k}
}

// TestPagedLanesAreMains holds the softmax and scores prims over a
// subgroup, a latent row's slot an item, to main's at lanes 32: two splits a
// 64-thread group, four slots a 128-thread one, a head past a whole number
// of lanes included.
func TestPagedLanesAreMains(t *testing.T) {
	shapes := append(append([]mainkernels.FlashShape(nil), stagedShapes...),
		mainkernels.FlashShape{Rows: 1, Heads: 4, KVHeads: 1, Dim: 48, MLA: 32, Page: 256, Splits: 2, Chunk: 64, Scale: 0.1},
		mainkernels.FlashShape{Rows: 2, Heads: 8, KVHeads: 1, Dim: 48, MLA: 32, Page: 64, Splits: 1, Chunk: 128, Scale: 0.1, Softcap: 30})
	for _, s := range shapes {
		name := fmt.Sprintf("r%d_h%d_kv%d_d%d_m%d_s%d_c%d", s.Rows, s.Heads, s.KVHeads, s.Dim, s.MLA, s.Splits, s.Chunk)
		t.Run("softmax_"+name, func(t *testing.T) {
			mk, err := mainkernels.PagedStagedSoftmax(s, 32)
			if err != nil {
				t.Fatal(err)
			}
			gs, err := softmaxProgram(s).Lower()
			if err != nil {
				t.Fatal(err)
			}
			same(t, name, []*ir.Kernel{mk}, planned(t, &gs[0].Row, compose.Facts{Target: sm70, RowGroup: 64, GroupRows: true}))
		})
		if s.MLA == 0 {
			continue
		}
		t.Run("scores_"+name, func(t *testing.T) {
			mk, err := mainkernels.PagedStagedScores(s, 32)
			if err != nil {
				t.Fatal(err)
			}
			pa := pagedOf(s)
			pa.Lanes = 32
			p := &compose.Program{Rows: s.Rows, Vals: []compose.Value{{Buf: -1}, {Buf: -1}, {Elem: compose.ValU32, Buf: -1}, {Elem: compose.ValU32, Buf: -1}, {Buf: -1}}}
			p.Prims = []compose.Prim{{Kind: compose.PrimScores, In: []int{0, 1, 2, 3}, Out: []int{4}, Paged: pa, Scale: s.Scale, Softcap: s.Softcap}}
			gs, err := p.Lower()
			if err != nil {
				t.Fatal(err)
			}
			ls, err := gs[0].Launches(sm70)
			if err != nil || len(ls) != 1 {
				t.Fatalf("%d launches: %v", len(ls), err)
			}
			if want := mainkernels.PagedStagedScoreItems(s) * 32; ls[0].Groups*ls[0].Width != want {
				t.Fatalf("%d groups of %d threads, main's %d threads", ls[0].Groups, ls[0].Width, want)
			}
			k, err := ls[0].Emit()
			if err != nil {
				t.Fatal(err)
			}
			same(t, name, []*ir.Kernel{mk}, []*ir.Kernel{k})
		})
	}
}

// TestPagedMergeIsMains holds the composer's merge to main's
// FlashAttentionMergeWide, with and without sinks.
func TestPagedMergeIsMains(t *testing.T) {
	for _, s := range stagedShapes {
		for _, sink := range []bool{false, true} {
			s.Sink = sink
			name := fmt.Sprintf("r%d_h%d_d%d_s%d_sink%v", s.Rows, s.Heads, s.Dim, s.Splits, sink)
			t.Run(name, func(t *testing.T) {
				// Main merges a latent row at the value's width.
				ms, vd := s, s.Dim
				if s.MLA > 0 {
					ms.Dim, ms.MLA, vd = s.MLA, 0, s.MLA
				}
				mk, err := mainkernels.FlashAttentionMergeWide(ms)
				if err != nil {
					t.Fatal(err)
				}
				g := &compose.Row{Width: s.Rows * s.Heads * vd, Rows: 1}
				part, out, sk := g.Buf(compose.BufRow), g.Buf(compose.BufOut), -1
				if sink {
					sk = g.Buf(compose.BufRow)
				}
				g.PagedMerge(part, out, sk, pagedOf(s))
				same(t, name, []*ir.Kernel{mk}, planned(t, g, compose.Facts{Target: sm70}))
			})
		}
	}
}
