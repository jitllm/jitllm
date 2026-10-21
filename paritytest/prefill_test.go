package paritytest

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/compose"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestPrefillIsMains holds the composer's staged prefill attention to
// main's PagedAttnScoresTiled, PagedPrefillSoftmax and PagedAttnAccTiled:
// split and direct, a run table or none, a subgroup an item or a thread.
func TestPrefillIsMains(t *testing.T) {
	type pc struct {
		heads, kv, dim, rows, splits, chunk, mla int
		mp, sink                                 bool
		softcap                                  float32
	}
	var cases []pc
	for _, hd := range [][3]int{{4, 2, 32}, {8, 8, 16}, {4, 1, 64}} {
		cases = append(cases,
			pc{heads: hd[0], kv: hd[1], dim: hd[2], rows: 32, splits: 2, chunk: 64},
			pc{heads: hd[0], kv: hd[1], dim: hd[2], rows: 32, splits: 0, chunk: 128},
			pc{heads: hd[0], kv: hd[1], dim: hd[2], rows: 8, splits: 0, chunk: 128, mp: true},
			pc{heads: hd[0], kv: hd[1], dim: hd[2], rows: 32, splits: 0, chunk: 64, sink: true, softcap: 30},
		)
	}
	cases = append(cases,
		pc{heads: 4, kv: 1, dim: 96, mla: 64, rows: 32, splits: 0, chunk: 128},
		pc{heads: 4, kv: 1, dim: 96, mla: 64, rows: 8, splits: 0, chunk: 128, mp: true},
	)
	for _, c := range cases {
		name := fmt.Sprintf("h%d_kv%d_d%d_m%d_r%d_s%d_c%d_map%v_sink%v", c.heads, c.kv, c.dim, c.mla, c.rows, c.splits, c.chunk, c.mp, c.sink)
		ms := mainkernels.FlashShape{Heads: c.heads, KVHeads: c.kv, Dim: c.dim, Rows: c.rows, Splits: c.splits, Chunk: c.chunk, Page: 64,
			Scale: 0.125, Softcap: c.softcap, Sink: c.sink, Map: c.mp, MLA: c.mla}
		ps := compose.Prefill{Heads: c.heads, KVHeads: c.kv, Dim: c.dim, Rows: c.rows, Splits: c.splits, Chunk: c.chunk, Page: 64,
			Scale: 0.125, Softcap: c.softcap, Sink: c.sink, Map: c.mp, MLA: c.mla}
		tables := func(g *compose.Row) (int, int) {
			if !c.mp {
				return -1, -1
			}
			return g.Buf(compose.BufU32), g.Buf(compose.BufU32)
		}
		W := max(1, c.splits) * c.chunk
		for _, qk := range [][2]int{{4, 4}, {2, 8}} {
			qt, kt := qk[0], qk[1]
			t.Run(fmt.Sprintf("scores_%s_q%d_k%d", name, qt, kt), func(t *testing.T) {
				mk, err := mainkernels.PagedAttnScoresTiled(ms, qt, kt)
				if err != nil {
					t.Fatal(err)
				}
				g := &compose.Row{Width: W, Rows: c.rows * c.heads, SpanGroup: c.rows}
				if c.mp {
					g.SpanGroup = qt
				}
				q, k, out, tab, desc := g.Buf(compose.BufRow), g.Buf(compose.BufRow), g.Buf(compose.BufOut), g.Buf(compose.BufU32), g.Buf(compose.BufU32)
				mp, bt := tables(g)
				g.PrefillScores(ps, q, k, out, tab, desc, mp, bt)
				same(t, name, []*ir.Kernel{mk}, planned(t, g, compose.Facts{Target: sm70, Step: compose.StepChunk, Tok: qt, RowTile: kt}))
			})
		}
		for _, lanes := range []int{32, 1} {
			for _, qt := range []int{4, 8} {
				t.Run(fmt.Sprintf("softmax_%s_l%d_q%d", name, lanes, qt), func(t *testing.T) {
					mk, err := mainkernels.PagedPrefillSoftmax(ms, lanes, qt)
					if err != nil {
						t.Fatal(err)
					}
					items := c.rows * c.heads * max(1, c.splits)
					g := &compose.Row{Width: 1, Rows: items}
					f := compose.Facts{Target: sm70}
					if lanes == 1 {
						f.Target.Lanes = 0
					}
					a, out, part, desc, sink := g.Buf(compose.BufRow), g.Buf(compose.BufOut), g.Buf(compose.BufOut), g.Buf(compose.BufU32), -1
					if c.sink && c.splits == 0 {
						sink = g.Buf(compose.BufRow)
					}
					mp, bt := tables(g)
					g.PrefillSoftmax(ps, a, out, part, desc, sink, mp, bt, qt)
					same(t, name, []*ir.Kernel{mk}, planned(t, g, f))
				})
			}
		}
		for _, qt := range []int{4, 8} {
			t.Run(fmt.Sprintf("acc_%s_q%d", name, qt), func(t *testing.T) {
				mk, err := mainkernels.PagedAttnAccTiled(ms, qt)
				if err != nil {
					t.Fatal(err)
				}
				vd := c.dim
				if c.mla > 0 {
					vd = c.mla
				}
				g := &compose.Row{Width: vd, Rows: c.rows * c.heads * max(1, c.splits), SpanGroup: c.rows}
				if c.mp {
					g.SpanGroup = qt
				}
				a, v, part, tab, desc := g.Buf(compose.BufRow), g.Buf(compose.BufRow), g.Buf(compose.BufOut), g.Buf(compose.BufU32), g.Buf(compose.BufU32)
				mp, bt := tables(g)
				g.PrefillAcc(ps, a, v, part, tab, desc, mp, bt)
				same(t, name, []*ir.Kernel{mk}, planned(t, g, compose.Facts{Step: compose.StepChunk, Tok: qt, RowTile: 1}))
			})
		}
	}
}
