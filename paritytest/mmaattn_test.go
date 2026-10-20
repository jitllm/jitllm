package paritytest

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

var sm80 = compose.Target{API: compose.APIPTX, ISA: ir.ISA{Family: ir.FamilyPTX, Version: 80}, Lanes: 32}

// TestPrefillScoresMMAIsMains holds the composer's staged prefill scores on
// the binary16 matrix instruction to main's PagedAttnScoresMMA.
func TestPrefillScoresMMAIsMains(t *testing.T) {
	type fc struct{ heads, kv, dim, rows, chunk, splits, nt int }
	var cases []fc
	for _, hd := range [][3]int{{8, 2, 64}, {4, 4, 128}, {4, 1, 32}} {
		for _, nt := range []int{1, 2, 4} {
			cases = append(cases, fc{hd[0], hd[1], hd[2], 32, 64, 1, nt})
		}
		cases = append(cases, fc{hd[0], hd[1], hd[2], 64, 128, 2, 2})
	}
	for _, c := range cases {
		name := fmt.Sprintf("h%d_kv%d_d%d_r%d_c%d_s%d_nt%d", c.heads, c.kv, c.dim, c.rows, c.chunk, c.splits, c.nt)
		t.Run(name, func(t *testing.T) {
			ms := mainkernels.FlashShape{Heads: c.heads, KVHeads: c.kv, Dim: c.dim, Rows: c.rows, Splits: c.splits, Chunk: c.chunk,
				Page: 64, Scale: 0.125}
			mk, err := mainkernels.PagedAttnScoresMMA(ms, c.nt)
			if err != nil {
				t.Fatal(err)
			}
			s := compose.Prefill{Heads: c.heads, KVHeads: c.kv, Dim: c.dim, Rows: c.rows, Splits: c.splits, Chunk: c.chunk, Page: 64, Scale: 0.125}
			if s.ScoresMMAWarps(c.nt) != mainkernels.PagedAttnScoresMMAWarps(ms, c.nt) {
				t.Fatalf("warps %d/%d", s.ScoresMMAWarps(c.nt), mainkernels.PagedAttnScoresMMAWarps(ms, c.nt))
			}
			g := &compose.Row{Width: 32 * s.ScoresMMAWarps(c.nt), Rows: 1}
			q, k, n, out := g.Buf(compose.BufRow), g.Buf(compose.BufRow), g.Buf(compose.BufU32), g.Buf(compose.BufOut)
			tab, desc := g.Buf(compose.BufU32), g.Buf(compose.BufU32)
			g.PrefillScoresMMA(s, c.nt, q, k, n, out, tab, desc)
			same(t, name, []*ir.Kernel{mk}, planned(t, g, compose.Facts{Target: sm80}))
		})
	}
}

// TestPrefillAccMMAIsMains holds the composer's staged prefill accumulate on
// the binary16 matrix instruction to main's PagedAttnAccMMA.
func TestPrefillAccMMAIsMains(t *testing.T) {
	type fc struct{ heads, kv, dim, rows, chunk, splits, mt, nt int }
	var cases []fc
	for _, hd := range [][3]int{{8, 2, 64}, {4, 4, 128}, {4, 1, 32}} {
		for _, tl := range [][2]int{{1, 1}, {2, 2}, {1, 2}} {
			if hd[2]%(16*tl[0]) == 0 {
				cases = append(cases, fc{hd[0], hd[1], hd[2], 32, 64, 0, tl[0], tl[1]})
				cases = append(cases, fc{hd[0], hd[1], hd[2], 32, 64, 2, tl[0], tl[1]})
			}
		}
	}
	for _, c := range cases {
		name := fmt.Sprintf("h%d_kv%d_d%d_r%d_c%d_s%d_m%dn%d", c.heads, c.kv, c.dim, c.rows, c.chunk, c.splits, c.mt, c.nt)
		t.Run(name, func(t *testing.T) {
			ms := mainkernels.FlashShape{Heads: c.heads, KVHeads: c.kv, Dim: c.dim, Rows: c.rows, Splits: c.splits, Chunk: c.chunk,
				Page: 64, Scale: 0.125}
			mk, err := mainkernels.PagedAttnAccMMA(ms, c.mt, c.nt)
			if err != nil {
				t.Fatal(err)
			}
			s := compose.Prefill{Heads: c.heads, KVHeads: c.kv, Dim: c.dim, Rows: c.rows, Splits: c.splits, Chunk: c.chunk, Page: 64, Scale: 0.125}
			if s.AccMMAWarps(c.mt, c.nt) != mainkernels.PagedAttnAccMMAWarps(ms, c.mt, c.nt) {
				t.Fatalf("warps %d/%d", s.AccMMAWarps(c.mt, c.nt), mainkernels.PagedAttnAccMMAWarps(ms, c.mt, c.nt))
			}
			g := &compose.Row{Width: 32 * s.AccMMAWarps(c.mt, c.nt), Rows: 1}
			a, v, n, out := g.Buf(compose.BufRow), g.Buf(compose.BufRow), g.Buf(compose.BufU32), g.Buf(compose.BufOut)
			tab, desc := g.Buf(compose.BufU32), g.Buf(compose.BufU32)
			g.PrefillAccMMA(s, c.mt, c.nt, a, v, n, out, tab, desc)
			same(t, name, []*ir.Kernel{mk}, planned(t, g, compose.Facts{Target: sm80}))
		})
	}
}
