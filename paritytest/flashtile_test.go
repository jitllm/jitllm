package paritytest

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/compose"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestPagedFlashTileIsMains holds the composer's collective-tile flash
// prefill to main's FlashPrefillTile over paged pools: head widths, query
// groups, a run table, splits, a softcap and a window.
func TestPagedFlashTileIsMains(t *testing.T) {
	type fc struct {
		heads, kv, dim, rows, splits, window int
		mp                                   bool
		softcap                              float32
	}
	var cases []fc
	for _, hd := range [][3]int{{8, 2, 64}, {4, 4, 128}, {16, 2, 64}, {4, 1, 32}} {
		cases = append(cases, fc{heads: hd[0], kv: hd[1], dim: hd[2], rows: 32})
		cases = append(cases, fc{heads: hd[0], kv: hd[1], dim: hd[2], rows: 32, mp: true, splits: 2})
	}
	cases = append(cases, fc{heads: 8, kv: 2, dim: 64, rows: 64, softcap: 30})
	for _, c := range cases {
		name := fmt.Sprintf("h%d_kv%d_d%d_r%d_s%d_map%v_cap%v_w%d", c.heads, c.kv, c.dim, c.rows, c.splits, c.mp, c.softcap, c.window)
		t.Run(name, func(t *testing.T) {
			tl, ok := mainkernels.FlashTileFor(c.dim)
			if !ok {
				t.Skip("no blocking")
			}
			fs := mainkernels.FlashPrefill70Shape{Heads: c.heads, KVHeads: c.kv, Dim: c.dim, Rows: c.rows, Scale: 0.125,
				Softcap: c.softcap, Window: c.window, Page: 64, Splits: c.splits, Map: c.mp}
			mk, err := mainkernels.FlashPrefillTile(fs, tl)
			if err != nil {
				t.Fatal(err)
			}
			s := compose.FlashTile{Heads: c.heads, KVHeads: c.kv, Dim: c.dim, Rows: c.rows, Page: 64, Scale: 0.125,
				Softcap: c.softcap, Window: c.window, Splits: c.splits, Map: c.mp, SG: tl.SG, TK: tl.TK}
			if s.Groups() != mainkernels.FlashPrefillTileGroups(fs, tl) || s.Threads() != tl.Threads() {
				t.Fatalf("geometry: groups %d/%d threads %d/%d", s.Groups(), mainkernels.FlashPrefillTileGroups(fs, tl), s.Threads(), tl.Threads())
			}
			g := &compose.Row{Width: s.Threads(), Rows: s.Groups()}
			q, k, v, n, out := g.Buf(compose.BufRow), g.Buf(compose.BufRow), g.Buf(compose.BufRow), g.Buf(compose.BufU32), g.Buf(compose.BufOut)
			tab, desc := g.Buf(compose.BufU32), g.Buf(compose.BufU32)
			mp := -1
			if c.mp {
				mp = g.Buf(compose.BufU32)
			}
			g.PagedFlashTile(s, q, k, v, n, out, tab, desc, mp)
			same(t, name, []*ir.Kernel{mk}, planned(t, g, compose.Facts{Target: compose.Target{API: compose.APIPTX,
				ISA: ir.ISA{Family: ir.FamilyPTX, Version: 80}, Lanes: 32}}))
		})
	}
}

// TestPagedFlash16IsMains holds the composer's binary16 m16n8 flash prefill
// to main's FlashPrefill80 (and its m16n8k8 twin): head widths past 128 too,
// a run table, splits and a softcap.
func TestPagedFlash16IsMains(t *testing.T) {
	type fc struct {
		heads, kv, dim, rows, splits int
		mp, k8                       bool
		softcap                      float32
	}
	var cases []fc
	for _, hd := range [][3]int{{8, 2, 64}, {4, 4, 128}, {4, 1, 32}, {4, 2, 256}} {
		cases = append(cases, fc{heads: hd[0], kv: hd[1], dim: hd[2], rows: 64})
		cases = append(cases, fc{heads: hd[0], kv: hd[1], dim: hd[2], rows: 64, mp: true, splits: 2})
		cases = append(cases, fc{heads: hd[0], kv: hd[1], dim: hd[2], rows: 64, k8: true, softcap: 30})
	}
	for _, c := range cases {
		name := fmt.Sprintf("h%d_kv%d_d%d_r%d_s%d_map%v_k8%v_cap%v", c.heads, c.kv, c.dim, c.rows, c.splits, c.mp, c.k8, c.softcap)
		t.Run(name, func(t *testing.T) {
			fs := mainkernels.FlashPrefill70Shape{Heads: c.heads, KVHeads: c.kv, Dim: c.dim, Rows: c.rows, Scale: 0.125,
				Softcap: c.softcap, Page: 64, Splits: c.splits, Map: c.mp}
			build := mainkernels.FlashPrefill80
			if c.k8 {
				build = mainkernels.FlashPrefill80K8
			}
			mk, err := build(fs)
			if err != nil {
				t.Fatal(err)
			}
			s := compose.Flash16{Heads: c.heads, KVHeads: c.kv, Dim: c.dim, Rows: c.rows, Page: 64, Scale: 0.125,
				Softcap: c.softcap, Splits: c.splits, Map: c.mp, K8: c.k8}
			if s.Groups() != mainkernels.FlashPrefill80Groups(fs) || s.Threads() != mainkernels.FlashPrefill80Threads {
				t.Fatalf("geometry: groups %d/%d", s.Groups(), mainkernels.FlashPrefill80Groups(fs))
			}
			g := &compose.Row{Width: s.Threads(), Rows: s.Groups()}
			q, k, v, n, out := g.Buf(compose.BufRow), g.Buf(compose.BufRow), g.Buf(compose.BufRow), g.Buf(compose.BufU32), g.Buf(compose.BufOut)
			tab, desc := g.Buf(compose.BufU32), g.Buf(compose.BufU32)
			mp := -1
			if c.mp {
				mp = g.Buf(compose.BufU32)
			}
			g.PagedFlash16(s, q, k, v, n, out, tab, desc, mp)
			same(t, name, []*ir.Kernel{mk}, planned(t, g, compose.Facts{Target: sm80}))
		})
	}
}
