package paritytest

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestPagedFlashIsMains holds the composer's paged flash decode to main's
// FlashDecodeKV: query-head groups, splits, rows, vector V loads, K's
// immediate addressing, a softcap and sinks.
func TestPagedFlashIsMains(t *testing.T) {
	type fc struct {
		heads, kv, dim, rows, splits int
		vec, kimm, sink                bool
		softcap                        float32
	}
	var cases []fc
	for _, hd := range [][3]int{{8, 2, 64}, {4, 4, 128}, {16, 2, 64}, {8, 1, 256}} {
		for _, sp := range []int{1, 2} {
			for _, vec := range []bool{false, true} {
				cases = append(cases, fc{heads: hd[0], kv: hd[1], dim: hd[2], rows: 1, splits: sp, vec: vec})
			}
		}
		cases = append(cases, fc{heads: hd[0], kv: hd[1], dim: hd[2], rows: 3, splits: 2, vec: true})
		cases = append(cases, fc{heads: hd[0], kv: hd[1], dim: hd[2], rows: 1, splits: 1, kimm: true, sink: true, softcap: 30})
	}
	for _, c := range cases {
		name := fmt.Sprintf("h%d_kv%d_d%d_r%d_s%d_vec%v_imm%v_sink%v_cap%v", c.heads, c.kv, c.dim, c.rows, c.splits, c.vec, c.kimm, c.sink, c.softcap)
		t.Run(name, func(t *testing.T) {
			ms := mainkernels.FlashShape{Heads: c.heads, KVHeads: c.kv, Dim: c.dim, Rows: c.rows, Splits: c.splits, Page: 64,
				Scale: 0.125, Softcap: c.softcap, Sink: c.sink, KImm: c.kimm, VecV: c.vec}
			mk, err := mainkernels.FlashDecodeKV(ms)
			if err != nil {
				t.Fatal(err)
			}
			s := compose.Flash{Heads: c.heads, KVHeads: c.kv, Dim: c.dim, Rows: c.rows, Splits: c.splits, Page: 64,
				Scale: 0.125, Softcap: c.softcap, Sink: c.sink, KImm: c.kimm, VecV: c.vec}
			if s.Width() != mainkernels.FlashKVWidth(ms) || s.Groups() != mainkernels.FlashKVGroups(ms) || s.Chunk() != mainkernels.FlashKVChunk(ms) {
				t.Fatalf("geometry: width %d/%d groups %d/%d chunk %d/%d", s.Width(), mainkernels.FlashKVWidth(ms),
					s.Groups(), mainkernels.FlashKVGroups(ms), s.Chunk(), mainkernels.FlashKVChunk(ms))
			}
			g := &compose.Row{Width: s.Width(), Rows: s.Groups()}
			q, k, v, out := g.Buf(compose.BufRow), g.Buf(compose.BufRow), g.Buf(compose.BufRow), g.Buf(compose.BufOut)
			tab, desc := g.Buf(compose.BufU32), g.Buf(compose.BufU32)
			sink := -1
			if c.sink {
				sink = g.Buf(compose.BufRow)
			}
			g.PagedFlash(s, q, k, v, tab, desc, out, sink)
			same(t, name, []*ir.Kernel{mk}, planned(t, g, compose.Facts{}))
		})
	}
}
