package paritytest

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

var metalM4 = compose.Target{API: compose.APIMSL, ISA: ir.ISA{Family: ir.FamilyMSL, Version: 30}, Lanes: 32}

// TestGemmTileIsMains holds the composer's GEMM on the collective binary16
// tiles to main's GemmTile, format by format: the blockings the tier takes,
// the 16-row tiles, a grouped mixture and a bias.
func TestGemmTileIsMains(t *testing.T) {
	type tc struct {
		mt, nt, wm, wn, kb int
		tall               bool
	}
	tiles := []tc{{4, 2, 2, 2, 1, false}, {2, 2, 2, 2, 2, false}, {4, 2, 2, 2, 1, true}, {2, 4, 2, 2, 4, false}}
	for _, q := range []format{{kernels.Q8_0, mainkernels.Q8_0}, {kernels.Q4_0, mainkernels.Q4_0}, {kernels.MXFP4, mainkernels.MXFP4},
		{kernels.Q5_0, mainkernels.Q5_0}, {kernels.Q4_K, mainkernels.Q4_K}, {kernels.Q6_K, mainkernels.Q6_K}} {
		for _, tl := range tiles {
			for _, ex := range []int{1, 4} {
				for _, bias := range []bool{false, true} {
					name := fmt.Sprintf("%v_m%dn%dw%dx%dk%d_tall%v_e%d_b%v", q.c, tl.mt, tl.nt, tl.wm, tl.wn, tl.kb, tl.tall, ex, bias)
					t.Run(name, func(t *testing.T) {
						rows, k, ntok := 128, 1024, 64
						ms := mainkernels.MatVecShape{T: q.m, K: k, Rows: rows, NTok: ntok, Experts: ex, Bias: bias}
						mk, err := mainkernels.GemmTile(ms, mainkernels.TileGemm{MT: tl.mt, NT: tl.nt, WM: tl.wm, WN: tl.wn, KB: tl.kb, Tall: tl.tall})
						if err != nil {
							t.Skip(err)
						}
						s := compose.GemmTile{T: q.c, Rows: rows, K: k, NTok: ntok, MT: tl.mt, NT: tl.nt, WM: tl.wm, WN: tl.wn, KB: tl.kb, Tall: tl.tall, Experts: ex}
						g := &compose.Row{Width: s.Threads(), Rows: s.Groups()}
						qs, d, sc, b, out := g.Buf(compose.BufU32), g.Buf(compose.BufU32), g.Buf(compose.BufU32), g.Buf(compose.BufU32), g.Buf(compose.BufTile)
						bi, sel := -1, -1
						if bias {
							bi = g.Buf(compose.BufTile)
						}
						if ex > 1 {
							sel = g.Buf(compose.BufU32)
						}
						if err := g.GemmTile(s, qs, d, sc, b, out, bi, sel); err != nil {
							t.Fatal(err)
						}
						same(t, name, []*ir.Kernel{mk}, planned(t, g, compose.Facts{Target: metalM4}))
					})
				}
			}
		}
	}
}
