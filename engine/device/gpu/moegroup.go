package gpu

import (
	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/composer/kernels"
)

// A mixture's prompt, batched. Batching needs the tokens routed to one expert
// to share its weight read, and the router decides that only at run time -- so
// the chunk is routed on the device, kernels.ExpertGroup* sort the (token,
// slot) pairs by expert with each expert's run padded to a whole matvec token
// group, the used column count is read back (the one word the grouped launches
// are sized by), and the expert matvecs run in that order with one expert per
// group (kernels.MatVec's grouped shape). ExpertCombineGrouped reads the
// outputs back in token order.
//
// The readback makes the block uncapturable, as a streamed block is.

// groupTok is how many sorted columns a grouped matvec thread carries, and so
// the padding unit of an expert's run: at a 128-row chunk and 8 of 128 experts
// an expert averages 8 tokens, and a group of 4 pads 1.5 of them.
const groupTok = 4

// voltaGroupTile is GemmVolta's grouped blocking for a matrix of rows rows at
// a padding unit of unit columns: four warps along the tokens, each unit/32
// n-tiles wide, and the widest m-tile the rows divide.
//
// Three m-tiles are tried where 128 does not divide: gpt-oss's 2880 rows are
// 30 blocks of 96, faster than 64x64 because the staged activations serve more
// rows. A 16-element format at MT 3 stages 192 items, which do not nest in 128
// threads, so the generator refuses it and voltaGroupedMV takes the next tile.
func voltaGroupTiles(q kernels.Quant, rows, unit int) []kernels.VoltaTile {
	if unit%32 != 0 {
		return nil
	}
	sub, _, _, _ := kernels.Layout(q)
	var out []kernels.VoltaTile
	for _, mt := range []int{4, 3, 2, 1} {
		tl := kernels.VoltaTile{MT: mt, NT: unit / 32, WM: 1, WN: 4, KB: max(32/sub, 1)}
		if rows%tl.Rows() == 0 {
			out = append(out, tl)
		}
	}
	return out
}

// sm70 is voltaOn's probe without its knobs, which the batched latent
// attention's scores ask too (mlabatch.go): this device has no integer matrix
// instruction and lowers MMAVolta. Cached per device.
func (g *devTier) sm70() bool {
	if g.voltaMoE == 0 {
		g.voltaMoE = -1
		probe := func(ker *ir.Kernel, err error) bool {
			if err != nil {
				return false
			}
			c, err := g.dev.Compile(ker)
			if err != nil {
				return false
			}
			closeKern(c)
			return true
		}
		if !g.mmaOff && !g.NoMMA {
			// Only a COMPILE refusal is the device's; a shape the generator
			// refuses says nothing about the card.
			if ker, err := kernels.MatVecMMA(kernels.MatVecShape{Center: g.center, T: kernels.Q8_0, K: 32, Rows: 16, NTok: 8,
				MT: 1, NT: 1}); err == nil {
				if c, err := g.dev.Compile(ker); err != nil {
					g.mmaOff, g.MMAWhy = true, err.Error()
				} else {
					closeKern(c)
				}
			}
		}
		tl := kernels.VoltaTile{MT: 4, NT: 1, WM: 1, WN: 4, KB: 1}
		if (g.mmaOff || g.NoMMA) && probe(kernels.GemmVolta(kernels.MatVecShape{Center: g.center, T: kernels.Q8_0, K: 32, Rows: 128,
			NTok: 32, Experts: 2}, tl)) {
			g.voltaMoE = 1
		}
	}
	return g.voltaMoE > 0
}

// tilesOK reports whether this device lowers the collective tile ops at the
// 8x8 binary16 shape GemmTile and FlashPrefillTile multiply in -- Metal's
// simdgroup_matrix, mma.sync on sm_75 and later -- asked once by compiling a
// probe (backend.LowersTile) instead of naming a backend. A device lowering
// them in sm_70's quad-pairs (backend.TileQuadPairs) runs these kernels' k of
// 8 on half its pairs, and GemmVolta is its staged GEMM: it is not asked. It
// is false once a built tile kernel has been refused (tileOff).
func (g *devTier) tilesOK() bool {
	if g.tileCap == 0 {
		g.tileCap = -1
		// A device that does not lower them is not a refusal (TileWhy is
		// what a device said about a kernel it was asked to build): it is
		// simply not asked.
		if ok, _ := backend.LowersTile(g.dev, backend.TileShape{M: 8, N: 8, K: 8, Elem: ir.TileF16, Acc: ir.TileF32}); ok && !backend.TileQuadPairs(g.dev) {
			g.tileCap = 1
		}
	}
	return g.tileCap == 1 && !g.tileOff
}

// tileF32 reports whether the device also lowers float32-operand tiles, which
// kernels.FlashPrefillTile multiplies P by V in: Metal's float
// simdgroup_matrix, and mma.sync from sm_80 (three tf32 products). sm_75 has
// the f16 tiles GemmTile needs and not these.
func (g *devTier) tileF32() bool {
	if !g.tilesOK() {
		return false
	}
	if g.tileF32Cap == 0 {
		g.tileF32Cap = -1
		if ok, _ := backend.LowersTile(g.dev, backend.TileShape{M: 8, N: 8, K: 8, Elem: ir.TileF32, Acc: ir.TileF32}); ok {
			g.tileF32Cap = 1
		}
	}
	return g.tileF32Cap == 1
}

// tileTall reports whether the device also lowers the 16-row tile
// (kernels.TileGemm.Tall): mma.sync's own m16, which the 8-row tile runs
// with half its rows zero. Metal's simdgroup_matrix is 8x8 only.
func (g *devTier) tileTall() bool {
	if g.tileTallCap == 0 {
		g.tileTallCap = -1
		if ok, _ := backend.LowersTile(g.dev, backend.TileShape{M: 16, N: 8, K: 8, Elem: ir.TileF16, Acc: ir.TileF32}); ok {
			g.tileTallCap = 1
		}
	}
	return g.tileTallCap == 1
}

// moeKernKey names a batched mixture's compiled kernel in devTier.ragKerns:
// the kind and the shape fields it is keyed on. A struct, where a formatted
// string was built for every lookup -- every bank of every mixture block of
// every step.
type moeKernKey struct {
}
