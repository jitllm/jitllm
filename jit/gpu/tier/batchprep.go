package tier

import (
	"errors"
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
)

// errNoBatch is what a block reports when a matvec in it has no batched twin --
// a mixture of experts, where a token batch and an expert batch want the
// same slot machinery for two different things.
var errNoBatch = errors.New("tier: no batched matvec for this block")

// BatchTok is how many token columns one thread of a batched matvec carries.
// A chunk of NTok tokens read one at a time reads the weights NTok times; with
// Tok columns per thread it reads them NTok/Tok times. Larger runs out of
// registers and occupancy. 16 won the sweep once attention was tiled; the best
// width moves with the kernel mix, so re-sweep after changing it.
// BatchShape.Tok overrides it.
const BatchTok = 16

// The batched prefill's tile geometry. Each is the fitted default; a tier's
// WithBatchShape and WithMMATile pins replace it for that tier (tilesFor).
//
// defBatchRowt is how many rows one thread carries. See MatVecShape.Rowt: Tok
// alone plateaus because the kernel issues one activation load per dot4, and a
// tile of rows shares those loads. 4 won by a wide margin in mvbench -ntok 128;
// 8 does not fit its accumulators.
//
// defMMAMT and defMMANT are the warp tile of the matrix-instruction matvec: MT
// blocks of 16 weight rows by NT blocks of 8 token columns. 4x4 beat narrower
// tiles and the dp4a tile in mvbench -mma -ntok 128 (see mmaTile).
//
// defAttnQTile and defAttnKTile are the query rows and key positions one thread
// of the batched scores kernel carries. maxKeyTile bounds the score-row
// padding, so it must not be smaller than the key tile. 4x2 won the end-to-end
// sweep: the query side is a broadcast (widening costs registers) while the key
// side is a strided walk (widening costs coalescing).
//
// defAttnATile is how many query rows one AttnAcc thread carries, so one V load
// serves all of them (see AttnAccTiled); 8 beat 1 end to end.
//
// defAttnNT is how many 8-position key tiles one warp of the tensor-core scores
// kernel carries, so one q fragment serves all of them; 2 beat 1, and 4 is no
// better.
//
// defBatchParts is the RMSNorm partial count for a batched chunk; see
// initScratch.
const (
	defBatchRowt               = 4
	defMMAMT, defMMANT         = 4, 4
	defAttnQTile, defAttnKTile = 4, 2
	defAttnATile               = 8
	defAttnNT                  = 2
	defBatchParts              = 64
)

// tiles is one tier's resolved batched geometry.
type tiles struct {
	rowt, mmaMT, mmaNT, qtile, ktile, atile, attnNT, parts int
}

// tilesFor applies a tier's pins to the defaults.
func tilesFor(kb knobs) tiles {
	t := tiles{rowt: defBatchRowt, mmaMT: defMMAMT, mmaNT: defMMANT, qtile: defAttnQTile,
		ktile: defAttnKTile, atile: defAttnATile, attnNT: defAttnNT, parts: defBatchParts}
	b := kb.batch
	if b.RowT >= 1 {
		t.rowt = b.RowT
	}
	for _, e := range []struct {
		v   int
		dst *int
	}{{b.QTile, &t.qtile}, {b.KTile, &t.ktile}, {b.ATile, &t.atile}, {b.Parts, &t.parts},
		{b.AttnNT, &t.attnNT}} {
		if e.v >= 1 && e.v <= 128 {
			*e.dst = e.v
		}
	}
	if kb.mmaMT > 0 && kb.mmaNT > 0 {
		t.mmaMT, t.mmaNT = kb.mmaMT, kb.mmaNT
	}
	return t
}

// maxKeyTile bounds how far past a row's causal width a kernel may write, so
// the score row is padded by it. The FMA tile writes up to ktile-1 past; the
// matrix kernel up to 8*attnNT-1; sm_70's AttnScoresMMA70 up to
// 32*volta70AttnMT-1.
//
// It must cover the widest tile: an overshoot writes -inf into the next row's
// scores, which softmaxes to NaN. volta70Attn refuses a tile wider than this.
const maxKeyTile = 64

// scoreStride is a block's score row: the context, padded for the batched
// arm -- AttnScoresTiled's last key tile can write up to maxKeyTile-1 past the
// causal width, which unpadded lands in the next row's scores -- and rounded
// to four floats, since AttnAccMMA70 reads a row's probabilities 16 bytes at a
// time and every batched kernel takes this one stride.
func scoreStride(p *nn.LayerPlan, rows int) int {
	if rows > 1 {
		return (p.MaxSeq + maxKeyTile + 3) &^ 3
	}
	return p.MaxSeq
}

// batchWidth is how many token positions one batched submission carries. It has
// to be a multiple of BatchTok, and larger only amortizes the launches -- the
// weight traffic is set by Tok alone.
const batchWidth = nn.MaxDevicePrefillChunk

// prepBatch builds the batched scratch on the first prompt that asks for one
// rather than at PrepLayer time.
//
// It is lazy because the scratch costs VRAM (mostly score buffers) that a
// decode-only process never uses, and separate because Alloc and Compile cannot
// run inside a Session.
func (g *devTier) prepBatch(width int) bool {
	// 8, not batchGrain: a ragged step on a device without a matrix
	// instruction runs at a multiple of 8 (see LayersRows).
	if g.bs == nil || width < 8 || width%8 != 0 {
		return false
	}
	if g.bbs == nil {
		g.bbs = map[int]*blockScratch{}
	}
	if g.bbs[width] == nil {
		if g.noRoomFor(width) {
			return false
		}
		tok := BatchTok
		if k := g.kb.batch.Tok; k >= 1 && batchWidth%k == 0 {
			tok = k
		}
		for tok > 1 && width%tok != 0 {
			tok /= 2
		}
		bs := g.initScratch(&g.bs.p, width)
		if bs == nil {
			return false
		}
		bs.tok = tok
		g.bbs[width] = bs
		// A width built after placement filled the card must not take what
		// the blocks were budgeted: it is refused, and the caller runs the
		// reserved width (batchFor) or narrower submissions. Without the
		// reservation the device asks the card, as it did before it. Nor
		// may it take the history admitted rows were promised (PromiseKV).
		// Free pool pages are not a reservation (PrewarmKV grows them into
		// whatever room placement left): they are given back first.
		if !g.NoScratchReserve && !g.kvCompact(g.promisedBytes(), nil) && width != g.promptW && width != g.stepW && !g.reserving {
			g.dropBatch(width)
			g.markNoRoom(width)
			g.ScratchRefused++
			g.LastErr = fmt.Sprintf("a %d-row batched scratch does not fit the budget (%d of %d bytes charged)", width, g.used, g.limit)
			return false
		}
	}
	return true
}

// batchGrain is the coarsest granularity every batched kernel divides: the
// matrix instruction covers 8 token columns and the shipped tile takes four of
// them, so a chunk width is a multiple of 32. A ragged tail is rounded up to
// this rather than to the full width, which is the difference between 67% and
// 33% waste on a 42-row tail.
const batchGrain = 8 * 4


// actWinFor is the activation window of the scratch set in use: a tier with a
// vision tower has a set per segment, each with its plan, and the window is a
// codegen input both tiers must agree on (nn.LayerPlan.ActWin).
func (g *devTier) actWinFor(ntok int) int {
	if g.bs != nil {
		return g.bs.p.ActWin
	}
	return 32
}

// batchMV compiles one matvec's batched twin for a prompt chunk: the staged
// int8 GEMM where Config.PreferGemmInt8 asks for it (int8Gemm, sm_80 on), the
// staged binary16 GEMM on the m16n8 instruction where the device has it
// (f16Gemm, sm_75 on), then intBatchMV's forms.
// pad32 is n rounded up to the activation quantizer's 32-element block: the
// width a ragged activation is quantized at, its tail zero.
func pad32(n int) int { return (n + 31) / 32 * 32 }


// ePad is the width a block's residual-reading matrices quantize their
// activation at: NEmbd rounded up to a whole quantizer block, or the padded
// width of a ragged tower (nn.LayerPlan.KPad). A ragged width only reaches a
// float-weight matrix, which reads the float rows and ignores the int8 ones.
func ePad(p *nn.LayerPlan) int {
	if p.KPad > 0 {
		return p.KPad
	}
	return pad32(p.NEmbd)
}

// qPad is ePad for the attention's output, qdim wide: padded where the
// output is the tower's own width.
func qPad(p *nn.LayerPlan, qdim int) int {
	if p.KPad > 0 && qdim == p.NEmbd {
		return p.KPad
	}
	return pad32(qdim)
}

// attendsIn reports whether a block of [lo, hi) on this device attends: not
// a linear block's recurrence.
func (g *devTier) attendsIn(lo, hi int) bool {
	for li := lo; li < hi; li++ {
		if l := g.layers[li]; l != nil && !l.linear {
			return true
		}
	}
	return false
}


// padOK reports whether batched twin bm may serve m: not one that reads the
// float activation when m's is the padded int8 rows (mv.pad).
func padOK(m, bm mv) bool { return !m.pad || bm.act == nil }

// mmaTile is the widest warp tile this shape admits: MT blocks of 16 rows by NT
// blocks of 8 token columns, each capped at the measured optimum and reduced to
// a divisor of the shape.
//
// 4x4 is measured, not derived: wider tiles run out of registers (MT*NT*4
// accumulators a lane) and narrower ones read the weights more times per chunk.
// mvbench -mma has the per-shape isolated numbers.
func (g *devTier) mmaTile(rows, ntok int) (int, int, bool) {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	if g.mmaOff || g.NoMMA {
		return 0, 0, false
	}
	mt, nt := g.tiles.mmaMT, g.tiles.mmaNT
	for mt > 1 && rows%(16*mt) != 0 {
		mt /= 2
	}
	for nt > 1 && ntok%(8*nt) != 0 {
		nt /= 2
	}
	// Below the smallest tile the shape simply cannot be covered by a warp: 16
	// rows and 8 columns are the instruction's own granularity.
	if rows%16 != 0 || ntok%8 != 0 {
		return 0, 0, false
	}
	return mt, nt, true
}


// kStride is the device K cache's transposed stride, or 0 for the row-major
// layout. See mainkernels.RoPERowsT: the scores kernel reads K with one thread per
// position, so [head][dim][position] coalesces where [position][head][dim] costs
// eight times the memory transactions.
//
// It is one value for the whole session because the rotation writes the layout
// and attention reads it; they cannot disagree. Config.NoKT pins it off, which
// is the bisection switch and the A/B arm.
func (g *devTier) kStride(p *nn.LayerPlan) int {
	if g.NoKT {
		return 0
	}
	// A latent cache is row-major: one row per position is written by two
	// kernels at two offsets and read whole by every head, so a transposed
	// layout would scatter both writes to help only the scores. Row-major
	// also keeps MigrateKV a straight copy.
	if p.MLA() {
		return 0
	}
	if c := g.capped(p); c.MaxSeq != p.MaxSeq {
		return c.MaxSeq + 1
	}
	// One slot past MaxSeq: PrepLayer allocates it for the batched prefill's
	// padded rows (their K goes to the trash slot), so the stride has to include
	// it or the last real position lands on top of another head's first.
	return p.MaxSeq + 1
}
// promptGrain is the grain a prompt chunk's own width is rounded to.
func (g *devTier) promptGrain() int { return batchGrain }

