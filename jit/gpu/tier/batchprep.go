package tier

import (
	"errors"
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
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

// prepBatch builds the batched scratch and the batched twin of every matvec,
// on the first prompt that asks for one rather than at PrepLayer time.
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
	defer g.scratchWin().close()
	// The layer walk is not guarded by the scratch: ReleaseLayers deletes
	// blocks, so a moved seam brings back blocks whose batched kernels are not
	// compiled, and finding that mid-submission (where Compile is not allowed)
	// would refuse every later batched prompt. The compiled kernels live in
	// g.kerns keyed by shape, which holds every width; batKern is then a map
	// lookup, safe inside a Session.
	tok := g.bbs[width].tok
	bb := g.bbs[width]
	// The walk is ~25 kernel lookups a block, every prompt (0.9 ms of a
	// 512-row Qwen3-0.6B prompt on a Volta-class card); it only has something
	// to find after a block was placed or released.
	if bb.walked == g.layerGen+1 {
		return true
	}
	for _, l := range g.layers {
		if l == nil || !l.ok {
			continue
		}
		// Only this scratch set's blocks: another set's are prepared with its
		// own batched scratch (gemma4.go), which this one may not be able to
		// run -- MiniMax-M3's dense lead set has no mixture scratch for the
		// blocks after it.
		if g.geoUsed && l.geo != g.geoCur {
			continue
		}
		// A recurrent block batches through GatedDeltaScan (or the fused rule,
		// or Mamba-1's scan); a scratch that could not build one refuses here.
		if l.linear && !bb.rec.ShortConv && bb.deltaScan == nil && bb.deltaFused == nil && bb.m1Scan == nil {
			g.LastErr = "tier: batched prefill: no chunked delta rule for a recurrent block"
			return false
		}
		// Every matvec a batched block can launch (prepLayer's list), or the rest
		// would compile lazily inside the session.
		for _, m := range []*mv{&l.mvq, &l.mvk, &l.mvv, &l.mvo, &l.mvg, &l.mvu, &l.mvd,
			&l.mvsg, &l.mvsu, &l.mvsd, &l.mvSG, &l.mvBA, &l.mvSO,
			&l.mvQA, &l.mvQB, &l.mvKVA, &l.mvKB, &l.mvVB,
			&l.mvFA, &l.mvFB, &l.mvGA, &l.mvGB, &l.mvpg, &l.mvpp,
			&l.mvIQ, &l.mvIK, &l.mvIW, &l.mvLL, &l.mvLR, &l.mvSI, &l.mvMQ,
			&l.mvCKV, &l.mvCG, &l.mvICKV, &l.mvICG,
			&l.mvHAD, &l.mvHAU, &l.mvHAI, &l.mvHFD, &l.mvHFU, &l.mvHFI, &l.mvNGK, &l.mvNGV} {
			if m.kern == nil {
				continue
			}
			if m.slots > 0 && l.mla && (m == &l.mvKB || m == &l.mvVB) {
				// MLA's absorb banks run grouped with the head as the expert
				// (mlabatch.go): every head's run is the whole chunk. A float
				// bank reads the float activation (emitMLARows' grouped).
				if bb.mlab == nil {
					return false
				}
				if _, ok := g.groupedMV(*m, bb.mlab.nh, bb.mlab.nh*width); !ok {
					return false
				}
				continue
			}
			if m.slots > 0 {
				// A mixture runs grouped (moegroup.go): a streamed bank holds
				// only the selection.
				mg := bb.mg
				// A hybrid block routes the rows on the device and runs their
				// experts on the host, or, with an expert cache, the cached ones
				// on the card through the grouped kernels over the cache
				// (emitGroupedMoE, streamBank.splitRows): built here for every
				// hybrid block, since a cache may be sized after placement. A
				// kernel the shape refuses leaves that block's rows on the host.
				hybrid := mg != nil && l.stream != nil && l.stream.hybrid
				if hybrid {
					if mg.volta && !kernels.IsFloat(m.q) {
						bias, nsrc := moeVoltaArgs(l, m, mg)
						g.voltaGroupedMV(*m, mg.nExp, mg.p, mg.unit, bias, nsrc)
					} else if !mg.volta && !kernels.IsFloat(m.q) {
						g.groupedMV(*m, mg.nExp, mg.p)
					}
					continue
				}
				if mg == nil || l.stream != nil {
					// Named either way: a refusal with a stale LastErr reads as
					// some other block's failure.
					g.LastErr = "tier: batched mixture: a streamed bank"
					if mg == nil {
						g.LastErr = fmt.Sprintf("tier: batched mixture: a block has experts and this "+
							"%d-row scratch has no mixture set", width)
					}
					return false
				}
				// A float bank has no scales to quantize against: the grouped
				// matvec reads the sorted float rows (moeFloatHalf), on
				// sm_70 too, where the tensor-core twin takes no float.
				if kernels.IsFloat(m.q) {
					if !g.moeFloatHalf(mg, &bb.p) {
						return false
					}
					if _, ok := g.groupedMV(*m, mg.nExp, mg.p); !ok {
						return false
					}
					continue
				}
				// On sm_70 the tensor-core twin or nothing: the scratch holds no
				// dp4a half there (newMoeGroup), so a bank without a Volta form
				// refuses the batch and the chunk goes row by row.
				if mg.volta {
					bias, nsrc := moeVoltaArgs(l, m, mg)
					if _, ok := g.voltaGroupedMV(*m, mg.nExp, mg.p, mg.unit, bias, nsrc); !ok {
						return false
					}
					continue
				}
				if _, ok := g.groupedMV(*m, mg.nExp, mg.p); !ok {
					return false
				}
				continue
			}
			if _, ok := g.batchMV(*m, width, tok); !ok {
				return false
			}
		}
	}
	// DeepSeek V4's grouped output at this width (ds4.go).
	for _, l := range g.layers {
		if l != nil && l.ok && !g.ds4Twins(l, width) {
			return false
		}
	}
	bb.walked = g.layerGen + 1
	return true
}

// batchGrain is the coarsest granularity every batched kernel divides: the
// matrix instruction covers 8 token columns and the shipped tile takes four of
// them, so a chunk width is a multiple of 32. A ragged tail is rounded up to
// this rather than to the full width, which is the difference between 67% and
// 33% waste on a 42-row tail.
const batchGrain = 8 * 4

// promptGrain is the grain a prompt chunk's own width is rounded to: the
// batched kernels' batchGrain, or on a device that runs a chunk's attention
// as FlashPrefill80 the row block that kernel takes (64 rows; a 32-row grain
// sent every width that is not a multiple of 64 to the staged kernels and
// the int8 operands). Callers hold g.mu.
func (g *devTier) promptGrain() int {
	if !g.kb.noAttnMMA && g.f16GemmK() == 16 {
		return max(batchGrain, 16*4)
	}
	return batchGrain
}

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

// quantizeRows is the quantizer of rows activations n wide: Quantize over
// them where n is a whole number of blocks, QuantizePadded with a zero tail
// where it is not (launch QuantizeThreads(rows*pad32(n)/32)). The window is
// clamped to the padded row, as the host's ragged quantizer clamps it.
func quantizeRows(rows, n, win int) (*ir.Kernel, error) {
	return quantizeRowsAt(rows, n, pad32(n), win)
}

// quantizeRowsAt is quantizeRows with the stored row kp wide, a whole number
// of blocks and n or more (a ragged tower's padded width, nn.LayerPlan.KPad).
func quantizeRowsAt(rows, n, kp, win int) (*ir.Kernel, error) {
	if n == kp {
		return oldkernels.Quantize(rows*n, oldkernels.ClampWindow(win, n))
	}
	return oldkernels.QuantizePadded(rows, n, kp, oldkernels.ClampWindow(win, kp))
}

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

func (g *devTier) batchMV(m mv, ntok, tok int) (mv, bool) {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	if g.PreferTiles {
		if bm, ok := g.tileMV(m, ntok); ok && padOK(m, bm) {
			return bm, true
		}
	}
	if bm, ok := g.int8Gemm(m, ntok); ok && padOK(m, bm) {
		return bm, true
	}
	if bm, ok := g.f16Gemm(m, ntok); ok && padOK(m, bm) {
		return bm, true
	}
	return g.intBatchMV(m, ntok, tok)
}

// padOK reports whether batched twin bm may serve m: not one that reads the
// float activation when m's is the padded int8 rows (mv.pad).
func padOK(m, bm mv) bool { return !m.pad || bm.act == nil }

// intBatchMV is batchMV without the m16n8 GEMM: the integer matrix
// instruction wherever the device has one (MatVecMMA is faster than the dp4a
// tile and bit-identical to it), then the other tiled forms, then the dp4a
// tile. The widest admissible warp tile is a divisor calculation (mmaTile),
// not a search. A ragged step takes it directly (ragMV): its few rows a
// sequence are decode's, and keep decode's int8 arithmetic.
func (g *devTier) intBatchMV(m mv, ntok, tok int) (mv, bool) {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	// A float weight has no tensor-core kernel. Asking would fail, and a
	// failure here switches MMA off for every other matvec (mmaOff).
	if mt, nt, ok := g.mmaTile(m.rows, ntok); ok && !kernels.IsFloat(m.q) {
		kk := kernKey{m.q, m.k, m.rows, -(ntok*100 + mt*10 + nt), m.bias != nil, 1, false, false, 0, false}
		c, cached := g.kerns[kk]
		if !cached {
			ker, err := oldkernels.MatVecMMA(oldkernels.MatVecShape{Center: g.center,
				T: m.q, K: m.k, Rows: m.rows, NTok: ntok, MT: mt, NT: nt,
				Bias: m.bias != nil, ActWin: g.actWinFor(ntok)})
			if err == nil {
				c, err = g.dev.Compile(ker)
			}
			if err != nil {
				// A backend without a matrix instruction is a property of the
				// device, not the shape, so it is recorded once instead of every
				// matvec re-discovering it with a Compile call.
				g.mmaOff, g.MMAWhy = true, err.Error()
				c = nil
			}
			g.kerns[kk] = c
		}
		if c != nil {
			return mv{kern: c, rows: m.rows, split: 1, bias: m.bias, q: m.q, k: m.k,
				threads: (m.rows / (16 * mt)) * (ntok / (8 * nt)) * 32}, true
		}
	}
	if bm, ok := g.tileMV(m, ntok); ok && padOK(m, bm) {
		return bm, true
	}
	if bm, ok := g.voltaMV(m, ntok); ok && padOK(m, bm) {
		return bm, true
	}
	return g.batchMVDot4(m, ntok, tok)
}

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

func (g *devTier) batchMVDot4(m mv, ntok, tok int) (mv, bool) {
	// Rows per thread is per shape because it only divides some; the
	// fallback is 2 and then 1.
	rowt := 1
	for _, r := range []int{g.tiles.rowt, 2} {
		if r > 1 && m.rows%r == 0 {
			rowt = r
			break
		}
	}
	// dot4Split passes the bias (this is every non-MMA backend's batched
	// path) and splits k where the tile leaves the grid thin.
	return g.dot4Split(m, ntok, tok, rowt, g.kb.batch.Split)
}

// kStride is the device K cache's transposed stride, or 0 for the row-major
// layout. See oldkernels.RoPERowsT: the scores kernel reads K with one thread per
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
