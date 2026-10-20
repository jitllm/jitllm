package tier

import (
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// ones32 is n float32 ones.
func ones32(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 1
	}
	return out
}

// volta70AttnMT and volta70AttnNT are AttnScoresMMA70's warp tile, 64 keys by 32
// queries, and AttnAccMMA70's query tile. The accumulate's 32 query rows are
// also the softmax's group (atile), since it walks its group's widest count.
const volta70AttnMT, volta70AttnNT = 2, 4

// volta70AccMT is AttnAccMMA70's dims per warp in 32s: one, so every 32 dims of
// a head is its own warp. The whole head per warp starved the card (too few
// warps, each walking its causal width serially); one per 32 dims roughly
// halved the accumulate on a Volta-class card.
const volta70AccMT = 1

// volta70LatMT is the same tile for a batched latent block (mlabatch.go): the
// whole row up to 128 dims a warp. It is zero below 32 dims, so every caller
// checks lat >= 32 first.
func volta70LatMT(dims int) int { return min(dims/32, 4) }

// volta70Attn builds bs's batched attention on sm_70's m8n8k4, both products,
// or neither: the scores alone would still need the accumulate's 32-row group
// in the softmax, and the pair is gated as a pair. It is for a device that
// lowers MMAVolta and not m16n8k16 (the caller tried that first), with the
// transposed f32 K cache these read; anything else keeps the FMA tiles.
func (g *devTier) volta70Attn(bs *blockScratch, kvDim, gqa int, scale float32) bool {
	p := &bs.p
	kst := g.kStride(p)
	if g.NoVolta || g.kb.noAttnMMA || g.KVF16 || bs.rows <= 1 || kst <= 0 || p.MLA() ||
		32*volta70AttnMT > maxKeyTile ||
		p.HeadDim%32 != 0 || bs.rows%(8*volta70AttnNT) != 0 || bs.sstride%4 != 0 {
		return false
	}
	comp := func(k *ir.Kernel, err error) backend.Kernel {
		if err != nil {
			return nil
		}
		c, err := g.dev.Compile(k)
		if err != nil {
			return nil
		}
		return c
	}
	sk := comp(oldkernels.AttnScoresMMA70(p.NHead, p.HeadDim, kvDim, gqa, bs.sstride, scale, bs.rows, kst,
		volta70AttnMT, volta70AttnNT, 0))
	ak := comp(oldkernels.AttnAccMMA70(p.NHead, p.HeadDim, kvDim, gqa, bs.sstride, bs.rows,
		volta70AccMT, volta70AttnNT))
	var sw backend.Kernel
	if p.SWAWindow > 0 {
		sw = comp(oldkernels.AttnScoresMMA70(p.NHead, p.HeadDim, kvDim, gqa, bs.sstride, scale, bs.rows, kst,
			volta70AttnMT, volta70AttnNT, windowArg(p)))
	}
	if sk == nil || ak == nil || (p.SWAWindow > 0 && sw == nil) {
		for _, k := range []backend.Kernel{sk, ak, sw} {
			if k != nil {
				g.closeKern(k)
			}
		}
		return false
	}
	bs.scores70, bs.acc70, bs.scores70W = sk, ak, sw
	bs.atile = 8 * volta70AttnNT
	g.AttnMMA = true
	// The one-kernel form where it takes the shape (no sinks, no chunked
	// window); the pair above stays built as everyone else's path and
	// NoFlashPrefill's arm.
	if !g.NoFlashPrefill && !p.AttnSinks && !p.SWAChunked && bs.rows%oldkernels.FlashPrefill70Rows == 0 {
		fs := oldkernels.FlashPrefill70Shape{Heads: p.NHead, KVHeads: p.NHead / gqa, Dim: p.HeadDim, Rows: bs.rows,
			KStride: kst, Scale: scale, Softcap: p.AttnSoftcap}
		fk := comp(oldkernels.FlashPrefill70(fs))
		var fw backend.Kernel
		if fk != nil && p.SWAWindow > 0 {
			fs.Window = p.SWAWindow
			if fw = comp(oldkernels.FlashPrefill70(fs)); fw == nil {
				g.closeKern(fk)
				fk = nil
			}
		}
		bs.fp70, bs.fp70W = fk, fw
	}
	return true
}

// mla70Attn is whether a batched latent block of rows rows takes its scores
// and its accumulate on sm_70's m8n8k4 (mlabatch.go): the device is sm_70, the
// cache is float32, the chunk is a whole number of the kernels' 32-row query
// groups and the latent a whole number of the accumulate's 128-wide warps.
func (g *devTier) mla70Attn(p *nn.LayerPlan, rows int) bool {
	lat := p.KVLoraRank
	return !g.NoVolta && !g.kb.noAttnMMA && !g.KVF16 && rows%(8*volta70AttnNT) == 0 &&
		32*volta70AttnMT <= maxKeyTile && p.KVRow()%4 == 0 && lat >= 32 &&
		lat%(32*volta70LatMT(lat)) == 0 && g.sm70()
}

// voltaBlock reports whether every matvec of block l runs, at this batch width,
// as sm_70's f16 twin (voltaMV), which reads the float activation through
// ActF16 and never the int8 one. A lookup: batchMV's twins are compiled by
// prepBatch before any session.
func (g *devTier) voltaBlock(l *layer, ntok, tok int) bool {
	if l.router != nil || l.linear || l.mla || l.stream != nil || l.clampB != nil {
		return false
	}
	n := 0
	for _, m := range []mv{l.mvq, l.mvk, l.mvv, l.mvo, l.mvg, l.mvu, l.mvd} {
		if m.kern == nil {
			continue
		}
		bm, ok := g.batchMV(m, ntok, tok)
		if !ok || bm.act == nil {
			return false
		}
		n++
	}
	return n > 0
}

// flashTileAttn builds oldkernels.FlashPrefillTile for a batched block where it
// takes the shape: a device that lowers the collective tiles (tilesOK), the
// transposed float32 K cache and float32 V, no sinks and no chunked window, a
// chunk that is a whole number of the kernel's query blocks. It leaves fpT nil
// otherwise, and the three-kernel path runs.
func (g *devTier) flashTileAttn(bs *blockScratch, kvDim, gqa int, scale float32) {
	p := &bs.p
	kst := g.kStride(p)
	if g.NoFlashPrefill || g.KVF16 || bs.rows <= 1 || kst <= 0 || !g.tileF32() ||
		p.MLA() || p.NonCausal || p.AttnSinks || p.SWAChunked || kvDim%4 != 0 {
		return
	}
	// Only at the 32-row blocking, as measured on Apple Silicon: a workgroup
	// re-stages K and V for every block of query rows, so a narrower block
	// (wide heads) lost to the three kernels. Staging K/V once per shared kv
	// head would fix the wide heads, but the float32 output in workgroup memory
	// leaves no room.
	t, ok := oldkernels.FlashTileFor(p.HeadDim)
	if !ok || t.SG < 4 || bs.rows%t.Rows() != 0 {
		return
	}
	comp := func(k *ir.Kernel, err error) backend.Kernel {
		if err != nil {
			return nil
		}
		c, err := g.dev.Compile(k)
		if err != nil {
			return nil
		}
		return c
	}
	fs := oldkernels.FlashPrefill70Shape{Heads: p.NHead, KVHeads: p.NHead / gqa, Dim: p.HeadDim, Rows: bs.rows,
		KStride: kst, Scale: scale, Softcap: p.AttnSoftcap}
	fk := comp(oldkernels.FlashPrefillTile(fs, t))
	var fw backend.Kernel
	if fk != nil && p.SWAWindow > 0 {
		fs.Window = p.SWAWindow
		if fw = comp(oldkernels.FlashPrefillTile(fs, t)); fw == nil {
			g.closeKern(fk)
			fk = nil
		}
	}
	bs.fpT, bs.fpTW, bs.fpTile = fk, fw, t
}

// spanKernel builds the decode scratch's row copy the folded head reads (a
// chunk's last row into the decode scratch) if it has none. Callers hold g.mu.
func (g *devTier) spanKernel(hb *blockScratch) error {
	if hb.spanK != nil {
		return nil
	}
	kk, e := oldkernels.CopySpan(hb.p.NEmbd)
	var k backend.Kernel
	if e == nil {
		k, e = g.dev.Compile(kk)
	}
	var ob backend.Buf
	if e == nil {
		w := g.scratchWin() // the scratch's own word (scratch.go)
		if ob, e = g.dev.Alloc(4); e != nil {
			g.closeKern(k)
		}
		w.close()
	}
	if e != nil {
		return e
	}
	hb.spanK, hb.spanOff = k, ob
	return nil
}
