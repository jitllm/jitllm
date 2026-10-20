package tier

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/jitllm/jitllm/jit/gpu/compose"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// initRopeTable builds the device's own rotary table for this scratch, or
// leaves it nil and records why in Stats.RopeTableWhy.
//
// It never fails the scratch: Layers is also given the host's finished
// cs/csSWA, so a device without the kernel uploads the table and computes the
// same logits. Only a silent absence would be wrong, hence the recorded reason.
// The planes depend on the model, not the position, so they go up once and a
// token writes four bytes a row instead of NRot*4. It is built even under
// Config.RopeTableHost so `jitllm verify -ab ropetab` can alternate both arms
// in one process.
func (g *devTier) initRopeTable(bs *blockScratch, p *nn.LayerPlan, rows int) error {
	why := func(f string, a ...any) error {
		g.RopeTableWhy = fmt.Sprintf(f, a...)
		return nil
	}
	switch {
	case p.NonCausal || p.NRot <= 0:
		// Not an absence: a ViT's rotary, where it has one, is a per-row table
		// the host builds per grid and hands over as cs; with none, bs.cs is
		// nil too and there is no table for either tier to build.
		return nil
	case len(p.RopeTab) == 0:
		return why("the plan carries no folded rotary planes")
	}
	npairs := p.NRot / 2
	if want := kernels.RopeTabBlock(npairs); len(p.RopeTab) != want {
		return why("the plan's rotary planes are %d words, the layout wants %d",
			len(p.RopeTab), want)
	}
	if p.SWAPeriod > 0 && len(p.RopeTabSWA) != kernels.RopeTabBlock(npairs) {
		return why("SWAPeriod is %d and the LOCAL rotary planes are %d words",
			p.SWAPeriod, len(p.RopeTabSWA))
	}
	r := compose.Row{Width: npairs, Rows: rows}
	out, planes, consts, pos := r.Buf(compose.BufOut), r.Buf(compose.BufRow), r.Buf(compose.BufU32), r.Buf(compose.BufU32)
	r.RopeTable(out, planes, consts, pos)
	c := g.rowKern(&r).k
	if c == nil {
		return why("%s formed no rotary table", g.dev.API())
	}
	// The buffers are allocated after the compile, so a device that refuses the
	// kernel spends nothing; an allocation failure from here on is a real error.
	up := func(dst *backend.Buf, v []float32) error {
		b, err := g.dev.Alloc(len(v) * 4)
		if err != nil {
			return err
		}
		*dst = b
		return b.Write(f32b(v))
	}
	if err := up(&bs.ropeTab, p.RopeTab); err != nil {
		return err
	}
	if p.SWAPeriod > 0 {
		if err := up(&bs.ropeTabSWA, p.RopeTabSWA); err != nil {
			return err
		}
	}
	if err := up(&bs.ropeConst, kernels.RopeTabConsts()); err != nil {
		return err
	}
	// (rows+1) words, the first being the valid row count as in bs.n, so a
	// ragged chunk's per-token staging is one small write.
	b, err := g.dev.Alloc((rows + 1) * 4)
	if err != nil {
		return err
	}
	// The table is zeroed once: the kernel writes only valid rows, so the padded
	// tail must already hold the zeros the host staging would have put there.
	// Nothing else writes bs.cs, so this deliberately un-poisons it under
	// WithPoisonScratch.
	z := make([]byte, rows*p.NRot*4)
	for _, t := range []backend.Buf{bs.cs, bs.csSWA} {
		if t == nil {
			continue
		}
		if err := t.Write(z); err != nil {
			b.Free()
			return err
		}
	}
	bs.rpos, bs.ropeTable, bs.ropeNPairs = b, c, npairs
	return nil
}

// initScratch allocates and compiles everything that does not depend on which
// block is running.
func (g *devTier) initScratch(p *nn.LayerPlan, rows int) (out *blockScratch) {
	// Everything the build allocates is the scratch's, charged (scratch.go).
	defer g.scratchWin().close()
	if rows < 1 {
		rows = 1
	}
	// The capped plan: every bound, stride and score-row width below reads bs.p,
	// so the scratch is built for what the device holds, not the full context.
	cp := g.capped(p)
	p = &cp
	// A vision block's attention may go in query chunks (visionChunk), whose
	// scratch is rounded up to whole chunks before anything is sized by it.
	if p.NonCausal && rows > 1 {
		_, rows = visionChunk(rows, p.NHead, scoreStride(p, rows), g.tiles.qtile, g.tiles.atile, g.planeBudget())
	}
	bs := &blockScratch{p: cp, rows: rows, qtile: 1, ktile: 1, atile: 1}
	// A build that fails part-way frees what it allocated: every refusal below
	// is a plain return nil, and on a full card -- the case that refuses --
	// what it leaked was what the next build needed.
	defer func() {
		if out == nil {
			freeScratch(bs, g.devShared)
		}
	}()
	// What the build allocated, by the device's own count: a tower's set is
	// resized from it (visrows.go). Read here rather than from the window,
	// which measures only when it is the outermost.
	defer func(a0 int64) {
		if out != nil {
			out.bytes = g.allocated() - a0
		}
	}(g.allocated())
	if rows > 1 {
		bs.qtile, bs.ktile, bs.atile = g.tiles.qtile, g.tiles.ktile, g.tiles.atile
		for rows%bs.qtile != 0 {
			bs.qtile /= 2
		}
		for rows%bs.atile != 0 {
			bs.atile /= 2
		}
	}
	// The norm reduction wants enough parts to spread across the device and few
	// enough that NormApply's re-sum stays cheap. 64 wins even at 128 rows: the
	// re-read is cached, and the reduction's thread count is what binds (see
	// docs/engineering-history/gpu-kernels.md).
	bs.parts = 64
	if rows > 1 {
		bs.parts = g.tiles.parts
	}
	for bs.parts > 1 && p.NEmbd%bs.parts != 0 {
		bs.parts /= 2
	}
	qdim := p.NHead * p.HeadDim
	// The position split is for one token only: at rows > 1 the grid already
	// exceeds the card's slots, so AttnAccTiled runs unsplit.
	if rows == 1 {
		if sp := accSplitFor(qdim, g.dev.Slots(), g.kb.mvAccSplit); sp > 1 && !g.sizePart(qdim*sp) {
			return nil
		}
	}
	// The cache row, not NKVHead*HeadDim: they differ only under MLA, where a
	// position is one KVLoraRank+NRot row for the whole layer.
	kvDim := p.KVRow()
	// bs.a and bs.ax are shared by every quantize in the block, so they are
	// sized for the widest vector that gets quantized; a short buffer is a
	// device-side out-of-bounds write, not a Go panic. A mixture quantizes
	// NExpertUsed*NFFNExp, which is not p.NFFN (the dense width), so it is
	// computed rather than assumed.
	bs.nExpert, bs.nUsed, bs.nFFNExp = p.NExpert, p.NExpertUsed, p.NFFNExp
	moeWide := 0
	if p.NExpert > 0 {
		moeWide = p.NExpertUsed * p.NFFNExp
	}
	wide := ePad(p)
	// The shared expert, the linear block's gated output (VHeads*VDim) and its
	// conv channels, and MLA's four widths below all quantize through the same
	// buffers, so all are in the max.
	inner, chans := 0, 0
	if r := cp.Recurrent; r.Conv > 0 {
		inner, chans = r.VHeads*r.VDim, r.Chans
	}
	mlaNopeW, mlaLatW, mlaOutW, mlaQAW := 0, 0, 0, 0
	if p.MLA() {
		bs.mlaLat, bs.mlaRot = p.KVLoraRank, p.NRot
		// The query's and kv_a's row: the latent and the rotary key. The cached
		// row (KVRow) also carries the indexer's key after them.
		bs.mlaRow, bs.mlaNopeW = p.KVLoraRank+p.NRot, p.HeadDim-p.NRot
		mlaNopeW = p.NHead * bs.mlaNopeW
		mlaLatW, mlaOutW, mlaQAW = p.NHead*p.KVLoraRank, p.NHead*p.HeadDimV, p.QLoraRank
	}
	// DeepSeek V4's query latent and grouped output (ds4.go).
	ds4QA, ds4OA := 0, 0
	if d := p.DS4; d != nil {
		ds4QA, ds4OA = p.QLoraRank, d.OGroups*d.OLoraRank
	}
	k3Lat := 0
	if p.K3 != nil {
		k3Lat = pad32(p.K3.Latent)
	}
	for _, n := range []int{p.NFFN, qPad(p, qdim), pad32(p.PLEDim), k3Lat, moeWide, p.NFFNShExp, pad32(inner), chans,
		mlaNopeW, mlaLatW, mlaOutW, mlaQAW, ds4QA, ds4OA} {
		if n > wide {
			wide = n
		}
	}
	var err error
	al := func(dst *backend.Buf, n int) {
		// cuMemAlloc(0) is an error, not an empty buffer; an absent buffer is
		// right, since no kernel reads it (e.g. a vision block's NRot is 0).
		if n == 0 {
			return
		}
		if err == nil {
			// The device refusing scratch is the card being full, the same
			// as a refused weight upload: counted as NoRoom, so a caller can
			// tell it from a shape this tier declines.
			if *dst, err = g.dev.Alloc(n); err != nil {
				g.NoRoom++
			}
		}
		// RULE 13: zero is the identity for a sum, so a kernel reading scratch it
		// did not write is right on the first session and wrong on the next.
		// WithPoisonScratch fills with NaN so the read shows in the first token.
		if err == nil && g.kb.poisonScratch && *dst != nil {
			(*dst).Write(nanFill(n))
		}
	}
	// Every AltUp stream, stream-major (altup.go); one stream elsewhere.
	al(&bs.x, rows*p.ResidW()*4)
	// x2 is every stream too where a hyper-connection or Kimi-K3 writes it.
	x2 := p.NEmbd
	if p.DS4 != nil || p.K3 != nil || p.Q4E != nil {
		x2 = p.ResidW()
	}
	al(&bs.x2, rows*x2*4)
	al(&bs.h, rows*p.NEmbd*4)
	al(&bs.mvOut, rows*p.NEmbd*4)
	al(&bs.part, rows*bs.parts*4)
	if lnBlock(p) {
		// LayerNorm's second pass needs partials of its own: the mean's must
		// survive into the apply, so the variance cannot overwrite them.
		al(&bs.part2, rows*bs.parts*4)
	}
	al(&bs.a, rows*wide)           // one byte per activation: packed int8
	al(&bs.ax, 3*rows*(wide/32)*4) // scale per 32, sum per 16
	bs.aCap, bs.axCap = rows*wide, 3*rows*(wide/32)*4
	if p.NExpert > 0 {
		k := p.NExpertUsed
		al(&bs.rlogits, p.NExpert*4)
		al(&bs.rsel, (k+1)*4)
		al(&bs.ident, k*4)
		al(&bs.rtop, (k+1)*4)
		al(&bs.rw, (k+1)*4) // slot k is ExpertRoute's bin
		al(&bs.rwProbe, (k+1)*4)
		al(&bs.rwHit, (k+1)*4)
		al(&bs.rselPred, (k+1)*4)
		al(&bs.esplit, max(p.ExpWidth(), p.NEmbd)*4)
		al(&bs.hsum, max(p.ExpWidth(), p.NEmbd)*4)
		if p.DenseMoE {
			al(&bs.hr, rows*p.NEmbd*4)
			al(&bs.rwS, k*4)
		}
		if p.ExpertGroups > 1 {
			al(&bs.rgs, p.ExpertGroups*4)
			al(&bs.rbm, p.NExpert*4)
		}
		al(&bs.eg, k*p.NFFNExp*4)
		al(&bs.eu, k*p.NFFNExp*4)
		al(&bs.eact, k*p.NFFNExp*4)
		al(&bs.edown, k*p.NEmbd*4)
		if p.MoEBias {
			al(&bs.rlogitsB, p.NExpert*4)
			al(&bs.egB, k*p.NFFNExp*4)
			al(&bs.euB, k*p.NFFNExp*4)
			al(&bs.edownB, k*p.NEmbd*4)
		}
	}
	if p.QKNorm || p.QKL2Norm {
		al(&bs.qn, rows*qdim*4)
		al(&bs.kn, rows*kvDim*4)
	}
	al(&bs.qr, rows*qdim*4)
	if p.RopeSplit {
		al(&bs.krot, rows*kvDim*4)
		al(&bs.kroff, rows*4)
	}
	if p.AttnTempScale != 0 && !p.NoPEGlobal {
		al(&bs.qt, rows*qdim*4)
	}
	if p.NFFNShExp > 0 {
		bs.nShExp = p.NFFNShExp
		al(&bs.shg, rows*p.NFFNShExp*4)
		al(&bs.shu, rows*p.NFFNShExp*4)
		al(&bs.shact, rows*p.NFFNShExp*4)
		al(&bs.shout, rows*p.NEmbd*4)
		al(&bs.shlogit, rows*4)
		al(&bs.shsum, rows*p.NEmbd*4)
	}
	if p.SubNorm {
		// BitNet's inner norms write their rows here, out of place (RULE 13).
		al(&bs.sub, rows*max(qdim, p.NFFN)*4)
	}
	if p.AttnOutGate {
		// Double width: with the gate folded in, l.wq has 2*qdim rows and the
		// matvec writes all of them.
		al(&bs.q, rows*2*qdim*4)
		al(&bs.qg, rows*qdim*4)
		al(&bs.ogate, rows*qdim*4)
		al(&bs.ogated, rows*qdim*4)
	} else {
		al(&bs.q, rows*qdim*4)
	}
	al(&bs.k, rows*kvDim*4)
	al(&bs.v, rows*kvDim*4)
	if p.ClampKQV != 0 || p.Clamps {
		al(&bs.qClamp, rows*qdim*4)
		al(&bs.kClamp, rows*kvDim*4)
		al(&bs.vClamp, rows*kvDim*4)
	}
	if p.Clamps {
		wide := max(p.NEmbd, qdim, p.NFFN)
		al(&bs.cin, rows*wide*4)
		al(&bs.cout, rows*wide*4)
		al(&bs.gClamp, rows*p.NFFN*4)
		al(&bs.uClamp, rows*p.NFFN*4)
	}
	// All three attention kernels share the score row (scoreStride).
	bs.sstride = scoreStride(p, rows)
	// Paged history reads through FlashDecodeKV, which keeps no score plane:
	// the planes, and every kernel below that reads a contiguous cache, are
	// not built (pagedSkip).
	paged := g.pagedPlan(p)
	// A non-causal block's attention may run in query chunks
	// (visionChunk), so its score planes are sized for a chunk rather than
	// the whole run.
	bs.arows = rows
	if p.NonCausal {
		bs.arows, _ = visionChunk(rows, p.NHead, bs.sstride, bs.qtile, bs.atile, g.planeBudget())
	}
	if !paged {
		al(&bs.att, bs.arows*p.NHead*bs.sstride*4)
	}
	if p.AttnSinks {
		al(&bs.noSink, p.NHead*4)
		al(&bs.sinkProbe, p.NHead*4)
		if err == nil {
			inf := make([]float32, p.NHead)
			for i := range inf {
				inf[i] = float32(math.Inf(-1))
			}
			err = bs.noSink.Write(f32b(inf))
		}
	}
	if p.AttnSoftcap != 0 && !paged {
		al(&bs.attCap, bs.arows*p.NHead*bs.sstride*4)
	}
	if !paged {
		al(&bs.prob, bs.arows*p.NHead*bs.sstride*4)
	}
	al(&bs.xb, rows*qdim*4)
	// An ungated FFN (a vision block, C6) has no gate to write here; at a
	// 4096-patch image the plane is 70 MB nothing reads.
	if !ungatedFFN(p) {
		al(&bs.g, rows*p.NFFN*4)
	}
	al(&bs.u, rows*p.NFFN*4)
	al(&bs.act, rows*p.NFFN*4)
	al(&bs.cs, rows*p.RopeW()*4)
	if p.SWAPeriod > 0 {
		al(&bs.csSWA, rows*p.RopeW()*4)
	}
	if err == nil {
		err = g.initRopeTable(bs, p, rows)
	}
	// A linear block's scratch, sized from the plan's own geometry. al() is a
	// no-op at zero, so other architectures allocate nothing here.
	if r := cp.Recurrent; r.Conv > 0 {
		bs.rec = r
		if r.WithAttn {
			al(&bs.side, rows*p.NEmbd*4)
		}
		if r.ShortConv {
			for _, b := range []*backend.Buf{&bs.scB, &bs.scC, &bs.scBx, &bs.scY} {
				al(b, rows*r.Chans*4)
			}
		}
		al(&bs.dMixed, rows*r.Chans*4)
		al(&bs.dConv, rows*r.Chans*4)
		al(&bs.dZ, rows*r.VHeads*r.VDim*4)
		al(&bs.dOut, rows*r.VHeads*r.VDim*4)
		al(&bs.dBA, rows*2*r.VHeads*4)
		// The gate buffers are per channel on KDA and per head on qwen3next.
		nGate := r.VHeads
		if r.ChanDecay {
			nGate = r.VHeads * r.KDim
		}
		for _, b := range []*backend.Buf{&bs.dAlpha, &bs.dDecay} {
			al(b, rows*nGate*4)
		}
		for _, b := range []*backend.Buf{&bs.dBRaw, &bs.dBeta} {
			al(b, rows*r.VHeads*4)
		}
		if r.Mamba1 {
			// dt's bottleneck before and after its norm, its quantization (a
			// scale per 32 and a sum per 16, as bs.ax), and B and C before
			// and after theirs.
			al(&bs.dLowRank, rows*r.Rank*4)
			al(&bs.m1LowN, rows*r.Rank*4)
			al(&bs.dLRa, rows*pad32(r.Rank))
			al(&bs.dLRax, 3*rows*(pad32(r.Rank)/32)*4)
			for _, b := range []*backend.Buf{&bs.m1B, &bs.m1C, &bs.m1BN, &bs.m1CN} {
				al(b, rows*r.KDim*4)
			}
		}
		if r.ChanDecay {
			// The rank IS KDim: KDA's f_a_proj and g_a_proj both project to
			// linear_head_dim, which is the state's key extent.
			al(&bs.dLowRank, rows*r.KDim*4)
			// The scale plane is three floats per 32-element group (a scale per 32
			// and a sum per 16), as bs.ax is sized. An undersized one ran on CUDA
			// and failed on Vulkan.
			al(&bs.dLRa, rows*pad32(r.KDim))
			al(&bs.dLRax, 3*rows*(pad32(r.KDim)/32)*4)
			al(&bs.dGateWaste, rows*nGate*4)
		}
		// q and k ping-pong between two buffers each: every device kernel is out
		// of place, and slice -> norm -> scale needs only a swap per step.
		qkDim := r.KHeads * r.KDim
		for _, b := range []*backend.Buf{&bs.dQ, &bs.dQn, &bs.dK, &bs.dKn} {
			al(b, rows*qkDim*4)
		}
		for _, b := range []*backend.Buf{&bs.dV, &bs.dOutN, &bs.dAct} {
			al(b, rows*r.VHeads*r.VDim*4)
		}
		// 1/sqrt(kDim), written once: it is geometry, and mainkernels.Scale takes its
		// factor from a one-element buffer.
		al(&bs.dScale, 4)
		// The recurrent descriptor (mainkernels.RecDescWords): the real row count,
		// which Conv1dShift reads -- a padded chunk submits surplus zero rows
		// and a zero column shifted into the state is a column the model never
		// produced, and it stays there -- and each sequence's slots in the
		// block's pool; for a ragged step, the run descriptor its rows are
		// grouped by (runDescWords, the longer of the two).
		al(&bs.dRows, 4*runDescWords(rows))
	}
	// MLA's scratch, allocated only where the model has a latent block.
	if p.MLA() {
		lat, rot, row := bs.mlaLat, bs.mlaRot, bs.mlaRow
		// The norm reductions are over the latent and the query latent, not over
		// NEmbd, so they need their own partial counts: bs.parts is chosen to
		// divide NEmbd and a 16-wide latent is not divisible by 64.
		bs.mlaParts, bs.mlaQAParts = partsFor(lat), partsFor(p.QLoraRank)
		al(&bs.mlaKV, row*4)
		al(&bs.mlaKVn, lat*4)
		al(&bs.mlaKVPart, bs.mlaParts*4)
		al(&bs.mlaNope, mlaNopeW*4)
		al(&bs.mlaPE, p.NHead*rot*4)
		al(&bs.mlaPErot, p.NHead*rot*4)
		al(&bs.mlaKPE, rot*4)
		al(&bs.mlaAbsRaw, mlaLatW*4)
		al(&bs.mlaAbs, p.NHead*row*4)
		al(&bs.mlaAcc, mlaLatW*4)
		al(&bs.mlaOut, mlaOutW*4)
		al(&bs.mlaAbsOff, p.NHead*4)
		al(&bs.mlaPEOff, p.NHead*4)
		al(&bs.mlaPOff, 4)
		al(&bs.mlaIdent, p.NHead*4)
		if p.QLoraRank != 0 {
			al(&bs.mlaQA, p.QLoraRank*4)
			al(&bs.mlaQAn, p.QLoraRank*4)
			al(&bs.mlaQAPart, bs.mlaQAParts*4)
		}
	}
	al(&bs.n, (rows+1)*4)
	al(&bs.koff, rows*4)
	al(&bs.kpos, rows*4)
	al(&bs.roff, rows*4)
	al(&bs.zero, 4)
	var tempTab []float32
	if p.QKL2Norm {
		al(&bs.l4Ones, p.HeadDim*4)
	}
	if p.AttnTempScale != 0 {
		tempTab = l4TempTab(p)
		al(&bs.tempTab, len(tempTab)*4)
	}
	if p.ExpertWeightIn && p.NExpert > 0 {
		al(&bs.wOnes, p.NExpertUsed*4)
	}
	if residScaled(p) {
		al(&bs.rscale, 4)
	}
	// Gemma 4's v norm and output scalar: v normed out of place, against
	// ones, and the 1.0 the scaled residual is copied back with.
	if p.VNorm {
		al(&bs.vn, rows*p.NKVHead*p.HeadDim*4)
	}
	if p.VNorm || p.QKL2Norm {
		al(&bs.hdOnes, p.HeadDim*4)
	}
	if p.GeomSplit {
		al(&bs.one, 4)
	}
	allocAltUp(bs, p, rows, al)
	if p.PLEDim != 0 {
		al(&bs.pleIn, rows*p.PLEWidth*4)
		al(&bs.pleG, rows*p.PLEDim*4)
		al(&bs.pleS, rows*p.PLEDim*4)
		al(&bs.pleA, rows*p.PLEDim*4)
		al(&bs.pleO, rows*p.NEmbd*4)
	}
	if err != nil {
		g.LastErr = err.Error()
		return nil
	}
	// Llama 4's three constant planes and Granite's residual scale, written once
	// because they are geometry.
	for _, c := range []struct {
		b backend.Buf
		v []float32
	}{{bs.l4Ones, ones32(p.HeadDim)}, {bs.tempTab, tempTab}, {bs.wOnes, ones32(p.NExpertUsed)},
		{bs.rscale, []float32{float32(p.ResidualScale)}}, {bs.hdOnes, ones32(p.HeadDim)},
		{bs.one, []float32{1}}} {
		if c.b == nil {
			continue
		}
		if err = c.b.Write(f32b(c.v)); err != nil {
			g.LastErr = err.Error()
			return nil
		}
	}
	if bs.ident != nil {
		id := make([]uint32, p.NExpertUsed)
		for i := range id {
			id[i] = uint32(i)
		}
		if err = bs.ident.Write(u32b(id)); err != nil {
			g.LastErr = err.Error()
			return nil
		}
	}
	// MLA's three constant buffers, written once: the two destination lists
	// interleave each head's absorbed latent and rotated key into one row-wide
	// query, and the identity is the selection both absorb banks are indexed with.
	if bs.mlaIdent != nil {
		id := make([]uint32, p.NHead)
		abs := make([]uint32, p.NHead)
		pe := make([]uint32, p.NHead)
		for h := 0; h < p.NHead; h++ {
			id[h] = uint32(h)
			if g.kb.mlaFault == MLAFaultSheetOff {
				// Head h absorbs through head h+1's sheet: in bounds, finite,
				// and a different model. See MLAFault.
				id[h] = uint32((h + 1) % p.NHead)
			}
			abs[h] = uint32(h * bs.mlaRow)
			pe[h] = uint32(h*bs.mlaRow + bs.mlaLat)
		}
		for _, w := range []struct {
			b backend.Buf
			v []uint32
		}{{bs.mlaIdent, id}, {bs.mlaAbsOff, abs}, {bs.mlaPEOff, pe}} {
			if err = w.b.Write(u32b(w.v)); err != nil {
				g.LastErr = err.Error()
				return nil
			}
		}
	}
	if err = bs.zero.Write([]byte{0, 0, 0, 0}); err != nil {
		g.LastErr = err.Error()
		return nil
	}
	if bs.dScale != nil {
		sc := float32(1 / math.Sqrt(float64(cp.Recurrent.KDim)))
		if err = bs.dScale.Write(f32b([]float32{sc})); err != nil {
			g.LastErr = err.Error()
			return nil
		}
	}
	// The rotated q of row r lands at r*qdim, which never changes -- unlike the
	// K cache offsets, which move with the position. Written once here.
	roff := make([]byte, 0, rows*4)
	for r := 0; r < rows; r++ {
		roff = append(roff, u32one(uint32(r*qdim))...)
	}
	if err = bs.roff.Write(roff); err != nil {
		g.LastErr = err.Error()
		return nil
	}
	// XD-RoPE's row-major k (bs.krot): row r at r*kvDim.
	if bs.kroff != nil {
		ko := make([]byte, 0, rows*4)
		for r := 0; r < rows; r++ {
			ko = append(ko, u32one(uint32(r*p.KVRow()))...)
		}
		if err = bs.kroff.Write(ko); err != nil {
			g.LastErr = err.Error()
			return nil
		}
	}

	bs.qkLanes = g.pickLanes()
	bs.qkWidth = bs.qkLanes
	if bs.qkLanes == 1 {
		bs.qkWidth = scalarWidth
	}
	// The per-kernel lanes, set when each head norm is built; see qNormL.
	bs.qNormL, bs.kNormL, bs.qkNormL, bs.outNormL = bs.qkLanes, bs.qkLanes, bs.qkLanes, bs.qkLanes
	if paged {
		if err := g.initPagedScratch(bs); err != nil {
			g.LastErr = "paged attention: " + err.Error()
			freeScratch(bs, g.devShared)
			return nil
		}
	}
	bs.hx = make([]float32, rows*max(wide, p.ResidW()))
	bs.hcs = make([]float32, rows*p.RopeW())
	if p.SWAPeriod > 0 {
		bs.hcsSWA = make([]float32, rows*p.RopeW())
	}
	bs.hraw = make([]byte, rows*p.ResidW()*4)
	bs.hout = unsafe.Slice((*float32)(unsafe.Pointer(&bs.hraw[0])), rows*p.ResidW())
	if rows > 1 && p.NExpert > 0 {
		bs.mg = g.newMoeGroup(p, rows)
	}
	return bs
}

// pickLanes settles, once per device, whether the kernels whose arithmetic
// needs a 32-lane subgroup (the butterfly softmax, the per-head norm) may be
// used. It asks the device for a promise: CUDA's warp and Apple's simdgroup are
// 32 by architecture; a Vulkan device promises 32 when its min and max subgroup
// sizes are both 32 or it lets the width be pinned at pipeline creation. The
// reported default subgroupSize is not a promise (see backend.GuaranteedLanes).
//
// A refusal selects the thread-per-head twins, which run every model correctly,
// so this is a selection rather than a capability gate. WithLanes(1) forces the
// scalar twins for a paired measurement; WithLanes(32) forces the kernel but not
// the promise, so a refusing backend still declines the pipeline and the block
// runs on the host. Reintroducing the bug is vulkan subgroup "force", which
// scripts/three-way.sh --violate subgroup-width sets.
func (g *devTier) pickLanes() int {
	if g.Lanes != 0 {
		return g.Lanes
	}
	ok, why := backend.GuaranteedLanes(g.dev, ir.SubgroupLanes)
	g.Lanes, g.LanesWhy = 1, why
	if ok {
		g.Lanes, g.LanesWhy = ir.SubgroupLanes, ""
	}
	switch g.kb.lanes {
	case 1:
		g.Lanes, g.LanesWhy = 1, "WithLanes(1)"
	case ir.SubgroupLanes:
		g.Lanes, g.LanesWhy = ir.SubgroupLanes, "WithLanes(32)"
		if why != "" {
			g.LanesWhy += ", over the device's own answer: " + why
		}
	}
	return g.Lanes
}

// scalarWidth is the launch width of every thread-per-head kernel in the
// subgroup-width class: the softmax twin and the head-norm twin. It is a plain
// grid width with no cross-lane meaning at all, which is the whole point of
// those kernels.
const scalarWidth = 64

// headNormGeom is the launch of a head norm built at lanes over heads heads:
// one 32-lane group per head, or one thread per head in scalarWidth groups. It
// follows the kernel's lanes, not the device's: headLanes may drop a head to the
// scalar twin, and msl lowers NTID as the declared 64, so a 32-wide launch of
// it would leave half the heads unwritten.
func headNormGeom(lanes, heads int) (groups, width int) {
	if lanes == 1 {
		return (heads + scalarWidth - 1) / scalarWidth, scalarWidth
	}
	return heads, lanes
}

// The softmax is picked by correctness, not speed. A load-time timing probe
// runs at pos=1, where the choice does not matter; the warp reduction's gain
// grows with context. `verify -ab softmax` at a long -n is how it is checked.

// softmaxAgrees launches k on a synthetic score matrix and compares against a
// float64 softmax on the host.
//
// The scores span several units so that exp() separates them, and each head's
// maximum sits at a different position -- a reduction that sees only one lane's
// stride of the row picks the wrong maximum and normalises by the wrong sum,
// which shows up as an error of order 1, not of order 1e-6. n is deliberately
// not a multiple of 32: it exercises both the lanes that own two positions and
// the lanes that own none.
func (g *devTier) softmaxAgrees(bs *blockScratch, k backend.Kernel, groups, width int) bool {
	// Every n the first tokens use: at n=1 a single lane owns the whole row, a
	// different path through the reduction than a long row. 31, 32 and 33
	// straddle the warp boundary.
	for _, n := range []int{1, 2, 31, 32, 33, 37} {
		if n > bs.p.MaxSeq {
			continue
		}
		if !g.softmaxAgreesAt(bs, k, groups, width, n) {
			return false
		}
	}
	return true
}

func (g *devTier) softmaxAgreesAt(bs *blockScratch, k backend.Kernel, groups, width, n int) bool {
	p := &bs.p
	scores := make([]float32, p.NHead*p.MaxSeq)
	for h := 0; h < p.NHead; h++ {
		for t := 0; t < n; t++ {
			scores[h*p.MaxSeq+t] = float32(4 * math.Sin(0.7*float64(t)+float64(h)))
		}
	}
	got := make([]float32, p.NHead*p.MaxSeq)
	// A sink row that sits inside the scores' range, so it moves both the
	// maximum and the denominator.
	sinks := make([]float32, p.NHead)
	for h := range sinks {
		sinks[h] = float32(3 * math.Cos(float64(h)))
	}
	var err error
	g.dev.Session(func(s backend.Session) {
		if err = s.Write(bs.att, f32b(scores)); err != nil {
			return
		}
		if err = s.Write(bs.n, u32one(uint32(n))); err != nil {
			return
		}
		bufs := []backend.Buf{bs.att, bs.n, bs.prob}
		if p.AttnSinks {
			if err = s.Write(bs.sinkProbe, f32b(sinks)); err != nil {
				return
			}
			bufs = append(bufs, bs.sinkProbe)
		}
		if err = s.Launch(k, groups, width, bufs...); err != nil {
			return
		}
		err = s.Read(bs.prob, f32b(got))
	})
	if err != nil {
		g.LastErr = err.Error()
		return false
	}
	for h := 0; h < p.NHead; h++ {
		row := scores[h*p.MaxSeq : h*p.MaxSeq+n]
		mx := math.Inf(-1)
		if p.AttnSinks {
			mx = float64(sinks[h])
		}
		for _, v := range row {
			if float64(v) > mx {
				mx = float64(v)
			}
		}
		e := make([]float64, n)
		var sum float64
		if p.AttnSinks {
			sum = math.Exp(float64(sinks[h]) - mx)
		}
		for i, v := range row {
			e[i] = math.Exp(float64(v) - mx)
			sum += e[i]
		}
		for i := range e {
			// 1e-4 is loose against float32 and an approximate device exp, and
			// tight against every way a cross-lane reduction can be wrong.
			if math.Abs(float64(got[h*p.MaxSeq+i])-e[i]/sum) > 1e-4 {
				return false
			}
		}
	}
	return true
}

// headOut is the buffer the head's answer is read from: the capped logits
// where the model has a final softcap, the projection's own otherwise.
func (bs *blockScratch) headOut() backend.Buf {
	if bs.headCap != nil {
		return bs.logitsCap
	}
	return bs.logits
}

// ragOut is headOut for the per-row head: the rows' capped logits where the
// model has a final softcap, the projection's own otherwise.
func (bs *blockScratch) ragOut(g *Config) backend.Buf {
	if bs.ragCapK != nil && !g.rowsFaulted(rowsNocap) {
		return bs.ragCap
	}
	return bs.ragLogits
}

// HeadResident reports whether the vocabulary projection is on the device.
//
// It is not derivable from the block count, which is the reason it exists:
// placeHead gives blocks up to make room for the projection, so the run with
// the better placement is the one showing fewer blocks.
func (g *devTier) HeadResident() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.bs != nil && g.bs.head != nil && g.bs.head.ok
}

// headScratch gives a device that holds no block the scratch the output
// projection runs in, built from the plan of the device that holds the last
// block (see GPU.PrepHead). A device that already has one keeps it.
func (g *devTier) headScratch(p *nn.LayerPlan) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.bs != nil {
		return true
	}
	g.initKVCap(p)
	g.bs = g.initScratch(p, 1)
	g.shapeChanged()
	return g.bs != nil
}

// dropHeadScratch frees what PrepHead allocated around the projection -- the
// norm, its bias, the logits and the softcap -- and refunds it, leaving the
// projection's resident to whoever holds it. Callers hold g.mu.
func (g *devTier) dropHeadScratch(bs *blockScratch) {
	g.shapeChanged()
	// A head that was dropped (its resident failed) replaces the cap it
	// compiled, and must not leak the one before.
	if bs.headCap != nil {
		bs.headCap.Close()
		bs.headCap = nil
	}
	if bs.logitsCap != nil {
		bs.logitsCap.Free()
		bs.logitsCap = nil
		g.refund(uint64(bs.mvHead.rows * 4))
	}
	// So does a PrepHead after a seam gave its blocks back and grew again: the
	// norm, its bias and the logits are allocated there, and overwriting them
	// leaked all three (and their charge) on every round trip. bs.head is set
	// only where those charges were made.
	if bs.head != nil {
		n := uint64(bs.p.NEmbd*4 + bs.mvHead.rows*4)
		if bs.hNormB != nil {
			n += uint64(bs.p.NEmbd * 4)
		}
		g.refund(n)
	}
	for _, b := range []*backend.Buf{&bs.hNorm, &bs.hNormB, &bs.logits} {
		if *b != nil {
			(*b).Free()
			*b = nil
		}
	}
	bs.head = nil
	g.freeAltNorms()
}
