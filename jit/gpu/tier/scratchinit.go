package tier

import (
	"fmt"
	"math"
	"unsafe"

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
	k, err := kernels.RopeTable(npairs, rows)
	if err != nil {
		return why("%v", err)
	}
	c, err := g.compileShared(k)
	if err != nil {
		return why("%s declined the rotary table: %v", g.dev.API(), err)
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
		closeKern(c)
		return err
	}
	if p.SWAPeriod > 0 {
		if err := up(&bs.ropeTabSWA, p.RopeTabSWA); err != nil {
			closeKern(c)
			return err
		}
	}
	if err := up(&bs.ropeConst, kernels.RopeTabConsts()); err != nil {
		closeKern(c)
		return err
	}
	// (rows+1) words, the first being the valid row count as in bs.n, so a
	// ragged chunk's per-token staging is one small write (see kernels.RopeTable).
	b, err := g.dev.Alloc((rows + 1) * 4)
	if err != nil {
		closeKern(c)
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
			closeKern(c)
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
			freeScratch(bs)
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
		// A latent block on sm_70 takes both products on m8n8k4, and the
		// accumulate walks the softmax's row group, so the group is fixed
		// here, before the softmax is built.
		if p.MLA() && g.mla70Attn(p, rows) {
			bs.atile, bs.mla70 = 8*volta70AttnNT, true
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
	if bs.arows < rows {
		g.allocAttnChunks(bs, qdim, al)
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
		// 1/sqrt(kDim), written once: it is geometry, and kernels.Scale takes its
		// factor from a one-element buffer.
		al(&bs.dScale, 4)
		// The recurrent descriptor (kernels.RecDescWords): the real row count,
		// which Conv1dShift reads -- a padded chunk submits surplus zero rows
		// and a zero column shifted into the state is a column the model never
		// produced, and it stays there -- and each sequence's slots in the
		// block's pool; for a ragged step, the run descriptor its rows are
		// grouped by (kernels.RunDescWords, the longer of the two).
		al(&bs.dRows, 4*kernels.RunDescWords(rows))
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

	r := cp.Recurrent
	actWin := p.ActWin
	if actWin < 32 {
		actWin = 32
	}
	// The plan's score scale, as every paged kernel reads it: the contiguous
	// tiles below serve only a vision block, where the two agree.
	scale := g.scoreScale(p)
	gqa := p.NHead / p.NKVHead
	// Settle the subgroup width before any kernel is built: it decides which
	// variant of two kernels this device gets, and it is one answer for the
	// device rather than one per kernel.
	bs.qkLanes = g.pickLanes()
	bs.qkWidth = bs.qkLanes
	if bs.qkLanes == 1 {
		bs.qkWidth = scalarWidth
	}
	// HeadNorm's warp form also needs a head that is a whole number of
	// subgroups; otherwise the kernel build fails and surfaces as a misleading
	// budget decline. headLanes drops such a head to the scalar form (the gated
	// delta rule's KDim/VDim can be any width).
	headLanes := func(width int) int {
		if bs.qkLanes == ir.SubgroupLanes && width%ir.SubgroupLanes != 0 {
			return 1
		}
		return bs.qkLanes
	}
	// The per-kernel lanes, set when each head norm is built; see qNormL.
	bs.qNormL, bs.kNormL, bs.qkNormL, bs.outNormL = bs.qkLanes, bs.qkLanes, bs.qkLanes, bs.qkLanes
	type job struct {
		dst  *backend.Kernel
		make func() (*ir.Kernel, error)
	}
	jobs := []job{
		// Each quantizer's window is clamped to its own row, as the host
		// clamps it: a session window wider than a row (a per-token model's)
		// must not take the next row's maximum.
		{&bs.quantE, func() (*ir.Kernel, error) {
			if p.KPad > 0 {
				return kernels.QuantizePadded(rows, p.NEmbd, p.KPad, kernels.ClampWindow(actWin, p.KPad))
			}
			return kernels.Quantize(rows*p.NEmbd, kernels.ClampWindow(actWin, p.NEmbd))
		}},
		{&bs.quantF, func() (*ir.Kernel, error) {
			if p.NFFN == 0 {
				return nil, nil
			}
			return kernels.Quantize(rows*p.NFFN, kernels.ClampWindow(actWin, p.NFFN))
		}},
		{&bs.subQA, func() (*ir.Kernel, error) {
			if !p.SubNorm {
				return nil, nil
			}
			return g.rmsQuantRows()(qdim, rows, float32(p.RMSEps), false, false, kernels.ClampWindow(actWin, qdim))
		}},
		{&bs.subQF, func() (*ir.Kernel, error) {
			if !p.SubNorm {
				return nil, nil
			}
			return g.rmsQuantRows()(p.NFFN, rows, float32(p.RMSEps), false, false, kernels.ClampWindow(actWin, p.NFFN))
		}},
		{&bs.quantAct, func() (*ir.Kernel, error) {
			if ungatedFFN(p) || p.NFFN == 0 {
				return nil, nil
			}
			// Optional: an activation with no gated form keeps two launches.
			if k, err := kernels.QuantizeAct(rows*p.NFFN, kernels.ClampWindow(actWin, p.NFFN), p.Act); err == nil {
				return k, nil
			}
			return nil, nil
		}},
		// Quantize bakes its element count, so the gated delta output
		// (VHeads*VDim wide, not NFFN) and the attention output (nHead*headDim,
		// not NEmbd) each get their own quantizer. Borrowing a kernel of another
		// width quantizes the wrong number of elements and runs fluently wrong.
		{&bs.quantD, func() (*ir.Kernel, error) {
			r := cp.Recurrent
			if r.Conv == 0 {
				return nil, nil
			}
			return quantizeRows(rows, r.VHeads*r.VDim, kernels.ClampWindow(actWin, pad32(r.VHeads*r.VDim)))
		}},
		{&bs.quantQ, func() (*ir.Kernel, error) {
			if w := qPad(p, qdim); w != qdim {
				return kernels.QuantizePadded(rows, qdim, w, kernels.ClampWindow(actWin, w))
			}
			return kernels.Quantize(rows*qdim, kernels.ClampWindow(actWin, qdim))
		}},
		// The norm is two kernels (RMSNorm) or three (LayerNorm: mean, variance
		// about it, apply); see blockScratch.normVar.
		{&bs.normPart, func() (*ir.Kernel, error) {
			if lnBlock(p) {
				return kernels.LayerNormPartRows(p.NEmbd, bs.parts, rows)
			}
			return kernels.NormPartRows(p.NEmbd, bs.parts, rows)
		}},
		{&bs.rms, func() (*ir.Kernel, error) {
			if lnBlock(p) {
				return nil, nil
			}
			return g.rmsRows()(p.NEmbd, rows, float32(p.RMSEps), false, false)
		}},
		{&bs.addRms, func() (*ir.Kernel, error) {
			if lnBlock(p) {
				return nil, nil
			}
			return g.rmsRows()(p.NEmbd, rows, float32(p.RMSEps), false, true)
		}},
		{&bs.rmsQ, func() (*ir.Kernel, error) {
			if lnBlock(p) {
				return nil, nil
			}
			// Optional: a width the window does not tile keeps the two
			// launches rather than failing the scratch.
			if k, err := g.rmsQuantRows()(p.NEmbd, rows, float32(p.RMSEps), false, false, kernels.ClampWindow(actWin, p.NEmbd)); err == nil {
				return k, nil
			}
			return nil, nil
		}},
		{&bs.addRmsQ, func() (*ir.Kernel, error) {
			if lnBlock(p) {
				return nil, nil
			}
			// Optional: a width the window does not tile keeps the two
			// launches rather than failing the scratch.
			if k, err := g.rmsQuantRows()(p.NEmbd, rows, float32(p.RMSEps), false, true, kernels.ClampWindow(actWin, p.NEmbd)); err == nil {
				return k, nil
			}
			return nil, nil
		}},
		{&bs.copyRows, func() (*ir.Kernel, error) {
			// A block with no pre-norm copies the residual into h instead.
			if !p.NoPreNorm && p.Q4E == nil {
				return nil, nil
			}
			return kernels.Restride(rows, p.NEmbd, p.NEmbd, p.NEmbd)
		}},
		{&bs.normApply, func() (*ir.Kernel, error) {
			if lnBlock(p) {
				return kernels.LayerNormApplyRows(p.NEmbd, bs.parts, float32(p.RMSEps), true, rows)
			}
			return kernels.NormApplyRows(p.NEmbd, bs.parts, float32(p.RMSEps), false, rows)
		}},
		// RoPE is not generated at all for a vision block with no rotary (NRot
		// is zero), so it cannot be launched by mistake. A ViT with a 2-D
		// rotary (Qwen2-VL, Pixtral) is the text block's rotation over a table
		// the host built per patch from its grid coordinates (nn.Rope.TableAt):
		// the kernels do not know a position has two axes.
		{&bs.ropeQ, func() (*ir.Kernel, error) {
			// MLA rotates through its own pair (mlaRopeQ/mlaRopeK): its rotary
			// halves are the tail of each query head and one shared key.
			if (p.NonCausal && p.NRot == 0) || p.MLA() {
				return nil, nil
			}
			if p.RopeSplit {
				return kernels.RoPERowsSplit(p.NHead, p.HeadDim, p.NRot, rows, 0)
			}
			return kernels.RoPERows(p.NHead, p.HeadDim, p.NRot, p.RopeNeox, rows)
		}},
		{&bs.ropeK, func() (*ir.Kernel, error) {
			if (p.NonCausal && p.NRot == 0) || p.MLA() {
				return nil, nil
			}
			if p.RopeSplit {
				return nil, nil // ropeKRow
			}
			return kernels.RoPERowsT(p.NKVHead, p.HeadDim, p.NRot, p.RopeNeox, rows, g.kStride(p))
		}},
		{&bs.ropeKRow, func() (*ir.Kernel, error) {
			// XD-RoPE's k is normed after the rotary, so it turns row-major
			// into bs.krot and reaches the cache, contiguous or paged, through
			// copyK.
			if !p.RopeSplit {
				return nil, nil
			}
			return kernels.RoPERowsSplit(p.NKVHead, p.HeadDim, p.NRot, rows, 0)
		}},

		{&bs.normVar, func() (*ir.Kernel, error) {
			if !lnBlock(p) {
				return nil, nil
			}
			return kernels.LayerNormVarRows(p.NEmbd, bs.parts, rows)
		}},
		{&bs.copyK, func() (*ir.Kernel, error) {
			// Needed wherever k reaches the cache unrotated or partly rotated: a
			// ViT with no rotary, a partial rotary (RoPERows leaves the tail
			// unwritten), Llama 4's NoPE layers, and a model with no rotary at all.
			// A non-causal set may hold a block that rotates nothing beside
			// ones that do (an encoder's decision head; layer.noRope).
			if p.MLA() || (!p.NonCausal && !partialRotary(p) && !p.NoPEGlobal && !p.NoPosEnc &&
				!p.RopeSplit) {
				return nil, nil
			}
			if st := g.kStride(p); st > 0 {
				return kernels.CopyAtRowsT(kvDim, rows, st)
			}
			return kernels.CopyAtRows(kvDim, rows)
		}},
		{&bs.qTemp, func() (*ir.Kernel, error) {
			if p.AttnTempScale == 0 {
				return nil, nil
			}
			return kernels.ScaleRowsByCount(rows, qdim, int(p.AttnTempFloor),
				int(p.AttnTempOffset), len(l4TempTab(p)))
		}},
		{&bs.actMulW, func() (*ir.Kernel, error) {
			if !p.ExpertWeightIn || p.NExpert == 0 {
				return nil, nil
			}
			return kernels.ActMulWeighted(p.NExpertUsed*p.NFFNExp, p.NFFNExp, p.Act, false)
		}},
		{&bs.copyQ, func() (*ir.Kernel, error) {
			if p.MLA() || !partialRotary(p) {
				return nil, nil
			}
			return kernels.CopyAtRows(qdim, rows)
		}},
		{&bs.copyV, func() (*ir.Kernel, error) {
			// MLA writes no V at all: the value is the key row's own prefix.
			if p.MLA() {
				return nil, nil
			}
			if g.KVF16 {
				return kernels.CopyAtF16(kvDim)
			}
			return kernels.CopyAtRows(kvDim, rows)
		}},
		// MLA's own sequence, built from existing kernels: the absorb is the
		// indexed matvec, scores and accumulate are the ordinary two at two widths,
		// and the rest are slices, rotations and copies.
		{&bs.mlaSliceNope, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			// q's nope half is the head of each head and its rotary half the
			// tail, at a stride of HeadDim. Both are compacted, because the
			// absorb wants one contiguous activation per head at stride nope
			// and the rotation wants nRot pairs from a base.
			return kernels.SliceRows(bs.mlaNopeW, p.HeadDim, 0, p.NHead)
		}},
		{&bs.mlaSlicePE, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			return kernels.SliceRows(bs.mlaRot, p.HeadDim, bs.mlaNopeW, p.NHead)
		}},
		{&bs.mlaSliceKPE, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			// The key's rotary half is the tail of the kv_a output, one for the
			// whole layer (the "mqa" in kv_a_proj_with_mqa).
			return kernels.SliceRows(bs.mlaRot, bs.mlaRow, bs.mlaLat, 1)
		}},
		{&bs.mlaRopeQ, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			// NHead "heads" of exactly nRot, because the halves were compacted
			// above -- so the rotation is over the whole of each one.
			return kernels.RoPERows(p.NHead, bs.mlaRot, bs.mlaRot, p.RopeNeox, 1)
		}},
		{&bs.mlaRopeK, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			// It rotates straight into the cache at this position's row plus the
			// latent's width; RoPERows takes its destination offset from a buffer.
			return kernels.RoPERows(1, bs.mlaRot, bs.mlaRot, p.RopeNeox, 1)
		}},
		{&bs.mlaCopyLat, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			return kernels.CopyAt(bs.mlaLat)
		}},
		// A NoPE MLA model (Kimi-Linear) still writes both destinations the rope
		// kernels would, so these copies replace them rather than a skipped launch.
		{&bs.mlaNoPEQ, func() (*ir.Kernel, error) {
			if !p.MLA() || !p.NoPosEnc {
				return nil, nil
			}
			return kernels.CopyAt(p.NHead * bs.mlaRot)
		}},
		{&bs.mlaNoPEK, func() (*ir.Kernel, error) {
			if !p.MLA() || !p.NoPosEnc {
				return nil, nil
			}
			return kernels.CopyAt(bs.mlaRot)
		}},
		{&bs.mlaCopyAbs, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			return kernels.CopyAtRows(bs.mlaLat, p.NHead)
		}},
		{&bs.mlaCopyPE, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			return kernels.CopyAtRows(bs.mlaRot, p.NHead)
		}},
		{&bs.mlaQuantNope, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			return kernels.Quantize(p.NHead*bs.mlaNopeW, actWin)
		}},
		{&bs.mlaQuantAcc, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			return kernels.Quantize(p.NHead*bs.mlaLat, actWin)
		}},
		{&bs.mlaQuantOut, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			return kernels.Quantize(p.NHead*p.HeadDimV, actWin)
		}},
		{&bs.mlaNormKVPart, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			return kernels.NormPart(bs.mlaLat, bs.mlaParts)
		}},
		{&bs.mlaNormKVApply, func() (*ir.Kernel, error) {
			if !p.MLA() {
				return nil, nil
			}
			return kernels.NormApply(bs.mlaLat, bs.mlaParts, latentEps(p), false)
		}},
		{&bs.mlaQuantQA, func() (*ir.Kernel, error) {
			if !p.MLA() || p.QLoraRank == 0 {
				return nil, nil
			}
			return kernels.Quantize(p.QLoraRank, actWin)
		}},
		{&bs.mlaNormQAPart, func() (*ir.Kernel, error) {
			if !p.MLA() || p.QLoraRank == 0 {
				return nil, nil
			}
			return kernels.NormPart(p.QLoraRank, bs.mlaQAParts)
		}},
		{&bs.mlaNormQAApply, func() (*ir.Kernel, error) {
			if !p.MLA() || p.QLoraRank == 0 {
				return nil, nil
			}
			return kernels.NormApply(p.QLoraRank, bs.mlaQAParts, latentEps(p), false)
		}},
		{&bs.scores, func() (*ir.Kernel, error) {
			// MLA's scores are MQA over the whole cache row in the existing kernel's
			// terms: head width and cache stride are the row, and gqa is NHead so
			// h/gqa is 0 for every head (as engine/model/forward.go does). The scale is the
			// plan's, not 1/sqrt(row).
			if p.MLA() {
				// A decode row takes the warp-per-(head, position) kernel
				// where the device guarantees the width: the thread-per-pair
				// one walks the whole 576-wide row serially (see
				// kernels.AttnScoresWarp).
				if rows == 1 && g.pickLanes() == ir.SubgroupLanes {
					bs.scoresWarp = true
					return kernels.AttnScoresWarp(p.NHead, p.KVRow(), p.KVRow(), p.NHead,
						bs.sstride, g.scoreScale(p))
				}
				return kernels.AttnScoresTiled(p.NHead, p.KVRow(), p.KVRow(), p.NHead,
					bs.sstride, g.scoreScale(p), rows, bs.qtile, bs.ktile, 0)
			}
			// The matrix instruction where the shape allows it: q.k is a matrix
			// product and the transposed K cache stores B the way mma wants it. A
			// refusal falls through to the FMA tile.
			if kst := g.kStride(p); bs.arows > 1 && kst > 0 && !g.NoMMA && !g.kb.noAttnMMA &&
				p.HeadDim%16 == 0 && bs.arows%16 == 0 {
				k, err := kernels.AttnScoresMMA(p.NHead, p.HeadDim, kvDim, gqa, bs.sstride,
					scale, bs.arows, kst, g.tiles.attnNT)
				if err == nil {
					if c, e := g.dev.Compile(k); e == nil {
						bs.scoresMMA, g.AttnMMA = c, true
						return nil, nil
					}
				}
			}
			// sm_70's m8n8k4 where m16n8k16 does not lower, for both products.
			// Not under query chunks: its kernels are built for the whole image.
			if bs.arows == rows && g.volta70Attn(bs, kvDim, gqa, scale) {
				return nil, nil
			}
			// The one-kernel form on Metal's matrix unit where it takes the
			// shape; the FMA tiles below stay built as everything else's path.
			g.flashTileAttn(bs, kvDim, gqa, scale)
			return kernels.AttnScoresTiled(p.NHead, p.HeadDim, kvDim, gqa, bs.sstride, scale,
				bs.arows, bs.qtile, bs.ktile, g.kStride(p))
		}},
		{&bs.scoresW, func() (*ir.Kernel, error) {
			// Built after bs.scores so it takes the same path: a model whose
			// global layers run the matrix instruction runs its local ones on
			// it too, and a device that refused it refuses both.
			if p.SWAWindow <= 0 {
				return nil, nil
			}
			if bs.scoresMMA != nil {
				k, err := kernels.AttnScoresMMAW(p.NHead, p.HeadDim, kvDim, gqa, bs.sstride,
					scale, bs.arows, g.kStride(p), g.tiles.attnNT, windowArg(p))
				if err == nil {
					if c, e := g.dev.Compile(k); e == nil {
						bs.scoresMMAW = c
						return nil, nil
					}
				}
				// A refusal falls through to the windowed FMA tile, which the
				// launch takes whenever scoresMMAW is nil.
			}
			return kernels.AttnScoresTiledW(p.NHead, p.HeadDim, kvDim, gqa, bs.sstride, scale,
				bs.arows, bs.qtile, bs.ktile, g.kStride(p), windowArg(p))
		}},
		{&bs.attnAcc, func() (*ir.Kernel, error) {
			// MLA accumulates at the latent's width (what it produces) over a
			// row-wide cache (the stride it walks), so the value is the key row's
			// prefix. No split: the split's partial plane is sized from qdim.
			if p.MLA() {
				bs.accSplit = 1
				// gqa is NHead so every head reads the row's own prefix; the fault arm
				// makes it head-major instead.
				gqa := p.NHead
				if g.kb.mlaFault == MLAFaultValueHeadMajor {
					gqa = 1
				}
				return kernels.AttnAcc(p.NHead, p.KVLoraRank, p.KVRow(), gqa, p.MaxSeq)
			}
			// The position split is a constant, not a tuned parameter: each thread's
			// trip count is ceil((n - s)/S), zero for every split past n, so one S is
			// correct at every depth. It exists because the unsplit kernel is a long
			// loop-carried chain over too few threads to fill the card at depth. S is
			// capped so the grid does not overshoot the device, and never above 32.
			if rows > 1 {
				bs.accSplit = 1
				return kernels.AttnAccTiled(p.NHead, p.HeadDim, kvDim, gqa, bs.sstride, bs.arows, bs.atile)
			}
			bs.accSplit = accSplitFor(qdim, g.dev.Slots(), g.kb.mvAccSplit)
			// WithAttnAccSplit pins it, to tell a real difference from f32
			// reassociation.
			if k := g.kb.accSplit; k >= 1 {
				bs.accSplit = k
			}
			if g.KVF16 {
				// The packed kernel handles split 1 too, so there is no
				// unsplit variant to fall back to here.
				return kernels.AttnAccSplitF16(p.NHead, p.HeadDim, kvDim, gqa, p.MaxSeq, bs.accSplit)
			}
			if bs.accSplit < 2 {
				return kernels.AttnAcc(p.NHead, p.HeadDim, kvDim, gqa, p.MaxSeq)
			}
			return kernels.AttnAccSplit(p.NHead, p.HeadDim, kvDim, gqa, p.MaxSeq, bs.accSplit)
		}},
		{&bs.attnRed, func() (*ir.Kernel, error) {
			if bs.accSplit < 2 {
				return nil, nil // no split, no reduction
			}
			return kernels.Reduce(qdim, bs.accSplit)
		}},
		{&bs.shRouterK, func() (*ir.Kernel, error) {
			if p.NFFNShExp == 0 {
				return nil, nil
			}
			return kernels.SharedRouterLogits(rows, p.NEmbd)
		}},
		{&bs.shActMul, func() (*ir.Kernel, error) {
			if p.NFFNShExp == 0 {
				return nil, nil
			}
			if ungatedFFN(p) {
				return kernels.Act(rows*p.NFFNShExp, p.Act)
			}
			return kernels.ActMul(rows*p.NFFNShExp, p.Act)
		}},
		{&bs.shQuant, func() (*ir.Kernel, error) {
			if p.NFFNShExp == 0 {
				return nil, nil
			}
			return kernels.Quantize(rows*p.NFFNShExp, actWin)
		}},
		{&bs.shAdd, func() (*ir.Kernel, error) {
			if p.NFFNShExp == 0 {
				return nil, nil
			}
			return kernels.SharedExpertAdd(p.NEmbd, rows)
		}},
		{&bs.splitQG, func() (*ir.Kernel, error) {
			if !p.AttnOutGate {
				return nil, nil
			}
			return kernels.SplitHeadGate(p.NHead, p.HeadDim, rows)
		}},
		{&bs.softcap, func() (*ir.Kernel, error) {
			if p.AttnSoftcap == 0 {
				return nil, nil
			}
			return kernels.Softcap(bs.arows*p.NHead*bs.sstride, p.AttnSoftcap)
		}},
		{&bs.clampQ, func() (*ir.Kernel, error) {
			if p.ClampKQV == 0 {
				return nil, nil
			}
			return kernels.Clamp(rows*qdim, p.ClampKQV)
		}},
		{&bs.clampKV, func() (*ir.Kernel, error) {
			if p.ClampKQV == 0 {
				return nil, nil
			}
			return kernels.Clamp(rows*kvDim, p.ClampKQV)
		}},
		{&bs.sigMul, func() (*ir.Kernel, error) {
			if !p.AttnOutGate {
				return nil, nil
			}
			return kernels.SigmoidMul(rows * qdim)
		}},
		{&bs.qkPart, func() (*ir.Kernel, error) {
			if !p.QKNorm || !p.QKNormWide || p.NKVHead != p.NHead || (p.NHead*p.HeadDim)%bs.parts != 0 {
				return nil, nil
			}
			return kernels.NormPartRows(p.NHead*p.HeadDim, bs.parts, rows)
		}},
		{&bs.qkApply, func() (*ir.Kernel, error) {
			if !p.QKNorm || !p.QKNormWide || p.NKVHead != p.NHead || (p.NHead*p.HeadDim)%bs.parts != 0 {
				return nil, nil
			}
			return kernels.NormApplyRows(p.NHead*p.HeadDim, bs.parts, float32(p.RMSEps), false, rows)
		}},
		{&bs.qkRms, func() (*ir.Kernel, error) {
			if !p.QKNorm || !p.QKNormWide || p.NKVHead != p.NHead || g.NoQKRms {
				return nil, nil
			}
			return g.rmsRows()(p.NHead*p.HeadDim, rows, float32(p.RMSEps), false, false)
		}},
		// A whole-projection norm (olmoe) is HeadNormRows with one group of the
		// whole width, each of q and k its own projection's.
		{&bs.qNorm, func() (*ir.Kernel, error) {
			if !p.QKNorm && !p.QKL2Norm {
				return nil, nil
			}
			if p.QKNorm && p.QKNormWide {
				w := p.NHead * p.HeadDim
				bs.qNormL = headLanes(w)
				return kernels.HeadNormRows(1, w, p.RMSEps, rows, bs.qNormL)
			}
			bs.qNormL = headLanes(p.HeadDim)
			return kernels.HeadNormRows(p.NHead, p.HeadDim, p.RMSEps, rows, bs.qNormL)
		}},
		{&bs.kNorm, func() (*ir.Kernel, error) {
			if !p.QKNorm && !p.QKL2Norm {
				return nil, nil
			}
			if p.QKNorm && p.QKNormWide {
				w := p.NKVHead * p.HeadDim
				bs.kNormL = headLanes(w)
				return kernels.HeadNormRows(1, w, p.RMSEps, rows, bs.kNormL)
			}
			bs.kNormL = headLanes(p.HeadDim)
			return kernels.HeadNormRows(p.NKVHead, p.HeadDim, p.RMSEps, rows, bs.kNormL)
		}},
		{&bs.actMul, func() (*ir.Kernel, error) {
			if ungatedFFN(p) {
				// Ungated: the activation alone replaces ActMul's product, which
				// would read a bs.g nothing wrote. xIELU's numbers are each
				// block's buffer (layer.xielu).
				if p.Act == kernels.ActXIELU {
					return kernels.XIELU(rows * p.NFFN)
				}
				return kernels.Act(rows*p.NFFN, p.Act)
			}
			if p.NFFN == 0 {
				return nil, nil
			}
			return kernels.ActMul(rows*p.NFFN, p.Act)
		}},
		// A second activation kernel because the element count is baked and a
		// DeepSeek has both shapes in one model (dense lead blocks, then
		// mixtures), while the scratch is one per tier.
		{&bs.actMulE, func() (*ir.Kernel, error) {
			if p.NExpert == 0 {
				return nil, nil
			}
			// One SwiGLU over all the slots at once: they are contiguous
			// and elementwise, so the slot boundary does not matter. Ungated
			// experts (Nemotron 3) take the activation alone.
			if ungatedFFN(p) {
				return kernels.Act(p.NExpertUsed*p.NFFNExp, p.Act)
			}
			return kernels.ActMul(p.NExpertUsed*p.NFFNExp, p.Act)
		}},
		{&bs.rank, func() (*ir.Kernel, error) {
			if p.NExpert == 0 {
				return nil, nil
			}
			return kernels.ExpertRank(routeOf(p))
		}},
		{&bs.weights, func() (*ir.Kernel, error) {
			if p.NExpert == 0 {
				return nil, nil
			}
			return kernels.ExpertWeights(routeOf(p))
		}},
		{&bs.pleSlice, func() (*ir.Kernel, error) {
			if p.PLEDim == 0 {
				return nil, nil
			}
			return kernels.PLESlice(rows, p.PLEWidth, p.PLEDim)
		}},
		{&bs.pleAct, func() (*ir.Kernel, error) {
			if p.PLEDim == 0 {
				return nil, nil
			}
			return kernels.ActMul(rows*p.PLEDim, p.Act)
		}},
		{&bs.pleQuant, func() (*ir.Kernel, error) {
			if p.PLEDim == 0 {
				return nil, nil
			}
			return quantizeRows(rows, p.PLEDim, actWin)
		}},
		{&bs.ewScale, func() (*ir.Kernel, error) {
			if !p.DenseMoE {
				return nil, nil
			}
			return kernels.ExpertWeightScale(1, p.NExpertUsed)
		}},
		{&bs.route, func() (*ir.Kernel, error) {
			r := routeOf(p)
			// The fused route renormalises or softmaxes all; sparsemixer's weights
			// are their own kernel (kernels.ExpertWeights).
			if p.NExpert == 0 || r.Gated() || r.Bias || r.Grouped() || r.SparseMixer != 0 ||
				kernels.ExpertRouteWidth(p.NExpert) == 0 {
				return nil, nil
			}
			// One warp for the renormalised route where the device has one: the
			// thread-per-expert form walks every logit. Same selection and weights
			// (kernels.ExpertRouteWarp).
			if r.Norm && bs.qkLanes == ir.SubgroupLanes {
				if k, err := kernels.ExpertRouteWarp(r); err == nil {
					bs.routeWidth = ir.SubgroupLanes
					return k, nil
				}
			}
			bs.routeWidth = kernels.ExpertRouteWidth(p.NExpert)
			return kernels.ExpertRoute(r)
		}},
		{&bs.gscore, func() (*ir.Kernel, error) {
			if p.NExpert == 0 || p.ExpertGroups <= 1 {
				return nil, nil
			}
			return kernels.ExpertGroupScore(routeOf(p))
		}},
		{&bs.gmask, func() (*ir.Kernel, error) {
			if p.NExpert == 0 || p.ExpertGroups <= 1 {
				return nil, nil
			}
			return kernels.ExpertGroupMask(routeOf(p))
		}},
		{&bs.rbias, func() (*ir.Kernel, error) {
			if p.NExpert == 0 || !p.MoEBias {
				return nil, nil
			}
			return kernels.Add(p.NExpert)
		}},
		{&bs.ebiasFF, func() (*ir.Kernel, error) {
			if p.NExpert == 0 || !p.MoEBias {
				return nil, nil
			}
			return kernels.IndexedBiasAdd(p.NExpertUsed, p.NFFNExp)
		}},
		{&bs.ebiasD, func() (*ir.Kernel, error) {
			if p.NExpert == 0 || !p.MoEBias {
				return nil, nil
			}
			return kernels.IndexedBiasAdd(p.NExpertUsed, p.NEmbd)
		}},
		{&bs.combine, func() (*ir.Kernel, error) {
			if p.NExpert == 0 {
				return nil, nil
			}
			return kernels.ExpertCombine(p.ExpWidth(), p.NExpertUsed)
		}},
		{&bs.quantX, func() (*ir.Kernel, error) {
			if p.NExpert == 0 {
				return nil, nil
			}
			// The SwiGLU product, quantized per slot: Slots independent
			// vectors of NFFNExp, laid out contiguously, which is exactly what
			// MatVecShape.SlotAct then reads.
			return kernels.Quantize(p.NExpertUsed*p.NFFNExp, actWin)
		}},
		{&bs.addE, func() (*ir.Kernel, error) { return kernels.Add(rows * p.NEmbd) }},
		{&bs.addS, func() (*ir.Kernel, error) {
			if !residScaled(p) {
				return nil, nil
			}
			return kernels.AddScaled(rows * p.NEmbd)
		}},
		{&bs.scaleX, func() (*ir.Kernel, error) {
			if !p.GeomSplit {
				return nil, nil
			}
			return kernels.Scale(rows * p.NEmbd)
		}},
		// Mamba-1's: the convolution's bias and SiLU, dt's quantization, the
		// three norms (dt over the rank, B and C over the state), and the
		// whole scan in one launch.
		{&bs.m1Bias, func() (*ir.Kernel, error) {
			if r.Conv == 0 || !r.Mamba1 {
				return nil, nil
			}
			return kernels.BiasAct(r.Chans, rows, kernels.ActSiLU)
		}},
		{&bs.m1Quant, func() (*ir.Kernel, error) {
			if r.Conv == 0 || !r.Mamba1 {
				return nil, nil
			}
			return quantizeRows(rows, r.Rank, actWin)
		}},
		{&bs.m1NormDt, func() (*ir.Kernel, error) {
			if r.Conv == 0 || !r.Mamba1 {
				return nil, nil
			}
			bs.m1NormDtL = headLanes(r.Rank)
			return kernels.HeadNormRows(1, r.Rank, p.RMSEps, rows, bs.m1NormDtL)
		}},
		{&bs.m1NormBC, func() (*ir.Kernel, error) {
			if r.Conv == 0 || !r.Mamba1 {
				return nil, nil
			}
			bs.m1NormBCL = headLanes(r.KDim)
			return kernels.HeadNormRows(1, r.KDim, p.RMSEps, rows, bs.m1NormBCL)
		}},
		{&bs.m1Scan, func() (*ir.Kernel, error) {
			if r.Conv == 0 || !r.Mamba1 {
				return nil, nil
			}
			d := deltaScanOf(r, rows, bs.qkLanes)
			bs.scanThreads = d.Threads()
			return kernels.GatedDeltaFused(d, kernels.DeltaFuse{Mamba1: true})
		}},
		// Mamba-2's: the whole update in one launch, its grouped norm, and the
		// copy a mixer-only block ends with.
		{&bs.ssdScan, func() (*ir.Kernel, error) {
			if r.Conv == 0 || !r.SSD {
				return nil, nil
			}
			d := deltaScanOf(r, rows, bs.qkLanes)
			bs.scanThreads = d.Threads()
			return kernels.GatedDeltaFused(d, kernels.DeltaFuse{Chans: r.Chans, SSD: true})
		}},
		{&bs.ssdNorm, func() (*ir.Kernel, error) {
			if r.Conv == 0 || !r.SSD || r.NoNorm {
				return nil, nil
			}
			w := r.VHeads * r.VDim / r.NormGroups
			bs.ssdNormL = headLanes(w)
			return kernels.GroupNormRows(r.NormGroups, w, p.RMSEps, rows, bs.ssdNormL)
		}},
		{&bs.copyRes, func() (*ir.Kernel, error) {
			if !r.SSD && !r.Mamba1 {
				return nil, nil
			}
			return kernels.Restride(rows, p.NEmbd, p.NEmbd, p.NEmbd)
		}},
		// The linear block's kernels, each nil when the model has no such block.
		{&bs.convRows, func() (*ir.Kernel, error) {
			if r.Conv == 0 {
				return nil, nil
			}
			return kernels.Conv1dRows(r.Conv, r.Chans, rows)
		}},
		{&bs.scMul, func() (*ir.Kernel, error) {
			if !r.ShortConv {
				return nil, nil
			}
			return kernels.ActMul(rows*r.Chans, kernels.ActIdentity)
		}},
		{&bs.convCentred, func() (*ir.Kernel, error) {
			if !r.Centred {
				return nil, nil
			}
			return kernels.Conv3Centred(r.Chans, rows)
		}},
		{&bs.convShift, func() (*ir.Kernel, error) {
			if r.Conv == 0 {
				return nil, nil
			}
			return kernels.Conv1dShift(r.Conv, r.Chans, rows)
		}},
		{&bs.ssmSiLU, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 {
				return nil, nil
			}
			return kernels.Act(rows*r.Chans, kernels.ActSiLU)
		}},
		{&bs.splitGates, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 {
				return nil, nil
			}
			return kernels.SplitDeltaGatesRows(r.KHeads, r.VHeads/r.KHeads, rows)
		}},
		{&bs.deltaGate, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 {
				return nil, nil
			}
			return kernels.DeltaGateRows(r.VHeads, rows)
		}},
		// KDA's gate kernels: two launches because decay runs over VHeads*KDim
		// channels and beta over VHeads heads. DeltaGate computes both, so each
		// launch keeps the half it wants and sends the other to dGateWaste.
		{&bs.deltaGateChan, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || !r.ChanDecay {
				return nil, nil
			}
			if r.DecayBound != 0 {
				// Kimi-K3's lower-bounded decay, the same params.
				return kernels.DeltaDecayBoundRows(r.VHeads*r.KDim, rows, r.DecayBound)
			}
			return kernels.DeltaGateRows(r.VHeads*r.KDim, rows)
		}},
		{&bs.quantLR, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || !r.ChanDecay {
				return nil, nil
			}
			return quantizeRows(rows, r.KDim, actWin)
		}},
		{&bs.deltaStepChan, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || !r.ChanDecay {
				return nil, nil
			}
			return kernels.GatedDeltaStepChan(r.VHeads, r.VDim, r.KDim, r.VHeads/r.KHeads)
		}},
		{&bs.outSigMul, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || !(r.ChanDecay || r.SigmoidGate) {
				return nil, nil
			}
			// SigmoidMul, not ActMul: KimiLinearRMSNormGated applies out*sigma(z)
			// where qwen3next applies out*SiLU(z), which differ by a factor of z.
			return kernels.SigmoidMul(rows * r.VHeads * r.VDim)
		}},
		{&bs.sliceQ, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 {
				return nil, nil
			}
			return kernels.SliceRows(r.KHeads*r.KDim, r.Chans, 0, rows)
		}},
		{&bs.sliceK, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 {
				return nil, nil
			}
			return kernels.SliceRows(r.KHeads*r.KDim, r.Chans, r.KHeads*r.KDim, rows)
		}},
		{&bs.sliceV, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 {
				return nil, nil
			}
			return kernels.SliceRows(r.VHeads*r.VDim, r.Chans, 2*r.KHeads*r.KDim, rows)
		}},
		{&bs.qkNorm, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 {
				return nil, nil
			}
			// The L2 norm is an RMSNorm with a unit weight and eps/n, one launch per
			// operand. KDA's l2norm hardcodes eps 1e-6 added to the sum (not
			// rms_norm_eps); dividing by KDim turns it into the mean form.
			// engine/model/delta.go's kdaL2Eps is the same number.
			eps := p.RMSEps
			if r.ChanDecay {
				eps = 1e-6
			}
			bs.qkNormL = headLanes(r.KDim)
			return kernels.HeadNormRows(r.KHeads, r.KDim, eps/float64(r.KDim),
				rows, bs.qkNormL)
		}},
		{&bs.qkScale, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 {
				return nil, nil
			}
			return kernels.Scale(rows * r.KHeads * r.KDim)
		}},
		{&bs.deltaStep, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 {
				return nil, nil
			}
			if r.KeyTiled && r.TailLen > 0 {
				return kernels.GatedDeltaStepTiledStride(r.VHeads, r.VDim, r.KDim, r.KHeads, r.StateLen)
			}
			if r.KeyTiled {
				return kernels.GatedDeltaStepTiled(r.VHeads, r.VDim, r.KDim, r.KHeads)
			}
			return kernels.GatedDeltaStep(r.VHeads, r.VDim, r.KDim, r.VHeads/r.KHeads)
		}},
		// The chunk's delta rule: GatedDeltaStep steps one token, so a batched
		// submission needs the scan. Decode takes it too where the device has a
		// subgroup: the step is a thread per state row, so warp loads are
		// uncoalesced, while the scan puts L lanes on a row. Without a subgroup the
		// scan's one-thread form holds the whole row in registers, so decode keeps
		// the step.
		{&bs.deltaFused, func() (*ir.Kernel, error) {
			// Decode only: the fused kernel forms q, k, v and the gates once per
			// state row inside the per-token loop. At one row that beats eleven
			// launches; over a chunk it is serial work on the scan's critical path
			// and measured slower.
			if r.Conv == 0 || r.ShortConv || r.Mamba1 || rows != 1 || bs.qkLanes != ir.SubgroupLanes {
				return nil, nil
			}
			d := deltaScanOf(r, rows, bs.qkLanes)
			if d.ScanLanes() == 1 {
				return nil, nil
			}
			// The two constants the unfused chain bakes: qkNorm's epsilon over
			// KDim (HeadNormRows' own argument, KDA's hardcoded 1e-6) and the
			// 1/sqrt(KDim) dScale holds.
			eps := p.RMSEps
			if r.ChanDecay {
				eps = 1e-6
			}
			bs.scanThreads = d.Threads()
			return kernels.GatedDeltaFused(d, kernels.DeltaFuse{Chans: r.Chans,
				Eps: float32(eps / float64(r.KDim)), Scale: float32(1 / math.Sqrt(float64(r.KDim))),
				BARep: r.VHeads / r.KHeads, Bound: r.DecayBound})
		}},
		{&bs.deltaScan, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 || bs.deltaFused != nil {
				return nil, nil
			}
			d := deltaScanOf(r, rows, bs.qkLanes)
			if rows == 1 && d.ScanLanes() == 1 {
				return nil, nil
			}
			bs.scanThreads = d.Threads()
			return kernels.GatedDeltaScan(d)
		}},
		{&bs.outNorm, func() (*ir.Kernel, error) {
			if r.Conv == 0 || r.ShortConv || r.Mamba1 {
				return nil, nil
			}
			bs.outNormL = headLanes(r.VDim)
			return kernels.HeadNormRows(r.VHeads, r.VDim, p.RMSEps, rows, bs.outNormL)
		}},
		{&bs.outActMul, func() (*ir.Kernel, error) {
			if r.Conv == 0 {
				return nil, nil
			}
			return kernels.ActMul(rows*r.VHeads*r.VDim, kernels.ActSiLU)
		}},
		{&bs.outNormGate, func() (*ir.Kernel, error) {
			// graph.NormGate's recipe: the output norm and its gate in one
			// launch, at outNorm's lanes. Built wherever outNorm is; each
			// block's plan decides whether it runs (prepNormGate).
			if r.Conv == 0 || r.ShortConv || r.Mamba1 {
				return nil, nil
			}
			gate := kernels.GateSiLU
			if r.ChanDecay || r.SigmoidGate {
				gate = kernels.GateSigmoid
			}
			return kernels.HeadNormGated(r.VHeads, r.VDim, p.RMSEps, rows, headLanes(r.VDim), gate)
		}},
	}
	for _, j := range altJobs(bs, p, rows, actWin) {
		jobs = append(jobs, job{j.dst, j.make})
	}
	// Gemma 4's clipped linears: matrix s's input clamp and output clamp at
	// its widths, rows of each.
	if p.Clamps {
		in := [7]int{p.NEmbd, p.NEmbd, p.NEmbd, qdim, p.NEmbd, p.NEmbd, p.NFFN}
		out := [7]int{qdim, kvDim, kvDim, p.NEmbd, p.NFFN, p.NFFN, p.NEmbd}
		for s := range 7 {
			ni, no, off := rows*in[s], rows*out[s], 4*s
			jobs = append(jobs,
				job{&bs.clampK[2*s], func() (*ir.Kernel, error) { return kernels.ClampAt(ni, off) }},
				job{&bs.clampK[2*s+1], func() (*ir.Kernel, error) { return kernels.ClampAt(no, off+2) }})
		}
	}
	// The contiguous cache's writers and readers, which paged history
	// replaces (initPagedScratch).
	pagedSkip := map[*backend.Kernel]bool{&bs.scores: true, &bs.scoresW: true, &bs.attnAcc: true,
		&bs.attnRed: true, &bs.softcap: true, &bs.copyK: true, &bs.ropeK: true, &bs.copyV: true}
	for _, j := range jobs {
		if paged && pagedSkip[j.dst] {
			continue
		}
		k, e := j.make()
		if e != nil {
			g.LastErr = e.Error()
			return nil
		}
		if k == nil {
			continue // an optional kernel this architecture does not need
		}
		c, e := g.compileShared(k)
		if e != nil {
			g.LastErr = e.Error()
			return nil
		}
		*j.dst = c
	}
	if paged {
		if err := g.initPagedScratch(bs); err != nil {
			g.LastErr = "paged attention: " + err.Error()
			freeScratch(bs)
			return nil
		}
		if err := g.initDS4(bs); err != nil {
			g.LastErr = "DeepSeek V4: " + err.Error()
			freeScratch(bs)
			return nil
		}
	} else {
		if !g.pickSoftmax(bs) {
			return nil
		}
		if err := g.initFlash(bs); err != nil {
			// Compilation failure retains the staged implementation.
			g.LastErr = err.Error()
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
	if err := g.initK3(bs); err != nil {
		g.LastErr = "Kimi-K3: " + err.Error()
		freeScratch(bs)
		return nil
	}
	if err := g.initQ4E(bs); err != nil {
		g.LastErr = "Qwen3.8-Flash-Next: " + err.Error()
		freeScratch(bs)
		return nil
	}
	if rows > 1 && p.MLA() {
		bs.mlab = g.newMLABatch(bs, p, rows)
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

// pickSoftmax takes the warp-per-head softmax when this device guarantees a
// 32-lane subgroup and the kernel then agrees with the host, and the
// one-thread-per-head kernel otherwise.
//
// It asks first (pickLanes) and probes second. The probe alone is not enough:
// an out-of-subgroup shuffle is undefined, and a device that picks its width
// per shader can pass on an idle device and produce NaN in a real submission
// (see docs/engineering-history/gpu-kernels.md). The probe stays because it
// catches a driver that promised a width and did not deliver, a shuffle
// lowered wrongly, and a deterministically narrow device. A device that
// refuses to compile the shuffle lands in the same fallback.
func (g *devTier) pickSoftmax(bs *blockScratch) bool {
	p := &bs.p
	// Every row of a chunk is an independent set of score rows, so a batched
	// softmax is simply nHeads*rows of them -- the kernel already takes the row
	// count as its first argument and needs no other change.
	heads := p.NHead * bs.arows
	build := func(lanes int) backend.Kernel {
		k, err := kernels.SoftmaxRowsSink(p.NHead, bs.sstride, lanes, bs.arows, bs.atile, p.AttnSinks)
		if err != nil {
			g.LastErr = err.Error()
			return nil
		}
		c, err := g.compileShared(k)
		if err != nil {
			g.LastErr = err.Error()
			return nil
		}
		return c
	}
	scalarGroups := (heads + scalarWidth - 1) / scalarWidth

	// The batched scratch inherits the decision rather than re-probing: the
	// subgroup width is a property of the device, and the rows==1 scratch is
	// always built first, so g.SoftmaxLanes is settled.
	if bs.rows > 1 {
		lanes, groups, width := g.SoftmaxLanes, heads, ir.SubgroupLanes
		if lanes != ir.SubgroupLanes {
			lanes, groups, width = 1, scalarGroups, scalarWidth
		}
		c := build(lanes)
		if c == nil {
			return false
		}
		bs.softmax, bs.smGroups, bs.smWidth = c, groups, width
		return true
	}

	// The wide kernel is the default on every device that guarantees the width;
	// WithLanes(1) forces the one-thread kernel. (An apparent Metal bug here was
	// the harness comparing two free-running greedy chains; diff at a fixed input,
	// see docs/engineering-history/gpu-kernels.md.)
	warpOK := g.pickLanes() == ir.SubgroupLanes
	g.SoftmaxWhy = "the warp reduction did not agree with the host"
	if !warpOK {
		g.SoftmaxWhy = g.LanesWhy
	}
	if c := build(ir.SubgroupLanes); warpOK && c != nil {
		// One workgroup per head, one warp wide: the group IS the warp, so the
		// kernel's lane == tid holds without assuming how a larger group is cut
		// into subgroups.
		if g.softmaxAgrees(bs, c, heads, 32) {
			bs.softmax, bs.smGroups, bs.smWidth = c, heads, ir.SubgroupLanes
			g.SoftmaxLanes = ir.SubgroupLanes
			bs.softmaxAlt = build(1)
			bs.smAltGroups, bs.smAltWidth = scalarGroups, scalarWidth
			return true
		}
		closeKern(c)
	} else if c != nil {
		closeKern(c)
	}
	c := build(1)
	if c == nil {
		return false
	}
	bs.softmax, bs.smGroups, bs.smWidth = c, scalarGroups, scalarWidth
	g.SoftmaxLanes = 1
	return true
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
