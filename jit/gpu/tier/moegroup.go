package tier

import (
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// A mixture's prompt, batched. Batching needs the tokens routed to one expert
// to share its weight read, and the router decides that only at run time -- so
// the chunk is routed on the device, oldkernels.ExpertGroup* sort the (token,
// slot) pairs by expert with each expert's run padded to a whole matvec token
// group, the used column count comes home (the one word the grouped launches
// are sized by), and the expert matvecs run in that order with one expert per
// group (oldkernels.MatVec's grouped shape). ExpertCombineGrouped reads the
// outputs back in token order.
//
// The readback makes the block uncapturable, as a streamed block is.

// groupTok is how many sorted columns a grouped matvec thread carries, and so
// the padding unit of an expert's run: at a 128-row chunk and 8 of 128 experts
// an expert averages 8 tokens, and a group of 4 pads 1.5 of them.
const groupTok = 4

// voltaUnit is the padding unit of an expert's run on sm_70's tensor cores.
// oldkernels.GemmVolta's grouped shape reads one expert per token block, so a run
// is padded to the block (32 or 64 columns rather than 4); the padding is
// compute, never a weight read. A 64-column block wins where an expert
// averages 48 tokens or more; below that the extra padding costs more than
// the staged activations save. Measured by backend.TestGemmVoltaGroupedSpeed;
// see docs/engineering-history/gpu-kernels.md, "A mixture's prompt on sm_70's
// tensor cores".
func voltaUnit(rows, k, nExp int) int {
	if rows*k >= 48*nExp {
		return 64
	}
	return 32
}

// voltaGroupTile is GemmVolta's grouped blocking for a matrix of rows rows at
// a padding unit of unit columns: four warps along the tokens, each unit/32
// n-tiles wide, and the widest m-tile the rows divide.
//
// Three m-tiles are tried where 128 does not divide: gpt-oss's 2880 rows are
// 30 blocks of 96, faster than 64x64 because the staged activations serve more
// rows. A 16-element format at MT 3 stages 192 items, which do not nest in 128
// threads, so the generator refuses it and voltaGroupedMV takes the next tile.
func voltaGroupTiles(q kernels.Quant, rows, unit int) []oldkernels.VoltaTile {
	if unit%32 != 0 {
		return nil
	}
	sub, _, _, _ := kernels.Layout(q)
	var out []oldkernels.VoltaTile
	for _, mt := range []int{4, 3, 2, 1} {
		tl := oldkernels.VoltaTile{MT: mt, NT: unit / 32, WM: 1, WN: 4, KB: max(32/sub, 1)}
		if rows%tl.Rows() == 0 {
			out = append(out, tl)
		}
	}
	return out
}

// moeGroup is a batched scratch's mixture half.
type moeGroup struct {
	rows, k, nExp, p int
	// unit is the padding of an expert's run: groupTok, or a GemmVolta token
	// block on sm_70 (volta). tselU names each unit-wide block's expert, h16
	// holds the sorted activations as binary16 for it.
	unit       int
	volta      bool
	tselU, h16 backend.Buf
	router, router16, rank, weights, gather, quantH, actMul, quantX, combine,
	biasR, biasFF, biasD backend.Kernel
	// actMulW is Llama 4's input-weighted activation, reading each sorted
	// column's weight through colPair (the (token, slot) pair it serves), and
	// ones the unit weights the downs are then combined at. nil otherwise.
	actMulW       backend.Kernel
	colPair, ones backend.Buf
	colPairH      []uint32
	// ewScale is Gemma 4's per-expert weight factor (nn.LayerPlan.DenseMoE),
	// writing rwS; nil elsewhere.
	ewScale backend.Kernel
	rwS     backend.Buf
	// gscore and gmask are the V3 grouped route's two passes, per row; nil on
	// an ungrouped route. rgs and rbm are their planes.
	gscore, gmask backend.Kernel
	rlog, rlogB, zsel, rsel, rtop, rw, perm, tsel, pos, colE,
	hg, ga, gax, eg, eu, egB, euB, eact, edown, edownB, rgs, rbm backend.Buf
	// hsum is a split step's host sum of its rows (streamBank.hostRowsLate).
	hsum backend.Buf
	// gcount..gtilesU group the chunk's pairs by expert on the device
	// (oldkernels.ExpertGroup*), into cnt and ends; endsH is ends read home, whose
	// last word is the used column count the grouped launches are sized by.
	gbcount, gcount, gends, gcols, gscatter, gtiles, gtilesU backend.Kernel
	bc, off, cnt, ends                                       backend.Buf
	endsH                                                    []uint32
}

// moeGroupWhyNot names a plan the grouped path does not run, or "".
func moeGroupWhyNot(p *nn.LayerPlan) string {
	if p.NExpert == 0 {
		return "not a mixture"
	}
	return oldkernels.MoERouteWhyNot(routeOf(p))
}

// newMoeGroup builds the mixture half of a rows-wide scratch, or nil with
// g.LastErr naming why.
func (g *devTier) newMoeGroup(p *nn.LayerPlan, rows int) *moeGroup {
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	if why := moeGroupWhyNot(p); why != "" {
		g.LastErr = "tier: batched mixture: " + why
		return nil
	}
	k, e := p.NExpertUsed, p.NExpert
	// volta means the experts run as a binary16 staged GEMM: sm_70's
	// GemmVolta or Metal's GemmTile (tileMoE). Both read one expert per
	// token block from ActF16T(Gather) and add the expert bias themselves, so
	// the scratch, the padding and the launch are the same and only the
	// kernel differs (voltaGroupedMV picks it).
	staged := g.voltaOn() || g.tileMoE()
	unit, volta := groupTok, staged && !g.moeNarrow(p, rows, k, e)
	if staged && !volta {
		g.MoENarrowScratches++
	}
	if volta {
		unit = voltaUnit(rows, k, e)
	}
	np := (rows*k + e*(unit-1) + unit - 1) / unit * unit
	m := &moeGroup{rows: rows, k: k, nExp: e, p: np, unit: unit, volta: volta,
		endsH: make([]uint32, e)}
	actWin := p.ActWin
	if actWin == 0 {
		actWin = 32
	}
	// ew is the experts' own width: n_embd, or Kimi-K3's latent, where the
	// gather reads and the combine writes; the router reads n_embd always.
	ew := p.ExpWidth()
	wide := pad32(max(ew, p.NFFNExp))
	route := routeOf(p)
	route.Rows = rows
	var err error
	kern := func(dst *backend.Kernel, mk func() (*ir.Kernel, error)) {
		if err != nil {
			return
		}
		var kk *ir.Kernel
		if kk, err = mk(); err == nil {
			*dst, err = g.dev.Compile(kk)
		}
	}
	buf := func(dst *backend.Buf, n int) {
		if err == nil {
			*dst, err = g.dev.Alloc(n)
		}
	}
	kern(&m.router, func() (*ir.Kernel, error) { return oldkernels.RouterMatVecTok(e, p.NEmbd, rows, false) })
	kern(&m.router16, func() (*ir.Kernel, error) { return oldkernels.RouterMatVecTok(e, p.NEmbd, rows, true) })
	kern(&m.rank, func() (*ir.Kernel, error) { return oldkernels.ExpertRank(route) })
	kern(&m.weights, func() (*ir.Kernel, error) { return oldkernels.ExpertWeights(route) })
	if route.Grouped() {
		// The V3 family's two planes, per row: a chunk ranks every row against
		// its own group scores, so both carry a row stride (MoERoute.Rows).
		kern(&m.gscore, func() (*ir.Kernel, error) { return oldkernels.ExpertGroupScore(route) })
		kern(&m.gmask, func() (*ir.Kernel, error) { return oldkernels.ExpertGroupMask(route) })
		buf(&m.rgs, rows*p.ExpertGroups*4)
		buf(&m.rbm, rows*e*4)
	}
	// Under volta the dp4a half is not built at all: the grouped GEMM gathers
	// straight into binary16 and adds the expert biases itself, so the float32
	// sorted copy, the int8 activation and the biased copies would be scratch
	// nothing reads (~0.3 GB on gpt-oss-120b). prepBatch refuses the batch
	// rather than falling back when a bank has no Volta form.
	if !volta {
		kern(&m.gather, func() (*ir.Kernel, error) { return oldkernels.GatherRows(ew, np, rows) })
		kern(&m.quantH, func() (*ir.Kernel, error) { return quantizeRows(np, ew, actWin) })
		kern(&m.quantX, func() (*ir.Kernel, error) { return quantizeRows(np, p.NFFNExp, actWin) })
	}
	pair := p.ExpertWeightIn
	kern(&m.gbcount, func() (*ir.Kernel, error) { return oldkernels.ExpertGroupBlockCount(rows, k, e) })
	kern(&m.gcount, func() (*ir.Kernel, error) { return oldkernels.ExpertGroupCount(rows, k, e) })
	kern(&m.gends, func() (*ir.Kernel, error) { return oldkernels.ExpertGroupEnds(e, unit) })
	kern(&m.gcols, func() (*ir.Kernel, error) { return oldkernels.ExpertGroupColumns(rows, e, np, pair) })
	kern(&m.gscatter, func() (*ir.Kernel, error) { return oldkernels.ExpertGroupScatter(rows, k, e, unit, np, pair) })
	kern(&m.gtiles, func() (*ir.Kernel, error) { return oldkernels.ExpertGroupTiles(e, np/groupTok, groupTok) })
	if volta {
		kern(&m.gtilesU, func() (*ir.Kernel, error) { return oldkernels.ExpertGroupTiles(e, np/unit, unit) })
	}
	nb := oldkernels.GroupBlocks(rows, k)
	buf(&m.bc, nb*e*4)
	buf(&m.off, nb*e*4)
	buf(&m.cnt, 2*e*4) // + ExpertGroupCount's trash row
	buf(&m.ends, e*4)
	kern(&m.actMul, func() (*ir.Kernel, error) {
		// Ungated experts (Nemotron 3) take the activation alone.
		if ungatedFFN(p) {
			return oldkernels.Act(np*p.NFFNExp, p.Act)
		}
		return oldkernels.ActMul(np*p.NFFNExp, p.Act)
	})
	kern(&m.combine, func() (*ir.Kernel, error) { return oldkernels.ExpertCombineGrouped(ew, k, rows) })
	if p.DenseMoE {
		kern(&m.ewScale, func() (*ir.Kernel, error) { return oldkernels.ExpertWeightScale(rows, k) })
		buf(&m.rwS, rows*k*4)
	}
	if p.ExpertWeightIn {
		kern(&m.actMulW, func() (*ir.Kernel, error) { return oldkernels.ActMulWeighted(np*p.NFFNExp, p.NFFNExp, p.Act, true) })
		buf(&m.colPair, (np+1)*4)
		buf(&m.ones, rows*k*4)
		if err == nil {
			err = m.ones.Write(f32b(ones32(rows * k)))
		}
	}
	for _, b := range []struct {
		dst *backend.Buf
		n   int
	}{{&m.rlog, rows * e * 4}, {&m.rsel, rows * (k + 1) * 4}, {&m.rtop, rows * (k + 1) * 4},
		{&m.rw, rows * k * 4}, {&m.perm, (np + 1) * 4}, {&m.tsel, np / groupTok * 4}, {&m.pos, rows * k * 4},
		{&m.colE, np * 4},
		{&m.eg, np * p.NFFNExp * 4}, {&m.eu, np * p.NFFNExp * 4}, {&m.eact, np * p.NFFNExp * 4},
		{&m.edown, (np + 1) * ew * 4}, {&m.hsum, rows * ew * 4}} {
		buf(b.dst, b.n)
	}
	if err == nil {
		// The spare column the grouping gives a pair that names no expert
		// reads this row of the down projection's output: zero, and never
		// written, since the kernels write only the columns in use.
		err = m.edown.WriteAt(np*ew*4, make([]byte, ew*4))
	}
	if volta {
		buf(&m.tselU, np/unit*4)
		buf(&m.h16, np*wide*2)
	} else {
		buf(&m.hg, np*ew*4)
		buf(&m.ga, np*wide)
		buf(&m.gax, 3*np*(wide/32)*4)
	}
	if p.MoEBias {
		// The router bias is added to every row: IndexedBiasAdd with a
		// selection of zeros reads bias row 0 for each of them.
		kern(&m.biasR, func() (*ir.Kernel, error) { return oldkernels.IndexedBiasAdd(rows, e) })
		buf(&m.rlogB, rows*e*4)
		buf(&m.zsel, rows*4)
		if !volta {
			kern(&m.biasFF, func() (*ir.Kernel, error) { return oldkernels.IndexedBiasAdd(np, p.NFFNExp) })
			kern(&m.biasD, func() (*ir.Kernel, error) { return oldkernels.IndexedBiasAdd(np, ew) })
			buf(&m.egB, np*p.NFFNExp*4)
			buf(&m.euB, np*p.NFFNExp*4)
			buf(&m.edownB, np*ew*4)
		}
		if err == nil {
			err = m.zsel.Write(make([]byte, rows*4))
		}
	}
	if err != nil {
		g.LastErr = "tier: batched mixture: " + err.Error()
		m.free(g.devShared)
		return nil
	}
	return m
}

func (m *moeGroup) free(g *devShared) {
	if m == nil {
		return
	}
	for _, k := range []backend.Kernel{m.router, m.router16, m.rank, m.weights, m.gather, m.quantH, m.actMul,
		m.quantX, m.combine, m.biasR, m.biasFF, m.biasD, m.gscore, m.gmask, m.actMulW,
		m.gbcount, m.gcount, m.gends, m.gcols, m.gscatter, m.gtiles, m.gtilesU, m.ewScale} {
		if k != nil {
			g.closeKern(k)
		}
	}
	for _, b := range []backend.Buf{m.rlog, m.rlogB, m.zsel, m.rsel, m.rtop, m.rw, m.perm, m.tsel,
		m.pos, m.colE, m.hg, m.ga, m.gax, m.eg, m.eu, m.egB, m.euB, m.eact, m.edown, m.edownB,
		m.rgs, m.rbm, m.colPair, m.ones, m.tselU, m.h16, m.bc, m.off, m.cnt, m.ends, m.rwS, m.hsum} {
		if b != nil {
			b.Free()
		}
	}
}

// voltaMoEWide is whether a scratch of rows rows picks the staged f16 GEMM for
// its experts: a few-row step (a speculation's verification) routes rows*k
// pairs over the bank, so most experts hold one column and the GEMM pads each
// to a 32-column block, where the dp4a grouped matvec pads to groupTok and reads
// the same weights. The class is the scratch's executed width, not the prompt's:
// VoltaMoEMinPairs is the pair count below which the dp4a half is built
// instead; 0 takes the bank's expert count (an expert averages a column), and
// 1 is the old behaviour, the staged GEMM at every width.
func (g *devTier) voltaMoEWide(rows, k, nExp int) bool {
	min := g.VoltaMoEMinPairs
	if min == 0 {
		// An expert averages a column or more.
		min = nExp
	}
	return rows*k >= min
}

// voltaOn is whether a batched mixture's experts take sm_70's f16 tensor
// cores: a card with no integer matrix instruction (or NoMMA) whose backend
// lowers the grouped GemmVolta, unless NoVolta or NoVoltaMoE.
//
// It asks the backend because the scratch is built before any matvec: mmaOff
// is learned lazily from the first batched matvec that fails to compile, and
// newMoeGroup has to fix the padding unit before that. So sm70 compiles two
// probes itself, once per device: the integer matrix instruction (whose
// refusal is what mmaOff records) and a small grouped GemmVolta (which SPIR-V
// and MSL refuse by name).
func (g *devTier) voltaOn() bool {
	return !g.NoVolta && !g.NoVoltaMoE && g.sm70()
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
			g.closeKern(c)
			return true
		}
		if !g.mmaOff && !g.NoMMA {
			// Only a COMPILE refusal is the device's; a shape the generator
			// refuses says nothing about the card.
			if ker, err := oldkernels.MatVecMMA(oldkernels.MatVecShape{Center: g.center, T: kernels.Q8_0, K: 32, Rows: 16, NTok: 8,
				MT: 1, NT: 1}); err == nil {
				if c, err := g.dev.Compile(ker); err != nil {
					g.mmaOff, g.MMAWhy = true, err.Error()
				} else {
					g.closeKern(c)
				}
			}
		}
		tl := oldkernels.VoltaTile{MT: 4, NT: 1, WM: 1, WN: 4, KB: 1}
		if (g.mmaOff || g.NoMMA) && probe(oldkernels.GemmVolta(oldkernels.MatVecShape{Center: g.center, T: kernels.Q8_0, K: 32, Rows: 128,
			NTok: 32, Experts: 2}, tl)) {
			g.voltaMoE = 1
		}
	}
	return g.voltaMoE > 0
}

// tileMoE is whether a batched mixture's experts run as the grouped
// oldkernels.GemmTile -- the collective tiles, wherever the device lowers them
// (tilesOK) -- as tileMV asks for the dense one, and off once the device has
// refused a tile.
func (g *devTier) tileMoE() bool {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	return !g.NoVolta && !g.NoVoltaMoE && g.tilesOK()
}

// tilesOK reports whether this device lowers the collective tile ops at the
// 8x8 binary16 shape GemmTile and FlashPrefillTile multiply in -- Metal's
// simdgroup_matrix, mma.sync on sm_75 and later -- asked once by compiling a
// probe (backend.LowersTile) instead of naming a backend. It is false once a
// built tile kernel has been refused (tileOff).
func (g *devTier) tilesOK() bool {
	if g.tileCap == 0 {
		g.tileCap = -1
		// A device that does not lower them is not a refusal (TileWhy is
		// what a device said about a kernel it was asked to build): it is
		// simply not asked.
		if ok, _ := backend.LowersTile(g.dev, backend.TileShape{M: 8, N: 8, K: 8, Elem: ir.TileF16, Acc: ir.TileF32}); ok {
			g.tileCap = 1
		}
	}
	return g.tileCap == 1 && !g.tileOff
}

// tileF32 reports whether the device also lowers float32-operand tiles, which
// oldkernels.FlashPrefillTile multiplies P by V in: Metal's float
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
// (oldkernels.TileGemm.Tall): mma.sync's own m16, which the 8-row tile runs
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

// tileGroupTiles is GemmTile's grouped blocking for rows at a padding unit of
// unit columns: two subgroups along the tokens, each unit/16 n-tiles wide, and
// the widest m-tile the rows divide -- tier.tileGemms' 64-row block first.
func tileGroupTiles(q kernels.Quant, rows, unit int) []oldkernels.TileGemm {
	if unit%16 != 0 {
		return nil
	}
	sub, _, _, _ := kernels.Layout(q)
	var out []oldkernels.TileGemm
	for _, mt := range []int{4, 2, 1} {
		tl := oldkernels.TileGemm{MT: mt, NT: unit / 16, WM: 2, WN: 2, KB: max(32/sub, 1)}
		if rows%tl.Rows() == 0 {
			out = append(out, tl)
		}
	}
	return out
}

// moeVoltaArgs is how emitGroupedMoE asks voltaGroupedMV for matvec m of
// layer l, which prepBatch must ask identically so the emit finds its kernels
// compiled: whether the expert bias rides the GEMM's epilogue, and the source
// row count the activation is gathered from (the normed rows for gate and up,
// 0 for the down, whose input is already in sorted order).
func moeVoltaArgs(l *layer, m *mv, mg *moeGroup) (bias bool, nsrc int) {
	switch m {
	case &l.mvg:
		return l.expGateB != nil, mg.rows
	case &l.mvu:
		return l.expUpB != nil, mg.rows
	}
	return l.expDownB != nil, 0
}

// moeKernKey names a batched mixture's compiled kernel in devTier.ragKerns:
// the kind and the shape fields it is keyed on. A struct, where a formatted
// string was built for every lookup -- every bank of every mixture block of
// every step.
type moeKernKey struct {
	kind                    string
	q                       kernels.Quant
	k, rows, bank, np, nsrc int
	vt                      oldkernels.VoltaTile
	tt                      oldkernels.TileGemm
	bias                    bool
}

// moeVoltaKey is what voltaGroupedMV is asked: a bank's shape and the padding,
// bias and source it is asked at.
type moeVoltaKey struct {
	q                             kernels.Quant
	k, rows, bank, np, unit, nsrc int
	bias                          bool
}

// moeVoltaMV is voltaGroupedMV's answer: the GEMM, the conversion it reads,
// its workgroup width, and how many row blocks it runs per token block in use.
// kern is nil where the bank has no tensor-core form.
type moeVoltaMV struct {
	kern, act backend.Kernel
	width     int
	rowBlocks int
}

// groups is the launch's workgroup count over used token blocks, the row
// block fastest: GemmVoltaGrouped and GemmTileGrouped, which are the same
// product.
func (v moeVoltaMV) groups(used int) int { return v.rowBlocks * used }

// moeKern is ragKerns' lookup, compiling on a miss. A failed compile is kept
// as nil, so the next ask is a lookup too.
func (g *devTier) moeKern(key moeKernKey, why string, mk func() (*ir.Kernel, error)) backend.Kernel {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	if c, hit := g.ragKerns[key]; hit {
		return c
	}
	ker, err := mk()
	var c backend.Kernel
	if err == nil {
		c, err = g.dev.Compile(ker)
	}
	if err != nil {
		g.LastErr = why + err.Error()
		c = nil
	}
	if g.ragKerns == nil {
		g.ragKerns = map[moeKernKey]backend.Kernel{}
	}
	g.ragKerns[key] = c
	return c
}

// voltaGroupedMV is the tensor-core twin of a bank's grouped matvec -- a
// grouped oldkernels.GemmVolta over np sorted columns in unit-wide blocks, with
// the expert bias in its epilogue where bias -- and the conversion it reads:
// ActF16TGather from nsrc source rows, or ActF16T of an already sorted input
// at nsrc 0. Compiled on first use (prepBatch makes that happen outside any
// session) and the answer kept, so inside a session it is one lookup on a
// struct: a step asks it for every bank of every mixture block. False where
// the format or the shape has no Volta form; the dp4a grouped matvec then
// serves.
//
// On Metal it is the grouped GemmTile (tileMoE), with the same parameters,
// the same activation conversions and the same grid order, so emitGroupedMoE
// launches either one through moeVoltaMV.
func (g *devTier) voltaGroupedMV(m mv, bank, np, unit int, bias bool, nsrc int) (moeVoltaMV, bool) {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	vk := moeVoltaKey{q: m.q, k: m.k, rows: m.rows, bank: bank, np: np, unit: unit, nsrc: nsrc, bias: bias}
	if v, hit := g.moeVolta[vk]; hit {
		return v, v.kern != nil
	}
	var v moeVoltaMV
	if oldkernels.Volta70OK(m.q) {
		v = g.buildVoltaGroupedMV(m, bank, np, unit, bias, nsrc)
	}
	if g.moeVolta == nil {
		g.moeVolta = map[moeVoltaKey]moeVoltaMV{}
	}
	g.moeVolta[vk] = v
	return v, v.kern != nil
}

// buildVoltaGroupedMV compiles what voltaGroupedMV keeps: the widest tile the
// device takes and its conversion, or nothing where either is refused.
func (g *devTier) buildVoltaGroupedMV(m mv, bank, np, unit int, bias bool, nsrc int) moeVoltaMV {
	const why = "tier: batched mixture on tensor cores: "
	var v moeVoltaMV
	s := oldkernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, NTok: np, Experts: bank, Bias: bias}
	key := moeKernKey{q: m.q, k: m.k, rows: m.rows, bank: bank, np: np, bias: bias}
	if g.tileMoE() {
		key.kind = "groupedtile"
		for _, tl := range tileGroupTiles(m.q, m.rows, unit) {
			tl.Tall = g.tileTall() && tl.MT%2 == 0
			key.tt = tl
			if v.kern = g.moeKern(key, why, func() (*ir.Kernel, error) { return oldkernels.GemmTile(s, tl) }); v.kern != nil {
				v.width, v.rowBlocks = tl.Threads(), oldkernels.GemmTileGrouped(s, tl, 1)
				break
			}
		}
	} else {
		key.kind = "groupedvolta"
		for _, tl := range voltaGroupTiles(m.q, m.rows, unit) {
			key.vt = tl
			if v.kern = g.moeKern(key, why, func() (*ir.Kernel, error) { return oldkernels.GemmVolta(s, tl) }); v.kern != nil {
				v.width, v.rowBlocks = tl.Threads(), oldkernels.GemmVoltaGrouped(s, tl, 1)
				break
			}
		}
	}
	if v.kern == nil {
		return moeVoltaMV{}
	}
	if nsrc > 0 {
		v.act = g.moeKern(moeKernKey{kind: "actf16tgather", q: m.q, k: m.k, np: np, nsrc: nsrc}, why,
			func() (*ir.Kernel, error) { return oldkernels.ActF16TGather(m.q, np, m.k, nsrc) })
	} else {
		v.act = g.moeKern(moeKernKey{kind: "actf16t", q: m.q, k: m.k, np: np}, why,
			func() (*ir.Kernel, error) { return oldkernels.ActF16T(m.q, np, m.k) })
	}
	if v.act == nil {
		return moeVoltaMV{}
	}
	return v
}

// moeFloatHalf gives m what a float bank's grouped matvec reads and writes
// where the scratch was built without it -- sm_70's, which builds no dp4a half
// (newMoeGroup): the gather into sorted float rows and, with expert biases,
// their elementwise adds, since a float bank's matvec does not add them. A
// no-op once built. It compiles and allocates, so it runs outside any session.
func (g *devTier) moeFloatHalf(m *moeGroup, p *nn.LayerPlan) bool {
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	var err error
	kern := func(dst *backend.Kernel, mk func() (*ir.Kernel, error)) {
		if err != nil || *dst != nil {
			return
		}
		var kk *ir.Kernel
		if kk, err = mk(); err == nil {
			*dst, err = g.dev.Compile(kk)
		}
	}
	buf := func(dst *backend.Buf, n int) {
		if err == nil && *dst == nil {
			*dst, err = g.dev.Alloc(n)
		}
	}
	ew := p.ExpWidth()
	kern(&m.gather, func() (*ir.Kernel, error) { return oldkernels.GatherRows(ew, m.p, m.rows) })
	buf(&m.hg, m.p*ew*4)
	if p.MoEBias {
		kern(&m.biasFF, func() (*ir.Kernel, error) { return oldkernels.IndexedBiasAdd(m.p, p.NFFNExp) })
		kern(&m.biasD, func() (*ir.Kernel, error) { return oldkernels.IndexedBiasAdd(m.p, ew) })
		buf(&m.egB, m.p*p.NFFNExp*4)
		buf(&m.euB, m.p*p.NFFNExp*4)
		buf(&m.edownB, m.p*ew*4)
	}
	if err != nil {
		g.LastErr = "tier: batched mixture, a float bank: " + err.Error()
		return false
	}
	return true
}

// groupedMV is the grouped twin of a bank's matvec, compiled on first use --
// which prepBatch makes happen outside any session, so inside one it is a
// lookup on a struct. A float bank reads the float activation its caller
// passes in the scale-plane slot (oldkernels.MatVec), column j at j*k, as the
// int8 one is read.
func (g *devTier) groupedMV(m mv, bank, np int) (backend.Kernel, bool) {
	// A build reached from inside a submission takes g.mu (subLock).
	defer g.subUnlock(g.subLock())
	key := moeKernKey{kind: "grouped", q: m.q, k: m.k, rows: m.rows, bank: bank, np: np}
	if c, hit := g.ragKerns[key]; hit {
		return c, c != nil
	}
	// A float bank narrower than 32 (MLA's absorbs at a small latent or nope
	// width) walks its own k; see oldkernels.MatVecShape.Sub.
	c := g.moeKern(key, "tier: batched mixture: ", func() (*ir.Kernel, error) {
		return oldkernels.MatVec(oldkernels.MatVecShape{Center: g.center, T: m.q, K: m.k, Rows: m.rows, Experts: bank,
			NTok: np, Tok: groupTok, ActWin: g.actWinFor(0), Sub: subFor(m.q, m.k)})
	})
	return c, c != nil
}

// emitGroupedMoE is a batched chunk's routed mixture, from the normed rows in
// bs.h to the weighted expert sum in bs.mvOut. It reads the selection home, so
// it runs inside the submission but outside any captured graph.
//
// Every launch goes through lc's reused buffer list, as the rest of the
// submission's do (see launcher): a step runs this for every mixture block, and
// a buffer list handed to the session directly was a heap object per launch.
// It writes no buffer lc's memo tracks (the shared f16 buffer and the int8
// activation), so it launches through lc.launch rather than lc.la.
//
// rin is the router's input rows: bs.h, where the experts read, for every
// mixture but Gemma 4's (nn.LayerPlan.DenseMoE), whose router has a norm of
// its own.
func (g *devTier) emitGroupedMoE(sid uint64, s backend.Session, lc *launcher, bs *blockScratch, l *layer, rows, valid int,
	rin, ein, eout backend.Buf, errp *error) {
	m, p := bs.mg, &bs.p
	ew := p.ExpWidth()
	la := func(k backend.Kernel, threads, group int, bufs ...backend.Buf) {
		if *errp == nil {
			*errp = lc.launch(k, (threads+group-1)/group, group, bufs...)
		}
	}
	if m == nil || m.rows != rows {
		*errp = errNoBatch
		return
	}
	e := m.nExp
	router := m.router
	if l.routerF16 {
		router = m.router16
	}
	la(router, oldkernels.RouterThreads(e*rows), oldkernels.RouterThreads(1), l.router, rin, m.rlog)
	logits := m.rlog
	if l.routerB != nil {
		la(m.biasR, rows*e, 128, m.rlog, m.zsel, l.routerB, m.rlogB)
		logits = m.rlogB
	}
	// The arity follows the route, exactly as the decode arm's does: CUDA
	// ignores a surplus buffer and Vulkan refuses one.
	switch {
	case m.gscore != nil:
		if l.expSelB != nil {
			la(m.gscore, rows*p.ExpertGroups, 128, logits, m.rgs, l.expSelB)
			la(m.gmask, rows*e, 128, logits, m.rgs, m.rbm, l.expSelB)
		} else {
			la(m.gscore, rows*p.ExpertGroups, 128, logits, m.rgs)
			la(m.gmask, rows*e, 128, logits, m.rgs, m.rbm)
		}
		la(m.rank, rows*e, 128, logits, m.rsel, m.rtop, m.rbm)
	case l.ds4 != nil:
		// DeepSeek V4 selects through a plane per row (ds4.go).
		la(m.rank, rows*e, 128, logits, m.rsel, m.rtop, g.ds4RouteBias(lc, bs, l, rows))
	case l.expSelB != nil:
		la(m.rank, rows*e, 128, logits, m.rsel, m.rtop, l.expSelB)
	default:
		la(m.rank, rows*e, 128, logits, m.rsel, m.rtop)
	}
	if p.NoExpertNorm && !p.ExpertSigmoid {
		la(m.weights, rows, 64, m.rtop, m.rw, logits)
	} else {
		la(m.weights, rows, 64, m.rtop, m.rw)
	}
	// pass is the card's grouped run over the selection m.rsel and weights
	// m.rw holds, its weighted sum written to dst.
	pass := func(dst backend.Buf) {
		gr := groupRun{g: g, m: m, l: l, p: p, rows: rows, e: e, ew: ew, ein: ein}
		if !gr.sort(groupLaunch{lc, errp}, errp, s) {
			return
		}
		cols := gr.cols
		if ungatedFFN(p) {
			// Ungated experts (Nemotron 3): up, the activation alone, down. No
			// such architecture has expert biases or input weighting.
			gr.grouped(groupLaunch{lc, errp}, errp, &l.mvu, l.up, nil, m.eu)
			gr.actK = nil
			la(m.actMul, cols*p.NFFNExp, 128, m.eu, m.eact)
			gr.grouped(groupLaunch{lc, errp}, errp, &l.mvd, l.down, nil, m.edown)
			la(m.combine, rows*ew, 128, m.edown, m.rw, m.pos, dst)
			g.GroupedMoE++
			return
		}
		gateB := gr.grouped(groupLaunch{lc, errp}, errp, &l.mvg, l.gate, l.expGateB, m.eg)
		// A different act kernel is a different layout; the SAME kernel over the
		// same source is the same conversion, so up reuses gate's unless its
		// format differs. The down reads eact, a different source: forget it.
		upB := gr.grouped(groupLaunch{lc, errp}, errp, &l.mvu, l.up, l.expUpB, m.eu)
		gr.actK = nil
		eg, eu := m.eg, m.eu
		if l.expGateB != nil && !gateB {
			la(m.biasFF, cols*p.NFFNExp, 128, m.eg, m.colE, l.expGateB, m.egB)
			eg = m.egB
		}
		if l.expUpB != nil && !upB {
			la(m.biasFF, cols*p.NFFNExp, 128, m.eu, m.colE, l.expUpB, m.euB)
			eu = m.euB
		}
		rw := m.rw
		if m.ewScale != nil {
			// Each routed weight times its expert's own factor (Gemma 4).
			la(m.ewScale, rows*m.k, 128, m.rw, m.rsel, l.expScale, m.rwS)
			rw = m.rwS
		}
		if m.actMulW != nil && !g.moeFaulted(moeWout) {
			la(m.actMulW, cols*p.NFFNExp, 128, eg, eu, m.rw, m.colPair, m.eact)
			rw = m.ones
		} else {
			la(m.actMul, cols*p.NFFNExp, 128, eg, eu, m.eact)
		}
		downB := gr.grouped(groupLaunch{lc, errp}, errp, &l.mvd, l.down, l.expDownB, m.edown)
		edown := m.edown
		if l.expDownB != nil && !downB {
			la(m.biasD, cols*ew, 128, m.edown, m.colE, l.expDownB, m.edownB)
			edown = m.edownB
		}
		la(m.combine, rows*ew, 128, edown, rw, m.pos, dst)
		g.GroupedMoE++
	}
	// A hybrid block's experts run on the host, every valid row of the chunk
	// (valid, not the padded width: a padding row's experts are host time for
	// nothing): the routing above is all the device does of them. So do a
	// streamed block's (its experts sent to the card a token at a time): its
	// bank holds a token's selection or its cache, not a chunk's, and a
	// refusal here came after the chunk's linear blocks had advanced, which a
	// hybrid model cannot take back.
	if l.stream != nil && (l.stream.hybrid || l.stream.host != nil) {
		if valid <= 0 || valid > rows {
			valid = rows
		}
		// With an expert cache on the card, the rows' cached experts run there
		// through the grouped kernels below, each read once for every row that
		// chose it, while the host runs the misses expert-major (splitRows):
		// a speculation step's verification costs the experts its rows touch,
		// not a decode token a row.
		st := l.stream
		// A prompt-sized chunk runs its experts on the card in waves through
		// the cache (prefillWaves) rather than reading every one on the host,
		// where graph.MoEWaves offers it (wavesRun): a prompt chunk's rows, a
		// bank streamed through the cache, and no per-expert bias or scale,
		// which index by expert id where a wave's slot ids are not.
		if !g.NoPrefillExperts && *errp == nil && g.wavesRun(l, m, valid) {
			*errp = st.prefillWaves(g, s, l, e, m.rsel, m.rw, m.k, rows, valid, func(first bool) error {
				if first {
					pass(eout)
				} else {
					pass(m.hsum)
					// The violation: the later waves' share dropped.
					if *errp == nil && !g.moeFaulted(moePrefillwave) {
						lc.la(bs.addE, valid*ew, eout, m.hsum, eout)
					}
				}
				return *errp
			})
			if *errp == nil {
				g.HybridRuns++
			}
			return
		}
		if hs := g.hostFns[sid][st.li]; st.cached() && !g.NoSplitRows && hs.one != nil && *errp == nil {
			if st.bank != l.down {
				st.forget()
				st.bank = l.down
			}
			hits := 0
			hits, *errp = st.splitRows(g, s, m.rsel, m.rw, ein, ew, m.k, rows, valid)
			if *errp != nil {
				return
			}
			g.HybridRuns++
			learn := func() {
				if *errp == nil && !g.NoRowAdmits {
					*errp = st.learnRows(g, s, l, g.hostFns[sid][st.li].uses, p.NExpert, m.k, valid)
				}
			}
			if hits == 0 {
				// Nothing on the card: the host's sum is the output.
				*errp = st.hostRowsLate(sid, g, s, eout, m.k, valid)
				learn()
				return
			}
			defer func() {
				if *errp == nil {
					*errp = st.hostRowsLate(sid, g, s, m.hsum, m.k, valid)
				}
				learn()
				// The violation: the host's share dropped, the card's hits alone.
				if *errp == nil && !g.moeFaulted(moeSplitrowshost) {
					lc.la(bs.addE, valid*ew, eout, m.hsum, eout)
				}
			}()
		} else {
			if *errp == nil {
				*errp = l.stream.runHostRows(sid, g, s, m.rsel, m.rw, ein, eout, ew, m.k, valid)
			}
			if *errp == nil {
				g.HybridRuns++
			}
			return
		}
	}
	pass(eout)
}

// groupRun is one grouped run of the routed experts over a chunk's rows: the
// selection sorted by expert (sort), then each bank's matvec over the rows
// that chose its experts (grouped), reading them gathered, quantized or
// converted once per source. The block encoder's pass and the device
// program's expert ops both run through it.
//
// The launch path and the error are each method's parameters, not fields:
// escape analysis does not tell one field from another, and a launch's
// buffers flow to the heap, so a field holding the session's error moved it
// there on every call.
type groupRun struct {
	g *devTier
	m *moeGroup
	l *layer
	p *nn.LayerPlan

	rows, e, ew int
	ein         backend.Buf // the rows the experts read

	cols, used int // the sorted columns, and the tiles they fill
	gathered   bool
	quantized  backend.Buf
	// actK and actSrc are the conversion h16 holds: the kernel and the
	// source it read. Both, so a caller cannot forget that down reads
	// another source than gate and up.
	actK    backend.Kernel
	actSrc  backend.Buf
	downSrc backend.Buf
}

// sort groups the selection by expert on the device and reads back the one
// size that depends on it, the used column count. False on an error.
func (r *groupRun) sort(gl groupLaunch, errp *error, s backend.Session) bool {
	m, rows, e := r.m, r.rows, r.e
	// The grouping runs on the device (oldkernels.ExpertGroup*) and only the used
	// column count comes home.
	nbe := oldkernels.GroupBlocks(rows, m.k) * e
	gl.la(m.gbcount, nbe, 128, m.rsel, m.bc)
	gl.la(m.gcount, nbe, 128, m.bc, m.off, m.cnt)
	gl.la(m.gends, e, 128, m.cnt, m.ends)
	if m.colPair != nil {
		gl.la(m.gcols, m.p, 128, m.ends, m.colE, m.perm, m.colPair)
		gl.la(m.gscatter, rows*m.k, 128, m.rsel, m.cnt, m.ends, m.off, m.perm, m.pos, m.colPair)
	} else {
		gl.la(m.gcols, m.p, 128, m.ends, m.colE, m.perm)
		gl.la(m.gscatter, rows*m.k, 128, m.rsel, m.cnt, m.ends, m.off, m.perm, m.pos)
	}
	gl.la(m.gtiles, m.p/groupTok, 128, m.ends, m.tsel)
	if m.volta {
		gl.la(m.gtilesU, m.p/m.unit, 128, m.ends, m.tselU)
	}
	if *errp == nil {
		*errp = s.Sync()
	}
	if *errp == nil {
		*errp = s.Read(m.ends, u32b(m.endsH))
	}
	if *errp != nil {
		return false
	}
	r.cols = int(m.endsH[e-1])
	r.used = r.cols / groupTok
	// downSrc is what a quantized down reads: the activated product.
	r.downSrc = m.eact
	if r.g.moeFaulted(moeDownsrc) {
		r.downSrc = m.eg
	}
	return true
}

// gather sorts the rows the experts read into column order, once. On the
// tensor cores the gather is the conversion: gate and up read the normed rows
// through ActF16TGather, so the float32 sorted copy (hg) exists only for the
// dp4a path, and the expert biases ride the GEMM's epilogue instead of three
// elementwise passes over the sorted outputs.
func (r *groupRun) gather(gl groupLaunch) {
	if !r.gathered {
		gl.la(r.m.gather, r.cols*r.ew, 128, r.ein, r.m.perm, r.m.hg)
		r.gathered = true
	}
}

// quantize is the dp4a matvecs' int8 activation, launched only when one of
// the three runs there; the tensor-core ones convert to binary16 once per
// source and layout (a Q6_K down reads a different step order from a Q4_K
// gate).
func (r *groupRun) quantize(gl groupLaunch, q backend.Kernel, n int, src backend.Buf) {
	if r.quantized != src {
		gl.la(q, oldkernels.QuantizeThreads(r.cols*pad32(n)/32), 128, src, r.m.ga, r.m.gax)
		r.quantized = src
	}
}

// grouped runs one bank over the sorted columns into dst and reports whether
// it added the bias itself.
func (r *groupRun) grouped(gl groupLaunch, errp *error, mv *mv, res *resident, bias, dst backend.Buf) bool {
	g, m, l := r.g, r.m, r.l
	down := mv == &l.mvd
	if kernels.IsFloat(mv.q) {
		// A float bank reads the sorted float rows in the scale plane's
		// slot and no int8 activation (oldkernels.MatVec), on sm_70 as
		// anywhere: moeFloatHalf built the gather there.
		c, ok := g.groupedMV(*mv, r.e, m.p)
		if !ok || m.gather == nil {
			if *errp == nil {
				*errp = errNoBatch
			}
			return false
		}
		src := m.eact
		if g.moeFaulted(moeFloatsrc) {
			src = m.eg
		}
		if !down {
			r.gather(gl)
			src = m.hg
		}
		// The int8 slots are bound and never read; sm_70's scratch has no
		// int8 buffers, so they take the source too.
		a, ax := m.ga, m.gax
		if a == nil {
			a, ax = src, src
		}
		gl.la(c, mv.rows*r.used, 128, res.qs, src, res.sc, a, ax, dst, m.tsel)
		g.GroupedFloat++
		return false
	}
	if !m.volta {
		c, ok := g.groupedMV(*mv, r.e, m.p)
		if !ok {
			if *errp == nil {
				*errp = errNoBatch
			}
			return false
		}
		if down {
			r.quantize(gl, m.quantX, r.p.NFFNExp, r.downSrc)
		} else {
			r.gather(gl)
			r.quantize(gl, m.quantH, r.ew, m.hg)
		}
		// Only the groups in use: the thread index is row fastest, then group.
		gl.la(c, mv.rows*r.used, 128, res.qs, res.d, res.sc, m.ga, m.gax, dst, m.tsel)
		return false
	}
	// prepBatch refused the batch unless every bank has its Volta form
	// compiled, so a refusal here is a broken invariant, not a fallback.
	withBias, nsrc := moeVoltaArgs(l, mv, m)
	vm, ok := g.voltaGroupedMV(*mv, r.e, m.p, m.unit, withBias, nsrc)
	if !ok {
		if *errp == nil {
			*errp = errNoBatch
		}
		return withBias
	}
	src := r.ein
	if down {
		src = r.downSrc
	}
	if vm.act != r.actK || src != r.actSrc {
		sub, _, _, _ := kernels.Layout(mv.q)
		if down {
			gl.la(vm.act, r.cols*mv.k/sub, 128, r.downSrc, m.h16)
		} else {
			gl.la(vm.act, r.cols*mv.k/sub, 128, r.ein, m.perm, m.h16)
		}
		r.actK, r.actSrc = vm.act, src
		g.ActF16Launches++
	}
	// Only the token blocks in use: the row block varies fastest.
	groups := vm.groups(r.cols / m.unit)
	if withBias {
		gl.la(vm.kern, groups*vm.width, vm.width, res.qs, res.d, res.sc, m.h16, dst, bias, m.tselU)
	} else {
		gl.la(vm.kern, groups*vm.width, vm.width, res.qs, res.d, res.sc, m.h16, dst, m.tselU)
	}
	g.GroupedVolta++
	return withBias
}

// groupLaunch launches a grouped run's kernels on lc, kernel k over threads
// in workgroups of group, latching the submission's first error into *err.
// A concrete type, not a function value: a launch through a function value
// moves its argument slice to the heap, which a warm step must not.
type groupLaunch struct {
	lc  *launcher
	err *error
}

func (g groupLaunch) la(k backend.Kernel, threads, group int, bufs ...backend.Buf) {
	if *g.err == nil {
		*g.err = g.lc.launch(k, (threads+group-1)/group, group, bufs...)
	}
}
