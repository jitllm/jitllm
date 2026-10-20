package tier

import (
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// groupTok is the padding of an expert's run of columns.
const groupTok = 4

// moeGroup is a batched scratch's mixture half: the routed rows' values,
// p columns of (token, expert) pairs, each expert's run padded to groupTok.
type moeGroup struct {
	rows, k, nExp, p int
	// unit is what each expert's run of columns is padded to (groupUnit).
	unit          int
	colPair, ones backend.Buf
	rwS           backend.Buf
	rlog, rlogB, rsel, rtop, rw, rbm,
	eg, eu, egB, euB, eact, edown, edownB backend.Buf
	// The grouping's planes (compose.Group*): the block counts, offsets and
	// totals, each run's end, each column's expert, row and pair's column,
	// each tile's expert; the sorted rows and their int8 operand. endsH is
	// ends read home, its last word the used column count.
	bc, off, cnt, ends, colE, perm, pos, tsel, hg, ga, gax backend.Buf
	endsH                                                  []uint32
}

func (m *moeGroup) free() {
	if m == nil {
		return
	}
	for _, b := range []backend.Buf{m.colPair, m.ones, m.rwS, m.rlog, m.rlogB, m.rsel, m.rtop, m.rw, m.rbm,
		m.eg, m.eu, m.egB, m.euB, m.eact, m.edown, m.edownB,
		m.bc, m.off, m.cnt, m.ends, m.colE, m.perm, m.pos, m.tsel, m.hg, m.ga, m.gax} {
		if b != nil {
			b.Free()
		}
	}
}

func (g *devTier) newMoeGroup(p *nn.LayerPlan, rows int) *moeGroup {
	defer g.scratchWin().close()
	k, e, ew := p.NExpertUsed, p.NExpert, p.ExpWidth()
	unit := g.groupUnit(rows, k, e)
	np := (rows*k + e*(unit-1) + unit - 1) / unit * unit
	m := &moeGroup{rows: rows, k: k, nExp: e, p: np, unit: unit, endsH: make([]uint32, e)}
	nb, wide := compose.GroupBlocks(rows, k), max(ew, p.NFFNExp)
	var err error
	buf := func(dst *backend.Buf, n int) {
		if err == nil {
			*dst, err = g.dev.Alloc(n)
		}
	}
	if p.ExpertGroups > 1 {
		buf(&m.rbm, rows*e*4)
	}
	if p.DenseMoE {
		buf(&m.rwS, rows*k*4)
	}
	if p.ExpertWeightIn {
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
		{&m.rw, rows * k * 4}, {&m.eg, np * p.NFFNExp * 4}, {&m.eu, np * p.NFFNExp * 4},
		{&m.eact, np * p.NFFNExp * 4}, {&m.edown, (np + 1) * ew * 4},
		{&m.bc, nb * e * 4}, {&m.off, nb * e * 4}, {&m.cnt, 2 * e * 4}, {&m.ends, e * 4}, {&m.colE, np * 4},
		{&m.perm, np * 4}, {&m.pos, rows * k * 4}, {&m.tsel, np / groupTok * 4}, {&m.hg, np * ew * 4},
		{&m.ga, np * wide}, {&m.gax, 3 * np * (wide / 32) * 4}} {
		buf(b.dst, b.n)
	}
	if err == nil {
		err = m.edown.WriteAt(np*ew*4, make([]byte, ew*4))
	}
	if p.MoEBias {
		buf(&m.rlogB, rows*e*4)
		buf(&m.egB, np*p.NFFNExp*4)
		buf(&m.euB, np*p.NFFNExp*4)
		buf(&m.edownB, np*ew*4)
	}
	if err != nil {
		g.LastErr = "tier: batched mixture: " + err.Error()
		m.free()
		return nil
	}
	return m
}

func ones32(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = 1
	}
	return v
}

// groupUnit is the columns each expert's run is padded to: a token group
// (groupTok), or on a device that lowers the 8x8 binary16 tile, a staged
// GEMM's tile of tokens -- 64 where the pairs reach 48 an expert, else 32.
func (g *devTier) groupUnit(rows, k, nExp int) int {
	if !g.tilesOK() {
		return groupTok
	}
	if rows*k >= 48*nExp {
		return 64
	}
	return 32
}

// tilesOK reports whether this device lowers the collective tile ops at the
// 8x8 binary16 shape, asked once by compiling a probe. Callers hold g.mu.
func (g *devTier) tilesOK() bool {
	if g.tileCap == 0 {
		g.tileCap = -1
		if ok, _ := backend.LowersTile(g.dev, backend.TileShape{M: 8, N: 8, K: 8, Elem: ir.TileF16, Acc: ir.TileF32}); ok {
			g.tileCap = 1
		}
	}
	return g.tileCap == 1
}

// tileF32 reports whether the device also lowers the float32 8x8 tile (the
// collective-tile flash prefill's P V): Metal's simdgroup_matrix, sm_80's
// tf32 products.
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
