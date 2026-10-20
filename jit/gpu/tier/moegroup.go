package tier

import (
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
)

// groupTok is the padding of an expert's run of columns.
const groupTok = 4

// moeGroup is a batched scratch's mixture half: the routed rows' values,
// p columns of (token, expert) pairs, each expert's run padded to groupTok.
type moeGroup struct {
	rows, k, nExp, p int
	colPair, ones    backend.Buf
	rwS              backend.Buf
	rlog, rlogB, rsel, rtop, rw, rbm,
	eg, eu, egB, euB, eact, edown, edownB backend.Buf
}

func (m *moeGroup) free() {
	if m == nil {
		return
	}
	for _, b := range []backend.Buf{m.colPair, m.ones, m.rwS, m.rlog, m.rlogB, m.rsel, m.rtop, m.rw, m.rbm,
		m.eg, m.eu, m.egB, m.euB, m.eact, m.edown, m.edownB} {
		if b != nil {
			b.Free()
		}
	}
}

func (g *devTier) newMoeGroup(p *nn.LayerPlan, rows int) *moeGroup {
	defer g.scratchWin().close()
	k, e, ew := p.NExpertUsed, p.NExpert, p.ExpWidth()
	np := (rows*k + e*(groupTok-1) + groupTok - 1) / groupTok * groupTok
	m := &moeGroup{rows: rows, k: k, nExp: e, p: np}
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
		{&m.eact, np * p.NFFNExp * 4}, {&m.edown, (np + 1) * ew * 4}} {
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
