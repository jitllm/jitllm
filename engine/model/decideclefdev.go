package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
)

// Clef's head on a device (nn.ClefHeadDevice): when the State's output
// projection is on a device -- every block placed there -- the head follows
// it. The rows the prefill handed over go up whole, with the request's
// pooling rows, option owners, type rows, mask and each option's lexical mean
// (the output embedding's rows, read on the host's generated kernels: the
// embedding is a gather, as the prompt's own token embedding is); one logit
// per option comes back.

// clefDevHead is the head as the device reads it, built once per model: its
// matrices f32, row by row through the generated row decoders.
func (m *Model) clefDevHead() *nn.ClefHead {
	m.bind.Lock()
	defer m.bind.Unlock()
	if m.clefDev != nil {
		return m.clefDev
	}
	h := m.clef
	mat := func(t tensor) []float32 {
		out := make([]float32, t.rows*t.k)
		for r := 0; r < t.rows; r++ {
			row := out[r*t.k : (r+1)*t.k]
			if t.packed != nil && len(t.packed.QS) != 0 {
				nn.RowPacked32JIT(row, t.typ, t.packed, r, t.rows, t.k)
			} else {
				nn.Row32JIT(row, t.typ, t.data, r, t.k)
			}
		}
		return out
	}
	dh := &nn.ClefHead{D: m.Cfg.NEmbd, W: h.width, Heads: h.heads, FFN: h.ffn, Routing: h.routing,
		Eps: float32(h.eps), RMSEps: float32(m.Cfg.RMSEps), OutNorm: m.outNorm,
		HidW: h.hidW, HidB: h.hidB, PMem: mat(h.pMem), PQ: mat(h.pQ), POQ: mat(h.pOQ), PG: mat(h.pG),
		POC: mat(h.pOC), POL: mat(h.pOL), SumW: h.sumW, SumB: h.sumB, FieldW: h.fieldW, FieldB: h.fieldB,
		OptW: h.optW, OptB: h.optB, Types: h.types, Sc: mat(h.sc), ScB: h.scB, ScOut: h.scOut, ScOutB: h.scOutB}
	copy(dh.Scales[:], h.scales)
	for i := range h.blocks {
		b := &h.blocks[i]
		db := nn.ClefHeadBlock{CrossW: b.crossW, CrossB: b.crossB, KVW: b.kvW, KVB: b.kvB,
			CQ: mat(b.cq), CK: mat(b.ck), CV: mat(b.cv), CO: mat(b.co), CBQ: b.cbq, CBK: b.cbk, CBV: b.cbv, CBO: b.cbo,
			FFNW: b.ffnW, FFNB: b.ffnB, Up: mat(b.up), BUp: b.bUp, Down: mat(b.down), BDown: b.bDown}
		if i >= h.routing {
			db.SelfW, db.SelfB = b.selfW, b.selfB
			db.SQ, db.SK, db.SV, db.SO = mat(b.sq), mat(b.sk), mat(b.sv), mat(b.so)
			db.SBQ, db.SBK, db.SBV, db.SBO = b.sbq, b.sbk, b.sbv, b.sbo
		}
		dh.Blocks = append(dh.Blocks, db)
	}
	m.clefDev = dh
	return dh
}

// clefOnDevice runs the head on the State's device when its output
// projection is there. ok false with no error: the host runs it.
func (d *Decider) clefOnDevice(p *clefPrompt, out []float32) (bool, error) {
	s := d.st
	dev, isDev := s.ld.(nn.ClefHeadDevice)
	if !isDev || s.head == nil || d.violation == "host-head" || s.m.outNormB != nil || s.c.LayerNorm {
		return false, nil
	}
	r, c := d.clef, d.m.Cfg
	D := c.NEmbd
	n, nq, no := len(p.ids), len(p.qs), len(p.opts)
	in := &d.clefIn
	in.N, in.NQ, in.NO = n, nq, no
	in.Resid = r.hid[:n*D]
	rows := nq + 1 + no
	in.Pool = growF(in.Pool, rows*n)
	clear(in.Pool)
	span := func(row int, sp clefSpan) {
		inv := 1 / float32(sp.hi-sp.lo)
		for j := sp.lo; j < sp.hi; j++ {
			in.Pool[row*n+j] = inv
		}
	}
	for i, sp := range p.qs {
		span(i, sp)
	}
	in.Pool[nq*n+n-1] = 1
	for i, sp := range p.opts {
		span(nq+1+i, sp)
	}
	if cap(in.Owner) < no {
		in.Owner = make([]uint32, no)
	}
	if cap(in.QType) < nq {
		in.QType = make([]uint32, nq)
	}
	in.Owner, in.QType = in.Owner[:no], in.QType[:nq]
	for i, o := range p.owner {
		in.Owner[i] = uint32(o)
	}
	for i, t := range p.qtype {
		in.QType[i] = uint32(t)
	}
	in.Mask = growF(in.Mask, nq*no)
	neg := float32(math.Inf(-1))
	for qi := 0; qi < nq; qi++ {
		for o := 0; o < no; o++ {
			in.Mask[qi*no+o] = neg
			if int(p.owner[o]) == qi {
				in.Mask[qi*no+o] = 0
			}
		}
	}
	in.Lex = r.lex[:no*D]
	for i, sp := range p.opts {
		if err := d.clefLexical(r.lex[i*D:(i+1)*D], p.ids[sp.lo:sp.hi]); err != nil {
			return false, err
		}
	}
	if !dev.ClefHead(d.m.clefDevHead(), in, out[:no]) {
		why := "no reason given"
		if e, ok := s.ld.(nn.ErrReporter); ok && e.Err() != "" {
			why = e.Err()
		}
		return false, fmt.Errorf("model: the device holding the head declined Clef's head: %s", why)
	}
	d.clefDevRuns++
	return true, nil
}
