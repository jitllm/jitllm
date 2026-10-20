package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
)

// Kev's pointer head on the device (nn.KevHeadDevice), on the decision
// head's kernels (dechead.go): the pointer row and the option rows go up,
// each is RMSNormed by the output norm and projected by its half of
// cls.output, and the query's projection is dotted with every key's over
// sqrt(P) -- one launch each, n scores back. The head's weights go up once
// (charged to the budget); a program is built per option count and kept.

type khState struct {
	heads map[*nn.KevHead]*khHead
	progs []*khProg
}

type khHead struct {
	bufs  map[*float32]backend.Buf
	bytes uint64
}

type khProg struct {
	h          *nn.KevHead
	n          int
	prog       []dhStep
	own        []backend.Buf
	q, k, outB backend.Buf
	inQ, inK   []float32
	out        []float32
	err        error
	run        func(backend.Session)
}

// khProgs is how many option counts a device keeps programs for.
const khProgs = 8

// KevHead runs Kev's head on the device holding the output projection.
func (g *GPU) KevHead(h *nn.KevHead, q, k, out []float32) bool {
	return false
}

// KevHead is the session's device's.
func (s *gpuSession) KevHead(h *nn.KevHead, q, k, out []float32) bool {
	defer s.leave(s.step())
	return s.g.KevHead(h, q, k, out)
}

func (g *devTier) khUpload(h *nn.KevHead) (*khHead, bool) {
	if g.kh.heads == nil {
		g.kh.heads = map[*nn.KevHead]*khHead{}
	}
	if r, ok := g.kh.heads[h]; ok {
		return r, true
	}
	all := [][]float32{h.OutNorm, h.WQ, h.WK, h.BQ, h.BK}
	var need uint64
	for _, v := range all {
		need += uint64(4 * len(v))
	}
	if !g.room(need) {
		g.LastErr = fmt.Sprintf("Kev's head needs %d bytes, %d of %d used", need, g.used, g.limit)
		return nil, false
	}
	r := &khHead{bufs: map[*float32]backend.Buf{}}
	for _, v := range all {
		b, err := g.dev.Alloc(4 * len(v))
		if err == nil {
			err = b.Write(f32b(v))
		}
		if err != nil {
			if b != nil {
				b.Free()
			}
			for _, x := range r.bufs {
				x.Free()
			}
			g.LastErr = fmt.Sprintf("Kev's head: %v", err)
			return nil, false
		}
		r.bufs[&v[0]] = b
		r.bytes += uint64(4 * len(v))
	}
	g.charge(r.bytes)
	g.kh.heads[h] = r
	return r, true
}

// khClose gives back every head and program. Callers hold g.mu.
func (g *devTier) khClose() {
	for _, r := range g.kh.heads {
		g.refund(r.bytes)
		for _, b := range r.bufs {
			b.Free()
		}
	}
	for _, p := range g.kh.progs {
		for _, b := range p.own {
			b.Free()
		}
	}
	g.kh = khState{}
}

func (p *khProg) khRun(s backend.Session) {
	w := func(b backend.Buf, x []byte) {
		if p.err == nil {
			p.err = s.Write(b, x)
		}
	}
	w(p.q, f32b(p.inQ))
	w(p.k, f32b(p.inK))
	for i := range p.prog {
		if p.err != nil {
			return
		}
		st := &p.prog[i]
		p.err = s.Launch(st.k, (st.threads+127)/128, 128, st.bufs...)
	}
	if p.err == nil {
		p.err = s.Read(p.outB, f32b(p.out))
	}
}
