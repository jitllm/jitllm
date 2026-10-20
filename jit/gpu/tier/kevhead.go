package tier

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
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
	g.mu.Lock()
	d := g.head
	g.mu.Unlock()
	if d == nil {
		return false
	}
	return d.kevHead(h, q, k, out)
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

func (g *devTier) kevHead(h *nn.KevHead, q, k, out []float32) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return false
	}
	n := len(out)
	if n < 1 || len(q) != h.D || len(k) != n*h.D {
		g.LastErr = fmt.Sprintf("Kev's head: %d query and %d key floats for %d options of %d", len(q), len(k), n, h.D)
		return false
	}
	hd, ok := g.khUpload(h)
	if !ok {
		return false
	}
	var pr *khProg
	for _, p := range g.kh.progs {
		if p.h == h && p.n == n {
			pr = p
			break
		}
	}
	if pr == nil {
		var err error
		if pr, err = g.khBuild(h, hd, n); err != nil {
			g.LastErr = "Kev's head: " + err.Error()
			return false
		}
		if len(g.kh.progs) == khProgs {
			for _, b := range g.kh.progs[0].own {
				b.Free()
			}
			g.kh.progs = append(g.kh.progs[:0], g.kh.progs[1:]...)
		}
		g.kh.progs = append(g.kh.progs, pr)
	}
	pr.inQ, pr.inK, pr.out, pr.err = q, k, out, nil
	g.dev.Session(pr.run)
	pr.inQ, pr.inK, pr.out = nil, nil, nil
	if pr.err != nil {
		g.LastErr = "Kev's head: " + pr.err.Error()
		return false
	}
	g.KevHeads++
	return true
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

// khBuild compiles the program for n options. Callers hold g.mu.
func (g *devTier) khBuild(h *nn.KevHead, hd *khHead, n int) (*khProg, error) {
	pr := &khProg{h: h, n: n}
	pr.run = pr.khRun
	D, P := h.D, h.P
	w := func(v []float32) backend.Buf { return hd.bufs[&v[0]] }
	var err error
	buf := func(floats int) backend.Buf {
		if err != nil {
			return nil
		}
		var b backend.Buf
		if b, err = g.dhNewBuf(floats); err != nil {
			return nil
		}
		pr.own = append(pr.own, b)
		return b
	}
	kern := func(key dhKey, build func() (*ir.Kernel, error)) backend.Kernel {
		if err != nil {
			return nil
		}
		var k backend.Kernel
		k, err = g.dhKern(key, build)
		return k
	}
	launch := func(k backend.Kernel, threads int, bufs ...backend.Buf) {
		pr.prog = append(pr.prog, dhStep{k, threads, bufs})
	}
	norm := func(dst, x backend.Buf, rows int) {
		launch(kern(dhKey{name: "norm", a: [10]int{D, rows}, f: [3]float32{h.RMSEps}, b: [2]bool{true, false}},
			func() (*ir.Kernel, error) { return oldkernels.DHNormRows(D, rows, h.RMSEps, true, false) }),
			rows, x, w(h.OutNorm), w(h.OutNorm), dst)
	}
	linear := func(dst, x backend.Buf, rows int, wt, bias []float32) {
		launch(kern(dhKey{name: "matabt", a: [10]int{1, rows, P, D, D, D, P, 0, 0, 0}, f: [3]float32{1}, b: [2]bool{true}},
			func() (*ir.Kernel, error) { return oldkernels.DHMatABt(1, rows, P, D, D, D, P, 0, 0, 0, 1, true) }),
			rows*P, x, w(wt), w(bias), dst)
	}
	pr.q, pr.k = buf(D), buf(n*D)
	qn, kn, qp, kp := buf(D), buf(n*D), buf(P), buf(n*P)
	pr.outB = buf(n)
	norm(qn, pr.q, 1)
	norm(kn, pr.k, n)
	linear(qp, qn, 1, h.WQ, h.BQ)
	linear(kp, kn, n, h.WK, h.BK)
	sc := float32(1 / math.Sqrt(float64(P)))
	launch(kern(dhKey{name: "matabt", a: [10]int{1, 1, n, P, P, P, n, 0, 0, 0}, f: [3]float32{sc}},
		func() (*ir.Kernel, error) { return oldkernels.DHMatABt(1, 1, n, P, P, P, n, 0, 0, 0, sc, false) }),
		n, qp, kp, kp, pr.outB)
	if err != nil {
		for _, b := range pr.own {
			b.Free()
		}
		return nil, err
	}
	return pr, nil
}
