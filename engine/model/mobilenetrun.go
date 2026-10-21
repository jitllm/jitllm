package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// The MobileNet-V5 tower's encode on the host (mobilenet.go has the tower).

// mnFaultKind is a MobileNet-V5 gate's violation, one of the towerFault values
// (vision.go) set on the tower's configuration; none in every real run.
type mnFaultKind = towerFault

const mnFaultNone = towerFaultNone

// mnColRows is how many output positions one im2col tile holds: a full
// convolution is a matmul over the tile, so the tile bounds the unfolded
// input rather than the whole picture's.
const mnColRows = 4096

// mnGrow sizes the encode's working memory for the tower, once, and puts
// every depthwise shape and the attention kernels in the State's JIT.
func (s *State) mnGrow() {
	v := s.vis
	if v.mn != nil {
		return
	}
	mt := v.t.mn
	r := &mnRun{}
	act := mt.hs * mt.ws * mt.stemOut
	var exp, pad, col, q, kv, scoresM int
	pin := func(h, w, k, st, c int) int {
		ho, _, _ := mnSame(h, k, st)
		wo, _, _ := mnSame(w, k, st)
		return ((ho-1)*st + k) * ((wo-1)*st + k) * c
	}
	stemK := s.vis.t.patchW.k
	pad = pin(v.t.Cfg.ImageSz, v.t.Cfg.ImageSz, 3, 2, 3)
	col = min(mt.hs*mt.ws, mnColRows) * stemK
	for li := s.lo; li < s.hi; li++ {
		b := s.m.layers[li].mn
		act = max(act, b.hin*b.win*b.cin, b.hout*b.wout*b.cout)
		switch b.kind {
		case mnEdge:
			exp = max(exp, b.hout*b.wout*b.cexp)
			pad = max(pad, pin(b.hin, b.win, 3, b.stride, b.cin))
			col = max(col, min(b.hout*b.wout, mnColRows)*pad32(9*b.cin))
		case mnInverted:
			exp = max(exp, b.hmid*b.wmid*b.cexp+b.hout*b.wout*b.cexp)
			kv = max(kv, b.hmid*b.wmid*b.cin)
			if b.dwStart != nil {
				pad = max(pad, pin(b.hin, b.win, b.kStart, b.sStart, b.cin))
			}
			if b.dwMid != nil {
				pad = max(pad, pin(b.hmid, b.wmid, b.kMid, b.sMid, b.cexp))
			}
		case mnAttention:
			n := b.hin * b.win
			q = max(q, n*b.heads*b.kd)
			kv = max(kv, n*b.cin)
			exp = max(exp, b.hkv*b.wkv*b.cin)
			scoresM = max(scoresM, b.hkv*b.wkv)
			if b.kDown != nil {
				pad = max(pad, pin(b.hin, b.win, b.kvK, 2, b.cin))
			}
			s.jit.AttnSetFor(b.kd, b.kd, b.kd, cpu.KVF32)
		}
	}
	a, bt := s.m.layers[s.lo+mt.tapA].mn, s.m.layers[s.lo+mt.tapB].mn
	ncat := a.hout * a.wout
	exp = max(exp, ncat*mt.fuseExp.rows)
	r.act[0], r.act[1] = make([]float32, act), make([]float32, act)
	r.exp, r.pad, r.col = make([]float32, exp), make([]float32, pad), make([]float32, col)
	r.q, r.ao = make([]float32, q), make([]float32, q)
	r.kin = make([]float32, kv)
	r.k, r.v = make([]float32, q), make([]float32, q)
	// One score row a pool chunk (mnChunk makes at most 4*Workers of them).
	r.scores = make([]float32, scoresM*4*s.jit.Workers())
	r.tapA = make([]float32, ncat*a.cout)
	r.cat = make([]float32, ncat*max(a.cout+bt.cout, mt.fuseExp.rows, mt.fuseProj.rows))
	r.pooled = make([]float32, mt.grid*mt.grid*max(mt.fuseProj.rows, v.t.Cfg.ProjDim))
	for _, sh := range mt.dw {
		s.jit.AddDWConv(sh)
	}
	r.fn = s.mnItems
	v.mn = r
}

// mnBegin is encodeBegin for the tower: the picture through the stem into the
// encode's activation, and the encode in progress set to its first block.
func (s *State) mnBegin(px []float32) error {
	v := s.vis
	t, c := v.t, v.t.Cfg
	mt := t.mn
	if len(px) != c.ImageSz*c.ImageSz*3 {
		return fmt.Errorf("model: Encode: %d pixels, want %dx%dx3", len(px), c.ImageSz, c.ImageSz)
	}
	s.mnGrow()
	r := v.mn
	r.cur = 0
	out := r.act[0][:mt.hs*mt.ws*mt.stemOut]
	if err := s.mnConv(out, px, c.ImageSz, c.ImageSz, 3, 3, 2, t.patchW); err != nil {
		return err
	}
	n := mt.hs * mt.ws
	if t.patchB != nil {
		s.axpyRows(out, t.patchB, nil, mt.stemOut, n, s.rowChunk(n, mt.stemOut), 1)
	}
	s.normRows(out, out, mt.stemNorm, nil, mt.stemOut, n, false, c.Eps, s.rowChunk(n, mt.stemOut))
	s.actAll(out, nn.ActGELU)
	r.h, r.w, r.c = mt.hs, mt.ws, mt.stemOut
	v.stage("stem", out)
	g := mt.grid
	v.gh, v.gw = g, g
	v.prog = encodeProg{on: true, next: s.lo, gh: g, gw: g}
	return nil
}

// mnPad copies src (h x w rows of c) into r.pad, zero-bordered for a k x k
// window at stride st under SAME padding, and returns the padded row's
// width in samples and the depthwise row shape.
func (s *State) mnPad(src []float32, h, w, c, k, st int) (wp int, sh cpu.DWShape) {
	r := s.vis.mn
	ho, top, _ := mnSame(h, k, st)
	wo, left, _ := mnSame(w, k, st)
	if s.tfault() == mnFaultSymmetric {
		top, left = k/2, k/2
	}
	hp, wp := (ho-1)*st+k, (wo-1)*st+k
	pad := r.pad[:hp*wp*c]
	n := min(w, wp-left)
	for y := 0; y < hp; y++ {
		row := pad[y*wp*c : (y+1)*wp*c]
		sy := y - top
		if sy < 0 || sy >= h {
			clear(row)
			continue
		}
		clear(row[:left*c])
		copy(row[left*c:], src[sy*w*c:sy*w*c+n*c])
		clear(row[(left+n)*c:])
	}
	return wp, cpu.DWShape{K: k, Stride: st, WP: wp, Chans: c, WOut: wo}
}

// mnDW is a depthwise convolution of src (h x w rows of c) into dst under
// SAME padding, one generated row kernel call per output row across the
// pool, and returns the output's grid.
func (s *State) mnDW(dst, src []float32, h, w, c, k, st int, filt []float32) (int, int) {
	r := s.vis.mn
	ho, _, _ := mnSame(h, k, st)
	wp, sh := s.mnPad(src, h, w, c, k, st)
	r.op, r.dwOut, r.dwIn, r.dwFilt, r.dwShape, r.dwStride = mnOpDW, dst, r.pad, filt, sh, st*wp*c
	r.chunk = s.mnChunk(ho)
	s.jit.Parallel(ho, r.chunk, r.fn)
	r.dwOut, r.dwIn, r.dwFilt = nil, nil, nil
	return ho, sh.WOut
}

// mnChunk is a region's chunk over n items: at most 4*Workers chunks, which
// is what the attention's score rows are sized for.
func (s *State) mnChunk(n int) int {
	w := 4 * s.jit.Workers()
	return max(1, (n+w-1)/w)
}

// mnItems is one chunk of the tower's pool region.
func (s *State) mnItems(lo, hi int) {
	r := s.vis.mn
	switch r.op {
	case mnOpDW:
		row := r.dwShape.WOut * r.dwShape.Chans
		for y := lo; y < hi; y++ {
			s.jit.DWConvRow(r.dwOut[y*row:(y+1)*row], r.dwIn[y*r.dwStride:], r.dwFilt, r.dwShape)
		}
	case mnOpAttn:
		m, kd, heads := r.attM, r.attKD, r.attHeads
		sc := r.scores[(lo/r.chunk)*m : (lo/r.chunk+1)*m]
		for i := lo; i < hi; i++ {
			for h := 0; h < heads; h++ {
				o := (i*heads + h) * kd
				r.att.AttnScores(sc, r.k, r.q[o:o+kd], m)
				nn.Softmax32JIT(sc, m)
				r.att.AttnAcc(r.ao[o:o+kd], r.v, sc, m)
			}
		}
	}
}

// mnConv is a full k x k convolution of src (h x w rows of c) at stride st
// under SAME padding, into dst: an im2col tile of output positions, tap-major
// as the converter laid the weight out, and the packed matmul over it.
func (s *State) mnConv(dst, src []float32, h, w, c, k, st int, wt tensor) error {
	r := s.vis.mn
	ho, _, _ := mnSame(h, k, st)
	wo, _, _ := mnSame(w, k, st)
	wp, _ := s.mnPad(src, h, w, c, k, st)
	kp := wt.k
	if kp < k*k*c {
		return fmt.Errorf("model: a %dx%d convolution over %d channels reads %d, and its weight is %d wide", k, k, c, k*k*c, kp)
	}
	for p0 := 0; p0 < ho*wo; p0 += mnColRows {
		p1 := min(p0+mnColRows, ho*wo)
		col := r.col[:(p1-p0)*kp]
		for p := p0; p < p1; p++ {
			y, x := p/wo, p%wo
			row := col[(p-p0)*kp : (p-p0+1)*kp]
			for ky := 0; ky < k; ky++ {
				at := ((y*st+ky)*wp + x*st) * c
				copy(row[ky*k*c:(ky+1)*k*c], r.pad[at:at+k*c])
			}
			clear(row[k*k*c:])
		}
		s.jit.NewInput()
		if err := s.mm(dst[p0*wt.rows:p1*wt.rows], wt, col, p1-p0); err != nil {
			return err
		}
	}
	return nil
}

// mnBlockRun runs block li of the tower on the host: the encode's activation
// in, the block's output in the other half.
func (s *State) mnBlockRun(li int) error {
	if err := s.m.pageIn(li); err != nil {
		return err
	}
	v := s.vis
	r, l := v.mn, &s.m.layers[li]
	b := l.mn
	eps := v.t.Cfg.Eps
	if r.h != b.hin || r.w != b.win || r.c != b.cin {
		return fmt.Errorf("model: tower block %d reads %dx%dx%d and the encode holds %dx%dx%d",
			li-s.lo, b.hin, b.win, b.cin, r.h, r.w, r.c)
	}
	x := r.act[r.cur][:b.hin*b.win*b.cin]
	y := r.act[1-r.cur][:b.hout*b.wout*b.cout]
	nin, nout := b.hin*b.win, b.hout*b.wout
	switch b.kind {
	case mnEdge:
		e := r.exp[:nout*b.cexp]
		if err := s.mnConv(e, x, b.hin, b.win, b.cin, 3, b.stride, l.up); err != nil {
			return err
		}
		s.normRows(e, e, b.norm1, nil, b.cexp, nout, false, eps, s.rowChunk(nout, b.cexp))
		s.actAll(e, nn.ActGELU)
		s.jit.NewInput()
		if err := s.mm(y, l.down, e, nout); err != nil {
			return err
		}
		s.normRows(y, y, b.norm2, nil, b.cout, nout, false, eps, s.rowChunk(nout, b.cout))
	case mnInverted:
		in, h, w := x, b.hin, b.win
		if b.dwStart != nil {
			t1 := r.kin[:b.hmid*b.wmid*b.cin]
			s.mnDW(t1, x, h, w, b.cin, b.kStart, b.sStart, b.dwStart)
			nm := b.hmid * b.wmid
			s.normRows(t1, t1, b.dwStartNorm, nil, b.cin, nm, false, eps, s.rowChunk(nm, b.cin))
			in, h, w = t1, b.hmid, b.wmid
		}
		nm := h * w
		e := r.exp[:nm*b.cexp]
		s.jit.NewInput()
		if err := s.mm(e, l.up, in, nm); err != nil {
			return err
		}
		s.normRows(e, e, b.norm1, nil, b.cexp, nm, false, eps, s.rowChunk(nm, b.cexp))
		s.actAll(e, nn.ActGELU)
		if b.dwMid != nil {
			e2 := r.exp[nm*b.cexp : nm*b.cexp+nout*b.cexp]
			s.mnDW(e2, e, h, w, b.cexp, b.kMid, b.sMid, b.dwMid)
			s.normRows(e2, e2, b.dwMidNorm, nil, b.cexp, nout, false, eps, s.rowChunk(nout, b.cexp))
			s.actAll(e2, nn.ActGELU)
			e = e2
		}
		s.jit.NewInput()
		if err := s.mm(y, l.down, e, nout); err != nil {
			return err
		}
		s.normRows(y, y, b.norm2, nil, b.cout, nout, false, eps, s.rowChunk(nout, b.cout))
	case mnAttention:
		xn := r.kin[:nin*b.cin]
		s.normRows(xn, x, b.norm1, nil, b.cin, nin, false, eps, s.rowChunk(nin, b.cin))
		qd := b.heads * b.kd
		q := r.q[:nin*qd]
		s.jit.NewInput()
		if err := s.mm(q, l.wq, xn, nin); err != nil {
			return err
		}
		s.scale(q, float32(1/math.Sqrt(float64(b.kd))))
		m := b.hkv * b.wkv
		for _, side := range []struct {
			down, norm []float32
			wt         tensor
			out        []float32
		}{{b.kDown, b.kDownNorm, l.wk, r.k[:m*b.kd]}, {b.vDown, b.vDownNorm, l.wv, r.v[:m*b.kd]}} {
			src := xn
			if side.down != nil {
				d := r.exp[:m*b.cin]
				s.mnDW(d, xn, b.hin, b.win, b.cin, b.kvK, 2, side.down)
				s.normRows(d, d, side.norm, nil, b.cin, m, false, eps, s.rowChunk(m, b.cin))
				src = d
			}
			s.jit.NewInput()
			if err := s.mm(side.out, side.wt, src, m); err != nil {
				return err
			}
		}
		r.op, r.att, r.attM, r.attKD, r.attHeads = mnOpAttn, s.jit.AttnSetFor(b.kd, b.kd, b.kd, cpu.KVF32), m, b.kd, b.heads
		r.chunk = s.mnChunk(nin)
		s.jit.Parallel(nin, r.chunk, r.fn)
		r.att = nil
		s.jit.NewInput()
		if err := s.mm(y, l.wo, r.ao[:nin*qd], nin); err != nil {
			return err
		}
	}
	if b.residual && s.tfault() != mnFaultNoResidual {
		s.addInto(y, x)
	}
	r.cur = 1 - r.cur
	r.h, r.w, r.c = b.hout, b.wout, b.cout
	if li-s.lo == v.t.mn.tapA {
		copy(r.tapA[:nout*b.cout], y)
	}
	return nil
}

// mnDevRun runs blocks [lo, hi) of the tower on the device: the encode's
// activation up, the last block's output back into the other half, and the
// fusion adapter's first input back as its block writes it.
func (s *State) mnDevRun(lo, hi int) error {
	v := s.vis
	r := v.mn
	b0, b1 := s.m.layers[lo].mn, s.m.layers[hi-1].mn
	if r.h != b0.hin || r.w != b0.win || r.c != b0.cin {
		return fmt.Errorf("model: tower block %d reads %dx%dx%d and the encode holds %dx%dx%d",
			lo-s.lo, b0.hin, b0.win, b0.cin, r.h, r.w, r.c)
	}
	cd, ok := s.ld.(nn.ConvDevice)
	if !ok {
		return fmt.Errorf("model: the device holds tower blocks [%d,%d) and cannot run a convolution", lo-s.lo, hi-s.lo)
	}
	tap, tapOut := -1, []float32(nil)
	if t := s.lo + v.t.mn.tapA; t >= lo && t < hi {
		a := s.m.layers[t].mn
		tap, tapOut = t, r.tapA[:a.hout*a.wout*a.cout]
	}
	x := r.act[r.cur][:b0.hin*b0.win*b0.cin]
	y := r.act[1-r.cur][:b1.hout*b1.wout*b1.cout]
	if !cd.ConvLayers(lo, hi, x, y, tap, tapOut) {
		why := "no reason given"
		if e, ok := s.ld.(nn.ErrReporter); ok && e.Err() != "" {
			why = e.Err()
		}
		return fmt.Errorf("model: the device failed tower block(s) [%d,%d): %s", lo-s.lo, hi-s.lo, why)
	}
	r.cur = 1 - r.cur
	r.h, r.w, r.c = b1.hout, b1.wout, b1.cout
	return nil
}

// mnFinish is encodeFinish for the tower: the fusion adapter over the last
// two stages' outputs, then the embedder.
func (s *State) mnFinish() ([]float32, error) {
	v := s.vis
	t, c := v.t, v.t.Cfg
	mt, r := t.mn, v.mn
	a, b := s.m.layers[s.lo+mt.tapA].mn, s.m.layers[s.lo+mt.tapB].mn
	ha, wa := a.hout, a.wout
	n := ha * wa
	ccat := a.cout + b.cout
	cat := r.cat[:n*ccat]
	fin := r.act[r.cur][:b.hout*b.wout*b.cout]
	fy, fx := ha/b.hout, wa/b.wout
	for y := 0; y < ha; y++ {
		for x := 0; x < wa; x++ {
			p := y*wa + x
			copy(cat[p*ccat:p*ccat+a.cout], r.tapA[p*a.cout:(p+1)*a.cout])
			q := (y/fy)*b.wout + x/fx
			if s.tfault() == mnFaultUpsampleT {
				q = (x/fx)*b.wout + y/fy
			}
			copy(cat[p*ccat+a.cout:(p+1)*ccat], fin[q*b.cout:(q+1)*b.cout])
		}
	}
	eps := c.Eps
	ce := mt.fuseExp.rows
	e := r.exp[:n*ce]
	s.jit.NewInput()
	if err := s.mm(e, mt.fuseExp, cat, n); err != nil {
		return nil, err
	}
	s.normRows(e, e, mt.fuseExpNorm, nil, ce, n, false, eps, s.rowChunk(n, ce))
	s.actAll(e, nn.ActGELU)
	E := c.NEmbd
	f := r.cat[:n*E]
	s.jit.NewInput()
	if err := s.mm(f, mt.fuseProj, e, n); err != nil {
		return nil, err
	}
	s.normRows(f, f, mt.fuseProjNorm, nil, E, n, false, eps, s.rowChunk(n, E))
	// The k x k average pool to the output grid: the sum, then one multiply.
	g, k := mt.grid, mt.pool
	nt := g * g
	pooled := r.pooled[:nt*E]
	for gy := 0; gy < g; gy++ {
		for gx := 0; gx < g; gx++ {
			dst := pooled[(gy*g+gx)*E : (gy*g+gx+1)*E]
			clear(dst)
			for dy := 0; dy < k; dy++ {
				for dx := 0; dx < k; dx++ {
					p := (gy*k+dy)*wa + gx*k + dx
					if s.tfault() == mnFaultPoolFirst {
						p = (gy*k)*wa + gx*k
					}
					nn.Axpy32JIT(dst, f[p*E:(p+1)*E], 1)
				}
			}
			if k > 1 {
				nn.Scale32JIT(dst, 1/float32(k*k))
			}
		}
	}
	s.normRows(pooled, pooled, mt.fNorm, nil, E, nt, false, eps, s.rowChunk(nt, E))
	v.stage("fusion", pooled)
	if v.beforeProj != nil {
		v.beforeProj(pooled)
	}
	// The embedder: sqrt(NEmbd), its norm, the matrix and an unweighted norm.
	s.scale(pooled, float32(math.Sqrt(float64(E))))
	s.normRows(pooled, pooled, t.projNorm, nil, E, nt, false, eps, s.rowChunk(nt, E))
	s.jit.NewInput()
	out := v.out[:nt*c.ProjDim]
	if err := s.mm(out, t.projW, pooled, nt); err != nil {
		return nil, err
	}
	s.normRows(out, out, mt.ones, nil, c.ProjDim, nt, false, eps, s.rowChunk(nt, c.ProjDim))
	v.stage("out", out)
	return out, nil
}
