package model

import (
	"unsafe"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
)

// A convolutional tower block on the CPU device (frontconv.go): composed
// once at the vision State's placement at its own shape -- its input grid's
// rows, each block a request of its own rows -- and run over the encode's
// own buffers (mnRun's activation halves), the block's input in one and its
// output into the other, as main's runner hands them over. Each run is
// counted served or declined by name (CPUBlockSteps); main's runner runs a
// block the CPU declines.

// cpuComposeConv composes convolutional block li for the CPU into cb.
func (s *State) cpuComposeConv(li int, cb *cpuBlock) {
	l := &s.m.layers[li]
	g := s.m.Composing().BlockGraph(li)
	switch {
	case !jit.HostCPULowers():
		cb.why = jit.DeclineISA
		return
	case g == nil:
		cb.why = jit.DeclineNoGraph
		return
	}
	vi := int32(li - s.m.tower.base)
	formats := map[jlm.Role]jlm.Type{}
	for _, o := range g.Ops {
		for _, w := range o.Weight {
			if l.mn.filter(w.Role) != nil {
				formats[w.Role] = jlm.TypeF32
			} else if e, ok := s.m.container.Find(w.Role, vi, -1); ok {
				formats[w.Role] = e.Type
			}
		}
	}
	rows := l.mn.hin * l.mn.win
	page := 1
	if s.kv.layers[li].p > 0 {
		a := s.nc()
		if a == nil || a.arena == nil {
			cb.why = jit.DeclineNoRoom
			return
		}
		page = a.page
		s.cpuPages(1)
		if len(s.cpu.rowsTab) < rows*4 {
			s.cpu.rowsTab = make([]uint32, rows*4)
		}
	}
	req := jit.Request{Graph: g, Formats: formats, Step: jit.Step{Rows: rows, Page: page, Keys: page, ActWindow: s.jit.ActWindow()},
		Device: s.composedCPU(), Apart: true}
	f, err := s.m.cpuFormOf(req, g, rows, true, true, false)
	if err != nil {
		cb.why = s.cpuFormDecline(err)
		return
	}
	cb.graph, cb.f, cb.why = g, f, jit.DeclineServes
	s.cpu.composed[li]++
}

// cpuConvRun runs convolutional block li on the CPU the composer's way
// over the encode's activation, its output into the other half, or says
// why not.
func (s *State) cpuConvRun(li int) jit.Decline {
	cb := s.cpu.blocks[li]
	if cb == nil {
		return jit.DeclineUnplaced
	}
	if cb.why != jit.DeclineServes {
		return cb.why
	}
	l, f := &s.m.layers[li], cb.f
	if why := s.cpuWeights(f, li, l); why != jit.DeclineServes {
		return why
	}
	r, b := s.vis.mn, l.mn
	nin, nout := b.hin*b.win, b.hout*b.wout
	u := max(f.unit, 1)
	x, y := r.act[r.cur], r.act[1-r.cur]
	if r.h != b.hin || r.w != b.win || r.c != b.cin || len(x) < (nin+u-1)/u*u*b.cin || len(y) < (nout+u-1)/u*u*b.cout {
		return jit.DeclineNoRoom
	}
	bd := &s.cpu.bound
	*bd = cpuBound{li: li, l: l, own: s.cpuOwn(f), n: nin}
	bd.io[0], bd.io[1] = unsafe.Pointer(&x[0]), unsafe.Pointer(&y[0])
	if pg := &s.kv.layers[li]; pg.p > 0 {
		if !s.lendNC(li) {
			return jit.DeclineNoRoom
		}
		defer s.unlend(li)
		for i := range nin {
			t := s.cpu.rowsTab[i*4 : i*4+4]
			t[0], t[1], t[2], t[3] = 0, 0, uint32(pg.p), uint32(i)
			s.cpuFaultRowWords(t)
		}
		bd.kv, bd.pages, bd.rows = pg, s.cpu.pages, s.cpu.rowsTab
	}
	s.cpu.call[0] = uint32(nin)
	for i := range f.codes {
		s.cpuCall(bd, &f.codes[i])
	}
	s.mnHandOver(li, y[:nout*b.cout])
	return jit.DeclineServes
}

// mnPages is a convolutional tower's history: an attention block's keys
// are its key grid's rows, one key head wide; a convolution keeps none.
func (s *State) mnPages() {
	for li := s.lo; li < s.hi; li++ {
		b, pg := s.m.layers[li].mn, &s.kv.layers[li]
		pg.p, pg.pp = 0, 0
		if b != nil && b.kind == mnAttention {
			pg.p = b.hkv * b.wkv
			pg.pp = pg.p * b.kd
		}
	}
}

// filter is the block's depthwise filter of role r as a weight (f32, its
// taps the rows), nil for a role it holds none of.
func (b *mnBlock) filter(r jlm.Role) *tensor {
	for i, fr := range [...]jlm.Role{jlm.RoleVMNDwStart, jlm.RoleVMNDwMid, jlm.RoleVMNKDown, jlm.RoleVMNVDown} {
		if fr == r && b.filt[i].rows > 0 {
			return &b.filt[i]
		}
	}
	return nil
}

// filters makes the block's depthwise filters' weights, each a view of the
// filter read at load.
func (b *mnBlock) filters() {
	for i, f := range [...]struct {
		v    []float32
		taps int
	}{{b.dwStart, b.kStart}, {b.dwMid, b.kMid}, {b.kDown, b.kvK}, {b.vDown, b.kvK}} {
		if len(f.v) == 0 || f.taps == 0 {
			continue
		}
		b.filt[i] = tensor{data: unsafe.Slice((*byte)(unsafe.Pointer(&f.v[0])), 4*len(f.v)), rows: f.taps * f.taps,
			k: len(f.v) / (f.taps * f.taps)}
	}
}

// convRoleWeight is convolutional block l's matrix or filter of role r.
func convRoleWeight(l *layer, r jlm.Role) *tensor {
	if t := l.mn.filter(r); t != nil {
		return t
	}
	for _, x := range convTensors(l) {
		if x.r == r && x.t.e != nil {
			return x.t
		}
	}
	return nil
}

// convRoleVector is convolutional block l's vector of role r.
func convRoleVector(l *layer, r jlm.Role) []float32 {
	for _, v := range convVectors(l.mn) {
		if v.Role == r && v.Taps == 0 {
			return v.V
		}
	}
	return nil
}

// filterT is filter(r) with its taps' rows and columns swapped, made once:
// a gate's violation (cpuFaultFilterT).
func (b *mnBlock) filterT(r jlm.Role) *tensor {
	t := b.filter(r)
	if t == nil {
		return nil
	}
	i := 0
	for &b.filt[i] != t {
		i++
	}
	if b.filtT[i].rows > 0 {
		return &b.filtT[i]
	}
	k := 1
	for k*k < t.rows {
		k++
	}
	src := unsafe.Slice((*float32)(unsafe.Pointer(&t.data[0])), t.rows*t.k)
	dst := make([]float32, len(src))
	for ky := range k {
		for kx := range k {
			copy(dst[(kx*k+ky)*t.k:(kx*k+ky+1)*t.k], src[(ky*k+kx)*t.k:(ky*k+kx+1)*t.k])
		}
	}
	b.filtT[i] = tensor{data: unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), 4*len(dst)), rows: t.rows, k: t.k}
	return &b.filtT[i]
}
