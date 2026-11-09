package model

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// ropeTabPlanes is one rotary configuration's folded constant planes, or nil
// where there is no such configuration -- a model with one rotary base has no
// local one, and a vision block has no rotary at all.
//
// nil is a real answer: LayerPlan.RopeTab is an alternative source for a table
// the tier is handed anyway, so nil means "upload the host's table".
//
// A multi-axis rotary (M-RoPE) has no planes to offer: a device builds a table
// from a row's CACHE position, and an image moves a row's rotary position off
// it, so the host's per-row table is what goes up (mrope.go).
func RopeTabPlanes(r *nn.Rope, nrot int) []float32 {
	if r == nil || nrot <= 0 || len(r.Runs) > 0 {
		return nil
	}
	return r.TabPlanes(nrot / 2)
}

// ropeAxesPlanes is the folded planes of a rotary over several position
// axes, nil for any other: the composer's table reads them with each row's
// positions on the axes (nn.RopeAxesDevice). A split rotary's table (XD-RoPE)
// holds a pair an element, element i+nrot/2 at pair i's frequency: the
// planes of the nrot/2 pairs twice.
func RopeAxesPlanes(r *nn.Rope, nrot int, split bool) []float32 {
	if r == nil || nrot <= 0 || len(r.Runs) == 0 {
		return nil
	}
	n := nrot / 2
	p := r.TabPlanes(n)
	if !split {
		return p
	}
	out := make([]float32, graph.RopeBlock(2*n))
	for k := range 2 * graph.RopeDigits {
		src, dst := p[graph.RopePlaneOff(k, n):][:n], out[graph.RopePlaneOff(k, 2*n):]
		copy(dst[copy(dst, src):], src)
	}
	out[graph.RopeScaleOff(2*n)] = p[graph.RopeScaleOff(n)]
	return out
}

// Model.splitBank is the Data of a bank that lives in expert pages. It is not
// the weight's bytes -- there is no one span of them -- and it exists because
// every consumer of an nn.Weight reads an empty Data as "no weight here".
// bankWeight is wt for a mixture's bank over fr. A bank in expert pages has no
// span to hand over, so the device gets Sheet instead: expert x's planes, held
// in their page of fr until the device has copied them. A float bank -- which
// the device reads row-major out of Data -- is gathered into one buffer, since
// that path has no Sheet; only the synthetic fixtures have one. It builds a
// Weight each call (its Sheet closure and Packed): the caller keeps it for the
// frames it was built over.
func (m *Model) BankWeight(fr *jlm.Frames, t Tensor) nn.Weight {
	c := m.Container
	if c == nil || t.E == nil || !c.ExpertPaged(t.E) {
		return Wt(t)
	}
	if t.Packed == nil {
		return m.gatheredBank(fr, t)
	}
	sheets := int(t.E.Dims[2])
	nrow := t.Rows / max(sheets, 1)
	return nn.Weight{T: t.Typ, Data: m.splitBank[:], Rows: t.Rows, K: t.K,
		Packed: &nn.Packed{Sheet: func(x int) (nn.Packed, func(), error) { return m.sheetOf(fr, t.E, x, nrow) }, Sheets: sheets}}
}

// gatheredBank is a float bank, which the device reads row-major out of Data:
// gathered into one buffer, since that path has no Sheet. Only the synthetic
// fixtures have one.
func (m *Model) gatheredBank(fr *jlm.Frames, t Tensor) nn.Weight {
	sheets := int(t.E.Dims[2])
	nrow := t.Rows / max(sheets, 1)
	var buf []byte
	for x := 0; x < sheets; x++ {
		p, release, err := m.sheetOf(fr, t.E, x, nrow)
		if err != nil {
			release()
			return Wt(Tensor{})
		}
		buf = append(buf, p.QS...)
		release()
	}
	return nn.Weight{T: t.Typ, Data: buf, Rows: t.Rows, K: t.K}
}

// sheetHold is the pins an expert sheet holds until the device has copied it:
// the expert's own page and the block's. The holds are recycled through
// Model.holdFree, with their release func built once, so handing a sheet over
// allocates nothing once the free list has filled.
type sheetHold struct {
	m       *Model
	l, lb   jlm.Lease
	live    bool // handed out and not yet put back: a second release is a no-op
	release func()
}

func (h *sheetHold) put() {
	m := h.m
	m.holdMu.Lock()
	live := h.live
	h.live = false
	m.holdMu.Unlock()
	if !live {
		return
	}
	h.l.Release()
	h.lb.Release()
	m.holdMu.Lock()
	m.holdFree = append(m.holdFree, h)
	m.holdMu.Unlock()
}

func (m *Model) newHold() *sheetHold {
	m.holdMu.Lock()
	defer m.holdMu.Unlock()
	if n := len(m.holdFree); n > 0 {
		h := m.holdFree[n-1]
		m.holdFree = m.holdFree[:n-1]
		h.live = true
		return h
	}
	h := &sheetHold{m: m, live: true}
	h.release = h.put
	return h
}

// sheetOf is expert x of bank e: its planes, held in their page until the
// returned release runs.
func (m *Model) sheetOf(fr *jlm.Frames, e *jlm.Entry, x, nrow int) (nn.Packed, func(), error) {
	c, cf := m.Container, fr
	p := c.ExpertPage(e, x)
	if p < 0 {
		return nn.Packed{}, noRelease, fmt.Errorf("model: %v expert %d has no expert page", e.Role, x)
	}
	h := m.newHold()
	// The block's own page is held too: a device assembles a bank one
	// expert page at a time, each fault can evict, and the block page holds
	// the router and shared expert the same upload reads.
	if b := int(e.Block); b >= 0 && b < int(c.H.NBlocks) {
		// Held only when in, as ensureExperts does: claiming the
		// frame of a block whose base is on the card held its size of
		// host budget empty for every sheet sent.
		if l, ok := cf.HoldFilled(b); ok {
			h.lb = l
		}
	}
	off, size := c.PageBounds(p)
	l, err := cf.Hold(p, []jlm.Range{{Off: off, N: size}})
	if err != nil {
		h.put()
		return nn.Packed{}, noRelease, err
	}
	h.l = l
	qs, d, sc := cf.Sheet(e, x)
	return nn.Packed{QS: qs, D: d, SC: sc}, h.release, nil
}

func noRelease() {}

// wt is t for a device: K is the width it was packed at, which is k but for
// a ragged tower's matrices (tensor.kp; nn.LayerPlan.KPad).
func Wt(t Tensor) nn.Weight {
	k := t.K
	if t.kp != 0 {
		k = t.kp
	}
	return nn.Weight{T: t.Typ, Data: t.Data, Rows: t.Rows, K: k, Packed: t.Packed, Entry: t.E}
}

// lnBias is a LayerNorm's bias as a device takes it: b itself, or zeros of
// the model's width where the file carries none (command-r). The host's
// kernel bakes the absence instead and never asks for this.
func (m *Model) LnBias(b []float32) []float32 {
	if b != nil {
		return b
	}
	m.lnZerosOnce.Do(func() { m.lnZeros = make([]float32, m.Cfg.NEmbd) })
	return m.lnZeros
}

// sinkSlot is 1 when any layer carries attention sinks, and the score rows
// need room for the one extra logit.
func (m *Model) SinkSlot() int {
	for i := range m.Layers {
		if m.Layers[i].Vector(jlm.RoleAttnSinks) != nil {
			return 1
		}
	}
	return 0
}

// headStages is what the model's head does past the projection, in the
// model's order.
func (m *Model) HeadStages() nn.HeadStage {
	var st nn.HeadStage
	if m.OutB != nil {
		st |= nn.HeadBias
	}
	if m.Cfg.FinalSoftcap != 0 {
		st |= nn.HeadCap
	}
	if m.Cfg.LogitScale != 1 {
		st |= nn.HeadScale
	}
	return st
}

// expSelBias is a representative selection bias, or nil when this model has
// none: it answers the shape question a device's plan asks, not a value one.
// A model must not mix biased and unbiased blocks (loadConfig refuses it).
func (m *Model) ExpSelBias() []float32 {
	for li := range m.Layers {
		if m.Cfg.MoEAt(li) && m.Layers[li].Vector(jlm.RoleExpProbsB) != nil {
			return m.Layers[li].Vector(jlm.RoleExpProbsB)
		}
	}
	return nil
}

// cpuStrides is each bank of block li's graph in the host's expert pages and
// the words from one expert's sheet to the next's there.
func (m *Model) CpuStrides(fr *jlm.Frames, li int, g *graph.Graph) map[jlm.Role]int {
	var st map[jlm.Role]int
	for _, o := range g.Ops {
		for _, w := range o.Weight {
			t := m.bankTensor(li, w.Role)
			if t == nil || t.E == nil {
				continue
			}
			if _, stride, ok := fr.BankView(t.E); ok {
				if st == nil {
					st = map[jlm.Role]int{}
				}
				st[w.Role] = int(stride / 4)
			}
		}
	}
	return st
}

// bankTensor is block li's routed bank of role r, nil for any other role.
func (m *Model) bankTensor(li int, r jlm.Role) *Tensor {
	if li >= len(m.Layers) || !jlm.ExpertBank(r) {
		return nil
	}
	l := &m.Layers[li]
	return [...]*Tensor{&l.Gate, &l.Up, &l.Down}[r-jlm.RoleExpGateBank]
}

// ropeMarksCap is the most rotary marks a sequence of maxSeq positions holds:
// a picture's grid and the text after it, a picture at least a row; none on
// a model whose rotary turns over one axis, every row at its cache position.
func (m *Model) RopeMarksCap(maxSeq int) int {
	if len(m.Rope.Runs) == 0 {
		return 0
	}
	return 2*maxSeq + 1
}

// cpuTablePlanes is rotary table i's planes as its composed form reads them
// (composer.TablePlanes) for an instance of context ctx: its rotary's
// (RopeFor), folded over its axes where it has several, or its sliding
// blocks'; nil where the model has no such rotary.
func (m *Model) CpuTablePlanes(i, ctx int) []float32 {
	r := m.RopeFor(ctx)
	switch {
	case i == 0 && len(r.Runs) > 0:
		return RopeAxesPlanes(r, m.Cfg.NRot, m.Cfg.RopeXD)
	case i == 0:
		return RopeTabPlanes(r, m.Cfg.NRot)
	}
	return RopeTabPlanes(m.RopeSWA, m.Cfg.LocalNRot())
}

// RopeFor is the rotary an instance of context ctx turns: the long factors'
// (RopeLong) where ctx passes the context the short factors were trained to,
// llama.cpp's rule on its context, and the model's rotary otherwise. An
// instance turns one rotary for its whole history.
func (m *Model) RopeFor(ctx int) *nn.Rope {
	if m.RopeLong != nil && ctx > m.LongCtx {
		return m.RopeLong
	}
	return &m.Rope
}

// namedVector is the vector a field of block l holds under r, and whether a
// field holds r at all: a role a field holds is read from it, nil or not (a
// gate's violation takes a vector out through its field), and every other
// role from the block's table (layer.vecs).
func (l *Layer) NamedVector(r jlm.Role) ([]float32, bool) {
	switch r {
	case jlm.RoleNextnENorm, jlm.RoleNextnHNorm:
		if l.Mtp == nil {
			return nil, true
		}
		if r == jlm.RoleNextnENorm {
			return l.Mtp.Enorm, true
		}
		return l.Mtp.Hnorm, true
	case jlm.RoleAttnNorm:
		return l.AttnNorm, true
	case jlm.RoleFFNNorm:
		return l.FfnNorm, true
	case jlm.RoleAttnNormBias:
		return l.AttnNormB, true
	case jlm.RoleFFNNormBias:
		return l.FfnNormB, true
	case jlm.RoleAttnQNorm:
		return l.QNorm, true
	case jlm.RoleAttnKNorm:
		return l.KNorm, true
	case jlm.RoleVClamp:
		return l.Clamp, true
	case jlm.RoleLayerOutScale:
		return l.OutScaleRow, true
	}
	return nil, false
}

// Held is block l's matrix of role r, nil for one it does not hold: what
// the loader bound under no role of its own -- a prediction block's
// matrices, the convolution's taps cut from a vector -- held under the role
// a graph reads it by.
func (l *Layer) Held(r jlm.Role) *Tensor {
	if t := l.Matrix(r); t != nil {
		return t
	}
	switch r {
	case jlm.RoleNextnEHProj:
		if l.Mtp == nil {
			return nil
		}
		return &l.Mtp.EhProj
	case jlm.RoleNextnEmbd:
		if l.Mtp == nil {
			return nil
		}
		return &l.Mtp.Embd
	case jlm.RoleSSMConv1d:
		return &l.SsmConvTaps
	}
	return nil
}

// Bytes is block l's matrix of role r as a form reads it: its bytes, its
// planes where it is packed, and its rows; nil for one it does not hold.
func (l *Layer) Bytes(r jlm.Role) ([]byte, *nn.Packed, int) { return l.Held(r).Bytes() }

// Floats is block l's vector of role r, nil for one it does not hold.
func (l *Layer) Floats(r jlm.Role) []float32 {
	if v, ok := l.NamedVector(r); ok {
		return v
	}
	return nn.VectorOf(l.Vecs, r)
}

// Bytes is t's bytes, its planes where it is packed, and its rows; nil for
// no tensor.
func (t *Tensor) Bytes() ([]byte, *nn.Packed, int) {
	if t == nil {
		return nil, nil, 0
	}
	return t.Data, t.Packed, t.Rows
}
