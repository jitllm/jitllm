package gpu

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/nn"
)

// A picture's rows apart from its blocks (nn.ComposedRows, nn.RowsPart): the
// tower's entry over the picture and its projector over its last block's
// rows, graphs of generic ops the model states, formed by the composer and
// run on the card holding the tower's last block, in a submission of their
// own, the rows uploaded and read back. The weights are held for the model
// as the head is, at no block's slot; a step forms nothing past the first.

// rowsState is a card's picture rows: the weights and vectors it holds of
// them, the buffers a call binds, and the call a submission runs.
type rowsState struct {
	w map[resKey]*resident
	v map[rowsVec]backend.Buf
	// in, out, req and pos are a call's rows, its request and its
	// positions, each of size bytes.
	in, out, req, pos     backend.Buf
	inN, outN, reqN, posN int
	// call is the call's live rows, its one word written from callW at every
	// call (composer.TableCall).
	call                  backend.Buf
	callN                 int
	callW                 [1]uint32
	sv                    *servedPart
	inB, outB, reqB, posB []byte
	err                   error
	to                    launchTo
	lc                    launcher
	fn                    func(backend.Session)
	// at is the view the call is the session of (see embRead.at).
	at        *devTier
	ins, outs [1]backend.Buf
	// coll is the row a head reads from every stream, and collOut and
	// collIn the collapse's output and the first stream it is copied over
	// (collapse.go).
	coll            backend.Buf
	collOut, collIn [1]backend.Buf
	// takeIn and takeOut are the folded head's row take's (servedRowTake).
	takeIn, takeOut [1]backend.Buf
}

// RunRows runs p over a picture's live rows on the card holding the
// highest block, in (at most p.Rows rows of the graph's input) from the
// host and out read back, req the request's words and pos the positions'
// (nil for none); false where the card does not, its reason counted
// (Stats.Steps).
func (g *GPU) RunRows(p *nn.RowsPart, live int, req, pos []uint32, in, out []float32) bool {
	return g.runRows(0, p, live, req, pos, in, out)
}

// runRows is RunRows in session sid's view of the card.
func (g *GPU) runRows(sid uint64, p *nn.RowsPart, live int, req, pos []uint32, in, out []float32) bool {
	g.mu.Lock()
	d := g.rowsTier()
	g.mu.Unlock()
	if d == nil {
		return false
	}
	v := d.as(sid)
	if v == nil {
		return false
	}
	defer v.done()
	return v.RunRows(p, live, req, pos, in, out)
}

// RunRows is nn.PictureRowsDevice under this session's device locks.
func (s *gpuSession) RunRows(p *nn.RowsPart, live int, req, pos []uint32, in, out []float32) bool {
	defer s.leave(s.enter())
	return s.g.runRows(s.sid, p, live, req, pos, in, out)
}

// PrepRows forms p on the card holding the highest block, its tensors and
// a call's buffers held: at a placement, so no picture forms it.
func (g *GPU) PrepRows(p *nn.RowsPart) bool {
	return g.prepRows(0, p)
}

// prepRows is PrepRows in session sid's view of the card.
func (g *GPU) prepRows(sid uint64, p *nn.RowsPart) bool {
	g.mu.Lock()
	d := g.rowsTier()
	g.mu.Unlock()
	if d == nil {
		return false
	}
	v := d.as(sid)
	if v == nil {
		return false
	}
	defer v.done()
	return v.PrepRows(p)
}

// PrepRows is GPU.PrepRows under this session's device locks.
func (s *gpuSession) PrepRows(p *nn.RowsPart) bool {
	defer s.leave(s.enter())
	return s.g.prepRows(s.sid, p)
}

// PrepRows is GPU.PrepRows on this card.
func (g *devTier) PrepRows(p *nn.RowsPart) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	defer g.scratchWin().close()
	was := g.prewarming
	g.prewarming = true
	defer func() { g.prewarming = was }()
	if _, err := g.servedRows(p); err != nil {
		g.LastErr = "the composed picture rows: " + err.Error()
		return false
	}
	gr := p.Graph
	inW, outW := gr.Values[gr.In[0]].Width, gr.Values[gr.Out[0]].Width
	if err := g.rowsBufs(4*p.Rows*inW, 4*max(p.Rows, outRows(p, p.Rows))*outW, 4*(composer.RequestRows+p.Rows), 4*(p.Rows+1)); err != nil {
		g.LastErr = "the composed picture rows: " + err.Error()
		return false
	}
	return true
}

// outRows is the rows p writes over live rows: its output's own where the
// graph states a grid of them, else live.
func outRows(p *nn.RowsPart, live int) int {
	if r := p.Graph.Values[p.Graph.Out[0]].Rows; r > 0 {
		return r
	}
	return live
}

// RunRows is GPU.RunRows on this card.
func (g *devTier) RunRows(p *nn.RowsPart, live int, req, pos []uint32, in, out []float32) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	defer g.scratchWin().close()
	sv, err := g.servedRows(p)
	if err != nil {
		g.LastErr = "the composed picture rows: " + err.Error()
		g.Steps.Decline(nn.PartRows, partNotFormed)
		return false
	}
	gr := p.Graph
	on := outRows(p, live)
	if live > p.Rows || len(in) > p.Rows*gr.Values[gr.In[0]].Width || len(out) < on*gr.Values[gr.Out[0]].Width {
		g.LastErr = fmt.Sprintf("the composed picture rows: %d live rows past the %d the form holds", live, p.Rows)
		g.Steps.Decline(nn.PartRows, partCaller)
		return false
	}
	rs := &g.rows
	inW, outW := gr.Values[gr.In[0]].Width, gr.Values[gr.Out[0]].Width
	if err := g.rowsBufs(4*p.Rows*inW, 4*max(p.Rows, on)*outW, 4*(composer.RequestRows+p.Rows), 4*(p.Rows+1)); err != nil {
		g.LastErr = "the composed picture rows: " + err.Error()
		g.Steps.Decline(nn.PartRows, partNoRoom)
		return false
	}
	rs.sv = sv
	rs.inB, rs.outB = f32b(in), f32b(out[:on*outW])
	rs.reqB, rs.posB = u32b(req), u32b(pos)
	rs.callW[0] = uint32(live)
	rs.err = nil
	rs.lc = launcher{to: &rs.to, err: &rs.err}
	rs.at = g
	if rs.fn == nil {
		rs.fn = rs.session
	}
	g.dev.Session(rs.fn)
	err = rs.err
	rs.sv, rs.inB, rs.outB, rs.reqB, rs.posB, rs.err, rs.at = nil, nil, nil, nil, nil, nil, nil
	if err != nil {
		g.LastErr = "the composed picture rows: " + err.Error()
		return false
	}
	return true
}

// session runs the call's rowsSession on the view that made it.
func (rs *rowsState) session(s backend.Session) { rs.at.rowsSession(s) }

// rowsSession is RunRows' submission: the rows and words written, the
// form run, its rows read back.
func (g *devTier) rowsSession(s backend.Session) {
	rs := &g.rows
	if rs.err = s.Write(rs.in, rs.inB); rs.err != nil {
		return
	}
	if len(rs.reqB) > 0 {
		if rs.err = s.Write(rs.req, rs.reqB); rs.err != nil {
			return
		}
	}
	if len(rs.posB) > 0 {
		if rs.err = s.Write(rs.pos, rs.posB); rs.err != nil {
			return
		}
	}
	if rs.err = s.Write(rs.call, u32b(rs.callW[:])); rs.err != nil {
		return
	}
	rs.to.s = s
	defer rs.to.leave()
	rs.ins[0], rs.outs[0] = rs.in, rs.out
	if err := rs.sv.runAs(g, &rs.lc, rs.sv.inst, nn.PartRows, rs.ins[:], rs.outs[:]); err != nil {
		rs.err = err
		return
	}
	rs.err = s.Read(rs.out, rs.outB)
}

// rowsBufs holds the call's buffers at the sizes given, each grown where
// it is short.
func (g *devTier) rowsBufs(in, out, req, pos int) error {
	rs := &g.rows
	for _, b := range [...]struct {
		buf  *backend.Buf
		n    *int
		want int
	}{{&rs.in, &rs.inN, in}, {&rs.out, &rs.outN, out}, {&rs.req, &rs.reqN, req}, {&rs.pos, &rs.posN, pos}, {&rs.call, &rs.callN, 4}} {
		if *b.buf != nil && *b.n >= b.want {
			continue
		}
		if *b.buf != nil {
			(*b.buf).Free()
			*b.buf = nil
		}
		nb, err := g.dev.Alloc(b.want)
		if err != nil {
			return err
		}
		*b.buf, *b.n = nb, b.want
	}
	return nil
}

// servedRows is p's form on the card, its weights and vectors held.
func (g *devTier) servedRows(p *nn.RowsPart) (*servedPart, error) {
	rs := &g.rows
	sk := servedKey{kind: nn.PartRows, graph: p.Graph, rows: p.Rows}
	if sv := g.served[sk]; sv != nil && rs.req != nil {
		sv.sl.tables[tableKey(composer.TableRequest)], sv.sl.tables[tableKey(composer.TablePos)] = rs.req, rs.pos
		sv.sl.tables[tableKey(composer.TableCall)] = rs.call
		return sv, nil
	}
	formats := map[jlm.Role]jlm.Type{}
	weights := map[jlm.Role]*resident{}
	vectors := map[jlm.Role]backend.Buf{}
	for _, o := range p.Graph.Ops {
		for _, b := range o.Weight {
			if w, ok := p.Weights[b.Role]; ok {
				r, err := g.rowsWeight(p.File, w)
				if err != nil {
					return nil, fmt.Errorf("%v: %w", b.Role, err)
				}
				t, _ := jlmType(r.t)
				formats[b.Role], weights[b.Role] = t, r
				continue
			}
			v, ok := p.Vectors[b.Role]
			if !ok || len(v) == 0 {
				return nil, fmt.Errorf("the rows read %v, which the model hands no tensor of", b.Role)
			}
			buf, err := g.rowsVector(p.Graph.Block, b.Role, v)
			if err != nil {
				return nil, fmt.Errorf("%v: %w", b.Role, err)
			}
			formats[b.Role], vectors[b.Role] = jlm.TypeF32, buf
			if b.Rows > 1 && b.K > 0 && len(v) == b.Rows*b.K {
				r, err := g.rowsWeight(p.File, nn.Weight{T: quant.F32, Data: f32b(v), Rows: b.Rows, K: b.K})
				if err != nil {
					return nil, fmt.Errorf("%v: %w", b.Role, err)
				}
				weights[b.Role] = r
			}
		}
	}
	if err := g.rowsBufs(4, 4, 4*(composer.RequestRows+p.Rows), 4*(p.Rows+1)); err != nil {
		return nil, err
	}
	sv, err := g.servedFor(sk, func() (composedPart, error) {
		return composedPart{kind: nn.PartRows, graph: p.Graph, formats: formats, weights: weights, vectors: vectors, apart: true,
			tables: map[composer.TableKey]backend.Buf{tableKey(composer.TableRequest): rs.req, tableKey(composer.TablePos): rs.pos,
				tableKey(composer.TableCall): rs.call}}, nil
	})
	if err != nil {
		return nil, err
	}
	sv.sl.tables[tableKey(composer.TableRequest)], sv.sl.tables[tableKey(composer.TablePos)] = rs.req, rs.pos
	sv.sl.tables[tableKey(composer.TableCall)] = rs.call
	return sv, nil
}

// rowsWeight is w held on the card for the picture rows, at no block's
// slot.
func (g *devTier) rowsWeight(f *jlm.File, w nn.Weight) (*resident, error) {
	rs := &g.rows
	data := w.Data
	if len(data) == 0 && w.Packed != nil {
		data = w.Packed.QS
	}
	if len(data) == 0 && w.Entry == nil {
		return nil, fmt.Errorf("no bytes")
	}
	q, ok := quantOf(w.T)
	if !ok {
		return nil, fmt.Errorf("format %v", w.T)
	}
	key := adhocKey(w.Entry, data, q, w.Rows, w.K)
	if r, ok := rs.w[key]; ok && r.ok {
		return r, nil
	}
	if _, ok := jlmType(q); !ok {
		return nil, fmt.Errorf("format %v has no container type", w.T)
	}
	if w.Entry != nil && f != nil && !f.InDense(w.Entry) {
		own, release, err := g.blockWeight(f, w)
		if err != nil {
			return nil, err
		}
		defer release()
		w, data = own, own.Data
	} else if w.Entry != nil {
		own, release, err := g.ownWeight(f, w)
		if err != nil {
			return nil, err
		}
		defer release()
		w = own
		if data = w.Data; len(data) == 0 && w.Packed != nil {
			data = w.Packed.QS
		}
	}
	r := g.resident(-1, 0, w.Entry, q, data, w.Rows, w.K, w.Packed)
	if r == nil || !r.ok {
		return nil, fmt.Errorf("not held: %s", g.LastErr)
	}
	if rs.w == nil {
		rs.w = map[resKey]*resident{}
	}
	rs.w[key] = r
	return r, nil
}

// rowsVec names a vector of the model: its block's and its role, the role
// alone naming one in every block of a chained tower.
type rowsVec struct {
	block int
	role  jlm.Role
}

// rowsVector is the model's vector of block li under r, v, uploaded for the
// rows once while the model holds the card: by its block and role, which
// name one vector of the model, never by v's address, which a later
// instance's vector may take once this one's is gone.
func (g *devTier) rowsVector(li int, r jlm.Role, v []float32) (backend.Buf, error) {
	rs := &g.rows
	key := rowsVec{li, r}
	if b, ok := rs.v[key]; ok {
		return b, nil
	}
	b, err := g.allocBytes(unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), 4*len(v)))
	if err != nil {
		return nil, err
	}
	if rs.v == nil {
		rs.v = map[rowsVec]backend.Buf{}
	}
	rs.v[key] = b
	return b, nil
}

// releaseRows gives back what the rows hold of the model: their weights'
// residents, and what closeRows frees. The buffers and vectors were charged
// as scratch (PrepRows and RunRows run in a scratch window), so they are
// freed in one too, which refunds them. Callers hold g.mu.
func (g *devTier) releaseRows() {
	defer g.scratchWin().close()
	for _, r := range g.rows.w {
		for k, v := range g.res {
			if v == r {
				delete(g.res, k)
				break
			}
		}
		if r != nil && r.ok {
			g.refund(r.bytes())
			g.freeResident(r)
		}
	}
	g.closeRows()
}

// closeRows frees the picture rows' buffers and vectors; its weights are
// the residents' (freed with them).
func (g *devTier) closeRows() {
	rs := &g.rows
	for _, b := range [...]*backend.Buf{&rs.in, &rs.out, &rs.req, &rs.pos, &rs.call, &rs.coll} {
		if *b != nil {
			(*b).Free()
			*b = nil
		}
	}
	for k, b := range rs.v {
		b.Free()
		delete(rs.v, k)
	}
	clear(rs.w)
}

// rowsTier is the card a rows part runs on: the one holding the highest
// block, or the first the GPU holds where no block is placed (a chained
// tower's blocks run as parts). Callers hold g.mu.
func (g *GPU) rowsTier() *devTier {
	if d := g.tail(); d != nil {
		return d
	}
	if len(g.devs) > 0 {
		return g.devs[0]
	}
	return nil
}

// blockWeight is w, an entry of a block's page in f, read into this card's
// frames over f: the page held while the upload copies its planes, which
// release lets go.
func (g *devTier) blockWeight(f *jlm.File, w nn.Weight) (nn.Weight, func(), error) {
	fr, err := g.framesOf(f)
	if err != nil {
		return w, nil, err
	}
	e := w.Entry
	qn, dn, sn := f.SpanLen(e)
	var rs []jlm.Range
	for _, r := range [...]jlm.Range{{Off: e.QSOff, N: qn}, {Off: e.DOff, N: dn}, {Off: e.SCOff, N: sn}} {
		if r.N != 0 {
			rs = append(rs, r)
		}
	}
	l, err := fr.Hold(f.BlockOf(e), rs)
	if err != nil {
		return w, nil, err
	}
	qs, d, sc := fr.Span(w.Entry)
	if len(qs) == 0 {
		l.Release()
		return w, nil, fmt.Errorf("reading the block's entry into the card's frames failed")
	}
	w.Data = qs
	p := nn.Packed{}
	if w.Packed != nil {
		p = *w.Packed
	}
	p.QS, p.D, p.SC = qs, d, sc
	w.Packed = &p
	return w, l.Release, nil
}
