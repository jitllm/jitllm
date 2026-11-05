package gpu

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// A convolutional tower's blocks on the card (nn.ConvDevice): Gemma 3n's
// MobileNet-V5, each block the graph the model's front end states for it
// (nn.LayerPlan.Graph), formed by the composer at its input grid's rows when
// the block is placed and run by ConvLayers, the activation staying on the
// card from block to block. The matrices are the block's ordinary slots, so
// the pager pages a convolutional block as it pages any other; its norms,
// depthwise filters and an attention block's key grid are its own. The
// activation is two buffers every convolutional block of the card shares,
// sized for the widest.

// convBlock is a convolutional block's own buffers on the card.
type convBlock struct {
	p nn.ConvPlan
	// rows is the grid's rows a call runs, in and out the floats it reads
	// and writes.
	rows, in, out int
	vec           map[jlm.Role]backend.Buf
	// filt is each depthwise filter as the composer binds a matrix of
	// float32 rows; it owns its buffer.
	filt map[jlm.Role]*resident
	// tab holds an attention block's history and descriptors: its key grid's
	// pool of keys and values, the pool's page, each query row's descriptor
	// (attending every key of the grid) and their query offsets; keys and
	// page are the grid's rows and the pool's page.
	k, v, pages, desc, qoff backend.Buf
	keys, page              int
	bytes                   uint64
}

// bufs is the block's own buffers, for auxBufs.
func (b *convBlock) bufs() []backend.Buf {
	if b == nil {
		return nil
	}
	out := []backend.Buf{b.k, b.v, b.pages, b.desc, b.qoff}
	for _, v := range b.vec {
		out = append(out, v)
	}
	for _, r := range b.filt {
		out = append(out, r.qs)
	}
	return out
}

// convScratch is the activation the card's convolutional blocks hand on,
// two halves of n floats each, cur the half holding a call's input.
type convScratch struct {
	act [2]backend.Buf
	n   int
	cur int
	// blocks is how many convolutional blocks the card holds: the last one
	// out frees the halves.
	blocks int
	call   convCall
	to     launchTo
	fn     func(backend.Session)
	// at is the view the call is the session of (see embRead.at).
	at *devTier
}

// session runs the call's convSession on the view that made it.
func (cs *convScratch) session(s backend.Session) { cs.at.convSession(s) }

// convCall is one ConvLayers submission's arguments and its latched error.
type convCall struct {
	lo, hi      int
	first, last bool
	x, out      []float32
	tap         int
	tapOut      []float32
	ins, outs   [1]backend.Buf
	err         error
	lc          launcher
}

// convMatrix is the slot of matrix role r in a convolutional block's
// tensors (layer.tensors).
func convMatrix(r jlm.Role) (int, bool) {
	switch r {
	case jlm.RoleVAttnQ:
		return 0, true
	case jlm.RoleVAttnK:
		return 1, true
	case jlm.RoleVAttnV:
		return 2, true
	case jlm.RoleVAttnOut:
		return 3, true
	case jlm.RoleVMNConvExp, jlm.RoleVMNPwExp:
		return 5, true
	case jlm.RoleVMNConvPwl, jlm.RoleVMNPwProj:
		return 6, true
	}
	return 0, false
}

// convVector is the vector of role r a convolutional block offers, and for
// a depthwise filter its side (its taps' rows).
func convVector(p *nn.ConvPlan, w *nn.ConvWeights, r jlm.Role) ([]float32, int) {
	switch r {
	case jlm.RoleVMNNorm1, jlm.RoleVMNPwExpNorm, jlm.RoleVAttnNorm:
		return w.Norm1, 0
	case jlm.RoleVMNNorm2, jlm.RoleVMNPwProjNorm:
		return w.Norm2, 0
	case jlm.RoleVMNDwStart:
		return w.DwStart, p.KStart
	case jlm.RoleVMNDwStartNorm:
		return w.DwStartNorm, 0
	case jlm.RoleVMNDwMid:
		return w.DwMid, p.KMid
	case jlm.RoleVMNDwMidNorm:
		return w.DwMidNorm, 0
	case jlm.RoleVMNKDown:
		return w.KDown, p.KVK
	case jlm.RoleVMNKDownNorm:
		return w.KDownNorm, 0
	case jlm.RoleVMNVDown:
		return w.VDown, p.KVK
	case jlm.RoleVMNVDownNorm:
		return w.VDownNorm, 0
	}
	return nil, 0
}

// attends says gr attends a history.
func attends(gr *graph.Graph) bool {
	for _, o := range gr.Ops {
		if o.Kind == graph.OpAttend {
			return true
		}
	}
	return false
}

// convGeometry is the rows, floats in and out and key grid of block p.
func convGeometry(p *nn.ConvPlan) (rows, in, out, keys int) {
	rows = p.HIn * p.WIn
	keys = rows
	if p.KVK > 0 {
		keys = p.HKV * p.WKV
	}
	return rows, rows * p.CIn, p.HOut * p.WOut * p.COut, keys
}

// prepConv is PrepLayer for a convolutional block: its matrices in the
// block's slots, its own buffers uploaded, its form made. A block the front
// end states no graph of is the audio conformer's, which no device composes
// yet.
func (g *devTier) prepConv(li int, p *nn.LayerPlan, w *nn.LayerWeights, mayPage bool) bool {
	c, cw := p.Conv, w.Conv
	switch {
	case p.Graph == nil:
		g.LastErr = fmt.Sprintf("block %d is a convolutional block the front end states no graph of", li)
		return false
	case cw == nil:
		g.LastErr = fmt.Sprintf("block %d: a convolutional plan offered no convolutional vectors", li)
		return false
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	// No history: a block already here is shared whole.
	if l := g.layers[li]; l != nil && l.ok {
		return true
	}
	rows, in, out, keys := convGeometry(c)
	cb := &convBlock{p: *c, rows: rows, in: in, out: out}
	var aux uint64
	for _, o := range p.Graph.Ops {
		for _, b := range o.Weight {
			if _, ok := convMatrix(b.Role); ok {
				continue
			}
			v, taps := convVector(c, cw, b.Role)
			if len(v) == 0 {
				g.LastErr = fmt.Sprintf("block %d binds %v, which the block offers no vector for", li, b.Role)
				return false
			}
			if taps > 0 && len(v) != taps*taps*b.K {
				g.LastErr = fmt.Sprintf("block %d: filter %v is %d floats, want %d", li, b.Role, len(v), taps*taps*b.K)
				return false
			}
			aux += uint64(4 * len(v))
		}
	}
	if attends(p.Graph) {
		cb.keys, cb.page = keys, max(keys, 64)
		aux += uint64(4 * (2*cb.page*c.KD + 1 + rows*composer.RowWords + rows*4))
	}
	ws := weightList(w)
	full, why, ok := g.pageSizeWhy(ws)
	if !ok {
		g.LastErr = fmt.Sprintf("block %d: %s", li, why)
		return false
	}
	g.dropGraph()
	g.widest = max(g.widest, full)
	need, _ := g.blockBytes(ws)
	need += aux
	if n := max(in, out); n > g.conv.n {
		need += uint64(8 * (n - g.conv.n))
	}
	g.kvCompact(need, nil)
	if !g.room(need) {
		if n := g.slotsIfPerm(aux); !mayPage || !g.stream[li] || n < minSlots {
			if mayPage {
				g.Declined++
			}
			g.LastErr = fmt.Sprintf("block %d needs %d bytes, %d of %d used", li, need, g.used, g.limit)
			return false
		}
		if !g.reclaim(need, li, 0, li+1) {
			g.Declined++
			g.LastErr = fmt.Sprintf("block %d needs %d bytes and nothing can be paged out (%d of %d used)", li, need, g.used, g.limit)
			return false
		}
	}
	if w.Ensure != nil {
		if err := w.Ensure(w); err != nil {
			g.LastErr = fmt.Sprintf("block %d: %v", li, err)
			return false
		}
		ws = weightList(w)
	}
	if !g.convGrow(max(in, out)) {
		return false
	}
	l := &layer{nonCausal: true, graph: p.Graph, conv: cb}
	ts := l.tensors()
	for i, x := range ws[:7] {
		if len(x.Data) == 0 {
			continue
		}
		q, _ := quantOf(x.T)
		r := g.resident(li, i, q, x.Data, x.Rows, x.K, x.Packed)
		if r == nil || !r.ok {
			if g.LastErr == "" {
				g.LastErr = fmt.Sprintf("block %d: %s not resident", li, weightRole(i))
			}
			g.dropTensors(l)
			g.convFree(cb)
			return false
		}
		*ts[i] = r
	}
	if err := g.convUpload(cb, c, cw, p.Graph); err != nil {
		g.dropTensors(l)
		g.convFree(cb)
		g.LastErr = fmt.Sprintf("block %d: %v", li, err)
		return false
	}
	cb.bytes = aux
	l.auxBytes = aux
	g.charge(aux)
	l.ok = true
	var pb, pr uint64
	for _, rp := range ts {
		r := *rp
		pb += r.bytes()
		if n := r.bytes(); n > 0 {
			if f, err := g.planesFootprint(r.t, r.nrows, r.k); err == nil && f > n {
				pr += f - n
			}
		}
	}
	l.pg = &page{ws: append([]nn.Weight(nil), ws...), bytes: pb, round: pr, in: true, ensure: w.Ensure}
	g.pagesIn++
	g.pageBytes += pb + pr
	if old := g.layers[li]; old != nil {
		g.dropLayerLocal(old)
	}
	if g.layers == nil {
		g.layers = map[int]*layer{}
	}
	g.layers[li] = l
	g.layerGen++
	g.placeGen.Add(1)
	g.conv.blocks++
	// The form is the placement's: no call forms one.
	was := g.prewarming
	g.prewarming = true
	_, err := g.servedConv(l, li)
	g.prewarming = was
	if err != nil {
		g.releaseBlock(li)
		g.afterRelease()
		g.Declined++
		g.LastErr = fmt.Sprintf("block %d: the composed convolutional block: %v", li, err)
		return false
	}
	g.trim()
	return true
}

// convUpload uploads block cb's vectors and filters the graph binds and an
// attention block's history and descriptors, once: no call writes them.
func (g *devTier) convUpload(cb *convBlock, c *nn.ConvPlan, cw *nn.ConvWeights, gr *graph.Graph) error {
	cb.vec, cb.filt = map[jlm.Role]backend.Buf{}, map[jlm.Role]*resident{}
	for _, o := range gr.Ops {
		for _, b := range o.Weight {
			if _, ok := convMatrix(b.Role); ok {
				continue
			}
			if cb.vec[b.Role] != nil || cb.filt[b.Role] != nil {
				continue
			}
			v, taps := convVector(c, cw, b.Role)
			buf, err := g.allocBytes(f32b(v))
			if err != nil {
				return err
			}
			if taps > 0 {
				cb.filt[b.Role] = &resident{qs: buf, t: layout.Float32, nrows: taps * taps, k: b.K, ok: true}
				continue
			}
			cb.vec[b.Role] = buf
		}
	}
	if cb.keys == 0 {
		return nil
	}
	pool := make([]float32, cb.page*c.KD)
	for i := range pool {
		pool[i] = float32(math.NaN())
	}
	var err error
	if cb.k, err = g.allocBytes(f32b(pool)); err != nil {
		return err
	}
	if cb.v, err = g.allocBytes(f32b(pool)); err != nil {
		return err
	}
	if cb.pages, err = g.allocBytes(make([]byte, 4)); err != nil {
		return err
	}
	desc := make([]byte, 4*cb.rows*composer.RowWords)
	for r := range cb.rows {
		d := desc[4*r*composer.RowWords:]
		binary.LittleEndian.PutUint32(d[4*composer.RowEnd:], uint32(cb.keys))
		binary.LittleEndian.PutUint32(d[4*composer.RowWrite:], uint32(r))
	}
	if cb.desc, err = g.allocBytes(desc); err != nil {
		return err
	}
	cb.qoff, err = g.allocBytes(make([]byte, 4*cb.rows*4))
	return err
}

// convFree frees block cb's own buffers.
func (g *devTier) convFree(cb *convBlock) {
	for _, b := range cb.bufs() {
		if b != nil {
			b.Free()
		}
	}
}

// convGrow grows the activation's halves to n floats each, poisoned (RULE
// 13). Callers hold g.mu.
func (g *devTier) convGrow(n int) bool {
	cs := &g.conv
	if n <= cs.n {
		return true
	}
	for i := range cs.act {
		have := cs.n
		if !g.regrow(&cs.act[i], &have, n, 4, true, true) {
			return false
		}
	}
	cs.n = n
	return true
}

// convLeft is a convolutional block leaving the card; the last one takes
// the activation with it. Its own buffers are auxBufs'. Callers hold g.mu.
func (g *devTier) convLeft(l *layer) {
	if l.conv == nil {
		return
	}
	l.conv = nil
	cs := &g.conv
	if cs.blocks--; cs.blocks > 0 {
		return
	}
	defer g.scratchWin().close()
	for i, b := range cs.act {
		if b != nil {
			b.Free()
			cs.act[i] = nil
		}
	}
	cs.n, cs.blocks = 0, 0
}

// servedConv is convolutional block li's form at its grid's rows, its
// weights bound again: a page-in replaces a block's buffers. Callers hold
// g.mu.
func (g *devTier) servedConv(l *layer, li int) (*servedPart, error) {
	cb := l.conv
	sk := servedKey{kind: nn.PartConv, graph: l.graph, rows: cb.rows, keys: cb.keys, page: cb.page}
	sv, err := g.servedFor(sk, func() (composedPart, error) {
		sl := composedPart{kind: nn.PartConv, graph: l.graph, formats: map[jlm.Role]jlm.Type{},
			weights: map[jlm.Role]*resident{}, vectors: map[jlm.Role]backend.Buf{},
			tables: map[composer.TableKey]backend.Buf{}, keys: cb.keys, page: cb.page, apart: true}
		if cb.keys > 0 {
			t := sl.tables
			t[tableKey(composer.TableKeys)], t[tableKey(composer.TableValues)] = cb.k, cb.v
			t[tableKey(composer.TablePages)], t[tableKey(composer.TableRows)], t[tableKey(composer.TableQOff)] = cb.pages, cb.desc, cb.qoff
		}
		for _, o := range l.graph.Ops {
			for _, b := range o.Weight {
				r := g.convWeight(l, b.Role)
				if r == nil {
					if v := cb.vec[b.Role]; v != nil {
						sl.vectors[b.Role], sl.formats[b.Role] = v, jlm.TypeF32
						continue
					}
					return sl, fmt.Errorf("the composed block %d binds %v, which the card holds nothing of", li, b.Role)
				}
				t, ok := jlmType(r.t)
				if !ok {
					return sl, fmt.Errorf("the composed block %d: format %d of %v has no container type", li, r.t, b.Role)
				}
				sl.weights[b.Role], sl.formats[b.Role] = r, t
			}
		}
		return sl, nil
	})
	if err != nil {
		return nil, err
	}
	for r := range sv.sl.weights {
		sv.sl.weights[r] = g.convWeight(l, r)
	}
	return sv, nil
}

// convWeight is block l's matrix or filter of role r, nil for a vector.
func (g *devTier) convWeight(l *layer, r jlm.Role) *resident {
	if i, ok := convMatrix(r); ok {
		return *l.tensors()[i]
	}
	return l.conv.filt[r]
}

// convOut is block li's output length in floats, 0 where this card does
// not hold it as a convolutional block.
func (g *devTier) convOut(li int) int {
	g.mu.Lock()
	defer g.mu.Unlock()
	if l := g.layers[li]; l != nil && l.conv != nil {
		return l.conv.out
	}
	return 0
}

// ConvLayers runs convolutional blocks [lo, hi) (nn.ConvDevice), paging them
// in as it goes: one submission a resident run, the activation on the card
// between them.
func (g *devTier) ConvLayers(lo, hi int, x, out []float32, tap int, tapOut []float32) bool {
	if g.injectedFail() {
		return false
	}
	for li := lo; li < hi; {
		if !g.pageInHold(li, lo, hi) {
			return false
		}
		// Held as a submission holds its range: no page-in of another
		// session's sends a block away before convOnce reads it.
		end := li + 1
		g.mu.Lock()
		for end < hi && g.holdPages(end, end+1) {
			end++
		}
		g.mu.Unlock()
		ok := g.convOnce(li, end, li == lo, end == hi, x, out, tap, tapOut)
		g.mu.Lock()
		g.dropPages(li, end)
		g.mu.Unlock()
		if !ok {
			return false
		}
		li = end
	}
	return true
}

// convOnce is one submission of resident blocks [lo, hi): x goes up first
// when the call starts here, out comes back when it ends here, and the tap
// block's output comes back as it is written.
func (g *devTier) convOnce(lo, hi int, first, last bool, x, out []float32, tap int, tapOut []float32) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	cs := &g.conv
	for li := lo; li < hi; li++ {
		l := g.layers[li]
		if l == nil || l.conv == nil || !l.ok {
			g.LastErr = fmt.Sprintf("block %d is not a convolutional block on this card", li)
			return false
		}
		if _, err := g.servedConv(l, li); err != nil {
			g.LastErr = fmt.Sprintf("block %d: the composed convolutional block: %v", li, err)
			g.Steps.Decline(nn.PartConv, partNotFormed)
			return false
		}
	}
	if b := g.layers[lo].conv; first && (len(x) != b.in || b.in > cs.n) {
		g.LastErr = fmt.Sprintf("block %d reads %d floats, handed %d", lo, b.in, len(x))
		return false
	}
	if b := g.layers[hi-1].conv; last && len(out) != b.out {
		g.LastErr = fmt.Sprintf("block %d writes %d floats, handed %d", hi-1, b.out, len(out))
		return false
	}
	c := &cs.call
	*c = convCall{lo: lo, hi: hi, first: first, last: last, x: x, out: out, tap: tap, tapOut: tapOut}
	c.lc = launcher{to: &cs.to, err: &c.err, g: g}
	cs.at = g
	if cs.fn == nil {
		cs.fn = cs.session
	}
	g.dev.Session(cs.fn)
	cs.at = nil
	err := c.err
	*c = convCall{}
	if err != nil {
		g.LastErr = err.Error()
		return false
	}
	g.ConvBlocks += int64(hi - lo)
	return true
}

// convSession is convOnce's submission, inside the backend's Session.
func (g *devTier) convSession(s backend.Session) {
	cs := &g.conv
	c := &cs.call
	cs.to.s = s
	defer cs.to.leave()
	if c.first {
		cs.cur = 0
		if c.err = s.Write(cs.act[0], f32b(c.x)); c.err != nil {
			return
		}
	}
	for li := c.lo; li < c.hi && c.err == nil; li++ {
		l := g.layers[li]
		sv := g.served[servedKey{kind: nn.PartConv, graph: l.graph, rows: l.conv.rows, keys: l.conv.keys, page: l.conv.page}]
		c.ins[0], c.outs[0] = cs.act[cs.cur], cs.act[1-cs.cur]
		if c.err = sv.run(g, &c.lc, sv.inst, c.ins[:], c.outs[:]); c.err != nil {
			return
		}
		cs.cur = 1 - cs.cur
		if li == c.tap {
			c.err = s.Read(cs.act[cs.cur], f32b(c.tapOut[:l.conv.out]))
		}
	}
	if c.last && c.err == nil {
		c.err = s.Read(cs.act[cs.cur], f32b(c.out))
	}
}

// ConvLayers runs convolutional blocks [lo, hi) card by card, the
// activation crossing the host between cards.
func (g *GPU) ConvLayers(lo, hi int, x, out []float32, tap int, tapOut []float32) bool {
	g.convMu.Lock()
	defer g.convMu.Unlock()
	g.mu.Lock()
	rs, ok := g.runsInto(g.convRuns[:0], lo, hi)
	g.convRuns = rs
	g.mu.Unlock()
	if !ok || len(rs) == 0 {
		return false
	}
	in := x
	for i, r := range rs {
		dst := out
		if i < len(rs)-1 {
			n := r.dev.convOut(r.hi - 1)
			if n == 0 {
				return false
			}
			k := i % 2
			if cap(g.convTmp[k]) < n {
				g.convTmp[k] = make([]float32, n)
			}
			dst = g.convTmp[k][:n]
		}
		if !r.dev.ConvLayers(r.lo, r.hi, in, dst, tap, tapOut) {
			return false
		}
		in = dst
	}
	return true
}

// ConvLayers forwards to the cards (nn.ConvDevice).
func (s *gpuSession) ConvLayers(lo, hi int, x, out []float32, tap int, tapOut []float32) bool {
	defer s.leave(s.step())
	return s.g.ConvLayers(lo, hi, x, out, tap, tapOut)
}
