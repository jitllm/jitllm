package gpu

import (
	"encoding/binary"
	"fmt"
	"slices"
	"unsafe"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// embPend is a prompt chunk whose rows EmbedRows promised and has not yet
// produced: the gather runs INSIDE the next submission whose x is dst, writing
// the chunk's scratch directly, so the rows never cross the bus in either
// direction. Anything else that reaches the device first makes it real on the
// host (materialize) before going on.
type embPend struct {
	dst []float32
	ids []int32
	// pos is the first row's position; composed says the composer gathers
	// the rows (ComposedEmbed), counted as kind: a decode step's or a
	// prompt's, from the scratch set geo current when they were promised
	// (a model of several geometries changes sets between blocks).
	pos      int
	composed bool
	kind     partKind
	geo      geoKey
}

// same reports whether x is exactly pe's destination -- the slice the caller
// was promised the rows in, handed back to Layers.
func (pe *embPend) same(x []float32) bool {
	return len(x) > 0 && len(x) == len(pe.dst) && unsafe.SliceData(x) == unsafe.SliceData(pe.dst)
}

// EmbedRows promises dst's first len(ids) rows as the input rows of tokens
// ids at positions pos onwards (nn.EmbedDevice). Where the composer serves
// them (ComposedEmbed) they are the front end's graph of the input rows over
// the tables the device holds, any rows a call carries, each decline named;
// otherwise the host writes them. False writes nothing.
//
// The gather rides the next submission rather than running here: reading the
// rows back into a fresh prompt buffer cost more than the host lookup it
// replaced. The chunk's Layers call gathers straight into its scratch
// (takeEmbed), which also skips uploading x.
func (g *devTier) EmbedRows(ids []int32, pos int, dst []float32) bool {
	n := len(ids)
	g.mu.Lock()
	defer g.mu.Unlock()
	bs := g.bs
	if n == 0 || n > batchWidth || bs == nil {
		return false
	}
	kind := partEmbed
	if n > 1 || g.prompt {
		kind = partEmbedRows
	}
	why := bs.embedServes()
	if why == partServes {
		why = bs.embedFits(ids, pos, dst)
	}
	if why == partServes && !g.embedBufs(bs) {
		why = partNoRoom
	}
	if why != partServes {
		g.PartsDeclined[kind]++
		g.StepDeclines[why]++
		return false
	}
	pe := &g.embBuf
	pe.dst, pe.ids, pe.pos, pe.composed, pe.kind, pe.geo = dst, append(pe.ids[:0], ids...), pos, true, kind, g.geoCur
	g.embPend = pe
	return true
}

// embedFits is why the rows of tokens ids at positions pos onwards into dst
// are not the composer's input rows, serves where they are: each token a row
// of the embedding, each position one of the position table's, and dst the
// rows' width, or every stream's where the graph writes the first or every
// stream.
func (bs *blockScratch) embedFits(ids []int32, pos int, dst []float32) partDecline {
	w := bs.embW
	n, s := len(ids)*w.k, bs.embStreams()
	switch {
	case s > 1 && len(dst) != n*s:
		// The streams past the first are the graph's too: the destination
		// is the whole stream-major residual.
		return partEmbedStreams
	case s == 1 && len(dst) != n && len(dst) != n*bs.p.Streams():
		return partEmbedToken
	}
	for _, id := range ids {
		if id < 0 || int(id) >= w.nrows {
			return partEmbedToken
		}
	}
	if p := bs.posW; p != nil && (pos < 0 || pos+len(ids) > p.nrows) {
		return partEmbedToken
	}
	return partServes
}

// embStreams is how many residual streams the input rows' graph writes: its
// outputs but the rows' per-layer inputs, its last where the plan has them.
func (bs *blockScratch) embStreams() int {
	n := len(bs.embedGraph.Out)
	if bs.p.PLEWidth != 0 {
		n--
	}
	return n
}

// embedOuts is the buffers input rows writing s streams write at R rows of
// bs: x, or every stream's view of it, then the per-layer inputs where the
// plan has them.
func (bs *blockScratch) embedOuts(s, R int) ([]backend.Buf, error) {
	if s <= 1 {
		bs.embOuts[0] = bs.x
		return bs.embOuts[:1], nil
	}
	in, out, err := bs.servedStreams(R, bs.p.Streams())
	if err != nil {
		return nil, err
	}
	n := copy(bs.embOuts[:], out[:s])
	if bs.p.PLEWidth != 0 {
		// servedStreams' inputs are the streams' views, then the per-layer
		// inputs.
		bs.embOuts[n] = in[len(in)-1]
		n++
	}
	return bs.embOuts[:n], nil
}

// prepEmbed holds the tables the head's input rows read (nn.Head.Embed)
// where the composer serves them: the token embedding, the head itself
// where it is the embedding, and a learned position table, each uploaded
// once for the model. A table that does not fit is named (bs.embWhy).
// Callers hold mu.
func (g *devTier) prepEmbed(bs *blockScratch, h *nn.Head) {
	bs.embW, bs.posW, bs.embWhy = nil, nil, partServes
	if h.Embed == nil {
		return
	}
	bs.embW = bs.head
	if !h.Embeds {
		bs.embW, bs.embWhy = g.embedTable(h.TokEmbd)
	}
	if len(h.PosEmbd.Data) > 0 && bs.embWhy == partServes {
		bs.posW, bs.embWhy = g.embedTable(h.PosEmbd)
	}
	clear(bs.embSW)
	clear(bs.embSV)
	p := h.Streams
	if p == nil || bs.embWhy != partServes {
		return
	}
	// The expansion's and the per-layer inputs' tensors, held as the
	// picture rows' are, at no block's slot.
	if bs.embSW == nil {
		bs.embSW, bs.embSV = map[jlm.Role]*resident{}, map[jlm.Role]backend.Buf{}
	}
	for role, w := range p.Weights {
		r, err := g.rowsWeight(w)
		if err != nil {
			g.LastErr = fmt.Sprintf("the input rows' %v: %v", role, err)
			bs.embWhy = partNoRoom
			return
		}
		bs.embSW[role] = r
	}
	for role, v := range p.Vectors {
		b, err := g.rowsVector(v)
		if err != nil {
			g.LastErr = fmt.Sprintf("the input rows' %v: %v", role, err)
			bs.embWhy = partNoRoom
			return
		}
		bs.embSV[role] = b
	}
}

// embedTable is w resident for the input rows, taken under paging only where
// it costs no block slot, as the head is (PrepHead).
func (g *devTier) embedTable(w nn.Weight) (*resident, partDecline) {
	q, ok := quantOf(w.T)
	if !ok || len(w.Data) == 0 {
		return nil, partEmbedFormat
	}
	// A table already held costs nothing more: a head given back for room
	// leaves its tables held (PrepHead).
	held := g.res[resKey{block: -1, nrows: w.Rows, k: w.K, t: q, p: &w.Data[0]}]
	if len(g.stream) > 0 && (held == nil || !held.ok) {
		if now, after := g.slots(), g.slotsIfPerm(uint64(len(w.Data))); after < now {
			return nil, partNoRoom
		}
	}
	r := g.resident(-1, 0, q, w.Data, w.Rows, w.K, w.Packed)
	if r == nil || !r.ok {
		return nil, partNoRoom
	}
	return r, partServes
}

// embedBufs makes bs's token and position words for the composer's gather
// if it has none, outside any Session: a word a row of the widest chunk,
// and the positions' live row count. Callers hold mu.
func (g *devTier) embedBufs(bs *blockScratch) bool {
	if bs.embTok != nil && bs.embPos != nil {
		return true
	}
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	for _, b := range [...]struct {
		buf   *backend.Buf
		words int
	}{{&bs.embTok, batchWidth}, {&bs.embPos, 1 + batchWidth}} {
		if *b.buf != nil {
			continue
		}
		nb, err := g.dev.Alloc(b.words * 4)
		if err != nil {
			g.LastErr = err.Error()
			return false
		}
		*b.buf = nb
	}
	return true
}

// embedWords stages the composer's gather of pe's rows at R rows: each row's
// token, then the live row count and each row's position, the rows past the
// promise's repeating its last.
func (g *devTier) embedWords(pe *embPend, R int) (tok, pos []byte) {
	n := len(pe.ids)
	tok = slices.Grow(g.subBytes.embIDs[:0], R*4)
	pos = slices.Grow(g.subBytes.embPos[:0], (R+1)*4)
	pos = binary.LittleEndian.AppendUint32(pos, uint32(n))
	for r := range R {
		i := min(r, n-1)
		tok = binary.LittleEndian.AppendUint32(tok, uint32(pe.ids[i]))
		pos = binary.LittleEndian.AppendUint32(pos, uint32(pe.pos+i))
	}
	g.subBytes.embIDs, g.subBytes.embPos = tok, pos
	return tok, pos
}

// embedRowsAt is the rows the composer gathers n promised rows of bs at on
// their own: one for one, else the narrowest width at least n whose form the
// placement formed, else the narrowest prompt width the lane holds a scratch
// for, else n.
func (g *devTier) embedRowsAt(bs *blockScratch, n int) int {
	if n <= 1 {
		return 1
	}
	t, _ := jlmType(bs.embW.t)
	formed, held := 0, 0
	for w := range g.bbs {
		if w < n {
			continue
		}
		if held == 0 || w < held {
			held = w
		}
		if g.served[servedKey{kind: partEmbed, graph: bs.embedGraph, rows: w, t: t}] != nil && (formed == 0 || w < formed) {
			formed = w
		}
	}
	switch {
	case formed > 0:
		return formed
	case held > 0:
		return held
	case g.promptsReserved():
		// Rows wider than every width held: the widest a promise holds,
		// formed at placement (prewarmComposedHead), its rows the call's.
		return batchWidth
	}
	return n
}

// dropEmbed forgets an unconsumed promise; see layersCall.
func (g *devTier) dropEmbed() {
	g.mu.Lock()
	g.embPend = nil
	g.mu.Unlock()
}

// dropEmbedOf is dropEmbed for session sid, from the router, which holds the
// device's view rather than the session's.
func (g *devTier) dropEmbedOf(sid uint64) {
	g.mu.Lock()
	if ds := g.sess[sid]; ds != nil {
		ds.embPend = nil
	}
	g.mu.Unlock()
}

// takeEmbed settles the promise against a submission of x: it returns it when
// x IS the promised destination (the caller then gathers into the scratch),
// and otherwise makes the rows real on the host first, so a path that uploads
// x -- a row at a time, a narrower chunk -- uploads the embedding.
func (g *devTier) takeEmbed(x []float32) *embPend {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.takeEmbedLocked(x)
}

// takeEmbedLocked is takeEmbed for a caller holding mu -- layersOnce runs
// under it, Session included.
func (g *devTier) takeEmbedLocked(x []float32) *embPend {
	pe := g.embPend
	g.embPend = nil
	if pe == nil {
		return nil
	}
	if pe.same(x) {
		return pe
	}
	g.materializeLocked(pe)
	return nil
}

// materializeLocked runs a promised gather on its own and reads the rows back
// into the promised slice. Callers hold mu.
func (g *devTier) materializeLocked(pe *embPend) {
	g.materializeComposed(pe)
}

// materializeComposed is materializeLocked for the composer's promise: its
// form of the input rows run in a submission of its own, into embCast, and
// the rows read back -- where the call that takes them is not this card's
// (a head on another card than block 0's), whose upload then moves them as
// the residual crosses between cards. Callers hold mu.
func (g *devTier) materializeComposed(pe *embPend) {
	if cur := g.geoCur; pe.geo != cur {
		g.useGeom(pe.geo)
		defer g.useGeom(cur)
	}
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	bs := g.bs
	if bs == nil || bs.embedServes() != partServes || !g.embedBufs(bs) {
		g.LastErr = "the composer's input rows were promised and are no longer held"
		return
	}
	R := g.embedRowsAt(bs, len(pe.ids))
	if bs.embCast == nil {
		if need := g.footprint(batchWidth * bs.embW.k * 4); !g.room(need) {
			g.LastErr = fmt.Sprintf("the composer's input rows need %d bytes, %d of %d used", need, g.used, g.limit)
			return
		}
		b, err := g.dev.Alloc(batchWidth * bs.embW.k * 4)
		if err != nil {
			g.LastErr = err.Error()
			return
		}
		bs.embCast = b
	}
	sv, err := g.servedEmbed(bs, R)
	if err != nil {
		g.LastErr = "the composed input rows: " + err.Error()
		return
	}
	rd := &g.embRd
	rd.tok, rd.pos = g.embedWords(pe, R)
	rd.rows = R
	if rd.composedFn == nil {
		rd.composedFn = g.composedSession
	}
	g.readBack(pe, sv, rd.composedFn)
}

// embRead is a promised gather made real in a submission of its own: what
// its session reads, and the session bodies, bound once. A closure over the
// call's locals, and the error it set, were heap objects a call -- a token's
// worth where the head is on another card than block 0's.
type embRead struct {
	pe       *embPend
	sv       *servedPart
	tok, pos []byte
	rows     int
	err      error
	lc       launcher

	composedFn func(backend.Session)
}

// readBack runs fn, one of embRead's session bodies, over the promise pe
// and, for the composer's, its form sv, and reports whether the rows came
// back.
func (g *devTier) readBack(pe *embPend, sv *servedPart, fn func(backend.Session)) bool {
	rd := &g.embRd
	rd.pe, rd.sv, rd.err = pe, sv, nil
	rd.lc = launcher{to: &g.embTo, err: &rd.err}
	g.dev.Session(fn)
	err := rd.err
	rd.pe, rd.sv, rd.err = nil, nil, nil
	if err != nil {
		g.LastErr = err.Error()
		return false
	}
	g.EmbedReadBacks++
	return true
}

// composedSession is materializeComposed's submission: the composer's form
// of the input rows, and the rows read back.
func (g *devTier) composedSession(s backend.Session) {
	rd, bs := &g.embRd, g.bs
	pe, sv := rd.pe, rd.sv
	if rd.err = s.Write(bs.embTok, rd.tok); rd.err != nil {
		return
	}
	if bs.posW != nil {
		if rd.err = s.Write(bs.embPos, rd.pos); rd.err != nil {
			return
		}
	}
	g.embTo.s = s
	defer g.embTo.leave()
	n, d := len(pe.ids), bs.embW.k
	if ns := bs.embStreams(); ns > 1 {
		// Every stream, through the scratch's own residual: the call that
		// takes the rows uploads them.
		outs, err := bs.embedOuts(ns, rd.rows)
		if err == nil {
			err = sv.runAs(g, &rd.lc, sv.inst, pe.kind, nil, outs)
		}
		for k := 0; k < ns && err == nil; k++ {
			err = s.Read(outs[k], f32b(pe.dst[k*n*d:(k+1)*n*d]))
		}
		// The per-layer inputs go where the call that takes the rows
		// writes them from (pleRows).
		if w := n * bs.p.PLEWidth; w != 0 && err == nil {
			if len(g.pleHost) < w {
				err = fmt.Errorf("tier: the input rows' per-layer inputs of %d rows, where the caller holds %d floats", n, len(g.pleHost))
			} else {
				err = s.Read(outs[ns], f32b(g.pleHost[:w]))
			}
		}
		rd.err = err
		return
	}
	out := [1]backend.Buf{bs.embCast}
	if err := sv.runAs(g, &rd.lc, sv.inst, pe.kind, nil, out[:]); err != nil {
		rd.err = err
		return
	}
	rd.err = s.Read(bs.embCast, f32b(pe.dst[:n*d]))
}
