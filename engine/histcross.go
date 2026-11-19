package engine

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// A history read across the CPU and a card. A block reading another block's
// history in place of its own (Gemma 3n's and Gemma 4's shared-KV layers)
// may run on the other side of the seam from that block: the side it runs on
// holds a copy of the history, written from the side that writes it before
// each run of a reader, from the first position the copy lacks to the run's
// end. On a card the copy is the tier's (nn.HistoryCopier); on the CPU
// device it is the session's own pages of the block (kvPages), which a block
// on a card otherwise leaves empty. Each side pages its copy as it pages any
// history -- at its own page size, so the two page apart -- and the rows
// cross as floats, converted from and to the CPU's format by the CPU
// device's conversions.
//
// A group parts only where its readers read no history but their source's
// and the source keeps none past its rows (crossGroups), and the device
// copies (nn.HistoryCopier); any other comes home whole (fixKVGroups).

// crossGroups reports, per block below len(src), whether its history group
// may part across the CPU and a card: no member reads a history past the one
// it binds in place of its own, and no block read that way keeps one past
// its rows.
func crossGroups(m *model.Model, src []int, more [][]graph.HistoryRead, group []int) []bool {
	ok := make([]bool, len(src))
	for li := range ok {
		ok[li] = true
	}
	for li := range src {
		if len(more[li]) == 0 && (src[li] == li || len(m.Composing().BlockGraph(src[li]).KeptHistories()) == 0) {
			continue
		}
		for b := range ok {
			if group[b] == group[li] {
				ok[b] = false
			}
		}
	}
	return ok
}

// crossable reports that block li's history group may part across the CPU
// and a card on the device the instance runs on.
func (s *Instance) crossable(li int) bool {
	return li >= 0 && li < len(s.crossOK) && s.crossOK[li] && s.routes[routeView].hcopy != nil
}

// cardReads reports that a block on a card reads block src's history in
// place of its own.
func (s *Instance) cardReads(src int) bool {
	for b := src + 1; b < s.hi; b++ {
		if s.histSrc(b) == src && s.devAt(b) {
			return true
		}
	}
	return false
}

// lanesRead reports that a block on the CPU device reads block src's history
// in place of its own.
func (s *Instance) lanesRead(src int) bool {
	for b := src + 1; b < s.hi; b++ {
		if s.histSrc(b) == src && !s.devAt(b) {
			return true
		}
	}
	return false
}

// crossRun brings the copies blocks [lo, hi), a run on route k, read of
// histories the other side writes up to rows [start, end) of session sess's
// sequence at base: the rows the run is about to read, written below it.
func (s *Instance) crossRun(sess *Session, k, lo, hi, base, start, end int) error {
	if !s.readsAcross || end <= start {
		return nil
	}
	card := k == routeView
	for li := lo; li < hi; li++ {
		src := s.histSrc(li)
		if src == li || src >= lo || s.devAt(src) == card {
			continue
		}
		seen := false
		for b := lo; b < li && !seen; b++ {
			seen = s.histSrc(b) == src
		}
		if seen {
			continue
		}
		if err := s.crossCopy(sess, src, card, base, start, end); err != nil {
			return err
		}
	}
	return nil
}

// crossCopy brings the copy of block src's history on the card (toCard) or
// on the CPU device up to end positions of session sess's sequence at base,
// from the first position it lacks: the rows below start were there before
// the step, those from it the step's.
func (s *Instance) crossCopy(sess *Session, src int, toCard bool, base, start, end int) error {
	kv := sess.kv
	pg := &kv.layers[src]
	if pg.p == 0 {
		return nil
	}
	hc := sess.routes[routeView].hcopy
	if hc == nil {
		return fmt.Errorf("model: block %d's history is read across the CPU and a card, and the device keeps no copy of a history", src)
	}
	l := kv.kvl.atLayer(s.c, src)
	conv := s.kvConvs[l.fmt]
	// What the run's rows read of a windowed history starts at its first
	// row's window: below it neither side need hold anything (a restored
	// prefix holds only its window).
	low := 0
	if base == 0 {
		low = pg.keyStart(start)
	}
	if toCard {
		from, ok := hc.CopyFrom(src, base, min(pg.sent[base], start))
		if !ok {
			return fmt.Errorf("model: no card holds the copy of block %d's history a block on it reads", src)
		}
		from = max(from, low)
		from = s.crossMap(sess, src, from, end)
		if from >= end {
			return nil
		}
		k, v := s.crossBufs((end - from) * l.kvDim())
		kv.window(src, base+from, end-from)
		if !pg.rowsOut(l, base+from, base+end, k, v, conv) {
			if err := s.kvCheck(); err != nil {
				return err
			}
			return fmt.Errorf("model: block %d's positions %d..%d would not leave the CPU device for its copy on the card", src, from, end-1)
		}
		if !hc.CopyRows(src, base, from, start, end, k, v) {
			return fmt.Errorf("model: block %d's positions %d..%d would not reach its copy on the card: %s", src, from, end-1, s.devWhy())
		}
		if pg.sent == nil {
			pg.sent = map[int]int{}
		}
		pg.sent[base] = end
		return nil
	}
	from := max(min(pg.copied[base], start), low)
	if base == 0 {
		from = s.crossAlias(sess, src, from, end)
	}
	if from >= end {
		pg.copy = true
		return nil
	}
	k, v := s.crossBufs((end - from) * l.kvDim())
	switch {
	case s.crossStale && pg.copied[base] > 0:
		clear(k)
		clear(v)
	case !hc.ReadRows(src, base, from, end, k, v):
		return fmt.Errorf("model: block %d's positions %d..%d would not come from its card to the CPU device's copy: %s", src, from, end-1, s.devWhy())
	}
	kv.writable(src, base+from)
	if from > pg.copied[base] && base == 0 {
		pg.genuine = max(pg.genuine, from) // what lies below it is not the card's history
	}
	if !pg.rowsIn(l, base+from, base+end, k, v, conv) {
		return fmt.Errorf("model: block %d's positions %d..%d would not go into the CPU device's copy", src, from, end-1)
	}
	pg.copy = true
	if pg.copied == nil {
		pg.copied = map[int]int{}
	}
	pg.copied[base] = end
	return nil
}

// migrated notes a move of block li's history of pos positions: sent to a
// card, the pages it left hold it as far as pos, a copy of it for a block on
// the CPU device reading it.
func (s *Instance) migrated(li, pos int, toDevice bool) {
	if pg := &s.kv.layers[li]; toDevice && pg.p > 0 {
		if pg.copied == nil {
			pg.copied = map[int]int{}
		}
		pg.copied[0] = pos
	}
}

// crossRows is crossRun for a step's rows, pos[i] the position of row i:
// across sessions (rowOwn) each row's session's sequence, else each
// sequence of the bound session's, its base the row's slot less its
// position.
func (s *Instance) crossRows(k, lo, hi int, pos, slot []int) error {
	if !s.readsAcross {
		return nil
	}
	for r := 0; r < len(pos); {
		sess, base := s.Session, 0
		if len(s.rowOwn) > 0 {
			sess = s.rowOwn[r]
		} else if slot != nil {
			base = slot[r] - pos[r]
		}
		start, end := pos[r], pos[r]+1
		e := r + 1
		for e < len(pos) {
			o, b := s.Session, 0
			if len(s.rowOwn) > 0 {
				o = s.rowOwn[e]
			} else if slot != nil {
				b = slot[e] - pos[e]
			}
			if o != sess || b != base {
				break
			}
			start, end = min(start, pos[e]), max(end, pos[e]+1)
			e++
		}
		if err := s.crossRun(sess, k, lo, hi, base, start, end); err != nil {
			return err
		}
		r = e
	}
	return nil
}

// crossMap maps, into the card's copy of block src's history from position
// from, the pages of it the prefix cache shares that the card names: a
// copy another session holds of the prefix there. It is where the rows the
// copy still lacks start.
func (s *Instance) crossMap(sess *Session, src, from, end int) int {
	kv := sess.kv
	pg := &kv.layers[src]
	d := kv.copyAt
	if kv.share == nil || d == nil || from%pg.p != 0 {
		return from
	}
	dev := d(src)
	if dev == nil || dev.SharePage(src) != pg.p {
		return from
	}
	for j := from / pg.p; (j+1)*pg.p <= end && pg.isShared(j); j++ {
		key := kv.pageKey(src, j)
		if key == "" || !dev.MapKV(src, j, key, (j+1)*pg.p) {
			break
		}
		from = (j + 1) * pg.p
		if pg.sent == nil {
			pg.sent = map[int]int{}
		}
		pg.sent[0] = max(pg.sent[0], from)
	}
	return from
}

// crossAlias is crossMap for the CPU device's copy of a card's block: the
// pages from from the store holds by name, aliased (faultShared) where it
// holds them.
func (s *Instance) crossAlias(sess *Session, src, from, end int) int {
	kv := sess.kv
	pg := &kv.layers[src]
	if kv.share == nil || from%pg.p != 0 {
		return from
	}
	for j := from / pg.p; (j+1)*pg.p <= end; j++ {
		if pg.resident(j) && !pg.isShared(j) {
			break // the copy's own page: its rows are written as they come
		}
		key := kv.pageKey(src, j)
		if !pg.resident(j) && (key == "" || kv.faultCopy(src, j, key) != nil) {
			break
		}
		from = (j + 1) * pg.p
		if pg.copied == nil {
			pg.copied = map[int]int{}
		}
		pg.copied[0] = max(pg.copied[0], from)
	}
	return from
}

// crossBufs is the instance's rows of n floats each for k and v a copy
// crosses through, grown to twice what it held, never shrunk: a copy on a
// card takes the rows of the page its token is in, one more a token.
func (s *Instance) crossBufs(n int) ([]float32, []float32) {
	if len(s.crossK) < n {
		n2 := max(n, 2*len(s.crossK))
		s.crossK, s.crossV = make([]float32, n2), make([]float32, n2)
	}
	return s.crossK[:n], s.crossV[:n]
}

// crossSettle makes the bound session's CPU copies the ones its blocks read
// now: a block on a card that a block on the CPU device reads keeps its
// pages here as the copy (written from the card at the reader's run), one
// no block here reads lets them go -- it keeps its history on the card
// alone -- and one that came home has its pages as its own again.
func (s *Instance) crossSettle() {
	if !s.readsAcross || s.kv == nil {
		return
	}
	for li := s.lo; li < s.hi && li < len(s.kv.layers); li++ {
		pg := &s.kv.layers[li]
		switch {
		case !s.devAt(li):
			pg.copy = false
		case s.lanesRead(li):
			pg.copy = pg.p > 0
		case pg.copy:
			pg.dropResident()
			pg.copy = false
		}
	}
}

// crossForget is what a change to the history other than a step's rows
// leaves of what the copies hold: nothing past at, from where they are
// written again whole.
func (s *kvPages) crossForget(at int) {
	s.copySealed = min(s.copySealed, at/max(s.p, 1))
	for b, n := range s.copied {
		s.copied[b] = min(n, at)
	}
	for b, n := range s.sent {
		s.sent[b] = min(n, at)
	}
}

// rowsOut gives positions [from, to) of the pages into k and v as floats,
// the row at from first, widened through conv where the history is not
// floats; false where a page is not resident.
func (s *kvPages) rowsOut(l kvLayout, from, to int, k, v []float32, conv nn.KVConvDevice) bool {
	kvDim, w := l.kvDim(), l.slots(l.kvDim())
	pl := s.layout(l)
	for pos := from; pos < to; {
		pg, off := s.page(pos)
		n := min(to-pos, s.p-off)
		if !s.resident(pg) {
			return false
		}
		i, o := pl.At(0, off, 0), (pos-from)*kvDim
		if !rowsConv(l, false, k[o:o+n*kvDim], s.mem.K(pg)[i:i+n*w], conv) ||
			!rowsConv(l, false, v[o:o+n*kvDim], s.mem.V(pg)[i:i+n*w], conv) {
			return false
		}
		pos += n
	}
	return true
}

// rowsIn writes positions [from, to) into the pages from k and v, the row
// at from first, narrowed through conv where the history is not floats: a
// page grown where it is not resident, one others share copied first.
func (s *kvPages) rowsIn(l kvLayout, from, to int, k, v []float32, conv nn.KVConvDevice) bool {
	kvDim, w := l.kvDim(), l.slots(l.kvDim())
	pl := s.layout(l)
	for pos := from; pos < to; {
		pg, off := s.page(pos)
		n := min(to-pos, s.p-off)
		s.grow(pg)
		s.cow(pg)
		if pos > from && pg < len(s.out) {
			s.out[pg] = false
		}
		i, o := pl.At(0, off, 0), (pos-from)*kvDim
		if !rowsConv(l, true, k[o:o+n*kvDim], s.mem.K(pg)[i:i+n*w], conv) ||
			!rowsConv(l, true, v[o:o+n*kvDim], s.mem.V(pg)[i:i+n*w], conv) {
			return false
		}
		pos += n
	}
	if s.win > 0 {
		s.genuine = max(s.genuine, s.keyStart(max(0, to-1-nn.KVWindowSlack)))
	}
	s.written = max(s.written, to)
	return true
}

// rowsConv moves rows between floats f and the history's own words h:
// copied where the history is floats, else narrowed into h (narrow) or
// widened from it through conv.
func rowsConv(l kvLayout, narrow bool, f, h []float32, conv nn.KVConvDevice) bool {
	if len(h) == 0 {
		return true
	}
	if l.fmt == nn.KVF32 {
		if narrow {
			copy(h, f)
		} else {
			copy(f, h)
		}
		return true
	}
	return conv != nil && conv.Convert(narrow, l.headDim, f, unsafe.Slice((*byte)(unsafe.Pointer(&h[0])), len(h)*4))
}
