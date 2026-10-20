package maintier

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	kernels "github.com/jitllm/jitllm/paritytest/mainkernels"
)

// A linear block's recurrent state is a session's, like an attention history,
// and a step across sessions (LayersSessions) runs several sessions' rows in
// one launch. A launch names buffers, not sessions, so every session's state
// for a block lives in ONE pool: a pair of buffers whose SLOTS are sequences'
// states, and each row of a step finds its own through the recurrent
// descriptor (kernels.RecDescWords). A session holds a SEAT -- a run of slots,
// one per row of its batch -- and the seat is the same in every linear block's
// pool on the device, so one descriptor serves every block of a submission.
//
// The pool is double-buffered as the pair before it was: ir.Validate refuses a
// kernel that loads and stores one buffer, so a step reads one half and writes
// the other. Which half is current is the SESSION's, per block (recPair.cur),
// since sessions step independently. A step across sessions reads one half for
// all its rows, so rows that disagree are first brought to one half
// (alignRec) -- a copy of a session's state that happens when sessions that
// stepped apart first step together, and not again while they keep stepping
// together.

// recPool is one linear block's recurrent states for every session on this
// device: the delta rule's state and the convolution's window, each in two
// halves of slots slots. In the shared form s[1] is nil and a step writes the
// device's recNext, which CopySlots moves home.
type recPool struct {
	s, c  [2]backend.Buf
	slots int
}

// recSeat is a session's run of slots, one per sequence of its batch.
type recSeat struct{ base, rows int }

// recSlotBytes is one slot of a pool in the device's current form, both halves
// of each buffer that has two.
func (g *devTier) recSlotBytes(shared bool) uint64 {
	n := 2*g.recC + 2*g.recS
	if shared {
		n -= g.recS
	}
	return uint64(n) * 4
}

// recPools is every linear block on this device that holds a pool.
func (g *devTier) recPools() []*layer {
	var ls []*layer
	for _, l := range g.layers {
		if l != nil && l.pool != nil {
			ls = append(ls, l)
		}
	}
	return ls
}

// seatPlace is the first run of rows slots that no seat but sid's holds.
// Callers hold g.mu.
func (g *devTier) seatPlace(sid uint64, rows int) int {
	type span struct{ a, b int }
	var taken []span
	for s, st := range g.recSeats {
		if s != sid {
			taken = append(taken, span{st.base, st.base + st.rows})
		}
	}
	slices.SortFunc(taken, func(x, y span) int { return x.a - y.a })
	base := 0
	for _, t := range taken {
		if base+rows <= t.a {
			break
		}
		base = max(base, t.b)
	}
	return base
}

// recEnd is one past the highest slot any seat holds.
func (g *devTier) recEnd() int {
	end := 0
	for _, st := range g.recSeats {
		end = max(end, st.base+st.rows)
	}
	return end
}

// newRecPool allocates a pool of slots slots in the given form. Every slot
// holds a defined value before anything can read it -- a summary of garbage is
// a history the model never saw (RULE 13) -- but the runs in filled are about
// to be carried in from the old pool by a device copy, so only the slots
// between them are zeroed from the host: a resize uploads zeros for the slots
// that changed hands, not for the whole pool.
func (g *devTier) newRecPool(slots int, shared bool, filled []recSeat) (*recPool, error) {
	p := &recPool{slots: slots}
	sB, cB := slots*g.recS*4, slots*g.recC*4
	zs := make([]byte, max(sB, cB))
	// zero writes zeros to every slot of b outside filled.
	zero := func(b backend.Buf, w int) error {
		if w == 0 {
			return nil // a short convolution's state, which it does not have
		}
		at := 0
		for _, r := range append(filled, recSeat{base: slots}) {
			if r.base > at {
				if err := b.WriteAt(at*w*4, zs[:(r.base-at)*w*4]); err != nil {
					return err
				}
			}
			at = max(at, r.base+r.rows)
		}
		return nil
	}
	var err error
	for i := 0; i < 2 && err == nil; i++ {
		if i == 0 || !shared {
			// A short convolution keeps no state beside its window, and an
			// empty buffer is no buffer on every backend: four bytes it never
			// reads stand in.
			if p.s[i], err = g.dev.Alloc(max(sB, 4)); err != nil {
				break
			}
			if err = zero(p.s[i], g.recS); err != nil {
				break
			}
		}
		if p.c[i], err = g.dev.Alloc(cB); err != nil {
			break
		}
		err = zero(p.c[i], g.recC)
	}
	if err != nil {
		p.free()
		return nil, err
	}
	return p, nil
}

func (p *recPool) free() {
	for i := 0; i < 2; i++ {
		if p.s[i] != nil {
			p.s[i].Free()
			p.s[i] = nil
		}
		if p.c[i] != nil {
			p.c[i].Free()
			p.c[i] = nil
		}
	}
}

// recResize rebuilds every pool at slots slots in the given form, keeping each
// seat's state: the doubled form keeps both halves of a slot, and a move to
// the shared form keeps the half the session reads next. The states cross on
// the device (backend.Device.Copy), one copy per run of adjacent seats, and
// never come to the host. A seat past the new size is the caller's to have
// dropped. Callers hold g.mu, outside a session.
func (g *devTier) recResize(slots int, shared bool) error {
	// Every session's seat is copied out of buffers about to be freed.
	g.quiesce()
	if g.recShared && !shared && len(g.recPools()) > 0 {
		return fmt.Errorf("a shared recurrent pool does not go back to two halves while a session holds it")
	}
	for _, l := range g.recPools() {
		old := l.pool
		keep := min(old.slots, slots)
		// carried is the seats of sessions holding this block whose state
		// pick selects, in slot order.
		carried := func(pick func(*recPair) bool) []recSeat {
			var seats []recSeat
			for sid, st := range g.recSeats {
				if rp := l.rec[sid]; rp != nil && st.base+st.rows <= keep && pick(rp) {
					seats = append(seats, st)
				}
			}
			return seatRuns(seats)
		}
		every := func(*recPair) bool { return true }
		// Every buffer of the new pool receives every held seat -- in the
		// shared form's one state, from whichever half the seat reads -- so
		// that is what the host need not zero.
		filled := carried(every)
		if recFaulted("resize") {
			filled = nil
		}
		np, err := g.newRecPool(slots, shared, filled)
		if err != nil {
			return err
		}
		// move copies the seats whose session passes pick from src to dst, at
		// the same slots.
		move := func(src, dst backend.Buf, w int, pick func(*recPair) bool) error {
			if src == nil || dst == nil || w == 0 || recFaulted("resize") {
				return nil
			}
			for _, r := range carried(pick) {
				off, n := r.base*w*4, r.rows*w*4
				if err := g.dev.Copy(dst, off, src, off, n); err != nil {
					return err
				}
				g.RecCarryBytes += uint64(n)
			}
			return nil
		}
		for i := 0; i < 2 && err == nil; i++ {
			if err = move(old.c[i], np.c[i], g.recC, every); err == nil && shared == g.recShared {
				err = move(old.s[i], np.s[i], g.recS, every)
			}
		}
		// Into the shared form, each session's current half becomes the one
		// state.
		for h := 0; h < 2 && err == nil && shared && !g.recShared; h++ {
			err = move(old.s[h], np.s[0], g.recS, func(rp *recPair) bool { return rp.cur == h })
		}
		if err != nil {
			np.free()
			return err
		}
		ob, nb := l.recBytes, uint64(slots)*g.recSlotBytes(shared)
		old.free()
		l.pool = np
		g.refund(ob)
		g.charge(nb)
		g.RecBytes += nb - ob
		l.recBytes = nb
		if shared {
			g.RecShared++
		}
	}
	g.recSlots, g.recShared = slots, shared
	// A recording names the old buffers by address.
	g.dropGraph()
	return nil
}

// recZero zeroes slots [base, base+rows) in every half of l's pool, or of
// every pool when l is nil: a seat a session takes must not start from the
// last holder's summary. Callers hold g.mu.
func (g *devTier) recZero(l *layer, base, rows int) error {
	ls := []*layer{l}
	if l == nil {
		ls = g.recPools()
	}
	return g.recZeroLayers(ls, base, rows)
}

// recZeroLayers is recZero over the pools of ls. The zeros are uploaded once,
// into a device buffer each half is copied from (Device.Copy) and kept with
// the pools for the next zeroing (recZeros): a write from
// the host per buffer put one upload of the whole slot -- 2 MiB a delta-rule
// state on a 4B qwen35 -- on the bus for every half of every block, about 30
// ms of a reset on a V100, where a copy on the card is microseconds. Callers
// hold g.mu, outside a session.
func (g *devTier) recZeroLayers(ls []*layer, base, rows int) error {
	n := rows * max(g.recS, g.recC) * 4
	if n == 0 {
		return nil
	}
	var zb backend.Buf
	var zeros []byte
	hostZeros := false
	if uint64(n) <= g.recZerosBytes {
		zb = g.recZeros
	}
	for _, l := range ls {
		if l == nil || l.pool == nil {
			continue
		}
		for i := 0; i < 2; i++ {
			for _, b := range []struct {
				buf backend.Buf
				w   int
			}{{l.pool.s[i], g.recS}, {l.pool.c[i], g.recC}} {
				if b.buf == nil || b.w == 0 {
					continue
				}
				if zb == nil && !hostZeros {
					var err error
					if roomFaulted("nozeros") || roomFaulted("nozeros-nohost") {
						err = fmt.Errorf("no room for the kept zeros (roomFault)")
					} else {
						zb, err = g.allocBytes(make([]byte, n))
					}
					if err != nil && roomFaulted("nozeros-nohost") {
						return err
					}
					if err != nil {
						// A card with no room for the kept zeros still has its
						// pools: their slots take zeros written from the host,
						// as a new pool's do (newRecPool).
						hostZeros = true
					} else {
						g.RecZeroSource += uint64(n)
						g.dropRecZeros()
						g.recZeros, g.recZerosBytes = zb, uint64(n)
					}
				}
				if hostZeros {
					if len(zeros) < rows*b.w*4 {
						zeros = make([]byte, rows*b.w*4)
					}
					if err := b.buf.WriteAt(base*b.w*4, zeros[:rows*b.w*4]); err != nil {
						return err
					}
					g.RecZeroBytes += uint64(rows * b.w * 4)
					g.RecZeroHost++
					continue
				}
				// Run with every other block's as one list, below.
				g.zeroList = append(g.zeroList, backend.CopyRange{Dst: b.buf, DstOff: base * b.w * 4, Src: zb, N: rows * b.w * 4})
			}
		}
	}
	if len(g.zeroList) == 0 {
		return nil
	}
	err := backend.CopyAll(g.dev, g.zeroList)
	if err == nil {
		for _, c := range g.zeroList {
			g.RecZeroBytes += uint64(c.N)
		}
		g.RecZeroCopies += uint64(len(g.zeroList))
	}
	clear(g.zeroList)
	g.zeroList = g.zeroList[:0]
	return err
}

// growSeat gives session sid a seat of at least rows slots: a batch
// keeps one state per sequence, and its seat grows to the highest sequence a
// step names -- a scheduler admits sequences into rows as they free, so the
// first steps need not name them all. The slots the seat had keep their
// states, wherever the seat lands; the slots it gains start zeroed. Every other
// session's seat is kept. When the pools' two halves do not fit at the new size
// (or Config.SharedRec asks), every pool takes the shared form. Callers hold
// g.mu, outside a session.
func (g *devTier) growSeat(sid uint64, rows int) bool {
	if rows > 1 && (roomFaulted("noseat") || roomFaulted("noseat-lie")) {
		g.LastErr = "rows: a batch's recurrent states do not fit in this device's budget (roomFault)"
		return false
	}
	old, had := g.recSeats[sid]
	if had && old.rows >= rows {
		return true
	}
	// The pools are rebuilt with every other session's seat carried across.
	g.quiesce()
	if g.recSeats == nil {
		g.recSeats = map[uint64]recSeat{}
	}
	// The seat's states, set aside on the device before the pools are
	// rebuilt: a rebuild carries the other seats, and this one may move.
	var kept seatStates
	wasShared := g.recShared
	if had {
		var err error
		if kept, err = g.readSeat(old); err != nil {
			g.LastErr = "rows: " + err.Error()
			return false
		}
		defer kept.free()
	}
	delete(g.recSeats, sid)
	restore := func() {
		if had {
			g.recSeats[sid] = old
		}
	}
	base := g.seatPlace(sid, rows)
	slots := max(g.recSlots, base+rows)
	pools := uint64(len(g.recPools()))
	cost := func(shared bool) uint64 {
		var held uint64
		for _, l := range g.recPools() {
			held += l.recBytes
		}
		next := pools * uint64(slots) * g.recSlotBytes(shared)
		if next <= held {
			return 0
		}
		return next - held
	}
	shared := g.recShared || g.SharedRec || !g.room(cost(false))
	need := cost(shared)
	var extra uint64
	if shared {
		if want := uint64(rows * g.recS * 4); g.recNextBytes < want {
			extra = want - g.recNextBytes
		}
	}
	if need+extra > 0 && !g.stateFits(need+extra, nil) {
		g.LastErr = fmt.Sprintf("rows: a batch's recurrent states do not fit in this device's budget: "+
			"%d sequences want %.1f MiB a block (%.1f MiB doubled), %.1f MiB used of %.1f",
			rows, float64(uint64(rows)*g.recSlotBytes(true))/(1<<20),
			float64(uint64(rows)*g.recSlotBytes(false))/(1<<20),
			float64(g.used)/(1<<20), float64(g.limit)/(1<<20))
		restore()
		return false
	}
	if slots != g.recSlots || shared != g.recShared {
		if err := g.recResize(slots, shared); err != nil {
			g.LastErr = "rows: " + err.Error()
			restore()
			return false
		}
	}
	// The batch's copy home, and a single sequence's for the session's own
	// decode and prompts.
	if shared && !(g.ensureRecNext(rows) && g.ensureRecNext(1)) {
		restore()
		return false
	}
	g.recSeats[sid] = recSeat{base, rows}
	if err := g.recZero(nil, base, rows); err != nil {
		g.LastErr = "rows: " + err.Error()
		return false
	}
	if had && !recFaulted("resize") {
		if err := g.writeSeat(sid, kept, base, old.rows, wasShared); err != nil {
			g.LastErr = "rows: " + err.Error()
			return false
		}
	}
	return true
}

// seatStates is a seat's states in every pool, each in a device buffer of its
// own: per pool, the delta state's and the window's two halves ([0] and [1] of
// each), nil where the form has no such half.
type seatStates map[*layer][2][2]backend.Buf

// readSeat copies st's slots of every pool into buffers of their own on the
// device, which a rebuild leaves alone: the seat may move, and its states
// must not come home through the host to follow it. Callers hold g.mu,
// outside a session, and free the result (seatStates.free).
func (g *devTier) readSeat(st recSeat) (seatStates, error) {
	out := seatStates{}
	for _, l := range g.recPools() {
		var v [2][2]backend.Buf
		for k, b := range [2]struct {
			buf [2]backend.Buf
			w   int
		}{{l.pool.s, g.recS}, {l.pool.c, g.recC}} {
			for h := 0; h < 2; h++ {
				// A short convolution's state slot is empty (recS 0).
				if b.buf[h] == nil || b.w == 0 {
					continue
				}
				n := st.rows * b.w * 4
				tmp, err := g.dev.Alloc(n)
				if err == nil {
					err = g.dev.Copy(tmp, 0, b.buf[h], st.base*b.w*4, n)
					g.RecCarryBytes += uint64(n)
				}
				if err != nil {
					if tmp != nil {
						tmp.Free()
					}
					out[l] = v
					out.free()
					return nil, err
				}
				v[k][h] = tmp
			}
		}
		out[l] = v
	}
	return out, nil
}

// free gives back the buffers readSeat took.
func (kept seatStates) free() {
	for _, v := range kept {
		for _, k := range v {
			for _, b := range k {
				if b != nil {
					b.Free()
				}
			}
		}
	}
}

// writeSeat puts states readSeat took back at slot base of every pool, on the
// device. A pool that went from two halves to the shared form takes, as its
// one state, the half the session reads next, as recResize does for every
// other seat. Callers hold g.mu, outside a session.
func (g *devTier) writeSeat(sid uint64, kept seatStates, base, rows int, wasShared bool) error {
	for l, v := range kept {
		if l.pool == nil {
			continue
		}
		cur := 0
		if rp := l.recOf(sid); rp != nil {
			cur = rp.cur
		}
		for k, b := range [2]struct {
			buf [2]backend.Buf
			w   int
		}{{l.pool.s, g.recS}, {l.pool.c, g.recC}} {
			for h := 0; h < 2; h++ {
				src := v[k][h]
				if k == 0 && g.recShared && !wasShared {
					src = v[0][cur]
				}
				if b.buf[h] == nil || src == nil {
					continue
				}
				if err := g.dev.Copy(b.buf[h], base*b.w*4, src, 0, rows*b.w*4); err != nil {
					return err
				}
				g.RecCarryBytes += uint64(rows * b.w * 4)
			}
		}
	}
	return nil
}

// recTidy gives back what no session holds: a pool no session is attached
// to, a seat whose session holds no linear block here, and the slots past
// the last seat -- so a session's leaving returns its state to the budget
// rather than only to the pool. Callers hold g.mu, outside a session.
func (g *devTier) recTidy() {
	for _, l := range g.layers {
		if l != nil && l.pool != nil && len(l.rec) == 0 {
			l.pool.free()
			l.pool = nil
			g.refund(l.recBytes)
			g.RecBytes -= min(g.RecBytes, l.recBytes)
			l.recBytes = 0
		}
	}
	for sid := range g.recSeats {
		held := false
		for _, l := range g.layers {
			if l != nil && l.rec[sid] != nil {
				held = true
				break
			}
		}
		if !held {
			delete(g.recSeats, sid)
		}
	}
	if len(g.recPools()) == 0 {
		g.recSlots, g.recShared = 0, false
		g.dropRecZeros()
		if g.recNext != nil {
			g.recNext.Free()
			g.refund(g.recNextBytes)
			g.recNext, g.recNextBytes = nil, 0
			g.dropGraph()
		}
		return
	}
	if end := g.recEnd(); end < g.recSlots {
		if err := g.recResize(end, g.recShared); err != nil {
			// The pools keep their size: room not given back, nothing lost.
			g.LastErr = "recurrent pools did not shrink: " + err.Error()
		}
	}
}

// dropRecZeros frees the kept zeros (recZeros). Callers hold g.mu.
func (g *devTier) dropRecZeros() {
	if g.recZeros != nil {
		g.recZeros.Free()
		g.recZeros, g.recZerosBytes = nil, 0
	}
}

// stepPar is the half of l's pool the running step reads: session sid's,
// or on a step across sessions the rows' common one (alignRec made
// them agree). Callers hold g.mu.
func (g *devTier) stepPar(sid uint64, l *layer) (int, bool) {
	if g.rag != nil && g.rag.sid != nil {
		sid = g.rag.sid[0]
	}
	rp := l.recOf(sid)
	if rp == nil {
		return 0, false
	}
	return rp.cur, true
}

// stepFlip moves every session of the running step to the half its step
// wrote, once a session however many of its rows the step carried. Callers
// hold g.mu.
func (g *devTier) stepFlip(sid uint64, l *layer) {
	if g.rag != nil && g.rag.sid != nil {
		for _, sid := range g.rag.sess {
			if rp := l.recOf(sid); rp != nil {
				rp.cur = 1 - rp.cur
				rp.steps++
			}
		}
		return
	}
	if rp := l.recOf(sid); rp != nil {
		rp.cur = 1 - rp.cur
		rp.steps++
	}
}

// groupRuns groups a ragged step's rows into runs, one a sequence: the rows
// of one session at one base (nn.RowsDevice's slots), each run's rows in
// position order, with the run's session and its seat row -- its sequence's
// place in the session's batch, (slot-pos)/seqLen. A step across sessions
// seats each session's one sequence at row 0.
//
// A run chains one state through its rows (kernels' run forms), so its
// positions must be consecutive: a gap would step the state past a token no
// row ran, a repeat would step it twice. Either is refused. It also fills
// rs.sess, the step's distinct sessions.
func (g *devTier) groupRuns(sid uint64, rs *ragStep) error {
	rs.runs, rs.runSid, rs.runSeat, rs.sess = rs.runs[:0], rs.runSid[:0], rs.runSeat[:0], rs.sess[:0]
	keys := rs.keys[:0]
	defer func() { rs.keys = keys[:0] }()
	for r := range rs.pos {
		k := runKey{sid, rs.slot[r] - rs.pos[r]}
		if rs.sid != nil {
			k.sid = rs.sid[r]
		}
		j := slices.Index(keys, k)
		if j < 0 {
			seat := 0
			if rs.sid == nil {
				if rs.seqLen <= 0 || k.base%rs.seqLen != 0 {
					return fmt.Errorf("row %d's sequence starts at slot %d, not a multiple of the %d positions a sequence spans",
						r, k.base, rs.seqLen)
				}
				seat = k.base / rs.seqLen
			}
			j = len(keys)
			keys = append(keys, k)
			// A run's row list keeps the backing an earlier step gave it.
			if len(rs.runs) < cap(rs.runs) {
				rs.runs = rs.runs[:len(rs.runs)+1]
				rs.runs[j] = rs.runs[j][:0]
			} else {
				rs.runs = append(rs.runs, nil)
			}
			rs.runSid, rs.runSeat = append(rs.runSid, k.sid), append(rs.runSeat, seat)
			if !slices.Contains(rs.sess, k.sid) {
				rs.sess = append(rs.sess, k.sid)
			}
		}
		rs.runs[j] = append(rs.runs[j], r)
	}
	for _, run := range rs.runs {
		slices.SortFunc(run, func(a, b int) int { return rs.pos[a] - rs.pos[b] })
		for k := 1; k < len(run); k++ {
			if a, b := run[k-1], run[k]; rs.pos[b] != rs.pos[a]+1 {
				return fmt.Errorf("rows %d and %d of one sequence are at positions %d and %d: "+
					"a sequence's rows in one step are consecutive positions", a, b, rs.pos[a], rs.pos[b])
			}
		}
	}
	if recFaulted("chain") {
		// Every row its own run at its sequence's seat: a chunk's rows each
		// read the state from before the step.
		var runs [][]int
		var sids []uint64
		var seats []int
		for j, run := range rs.runs {
			for _, r := range run {
				runs, sids, seats = append(runs, []int{r}), append(sids, rs.runSid[j]), append(seats, rs.runSeat[j])
			}
		}
		rs.runs, rs.runSid, rs.runSeat = runs, sids, seats
	}
	return nil
}

// RecMark is recMark for the zero session: a caller that never attached.
func (g *devTier) RecMark() {
	g.recMark(0)
}

// recMark records the half each of session sid's linear blocks reads
// (nn.RecRewinder).
func (g *devTier) recMark(sid uint64) {
	v := g.as(sid)
	if v == nil {
		return
	}
	defer v.done()
	g = v
	g.mu.Lock()
	defer g.mu.Unlock()
	for _, l := range g.layers {
		if rp := l.recOf(sid); l != nil && l.linear && rp != nil {
			rp.mark, rp.markSteps, rp.marked = rp.cur, rp.steps, true
		}
	}
}

// RecRewind is recRewind for the zero session: a caller that never attached.
func (g *devTier) RecRewind() bool {
	return g.recRewind(0)
}

// recRewind takes session sid's linear blocks back to the state
// RecMark saw. A step reads one half of the pool and writes the other, so
// after exactly one step the marked state is still there, untouched, and the
// rewind is a flip back. After more than one, or in the shared form (one copy
// of the delta state, written in place through recNext), it is gone: false,
// with nothing changed. Every block is checked before any is flipped.
func (g *devTier) recRewind(sid uint64) bool {
	v := g.as(sid)
	if v == nil {
		return false
	}
	defer v.done()
	g = v
	back, ok := g.rewindable(sid)
	if !ok {
		return false
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	for _, rp := range back {
		rp.cur, rp.marked = rp.mark, false
		rp.steps++
	}
	// The recording a key's parity names is still right: the parity is cur.
	return true
}

// rewindable is RecRewind's check alone: the blocks a rewind would flip, or
// false when the marked state is gone from any of them.
func (g *devTier) rewindable(sid uint64) ([]*recPair, bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	var back []*recPair
	for _, l := range g.layers {
		rp := l.recOf(sid)
		if l == nil || !l.linear || rp == nil {
			continue
		}
		switch {
		case !rp.marked:
			g.LastErr = "recurrent rewind: a linear block with no mark"
			return nil, false
		case rp.steps == rp.markSteps:
			continue // not stepped since the mark: already there
		case g.recShared || rp.steps != rp.markSteps+1:
			g.LastErr = fmt.Sprintf("recurrent rewind: %d step(s) since the mark, shared form %v",
				rp.steps-rp.markSteps, g.recShared)
			return nil, false
		}
		back = append(back, rp)
	}
	return back, true
}

// runKey is a sequence as groupRuns tells them apart: its session and the
// slot its positions start at.
type runKey struct {
	sid  uint64
	base int
}

// recDesc is the recurrent descriptor the running call stages: for a chunk or
// a decode, the real row count and session sid's one slot pair
// (kernels.RecDescWords); for a ragged step, the run descriptor
// (kernels.RunDescWords) over the step's runs (groupRuns), each run's slots
// being its sequence's seat row of its session's seat. In the shared form a
// run's out slot is its place in recNext. Callers hold g.mu.
func (g *devTier) recDesc(sid uint64, nrow int) []byte {
	rag := g.rag
	slots := func(sid uint64, seat, j int) (uint32, uint32) {
		if recFaulted("slot") && rag != nil && rag.sid != nil {
			sid = rag.sid[0]
		}
		in := g.recSeats[sid].base + seat
		out := in
		if g.recShared {
			out = j
		}
		return uint32(in), uint32(out)
	}
	// The words are the devTier's, reused: the descriptor is written to the
	// device before the submission goes on, and read by nothing after.
	words := func(k int) []uint32 {
		w := slices.Grow(g.recDescW[:0], k)[:k]
		clear(w)
		g.recDescW = w
		return w
	}
	if rag == nil || len(rag.runs) == 0 {
		in, out := slots(sid, 0, 0)
		w := words(3)
		w[0], w[1], w[2] = uint32(nrow), in, out
		return u32view(w)
	}
	n, S := len(rag.pos), len(rag.runs)
	w := words(kernels.RunDescWords(n))
	w[0] = uint32(nrow)
	at := 0
	for j := 0; j < n; j++ {
		// A step of S runs repeats run S-1 in the entries past it.
		s, start := min(j, S-1), at
		if j >= S {
			start = at - len(rag.runs[s])
		}
		w[1+2*j], w[2+2*j] = slots(rag.runSid[s], rag.runSeat[s], s)
		w[1+2*n+2*j], w[2+2*n+2*j] = uint32(start), uint32(len(rag.runs[s]))
		if j < S {
			for k, r := range rag.runs[j] {
				w[1+4*n+at+k] = uint32(r)
				w[1+5*n+r] = uint32(at + k)
				w[1+6*n+r] = uint32(j)
			}
			at += len(rag.runs[j])
		}
	}
	return u32view(w)
}

// slotCopy is CopySlots for w-float slots over rows sequences, compiled on
// first use -- outside any session, which ensureRecNext and alignRec are.
// Callers hold g.mu.
func (g *devTier) slotCopy(w, rows int) (backend.Kernel, error) {
	key := [2]int{w, rows}
	if k, ok := g.recCopies[key]; ok {
		return k, nil
	}
	kk, err := kernels.CopySlots(w, rows)
	if err != nil {
		return nil, err
	}
	k, err := g.dev.Compile(kk)
	if err != nil {
		return nil, err
	}
	if g.recCopies == nil {
		g.recCopies = map[[2]int]backend.Kernel{}
	}
	g.recCopies[key] = k
	return k, nil
}

// linearIn reports whether any block in [lo, hi) is a linear one. Callers
// hold g.mu.
func (g *devTier) linearIn(lo, hi int) bool {
	for li := lo; li < hi; li++ {
		if l := g.layers[li]; l != nil && l.linear {
			return true
		}
	}
	return false
}

// prepRecRows readies the recurrent pools for a ragged step over [lo, hi)
// whose rows groupRuns grouped, outside any session. One session's batch keeps
// a state per sequence, its seat grown to the highest sequence the step names
// (growSeat keeps the states it had). A step across sessions takes each
// session's one state where it is, after bringing the sessions to one half of
// every pool. Callers hold g.mu.
func (g *devTier) prepRecRows(sid uint64, lo, hi int, rs *ragStep) bool {
	if rs.sid == nil {
		seats := 0
		for _, q := range rs.runSeat {
			seats = max(seats, q+1)
		}
		if !g.growSeat(sid, seats) {
			return false
		}
	} else {
		for _, s := range rs.sess {
			if st, ok := g.recSeats[s]; !ok || st.rows != 1 {
				g.LastErr = fmt.Sprintf("sessions: session %d holds %d recurrent state(s) here; "+
					"a step across sessions takes one a session", s, st.rows)
				return false
			}
			for li := lo; li < hi; li++ {
				if l := g.layers[li]; l != nil && l.linear && l.recOf(s) == nil {
					g.LastErr = fmt.Sprintf("sessions: block %d holds no recurrent state for session %d", li, s)
					return false
				}
			}
		}
		if !g.alignRec(lo, hi, rs.sess) {
			return false
		}
	}
	// recNext holds a next state a run, and the copy home is compiled for the
	// step's row count, which the run descriptor is sized by.
	return !g.recShared || g.ensureRecNext(len(rs.pos))
}

// alignRec brings every session of a step across sessions (sid, each once) to
// one half of each linear block's pool in [lo, hi): the half most of them
// read, a session in the minority having its state copied across. Sessions
// that keep stepping together stay aligned, so this copies only when the
// step's company changes. Callers hold g.mu, outside a session.
func (g *devTier) alignRec(lo, hi int, sid []uint64) bool {
	// It copies other sessions' states between halves.
	g.quiesce()
	for li := lo; li < hi; li++ {
		l := g.layers[li]
		if l == nil || !l.linear {
			continue
		}
		ones := 0
		for _, s := range sid {
			ones += l.recOf(s).cur
		}
		want := 0
		if 2*ones > len(sid) || 2*ones == len(sid) && l.recOf(sid[0]).cur == 1 {
			want = 1
		}
		for _, s := range sid {
			rp := l.recOf(s)
			if rp.cur == want {
				continue
			}
			if !recFaulted("parity") && !g.moveHalf(l, g.recSeats[s], rp.cur, want) {
				return false
			}
			rp.cur = want
			g.RecAligns++
		}
	}
	return true
}

// moveHalf copies a seat's state in l's pool from half `from` to half `to`,
// on the device: the window always, the delta state where it has two halves.
// Callers hold g.mu, outside a session.
func (g *devTier) moveHalf(l *layer, st recSeat, from, to int) bool {
	for _, b := range []struct {
		src, dst backend.Buf
		w        int
	}{{l.pool.c[from], l.pool.c[to], g.recC}, {l.pool.s[from], l.pool.s[to], g.recS}} {
		if b.src == nil || b.dst == nil || b.w == 0 {
			continue // the shared form's one state has no other half, a short convolution none
		}
		off := st.base * b.w * 4
		if err := g.dev.Copy(b.dst, off, b.src, off, st.rows*b.w*4); err != nil {
			g.LastErr = "sessions: " + err.Error()
			return false
		}
	}
	return true
}

// seatRuns merges seats that sit back to back into one run, in slot order, so
// a pool's resize is one copy per run rather than one per session.
func seatRuns(seats []recSeat) []recSeat {
	slices.SortFunc(seats, func(a, b recSeat) int { return a.base - b.base })
	var runs []recSeat
	for _, st := range seats {
		if k := len(runs) - 1; k >= 0 && runs[k].base+runs[k].rows == st.base {
			runs[k].rows += st.rows
			continue
		}
		runs = append(runs, st)
	}
	return runs
}
