package run

import (
	"fmt"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/page"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
)

// Residency is a model's weights on one device, shared by every runner of
// the model there (WithResidency): the paging of block paging
// (docs/design/block-paging.md) on the plan, its policy the one the old
// engine path's device tier runs (page.Pool, page.Rec, page.Pick,
// page.BudgetFor). The dense region's weights are permanent; every block's
// are pageable, whatever the model's size: a block is paged in when a
// launch binds it and out when the budget needs its room, and on a model
// that fits eviction never fires. The sessions' own buffers -- the KV
// pages, the scratch -- are charged as the permanent part, and a page goes
// out to make room for them, since a page can come back and history
// cannot. Every block is in the one block space (jlm.File.BlockOf): a
// tower's blocks page as the text's do. Its state changes on the device's
// own thread alone (backend.Device.Session), where every launch reading it
// runs: two sessions of a model on two goroutines meet there.
type Residency struct {
	f     Source
	dev   backend.Device
	pool  *page.Pool
	limit uint64
	// used is every byte charged: resident pages, dense weights, sessions'
	// buffers; pageBytes the resident pages' share.
	used, pageBytes uint64
	of              map[wkey]*wstore
	blocks          map[int]*wblock
	dense           []*wstore
	// gen counts page-ins and page-outs: a session rebinds its arguments,
	// and drops its recordings, when it moved.
	gen      int
	pagesIn  int
	PageIns  int
	PageOuts int
	ready    bool
	// guard is the poisoned bytes past every buffer (WithGuard), poison
	// the 0xff a buffer's tail is written with, reused.
	guard  int
	poison []byte
}

// wkey names a weight storage in every runner of the model.
type wkey struct {
	seg          jit.Segment
	role         jlm.Role
	block, index int32
	region       int8
}

// wstore is one weight region: its bytes as the container holds them, and
// its device buffer while resident.
type wstore struct {
	key   wkey
	e     *jlm.Entry
	bytes uint64
	buf   backend.Buf
}

// wblock is one block's pageable weights and its page record.
type wblock struct {
	rec    page.Rec
	stores []*wstore
}

// NewResidency is f's weights on dev, spending from pool (nil: an account of
// the device's own) within limit bytes (0: page.BudgetFor of what the device
// reports free).
func NewResidency(f Source, dev backend.Device, pool *page.Pool, limit uint64) *Residency {
	if pool == nil {
		pool = page.NewPool(dev.Name(), 0)
	}
	if limit == 0 {
		free, total, err := dev.Mem()
		limit, _ = page.BudgetFor(free, total, err, 0, "", defaultBudget)
	}
	return &Residency{f: f, dev: dev, pool: pool, limit: limit, of: map[wkey]*wstore{}, blocks: map[int]*wblock{}}
}

// defaultBudget is the budget when neither the caller nor the device says,
// the device tier's.
const defaultBudget = 2 << 30

// WithResidency has the runner bind res's weights in place of holding its
// own: a model's runners (a token a step, a prompt's chunks) share them.
func WithResidency(res *Residency) Option { return func(r *Runner) { r.res = res } }

// Blocks is how many blocks have pageable weights, and Resident how many of
// them are in.
func (res *Residency) Blocks() int   { return len(res.blocks) }
func (res *Residency) Resident() int { return res.pagesIn }

// Widest is the largest block's pageable bytes, and Permanent what is
// charged that no page-out gives back: a budget of Permanent+n*Widest holds
// n blocks.
func (res *Residency) Widest() uint64 {
	var w uint64
	for _, b := range res.blocks {
		w = max(w, b.rec.Bytes)
	}
	return w
}
func (res *Residency) Permanent() uint64 { return res.used - res.pageBytes }

// add registers weight storage st, uploading on se what is resident.
func (res *Residency) add(se backend.Session, st jit.Storage) (*wstore, error) {
	k := wkey{st.Segment, st.Role, int32(st.Block), st.Index, int8(st.Region)}
	if s := res.of[k]; s != nil {
		if uint64(st.Bytes) > s.bytes {
			return nil, fmt.Errorf("run: a runner reads %d bytes of tensor %v of block %d, which the model's weights hold in %d", st.Bytes, st.Role, st.Block, s.bytes)
		}
		return s, nil
	}
	if st.Region < 0 || st.Region > 2 {
		return nil, fmt.Errorf("run: tensor %v of block %d has no region %d", st.Role, st.Block, st.Region)
	}
	view := jit.Model(res.f)
	if st.Segment == jit.SegmentVision {
		view = res.f.Segment()
	}
	e, ok := view.Find(st.Role, int32(st.Block), st.Index)
	if !ok {
		return nil, fmt.Errorf("run: no tensor %v of block %d in the container", st.Role, st.Block)
	}
	n, err := res.length(e, int(st.Region))
	if err != nil {
		return nil, err
	}
	s := &wstore{key: k, e: e, bytes: max(uint64(st.Bytes), n)}
	res.of[k] = s
	if res.f.InDense(e) {
		if err := res.upload(se, s); err != nil {
			return nil, err
		}
		res.dense = append(res.dense, s)
		res.charge(s.bytes)
		return s, nil
	}
	b := res.f.BlockOf(e)
	wb := res.blocks[b]
	if wb == nil {
		wb = &wblock{}
		res.blocks[b] = wb
	}
	if wb.rec.In {
		// A block already in gains a region: it comes in with the rest.
		if err := res.upload(se, s); err != nil {
			return nil, err
		}
		res.charge(s.bytes)
		res.pageBytes += s.bytes
	}
	wb.stores = append(wb.stores, s)
	wb.rec.Bytes += s.bytes
	return s, nil
}

// length is region i of e's bytes as the container holds it.
func (res *Residency) length(e *jlm.Entry, i int) (uint64, error) {
	if res.f.ExpertPaged(e) {
		// A verbatim bank has no planes to split, and its whole span is
		// SpanLen's (jlm.File.SheetSpan).
		if qs, d, sc, sheets, err := res.f.SheetSpan(e); err == nil {
			return [3]uint64{qs, d, sc}[i] * sheets, nil
		}
	}
	qs, d, sc := res.f.SpanLen(e)
	return [3]uint64{qs, d, sc}[i], nil
}

// fill pages in every block the budget holds, in block order: a model that
// fits is wholly resident before its first step, and pays nothing after.
func (res *Residency) fill(se backend.Session) error {
	if res.ready {
		return nil
	}
	res.ready = true
	for b := 0; b < res.maxBlock()+1; b++ {
		wb := res.blocks[b]
		if wb == nil || wb.rec.In {
			continue
		}
		if !res.room(wb.rec.Bytes) {
			break
		}
		if err := res.pageIn(se, b, 0, 0); err != nil {
			return err
		}
	}
	return nil
}

func (res *Residency) maxBlock() int {
	m := -1
	for b := range res.blocks {
		m = max(m, b)
	}
	return m
}

// room reports whether n more bytes fit the device's limit and its pool.
func (res *Residency) room(n uint64) bool {
	return res.used+n <= res.limit && res.pool.Room(n)
}

func (res *Residency) charge(n uint64) { res.used += n; res.pool.Take(n) }

func (res *Residency) refund(n uint64) {
	n = min(n, res.used)
	res.used -= n
	res.pool.Give(n)
}

// victim is the block a page-in of cur evicts while scan [lo, hi) runs, by
// the shared policy; held blocks (a launch binding them) are no candidates.
func (res *Residency) victim(cur, lo, hi int) int {
	p := page.NewPick(cur, lo, hi)
	for b, wb := range res.blocks {
		p.Offer(b, &wb.rec)
	}
	return p.Victim()
}

// reclaim pages blocks out until n bytes fit, se synced first so no launch
// queued on it still reads a victim.
func (res *Residency) reclaim(se backend.Session, n uint64, cur, lo, hi int) error {
	synced := false
	for !res.room(n) {
		v := res.victim(cur, lo, hi)
		if v < 0 {
			return fmt.Errorf("run: %d bytes do not fit the device's budget of %d with every block it can give back out (%d charged)", n, res.limit, res.used)
		}
		if !synced {
			if err := se.Sync(); err != nil {
				return err
			}
			synced = true
		}
		res.pageOut(v)
	}
	return nil
}

// pageIn brings block b's weights onto the device, evicting by the policy.
func (res *Residency) pageIn(se backend.Session, b, lo, hi int) error {
	wb := res.blocks[b]
	if wb == nil || wb.rec.In {
		return nil
	}
	if err := res.reclaim(se, wb.rec.Bytes, b, lo, hi); err != nil {
		return err
	}
	for _, s := range wb.stores {
		if err := res.upload(se, s); err != nil {
			return err
		}
	}
	wb.rec.In = true
	res.charge(wb.rec.Bytes)
	res.pageBytes += wb.rec.Bytes
	res.pagesIn++
	res.PageIns++
	res.gen++
	return nil
}

// pageOut frees block b's weights, its record kept.
func (res *Residency) pageOut(b int) {
	wb := res.blocks[b]
	if wb == nil || !wb.rec.In {
		return
	}
	for _, s := range wb.stores {
		if s.buf != nil {
			s.buf.Free()
			s.buf = nil
		}
	}
	wb.rec.In = false
	res.refund(wb.rec.Bytes)
	res.pageBytes -= min(res.pageBytes, wb.rec.Bytes)
	res.pagesIn--
	res.PageOuts++
	res.gen++
}

// upload is s's bytes in a buffer of its own, its page held while they are
// copied (Runner.weight's rule).
func (res *Residency) upload(se backend.Session, s *wstore) error {
	data, release, err := regionOf(res.f, s.e, int(s.key.region))
	if err != nil {
		return err
	}
	defer release()
	// Poisoned past the bytes (RULE 13): a fresh buffer is often zero,
	// and a kernel reading past its weight would read nothing visible.
	n := int(s.bytes) + res.guard
	b, err := res.dev.Alloc(n)
	if err != nil {
		return err
	}
	tail := n - len(data)
	if len(res.poison) < tail {
		res.poison = make([]byte, tail)
		for i := range res.poison {
			res.poison[i] = 0xff
		}
	}
	if err = se.WriteAt(b, 0, data); err == nil && tail > 0 {
		err = se.WriteAt(b, len(data), res.poison[:tail])
	}
	if err != nil {
		b.Free()
		return err
	}
	s.buf = b
	return nil
}

// SetBudget retargets the weights' device budget while they run, paging
// blocks out until they fit (the device tier's SetBudget): a block paged
// out keeps its record and comes back when a launch binds it.
func (res *Residency) SetBudget(bytes uint64) {
	res.dev.Session(func(se backend.Session) {
		res.limit = bytes
		synced := false
		for res.used > res.limit && res.pagesIn > 0 {
			v := res.victim(-1, 0, 0)
			if v < 0 {
				return
			}
			if !synced {
				if se.Sync() != nil {
					return
				}
				synced = true
			}
			res.pageOut(v)
		}
	})
}

// ensure pages in the blocks a launch binds, each held while the others
// come in so none of them is the victim.
func (res *Residency) ensure(se backend.Session, need []int, lo, hi int) error {
	for _, b := range need {
		if wb := res.blocks[b]; wb != nil {
			wb.rec.Uses++
		}
	}
	defer func() {
		for _, b := range need {
			if wb := res.blocks[b]; wb != nil {
				wb.rec.Uses--
			}
		}
	}()
	for _, b := range need {
		if err := res.pageIn(se, b, lo, hi); err != nil {
			return err
		}
	}
	return nil
}

// Close frees every buffer.
func (res *Residency) Close() {
	res.dev.Session(func(backend.Session) {
		for b := range res.blocks {
			res.pageOut(b)
		}
		for _, s := range res.dense {
			if s.buf != nil {
				s.buf.Free()
				s.buf = nil
			}
		}
		res.dense = nil
	})
}
