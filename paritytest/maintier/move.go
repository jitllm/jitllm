package maintier

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
)

// A block moving between two devices of one tier: the target takes the block
// (staged ahead, or prepared now), the session's history is copied from the
// source's pages into pages the target allocates, the route switches and the
// source frees the block. The history never comes home to the State's cache:
// it goes card to card (backend.PeerCopier) where the backend can, and
// otherwise through a host buffer in chunks, the read of one overlapping the
// write of the last.

// errNotDirect says a move cannot run in the tier -- a device that keeps the
// contiguous cache, a batch's flat rows -- and nothing was changed: the caller
// moves the block through its own cache instead.
var errNotDirect = nn.ErrNotDirect

// moveChunk is the staged path's buffer: big enough that a transfer runs at
// the link's rate, small enough that two of them are nothing beside a card.
const moveChunk = 16 << 20

// StageBlock is stageBlock for the zero session.
func (g *GPU) StageBlock(li int, to string, p *nn.LayerPlan, w *nn.LayerWeights, positions int) bool {
	return g.stageBlock(0, li, to, p, w, positions)
}

// retiree is one thing a move left on its source: block li, or the
// projection (li -1).
type retiree struct {
	d  *devTier
	li int
}

// RetireMoved frees, on another goroutine, what the moves since the last call
// left on their sources: the blocks' weights and history there, and a
// projection that followed them. The routes already name the targets, so no
// step reads them; the frees -- a driver free of every tensor, a graph
// dropped -- are a few milliseconds a block, and inside the pause they were
// most of it. Every placement call settles them first.
func (g *GPU) RetireMoved() {
	g.mu.Lock()
	list := g.retire
	g.retire = nil
	g.mu.Unlock()
	if len(list) == 0 {
		return
	}
	g.retiring.Add(1)
	go func() {
		defer g.retiring.Done()
		for _, r := range list {
			if r.li < 0 {
				r.d.dropHead()
				continue
			}
			r.d.ReleaseLayers(r.li, r.li+1)
		}
	}()
}

// settle waits for RetireMoved's frees.
func (g *GPU) settle() { g.retiring.Wait() }

// stageBlock prepares block li on the named device -- its weights uploaded,
// its kernels built, this session's history structures made -- without
// routing the block there, so a move's commit (moveBlock) has only the
// history left to copy. It runs while the session keeps stepping on the
// block's current device, which is why it takes no call lock: it is refused
// unless the target routes no block at all, so no step reads anything it
// changes. False means it was not staged; the commit then prepares the block
// itself.
func (g *GPU) stageBlock(sid uint64, li int, to string, p *nn.LayerPlan, w *nn.LayerWeights, positions int) bool {
	g.settle()
	d, err := g.device(to)
	if err != nil {
		return false
	}
	g.mu.Lock()
	src := g.own[li]
	busy := src == d || g.staged[li] == d
	for _, o := range g.own {
		busy = busy || o == d
	}
	g.mu.Unlock()
	if busy {
		return false
	}
	done, ok := g.admit(p.Model, fmt.Sprintf("block %d", li))
	if !ok {
		return false
	}
	defer done()
	if !d.prepLayer(sid, li, p, w, false) {
		return false
	}
	// The history's pages, grown ahead (the pool's growth copies and drops
	// graphs, a few milliseconds a block at the commit), and the cards' peer
	// access, enabled ahead: the commit is left the copy.
	d.prewarmKV(positions)
	if pc, ok := d.dev.(backend.PeerCopier); ok && src != nil && d.dev.API() == src.dev.API() {
		pc.PeerAccess(src.dev)
	}
	g.mu.Lock()
	if g.staged == nil {
		g.staged = map[int]*devTier{}
	}
	g.staged[li] = d
	g.mu.Unlock()
	return true
}

// stageHead prepares the output projection on the named device as
// stageBlock prepares a block: uploaded beside the session's steps, on a
// device that routes nothing, so the commit's PrepHeadOn finds it there
// (devTier.adoptHead) instead of uploading inside the pause.
func (g *GPU) stageHead(to string, h *nn.Head) bool {
	g.settle()
	d, err := g.device(to)
	if err != nil {
		return false
	}
	g.mu.Lock()
	busy := g.head == d
	for _, o := range g.own {
		busy = busy || o == d
	}
	g.mu.Unlock()
	if busy {
		return false
	}
	done, ok := g.admit(h.Model, "output projection")
	if !ok {
		return false
	}
	defer done()
	if !d.PrepHead(h) {
		return false
	}
	g.mu.Lock()
	g.stagedHead = d
	g.mu.Unlock()
	return true
}

// StageHead is stageHead; it carries no session.
func (g *GPU) StageHead(to string, h *nn.Head) bool { return g.stageHead(to, h) }

// Unstage frees what StageBlock prepared for blocks [lo, hi), and the
// projection StageHead prepared, that no move committed: a move abandoned
// before its token boundary, or a block its target then refused.
func (g *GPU) Unstage(lo, hi int) {
	g.settle()
	g.mu.Lock()
	head := g.stagedHead
	g.stagedHead = nil
	if head == g.head {
		head = nil
	}
	var drop []*devTier
	var at []int
	for li := lo; li < hi; li++ {
		if d := g.staged[li]; d != nil {
			delete(g.staged, li)
			if g.own[li] != d {
				drop, at = append(drop, d), append(at, li)
			}
		}
	}
	g.mu.Unlock()
	for i, d := range drop {
		d.ReleaseLayers(at[i], at[i]+1)
	}
	if head != nil {
		head.dropHead()
	}
}

// MoveBlock is moveBlock for the zero session.
func (g *GPU) MoveBlock(li int, to string, p *nn.LayerPlan, w *nn.LayerWeights, pos int) (nn.BlockMoved, error) {
	return g.moveBlock(0, li, to, p, w, pos)
}

// moveBlock moves block li of session sid to the named device, its history
// copied device to device; pos is the session's position. Moved false with
// no error is the target refusing the block, which stays where it was;
// errNotDirect is a move the tier cannot make, with nothing changed.
func (g *GPU) moveBlock(sid uint64, li int, to string, p *nn.LayerPlan, w *nn.LayerWeights, pos int) (nn.BlockMoved, error) {
	g.settle()
	var out nn.BlockMoved
	d, err := g.device(to)
	if err != nil {
		return out, err
	}
	g.mu.Lock()
	src, staged := g.own[li], g.staged[li] == d
	g.mu.Unlock()
	if staged {
		d.mu.Lock()
		staged = d.layers[li] != nil
		d.mu.Unlock()
	}
	if !staged && w == nil {
		return out, nn.ErrNotStaged
	}
	if src == nil {
		return out, fmt.Errorf("tier: block %d is on no device", li)
	}
	if src == d {
		out.Moved = true
		return out, nil
	}
	src.mu.Lock()
	shared := src.sharedBeyond(li, sid)
	src.mu.Unlock()
	if shared {
		return out, fmt.Errorf("tier: block %d is on %s and another session is using it, "+
			"so it cannot move to %s", li, label(src.dev, src.ord), to)
	}
	if !staged {
		done, ok := g.admit(p.Model, fmt.Sprintf("block %d", li))
		if !ok {
			return out, nil
		}
		took := false
		for _, mayPage := range [2]bool{false, true} {
			if took = d.prepLayer(sid, li, p, w, mayPage); took {
				break
			}
		}
		done()
		if !took {
			return out, nil
		}
	}
	// Both paged is what a page copy needs; a device keeping the contiguous
	// cache is moved through the State's cache instead.
	if !src.paged || !d.paged {
		d.ReleaseLayers(li, li+1)
		g.mu.Lock()
		delete(g.staged, li)
		g.mu.Unlock()
		return out, errNotDirect
	}
	path, n, err := copyHistory(sid, li, src, d, pos)
	if err != nil {
		// The source still holds the block and its history; the target's copy
		// goes.
		d.ReleaseLayers(li, li+1)
		g.mu.Lock()
		delete(g.staged, li)
		g.mu.Unlock()
		return out, err
	}
	g.mu.Lock()
	g.own[li], g.geo[li] = d, geoOf(p)
	g.split = g.split || p.GeomSplit
	delete(g.staged, li)
	g.mu.Unlock()
	// The source's copy is freed once the caller has moved what it is
	// moving (RetireMoved), outside the pause.
	g.mu.Lock()
	g.retire = append(g.retire, retiree{src, li})
	g.mu.Unlock()
	out.Moved, out.Path, out.Bytes = true, path, n
	return out, nil
}

// copyHistory copies session sid's history on block li from src's pages into
// pages dst allocates for it: the block's own pool and, on a DeepSeek V4
// block, its entries' (entKey). It reports the slowest path any part took and
// the bytes copied. Pages copy raw when both pools page alike; otherwise the
// history is read into the device-independent row layout and written again
// (pagedMigrate), still without the State's cache.
func copyHistory(sid uint64, li int, src, dst *devTier, pos int) (string, uint64, error) {
	sv, dv := src.as(sid), dst.as(sid)
	if sv == nil || dv == nil {
		if sv != nil {
			sv.done()
		}
		if dv != nil {
			dv.done()
		}
		return "", 0, fmt.Errorf("tier: no lane for block %d's history on one of the devices", li)
	}
	defer sv.done()
	defer dv.done()
	// One order for the two locks, whichever way the block moves.
	a, b := sv, dv
	if dst.ord < src.ord {
		a, b = dv, sv
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	b.mu.Lock()
	defer b.mu.Unlock()
	path, total := "peer", uint64(0)
	worse := func(p string) {
		rank := map[string]int{"host": 0, "staged": 1, "peer": 2}
		if rank[p] < rank[path] {
			path = p
		}
	}
	skp, dkp := sv.kvPages(), dv.kvPages()
	s := seqID{sid, 0}
	for _, key := range []int{li, entKey(li)} {
		spl := skp.layers[key]
		if spl == nil || len(spl.owned[s]) == 0 {
			continue
		}
		for o := range spl.owned {
			if o.sid == sid && o.base != 0 {
				return "", 0, errNotDirect // a batch's rows: the State moves them
			}
		}
		dpl := dkp.layers[key]
		if dpl == nil {
			return "", 0, fmt.Errorf("tier: block %d has no pages on %s", li, label(dst.dev, dst.ord))
		}
		n := spl.positions(pos)
		if key == li {
			n = pos
		}
		P := skp.p
		npages := (n + P - 1) / P
		raw := skp.p == dkp.p && spl.geom == dpl.geom && skp.evicted[s] == 0 &&
			len(spl.home[s]) == 0 && len(spl.owned[s]) >= npages && !slices.Contains(spl.owned[s][:npages], 0)
		if raw {
			if err := dv.rewind(sid, dkp, s, n); err != nil {
				return "", 0, err
			}
			if have := len(dpl.owned[s]); npages > have {
				if err := dv.allocPages(sid, dkp, dpl, s, npages-have); err != nil {
					return "", 0, err
				}
			}
			if err := dv.flushTabs(); err != nil {
				return "", 0, err
			}
			raw = dkp.evicted[s] == 0 && len(dpl.owned[s]) >= npages && !slices.Contains(dpl.owned[s][:npages], 0)
		}
		if !raw {
			kv := dpl.geom.kvRow
			k, v := make([]float32, n*kv), make([]float32, n*kv)
			if err := sv.pagedMigrate(sid, key, 0, k, v, n, false); err != nil {
				return "", 0, err
			}
			if err := dv.pagedMigrate(sid, key, 0, k, v, n, true); err != nil {
				return "", 0, err
			}
			worse("host")
			total += uint64(n*kv) * 8
			continue
		}
		kw, vw := spl.geom.kWords(P), spl.geom.vWords(P)
		sids, dids := spl.owned[s][:npages], dpl.owned[s][:npages]
		for _, half := range [2]struct {
			d, s  backend.Buf
			words int
		}{{dpl.k, spl.k, kw}, {dpl.v, spl.v, vw}} {
			if half.words == 0 {
				continue
			}
			p, err := copyPages(dv, half.d, dids, sv, spl, half.s, sids, half.words)
			if err != nil {
				return "", 0, err
			}
			worse(p)
			total += uint64(npages * half.words * 4)
		}
	}
	if total == 0 {
		path = ""
	}
	return path, total, nil
}

// copyPages copies pages sids of src's buffer sb (layer pool spl) into pages
// dids of dst's buffer db, words 32-bit words a page. A backend that copies
// between its own devices does it card to card, one call a run of pages
// consecutive on both sides (backend.PeerCopier: "peer" when the cards
// address each other, "staged" when the driver bounces the bytes through the
// host). Otherwise each group of pages is gathered on the source and read in
// one transfer, and written to the target a run at a time, the read of the
// next group beside the writes of the last ("staged").
func copyPages(dst *devTier, db backend.Buf, dids []uint32, src *devTier, spl *kvLayerPool, sb backend.Buf, sids []uint32, words int) (string, error) {
	n := len(sids)
	if pc, ok := dst.dev.(backend.PeerCopier); ok && dst.dev.API() == src.dev.API() {
		path := "peer"
		for j := 0; j < n; {
			r := 1
			for j+r < n && sids[j+r] == sids[j]+uint32(r) && dids[j+r] == dids[j]+uint32(r) {
				r++
			}
			direct, err := pc.CopyPeer(db, int(dids[j])*words*4, src.dev, sb, int(sids[j])*words*4, r*words*4)
			if err != nil {
				return "", err
			}
			if !direct {
				path = "staged"
			}
			j += r
		}
		return path, nil
	}
	group := max(1, moveChunk/(words*4))
	type part struct {
		at   int
		rows []float32
		err  error
	}
	full := make(chan part, 1)
	go func() {
		defer close(full)
		for at := 0; at < n; at += group {
			rows, err := src.readPages(spl, sb, sids[at:min(n, at+group)], words)
			full <- part{at, rows, err}
			if err != nil {
				return
			}
		}
	}()
	var werr error
	for pt := range full {
		if werr != nil {
			continue
		}
		if werr = pt.err; werr != nil {
			continue
		}
		ids := dids[pt.at : pt.at+len(pt.rows)/words]
		for j := 0; j < len(ids) && werr == nil; {
			r := 1
			for j+r < len(ids) && ids[j+r] == ids[j]+uint32(r) {
				r++
			}
			werr = db.WriteAt(int(ids[j])*words*4, f32b(pt.rows[j*words:(j+r)*words]))
			j += r
		}
	}
	return "staged", werr
}
