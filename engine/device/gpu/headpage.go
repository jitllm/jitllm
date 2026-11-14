package gpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/nn"
)

// headPage is the output projection where the caller's placement streams it
// (nn.Head.Stream): its planes page through the card's slots as a streamed
// block's weights do -- in for the call that reads them, a victim when a
// page-in wants the room -- behind the one resident every reader of the head
// holds (bs.head, a lane's shared head), which a page-in fills again.
type headPage struct {
	r    *resident
	w    nn.Weight
	f    *jlm.File
	q    layout.Quant
	foot uint64 // the planes' footprint: what a page-in finds room for
	in   bool
	// uses is the submissions reading it: no victim while any does.
	uses int
}

// headSlot is the projection where victim names it beside the blocks: past
// every block index, and not negative, which victim's callers read as none.
const headSlot = 1 << 30

// headPlaced says bs holds the head: resident, or streamed and paged out.
// Callers hold g.mu.
func (g *devTier) headPlaced(bs *blockScratch) bool {
	return bs != nil && bs.head != nil && (bs.head.ok || g.headPg != nil && g.headPg.r == bs.head)
}

// streamHead makes the projection PrepHead placed in bs a page of planes
// bytes: a slot holds it as it passes, so the widest slot is at least it.
// Callers hold g.mu.
func (g *devTier) streamHead(bs *blockScratch, h *nn.Head, q layout.Quant, planes uint64) {
	g.headPg = &headPage{r: bs.head, w: h.W, f: h.File, q: q, foot: planes, in: true}
	g.pageBytes += planes
	g.widest = max(g.widest, planes)
	g.Slots = g.slots()
	g.HeadsStreamed++
}

// unstreamHead forgets the projection's page as bs lets the head go; planes
// paged in are the resident's, freed by whoever frees it. Callers hold g.mu.
func (g *devTier) unstreamHead(bs *blockScratch) {
	hp := g.headPg
	if hp == nil || hp.r != bs.head {
		return
	}
	if hp.in {
		g.pageBytes -= min(g.pageBytes, hp.foot)
	}
	g.headPg = nil
}

// headVictim says victim takes the projection over best, the block whose
// next use is bestD away with the scan at cur in [lo, hi). The head runs
// after the scan's last block, so inside the scan its next use is hi-cur
// away and a tie is block lo's, read after it; outside the scan it is read
// after every block, so it goes first. Callers hold g.mu.
func (g *devTier) headVictim(cur, lo, hi, best, bestD int) bool {
	hp := g.headPg
	if hp == nil || cur == headSlot || !hp.in || hp.uses > 0 || hp.foot == 0 {
		return false
	}
	if lo <= cur && cur < hi {
		return best < 0 || hi-cur > bestD
	}
	return true
}

// pageOutHead frees the streamed projection's planes, leaving the head
// placed. Callers hold g.mu.
func (g *devTier) pageOutHead() uint64 {
	hp := g.headPg
	if hp == nil || !hp.in || hp.uses > 0 {
		return 0
	}
	// A recording names the planes' addresses. See dropGraph.
	g.dropGraph()
	r := hp.r
	for k, v := range g.res {
		if v == r {
			delete(g.res, k)
			break
		}
	}
	n := r.bytes()
	g.refund(n)
	g.freeResident(r)
	hp.in = false
	g.pageBytes -= min(g.pageBytes, hp.foot)
	g.HeadPageOuts++
	return n
}

// pageInHead uploads the streamed projection again into the resident its
// readers hold, read afresh into the card's frames: what the frames held at
// placement is another tensor's by now. Callers hold g.mu.
func (g *devTier) pageInHead(hp *headPage) bool {
	g.dropGraph()
	w, release, err := g.ownWeight(hp.f, hp.w)
	if err != nil {
		g.LastErr = fmt.Sprintf("the output projection: %v", err)
		return false
	}
	defer release()
	key := adhocKey(w.Entry, w.Data, hp.q, w.Rows, w.K)
	r := g.resident(-1, 0, w.Entry, hp.q, w.Data, w.Rows, w.K, w.Packed)
	if r == nil || !r.ok {
		if g.res[key] == r {
			delete(g.res, key)
		}
		return false
	}
	*hp.r = *r
	g.res[key] = hp.r
	hp.in = true
	g.pageBytes += hp.foot
	g.HeadPageIns++
	return true
}

// headRoom pages streamed blocks out until the projection fits, the scan at
// cur, and reports whether it does. wait lets a submission reading the only
// victims finish first; a caller holding pages of its own does not wait, so
// no submission waits holding a page another waits for. Callers hold g.mu.
func (g *devTier) headRoom(hp *headPage, cur, lo, hi int, wait bool) bool {
	for !g.room(hp.foot) && !g.kvCompact(hp.foot, nil) {
		if v := g.victim(cur, lo, hi, true); v >= 0 {
			g.pageOut(v)
			continue
		}
		if !wait || !g.victimHeld(cur, true) {
			return false
		}
		g.idle.Wait()
		if g.headPg != hp || hp.in {
			return true
		}
	}
	return true
}

// headHold pages the streamed projection in for a submission about to read
// it, the scan at cur, and holds it there until headLetGo. True at once
// where the head does not stream. wait is headRoom's. Callers do not hold
// g.mu.
func (g *devTier) headHold(cur, lo, hi int, wait bool) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	hp := g.headPg
	if hp == nil {
		return true
	}
	if !hp.in {
		g.quiesce()
	}
	if !hp.in {
		if !g.headRoom(hp, cur, lo, hi, wait) {
			if wait {
				g.LastErr = fmt.Sprintf("the output projection needs %d bytes and no streamed block "+
					"can be paged out for it", hp.foot)
				g.noteCause(nn.FailRoom)
			}
			return false
		}
		if g.headPg != hp {
			return true
		}
		if !hp.in && !g.pageInHead(hp) {
			return false
		}
	}
	hp.uses++
	return true
}

// headLetGo ends what headHold began, and wakes a page-in waiting for the
// room the projection holds.
func (g *devTier) headLetGo() {
	g.mu.Lock()
	if hp := g.headPg; hp != nil && hp.uses > 0 {
		hp.uses--
		g.idle.Broadcast()
	}
	g.mu.Unlock()
}
