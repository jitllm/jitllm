package tier

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
)

// Submissions in flight.
//
// A session's submission runs outside g.mu (layersOnce), so another session's
// bookkeeping goes on beside it. Most of that bookkeeping touches only what
// its own session reads -- its pages, its table run, its lane -- and needs
// nothing more. What changes something every submission may read -- a buffer
// a recording names, a block's weights, a page another sequence's descriptor
// points at, a recurrent pool's slots -- waits for every submission in flight
// to complete first (quiesce), and holds g.mu while it changes it, so none
// starts until it is done. A submission whose range writes what is the
// device's rather than the session's (a vision tower's set and k/v pair, a
// streamed block's expert cache, a recurrent pool, the rows of several
// sessions) runs alone: it quiesces and keeps g.mu across its Session, which
// is the whole device as it was before sessions ran at once
// (exclusiveRange).
//
// A KV page released while submissions are in flight is fenced with the
// ticket the next submission will take (subClock.seq), and is reused only
// once every submission holding a ticket up to it has completed -- or at once
// when none is in flight -- so a page a running submission might still read
// is never handed to another sequence under it (kvLayerPool.fence).

// subClock is the device's submission tickets: seq is the next one to be
// taken, live how many are in flight and oldest the smallest of them (seq
// when none is).
type subClock struct {
	seq, oldest uint64
	live        int
}

// beginSub takes a ticket for a submission about to run outside g.mu. Its
// caller waited for any quiesce in progress before its prologue and has held
// g.mu since (layersOnce), so none can be: a wait here would let the state
// the prologue read change under the submission. Callers hold g.mu.
func (g *devTier) beginSub() (uint64, error) {
	if g.quiet > 0 {
		return 0, fmt.Errorf("tier: a submission began while a quiesce waited")
	}
	t := g.clk.seq
	g.clk.seq++
	if len(g.tickets) > 0 {
		g.SubsBeside++
	}
	g.tickets = append(g.tickets, t)
	g.clk.live = len(g.tickets)
	g.clk.oldest = g.tickets[0]
	return t, nil
}

// endSub gives ticket t back. Callers hold g.mu.
func (g *devTier) endSub(t uint64) {
	if i := slices.Index(g.tickets, t); i >= 0 {
		g.tickets = slices.Delete(g.tickets, i, i+1)
	}
	g.clk.live = len(g.tickets)
	g.clk.oldest = g.clk.seq
	if len(g.tickets) > 0 {
		g.clk.oldest = slices.Min(g.tickets)
	}
	if len(g.tickets) == 0 {
		g.idle.Broadcast()
	}
}

// runUnlocked runs f as the session's submission with g.mu let go, and takes
// it again however f ends. Callers hold g.mu.
func (g *devTier) runUnlocked(f func(backend.Session)) {
	g.mu.Unlock()
	defer g.mu.Lock()
	g.runSub(f)
}

// exclusiveRange reports that a submission of blocks [lo, hi) in scratch bs
// writes what is the device's rather than its session's, and so runs alone:
// a vision tower's block (the set and the k/v pair are one per device), a
// linear block (the recurrent pools, their "next state" buffer and every
// session's half flips), a streamed block (the expert cache and its staging),
// a DeepSeek V4 block (its compressed history's staging), a step over several
// sessions' rows, or a call streaming evicted history through free pages
// another session's prologue could take. Callers hold g.mu.
func (g *devTier) exclusiveRange(lo, hi int, rag *ragStep, bs *blockScratch) bool {
	if rag != nil && rag.sid != nil {
		return true
	}
	for li := lo; li < hi; li++ {
		l := g.layers[li]
		if l == nil {
			continue
		}
		if l.nonCausal || l.linear || l.withAttn || l.stream != nil || l.ds4 != nil || l.q4e != nil || l.conv != nil {
			return true
		}
	}
	return false
}

// evictedFor reports that a call of session sid -- or, in a step over several
// sessions' rows, of any of them -- reads history with pages at home: it
// streams them through the pool's free pages (pagedstream.go), which another
// session's prologue could take, so it runs alone. Callers hold g.mu.
func (g *devTier) evictedFor(sid uint64, rag *ragStep) bool {
	if g.kvp == nil || len(g.kvp.evicted) == 0 {
		return false
	}
	for s, n := range g.kvp.evicted {
		if n == 0 {
			continue
		}
		if s.sid == sid || rag != nil && slices.Contains(rag.sid, s.sid) {
			return true
		}
	}
	return false
}

// quiesce waits until no submission is in flight, and keeps new ones from
// starting while it waits; the caller then changes what they read before it
// lets g.mu go. It must not be called from inside a submission -- it would
// wait for itself. Callers hold g.mu.
func (g *devTier) quiesce() error {
	if g.inSub {
		return fmt.Errorf("tier: quiesce from inside a submission")
	}
	if g.prologueHeld {
		return fmt.Errorf("tier: quiesce inside a submission's prologue")
	}
	if len(g.tickets) == 0 {
		return nil
	}
	g.quiet++
	for len(g.tickets) > 0 {
		g.idle.Wait()
	}
	if g.quiet--; g.quiet == 0 {
		g.idle.Broadcast()
	}
	return nil
}
