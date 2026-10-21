package tier

import (
	"unsafe"

	"github.com/jitllm/jitllm/engine/nn"
)

// A lone matvec's weight is uploaded once and keyed on its address (resKey.p),
// and so is its pack in the arena and a prewarm's stage. The host's pager hands
// one address to block after block: a frame refilled, or memory given back and
// mapped again. Forget is the host saying so, and every copy keyed inside the
// range goes; see docs/engineering-history/placement.md 15v.

var (
	_ nn.CopyForgetter = (*GPU)(nil)
	_ nn.CopyForgetter = (*gpuSession)(nil)
)

// Forget drops every copy the tier keyed on an address in [p, p+n), on every
// device and in the arena. It never waits for a device that is busy: a device
// whose lock is held -- by a submission, or by the very upload whose page-in
// asked -- gets the range queued, and applies it before its next lookup by
// address.
func (g *GPU) Forget(p unsafe.Pointer, n uintptr) {
	lo := uintptr(p)
	hi := lo + n
	for _, d := range g.devs {
		d.forget(lo, hi)
	}
	g.Config.arena.Forget(lo, hi)
}

// Forget forwards to the tier: the copies are the model's, not a session's.
func (s *gpuSession) Forget(p unsafe.Pointer, n uintptr) { s.g.Forget(p, n) }

// forget applies [lo, hi) now when the device is free and queues it when not.
func (g *devTier) forget(lo, hi uintptr) {
	if g.mu.TryLock() {
		defer g.mu.Unlock()
		g.applyForgets()
		g.applyForget(lo, hi)
		return
	}
	g.forgetMu.Lock()
	defer g.forgetMu.Unlock()
	g.forgets = append(g.forgets, [2]uintptr{lo, hi})
	g.forgetPend.Store(true)
}

// applyForgets applies every queued range. It is the first thing every lookup
// by address does. Callers hold g.mu.
func (g *devTier) applyForgets() {
	if !g.forgetPend.Load() {
		return
	}
	g.forgetMu.Lock()
	rs := g.forgets
	g.forgets = nil
	g.forgetPend.Store(false)
	g.forgetMu.Unlock()
	for _, r := range rs {
		g.applyForget(r[0], r[1])
	}
}

// applyForget drops the copies keyed in [lo, hi): uploaded weights, with their
// device memory and its charge, staged packs, and any prewarm pack still
// reading from there. A dropped copy that was the output projection takes the
// head's scratch with it, so PrepHead builds it again. Callers hold g.mu.
func (g *devTier) applyForget(lo, hi uintptr) {
	if g.closed {
		return
	}
	in := func(p *byte) bool {
		at := uintptr(unsafe.Pointer(p))
		return p != nil && at >= lo && at < hi
	}
	for _, pk := range g.packing {
		if pk.at >= lo && pk.at < hi {
			pk.spoiled = true
		}
	}
	head := false
	for k, r := range g.res {
		if !in(k.p) {
			continue
		}
		delete(g.res, k)
		if r == nil || !r.ok {
			continue
		}
		if g.bs != nil && g.bs.head == r {
			g.dropHeadScratch(g.bs)
			head = true
		}
		g.refund(r.bytes())
		g.freeResident(r)
		g.Forgotten++
	}
	for k, st := range g.stage {
		if !in(k.p) {
			continue
		}
		delete(g.stage, k)
		g.stageBytes -= st.bytes()
		g.Config.stage.give(st.bytes())
	}
	// A recording that launched the head names its buffers.
	if head {
		g.dropGraph()
	}
}

// freeResident frees an uploaded tensor's device buffers and lets go of the
// arena entry an imported one aliases. The caller has taken it out of g.res
// and refunded its charge. Callers hold g.mu.
func (g *devTier) freeResident(r *resident) {
	r.qs.Free()
	r.d.Free()
	if r.sc != nil {
		r.sc.Free()
	}
	if r.imported {
		g.arena.Unimport(r.pk)
		r.pk = nil
	}
	r.ok = false
}

// packing is one prewarm pack running outside g.mu: where its source bytes
// start, and whether a forget covered them meanwhile.
type packing struct {
	at      uintptr
	spoiled bool
}

// donePacking takes pk off the in-flight list. Callers hold g.mu.
func (g *devTier) donePacking(pk *packing) {
	for i, p := range g.packing {
		if p == pk {
			g.packing = append(g.packing[:i], g.packing[i+1:]...)
			return
		}
	}
}
