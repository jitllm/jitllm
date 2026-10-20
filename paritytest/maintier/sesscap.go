package maintier

import (
	"reflect"

	"github.com/jitllm/jitllm/engine/nn"
)

// A device serves several sessions over one scratch: their histories are
// paged (pagedattn.go), so no kernel bakes a session's capacity and switching
// sessions is a change of whose pages a call reads, and of the recording.

// moveHead hands the output projection from src to dst. The head is the
// model's, not a capacity's: it rides whichever text scratch is current, and
// src forgets it so that freeing src cannot free the head's buffers.
func moveHead(dst, src *blockScratch) {
	if dst == nil || src == nil || dst == src {
		return
	}
	dst.head, dst.mvHead = src.head, src.mvHead
	dst.hNorm, dst.hNormB, dst.logits, dst.hRaw, dst.hOut = src.hNorm, src.hNormB, src.logits, src.hRaw, src.hOut
	dst.logitsCap, dst.headCap = src.logitsCap, src.headCap
	dst.headEmbeds, dst.embScale = src.headEmbeds, src.embScale
	src.head, src.mvHead = nil, mv{}
	src.hNorm, src.hNormB, src.logits, src.hRaw, src.hOut = nil, nil, nil, nil, nil
	src.logitsCap, src.headCap = nil, nil
}

// sameShape reports that two plans differ at most in capacity, so a scratch
// built for one serves the other once rebuilt at the new capacity -- and the
// scratch built for the old capacity is still right for any session that has
// it.
func sameShape(a, b nn.LayerPlan) bool {
	a.MaxSeq, b.MaxSeq = 0, 0
	return reflect.DeepEqual(a, b)
}

// sharedBeyond reports that a session other than sid holds history on block
// li: the block cannot move between devices or leave the card without taking
// that history with it. Callers hold g.mu.
func (g *devTier) sharedBeyond(li int, sid uint64) bool {
	l := g.layers[li]
	if l == nil {
		return false
	}
	for s := range l.kv {
		if s != sid {
			return true
		}
	}
	for s := range l.rec {
		if s != sid {
			return true
		}
	}
	return false
}

// heldBy is what session sid holds on this device: its history on every
// block.
func (g *devTier) heldBy(sid uint64) uint64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	var n uint64
	for _, l := range g.layers {
		if l == nil {
			continue
		}
		if kvp := l.kv[sid]; kvp != nil {
			n += kvp.bytes
		}
		if l.rec[sid] != nil {
			n += uint64(g.recSeats[sid].rows) * g.recSlotBytes(g.recShared)
		}
	}
	return n + g.pagedHeldBy(sid)
}
