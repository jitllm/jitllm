package tier

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
)

// The resident head is the model's: one projection per device, and a second
// State on this tier finds it placed. A multi-token-prediction block projects
// through the same matrix with its own head norm, so the head carries more
// than one norm: the one it was built with, and one more per norm another
// State brings. The projection is never duplicated -- it is the large part.

// altNorm is a head norm other than the one the head was built with.
type altNorm struct {
	src []float32 // the host slice it was uploaded from, by identity
	buf backend.Buf
}

// adoptHead answers PrepHead for a head already resident: the same projection
// is accepted, with its norm uploaded beside the resident one when it differs,
// and a different projection of the same shape is refused -- reusing the
// resident one would project through another tensor's weights. Callers hold
// g.mu.
func (g *devTier) adoptHead(h *nn.Head) bool {
	if len(h.W.Data) == 0 || &h.W.Data[0] != g.headSrc {
		g.LastErr = "the head resident on this device is another projection of the same shape"
		return false
	}
	if _, _, ok := g.headNormFor(h); ok {
		return true
	}
	// A LayerNorm head carries a bias the alternate does not; no model with a
	// prediction block has one.
	if g.bs.hNormB != nil {
		g.LastErr = "a second norm for a LayerNorm head"
		return false
	}
	if len(h.Norm) != g.bs.p.NEmbd {
		g.LastErr = fmt.Sprintf("a %d-wide head norm for a %d-wide head", len(h.Norm), g.bs.p.NEmbd)
		return false
	}
	b, err := g.dev.Alloc(len(h.Norm) * 4)
	if err != nil {
		g.LastErr = "the head's second norm: " + err.Error()
		return false
	}
	if err := b.Write(f32b(h.Norm)); err != nil {
		b.Free()
		g.LastErr = "the head's second norm: " + err.Error()
		return false
	}
	g.charge(uint64(len(h.Norm) * 4))
	g.altNorms = append(g.altNorms, altNorm{src: h.Norm, buf: b})
	g.dropGraph()
	return true
}

// joinHead is PrepHead for a head another State placed: adoptHead under the
// tier's lock, without the graph drop a fresh head's upload needs.
func (g *devTier) joinHead(h *nn.Head) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.bs == nil || g.bs.head == nil || !g.bs.head.ok {
		return false
	}
	return g.adoptHead(h)
}

// headNormFor is the norm buffer a call's head reads and its index for the
// graph key: 0 for the norm the head was built with -- the same slice, or the
// same values -- and i+1 for altNorms[i]. false when h brings a norm this
// device was never given (its PrepHead was not asked). Callers hold g.mu.
func (g *devTier) headNormFor(h *nn.Head) (backend.Buf, int, bool) {
	if h == nil || len(h.Norm) == 0 || slices.Equal(h.Norm, g.headNormHost) {
		return g.bs.hNorm, 0, true
	}
	for i, a := range g.altNorms {
		if len(a.src) > 0 && &a.src[0] == &h.Norm[0] {
			return a.buf, i + 1, true
		}
	}
	return nil, 0, false
}

// freeAltNorms gives every second norm back, with the head that served them.
// Callers hold g.mu.
func (g *devTier) freeAltNorms() {
	for _, a := range g.altNorms {
		a.buf.Free()
		g.refund(uint64(len(a.src) * 4))
	}
	g.altNorms = nil
}

// dropHead gives back the output projection a device holds once it holds no
// block: the projection, its norm and logits, and the scratch they ran in.
// A head that moved to another device left this copy behind with nothing to
// run it -- the model's largest single tensor, charged to a card that serves
// none of the model -- so the move frees it (GPU.PrepHeadOn). A device that
// still holds a block keeps its head: the scratch is that block's too.
func (g *devTier) dropHead() {
	g.mu.Lock()
	defer g.mu.Unlock()
	bs := g.bs
	if bs == nil || bs.head == nil || len(g.layers) > 0 {
		return
	}
	r := bs.head
	g.dropGraph()
	g.dropHeadScratch(bs)
	if r.ok && !r.shared {
		for k, v := range g.res {
			if v == r {
				delete(g.res, k)
				break
			}
		}
		if n := r.bytes(); n > 0 {
			g.refund(n)
		}
		g.freeResident(r)
	}
	g.headSrc, g.headNormHost = nil, nil
	g.dropAllScratch()
}
