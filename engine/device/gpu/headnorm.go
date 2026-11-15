package gpu

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// The resident head is the model's: one projection per device, and a second
// Instance on this tier finds it placed. A multi-token-prediction block projects
// through the same matrix with a norm of its own, so the head carries more
// than one set of vectors: the one it was built with, and one more per set
// another Instance brings. The projection is never duplicated -- it is the
// large part.

// altVecs is a set of the head's vectors other than the one it was built
// with: the host vectors it was uploaded from, by identity, and their buffers.
type altVecs struct {
	src  []nn.RoleVector
	bufs []roleBuf
	n    uint64
}

// adoptHead answers PrepHead for a head already resident: the same projection
// is accepted, with its vectors uploaded beside the resident ones when they
// differ, and a different projection of the same shape is refused -- reusing
// the resident one would project through another tensor's weights. The same
// projection is the same container entry, wherever the instance offering it
// holds its bytes; one that is no entry is known by its address. Callers hold
// g.mu.
func (g *devTier) adoptHead(h *nn.Head) bool {
	same := h.W.Entry != nil && h.W.Entry == g.headEntry
	if h.W.Entry == nil {
		same = len(h.W.Data) > 0 && &h.W.Data[0] == g.headSrc
	}
	if !same {
		g.LastErr = "the head resident on this device is another projection of the same shape"
		return false
	}
	if _, _, ok := g.headVecsFor(h); ok {
		return true
	}
	spec, _ := headShape(g.bs.headGraph, g.bs.headMix)
	bufs, n, err := g.upHeadVecs(h, spec)
	if err != nil {
		g.LastErr = "the head's second vectors: " + err.Error()
		return false
	}
	g.altVecs = append(g.altVecs, altVecs{src: headSrc(h, spec), bufs: bufs, n: n})
	g.dropGraph()
	return true
}

// headSrc is the vectors of h the head's graph binds (spec), the host slices
// themselves.
func headSrc(h *nn.Head, spec headSpec) []nn.RoleVector {
	var vs []nn.RoleVector
	spec.eachVec(func(b graph.Binding, _ bool) { vs = append(vs, nn.RoleVector{Role: b.Role, V: h.Vector(b.Role)}) })
	return vs
}

// upHeadVecs uploads every vector of h the head's graph binds (spec), each as
// wide as the graph reads it, and charges them: their buffers and their
// bytes.
func (g *devTier) upHeadVecs(h *nn.Head, spec headSpec) ([]roleBuf, uint64, error) {
	var bufs []roleBuf
	var n uint64
	var err error
	spec.eachVec(func(b graph.Binding, _ bool) {
		if err != nil {
			return
		}
		v := h.Vector(b.Role)
		if len(v) != b.K {
			err = fmt.Errorf("the head's %v is %d wide, and its graph reads %d", b.Role, len(v), b.K)
			return
		}
		var buf backend.Buf
		if buf, err = g.dev.Alloc(len(v) * 4); err != nil {
			return
		}
		if err = buf.Write(f32b(v)); err != nil {
			buf.Free()
			return
		}
		bufs = append(bufs, roleBuf{role: b.Role, buf: buf})
		n += uint64(len(v) * 4)
	})
	if err != nil {
		for _, b := range bufs {
			b.buf.Free()
		}
		return nil, 0, err
	}
	g.charge(n)
	return bufs, n, nil
}

// joinHead is PrepHead for a head another instance placed: adoptHead under the
// tier's lock, without the graph drop a fresh head's upload needs. It holds
// the head's mix (nn.Head.Mix) as PrepHead does.
func (g *devTier) joinHead(h *nn.Head) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.headPlaced(g.bs) {
		return false
	}
	g.headGraphs(h)
	g.bs.headMix = h.Mix
	return g.adoptHead(h)
}

// headVecsFor is the vectors a call's head reads and their index for the
// graph key: 0 for the set the head was built with -- the same slices, or the
// same values -- and i+1 for altVecs[i]. false when h brings vectors this
// device was never given (its PrepHead was not asked). Callers hold g.mu.
func (g *devTier) headVecsFor(h *nn.Head) ([]roleBuf, int, bool) {
	if h == nil {
		return g.bs.hVecs, 0, true
	}
	same := true
	for _, hv := range g.headVecHost {
		if !slices.Equal(h.Vector(hv.Role), hv.V) {
			same = false
			break
		}
	}
	if same {
		return g.bs.hVecs, 0, true
	}
	for i, a := range g.altVecs {
		if sameSrc(a.src, h) {
			return a.bufs, i + 1, true
		}
	}
	return nil, 0, false
}

// sameSrc says h's vectors are the host slices src was uploaded from.
func sameSrc(src []nn.RoleVector, h *nn.Head) bool {
	for _, s := range src {
		v := h.Vector(s.Role)
		if len(v) != len(s.V) || len(v) > 0 && &v[0] != &s.V[0] {
			return false
		}
	}
	return true
}

// freeAltVecs gives every second set of vectors back, with the head that
// served them. Callers hold g.mu.
func (g *devTier) freeAltVecs() {
	for _, a := range g.altVecs {
		for _, b := range a.bufs {
			b.buf.Free()
		}
		g.refund(a.n)
	}
	g.altVecs = nil
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
	g.giveBackHead(bs)
	g.dropAllScratch()
	g.dropIfEmpty()
}

// giveBackHead frees the head PrepHead placed: the scratch around it and,
// where no other holder shares it, the projection. Callers hold g.mu.
func (g *devTier) giveBackHead(bs *blockScratch) {
	r := bs.head
	g.dropGraph()
	g.dropHeadScratch(bs)
	if r != nil && r.ok && !r.shared {
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
	g.headEntry, g.headSrc, g.headVecHost = nil, nil, nil
}
