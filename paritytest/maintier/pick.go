package maintier

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	kernels "github.com/jitllm/jitllm/paritytest/mainkernels"
)

// The label pick (nn.Head.Pick): after the head, the logits at a handful of
// vocabulary ids are gathered on the device (kernels.GatherRows, rows of one
// float) and those alone come home in place of the vocabulary's. A decision
// model's readout reads a question's label ids (model.Decider): a few
// hundred bytes where the row is the vocabulary's 65536-151936 floats.

// pickMax is the most ids a pick reads. It sizes the lane's buffers once, so a
// recording, which names them, never outlives a regrown one. It is this
// tier's choice, not a device limit: a longer pick reads the row back and the
// caller gathers there. Lev's codes are at most 255; d1's groups two a code.
const pickMax = 512

// prepPick readies the pick for head: the device's gather over its rows and
// this lane's buffers. False leaves the call to read the logits back.
// Callers hold g.mu, outside any Session.
func (g *devTier) prepPick(head *nn.Head) bool {
	n, k := head.W.Rows, len(head.Pick)
	if k < 1 || k > pickMax || len(head.PickVals) < k {
		return false
	}
	w := g.scratchWin()
	defer w.close()
	if g.pickK == nil {
		kk, err := kernels.GatherRows(1, pickMax, n)
		if err != nil {
			g.LastErr = "the label pick: " + err.Error()
			return false
		}
		c, err := g.dev.Compile(kk)
		if err != nil {
			g.LastErr = "the label pick: " + err.Error()
			return false
		}
		g.pickK = c
	}
	// The picked rows' own projection, compiled once for the head here:
	// outside any session, where Compile is allowed. A float head keeps the
	// whole matvec and the gather.
	if hr := g.headRes(); hr != nil && g.pickMV == nil && !g.NoPickMV && !kernels.IsFloat(hr.t) {
		if c, err := g.idKernel(kernels.MatVecShape{Center: g.center, T: hr.t, K: hr.k, Rows: pickMax,
			Split: 1, PickOf: hr.nrows}); err == nil {
			g.pickMV = c
		}
	}
	l := g.lane
	if l.pickIDs != nil {
		return true
	}
	var bufs [2]backend.Buf
	for i := range bufs {
		b, err := g.dev.Alloc(4 * pickMax)
		if err != nil {
			if i > 0 {
				bufs[0].Free()
			}
			g.LastErr = fmt.Sprintf("the label pick's buffers: %v", err)
			return false
		}
		bufs[i] = b
	}
	l.pickIDs, l.pickOut = bufs[0], bufs[1]
	return true
}

// pickStage is the pick's ids as the gather reads them: every slot past the
// pick's an id past the vocabulary, which the gather writes as zero.
func (g *devTier) pickStage(head *nn.Head) []byte {
	if cap(g.pickHost) < 4*pickMax {
		g.pickHost = make([]byte, 4*pickMax)
	}
	b := g.pickHost[:4*pickMax]
	n := uint32(head.W.Rows)
	for i := range pickMax {
		id := n
		if i < len(head.Pick) && head.Pick[i] >= 0 {
			id = uint32(head.Pick[i])
		}
		b[4*i], b[4*i+1], b[4*i+2], b[4*i+3] = byte(id), byte(id>>8), byte(id>>16), byte(id>>24)
	}
	return b
}

// freePick gives back the lane's pick buffers.
func (l *lane) freePick() {
	for _, b := range [...]*backend.Buf{&l.pickIDs, &l.pickOut} {
		if *b != nil {
			(*b).Free()
			*b = nil
		}
	}
}

// closePick closes the device's gather: it is built for a head's rows, so it
// goes with the model.
func (g *devTier) closePick() {
	if g.pickK != nil {
		g.pickK.Close()
		g.pickK = nil
	}
}

// headRes is the head's resident weights, nil while no head is placed.
func (g *devTier) headRes() *resident {
	if g.bs == nil {
		return nil
	}
	return g.bs.head
}
