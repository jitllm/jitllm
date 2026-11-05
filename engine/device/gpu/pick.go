package gpu

import (
	"fmt"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// The label pick (nn.Head.Pick): after the head, the logits at a handful of
// vocabulary ids are gathered on the device by the composer's form and those
// alone are read back in place of the vocabulary's. A decision
// model's readout reads a question's label ids (model.Decider): a few
// hundred bytes where the row is the vocabulary's 65536-151936 floats.

// pickMax is the most ids a pick reads. It sizes the lane's buffers once, so a
// recording, which names them, never outlives a regrown one. It is this
// tier's choice, not a device limit: a longer pick reads the row back and the
// caller gathers there.
const pickMax = nn.PickMax

// prepPick readies the pick for head where the composer picks (composed):
// this lane's buffers. False leaves the call to read the logits back.
// Callers hold g.mu, outside any Session.
func (g *devTier) prepPick(head *nn.Head, composed bool) bool {
	if k := len(head.Pick); !composed || k < 1 || k > pickMax || len(head.PickVals) < k {
		return false
	}
	w := g.scratchWin()
	defer w.close()
	return g.pickBufs()
}

// pickBufs makes this lane's pick buffers: the ids and the picked logits.
func (g *devTier) pickBufs() bool {
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

// runPick is the composer's label pick over bs's logits into the lane's.
func (g *devTier) runPick(lc *launcher, bs *blockScratch) error {
	sv, err := g.servedSample(bs.pickGraph)
	if err != nil {
		return err
	}
	return sv.run(g, lc, sv.inst, []backend.Buf{bs.headOut(), g.pickIDs}, []backend.Buf{g.pickOut})
}

// runPickHead is the composer's head projecting the picked rows of in alone
// into the lane's picked logits: the step's head and its pick in one form.
func (g *devTier) runPickHead(lc *launcher, bs *blockScratch, norm, in backend.Buf) error {
	sv, err := g.servedPickHead(bs, norm)
	if err != nil {
		return err
	}
	if err := sv.run(g, lc, sv.inst, []backend.Buf{in, g.pickIDs}, []backend.Buf{g.pickOut}); err != nil {
		return err
	}
	g.Steps.Serve(nn.PartSample)
	g.PickRows++
	return nil
}

// pickRowsExact says the composer's head over the picked rows is the head's
// own arithmetic at them, so a pick is the row's value to the bit: a Q8_0
// head, whose row-selected matvec is the plain matvec's form
// (decomposeMatVecQ8). Any other format reads the rows by expert matvec's
// decode, a different order of the same sum, and picks by the gather.
func (bs *blockScratch) pickRowsExact() bool {
	if bs.head == nil {
		return false
	}
	t, ok := jlmType(bs.head.t)
	return ok && t == jlm.TypeQ8
}
