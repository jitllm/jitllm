package tier

import (
	"encoding/binary"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// The device sampler (nn.Head.SampleK): after the head, the composed
// sampler (graph.SampleTopK) applies the repeat penalty and selects k
// candidates in the token's submission, and they are read back in place of
// the vocabulary's logits. The softmax over them, the cuts and the draw are
// the host sampler's (model.Sampler), on the same candidates it would have
// chosen from the whole row.

// sampleMaxK is the largest k the device serves and sampleMaxHist the longest
// history. They size the lane's buffers once, so a recording, which names
// those buffers, never outlives a regrown one. They are this tier's choice,
// not a device limit: a larger k or history reads the row back and the host
// sampler selects from it, as it does with no top-k at all.
const (
	sampleMaxK    = nn.SampleMaxK
	sampleMaxHist = nn.SampleMaxHist
)

// prepSample readies this lane's buffers for head's sampler, false where its
// k or history is past the bounds: the call reads the logits back. Callers
// hold g.mu, outside any Session.
func (g *devTier) prepSample(head *nn.Head) bool {
	n, k := head.W.Rows, head.SampleK
	if k < 1 || k >= n || k > sampleMaxK || len(head.SampleVals) < k || len(head.SampleIDs) < k ||
		len(head.SampleArgs) < 2 || len(head.SampleArgs) > kernels.SampleArgsWords(sampleMaxHist) ||
		int(head.SampleArgs[0]) != len(head.SampleArgs)-2 {
		return false
	}
	w := g.scratchWin()
	defer w.close()
	return g.sampleBufs()
}

// sampleBufs makes this lane's sampler buffers, sized for the bounds.
func (g *devTier) sampleBufs() bool {
	l := g.lane
	if l.sampleArgs != nil {
		return true
	}
	sizes := [...]int{4 * kernels.SampleArgsWords(sampleMaxHist), 4 * sampleMaxK, 4 * sampleMaxK, 4}
	var bufs [len(sizes)]backend.Buf
	for i, sz := range sizes {
		b, err := g.dev.Alloc(sz)
		if err != nil {
			for _, f := range bufs[:i] {
				f.Free()
			}
			g.LastErr = "the sampler's buffers: " + err.Error()
			return false
		}
		bufs[i] = b
	}
	l.sampleArgs, l.sampleV, l.sampleI, l.samplePenalty = bufs[0], bufs[1], bufs[2], bufs[3]
	l.sampleStage = make([]byte, 4*kernels.SampleArgsWords(sampleMaxHist))
	return true
}

// stageSample is head's argument words as the composed sampler reads them:
// the count, the penalty's bits and the history, an id no row has past it.
func (l *lane) stageSample(head *nn.Head) []byte {
	b := l.sampleStage
	for i := range len(b) / 4 {
		w := uint32(0xffffffff)
		if i < len(head.SampleArgs) {
			w = head.SampleArgs[i]
		}
		binary.LittleEndian.PutUint32(b[4*i:], w)
	}
	return b
}

// freeSample gives back the lane's sampler buffers.
func (l *lane) freeSample() {
	for _, b := range [...]*backend.Buf{&l.sampleArgs, &l.sampleV, &l.sampleI, &l.samplePenalty} {
		if *b != nil {
			(*b).Free()
			*b = nil
		}
	}
}
