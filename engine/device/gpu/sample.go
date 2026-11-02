package gpu

import (
	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
)

// The device sampler (nn.Head.SampleK): after the head, the composed walk
// (graph.SampleWalk) draws a top-k draw's token in the token's submission
// -- the penalty, the k best, the cuts and the walk at the caller's uniform
// -- and the token is read back in place of the vocabulary's logits.

// sampleMaxK is the largest k the device serves. It sizes nothing a
// recording names; a larger k reads the row back and the CPU draws.
const sampleMaxK = nn.SampleMaxK

// prepSample readies this lane's buffers for head's sampler, false where its
// k is past the bound or its window is not a row over the vocabulary: the
// call reads the logits back. Callers hold g.mu, outside any Session.
func (g *devTier) prepSample(head *nn.Head) bool {
	n, k := head.W.Rows, head.SampleK
	if k < 1 || k >= n || k > sampleMaxK || head.SamplePenalty != 1 && len(head.SampleWindow) != n {
		return false
	}
	w := g.scratchWin()
	defer w.close()
	return g.sampleBufs(n)
}

// sampleBufs makes this lane's sampler buffers for a vocabulary of n tokens,
// false where they are another vocabulary's.
func (g *devTier) sampleBufs(n int) bool {
	l := g.lane
	if l.sampleWindow != nil {
		return l.sampleN == n
	}
	sizes := [...]int{4 * n, 4, 4 * len(nn.Head{}.WalkArgs)}
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
	l.sampleWindow, l.samplePenalty, l.sampleWalk, l.sampleN = bufs[0], bufs[1], bufs[2], n
	return true
}

// freeSample gives back the lane's sampler buffers.
func (l *lane) freeSample() {
	for _, b := range [...]*backend.Buf{&l.sampleWindow, &l.samplePenalty, &l.sampleWalk} {
		if *b != nil {
			(*b).Free()
			*b = nil
		}
	}
}
