package tier

import (
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
)

// The device sampler (nn.Head.SampleK): after the head, the repeat penalty
// and the two top-k passes run in the token's submission and k candidates
// come home in place of the vocabulary's logits. The softmax over them, the
// cuts and the draw are the host sampler's (model.Sampler), on the same
// candidates it would have chosen from the whole row.

// sampleMaxK is the largest k the device serves and sampleMaxHist the longest
// history. They size the lane's buffers once, so a recording, which names
// those buffers, never outlives a regrown one. They are this tier's choice,
// not a device limit: a larger k or history reads the row back and the host
// sampler selects from it, as it does with no top-k at all.
const (
	sampleMaxK    = 256
	sampleMaxHist = 256
)

// sampleKerns is one k's two top-k passes.
type sampleKerns struct{ p1, p2 backend.Kernel }

// freeSample gives back the lane's sampler buffers.
func (l *lane) freeSample() {
	for _, b := range [...]*backend.Buf{&l.sampleArgs, &l.samplePen, &l.sampleV1, &l.sampleI1, &l.sampleV, &l.sampleI} {
		if *b != nil {
			(*b).Free()
			*b = nil
		}
	}
}

// closeSample closes the device's sampler kernels: they are built for a
// head's rows, so they go with the model.
func (g *devTier) closeSample() {
	if g.samplePenK != nil {
		g.samplePenK.Close()
		g.samplePenK = nil
	}
	for k, ks := range g.sampleKs {
		ks.p1.Close()
		ks.p2.Close()
		delete(g.sampleKs, k)
	}
}
