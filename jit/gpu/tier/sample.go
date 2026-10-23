package tier

import (
	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
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

// prepSample readies the sampler for head: the device's kernels for its rows
// and k, and this lane's buffers. False leaves the call to read the logits
// back. Callers hold g.mu, outside any Session.
func (g *devTier) prepSample(head *nn.Head) bool {
	n, k := head.W.Rows, head.SampleK
	if k < 1 || k >= n || k > sampleMaxK || len(head.SampleVals) < k || len(head.SampleIDs) < k ||
		len(head.SampleArgs) < 2 || len(head.SampleArgs) > kernels.SampleArgsWords(sampleMaxHist) ||
		int(head.SampleArgs[0]) != len(head.SampleArgs)-2 {
		return false
	}
	w := g.scratchWin()
	defer w.close()
	if g.samplePenK == nil {
		kk, err := kernels.SamplePenalty(n)
		if err != nil {
			return false
		}
		c, err := g.dev.Compile(kk)
		if err != nil {
			g.LastErr = "the sampler's penalty: " + err.Error()
			return false
		}
		g.samplePenK = c
	}
	if _, ok := g.sampleKs[k]; !ok {
		groups := kernels.SampleSlices(n)
		k1, err := kernels.SampleTopK(n, kernels.SampleSlice, k, false)
		if err != nil {
			return false
		}
		k2, err := kernels.SampleTopK(groups*k, groups*k, k, true)
		if err != nil {
			return false
		}
		c1, err := g.dev.Compile(k1)
		if err != nil {
			g.LastErr = "the sampler's first pass: " + err.Error()
			return false
		}
		c2, err := g.dev.Compile(k2)
		if err != nil {
			c1.Close()
			g.LastErr = "the sampler's second pass: " + err.Error()
			return false
		}
		if g.sampleKs == nil {
			g.sampleKs = map[int]sampleKerns{}
		}
		g.sampleKs[k] = sampleKerns{c1, c2}
	}
	l := g.lane
	if l.sampleArgs != nil {
		return true
	}
	cand := kernels.SampleSlices(n) * sampleMaxK
	sizes := [...]int{4 * kernels.SampleArgsWords(sampleMaxHist), 4 * n, 4 * cand, 4 * cand, 4 * sampleMaxK, 4 * sampleMaxK}
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
	l.sampleArgs, l.samplePen, l.sampleV1, l.sampleI1, l.sampleV, l.sampleI =
		bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], bufs[5]
	return true
}

// launchSample is the sampler's launches over the head's n logits: the
// penalty when it is on (pen), then the two top-k passes.
func (g *devTier) launchSample(lc *launcher, logits backend.Buf, n, k int, pen bool) error {
	ks := g.sampleKs[k]
	groups := kernels.SampleSlices(n)
	src := logits
	if pen {
		if err := lc.launch(g.samplePenK, kernels.SampleGroups(n), kernels.SampleGroup,
			logits, g.sampleArgs, g.samplePen); err != nil {
			return err
		}
		src = g.samplePen
	}
	if err := lc.launch(ks.p1, groups, kernels.SampleGroup,
		src, src, g.sampleV1, g.sampleI1); err != nil {
		return err
	}
	return lc.launch(ks.p2, 1, kernels.SampleGroup, g.sampleV1, g.sampleI1, g.sampleV, g.sampleI)
}

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
