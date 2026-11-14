package gpu

import (
	"slices"

	"github.com/jitllm/jitllm/nn"
)

// streamRows copies a residual of `from` rows into one of `to` rows: the
// first min(to, from) rows of every stream, each stream at its own row count
// (stream-major). With one stream it is a prefix copy.
func streamRows(dst, src []float32, p *nn.LayerPlan, to, from int) {
	if p.Streams() == 1 {
		n := min(len(dst), len(src))
		copy(dst[:n], src[:n])
		return
	}
	d, rows := p.NEmbd, min(to, from)
	for k := 0; k < p.Streams(); k++ {
		copy(dst[k*to*d:(k*to+rows)*d], src[k*from*d:(k*from+rows)*d])
	}
}

// streamCut is rows [off, off+m) of every one of the streams of src, a
// stream-major residual of n rows d wide, as a residual of m rows in the
// device's scratch for it; streamPut writes them back. One stream's rows are
// src's own, uncopied.
func (d *devTier) streamCut(src []float32, streams, w, off, m, n int) []float32 {
	if streams == 1 {
		return src[off*w : (off+m)*w]
	}
	dst := slices.Grow(d.cutX[:0], streams*m*w)[:streams*m*w]
	d.cutX = dst
	for k := range streams {
		copy(dst[k*m*w:(k+1)*m*w], src[(k*n+off)*w:(k*n+off+m)*w])
	}
	return dst
}

func streamPut(dst, cut []float32, streams, w, off, m, n int) {
	if streams == 1 {
		return
	}
	for k := range streams {
		copy(dst[(k*n+off)*w:(k*n+off+m)*w], cut[k*m*w:(k+1)*m*w])
	}
}

// sideRows is a call's per-row inputs as the session staged them for the
// call's n rows: the per-layer inputs, the tokens and the cut rows.
type sideRows struct {
	ple, cut []float32
	toks     []int32
}

// side is the per-row inputs staged for the next call.
func (d *devTier) side() sideRows {
	d.mu.Lock()
	defer d.mu.Unlock()
	return sideRows{ple: d.pleHost, cut: d.cutHost, toks: d.stepToks}
}

// sideAt stages s's rows from off of its n for a call of those rows; off 0
// stages s as it was.
func (d *devTier) sideAt(s sideRows, off, n int) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.pleHost, d.cutHost, d.stepToks = s.ple, s.cut, s.toks
	if n < 1 || off == 0 {
		return
	}
	if len(s.ple)%n == 0 {
		d.pleHost = s.ple[off*(len(s.ple)/n):]
	}
	if len(s.cut)%n == 0 {
		d.cutHost = s.cut[off*(len(s.cut)/n):]
	}
	if len(s.toks) >= n {
		d.stepToks = s.toks[off:]
	}
}

// realEmbed makes a promised gather real now: a cut copies its rows out of
// the promised slice, which the gather would otherwise fill only after. False
// when it could not: nothing wrote them.
func (d *devTier) realEmbed() bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	if pe := d.embPend; pe != nil {
		d.embPend = nil
		return d.materializeLocked(pe)
	}
	return true
}

// streams is how many residual streams this device's blocks carry.
func (d *devTier) streams() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.bs == nil {
		return 1
	}
	return d.bs.p.Streams()
}
