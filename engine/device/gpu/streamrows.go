package gpu

import (
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

// streams is how many residual streams this device's blocks carry.
func (d *devTier) streams() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.bs == nil {
		return 1
	}
	return d.bs.p.Streams()
}
