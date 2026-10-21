//go:build amd64

package cpu

import (
	"math"
	"math/rand"
	"testing"
)

// TestEmitDWConvSSEMatchesReference gates the SSE tier's depthwise row at
// filters 1, 3 and 5, strides 1 and 2 and every ragged channel count
// hybWidths names, inputs and output unaligned with guards after them.
func TestEmitDWConvSSEMatchesReference(t *testing.T) {
	em := EmittersFor(TierSSE)
	for _, k := range []int{1, 3, 5} {
		for _, st := range []int{1, 2} {
			for _, ch := range hybWidths {
				s := DWShape{K: k, Stride: st, WP: 4*st + k, Chans: ch, WOut: 5}
				b, err := em.DWConv(s)
				if err != nil {
					t.Fatalf("%+v: %v", s, err)
				}
				c := hybMapSSE(t, s.Key(), b)
				rnd := rand.New(rand.NewSource(int64(k*1000 + st*100 + ch)))
				in, _ := hybGuarded(s.K * s.WP * s.Chans)
				w, _ := hybGuarded(s.K * s.K * s.Chans)
				out, outAll := hybGuarded(s.WOut * s.Chans)
				for i := range in {
					in[i] = float32(rnd.NormFloat64())
				}
				for i := range w {
					w[i] = float32(rnd.NormFloat64() * 0.3)
				}
				c.Call(&Args{Out: &out[0], AScale: &w[0], Q32: &in[0]})
				c.Close()
				for x := 0; x < s.WOut; x++ {
					for ch := 0; ch < s.Chans; ch++ {
						var acc float64
						for ky := 0; ky < s.K; ky++ {
							for kx := 0; kx < s.K; kx++ {
								acc += float64(in[(ky*s.WP+x*s.Stride+kx)*s.Chans+ch]) *
									float64(w[(ky*s.K+kx)*s.Chans+ch])
							}
						}
						if d := math.Abs(float64(out[x*s.Chans+ch]) - acc); d > 1e-4*(1+math.Abs(acc)) {
							t.Fatalf("%+v: out[%d][%d] = %v, want %v", s, x, ch, out[x*s.Chans+ch], acc)
						}
					}
				}
				hybCheckGuard(t, s.Key(), outAll, s.WOut*s.Chans)
			}
		}
	}
}
