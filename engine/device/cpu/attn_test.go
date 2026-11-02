//go:build amd64 || arm64

package cpu

import (
	"math"
	"math/rand"
	"testing"
)

// The attention kernels are checked against a float64 evaluation of the same
// sums, at every head dimension and KV stride the supported models use.
//
// The tolerance is float32's, not float64's: the kernel accumulates in float32
// across four chains and the reference in float64 in one, so the difference is
// summation order in a narrower type. A real indexing bug (the wrong row
// stride, the wrong lane) lands orders of magnitude above this.
func TestAttnKernels(t *testing.T) {
	for _, shape := range []struct{ hd, kvStride, npos int }{
		{64, 256, 1}, {64, 256, 7}, {64, 256, 821}, // tinyllama: 4 kv heads x 64
		{256, 256, 1}, {256, 256, 33}, {256, 256, 500}, // gemma: 1 kv head x 256
		{128, 512, 64}, {8, 8, 3},
		// Ragged head dimensions: all tail (1, 3, 4), a vector and a tail on
		// one host or both (12, 17, 33), and a stride that is not the head.
		{1, 1, 5}, {3, 5, 9}, {4, 4, 1}, {12, 20, 7}, {17, 17, 33}, {33, 40, 100},
	} {
		rng := rand.New(rand.NewSource(int64(shape.hd*7919 + shape.npos)))
		q := make([]float32, shape.hd)
		for i := range q {
			q[i] = float32(rng.NormFloat64())
		}
		kv := make([]float32, shape.npos*shape.kvStride)
		for i := range kv {
			kv[i] = float32(rng.NormFloat64())
		}
		att := make([]float32, shape.npos)
		for i := range att {
			att[i] = float32(rng.Float64())
		}

		// scores
		kern := onHost(t)(hostTable().AttnScores(shape.hd, shape.kvStride, KVF32))
		got := sentinelRow(shape.npos)
		args := Args{Out: &got[0], W: (*byte)(nil), Rows: int64(shape.npos), Q32: &q[0]}
		args.W = f32bytes(kv)
		kern.Call(&args)
		kern.Close()
		for i := shape.npos; i < len(got); i++ {
			if math.Float32bits(got[i]) != softmaxSentinel {
				t.Fatalf("scores hd=%d: wrote score %d past npos", shape.hd, i)
			}
		}

		var sse, sy2 float64
		for tt := 0; tt < shape.npos; tt++ {
			var want float64
			for i := 0; i < shape.hd; i++ {
				want += float64(q[i]) * float64(kv[tt*shape.kvStride+i])
			}
			d := float64(got[tt]) - want
			sse += d * d
			sy2 += want * want
		}
		if nmse := sse / sy2; nmse > 1e-12 || math.IsNaN(nmse) {
			t.Errorf("scores hd=%d stride=%d npos=%d: NMSE %.3e", shape.hd, shape.kvStride, shape.npos, nmse)
		}

		// weighted accumulation
		kern = onHost(t)(hostTable().AttnAcc(shape.hd, shape.kvStride, KVF32))
		out := sentinelRow(shape.hd)
		args = Args{Out: &out[0], W: f32bytes(kv), AScale: &att[0], Rows: int64(shape.npos)}
		kern.Call(&args)
		kern.Close()
		for i := shape.hd; i < len(out); i++ {
			if math.Float32bits(out[i]) != softmaxSentinel {
				t.Fatalf("acc hd=%d: wrote dimension %d past the head", shape.hd, i)
			}
		}

		sse, sy2 = 0, 0
		for i := 0; i < shape.hd; i++ {
			var want float64
			for tt := 0; tt < shape.npos; tt++ {
				want += float64(att[tt]) * float64(kv[tt*shape.kvStride+i])
			}
			d := float64(out[i]) - want
			sse += d * d
			sy2 += want * want
		}
		accBound := 1e-12
		if HostTier() == TierSSE {
			// No FMA on this tier: each position is out + float32(v*w), a
			// product rounded before the add, and the kernel is held to that
			// to the bit (attnAxpyModelSSE's arithmetic) before the float64
			// bound. At hd=1 the NMSE is one element's relative error, nothing
			// averages it, and a cancelling sum on these inputs reaches 1.3e-12.
			model := make([]float32, shape.hd)
			for tt := 0; tt < shape.npos; tt++ {
				for i := range model {
					model[i] = model[i] + float32(kv[tt*shape.kvStride+i]*att[tt])
				}
			}
			for i := range model {
				if math.Float32bits(out[i]) != math.Float32bits(model[i]) {
					t.Fatalf("acc hd=%d stride=%d npos=%d: dimension %d is %v, the unfused accumulate gives %v",
						shape.hd, shape.kvStride, shape.npos, i, out[i], model[i])
				}
			}
			accBound = 1e-10
		}
		if nmse := sse / sy2; nmse > accBound || math.IsNaN(nmse) {
			t.Errorf("acc hd=%d stride=%d npos=%d: NMSE %.3e", shape.hd, shape.kvStride, shape.npos, nmse)
		}
	}
}
