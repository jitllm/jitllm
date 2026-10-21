package cpu

import (
	"math"
	"math/rand"
	"testing"
)

// TestAttnPairedIsBitIdentical holds the paired attention kernels to two calls
// of the single-head ones, bit for bit: they share a load and change no
// arithmetic, so an NMSE bar would hide a swapped accumulator, a cursor
// advanced once for two heads, or a reduction reading the other head's chain.
//
// Shapes cover npos 1 and 2 (where a paired reduction can alias), an odd 7,
// real depths, and 1537 to catch an off-by-one past 1536.
func TestAttnPairedIsBitIdentical(t *testing.T) {
	for _, shape := range []struct{ hd, kvStride, npos int }{
		{64, 256, 1}, {64, 256, 2}, {64, 256, 7},
		{64, 256, 512}, {64, 256, 1536}, {64, 256, 1537},
		{256, 256, 33},                       // gemma: one kv head of 256
		{128, 512, 64},                       // a kv head that is not the whole stride
		{8, 8, 3},                            // a head of one vector
		{3, 5, 9}, {12, 20, 7}, {17, 17, 33}, // ragged: the tails must pair too
	} {
		rng := rand.New(rand.NewSource(int64(shape.hd*7919 + shape.npos)))
		f32 := func(n int) []float32 {
			v := make([]float32, n)
			for i := range v {
				v[i] = float32(rng.NormFloat64())
			}
			return v
		}
		q0, q1 := f32(shape.hd), f32(shape.hd)
		kv := f32(shape.npos * shape.kvStride)
		w0, w1 := f32(shape.npos), f32(shape.npos)

		// --- scores: two single-head calls, then one paired call ---
		single := onHost(t)(hostTable().AttnScores(shape.hd, shape.kvStride, KVF32))
		want0, want1 := make([]float32, shape.npos), make([]float32, shape.npos)
		a0 := Args{Out: &want0[0], W: f32bytes(kv), Rows: int64(shape.npos), Q32: &q0[0]}
		single.Call(&a0)
		a1 := Args{Out: &want1[0], W: f32bytes(kv), Rows: int64(shape.npos), Q32: &q1[0]}
		single.Call(&a1)
		single.Close()

		paired, err := Map(emitOrFail(t, func() ([]byte, error) { return hostTable().AttnScores2(shape.hd, shape.kvStride, KVF32) }))
		if err != nil {
			t.Fatal(err)
		}
		got0, got1 := make([]float32, shape.npos), make([]float32, shape.npos)
		ap := Args{Out: &got0[0], W: f32bytes(kv), Rows: int64(shape.npos),
			Q32: &q0[0], Out2: &got1[0], Q2: &q1[0]}
		paired.Call(&ap)
		paired.Close()
		diffBits(t, "scores head0", shape.hd, shape.npos, want0, got0)
		diffBits(t, "scores head1", shape.hd, shape.npos, want1, got1)

		// --- accumulate ---
		singleA, err := Map(emitOrFail(t, func() ([]byte, error) { return hostTable().AttnAcc(shape.hd, shape.kvStride, KVF32) }))
		if err != nil {
			t.Fatal(err)
		}
		aw0, aw1 := make([]float32, shape.hd), make([]float32, shape.hd)
		b0 := Args{Out: &aw0[0], W: f32bytes(kv), AScale: &w0[0], Rows: int64(shape.npos)}
		singleA.Call(&b0)
		b1 := Args{Out: &aw1[0], W: f32bytes(kv), AScale: &w1[0], Rows: int64(shape.npos)}
		singleA.Call(&b1)
		singleA.Close()

		pairedA, err := Map(emitOrFail(t, func() ([]byte, error) { return hostTable().AttnAcc2(shape.hd, shape.kvStride, KVF32) }))
		if err != nil {
			t.Fatal(err)
		}
		ag0, ag1 := make([]float32, shape.hd), make([]float32, shape.hd)
		bp := Args{Out: &ag0[0], W: f32bytes(kv), AScale: &w0[0], Rows: int64(shape.npos),
			Out2: &ag1[0], AScale2: &w1[0]}
		pairedA.Call(&bp)
		pairedA.Close()
		diffBits(t, "acc head0", shape.hd, shape.npos, aw0, ag0)
		diffBits(t, "acc head1", shape.hd, shape.npos, aw1, ag1)
	}
}

func emitOrFail(t *testing.T, f func() ([]byte, error)) []byte {
	t.Helper()
	b, err := f()
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func diffBits(t *testing.T, what string, hd, npos int, want, got []float32) {
	t.Helper()
	for i := range want {
		if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
			t.Fatalf("%s hd=%d npos=%d: element %d is %v, the single-head kernel gives %v "+
				"-- the paired kernel must be bit-identical, not close",
				what, hd, npos, i, got[i], want[i])
		}
	}
}
