package cpu

import (
	"math"
	"testing"
	"unsafe"
)

// TestAttnScoresTiledMatchesReference runs the qt-wide score kernel and compares
// every element against a plain Go dot product.
//
// It is not bit-compared against EmitAttnScores: the single-query kernel runs
// four accumulator chains, the tiled one one chain per query (qt chains already
// cover the FMA latency), so the last bits differ.
//
// The geometry is the tower's: hd=64, a 768-wide residual so kvStride and
// qStride are both 768, and a padded score stride.
func TestAttnScoresTiledMatchesReference(t *testing.T) {
	for _, hd := range []int{64, 17, 12} { // 17 and 12: the tail, per query
		attnTiledCase(t, hd)
	}
}

func attnTiledCase(t *testing.T, hd int) {
	const (
		stride      = 768 // kv and q rows both step by the residual width
		scoreStride = 136 // deliberately not a round number
		npos        = 37  // and not a multiple of anything
	)
	for _, qt := range []int{2, 4, 8} {
		code := onHost(t)(hostTable().AttnScoresTiled(hd, stride, stride, scoreStride, qt, KVF32))
		defer code.Close()

		k := make([]float32, npos*stride)
		q := make([]float32, qt*stride)
		for i := range k {
			k[i] = float32((i%71)-35) / 16
		}
		for i := range q {
			q[i] = float32((i%53)-26) / 8
		}
		scores := make([]float32, qt*scoreStride)
		args := Args{
			Out: &scores[0], W: (*byte)(unsafe.Pointer(&k[0])),
			Rows: int64(npos), Q32: &q[0],
		}
		code.Call(&args)

		worst := 0.0
		for j := 0; j < qt; j++ {
			for p := 0; p < npos; p++ {
				var want float64
				for d := 0; d < hd; d++ {
					want += float64(q[j*stride+d]) * float64(k[p*stride+d])
				}
				got := float64(scores[j*scoreStride+p])
				if d := math.Abs(got - want); d/(math.Abs(want)+1e-9) > worst {
					worst = d / (math.Abs(want) + 1e-9)
				}
			}
		}
		if worst > 1e-5 {
			t.Errorf("hd=%d qt=%d: worst relative error %.3e, want <= 1e-5", hd, qt, worst)
		}
		// Nothing outside the npos prefix of each row may be touched: the tower
		// pads its score rows and the softmax fills the tail itself.
		for j := 0; j < qt; j++ {
			for p := npos; p < scoreStride; p++ {
				if scores[j*scoreStride+p] != 0 {
					t.Fatalf("hd=%d qt=%d: wrote past npos at row %d col %d", hd, qt, j, p)
				}
			}
		}
		t.Logf("hd=%d qt=%d: worst relative error %.3e over %d rows x %d keys", hd, qt, worst, qt, npos)
	}
}
