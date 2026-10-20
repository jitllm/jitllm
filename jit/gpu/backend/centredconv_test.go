package backend_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// TestConv3CentredMatchesTheLoop runs LFM2's encoder convolution on every
// device against the loop it is: three taps centred on the row, a neighbour
// read only inside the row's window. The windows are a text call's (every
// row the whole call) and a media prefix's (the prefix's rows the prefix),
// whose last prefix row must not read the first text row; a row count of one
// reads no neighbour at all. Every output starts NaN.
func TestConv3CentredMatchesTheLoop(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	rng := rand.New(rand.NewSource(7))
	for _, d := range devs {
		for _, tc := range []struct{ chans, rows, prefix int }{{8, 1, 0}, {5, 7, 0}, {8, 6, 3}, {33, 9, 1}} {
			x := convRand(rng, tc.rows*tc.chans)
			w := convRand(rng, 3*tc.chans)
			win := make([]float32, 2*tc.rows)
			los, his := make([]int, tc.rows), make([]int, tc.rows)
			for r := 0; r < tc.rows; r++ {
				his[r] = tc.rows
				if r < tc.prefix {
					his[r] = tc.prefix
				}
				win[2*r] = math.Float32frombits(uint32(los[r]))
				win[2*r+1] = math.Float32frombits(uint32(his[r]))
			}
			want := make([]float64, tc.rows*tc.chans)
			for r := 0; r < tc.rows; r++ {
				for c := 0; c < tc.chans; c++ {
					v := float64(w[tc.chans+c]) * float64(x[r*tc.chans+c])
					if r-1 >= los[r] {
						v += float64(w[c]) * float64(x[(r-1)*tc.chans+c])
					}
					if r+1 < his[r] {
						v += float64(w[2*tc.chans+c]) * float64(x[(r+1)*tc.chans+c])
					}
					want[r*tc.chans+c] = v
				}
			}
			k, err := oldkernels.Conv3Centred(tc.chans, tc.rows)
			got := convRun(t, d, k, err, tc.rows*tc.chans, tc.rows*tc.chans, x, w, win)
			convCheck(t, d.Name(), got, want, 1e-5)
		}
	}
}
