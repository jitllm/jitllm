package model

import (
	"math"
	"testing"
)

// towerCbRef is the tower's numerical oracle, from llama.cpp's own graph dump.
// It catches layout bugs that shape and finiteness checks cannot, such as the
// flattened patch order (channel-planar [c][ky][kx], since a GGUF conv2d kernel
// is [kx, ky, c, out]).
//
// `llama-mtmd-debug -p encode --image cb` synthesises the image in memory as raw
// floats, so the comparison has no png decode, resize or normalisation in it.
// The values are its node_395, the projector output:
//
//	llama-mtmd-debug -m SmolVLM-256M-Instruct-Q8_0.gguf \
//	  --mmproj mmproj-SmolVLM-256M-Instruct-Q8_0.gguf --image cb -n 512 -p encode
//
// It prints three leading and three trailing values per axis plus the sum of
// the whole tensor, which is why the table is shaped the way it is.
func towerCbRef() struct {
	rows [6]int
	vals [6][6]float64
	sum  float64
} {
	return struct {
		rows [6]int
		vals [6][6]float64
		sum  float64
	}{
		rows: [6]int{0, 1, 2, 61, 62, 63},
		vals: [6][6]float64{
			{2.9709, -0.2394, 7.9154, -0.0333, 3.9492, -6.2315},
			{2.3362, 1.1205, 4.9934, 1.0887, 3.3559, -5.4497},
			{2.0612, 0.4978, 5.8252, -0.2149, 3.9973, -5.3926},
			{8.0296, 3.1371, 5.3551, 4.4943, 8.4025, -3.6975},
			{10.8150, 1.2275, 8.8473, 2.4137, 12.9995, -7.4657},
			{15.4850, 4.6529, 1.5212, 3.8733, 7.9017, 0.0195},
		},
		sum: 1605.572876,
	}
}

// towerCb is the same checkerboard llama-mtmd-debug's "cb" builds: 1.0 and 0.0
// on all three channels, raw: the debug tool applies no image_mean/image_std
// (its IM2COL node prints 1, 0, 1, not 1, -1, 1).
func towerCb(n int) []float32 {
	px := make([]float32, n*n*3)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			v := 1.0
			if (x+y)%2 == 1 {
				v = 0
			}
			for ch := 0; ch < 3; ch++ {
				px[(y*n+x)*3+ch] = float32(v)
			}
		}
	}
	return px
}

// TestTowerMatchesLlamaCpp holds the SmolVLM tower's projector output on the
// checkerboard to towerCbRef.
func TestTowerMatchesLlamaCpp(t *testing.T) {
	vlm, tw := openTower(t) // skips only with neither the mmproj nor the merged container
	defer vlm.Close()
	c := tw.Cfg
	s := tw.testState()
	defer s.Close()

	out, err := s.Encode(towerCb(c.ImageSz))
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != c.Tokens()*c.ProjDim {
		t.Fatalf("%d floats, want %d", len(out), c.Tokens()*c.ProjDim)
	}

	// The bar is relative RMS: jitllm quantizes activations to int8 per matmul
	// where clip runs f32/f16, which compounds to ~1.5% over twelve blocks,
	// while a permutation moves values by O(1).
	var sse, sy2 float64
	for i, row := range towerCbRef().rows {
		for j := 0; j < 6; j++ {
			col := j
			if j >= 3 {
				col = c.ProjDim - 6 + j
			}
			got, want := float64(out[row*c.ProjDim+col]), towerCbRef().vals[i][j]
			sse += (got - want) * (got - want)
			sy2 += want * want
		}
	}
	if sy2 <= 0 || math.IsNaN(sse) {
		t.Fatalf("degenerate comparison: sse %g sy2 %g", sse, sy2)
	}
	if rel := math.Sqrt(sse / sy2); rel > 0.05 {
		t.Errorf("projector output differs from llama.cpp: relative RMS %.4f over 36 sampled values", rel)
		for i, row := range towerCbRef().rows {
			t.Logf("row %2d: got %8.4f %8.4f %8.4f ... want %8.4f %8.4f %8.4f",
				row, out[row*c.ProjDim], out[row*c.ProjDim+1], out[row*c.ProjDim+2],
				towerCbRef().vals[i][0], towerCbRef().vals[i][1], towerCbRef().vals[i][2])
		}
	}

	// The sum survives any permutation but sees every element, so it
	// complements the samples. It is a near-cancellation, so the bar is
	// percent.
	var sum float64
	for _, v := range out {
		sum += float64(v)
	}
	if d := math.Abs(sum-towerCbRef().sum) / math.Abs(towerCbRef().sum); d > 0.05 {
		t.Errorf("embedding sum %.4f, llama.cpp %.4f (%.1f%% off)", sum, towerCbRef().sum, d*100)
	}
}
