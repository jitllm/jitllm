package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// int8Tiles are the GemmInt8 blockings TestMatVecMMAMatchesDot4 sweeps on
// every shape a tile divides: 2x2 and 1x4 warp grids, one and several
// m-tiles and n-tiles, a 32-row block whose staging items are fewer than its
// threads.
var int8Tiles = []mainkernels.Int8Tile{
	{MT: 4, NT: 4, WM: 2, WN: 2}, {MT: 2, NT: 2, WM: 2, WN: 2}, {MT: 1, NT: 1, WM: 2, WN: 2},
	{MT: 4, NT: 2, WM: 1, WN: 4}, {MT: 1, NT: 2, WM: 4, WN: 1},
	// The deep stage (MMQ's shape): eight warps, 64 and 128 elements of k a
	// trip, one buffer and two; and sm_75's m8n8k16 at one step and four.
	{MT: 2, NT: 4, WM: 4, WN: 2, D: 2}, {MT: 4, NT: 2, WM: 2, WN: 4, D: 4, Single: true},
	{MT: 2, NT: 2, WM: 2, WN: 2, M8: true}, {MT: 4, NT: 4, WM: 2, WN: 2, D: 4, Single: true, M8: true},
	{MT: 4, NT: 2, WM: 2, WN: 4, D: 4, Single: true, M8: true},
	// ldmatrix weight fragments, on both instructions.
	{MT: 2, NT: 2, WM: 2, WN: 2, LdM: true}, {MT: 2, NT: 4, WM: 4, WN: 2, D: 4, Single: true, LdM: true},
	{MT: 2, NT: 2, WM: 2, WN: 2, M8: true, LdM: true}, {MT: 4, NT: 2, WM: 2, WN: 4, D: 4, Single: true, M8: true, LdM: true},
}

// int8M8Tiles are int8Tiles' m8n8k16 forms: all a device with only sm_75's
// integer instruction runs.
var int8M8Tiles = func() (out []mainkernels.Int8Tile) {
	for _, tl := range int8Tiles {
		if tl.M8 {
			out = append(out, tl)
		}
	}
	return
}()

// gemmInt8Case runs GemmInt8 on the fixture run's buffers for every tile in
// int8Tiles that divides the shape, unsplit and split in two, and holds it
// to want, the dp4a arm's output: bit for bit unsplit (the fold is the
// same), and to NMSE 1e-10 split, where Reduce adds the partials in another
// order. run launches a kernel into a fresh poisoned output of n floats per
// split slice and returns the bytes. A format GemmInt8 has no decode for is
// refused by name.
func gemmInt8Case(t *testing.T, d backend.Device, q mainkernels.Quant, nrows, k, ntok int, want []byte, tiles []mainkernels.Int8Tile,
	run func(kern backend.Kernel, groups, w, n int) []byte) {
	t.Helper()
	if !mainkernels.Int8OK(q) {
		if _, err := mainkernels.GemmInt8(mainkernels.MatVecShape{T: q, K: k, Rows: nrows, NTok: ntok},
			mainkernels.Int8Tile{MT: 1, NT: 1, WM: 1, WN: 1}); err == nil {
			t.Fatalf("GemmInt8 built %v, which it has no decode for", q)
		}
		return
	}
	ran := 0
	for _, tl := range tiles {
		for _, split := range []int{1, 2} {
			s := mainkernels.MatVecShape{T: q, K: k, Rows: nrows, NTok: ntok, Split: split}
			kk, err := mainkernels.GemmInt8(s, tl)
			if err != nil {
				if strings.Contains(err.Error(), "does not divide") || strings.Contains(err.Error(), "do not tile") ||
					strings.Contains(err.Error(), "shared memory") {
					continue
				}
				t.Fatalf("%+v split %d: %v", tl, split, err)
			}
			kern, err := d.Compile(kk)
			if err != nil {
				if d.API() == "ptx" {
					t.Fatalf("%s %+v: %v", d.Name(), tl, err)
				}
				t.Logf("%s: %v", d.API(), err)
				return
			}
			n := nrows * ntok
			got := run(kern, mainkernels.GemmInt8Groups(s, tl), tl.Threads(), n*split)
			kern.Close()
			ran++
			bad := 0
			var worst, sse, sy2 float64
			for i := 0; i < n; i++ {
				var g float64
				for sg := 0; sg < split; sg++ {
					g += float64(math.Float32frombits(binary.LittleEndian.Uint32(got[4*(sg*n+i):])))
				}
				w := float64(math.Float32frombits(binary.LittleEndian.Uint32(want[4*i:])))
				if math.IsNaN(g) || math.IsInf(g, 0) {
					t.Fatalf("%+v split %d: out[tok %d][row %d] is %v", tl, split, i/nrows, i%nrows, g)
				}
				e := g - w
				sse, sy2 = sse+e*e, sy2+w*w
				if split == 1 && float32(g) != float32(w) {
					if bad < 5 {
						t.Errorf("%+v: out[tok %d][row %d]: gemm %v, dp4a %v", tl, i/nrows, i%nrows, g, w)
					}
					bad++
					worst = math.Max(worst, math.Abs(e))
				}
			}
			name := fmt.Sprintf("%+v split %d", tl, split)
			if bad > 0 {
				t.Fatalf("%s: %d/%d elements differ, worst |delta| %g -- this path must be bit-identical", name, bad, n, worst)
			}
			if nmse := sse / sy2; split > 1 && (nmse > 1e-10 || math.IsNaN(nmse)) {
				t.Fatalf("%s: NMSE %.3e against the dp4a arm", name, nmse)
			}
		}
	}
	t.Logf("GemmInt8: %d tile and split forms held", ran)
}
