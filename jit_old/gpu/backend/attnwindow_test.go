package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestAttnScoresWindowMasksExactlyTheOldKeys holds every windowed scores kernel
// to its unwindowed twin on real hardware: bit-identical at every key the window
// keeps, -inf at every key before it.
//
// Equality, not a bound, because the window is a mask and nothing else: the
// two kernels run the same dot products in the same order, so any difference
// in a kept score is the mask reaching where it should not.
func TestAttnScoresWindowMasksExactlyTheOldKeys(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
	}
	mma := map[backend.Device]bool{}
	for _, d := range mmaDevices(t, devs, ir.MMAShape{M: 16, N: 8, K: 16, Kind: ir.MMAF16}) {
		mma[d] = true
	}
	const nHeads, headDim, nKV, W = 8, 64, 2, 16
	for _, d := range devs {
		t.Run(d.API(), func(t *testing.T) {
			for _, c := range []struct {
				kind      string
				rows, cap int
			}{
				{"decode", 1, 10}, // inside the window: nothing masked
				{"decode", 1, 40}, // past it: the first 24 keys
				{"tiled", 16, 64}, // every row past it
				{"tiled", 16, 20}, // rows straddling it
				{"mma", 16, 64},
				{"mma", 16, 20},
			} {
				if c.kind == "mma" && !mma[d] {
					continue
				}
				t.Run(fmt.Sprintf("%s/rows%d/cap%d", c.kind, c.rows, c.cap), func(t *testing.T) {
					windowCase(t, d, c.kind, nHeads, headDim, nKV, W, c.rows, c.cap)
				})
			}
		})
	}
}

func windowCase(t *testing.T, d backend.Device, kind string, nHeads, headDim, nKV, W, rows, cap int) {
	t.Helper()
	const maxSeq = 256
	kvDim, gqa := nKV*headDim, nHeads/nKV
	sstride, kStride := maxSeq+8, maxSeq+1
	scale := float32(1) / float32(math.Sqrt(float64(headDim)))
	build := func(window int) backend.Kernel {
		var k *ir.Kernel
		var err error
		switch kind {
		case "decode":
			k, err = mainkernels.AttnScoresTiledW(nHeads, headDim, kvDim, gqa, sstride, scale, 1, 1, 1, kStride, window)
		case "tiled":
			k, err = mainkernels.AttnScoresTiledW(nHeads, headDim, kvDim, gqa, sstride, scale, rows, 4, 2, kStride, window)
		case "mma":
			k, err = mainkernels.AttnScoresMMAW(nHeads, headDim, kvDim, gqa, sstride, scale, rows, kStride, 1, window)
		}
		if err != nil {
			t.Fatal(err)
		}
		c, err := d.Compile(k)
		if err != nil {
			t.Fatal(err)
		}
		return c
	}
	plain, win := build(0), build(W)
	defer plain.Close()
	defer win.Close()

	rng := rand.New(rand.NewSource(int64(rows*31 + cap)))
	q := make([]float32, rows*nHeads*headDim)
	for i := range q {
		q[i] = float32(rng.NormFloat64())
	}
	kc := make([]float32, kvDim*kStride)
	for i := range kc {
		kc[i] = float32(rng.NormFloat64())
	}
	// pN: a one-row kernel reads its count at [0]; a batched one reads the
	// uniform width at [0] and row r's causal count at [1+r].
	pn := []uint32{uint32(cap)}
	counts := []int{cap}
	if rows > 1 {
		counts = counts[:0]
		for r := 0; r < rows; r++ {
			n := cap - rows + 1 + r
			pn = append(pn, uint32(n))
			counts = append(counts, n)
		}
	}
	var bufs []backend.Buf
	defer func() {
		for _, b := range bufs {
			b.Free()
		}
	}()
	up := func(n int, p []byte) backend.Buf {
		b, err := d.Alloc(n)
		if err != nil {
			t.Fatal(err)
		}
		if p != nil {
			if err := b.Write(p); err != nil {
				t.Fatal(err)
			}
		}
		bufs = append(bufs, b)
		return b
	}
	bQ, bK, bN := up(len(q)*4, f32bytes(q)), up(len(kc)*4, f32bytes(kc)), up(len(pn)*4, u32bytes(pn))
	nOut := rows * nHeads * sstride
	threads := map[string]int{
		"decode": nHeads * cap,
		"tiled":  (rows / 4) * nHeads * ((cap + 1) / 2),
		"mma":    (rows / 16) * nHeads * ((cap + 7) / 8) * 32,
	}[kind]
	run := func(k backend.Kernel) []float32 {
		out := up(nOut*4, nil)
		if err := k.Launch((threads+127)/128, 128, bQ, bK, bN, out); err != nil {
			t.Fatal(err)
		}
		raw := make([]byte, nOut*4)
		if err := out.Read(raw); err != nil {
			t.Fatal(err)
		}
		v := make([]float32, nOut)
		for i := range v {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return v
	}
	a, b := run(plain), run(win)
	masked, kept := 0, 0
	for r := 0; r < rows; r++ {
		lo := counts[r] - min(counts[r], W)
		for h := 0; h < nHeads; h++ {
			for p := 0; p < counts[r]; p++ {
				i := (r*nHeads+h)*sstride + p
				if p < lo {
					if !math.IsInf(float64(b[i]), -1) {
						t.Fatalf("row %d head %d key %d is %v; the window starts at %d", r, h, p, b[i], lo)
					}
					masked++
					continue
				}
				if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
					t.Fatalf("row %d head %d key %d: %v windowed against %v plain -- a kept "+
						"score changed", r, h, p, b[i], a[i])
				}
				kept++
			}
		}
	}
	t.Logf("%d kept identical, %d masked", kept, masked)
}
