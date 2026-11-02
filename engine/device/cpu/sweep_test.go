//go:build amd64 && linux

package cpu

import (
	"github.com/jitllm/jitllm/internal/oracle"
	"math/rand"
	"testing"
	"time"

	"github.com/jitllm/jitllm/format/quant"
)

// TestPackWidthSweep finds where interleaving stops paying. Measured on one core
// against a DRAM-resident set, best-of-N per width, all widths in one process so
// thermal drift affects them equally.
func TestPackWidthSweep(t *testing.T) {
	if testing.Short() {
		t.Skip("allocates 128 MB")
	}
	if ggufDeclined(t, quant.Q8_0) || ggufDeclined(t, quant.Q4_0) {
		return
	}
	const k = 2048
	nb := k / 32
	for _, wt := range []quant.Type{quant.Q8_0, quant.Q4_0} {
		rowBytes := RowBytes(wt, k)
		rows := 128 << 20 / rowBytes
		rng := rand.New(rand.NewSource(29))
		w := make([]byte, rows*rowBytes)
		for i := range w {
			w[i] = byte(rng.Intn(256))
		}
		x := make([]float32, k)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		q := make([]int8, k)
		pairs := make([]float32, 2*nb)
		oracle.QuantizeQ8(q, pairs, x, BiasC(wt))
		konst := KernelConst(wt)
		out := make([]float32, rows)

		base := 0.0
		t.Logf("--- %s, k=%d, %d MB DRAM-resident, 1 core ---", wt, k, rows*rowBytes>>20)
		for _, width := range []int{1, 2, 3, 4, 5, 6, 7, 8} {
			var code []byte
			var err error
			if width == 1 {
				code, err = Emit(Spec{W: wt, Rows: 1, Accs: 1, Cols: 1})
			} else {
				code, err = EmitInterleaved(wt, width, nb)
			}
			if err != nil {
				t.Logf("  pack%-2d  unsupported: %v", width, err)
				continue
			}
			c := mustMap(t, code)
			chunk := RowsPerCall(nb) / width * width
			if chunk == 0 {
				chunk = width
			}
			run := func() time.Duration {
				start := time.Now()
				for r := 0; r+width <= rows; r += chunk {
					n := chunk
					if r+n > rows {
						n = (rows - r) / width * width
					}
					if n == 0 {
						break
					}
					rc := int64(n)
					if width > 1 {
						rc = int64(n / width)
					}
					args := Args{Out: &out[r], W: &w[r*rowBytes], A: &q[0], AScale: &pairs[0],
						Rows: rc, K: int64(nb), RowStr: int64(rowBytes), Scr: &konst[0]}
					c.Call(&args)
				}
				return time.Since(start)
			}
			run()
			best := time.Hour
			for i := 0; i < 7; i++ {
				if d := run(); d < best {
					best = d
				}
			}
			c.Close()
			gbs := float64(rows) * float64(rowBytes) / best.Seconds() / 1e9
			if width == 1 {
				base = gbs
			}
			t.Logf("  pack%-2d %6.2f GB/s  %5.3fx  %5d B", width, gbs, gbs/base, len(code))
		}
	}
}
