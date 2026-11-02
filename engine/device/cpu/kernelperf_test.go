//go:build amd64 && linux

package cpu

import (
	"github.com/jitllm/jitllm/internal/oracle"
	"math/rand"
	"testing"
	"time"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/dev/bench"
)

// TestQ40Throughput measures the generated kernel on a DRAM-resident weight set
// and reports it as a fraction of a wall measured in the same process.
//
// Logging only, never a gate: a throughput assertion on a noisy box makes the
// file flaky, and a flaky file gets skipped along with its NMSE checks.
func TestQ40Throughput(t *testing.T) {
	if testing.Short() {
		t.Skip("allocates 64 MB of weights")
	}
	if ggufDeclined(t, quant.Q4_0) {
		return
	}
	const k = 2048                   // tinyllama's attention width
	const rowBytes = k / 32 * 18     // 1152 bytes per Q4_0 row
	const rows = 64 << 20 / rowBytes // ~64 MB, far past the last-level cache

	rng := rand.New(rand.NewSource(1))
	packed := make([]byte, rows*rowBytes)
	for i := range packed {
		packed[i] = byte(rng.Intn(256))
	}
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	q := make([]int8, k)
	pairs := make([]float32, 2*(k/32))
	if err := oracle.QuantizeQ8(q, pairs, x, BiasC(quant.Q4_0)); err != nil {
		t.Fatal(err)
	}
	out := make([]float32, rows)
	mask := KernelConst(quant.Q4_0)

	code, err := Emit(Spec{W: quant.Q4_0, Rows: 1, Accs: 1, Cols: 1})
	if err != nil {
		t.Fatal(err)
	}
	kern := mustMap(t, code)
	defer kern.Close()

	// Chunk rows against the I3 budget rather than making one enormous call: a
	// goroutine inside generated code cannot be async-preempted, and entry is
	// cheap enough that calling more often costs nothing measurable.
	chunk := RowsPerCall(k / 32)
	run := func() time.Duration {
		start := time.Now()
		for r := 0; r < rows; r += chunk {
			n := chunk
			if r+n > rows {
				n = rows - r
			}
			args := Args{
				Out: &out[r], W: &packed[r*rowBytes], A: &q[0], AScale: &pairs[0],
				Rows: int64(n), K: int64(k / 32), Scr: &mask[0],
			}
			kern.Call(&args)
		}
		return time.Since(start)
	}
	run()
	best := time.Hour
	for i := 0; i < 5; i++ {
		if d := run(); d < best {
			best = d
		}
	}

	bytes := uint64(rows) * rowBytes
	t.Log(bench.Report("Q4_0 matvec, generated, 1 core", bytes, best, 1))
	t.Logf("%d rows x %d in %v, %d rows per call, kernel %d bytes",
		rows, k, best.Round(time.Millisecond), chunk, len(code))
	t.Logf("reference tier is 0.14 GB/s; this is %.0fx",
		(float64(bytes)/best.Seconds())/0.14e9)
}
