//go:build amd64 || arm64

package nn

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestLogSoftmaxMatchesTheOracle holds LogSoftmax32JIT to the f64 definition
// on the host tier at every ragged width from 1 to a few vectors, at a
// vocabulary width, with a guard after the row. Rows also sit at offsets of
// +-300, where exp overflows or underflows f32: a kernel that dropped the
// maximum's subtraction returns infinities there, not a near miss.
func TestLogSoftmaxMatchesTheOracle(t *testing.T) {
	checkLogSoftmax(t)
}

func checkLogSoftmax(t *testing.T) {
	t.Helper()
	const guard = 2 * cpu.ElemLanes
	const guardBits = 0x7fc0beef
	widths := []int{}
	for n := 1; n <= 5*cpu.ElemLanes+3; n++ {
		widths = append(widths, n)
	}
	widths = append(widths, 1027, 32000)
	for _, off := range []float64{0, 300, -300} {
		for _, n := range widths {
			row := make([]float32, n+guard)
			for i := 0; i < n; i++ {
				row[i] = float32(off + math.Sin(float64(i)*0.37+off)*float64(1+i%13))
			}
			for i := n; i < len(row); i++ {
				row[i] = math.Float32frombits(guardBits)
			}
			want := oracle.LogSoftmax32(row[:n])
			LogSoftmax32JIT(row, n)
			for i := n; i < len(row); i++ {
				if math.Float32bits(row[i]) != guardBits {
					t.Fatalf("n=%d off=%v: wrote element %d past the end", n, off, i)
				}
			}
			for i := 0; i < n; i++ {
				if math.IsNaN(want[i]) || math.IsInf(want[i], 0) {
					t.Fatalf("n=%d: the oracle is not finite at %d", n, i)
				}
				// The row's own f32 rounding at |x| ~ 300 is ~3e-5; the
				// kernel's ln(sum) adds ~1e-6.
				tol := 2e-6*(1+math.Abs(want[i])) + 4e-8*math.Abs(off)
				if d := math.Abs(float64(row[i]) - want[i]); !(d <= tol) {
					t.Fatalf("n=%d off=%v: element %d is %v, the oracle %v (|d| %.3g)", n, off, i, row[i], want[i], d)
				}
			}
		}
	}
}
