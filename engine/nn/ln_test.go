//go:build amd64 || arm64

package nn

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// TestLnMatchesTheDefinition holds Ln32JIT to math.Log at every ragged width
// from 1 to a few vectors, with a guard after the row, over inputs from the
// log-mel floor (2^-24) to 1e6. The violation: a row left as it was reads
// past the bound at every element (ln x = x only near x = 0.567).
func TestLnMatchesTheDefinition(t *testing.T) { checkLn(t) }

func checkLn(t *testing.T) {
	t.Helper()
	const guard = 2 * cpu.ElemLanes
	const guardBits = 0x7fc0beef
	for n := 1; n <= 5*cpu.ElemLanes+3; n++ {
		row := make([]float32, n+guard)
		for i := 0; i < n; i++ {
			// From 2^-24 up through 1e6, a factor of ~2.9 a step.
			row[i] = float32(math.Exp(-16.6 + float64(i%31)*1.06))
		}
		for i := n; i < len(row); i++ {
			row[i] = math.Float32frombits(guardBits)
		}
		in := append([]float32(nil), row[:n]...)
		Ln32JIT(row[:n])
		for i := n; i < len(row); i++ {
			if math.Float32bits(row[i]) != guardBits {
				t.Fatalf("n=%d: wrote element %d past the end", n, i)
			}
		}
		for i := 0; i < n; i++ {
			want := math.Log(float64(in[i]))
			// The atanh series is ~1e-6 absolute; e*ln2 adds a rounding.
			if d := math.Abs(float64(row[i]) - want); !(d <= 3e-6*(1+math.Abs(want))) {
				t.Fatalf("n=%d: ln(%v) is %v, want %v (|d| %.3g)", n, in[i], row[i], want, d)
			}
			if math.Abs(float64(in[i])-want) <= 3e-6*(1+math.Abs(want)) {
				t.Fatalf("n=%d: element %d is its own log (%v): the gate cannot see a kernel that does nothing", n, i, in[i])
			}
		}
	}
}
