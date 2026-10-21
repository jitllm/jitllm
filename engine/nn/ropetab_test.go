//go:build amd64 || arm64

package nn

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// The rotary table's gates. The oracle is math.Cos and math.Sin in float64
// over the same frequency recurrence the table uses.

// ropeOracle is the float64 table, the body Rope.Table had before the kernel,
// kept in the test so a release binary contains no Go arm.
func ropeOracle(r Rope, pos, npairs int) []float64 {
	mscale := r.Scale
	if mscale == 0 {
		mscale = 1
	}
	out := make([]float64, 2*npairs)
	theta := float64(pos)
	step := math.Pow(r.Base, -2/float64(r.NRot))
	for p := 0; p < npairs; p++ {
		th := theta
		if p < len(r.Freqs) {
			th /= float64(r.Freqs[p])
		}
		out[2*p] = math.Cos(th) * mscale
		out[2*p+1] = math.Sin(th) * mscale
		theta *= step
	}
	return out
}

// ropeConfigs is the rotary geometry of the models tested here plus the ragged
// widths no model has, so the tail path is covered by something.
func ropeConfigs() []struct {
	name string
	r    Rope
} {
	yarn := YarnFreqs(64, 150000, 32, 4096, 32, 1, true)
	return []struct {
		name string
		r    Rope
	}{
		{"llama3.2-1b", Rope{NRot: 64, Base: 500000}},
		{"tinyllama", Rope{NRot: 64, Base: 10000}},
		{"qwen3", Rope{NRot: 128, Base: 1000000}},
		{"gemma2", Rope{NRot: 256, Base: 10000}},
		{"phi3", Rope{NRot: 96, Base: 10000}},
		{"qwen3next-partial", Rope{NRot: 64, Base: 10000000}},
		{"scaled", Rope{NRot: 128, Base: 1000000, Scale: 0.8}},
		{"yarn-freqs", Rope{NRot: 64, Base: 150000, Freqs: yarn}},
		{"ragged-18", Rope{NRot: 36, Base: 10000}},
		{"ragged-26", Rope{NRot: 52, Base: 500000}},
		{"ragged-9", Rope{NRot: 18, Base: 10000}},
		// Narrower than a vector on every tier: stories260K's head dimension
		// is eight, so four pairs is a shipped shape.
		{"stories260K", Rope{NRot: 8, Base: 10000}},
		{"pairs-3", Rope{NRot: 6, Base: 10000}},
		{"pairs-1", Rope{NRot: 2, Base: 10000}},
		{"pairs-5", Rope{NRot: 10, Base: 500000}},
		// Past the arm64 kernel's immediates: from 512 pairs the scale word
		// is beyond LDR s's reach, from about 2340 the last plane beyond LDR
		// q's, and both are read through a register there. A Mamba-1 model
		// asked for 2048 pairs and the emitter panicked.
		{"pairs-512", Rope{NRot: 1024, Base: 10000}},
		{"pairs-2048", Rope{NRot: 4096, Base: 10000}},
		{"pairs-2401", Rope{NRot: 4802, Base: 10000}},
	}
}

// ropePositions sweeps the whole angle range rather than the small positions a
// decode test naturally reaches: 0 and 1 are the degenerate ends, the powers of
// two walk every digit of the decomposition into use, and 131071 is the top of
// the longest context among them.
func ropePositions() []int {
	p := []int{0, 1, 2, 3, 7, 63, 127, 128, 129, 255, 511, 1000, 1023, 1024,
		4095, 4096, 8191, 16383, 16384, 16385, 32768, 65535, 100000, 131071}
	for _, x := range []int{1 << 18, 1<<21 - 1, 1 << 21, 1<<24 + 12345} {
		p = append(p, x)
	}
	return p
}

// TestRopeTableMatchesTheTranscendentals is gate one: the generated table
// against math.Cos and math.Sin over a wide angle sweep.
//
// The 3e-07 absolute bound is derived: the reduction leaves ~1.3e-7 radians
// and the polynomials ~1e-8, so it is about twice the budget, leaving room for
// the SSE tier's extra roundings while every structural error (a quadrant, a
// sin/cos swap, a lost digit plane) is O(1).
func TestRopeTableMatchesTheTranscendentals(t *testing.T) {
	const bound = 3e-7
	var worst float64
	var worstAt string
	ran := 0
	for _, c := range ropeConfigs() {
		npairs := c.r.NRot / 2
		cs := make([]float32, c.r.NRot)
		for _, pos := range ropePositions() {
			if pos >= cpu.RopeTabMaxPos {
				continue
			}
			before := RopeTableCalls()
			c.r.Table(cs, pos)
			if RopeTableCalls() != before+1 {
				t.Fatalf("%s at %d: the kernel did not run -- this gate proves nothing "+
					"about generated code (RULE 10)", c.name, pos)
			}
			ran++
			want := ropeOracle(c.r, pos, npairs)
			for i := range want {
				d := math.Abs(float64(cs[i]) - want[i])
				if d > worst {
					worst, worstAt = d, c.name
				}
				if d > bound {
					t.Errorf("%s pos %d entry %d: %.9g against %.9g, |d| %.3e > %.1e",
						c.name, pos, i, cs[i], want[i], d, bound)
				}
			}
		}
	}
	if ran == 0 {
		t.Fatal("nothing ran")
	}
	t.Logf("%d tables, worst |d| %.3e (%s) against a %.1e bound", ran, worst, worstAt, bound)
}

// TestRopeTableAgainstTheF64Table is gate two: where the kernel lands against
// the float64 table it replaced (as stored in float32), per position, which is
// what decides whether a model's output moves.
func TestRopeTableAgainstTheF64Table(t *testing.T) {
	r := Rope{NRot: 128, Base: 500000}
	npairs := r.NRot / 2
	cs := make([]float32, r.NRot)
	for _, pos := range []int{5, 4096, 32768, 131071} {
		r.Table(cs, pos)
		want := ropeOracle(r, pos, npairs)
		var worst float64
		for i := range want {
			// The f64 table is stored as float32, so the comparison is against
			// its rounded value.
			if d := math.Abs(float64(cs[i]) - float64(float32(want[i]))); d > worst {
				worst = d
			}
		}
		t.Logf("pos %-7d worst |d| %.3e", pos, worst)
		if worst > 3e-7 {
			t.Errorf("pos %d: %.3e against the f64 table", pos, worst)
		}
	}
}

// TestRopeTableExactnessHolds is the arithmetic the whole design rests on,
// checked rather than argued: every digit-by-head product is exact in float32
// and so is their sum, for every real configuration and every digit value.
//
// A tolerance gate cannot see this: rounded products would still land within a
// few e-7 at the positions a sweep visits.
func TestRopeTableExactnessHolds(t *testing.T) {
	for _, c := range ropeConfigs() {
		npairs := c.r.NRot / 2
		tab := ropeTabFor(c.r, npairs)
		for p := 0; p < npairs; p++ {
			for d := 0; d < cpu.RopeTabDigits; d++ {
				h := tab.block[cpu.RopeTabPlaneOff(d, npairs)+p]
				for _, digit := range []int{1, 63, 64, 100, 127} {
					got := float64(float32(digit) * h)
					want := float64(digit) * float64(h)
					if got != want {
						t.Fatalf("%s pair %d digit-plane %d: %d*%v rounded (%v against %v)",
							c.name, p, d, digit, h, got, want)
					}
				}
			}
			// And the four-term sum at the worst case: every digit 127.
			var sum float32
			var exact float64
			for d := 0; d < cpu.RopeTabDigits; d++ {
				h := tab.block[cpu.RopeTabPlaneOff(d, npairs)+p]
				sum += 127 * h
				exact += 127 * float64(h)
			}
			if float64(sum) != exact {
				t.Fatalf("%s pair %d: the four-term sum rounded (%v against %v)",
					c.name, p, float64(sum), exact)
			}
		}
	}
}

// TestRopeTableRefusesAPositionItCannotFold is the guard on the decomposition's
// own bound. Four base-128 digits cover 2^28 positions; a digit above that
// would be silently dropped, so the position is refused.
func TestRopeTableRefusesAPositionItCannotFold(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("a position past the decomposition's reach was accepted -- the " +
				"top digit would have been dropped and the angle silently wrong")
		}
	}()
	r := Rope{NRot: 64, Base: 10000}
	r.Table(make([]float32, 64), cpu.RopeTabMaxPos)
}

// TestRopeTableWritesNothingPastItsTable gates the tail: a table is written
// into a sub-slice of one batch buffer (engine/model/prefill.go), so a store past the
// end silently overwrites the next position's table. The ragged widths matter
// because only the store is narrowed.
func TestRopeTableWritesNothingPastItsTable(t *testing.T) {
	const guard = float32(-12345.5)
	for _, c := range ropeConfigs() {
		npairs := c.r.NRot / 2
		buf := make([]float32, 2*npairs+32)
		for i := range buf {
			buf[i] = guard
		}
		for _, pos := range []int{0, 1, 129, 4096, 131071} {
			c.r.Table(buf[:2*npairs], pos)
			for i := 2 * npairs; i < len(buf); i++ {
				if buf[i] != guard {
					t.Fatalf("%s (%d pairs) at pos %d: word %d past the table was written "+
						"(%v) -- in a batch buffer that is the next position's table",
						c.name, npairs, pos, i-2*npairs, buf[i])
				}
			}
		}
	}
}
