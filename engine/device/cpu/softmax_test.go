package cpu

import (
	"math"
	"testing"
	"unsafe"
)

// TestEmitSoftmaxMatchesReference gates the generated softmax against a float64
// three-pass reference, over the row lengths a real decode produces.
//
// The lengths are deliberately not round: a score row is pos+1, and every
// ragged length exercises the tail.
func TestEmitSoftmaxMatchesReference(t *testing.T) {
	code := onHost(t)(hostTable().Softmax())
	defer code.Close()
	consts := expConsts()

	for _, n := range []int{1, 2, 3, 7, 8, 9, 13, 31, 32, 33, 64, 127, 512, 1537} {
		row := sentinelRow(n)
		want := make([]float64, n)
		for i := 0; i < n; i++ {
			// A wide range, including values far enough apart that a missing
			// max-subtraction would overflow rather than merely drift.
			v := math.Sin(float64(i)*0.9) * float64(3+i%23)
			row[i] = float32(v)
			want[i] = v
		}

		// float64 three-pass reference.
		mx := want[0]
		for _, v := range want {
			if v > mx {
				mx = v
			}
		}
		var sum float64
		for i, v := range want {
			want[i] = math.Exp(v - mx)
			sum += want[i]
		}
		for i := range want {
			want[i] /= sum
		}

		callSoftmax(t, code, consts, row, n)

		var sse, sy2 float64
		for i := 0; i < n; i++ {
			d := float64(row[i]) - want[i]
			sse, sy2 = sse+d*d, sy2+want[i]*want[i]
		}
		if nmse := sse / sy2; nmse > 1e-10 {
			t.Errorf("n=%d: NMSE %.3e against the float64 reference", n, nmse)
		}
		// A row's probabilities sum to 1 over the real entries: a tail lane
		// that leaked exp(-max) into the sum would show up here rather than in
		// the NMSE above.
		var tot float64
		for i := 0; i < n; i++ {
			tot += float64(row[i])
		}
		if math.Abs(tot-1) > 1e-5 {
			t.Errorf("n=%d: probabilities sum to %.6f, not 1 -- a tail lane leaked into the sum", n, tot)
		}
	}
}

// TestEmitSoftmaxMaxIsTheRealMax puts two saturating entries in every pair of
// lane positions and checks their ratio, which is the only thing a wrong
// maximum can destroy. Softmax is shift-invariant, so a wrong maximum breaks
// only once x-m passes exp's +88 clamp, and a single over-clamped entry still
// normalises to ~1; two saturating entries tie instead of being 50 nats apart.
func TestEmitSoftmaxMaxIsTheRealMax(t *testing.T) {
	code := onHost(t)(hostTable().Softmax())
	defer code.Close()
	consts := expConsts()

	const n = 16
	for hi := 0; hi < n; hi++ {
		lo := (hi + 5) % n
		row := sentinelRow(n)
		for i := range row[:n] {
			row[i] = float32(i)
		}
		row[hi], row[lo] = 200, 150
		callSoftmax(t, code, consts, row, n)

		// The two are 50 apart, so the smaller must be exp(-50) = 1.9e-22 of
		// the larger. A reduction that missed the true maximum clamps both and
		// reports 1.0.
		if row[hi] < 0.999 {
			t.Errorf("peaks at %d/%d: the larger holds %.6g of the mass, want ~1", hi, lo, row[hi])
		}
		if ratio := float64(row[lo]) / float64(row[hi]); ratio > 1e-20 {
			t.Errorf("peaks at %d/%d: ratio %.3g, want ~%.3g -- the max reduction "+
				"missed one of those lanes and the clamp tied them",
				hi, lo, ratio, math.Exp(-50))
		}
	}
}

// TestEmitSoftmaxDeepNegativeRow holds the kernel to rows that sit entirely
// below exp's low clamp (qwen2's scores reach -872). A tail lane that is not
// neutral, or a maximum seeded from a constant, would become the row's maximum
// there.
func TestEmitSoftmaxDeepNegativeRow(t *testing.T) {
	code := onHost(t)(hostTable().Softmax())
	defer code.Close()
	consts := expConsts()

	for _, n := range []int{1, 2, 5, 9, 17} {
		for _, base := range []float64{-872.46, -1e5, -3.4e30} {
			row := sentinelRow(n)
			want := make([]float64, n)
			for i := 0; i < n; i++ {
				v := base + float64(i)*0.5
				row[i] = float32(v)
				want[i] = v
			}
			mx := want[n-1] // ascending, so the last is the maximum
			var sum float64
			for i, v := range want {
				want[i] = math.Exp(v - mx)
				sum += want[i]
			}
			for i := range want {
				want[i] /= sum
			}
			callSoftmax(t, code, consts, row, n)

			var sse, sy2 float64
			for i := 0; i < n; i++ {
				d := float64(row[i]) - want[i]
				sse, sy2 = sse+d*d, sy2+want[i]*want[i]
			}
			if nmse := sse / sy2; nmse > 1e-10 {
				t.Errorf("n=%d base=%g: NMSE %.3e -- got %v, want %v",
					n, base, nmse, row[:n], want)
			}
		}
	}
}

const softmaxSentinel = 0x7fc0beef

// sentinelRow is n entries followed by a guard of NaNs with a payload, which a
// kernel that writes past n would disturb.
func sentinelRow(n int) []float32 {
	row := make([]float32, n+2*ElemLanes)
	for i := n; i < len(row); i++ {
		row[i] = math.Float32frombits(softmaxSentinel)
	}
	return row
}

// callSoftmax runs the kernel over row[:n] and fails if it wrote the guard.
func callSoftmax(t *testing.T, code *Code, consts []float32, row []float32, n int) {
	t.Helper()
	args := Args{
		Out:  &row[0],
		Scr:  (*byte)(unsafe.Pointer(&consts[0])),
		K:    int64(n / ElemLanes),
		Rows: int64(n % ElemLanes),
	}
	code.Call(&args)
	for i := n; i < len(row); i++ {
		if math.Float32bits(row[i]) != softmaxSentinel {
			t.Fatalf("n=%d: wrote element %d past the end (%v)", n, i, row[i])
		}
	}
}
