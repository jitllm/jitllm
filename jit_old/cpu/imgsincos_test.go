//go:build amd64 || arm64

package cpu

import (
	"math"
	"testing"
	"unsafe"
)

// sinCosRef is the table as engine/model built it in Go before the kernel:
// the frequency's power in float64 rounded to float32, the angle a float32
// product, its sine and cosine in float64 rounded to float32. fold is the
// violation: the angle folded in float64, as the rotary kernel folds its own.
func sinCosRef(k, rows int, fold bool) (sin, cos []float32) {
	sin, cos = make([]float32, rows*k), make([]float32, rows*k)
	for p := 0; p < rows; p++ {
		for i := 0; i < k; i++ {
			om := float32(1 / math.Pow(10000, float64(float32(i)/float32(k))))
			a := float64(float32(p) * om)
			if fold {
				a = float64(p) * float64(om)
			}
			sin[p*k+i], cos[p*k+i] = float32(math.Sin(a)), float32(math.Cos(a))
		}
	}
	return sin, cos
}

// TestSinCosTabMatchesTheReference gates SinCosTab on every tier this host
// executes against the Go table it replaced, bit for bit: at every
// frequency count a real resampler width makes (d/4 for d a whole number of
// 128-wide heads, MiniCPM-V's 2304, 3584 and 4096 among them) over the
// positions a slice grid reaches, and at ragged counts 1..9 so the pair loop's
// odd frequency runs, with guards after both tables. Folding the angle in
// float64 must fail it.
func TestSinCosTabMatchesTheReference(t *testing.T) {
	type shape struct{ k, rows int }
	var shapes []shape
	for k := 1; k <= 9; k++ {
		shapes = append(shapes, shape{k, 13})
	}
	for _, k := range []int{32, 64, 96, 128, 256, 384, 512, 576, 768, 896, 1024, 1152, 1280, 1536, 2048} {
		shapes = append(shapes, shape{k, 72})
	}
	scr := SinCosConsts(10000)
	for _, tr := range imgTiers() {
		b, err := tr.em.SinCosTab()
		c := imgMap(t, tr, "sincostab", b, err)
		for _, violate := range []bool{false, true} {
			bad, n := 0, 0
			for _, s := range shapes {
				sin, cos := make([]float32, s.rows*s.k+4), make([]float32, s.rows*s.k+4)
				for i := range sin {
					sin[i], cos[i] = float32(math.NaN()), float32(math.NaN())
				}
				c.Call(&Args{
					Out:  &sin[0],
					Out2: &cos[0],
					K:    int64(s.k),
					Rows: int64(s.rows),
					Scr:  (*byte)(unsafe.Pointer(&scr[0])),
				})
				for i := s.rows * s.k; i < len(sin); i++ {
					if !math.IsNaN(float64(sin[i])) || !math.IsNaN(float64(cos[i])) {
						t.Fatalf("%s K %d: wrote past the tables at %d", tr.name, s.k, i)
					}
				}
				ws, wc := sinCosRef(s.k, s.rows, violate)
				for i := range ws {
					n++
					if math.Float32bits(sin[i]) != math.Float32bits(ws[i]) || math.Float32bits(cos[i]) != math.Float32bits(wc[i]) {
						bad++
						if !violate {
							t.Fatalf("%s K %d position %d frequency %d: (%v, %v), want (%v, %v)", tr.name, s.k,
								i/s.k, i%s.k, sin[i], cos[i], ws[i], wc[i])
						}
					}
				}
			}
			if violate && bad == 0 {
				t.Fatalf("%s: the angle folded in float64 agreed everywhere -- this gate proves nothing", tr.name)
			}
			if violate {
				t.Logf("%s: %d entries bit-exact; the float64 fold differs at %d", tr.name, n, bad)
			}
		}
		c.Close()
	}
}

// TestSinCosMarginsHoldTheTable is the margin the kernel's exactness rests on:
// every true value at the real widths sits further from a float32 rounding
// boundary than the kernel's float64 error (a few ulp, 1e-15 relative), so
// any float64 evaluation that good rounds to the same float32 the Go table
// did. It reads the margin off math's own values, which are within an ulp.
func TestSinCosMarginsHoldTheTable(t *testing.T) {
	margin := func(v float64) float64 {
		f := float32(v)
		lo := float64(math.Nextafter32(f, float32(math.Inf(-1))))
		hi := float64(math.Nextafter32(f, float32(math.Inf(1))))
		d := math.Min(math.Abs(v-(float64(f)+lo)/2), math.Abs(v-(float64(f)+hi)/2))
		return d / math.Abs(v)
	}
	worst := 1.0
	for _, k := range []int{384, 576, 896, 1024} {
		for i := 0; i < k; i++ {
			om := 1 / math.Pow(10000, float64(float32(i)/float32(k)))
			worst = math.Min(worst, margin(om))
			for p := 1; p < 72; p++ {
				a := float64(float32(p) * float32(om))
				for _, v := range []float64{math.Sin(a), math.Cos(a)} {
					if v != 0 {
						worst = math.Min(worst, margin(v))
					}
				}
			}
		}
	}
	t.Logf("the nearest a true value comes to a float32 rounding boundary: %.2e relative", worst)
	if worst < 1e-14 {
		t.Fatalf("a value sits %.2e from a rounding boundary: the kernel's float64 error could flip it", worst)
	}
}
