package cpu

import (
	"math"
	"testing"
	"unsafe"
)

// rowStatRef is the float64 definition of the two row statistics: the
// gaussian top-k max(x - (mean + c*sd), 0), sd the population deviation, and
// the magnitude match x * rms(ref) / sqrt(max(mean(x^2), eps)).
func rowStatRef(mode RowMode, x, ref []float32, c, eps float64) []float64 {
	n := float64(len(x))
	out := make([]float64, len(x))
	switch mode {
	case RowGauss:
		var mean, v float64
		for _, e := range x {
			mean += float64(e)
		}
		mean /= n
		for _, e := range x {
			d := float64(e) - mean
			v += d * d
		}
		cut := mean + c*math.Sqrt(v/n)
		for i, e := range x {
			out[i] = math.Max(float64(e)-cut, 0)
		}
	case RowStd:
		var mean, v float64
		for _, e := range x {
			mean += float64(e)
		}
		mean /= n
		for _, e := range x {
			d := float64(e) - mean
			v += d * d
		}
		sd := 0.0
		if n > 1 {
			sd = math.Sqrt(v / (n - 1))
		}
		for i, e := range x {
			out[i] = (float64(e) - mean) / (sd + eps)
		}
	case RowMag:
		var sr, sx float64
		for i := range x {
			sr += float64(ref[i]) * float64(ref[i])
			sx += float64(x[i]) * float64(x[i])
		}
		s := math.Sqrt(sr/n) / math.Sqrt(math.Max(sx/n, eps))
		for i, e := range x {
			out[i] = float64(e) * s
		}
	}
	return out
}

// rowStatInputs is a gate-like row with a mean of its own (so the gaussian's
// mean is not zero) and a reference about three times its size (so the match
// scales by something other than one).
func rowStatInputs(n int) (x, ref []float32) {
	x, ref = make([]float32, n), make([]float32, n)
	for i := range x {
		x[i] = float32(0.7 + math.Sin(float64(i)*0.37)*float64(1+i%7))
		ref[i] = float32(3 * math.Cos(float64(i)*0.23) * float64(1+i%5))
	}
	return x, ref
}

// runRowStat runs one row-statistic kernel on x (with ref for the match) and
// checks nothing past the row was written.
func runRowStat(t *testing.T, c *Code, name string, x, ref []float32, kc []float32) []float32 {
	t.Helper()
	n := len(x)
	y := sentinelRow(n)
	args := Args{Out: &y[0], A: (*int8)(unsafe.Pointer(&x[0])), Scr: (*byte)(unsafe.Pointer(&kc[0])), K: int64(n)}
	if ref != nil {
		args.AScale = &ref[0]
	}
	c.Call(&args)
	for i := n; i < len(y); i++ {
		if math.Float32bits(y[i]) != softmaxSentinel {
			t.Fatalf("%s n=%d: wrote element %d past the end", name, n, i)
		}
	}
	return y[:n]
}

// TestRowStatsMatchReference gates Gemma 3n's gaussian top-k and AltUp's
// magnitude match against float64 at whole and ragged widths, the gaussian at
// a multiple of the deviation low enough that a few-element row keeps
// something (Gemma 3n's own, 1.645, keeps 5%). The gaussian's
// cutoff is checked to cut: some elements survive and some are zeroed, or
// the kernel could be a ReLU and pass.
func TestRowStatsMatchReference(t *testing.T) {
	const c, eps = 0.3, 1e-5
	for _, n := range []int{1, 2, 3, 5, 7, 8, 9, 13, 24, 33, 35, 64, 100, 128, 768, 2048, 8192} {
		x, ref := rowStatInputs(n)
		for _, m := range []struct {
			mode RowMode
			code []byte
			kc   []float32
			ref  []float32
		}{
			{RowGauss, hostBytes(t)(hostTable().RowStat(n, RowGauss)), []float32{1 / float32(n), 0, c}, nil},
			{RowMag, hostBytes(t)(hostTable().RowStat(n, RowMag)), []float32{1 / float32(n), eps, 1}, ref},
			{RowStd, hostBytes(t)(hostTable().RowStat(n, RowStd)), rowStdKons(n, eps), nil},
		} {
			name := map[RowMode]string{RowGauss: "gausstopk", RowMag: "magmatch", RowStd: "rowstd"}[m.mode]
			code := onHost(t)(m.code, nil)
			got := runRowStat(t, code, name, x, m.ref, m.kc)
			code.Close()
			want := rowStatRef(m.mode, x, ref, c, eps)
			var se, sy, kept float64
			for i := range want {
				d := float64(got[i]) - want[i]
				se, sy = se+d*d, sy+want[i]*want[i]
				if want[i] > 0 {
					kept++
				}
			}
			if sy == 0 {
				if n > 8 {
					t.Errorf("%s n=%d: the reference is all zero", name, n)
				}
				continue
			}
			if nmse := se / sy; !(nmse <= 1e-12) {
				t.Errorf("%s n=%d: NMSE %.3e against float64", name, n, nmse)
			}
			if m.mode == RowGauss && n >= 64 && (kept == 0 || kept == float64(n)) {
				t.Errorf("gausstopk n=%d: %v of %d kept, so the cutoff cut nothing or everything", n, kept, n)
			}
		}
	}
}

// rowStdKons is RowStd's constants: [1/n, eps, 1, 1/(n-1)], the last 0 for
// one element.
func rowStdKons(n int, eps float32) []float32 {
	var unb float32
	if n > 1 {
		unb = 1 / float32(n-1)
	}
	return []float32{1 / float32(n), eps, 1, unb}
}
