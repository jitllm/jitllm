package tier

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// TestFloatMatVecIsServed runs a lone F32, F16 and BF16 matvec through the
// per-matvec device path against a float64 dot of the same weights and a
// float activation. It declined these once (counted as "no kernel"), which
// RULE 8 does not allow; and a float weight must read the FLOAT activation
// (kernels.MatVec's scale-plane slot), which the bound below -- far under
// what int8 activation rounding would give -- holds it to.
func TestFloatMatVecIsServed(t *testing.T) {
	g, err := OpenWith(WithDevices("auto"), WithDeviceTune(TuneOff), WithMinMatVecBytes(0))
	if err != nil || g == nil {
		t.Skipf("no device: %v", err)
	}
	defer g.Close()
	const rows, k = 96, 256
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(math.Sin(0.37*float64(i))) * 3
	}
	for _, typ := range []quant.Type{quant.F32, quant.F16, quant.BF16} {
		bpe := 4
		if typ != quant.F32 {
			bpe = 2
		}
		w := make([]byte, rows*k*bpe)
		want := make([]float64, rows)
		for r := 0; r < rows; r++ {
			for i := 0; i < k; i++ {
				v := float32(math.Cos(0.11*float64(r*k+i))) * 0.5
				o := r*k + i
				switch typ {
				case quant.F32:
					binary.LittleEndian.PutUint32(w[4*o:], math.Float32bits(v))
				case quant.F16:
					h := quant.EncodeHalf(v)
					binary.LittleEndian.PutUint16(w[2*o:], h)
					v = float32(quant.DecodeHalf(h))
				default:
					h := uint16(math.Float32bits(v) >> 16)
					binary.LittleEndian.PutUint16(w[2*o:], h)
					v = math.Float32frombits(uint32(h) << 16)
				}
				want[r] += float64(v) * float64(x[i])
			}
		}
		out := make([]float32, rows)
		for i := range out {
			out[i] = float32(math.NaN())
		}
		if !g.MatVec(out, typ, w, x, rows, k) {
			t.Fatalf("%v: the device declined a lone float matvec", typ)
		}
		var num, den float64
		for r := range want {
			d := float64(out[r]) - want[r]
			num, den = num+d*d, den+want[r]*want[r]
		}
		nmse := num / den
		if math.IsNaN(nmse) || !(nmse < 1e-10) {
			t.Fatalf("%v: NMSE %.3e against a float dot -- the kernel is not reading the "+
				"float activation", typ, nmse)
		}
		t.Logf("%v: NMSE %.2e", typ, nmse)
	}
}
