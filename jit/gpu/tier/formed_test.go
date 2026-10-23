package tier

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestServedMatVecIsComposed runs a lone quantized matvec through the
// per-matvec device path for every packed type a device has a form for, and
// holds it to the oracle's matvec over the same int8 form of the row
// (oracle.PackActivationsInto, dequantized). The row is uploaded as floats
// and quantized by the composer's kernels; the stats count that they ran.
// The float arm forms it with no int8 form, the weights decoded and the row
// kept float, as the shadow's reference slice is (shadow.go), and holds it
// to the oracle over the row itself.
func TestServedMatVecIsComposed(t *testing.T) {
	ran := 0
	for _, spec := range testdevices.Specs("cuda", "vulkan", "metal") {
		for _, float := range []bool{false, true} {
			g, err := OpenWith(WithDevices(spec), WithDeviceTune(TuneOff), WithMinMatVecBytes(0))
			if err != nil || g == nil {
				t.Logf("%s: no device (%v)", spec, err)
				continue
			}
			ran++
			name := spec
			if float {
				name += "/float"
				for _, d := range g.devs {
					d.composer = jit.Default().WithoutInt8(jit.DeviceGPU)
				}
			}
			t.Run(name, func(t *testing.T) { servedMatVecIsComposed(t, g, float) })
			g.Close()
		}
	}
	if ran == 0 {
		t.Fatal("no device of any backend: this gate proved nothing")
	}
}

// servedMatVecIsComposed is TestServedMatVecIsComposed on the device of g.
func servedMatVecIsComposed(t *testing.T, g *GPU, float bool) {
	rng := rand.New(rand.NewSource(3))
	const rows, k = 96, 512
	checked := 0
	for _, typ := range quant.PackedTypes {
		q, ok := quantOf(typ)
		if !ok {
			t.Logf("%v: no device form (%s)", typ, kernels.DeviceWhyNot(q))
			continue
		}
		w := formedTestWeights(q, rows, k, rng)
		x := make([]float32, k)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		// The oracle's dot over the row's own int8 form: what the device's
		// integer dot computes, up to the order its floats are summed in.
		a, as, sum := make([]uint32, k/4), make([]float32, k/32), make([]float32, k/16)
		if err := oracle.PackActivationsInto(a, as, sum, x, 32); err != nil {
			t.Fatal(err)
		}
		xq := make([]float32, k)
		for i := range xq {
			xq[i] = float32(int8(a[i/4]>>(8*(i%4)))) * as[i/32]
		}
		if float {
			xq = x
		}
		want := make([]float32, rows)
		if err := oracle.MatVec(want, typ, w, xq, rows, k); err != nil {
			t.Fatal(err)
		}
		out := make([]float32, rows)
		for i := range out {
			out[i] = float32(math.NaN())
		}
		before := g.Stats().Formed
		if !g.MatVec(out, typ, w, x, rows, k) {
			t.Fatalf("%v: the device declined a lone matvec (%s)", typ, g.Err())
		}
		if g.Stats().Formed != before+1 {
			t.Fatalf("%v: the matvec was served but not by the composer's kernels", typ)
		}
		var num, den float64
		for r := range want {
			d := float64(out[r]) - float64(want[r])
			num, den = num+d*d, den+float64(want[r])*float64(want[r])
		}
		nmse := num / den
		if math.IsNaN(nmse) || !(nmse < 1e-9) {
			t.Fatalf("%v: NMSE %.3e against the oracle (float row %v)", typ, nmse, float)
		}
		t.Logf("%v: NMSE %.2e", typ, nmse)
		checked++
	}
	if checked == 0 {
		t.Fatal("no packed type was served: this gate proved nothing")
	}
}

// formedTestWeights is random GGUF blocks of q with finite scales near one:
// random bits reach f16's Inf/NaN exponent one time in 32.
func formedTestWeights(q kernels.Quant, rows, k int, rng *rand.Rand) []byte {
	f16At := map[kernels.Quant][]int{
		kernels.Q4_0: {0}, kernels.Q8_0: {0}, kernels.Q5_0: {0}, kernels.Q5_1: {0, 2},
		kernels.Q4_K: {0, 2}, kernels.Q5_K: {0, 2}, kernels.Q3_K: {108}, kernels.Q6_K: {208}, kernels.TQ2: {64},
	}
	nblk, bb := k/q.Elems(), q.BlockBytes()
	raw := make([]byte, rows*nblk*bb)
	rng.Read(raw)
	for i := 0; i < rows*nblk; i++ {
		switch {
		case q == kernels.MXFP4:
			raw[i*bb] = byte(120 + rng.Intn(10))
		case q == kernels.TQ2F:
			binary.LittleEndian.PutUint32(raw[i*bb+64:], math.Float32bits(0.5+rng.Float32()))
		default:
			for _, off := range f16At[q] {
				binary.LittleEndian.PutUint16(raw[i*bb+off:], uint16(0x2000|rng.Intn(0x0C00)))
			}
		}
	}
	return raw
}
