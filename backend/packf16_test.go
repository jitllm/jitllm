package backend_test

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/ir"
)

// TestPackF16RoundTrip runs OpPackF16 and OpCvtF16H against each other on real
// hardware, on every backend present.
//
// OpPackF16 is on the path of every stored key and value in an f16 KV cache,
// and a halves-swapped lowering would run and answer nonsense. It packs two
// distinct values and reads both halves back, so an order swap fails rather
// than cancelling.
//
// The comparison is against Go's own float32->float16->float32, not against the
// input: binary16 has 11 bits of mantissa and the point of the op is to lose
// the rest. What must match is which bits are lost.
func TestPackF16RoundTrip(t *testing.T) {
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	// index 3 is deliberately not representable in binary16.
	in := []float32{0, 1, -1, 3.14159, -0.5, 2048.5, -2.71828, 1024, 0.25, 1234.5}
	const n = 5 // pairs

	for _, d := range devs {
		d := d
		t.Run(d.API(), func(t *testing.T) {
			defer d.Close()
			b := ir.New("k", [3]int{n, 1, 1})
			pIn := b.Param("pIn", ir.F32)
			pLo := b.Param("pLo", ir.F32)
			i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
				b.Const(ir.U32, n-1))
			two := b.Mul(ir.U32, i, b.Const(ir.U32, 2))
			lo := b.Load(ir.F32, pIn, two, 0)
			hi := b.Load(ir.F32, pIn, two, 1)
			// The packed word is stored raw and decoded on the host: a
			// driver may fold Unpack(Pack(x)) back to x on the device, which
			// hides the rounding. Crossing memory checks the bits the KV
			// cache will actually contain.
			b.Store(pLo, i, b.Bitcast(ir.F32, b.PackF16(lo, hi)), 0)
			k, err := mustBuild(t, b), error(nil)
			if err = k.Validate(); err != nil {
				t.Fatalf("validate: %v", err)
			}
			kern, err := d.Compile(k)
			if err != nil {
				t.Fatalf("compile: %v", err)
			}
			defer kern.Close()

			bufIn, _ := d.Alloc(len(in) * 4)
			bufLo, _ := d.Alloc(n * 4)
			defer bufIn.Free()
			defer bufLo.Free()
			if err := bufIn.Write(packBytes(in)); err != nil {
				t.Fatal(err)
			}
			if err := kern.Launch(1, n, bufIn, bufLo); err != nil {
				t.Fatal(err)
			}
			gotLo := make([]byte, n*4)
			if err := bufLo.Read(gotLo); err != nil {
				t.Fatal(err)
			}
			for j := 0; j < n; j++ {
				w := uint32(gotLo[j*4]) | uint32(gotLo[j*4+1])<<8 |
					uint32(gotLo[j*4+2])<<16 | uint32(gotLo[j*4+3])<<24
				gl, gh := f16dec(uint16(w)), f16dec(uint16(w>>16))
				for _, c := range []struct {
					name      string
					got, want float64
				}{{"lo", gl, float64(in[2*j])}, {"hi", gh, float64(in[2*j+1])}} {
					if c.want == 0 {
						if c.got != 0 {
							t.Errorf("pair %d %s: got %v, want 0", j, c.name, c.got)
						}
						continue
					}
					// binary16 keeps 11 significant bits, so 2^-10 relative is
					// the loosest a correct rounding can be. A swapped splice
					// misses by far more than this, which is why the pairs hold
					// distinct magnitudes.
					if rel := math.Abs(c.got-c.want) / math.Abs(c.want); rel > 1.0/1024 {
						t.Errorf("pair %d %s: got %v, want %v within 2^-10 (rel %.2g)",
							j, c.name, c.got, c.want, rel)
					}
				}
			}
		})
	}
}

func packBytes(v []float32) []byte {
	out := make([]byte, len(v)*4)
	for i, x := range v {
		u := math.Float32bits(x)
		out[i*4], out[i*4+1], out[i*4+2], out[i*4+3] = byte(u), byte(u>>8), byte(u>>16), byte(u>>24)
	}
	return out
}

// f16dec is gguf/dequant.go's decoder, duplicated here so the test does not
// depend on an unexported symbol in another package.
func f16dec(u uint16) float64 {
	sign := uint32(u>>15) << 31
	exp := int32(u>>10) & 0x1F
	man := uint32(u & 0x3FF)
	switch {
	case exp == 0 && man == 0:
		return math.Float64frombits(uint64(sign) << 32)
	case exp == 0: // subnormal
		f := float64(man) / 1024 * math.Pow(2, -14)
		if sign != 0 {
			f = -f
		}
		return f
	case exp == 0x1F:
		if sign != 0 {
			return math.Inf(-1)
		}
		return math.Inf(1)
	}
	f := (1 + float64(man)/1024) * math.Pow(2, float64(exp-15))
	if sign != 0 {
		f = -f
	}
	return f
}
