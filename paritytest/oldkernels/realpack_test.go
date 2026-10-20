package oldkernels_test

import (
	"math"
	"os"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testmodels"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestPackAgainstRealWeights reconstructs every packed weight and compares it
// against quant.Dequant, on real model tensors, for every format that has a GPU
// kernel. Synthetic blocks once passed while a model decoded wrong (the bug was
// in subnormal scales); real weights exercise the scale tables, bit layouts and
// exponent ranges models contain, and quant.Dequant is verified bit-exact
// against libggml.
func TestPackAgainstRealWeights(t *testing.T) {
	models := testmodels.Glob("*.gguf")
	if len(models) == 0 {
		t.Skip("MODEL MISSING: no *.gguf in " + testmodels.Dir() + " (set JITLLM_MODELS to the model directory) -- this gate proved nothing")
	}
	want := map[quant.Type]kernels.Quant{}
	for _, f := range packedRefFormats {
		want[f.g] = f.k
	}
	seen := map[quant.Type]int{}
	for _, path := range models {
		f, err := gguf.Open(path)
		if err != nil {
			continue
		}
		for i := range f.Tensors {
			ti := &f.Tensors[i]
			q, ok := want[ti.Type]
			// An expert bank is 3-D, and it is the only place gpt-oss keeps
			// MXFP4: read it as its experts' rows stacked, which is how the
			// bytes lie.
			if !ok || len(ti.Dims) < 2 || len(ti.Dims) > 3 || seen[ti.Type] >= 2 {
				continue
			}
			k, all := int(ti.Dims[0]), int(ti.Dims[1])
			if len(ti.Dims) == 3 {
				all *= int(ti.Dims[2])
			}
			if k%q.Elems() != 0 {
				continue
			}
			nrows := 24 // enough rows to catch a stride error
			if nrows > all {
				nrows = all
			}
			w := f.Bytes(ti)
			checkPack(t, q, ti.Type, ti.Name, w, nrows, all, k)
			seen[ti.Type]++
		}
		f.Close()
	}
	for gt, q := range want {
		if seen[gt] == 0 {
			t.Logf("no %s tensor available to check", q)
		} else {
			t.Logf("%-5s checked in %d tensor(s)", q, seen[gt])
		}
	}
}

func checkPack(t *testing.T, q kernels.Quant, g quant.Type, name string, w []byte, nrows, allRows, k int) {
	t.Helper()
	qs, dw, scw, err := kernels.PackWeights(q, w, nrows, k)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	sub, bits, biasK, biasArr := kernels.Layout(q)
	perSuper, scOff := kernels.ScaleLayout(q)
	hiW := kernels.HiPlane(q)
	nsub := k / sub
	// pw primary words then hw secondary ones. Restated here rather than asked
	// for, as the scale layout below is: this gate is an independent reader,
	// so it has to be edited in step with the packer.
	pw, hw := sub*bits/32, sub*hiW/32
	words := pw + hw
	ref := make([]float64, k)
	rowBytes := len(w) / allRows
	for r := 0; r < nrows; r++ {
		if err := quant.Dequant(g, w[r*rowBytes:(r+1)*rowBytes], ref); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		for si := 0; si < nsub; si++ {
			// Rebuild scale and bias exactly as the kernel does.
			// One d word per super-block per row: d low, dmin high (zero where
			// the format has no minimum).
			sup := si / perSuper
			// The index is restated, not called: kernels.DIndex on both sides
			// would agree with itself whatever it computed.
			var dword uint32
			var d, dmin float64
			switch {
			case g == quant.MXFP4:
				// Four rows to a word and no f16: MXFP4's super-scale is an E8M0
				// byte, biased so the byte shifted left 23 is the f32. The plane
				// is a quarter the length, so a narrow index would read past it.
				dword = dw[sup*((nrows+3)/4)+r/4]
				d = float64(math.Float32frombits(uint32(uint8(dword>>(8*uint(r%4)))) << 23))
			case perSuper == 1:
				// Two rows to a word on the narrow formats, low half first; the
				// plane is half the length, so the wide index would go out of range.
				dword = dw[sup*((nrows+1)/2)+r/2]
				if r&1 == 1 {
					dword >>= 16
				}
				d = float64(kernels.F16(uint16(dword)))
			default:
				dword = dw[sup*nrows+r]
				d = float64(kernels.F16(uint16(dword)))
				dmin = float64(kernels.F16(uint16(dword >> 16)))
			}
			sc, bias := d, 0.0
			if perSuper == 8 && biasArr {
				// Container v27's 96-bit stream, restated bit by bit: field
				// f = 2*sub (+1 for the minimum) is stream bits [6f, 6f+6),
				// the three words low first.
				field := func(f int) int {
					v := 0
					for i := 0; i < 6; i++ {
						pos := 6*f + i
						w := scw[(sup*3+pos/32)*nrows+r]
						v |= int(w>>uint(pos%32)&1) << uint(i)
					}
					return v
				}
				sc = d * float64(field(2*(si%8))+scOff)
				bias = dmin * float64(field(2*(si%8)+1))
			} else if perSuper > 1 {
				per, stride := 4, 8
				if biasArr {
					per, stride = 2, 16
				}
				w := scw[(si/per)*nrows+r]
				base := uint(si%per) * uint(stride)
				sci := int((w >> base) & 0xFF)
				sc = d * float64(sci+scOff)
				if biasArr {
					mi := int((w >> (base + 8)) & 0xFF)
					bias = dmin * float64(mi)
				}
			}
			if !biasArr {
				bias = float64(biasK) * sc
			}
			for l := 0; l < sub; l++ {
				var qv int32
				byteIdx, hiNib := l, false
				if bits == 4 {
					if l >= sub/2 {
						byteIdx, hiNib = l-sub/2, true
					}
				}
				word := qs[(si*words+byteIdx/4)*nrows+r]
				bv := byte(word >> uint(8*(byteIdx%4)))
				if bits == 4 {
					if hiNib {
						qv = int32(bv >> 4)
					} else {
						qv = int32(bv & 0xF)
					}
					if g == quant.MXFP4 {
						// An e2m1 code: bit 3 the sign, the low three
						// bits {0, .5, 1, 1.5, 2, 3, 4, 6}, doubled to
						// integers, biased by 12. Restated, not read from
						// kernels.Codes, for the reason the index is.
						mag := [8]int32{0, 1, 2, 3, 4, 6, 8, 12}[qv&7]
						if qv&8 != 0 {
							mag = -mag
						}
						qv = mag + 12
					}
				} else {
					qv = int32(int8(bv))
				}
				if hw > 0 {
					// Byte l%4 of hi word l/4/lanes, bit hiW*((l/4)%lanes).
					lanes := 8 / hiW
					g := l / 4
					hword := qs[(si*words+pw+g/lanes)*nrows+r]
					code := (hword >> (8*uint(l%4) + uint(hiW*(g%lanes)))) & (1<<uint(hiW) - 1)
					qv |= int32(code) << uint(bits)
				}
				got := sc*float64(qv) - bias
				exp := ref[si*sub+l]
				if math.Abs(got-exp) > 1e-5*math.Abs(exp)+1e-12 {
					t.Fatalf("%s %s row %d sub %d elem %d: packed %g, gguf %g",
						q, name, r, si, l, got, exp)
				}
			}
		}
	}
}

var _ = os.Stat
