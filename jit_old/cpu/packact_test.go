package cpu

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
)

// TestPackActMatchesGoPacker holds the generated activation packer to the Go one
// bit for bit, on every output it writes. The output is integers, so an NMSE
// would pass a kernel that rounds exact halves to nearest-even (VCVTPS2DQ)
// instead of half away. The zero window is routine: MatMul zero-fills a ragged
// tile's padding rows, and 1/0 = +Inf, 0*Inf = NaN without the mask.
func TestPackActMatchesGoPacker(t *testing.T) {
	// The packer feeds the GGUF GEMM, the AVX2 tier's alone.
	if ggufDeclined(t, quant.Q4_0) {
		return
	}
	for _, tok := range []int{1, 4, 16} {
		for _, tp := range []quant.Type{quant.Q4_0, quant.Q3_K, quant.Q6_K, quant.Q4_K} {
			for _, window := range []int{32, 256} {
				for _, mode := range []string{"normal", "zero", "atmax"} {
					packActCase(t, tok, tp, window, mode)
				}
			}
		}
	}
}

func packActCase(t *testing.T, tok int, tp quant.Type, window int, mode string) {
	t.Helper()
	const k = 512
	per := window / Q8Block
	nb := k / Q8Block
	rng := rand.New(rand.NewSource(int64(tok*7919 + int(tp)*131 + window + len(mode))))

	x := make([]float32, tok*k)
	for i := range x {
		switch mode {
		case "zero":
			x[i] = 0
		case "atmax":
			// Values that make |v*inv| land exactly on 127 and on .5
			// boundaries, which is where a nearest-even convert diverges from
			// round-half-away.
			x[i] = float32(rng.Intn(255)-127) / 2
		default:
			x[i] = float32(rng.NormFloat64() * 3)
		}
	}
	mk := func() ([]int8, []float32, []int32, []int32) {
		return make([]int8, tok*k), make([]float32, nb*tok),
			make([]int32, nb*tok), make([]int32, 2*nb*tok)
	}
	dA, sA, mA, hA := mk()
	if err := packGEMM(tp, dA, sA, mA, hA, x, tok, k, 0, tok, 0, nb, window); err != nil {
		t.Fatal(err)
	}

	// The generated kernel, driven exactly as the caller will drive it: one
	// call per (token, window).
	dB, sB, mB, hB := mk()
	wantHalf := NeedsHalfSums(tp)
	code := mustMap(t, EmitPackAct(tok, wantHalf))
	defer code.Close()
	konst := []float32{127, 1, 0.5,
		math.Float32frombits(0x7FFFFFFF), math.Float32frombits(0x80000000),
		math.Float32frombits(uint32(int32(BiasC(tp))))}
	for n := 0; n < tok; n++ {
		row := x[n*k : (n+1)*k]
		for b0 := 0; b0 < nb; b0 += per {
			w1 := (b0 + per) * Q8Block
			if w1 > k {
				w1 = k
			}
			blocks := per
			if b0+blocks > nb {
				blocks = nb - b0
			}
			args := Args{
				A:        (*int8)(unsafe.Pointer(&row[b0*Q8Block])),
				K:        int64(w1 - b0*Q8Block),
				Q32:      &row[b0*Q8Block],
				Rows:     int64(blocks),
				W:        (*byte)(unsafe.Pointer(&dB[((b0*Q8Block/4)*tok+n)*4])),
				Out:      &sB[b0*tok+n],
				ASum:     &mB[b0*tok+n],
				AHalfSum: &hB[2*b0*tok+n],
				Scr:      (*byte)(unsafe.Pointer(&konst[0])),
			}
			code.Call(&args)
		}
	}

	name := tp.String() + "/tok" + itoa(tok) + "/win" + itoa(window) + "/" + mode
	for i := range dA {
		if dA[i] != dB[i] {
			t.Fatalf("%s: dst[%d] = %d, want %d", name, i, dB[i], dA[i])
		}
	}
	for i := range sA {
		if sA[i] != sB[i] {
			t.Fatalf("%s: scale[%d] = %v, want %v", name, i, sB[i], sA[i])
		}
	}
	for i := range mA {
		if mA[i] != mB[i] {
			t.Fatalf("%s: sum[%d] = %d, want %d", name, i, mB[i], mA[i])
		}
	}
	if wantHalf {
		for i := range hA {
			if hA[i] != hB[i] {
				t.Fatalf("%s: half[%d] = %d, want %d", name, i, hB[i], hA[i])
			}
		}
	}
}
