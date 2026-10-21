//go:build amd64 || arm64

package cpu

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
)

// TestGEMMMatchesReference holds every tile shape to a float64 evaluation of
// the same int8 activations and per-block scales the kernel computes with; only
// summation order may differ. A mis-indexed row, token or k-group (all three
// addressed by baked displacement) yields a plausible number, not garbage.
func TestGEMMMatchesReference(t *testing.T) {
	for _, wt := range []quant.Type{quant.Q8_0, quant.Q4_0, quant.Q4_K, quant.Q6_K, quant.Q3_K, quant.Q5_K} {
		for _, tile := range [][2]int{{1, 1}, {2, 1}, {4, 1}, {6, 1}, {2, 2}, {3, 2}, {6, 2}, {2, 3}, {3, 4}, {2, 6}} {
			mr, nr := tile[0], tile[1]
			ks := []int{32, 256, 2048}
			if wt.BlockElems() == 256 {
				ks = []int{256, 512, 2048}
			}
			for _, k := range ks {
				t.Run(wt.String()+"/"+name(mr, nr, k), func(t *testing.T) { gemmVsReference(t, wt, mr, nr, k) })
			}
		}
	}
}

func name(mr, nr, k int) string {
	return itoa(mr) + "x" + itoa(nr) + "/k" + itoa(k)
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var b [8]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	return string(b[i:])
}

func gemmVsReference(t *testing.T, wt quant.Type, mr, nr, k int) {
	gemmVsReferenceWin(t, wt, mr, nr, k, Q8Block)
}

// gemmVsReferenceWin packs the activations at the same window the kernel was
// emitted for. Passing different ones is the violation the wide-window test
// exists to catch, and on arm64 it is a wrong answer rather than an error.
func gemmVsReferenceWin(t *testing.T, wt quant.Type, mr, nr, k, window int) {
	const groups = 3          // row groups, so the outer loop and its stride are exercised
	nbAct := k / 32           // activation scale blocks: always 32-wide
	nb := BlocksPerRow(wt, k) // weight blocks: 256-wide for the k-quants
	rows := mr * groups
	tok := GEMMTokens(nr)

	code, err := EmitGEMMWindow(wt, mr, nr, nb, window)
	if err != nil {
		t.Skipf("tile rejected: %v", err)
	}
	kern := mustMap(t, code)
	defer kern.Close()

	rng := rand.New(rand.NewSource(int64(mr*1013 + nr*17 + k)))
	packed, exact := buildWeights(rng, wt, rows, k)

	x := make([]float32, tok*k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	q := make([]int8, tok*k)
	scale := make([]float32, nbAct*tok)
	sum := make([]int32, nbAct*tok)
	half := make([]int32, 2*nbAct*tok)
	if err := PackGEMMActivationsRange(wt, q, scale, sum, half, x, tok, k, 0, tok, window); err != nil {
		t.Fatal(err)
	}
	konst := KernelConst(wt)
	out := make([]float32, rows*tok)
	scratch := make([]float32, GEMMScratchMax)

	args := Args{
		Out: &out[0], W: &packed[0], A: &q[0], AScale: &scale[0],
		Rows: int64(groups), K: int64(nb), RowStr: int64(RowBytes(wt, k)),
		Scr: &konst[0], Cols: int64(tok), AHalfSum: &half[0],
		OutStr: int64(tok * 4), ASum: &sum[0],
		Scratch: (*byte)(unsafe.Pointer(&scratch[0])),
	}
	kern.Call(&args)

	// Reference: the same arithmetic, in float64, from the dequantized weights
	// and the int8 activations the kernel was actually handed.
	var sse, sy2 float64
	for r := 0; r < rows; r++ {
		for n := 0; n < tok; n++ {
			var want float64
			for b := 0; b < nbAct; b++ {
				d := float64(scale[b*tok+n])
				for i := 0; i < 32; i++ {
					ki := b*32 + i
					qi := float64(q[((ki/4)*tok+n)*4+ki%4])
					want += exact[r][ki] * qi * d
				}
			}
			got := float64(out[r*tok+n])
			sse += (got - want) * (got - want)
			sy2 += want * want
		}
	}
	// A degenerate reference (all-zero or NaN) is a failure, not a pass: a
	// guard like `if sy2 > 0` would leave nmse at 0 and report a perfect
	// match.
	if math.IsNaN(sy2) || math.IsInf(sy2, 0) || sy2 <= 0 {
		t.Fatalf("%dx%d k=%d: reference is degenerate (sy2=%v) -- the comparison proves nothing",
			mr, nr, k, sy2)
	}
	nmse := sse / sy2
	if nmse > 1e-10 || math.IsNaN(nmse) {
		t.Errorf("%dx%d k=%d: NMSE %.3e exceeds 1e-10", mr, nr, k, nmse)
	}
	t.Logf("%dx%d k=%d rows=%d tok=%d: %d bytes, NMSE %.2e", mr, nr, k, rows, tok, len(code), nmse)
}
