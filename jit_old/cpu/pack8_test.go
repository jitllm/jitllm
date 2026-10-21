//go:build amd64 && linux

package cpu

import (
	"github.com/jitllm/jitllm/internal/oracle"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

func packVsReference(t *testing.T, wt quant.Type, rows, k int) {
	if ggufDeclined(t, wt) {
		return
	}
	nb := k / 32
	code, err := EmitInterleaved(wt, rows, nb)
	if err != nil {
		t.Fatal(err)
	}
	kern := mustMap(t, code)
	defer kern.Close()

	const nrows = 64
	rng := rand.New(rand.NewSource(int64(rows*77 + k)))
	packed, exact := buildWeights(rng, wt, nrows, k)
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	q := make([]int8, k)
	pairs := make([]float32, 2*nb)
	if err := oracle.QuantizeQ8(q, pairs, x, BiasC(wt)); err != nil {
		t.Fatal(err)
	}
	konst := KernelConst(wt)
	out := make([]float32, nrows)
	args := Args{Out: &out[0], W: &packed[0], A: &q[0], AScale: &pairs[0],
		Rows: int64(nrows / rows), K: int64(nb),
		RowStr: int64(RowBytes(wt, k)), Scr: &konst[0]}
	kern.Call(&args)

	var sse, sy2 float64
	for r := 0; r < nrows; r++ {
		var want float64
		for b := 0; b < nb; b++ {
			dx := float64(pairs[2*b])
			for i := 0; i < 32; i++ {
				want += exact[r][b*32+i] * float64(q[b*32+i]) * dx
			}
		}
		d := float64(out[r]) - want
		sse += d * d
		sy2 += want * want
	}
	if nmse := sse / sy2; nmse > 1e-10 || math.IsNaN(nmse) {
		t.Errorf("%s x%d k=%d: NMSE %.3e", wt, rows, k, nmse)
	}
	t.Logf("%s x%d k=%d: %d bytes, NMSE ok", wt, rows, k, len(code))
}

func TestPack8MatchesReference(t *testing.T) {
	for _, wt := range []quant.Type{quant.Q4_0, quant.Q8_0} {
		for _, rows := range []int{2, 4, 8} {
			for _, k := range []int{2048, 16384} {
				packVsReference(t, wt, rows, k)
			}
		}
	}
}
