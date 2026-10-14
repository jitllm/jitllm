package backend_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestMatVecPickMatchesTheReference holds MatVec's PickOf form -- output row
// r is the weight's row pick[r] -- to a float64 reference over the formats
// whose d plane differs (one row a word, two, four), picks that are unsorted,
// repeated, odd and even, and pick slots past the matrix (the label pick's
// padding, which read its last row). A pick that reads the wrong row, or the
// wrong row's scale (the d plane addressed by the output row), misses the
// reference by the rows' difference: that mutation fails it.
func TestMatVecPickMatchesTheReference(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const rows, k, npick = 1024, 512, 64
	ran := 0
	for _, d := range devs {
		defer d.Close()
		for _, q := range []kernels.Quant{kernels.Q8_0, kernels.Q4_K, kernels.Q6_K, kernels.Q4_0, kernels.Q5_K, kernels.Q3_K} {
			rng := rand.New(rand.NewSource(int64(q) + 7))
			raw := rawWeights(q, rows, k, rng)
			qs, dw, scw, err := kernels.PackWeights(q, raw, rows, k)
			if err != nil {
				t.Fatal(err)
			}
			if len(scw) == 0 {
				scw = []uint32{0}
			}
			wf := make([]float32, rows*k)
			if err := quant.Dequant32(ggufOf[q], raw, wf); err != nil {
				t.Fatal(err)
			}
			x := make([]float32, k)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}
			av, as, asum, err := kernels.PackActivations(x)
			if err != nil {
				t.Fatal(err)
			}
			pick := make([]uint32, npick)
			for i := range pick {
				pick[i] = uint32(rng.Intn(rows))
			}
			pick[1], pick[2], pick[3] = pick[0], 0, rows-1 // repeated, first, last
			pick[npick-1] = rows + 5                       // padding: reads the last row
			ker, err := kernels.MatVec(kernels.MatVecShape{T: q, K: k, Rows: npick, Split: 1, PickOf: rows})
			if err != nil {
				t.Fatalf("%v: %v", q, err)
			}
			kern, err := d.Compile(ker)
			if err != nil {
				t.Fatalf("%s %v: %v", d.API(), q, err)
			}
			g := newGPU(t, d)
			bA, bAX := g.up(u32bytes(av)), g.up(f32bytes(append(append([]float32{}, as...), asum...)))
			pb := make([]byte, 4*npick)
			for i, v := range pick {
				binary.LittleEndian.PutUint32(pb[4*i:], v)
			}
			out := g.up(f32bytes(nanFill(npick)))
			if err := kern.Launch((npick+127)/128, 128, g.up(u32bytes(qs)), g.up(u32bytes(dw)), g.up(u32bytes(scw)),
				bA, bAX, out, g.up(pb)); err != nil {
				t.Fatal(err)
			}
			got := make([]byte, 4*npick)
			if err := out.Read(got); err != nil {
				t.Fatal(err)
			}
			var se, sy float64
			for i, id := range pick {
				r := min(int(id), rows-1)
				var ref float64
				for j := 0; j < k; j++ {
					ref += float64(wf[r*k+j]) * float64(x[j])
				}
				v := float64(math.Float32frombits(binary.LittleEndian.Uint32(got[4*i:])))
				if math.IsNaN(v) || math.IsInf(v, 0) {
					t.Fatalf("%s %v: pick %d (row %d) is %v -- never written", d.API(), q, i, r, v)
				}
				se += (v - ref) * (v - ref)
				sy += ref * ref
			}
			nm := se / sy
			t.Logf("%s %v: %d picked rows of %d, NMSE %.3e", d.API(), q, npick, rows, nm)
			if nm > 1e-4 {
				t.Fatalf("%s %v: the picked rows read NMSE %.3e against the reference", d.API(), q, nm)
			}
			g.free()
			kern.Close()
			ran++
		}
	}
	if ran == 0 {
		t.Skip("no backend ran the kernel")
	}
}
