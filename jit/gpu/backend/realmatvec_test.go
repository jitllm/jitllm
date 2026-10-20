package backend_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testmodels"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestMatVecRealWeights runs the GPU kernel over real model tensors of every
// supported format and holds it to NMSE < 1e-10 against a float64 evaluation of
// the same quantities.
//
// Real tensors cover the formats the synthetic test has no generator for
// (16-element sub-blocks, the second activation-sum granularity, two-plane
// payloads); quant.Dequant is the oracle either way.
func TestMatVecRealWeights(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend")
	}
	models := testmodels.Glob("*.gguf")
	if len(models) == 0 {
		t.Skip("MODEL MISSING: no *.gguf in " + testmodels.Dir() + " (set JITLLM_MODELS to the model directory) -- this gate proved nothing")
	}
	// Every packed format, derived from quant.PackedTypes rather than a
	// hand-written list that could drift.
	supported := map[quant.Type]mainkernels.Quant{}
	for _, t := range quant.PackedTypes {
		if q, ok := mainkernels.QuantOf(t); ok {
			supported[t] = q
		}
	}
	for _, d := range devs {
		defer d.Close()
		seen := map[quant.Type]int{}
		for _, path := range models {
			f, err := gguf.Open(path)
			if err != nil {
				continue
			}
			for i := range f.Tensors {
				ti := &f.Tensors[i]
				q, ok := supported[ti.Type]
				if !ok || len(ti.Dims) != 2 || seen[ti.Type] >= 1 {
					continue
				}
				k, all := int(ti.Dims[0]), int(ti.Dims[1])
				if k%q.Elems() != 0 || all < 64 {
					continue
				}
				seen[ti.Type]++
				// And the row tile, on every format: the tile's payload,
				// both scale planes and the secondary plane are read at a
				// compile-time displacement from one index, which the
				// two-plane formats and Q3_K's 16-element sub-block
				// exercise differently.
				for _, split := range []int{1, 4} {
					for _, rowt := range []int{1, 2, 4} {
						name := q.String() + "/split" + itoa(split) + "/rowt" + itoa(rowt)
						t.Run(d.API()+"/"+name, func(t *testing.T) {
							realCase(t, d, q, f.Bytes(ti), 64, all, k, split, rowt)
						})
					}
				}
			}
			f.Close()
		}
	}
}

func realCase(t *testing.T, d backend.Device, q mainkernels.Quant, w []byte, nrows, allRows, k, split, rowt int) {
	sub, _, _, _ := mainkernels.Layout(q)
	if (k/sub)%split != 0 {
		t.Skipf("split %d does not divide %d sub-blocks", split, k/sub)
	}
	rowBytes := len(w) / allRows
	w = w[:nrows*rowBytes]

	qs, dw, scw, err := mainkernels.PackWeights(q, w, nrows, k)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(11))
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64() / (0.2 + math.Abs(rng.NormFloat64())))
	}
	av, as, asum, err := mainkernels.PackActivations(x)
	if err != nil {
		t.Fatal(err)
	}
	ax := append(append([]float32{}, as...), asum...)

	ker, err := mainkernels.MatVec(mainkernels.MatVecShape{T: q, K: k, Rows: nrows, Split: split, Rowt: rowt})
	if err != nil {
		t.Fatal(err)
	}
	kern, err := d.Compile(ker)
	if err != nil {
		t.Fatal(err)
	}
	defer kern.Close()

	var bufs []backend.Buf
	defer func() {
		for _, b := range bufs {
			b.Free()
		}
	}()
	up := func(p []byte) backend.Buf {
		b, err := d.Alloc(len(p))
		if err != nil {
			t.Fatal(err)
		}
		if err := b.Write(p); err != nil {
			t.Fatal(err)
		}
		bufs = append(bufs, b)
		return b
	}
	if len(scw) == 0 {
		scw = []uint32{0}
	}
	bQS, bD, bSC := up(u32bytes(qs)), up(u32bytes(dw)), up(u32bytes(scw))
	bA, bAX := up(u32bytes(av)), up(f32bytes(ax))
	// RULE 13: poisoned, not zeroed. A tile row the kernel never writes reads
	// back as 0.0 out of a fresh allocation, which is a plausible dot product;
	// NaN is not, and it survives the Reduce that sums the split's segments.
	bPart := up(f32bytes(nanFill(nrows * split)))
	bOut := bPart
	if split > 1 {
		bOut = up(f32bytes(nanFill(nrows)))
	}
	const width = 128
	// The grid divides by the tile: surplus threads are clamped to the last
	// work item (RULE 13), so an oversized grid would silently recompute it.
	if err := kern.Launch(((nrows/rowt)*split+width-1)/width, width, bQS, bD, bSC, bA, bAX, bPart); err != nil {
		t.Fatal(err)
	}
	if split > 1 {
		rk, err := mainkernels.Reduce(nrows, split)
		if err != nil {
			t.Fatal(err)
		}
		rkern, err := d.Compile(rk)
		if err != nil {
			t.Fatal(err)
		}
		defer rkern.Close()
		if err := rkern.Launch((nrows+width-1)/width, width, bPart, bOut); err != nil {
			t.Fatal(err)
		}
	}
	raw := make([]byte, nrows*4)
	if err := bOut.Read(raw); err != nil {
		t.Fatal(err)
	}

	// Reference: gguf's weights against the same quantized activations.
	gt, ok := ggufOf[q] // derived from quant.PackedTypes; a copy here stopped at six
	if !ok {
		t.Fatalf("%s: no source type to dequantize the reference with", q)
	}
	ref := make([]float64, k)
	var sse, sy2 float64
	for r := 0; r < nrows; r++ {
		if err := quant.Dequant(gt, w[r*rowBytes:(r+1)*rowBytes], ref); err != nil {
			t.Fatal(err)
		}
		var want float64
		for b := 0; b < k/32; b++ {
			da := float64(as[b])
			for i := 0; i < 32; i++ {
				qi := int8(av[b*8+i/4] >> uint(8*(i%4)))
				want += ref[b*32+i] * float64(qi) * da
			}
		}
		got := float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[r*4:])))
		// The poison is checked before the bound and by row: NaN > bound is
		// false, and with a tile the unwritten row is the diagnosis.
		if math.IsNaN(got) || math.IsInf(got, 0) {
			t.Fatalf("%s rows=%d k=%d split=%d rowt=%d: row %d came back %v -- "+
				"the kernel never wrote it (output was poisoned, not zeroed)",
				q, nrows, k, split, rowt, r, got)
		}
		dd := got - want
		sse += dd * dd
		sy2 += want * want
	}
	nmse := 0.0
	if sy2 > 0 {
		nmse = sse / sy2
	}
	if nmse > 1e-10 || math.IsNaN(nmse) {
		t.Errorf("%s rows=%d k=%d split=%d rowt=%d: NMSE %.3e exceeds 1e-10",
			q, nrows, k, split, rowt, nmse)
	}
}

var _ = unsafe.Pointer(nil)
