//go:build amd64 || arm64

package nn

import (
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// TestChunkBytesCapNarrowsAndKeepsTheAnswer is the gate on WithChunkBytes.
//
// It asserts both halves: the answer is bit-identical by construction, so
// equality alone passes when the cap never reached the chunk, and a counter
// alone says nothing about the answer.
func TestChunkBytesCapNarrowsAndKeepsTheAnswer(t *testing.T) {
	const nrows, k, ntok = 3072, 768, 8
	gt := quant.Q4_K
	if !cpu.PackedSupported(gt) {
		t.Fatalf("%s lost its packed kernel", gt)
	}
	q, _ := kernels.QuantOf(gt)
	rng := rand.New(rand.NewSource(11))
	nb := uint64(nrows*k) / gt.BlockElems()
	bb := gt.BlockBytes()
	src := make([]byte, nb*bb)
	for i := range src {
		src[i] = byte(rng.Intn(256))
	}
	// A uniformly random f16 is Inf or NaN about one time in 32, and a poisoned
	// oracle reports as a pass.
	for b := uint64(0); b < nb; b++ {
		for _, o := range []int{0, 2} {
			src[b*bb+uint64(o)], src[b*bb+uint64(o)+1] = 0x00, 0x3C // 1.0
		}
	}
	qs, d, sc, err := kernels.PackWeights(q, src, nrows, k)
	if err != nil {
		t.Fatal(err)
	}
	pk := &Packed{QS: u32b(qs), D: u32b(d), SC: u32b(sc)}
	x := make([]float32, ntok*k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	perRow := (len(pk.QS) + len(pk.D) + len(pk.SC)) / nrows

	run := func(cap int) ([]float32, int) {
		// GEMM off: the cap is the tiled path's locality instrument (packedChunk),
		// and the weight-stationary GEMM reads each weight once per token block
		// whatever the chunk, so it has no cap to honour.
		opts := []Option{WithTune(TuneOff), WithQuietTuner(true), WithGEMMTokens(-1)}
		if cap > 0 {
			opts = append(opts, WithChunkBytes(cap))
		}
		j := NewJIT(k, nrows, []quant.Type{gt}, opts...)
		if j == nil {
			t.Skip("no JIT on this build")
		}
		defer j.Close()
		out := make([]float32, ntok*nrows)
		if !j.MatMulPacked(out, gt, pk, x, nrows, k, ntok) {
			t.Skipf("the batch declined on this architecture -- %s has no tiled kernel here", gt)
		}
		return out, j.BatchedChunkRows()
	}

	base, baseRows := run(0)
	if baseRows <= 0 {
		t.Fatalf("the uncapped run reported %d chunk rows, so there is nothing "+
			"for a cap to narrow and this gate would prove nothing", baseRows)
	}
	// A cap of a quarter the uncapped slice must land at roughly a quarter the
	// rows. Asking for "smaller" alone would pass on a cap that clipped to one
	// group.
	small := perRow * baseRows / 4
	capped, capRows := run(small)
	t.Logf("uncapped %d rows (%d B/row, %.2f MiB); capped at %.2f MiB -> %d rows",
		baseRows, perRow, float64(baseRows*perRow)/(1<<20),
		float64(small)/(1<<20), capRows)

	if capRows >= baseRows {
		t.Errorf("a cap of %d bytes left the chunk at %d rows (uncapped %d) -- "+
			"the cap did not reach packedChunk", small, capRows, baseRows)
	}
	if got := capRows * perRow; got > small {
		t.Errorf("the capped chunk covers %d bytes, over its %d-byte cap", got, small)
	}
	// A cap above the whole weight must change nothing, which is the half
	// that catches a cap applied as a floor or an off-by-one that always clips.
	if _, wide := run(perRow * nrows * 4); wide != baseRows {
		t.Errorf("a cap of four whole weights moved the chunk to %d rows, want "+
			"the uncapped %d -- the cap is narrowing when it has no reason to",
			wide, baseRows)
	}
	for i := range base {
		if base[i] != capped[i] {
			t.Fatalf("element %d: uncapped %v, capped %v -- the cap moved a pool "+
				"boundary and changed the ARITHMETIC, which it must not",
				i, base[i], capped[i])
		}
	}
	// The cap must not reach a decode matvec, which reads each weight once
	// and has no reuse to keep: a narrower chunk there is pure added dispatch.
	dec := NewJIT(k, nrows, []quant.Type{gt}, WithTune(TuneOff), WithQuietTuner(true),
		WithChunkBytes(small))
	if dec == nil {
		t.Skip("no JIT on this build")
	}
	defer dec.Close()
	groups := nrows / grpOf(gt)
	want := packedChunk(dec, groups, gt, k, grpOf(gt))
	if got := dec.batchedChunk(groups, gt, k, grpOf(gt), 1, pk, nrows); got != want {
		t.Errorf("at ntok=1 the cap narrowed the chunk to %d groups, want the "+
			"uncapped %d -- a decode matvec reads each weight once and has no "+
			"reuse to keep, so this is pure added dispatch", got, want)
	}
}
