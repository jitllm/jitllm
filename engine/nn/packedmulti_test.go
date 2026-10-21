package nn

import (
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// packSheet builds one valid packed weight of (nrows, k). Scales vary per block
// (see TestPackedMatVecTailAgrees), and every sheet gets its own seed so a
// batch that reads another expert's bytes cannot agree with the loop by
// accident.
func packSheet(t testing.TB, gt quant.Type, nrows, k, seed int) *Packed {
	t.Helper()
	rng := rand.New(rand.NewSource(int64(seed)))
	q, _ := kernels.QuantOf(gt)
	nb := uint64(nrows*k) / gt.BlockElems()
	bb := gt.BlockBytes()
	src := make([]byte, nb*bb)
	for i := range src {
		src[i] = byte(rng.Intn(256))
	}
	if !quant.PlantScales(gt, src, seed) {
		panic("no scale planter for " + gt.String() + " -- its blocks would carry random scales")
	}
	qs, d, sc, err := kernels.PackWeights(q, src, nrows, k)
	if err != nil {
		t.Fatal(err)
	}
	return &Packed{QS: u32b(qs), D: u32b(d), SC: u32b(sc)}
}

// poison fills a result buffer with something no correct answer contains. The
// packed kernels accumulate and the caller owns the zero, which a zeroed
// buffer would hide.
func poison(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(1e3 * (float64(i%7) - 3))
	}
	return v
}

// TestMatVecPackedMultiMatchesTheLoopExactly is bit equality, not an NMSE
// bound, because MatVecPackedMulti changes dispatch and not arithmetic. It
// fails against dropping the per-matrix row/stride re-read and against assuming
// a worker's range never spans matrices.
func TestMatVecPackedMultiMatchesTheLoopExactly(t *testing.T) {
	type shape struct{ m, rows, k int }
	// The real mixtures' own shapes: gate/up is (NFFNExp, NEmbd) and down is
	// (NEmbd, NFFNExp).
	shapes := []shape{
		{8, 1024, 2048}, // olmoe-1b-7b gate/up, top-8
		{8, 2048, 1024}, // olmoe-1b-7b down
		{4, 768, 1024},  // Qwen3-MoE-4x0.6B gate/up, top-4
		{2, 16, 256},    // the smallest batch and one fused group
	}
	maxK, maxRows := 0, 0
	for _, sh := range shapes {
		maxK, maxRows = max(maxK, sh.k), max(maxRows, sh.rows)
	}
	ran := 0
	for _, gt := range quant.PackedTypes {
		// Both batched entry points are built on the fused kernel; a format
		// without one is logged and skipped, and requireRan below fails if
		// none qualified where one exists.
		if !cpu.PackedFusedSupported(gt) {
			t.Logf("%s: no FUSED packed kernel on this architecture", gt)
			continue
		}
		// The JIT is sized like a model's, not like the case.
		j := NewJIT(maxK, maxRows, []quant.Type{gt}, WithTune(TuneOff), WithMixtureBatch(1), WithGEMMExact(true)) // exact (the float forms), and batched on every host
		if j == nil {
			t.Fatalf("%s: no JIT", gt)
		}
		for _, sh := range shapes {
			if sh.k%int(gt.BlockElems()) != 0 {
				continue
			}
			ps := make([]*Packed, sh.m)
			for i := range ps {
				ps[i] = packSheet(t, gt, sh.rows, sh.k, i+1)
			}
			rng := rand.New(rand.NewSource(int64(sh.rows)))
			x := make([]float32, sh.k)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}
			// The loop this replaces, one region per matrix.
			want := make([][]float32, sh.m)
			for i := range want {
				want[i] = poison(sh.rows)
				if !j.MatVecPacked(want[i], gt, ps[i], x, sh.rows, sh.k) {
					t.Fatalf("%s %v: MatVecPacked declined", gt, sh)
				}
			}
			got := make([][]float32, sh.m)
			for i := range got {
				got[i] = poison(sh.rows)
			}
			// A false here would mean the batch never ran and the comparison
			// proves nothing.
			if !j.MatVecPackedMulti(got, gt, ps, x, sh.rows, sh.k) {
				t.Fatalf("%s m=%d %dx%d: MatVecPackedMulti DECLINED -- the batch "+
					"never ran", gt, sh.m, sh.rows, sh.k)
			}
			for i := range got {
				for r := 0; r < sh.rows; r++ {
					if got[i][r] != want[i][r] {
						t.Fatalf("%s m=%d %dx%d: matrix %d row %d: batch %v, loop %v",
							gt, sh.m, sh.rows, sh.k, i, r, got[i][r], want[i][r])
					}
				}
			}
			ran++
		}
		j.Close()
	}
	requireRan(t, ran)
	t.Logf("%d (format, shape) batches bit-identical to the per-matrix loop", ran)
}

// TestMatVecPackedMultiDeclinesRatherThanGuessing pins the shapes it must not
// serve, because the caller keeps its loop on a false and a wrong true would be
// silent.
func TestMatVecPackedMultiDeclinesRatherThanGuessing(t *testing.T) {
	gt := quant.Q4_K
	if !cpu.PackedFusedSupported(gt) {
		t.Skipf("%s: no FUSED packed kernel on this architecture", gt)
	}
	j := NewJIT(1024, 512, []quant.Type{gt}, WithMixtureBatch(1))
	if j == nil {
		t.Fatal("no JIT")
	}
	defer j.Close()
	const rows, k = 512, 1024
	ps := []*Packed{packSheet(t, gt, rows, k, 1), packSheet(t, gt, rows, k, 2)}
	outs := [][]float32{poison(rows), poison(rows)}
	x := make([]float32, k)

	// One matrix is not a batch: MatVecPacked already serves it.
	if j.MatVecPackedMulti(outs[:1], gt, ps[:1], x, rows, k) {
		t.Error("m=1 was served; it must fall through to MatVecPacked")
	}
	// A row count the fused kernel's group does not divide.
	if rows%cpu.PackedFusedGroup == 0 {
		if j.MatVecPackedMulti(outs, gt, ps, x, rows+1, k) {
			t.Errorf("rows=%d is not a multiple of %d and was served",
				rows+1, cpu.PackedFusedGroup)
		}
	}
	// Mismatched lengths, a nil sheet, and a short output.
	if j.MatVecPackedMulti(outs[:1], gt, ps, x, rows, k) {
		t.Error("len(outs) != len(ps) was served")
	}
	if j.MatVecPackedMulti(outs, gt, []*Packed{ps[0], nil}, x, rows, k) {
		t.Error("a nil sheet was served")
	}
	if j.MatVecPackedMulti([][]float32{outs[0], outs[1][:rows-1]}, gt, ps, x, rows, k) {
		t.Error("a short output was served")
	}
}

// TestMatVecPackedGatherMatchesTheLoopExactly is the down projection's twin of
// the gate above: m matrices with m different activations, bit-identical to
// the per-matrix loop. The activations are strided with slack between, as the
// caller holds them, so indexing at mi*k instead of mi*xstride, or pointing
// every matrix at activation 0, fails.
func TestMatVecPackedGatherMatchesTheLoopExactly(t *testing.T) {
	type shape struct{ m, rows, k int }
	shapes := []shape{
		{8, 2048, 1024}, // olmoe-1b-7b down, top-8
		{4, 1024, 768},  // Qwen3-MoE-shaped down
		{2, 16, 256},    // the smallest batch and one fused group
	}
	maxK, maxRows := 0, 0
	for _, sh := range shapes {
		maxK, maxRows = max(maxK, sh.k), max(maxRows, sh.rows)
	}
	ran := 0
	for _, gt := range quant.PackedTypes {
		if !cpu.PackedFusedSupported(gt) {
			t.Logf("%s: no FUSED packed kernel on this architecture", gt)
			continue
		}
		j := NewJIT(maxK, maxRows, []quant.Type{gt}, WithTune(TuneOff), WithMixtureBatch(1), WithGEMMExact(true)) // exact (the float forms), and batched on every host
		if j == nil {
			t.Fatalf("%s: no JIT", gt)
		}
		for _, sh := range shapes {
			if sh.k%int(gt.BlockElems()) != 0 {
				continue
			}
			// A stride wider than k, with junk in the slack, so an off-by-one
			// in the activation addressing cannot read zeros and pass.
			stride := sh.k + ElemLanes
			rng := rand.New(rand.NewSource(int64(sh.rows)))
			x := make([]float32, (sh.m-1)*stride+sh.k)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}
			ps := make([]*Packed, sh.m)
			for i := range ps {
				ps[i] = packSheet(t, gt, sh.rows, sh.k, i+1)
			}
			want := make([][]float32, sh.m)
			for i := range want {
				want[i] = poison(sh.rows)
				// Each matrix against its own activation, one call each.
				j.NewInput()
				if !j.MatVecPacked(want[i], gt, ps[i], x[i*stride:i*stride+sh.k], sh.rows, sh.k) {
					t.Fatalf("%s %v: MatVecPacked declined", gt, sh)
				}
			}
			got := make([][]float32, sh.m)
			for i := range got {
				got[i] = poison(sh.rows)
			}
			if !j.MatVecPackedGather(got, gt, ps, x, stride, sh.rows, sh.k) {
				t.Fatalf("%s m=%d %dx%d: MatVecPackedGather DECLINED -- the "+
					"batch never ran", gt, sh.m, sh.rows, sh.k)
			}
			for i := range got {
				for r := 0; r < sh.rows; r++ {
					if got[i][r] != want[i][r] {
						t.Fatalf("%s m=%d %dx%d: matrix %d row %d: gather %v, loop %v",
							gt, sh.m, sh.rows, sh.k, i, r, got[i][r], want[i][r])
					}
				}
			}
			ran++
		}
		j.Close()
	}
	requireRan(t, ran)
	t.Logf("%d (format, shape) gathers bit-identical to the per-matrix loop", ran)
}

// TestMatVecPackedGatherKeepsTheSingleActivationCache pins the property the
// shared expert depends on: a gather must not disturb MatVecPacked's own
// activation cache, which is a different buffer (f.q against f.mq).
func TestMatVecPackedGatherKeepsTheSingleActivationCache(t *testing.T) {
	gt := quant.Q4_K
	if !cpu.PackedFusedSupported(gt) {
		t.Skipf("%s: no FUSED packed kernel on this architecture", gt)
	}
	const rows, k = 512, 1024
	j := NewJIT(k, rows, []quant.Type{gt}, WithTune(TuneOff), WithMixtureBatch(1), WithGEMMExact(true)) // exact (the float forms), and batched on every host
	if j == nil {
		t.Fatal("no JIT")
	}
	defer j.Close()
	h := make([]float32, k)
	rng := rand.New(rand.NewSource(7))
	for i := range h {
		h[i] = float32(rng.NormFloat64())
	}
	p := packSheet(t, gt, rows, k, 1)
	first := poison(rows)
	if !j.MatVecPacked(first, gt, p, h, rows, k) {
		t.Fatal("declined")
	}
	stride := k + ElemLanes
	x := make([]float32, stride+k)
	ps := []*Packed{packSheet(t, gt, rows, k, 2), packSheet(t, gt, rows, k, 3)}
	outs := [][]float32{poison(rows), poison(rows)}
	if !j.MatVecPackedGather(outs, gt, ps, x, stride, rows, k) {
		t.Fatal("gather declined")
	}
	again := poison(rows)
	if !j.MatVecPacked(again, gt, p, h, rows, k) {
		t.Fatal("declined on the second call")
	}
	for r := 0; r < rows; r++ {
		if first[r] != again[r] {
			t.Fatalf("row %d: %v before the gather, %v after", r, first[r], again[r])
		}
	}
}

// requireRan separates "nothing ran because the capability is absent" from
// "nothing ran because the gate is broken", which look identical in a count:
// it skips, saying why, only when no format has a fused packed kernel, and
// fails when one exists and nothing ran.
func requireRan(t *testing.T, ran int) {
	t.Helper()
	if ran > 0 {
		return
	}
	for _, gt := range quant.PackedTypes {
		if cpu.PackedFusedSupported(gt) {
			t.Fatalf("%s HAS a fused packed kernel and no shape ran: this gate "+
				"proved nothing", gt)
		}
	}
	t.Skip("no FUSED packed kernel for any format on this architecture, so the " +
		"batched mixture dispatch does not exist here -- nothing to compare")
}

// TestMatVecPackedGatherActIsTheActivationThenTheGather: the gated
// activation taken inside the gather's quantize region is ActMul32JIT over
// each activation and then MatVecPackedGather, bit for bit, on every packed
// format and every gated activation; the slack past each activation is left
// as it was. A wrong up stride fails it.
func TestMatVecPackedGatherActIsTheActivationThenTheGather(t *testing.T) {
	const m, rows, k = 4, 256, 512
	stride := k + ElemLanes
	ran := 0
	for _, gt := range quant.PackedTypes {
		if !cpu.PackedFusedSupported(gt) || k%int(gt.BlockElems()) != 0 {
			continue
		}
		j := NewJIT(k, rows, []quant.Type{gt}, WithTune(TuneOff), WithMixtureBatch(1), WithGEMMExact(true))
		if j == nil {
			t.Fatalf("%s: no JIT", gt)
		}
		ps := make([]*Packed, m)
		for i := range ps {
			ps[i] = packSheet(t, gt, rows, k, i+1)
		}
		for _, act := range []ActKind{ActSiLU, ActGELU} {
			if gatedIndex(act) < 0 {
				continue
			}
			rng := rand.New(rand.NewSource(int64(act) + 7))
			x := make([]float32, (m-1)*stride+k)
			up := make([]float32, len(x))
			for i := range x {
				x[i], up[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
			}
			x2 := append([]float32(nil), x...)
			for i := 0; i < m; i++ {
				ActMul32JIT(x2[i*stride:i*stride+k], up[i*stride:i*stride+k], act)
			}
			want, got := make([][]float32, m), make([][]float32, m)
			for i := range want {
				want[i], got[i] = poison(rows), poison(rows)
			}
			if !j.MatVecPackedGather(want, gt, ps, x2, stride, rows, k) {
				t.Fatalf("%s: MatVecPackedGather declined", gt)
			}
			if !j.MatVecPackedGatherAct(got, gt, ps, x, up, act, stride, rows, k) {
				t.Fatalf("%s %v: MatVecPackedGatherAct declined", gt, act)
			}
			for i := range got {
				for r := range got[i] {
					if got[i][r] != want[i][r] {
						t.Fatalf("%s %v: matrix %d row %d: fused %v, apart %v", gt, act, i, r, got[i][r], want[i][r])
					}
				}
			}
			if !slices.Equal(x, x2) {
				t.Fatalf("%s %v: the activations in place differ from the activation kernel's", gt, act)
			}
			ran++
		}
		j.Close()
	}
	requireRan(t, ran)
	t.Logf("%d (format, activation) fused gathers bit-identical", ran)
}
