package nn

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/cpu"
)

// TestFusedKSplitIsBitIdentical walks the fused decode matvec's k in pieces --
// one super-block per call, and the budget's own piece at a k large enough to
// need one -- and demands the same bits as the whole-k call. The kernel adds
// each super-block into Out in k order, so pieces perform the same adds; what a
// piece can get wrong is its hand-computed start offsets.
//
// The bank case (Row > 0 inside a Stride-row buffer) is a mixture's expert,
// whose planes are interleaved with the other experts'.
func TestFusedKSplitIsBitIdentical(t *testing.T) {
	ran := 0
	for _, gt := range quant.PackedTypes {
		if !cpu.PackedFusedSupported(gt) {
			continue
		}
		step := kStep(gt)
		grp := grpOf(gt)
		for _, sh := range []struct{ rows, k, bankRow int }{
			{grp, 3 * step, 0},
			{5 * grp, 7 * step, 0},
			{64, 4 * step, 64},
			{2048, (14336 / step) * step, 0}, // large enough for the budget's own piece
		} {
			if sh.k < 2*step {
				continue
			}
			stride := sh.rows + sh.bankRow
			pk := packOne(t, gt, stride, sh.k)
			pk.Row, pk.Stride = sh.bankRow, stride
			x := prevnniAct(sh.k)
			run := func(ks int) []float32 {
				j := NewJIT(sh.k, stride, []quant.Type{gt}, WithFusedKSplit(ks), WithFusedKSlices(-1))
				if j == nil {
					t.Fatal("no JIT")
				}
				defer j.Close()
				if j.packedFused[gt] == nil {
					t.Fatalf("%s: PackedFusedSupported but the JIT built no fused kernel", gt)
				}
				if ks == 0 && sh.rows >= 2048 {
					per := max(1, (sh.rows/grp+j.pool.N()-1)/j.pool.N())
					if j.fusedPiece(gt, sh.k, per*grp) >= sh.k/step {
						t.Logf("%s %dx%d: the budget did not split k at %d workers", gt, sh.rows, sh.k, j.pool.N())
					}
				}
				out := make([]float32, sh.rows+1)
				out[sh.rows] = 12345
				if !j.MatVecPacked(out, gt, pk, x, sh.rows, sh.k) {
					t.Fatalf("%s %dx%d: MatVecPacked declined", gt, sh.rows, sh.k)
				}
				if out[sh.rows] != 12345 {
					t.Fatalf("%s %dx%d: wrote past the rows", gt, sh.rows, sh.k)
				}
				return out[:sh.rows]
			}
			want := run(-1)
			for _, c := range []struct{ ks int }{{1}, {2}, {0}} {
				got := run(c.ks)
				for r := range want {
					if math.Float32bits(got[r]) != math.Float32bits(want[r]) || math.IsNaN(float64(want[r])) {
						t.Fatalf("%s %dx%d bank row %d, piece %d: row %d is %v, whole k gives %v",
							gt, sh.rows, sh.k, sh.bankRow, c.ks, r, got[r], want[r])
					}
				}
			}
			ran++
		}
	}
	if ran == 0 {
		if cpu.PackedFusedSupported(quant.Q4_K) {
			t.Fatal("a fused kernel exists and no shape ran -- this gate proved nothing")
		}
		t.Skip("no fused packed kernel on this host, so there is no k to split")
	}
	t.Logf("%d (format, shape) pair(s) bit-identical under every k piece", ran)
}

// TestFusedKSplitReachesTheMixturePaths holds MatVecPackedMulti and
// MatVecPackedGather -- a mixture's gate/up and down projections, which share
// the k walk -- to the same bits under every piece, on experts that are rows
// of one Stride-row bank, as a real mixture's are.
func TestFusedKSplitReachesTheMixturePaths(t *testing.T) {
	ran := 0
	for _, gt := range quant.PackedTypes {
		if !cpu.PackedFusedSupported(gt) {
			continue
		}
		step := kStep(gt)
		grp := grpOf(gt)
		const m = 3
		rows, k := 4*grp, 5*step
		if k < 2*step {
			continue
		}
		bank := packOne(t, gt, m*rows, k)
		ps := make([]*Packed, m)
		for e := range ps {
			cp := *bank
			cp.Row, cp.Stride = e*rows, m*rows
			ps[e] = &cp
		}
		x := prevnniAct(m * k)
		run := func(ks int, gather bool) [][]float32 {
			j := NewJIT(k, m*rows, []quant.Type{gt}, WithFusedKSplit(ks), WithMixtureBatch(1))
			if j == nil {
				t.Fatal("no JIT")
			}
			defer j.Close()
			outs := make([][]float32, m)
			for e := range outs {
				outs[e] = make([]float32, rows)
			}
			ok := false
			if gather {
				ok = j.MatVecPackedGather(outs, gt, ps, x, k, rows, k)
			} else {
				ok = j.MatVecPackedMulti(outs, gt, ps, x[:k], rows, k)
			}
			if !ok {
				t.Fatalf("%s: the batched path declined (gather %v)", gt, gather)
			}
			return outs
		}
		for _, gather := range []bool{false, true} {
			want := run(-1, gather)
			for _, ks := range []int{1, 2, 0} {
				got := run(ks, gather)
				for e := range want {
					for r := range want[e] {
						if math.Float32bits(got[e][r]) != math.Float32bits(want[e][r]) || math.IsNaN(float64(want[e][r])) {
							t.Fatalf("%s gather %v piece %d: expert %d row %d is %v, whole k gives %v",
								gt, gather, ks, e, r, got[e][r], want[e][r])
						}
					}
				}
			}
			ran++
		}
	}
	if ran == 0 {
		if cpu.PackedFusedSupported(quant.Q4_K) {
			t.Fatal("a fused kernel exists and no format ran -- this gate proved nothing")
		}
		t.Skip("no fused packed kernel on this host")
	}
}

// TestFusedKSlicesMatchWholeK splits a matrix over k across workers -- the
// short-rows, long-k shape fusedSlices takes -- and holds the summed partials
// to the whole-k call within a reassociation bound (measured worst ~4e-13; a
// slice at the wrong super-block reads NMSE ~1).
func TestFusedKSlicesMatchWholeK(t *testing.T) {
	ran, worst := 0, 0.0
	for _, gt := range quant.PackedTypes {
		if !cpu.PackedFusedSupported(gt) {
			continue
		}
		step := kStep(gt)
		grp := grpOf(gt)
		for _, sh := range []struct{ rows, k, bankRow int }{
			{4 * grp, 7 * step, 0},
			{64, 9 * step, 64},
			{1024, (14336 / step) * step, 0},
		} {
			stride := sh.rows + sh.bankRow
			pk := packOne(t, gt, stride, sh.k)
			pk.Row, pk.Stride = sh.bankRow, stride
			x := prevnniAct(sh.k)
			run := func(slices int) []float32 {
				j := NewJIT(sh.k, stride, []quant.Type{gt}, WithFusedKSlices(slices))
				if j == nil {
					t.Fatal("no JIT")
				}
				defer j.Close()
				if slices > 1 && j.pool.N() < 2 {
					t.Skip("one worker: nothing to slice across")
				}
				out := make([]float32, sh.rows+1)
				out[sh.rows] = 12345
				if !j.MatVecPacked(out, gt, pk, x, sh.rows, sh.k) {
					t.Fatalf("%s %dx%d: MatVecPacked declined", gt, sh.rows, sh.k)
				}
				if out[sh.rows] != 12345 {
					t.Fatalf("%s %dx%d: wrote past the rows", gt, sh.rows, sh.k)
				}
				return out[:sh.rows]
			}
			want := run(-1)
			for _, sl := range []int{2, 3, 0} {
				got := run(sl)
				var num, den float64
				for r := range want {
					d := float64(got[r]) - float64(want[r])
					num += d * d
					den += float64(want[r]) * float64(want[r])
				}
				worst = max(worst, num/den)
				if !(num/den <= 1e-10) {
					t.Fatalf("%s %dx%d bank row %d, %d slices: NMSE %.3e against whole k (row 0 %v, want %v)",
						gt, sh.rows, sh.k, sh.bankRow, sl, num/den, got[0], want[0])
				}
			}
			ran++
		}
	}
	if ran == 0 {
		if cpu.PackedFusedSupported(quant.Q4_K) {
			t.Fatal("a fused kernel exists and no format ran -- this gate proved nothing")
		}
		t.Skip("no fused packed kernel on this host")
	}
	t.Logf("%d (format, shape) pair(s) within NMSE 1e-10 under 2, 3 and the default slices; worst %.3e", ran, worst)
}

// kStep is the k granule the shapes here are built from: one outer step of
// the kernel, widened to the source block when the packer reads whole blocks
// wider than that step (TQ2_0's 256 against its 32-wide step).
func kStep(t quant.Type) int {
	return max(cpu.PackedOuterElems(t), int(t.BlockElems()))
}
