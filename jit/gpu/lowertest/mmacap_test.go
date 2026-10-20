package lowertest

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
	"github.com/jitllm/jitllm/jit/gpu/msl"
	"github.com/jitllm/jitllm/jit/gpu/ptx"
	"github.com/jitllm/jitllm/jit/gpu/spirv"
)

// Which backends lower ir.OpMMA and the tile ops is a list in the repo, asserted
// once here, rather than a skipped subtest per (backend x shape x format) in
// the MMA sweeps, which would hide any real skip.
//
// ir.OpMMA is the fragment form (per-lane layout from MMAShape.Frags) and is
// ptx-only by design. The collective tile ops (opaque tiles loaded with a
// stride) lower on all three: spirv to cooperative matrices, msl to
// simdgroup_matrix, ptx to mma.sync over 8x8 atoms (ptx/tile.go). So a tile
// kernel reaches every card's matrix unit; the fragment form stays PTX's.
var (
	mmaImplemented = []string{"ptx"}
	mmaDeclined    = []string{"spirv", "msl"}

	tileImplemented = []string{"spirv", "msl", "ptx"}
	tileDeclined    = []string{}
)

// mmaProbe is the smallest kernel that contains an OpMMA: one tile, results
// stored so nothing is dead. Every lowerer sees the same one.
func mmaProbe(sh ir.MMAShape) *ir.Kernel {
	b := ir.New("mmaprobe", [3]int{32, 1, 1})
	pA := b.Param("pA", ir.U32)
	pB := b.Param("pB", ir.U32)
	// The output is the tile's own element type: an integer tile stores i32,
	// a float one f32, and a load or store must match its parameter.
	pOut := b.Param("pOut", sh.Elem())
	na, nb, nc := sh.Frags()
	a := make([]ir.Value, na)
	for i := range a {
		a[i] = b.Load(ir.U32, pA, b.Const(ir.U32, int64(i)), 0)
	}
	bb := make([]ir.Value, nb)
	for i := range bb {
		bb[i] = b.Load(ir.U32, pB, b.Const(ir.U32, int64(i)), 0)
	}
	c := make([]ir.Value, nc)
	for i := range c {
		c[i] = b.Const(sh.Elem(), 0)
	}
	d := b.MMA(sh, a, bb, c)
	for i, v := range d {
		b.Store(pOut, b.Const(ir.U32, int64(i)), v, 0)
	}
	return b.Done()
}

func TestMMACoversExactlyWhatItClaims(t *testing.T) {
	all := map[string]func(*ir.Kernel) error{
		"ptx":   func(k *ir.Kernel) error { _, e := ptx.Lower(k, "sm_86"); return e },
		"spirv": func(k *ir.Kernel) error { _, e := spirv.Emit(k); return e },
		"msl":   func(k *ir.Kernel) error { _, e := msl.Emit(k); return e },
	}
	// Every backend is accounted for, so a new one needs a decision here.
	seen := map[string]int{}
	for _, b := range mmaImplemented {
		seen[b]++
	}
	for _, b := range mmaDeclined {
		seen[b]++
	}
	if len(seen) != len(all) {
		t.Fatalf("%d backends lower kernels and this file accounts for %d; a backend "+
			"added without an entry above silently loses its MMA coverage", len(all), len(seen))
	}
	for b, c := range seen {
		if _, ok := all[b]; !ok {
			t.Errorf("%q is listed here and is not a backend", b)
		} else if c != 1 {
			t.Errorf("%s appears %d times across the implemented and declined lists", b, c)
		}
	}

	// The s8 shape the quantized matvec uses and the f16 one attention wants:
	// a backend could lower one and not the other.
	shapes := []ir.MMAShape{{M: 16, N: 8, K: 16, Kind: ir.MMAS8}, {M: 16, N: 8, K: 16, Kind: ir.MMAF16}}
	for _, sh := range shapes {
		k := mmaProbe(sh)
		if err := k.Validate(); err != nil {
			t.Fatalf("the probe itself does not validate: %v", err)
		}
		for _, b := range mmaImplemented {
			if err := all[b](k); err != nil {
				t.Errorf("%s is listed as implemented and refused %v: %v", b, sh.Kind, err)
			}
		}
		for _, b := range mmaDeclined {
			err := all[b](k)
			if err == nil {
				t.Errorf("%s lowered %v and is listed as declined; move it and turn "+
					"the MMA sweeps back on for it", b, sh.Kind)
				continue
			}
			// The refusal must say why, so a real constraint can be told from
			// work not done yet.
			if !strings.Contains(err.Error(), "matrix") {
				t.Errorf("%s: the refusal does not name what is missing: %v", b, err)
			}
			t.Logf("%s declines %v: %v", b, sh.Kind, err)
		}
	}
}

// TestTileCoversExactlyWhatItClaims is the twin gate for the collective tile
// ops. It asserts the lowering, not the device: whether a card enumerates a
// given (M,N,K) is refused separately at Compile (vulkan.Ctx.SupportsTile).
func TestTileCoversExactlyWhatItClaims(t *testing.T) {
	all := map[string]func(*ir.Kernel) error{
		"ptx":   func(k *ir.Kernel) error { _, e := ptx.Lower(k, "sm_86"); return e },
		"spirv": func(k *ir.Kernel) error { _, e := spirv.Emit(k); return e },
		"msl":   func(k *ir.Kernel) error { _, e := msl.Emit(k); return e },
	}
	seen := map[string]int{}
	for _, b := range append(append([]string{}, tileImplemented...), tileDeclined...) {
		seen[b]++
	}
	if len(seen) != len(all) {
		t.Fatalf("%d backends lower kernels and this file accounts for %d; a backend "+
			"added without an entry above silently loses its tile coverage", len(all), len(seen))
	}

	// The backends narrow at different places:
	//
	//	spirv  the shape is in the type, so the lowering takes any of them and
	//	       the driver refuses what it did not enumerate (not visible here).
	//	msl    the shape is a template parameter and simdgroup_matrix is 8x8
	//	       only, so the lowering refuses 16x8x16 itself.
	//	ptx    any side a multiple of 8, binary16 or float32 (three tf32
	//	       products) operands; an int8 tile is refused.
	for _, c := range []struct {
		m, n, k int
		elem    ir.TileElem
		lowers  []string // the backends whose LOWERING accepts this shape
	}{
		{16, 8, 16, ir.TileF16, []string{"spirv", "ptx"}},
		{8, 8, 8, ir.TileF32, []string{"spirv", "msl", "ptx"}},
		{16, 8, 32, ir.TileS8, []string{"spirv"}},
	} {
		k, err := mainkernels.TileProbe(c.m, c.n, c.k, 2, c.elem)
		if err != nil {
			t.Fatalf("TileProbe: %v", err)
		}
		if err := k.Validate(); err != nil {
			t.Fatalf("the probe itself does not validate: %v", err)
		}
		lowers := map[string]bool{}
		for _, b := range c.lowers {
			if !contains(tileImplemented, b) {
				t.Fatalf("%q lowers a shape and is not in tileImplemented", b)
			}
			lowers[b] = true
		}
		for _, b := range tileImplemented {
			err := all[b](k)
			switch {
			case lowers[b] && err != nil:
				t.Errorf("%s is listed as lowering %dx%dx%d %s and refused it: %v",
					b, c.m, c.n, c.k, c.elem, err)
			case !lowers[b] && err == nil:
				t.Errorf("%s lowered %dx%dx%d %s and is not listed as doing so; the "+
					"shape table above is what a kernel picks from", b, c.m, c.n, c.k, c.elem)
			case !lowers[b]:
				t.Logf("%s narrows %dx%dx%d %s at the lowering: %v", b, c.m, c.n, c.k, c.elem, err)
			}
		}
		for _, b := range tileDeclined {
			err := all[b](k)
			if err == nil {
				t.Errorf("%s lowered a %dx%dx%d %s tile and is listed as declined; move it",
					b, c.m, c.n, c.k, c.elem)
				continue
			}
			if !strings.Contains(err.Error(), "tile") && !strings.Contains(err.Error(), "matrix") {
				t.Errorf("%s: the refusal does not name what is missing: %v", b, err)
			}
			t.Logf("%s declines %dx%dx%d %s: %v", b, c.m, c.n, c.k, c.elem, err)
		}
	}
}

func contains(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}
