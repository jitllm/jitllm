package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/composer/kernels"
	"github.com/jitllm/jitllm/composer/ptx"
)

// voltaTileCase is one TestVoltaTilesMatchTheReference kernel: D = A*B + C
// chained twice (D2 = A*B + D) over views of the given majors and strides
// wider than the tile, the accumulator loaded or splat, and A and B stored
// back beside D.
type voltaTileCase struct {
	m, n, k          int
	aCol, bCol, oCol bool
	loadC            bool
}

// TestVoltaTilesMatchTheReference forces sm_70's lowering of the collective
// tiles (the m8n8k4 quad-pair layout, composer/ptx/tile.go) on every CUDA device
// of sm_70 or later -- a card runs PTX built for an older target -- and holds
// each case to a float64 product over small integers, where binary16
// operands and float32 sums are exact, so the compare is equality. The cases
// sweep sides of 8 to 32 and k of 8 to 48 (a k of 8 past the last 16 runs
// half the quad-pairs on zeros), both majors of every view, an accumulator
// loaded and splat, and an operand stored back, which reads every register a
// lane holds and no other. Main's staged tile GEMM then runs at sm_70 too.
func TestVoltaTilesMatchTheReference(t *testing.T) {
	gpuLock(t)
	d, err := backend.OpenCUDAWith(0, backend.Opts{CUDATarget: "sm_70"})
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	defer d.Close()
	// A card below sm_70 has no m8n8k4 to run what was built for it.
	pk, err := kernels.MMAVoltaProbe()
	if err != nil {
		t.Fatal(err)
	}
	probe, err := d.Compile(pk)
	if err != nil {
		t.Skipf("%s: no sm_70 matrix instruction: %v", d.API(), err)
	}
	probe.Close()
	var cases []voltaTileCase
	for _, m := range []int{8, 16, 24, 32} {
		for _, n := range []int{8, 16, 24, 32} {
			for _, k := range []int{8, 16, 24, 48} {
				i := len(cases)
				cases = append(cases, voltaTileCase{m: m, n: n, k: k,
					aCol: i&1 != 0, bCol: i&2 == 0, oCol: i&4 != 0, loadC: i&8 == 0})
			}
		}
	}
	pairs := 0
	for _, c := range cases {
		// A kernel whose tiles are all in sixteens takes the output-split
		// layout, whose store sums nothing across the pairs; any other the
		// k-split, whose store does.
		kk, _, err := voltaTileKernel(c)
		if err != nil {
			t.Fatal(err)
		}
		src, err := ptx.Lower(kk, "sm_70")
		if err != nil {
			t.Fatal(err)
		}
		split := c.m%16 == 0 && c.n%16 == 0
		if strings.Contains(src, "shfl.sync") == split {
			t.Fatalf("%+v: output-split %v, yet the store's shuffles say otherwise", c, split)
		}
		if split {
			pairs++
		}
		voltaTileCaseRun(t, d, c)
	}
	if pairs == 0 || pairs == len(cases) {
		t.Fatalf("%d of %d cases took the output-split layout: the gate holds one layout only", pairs, len(cases))
	}
	g := 0
	for _, q := range []kernels.Quant{kernels.Q4_K, kernels.Q8_0, kernels.Q6_K} {
		for _, sh := range gemmTileCases[:3] {
			sub, _, _, _ := kernels.Layout(q)
			tl := sh.tl
			tl.KB = max(tl.KB*32/sub, 1)
			s := kernels.MatVecShape{T: q, K: sh.k, Rows: sh.rows, NTok: sh.ntok, Bias: sh.bias}
			kk, err := kernels.GemmTile(s, tl)
			if err != nil {
				t.Fatalf("%v %+v: %v", q, sh, err)
			}
			kern, err := d.Compile(kk)
			if err != nil {
				t.Fatalf("%s sm_70 gemmtile %v: %v", d.API(), q, err)
			}
			mma70Case(t, d, kern, s, fmt.Sprintf("gemmtile sm_70 %v %+v", q, tl),
				kernels.GemmTileGroups(s, tl), tl.Threads(), true)
			kern.Close()
			g++
		}
	}
	t.Logf("%s at sm_70: %d tile cases (%d output-split) and %d staged GEMMs", d.API(), len(cases), pairs, g)
}

// voltaTileKernel builds c's kernel. Every view's stride is its minor extent
// plus 8, so an access that ignored the stride reads the wrong elements.
func voltaTileKernel(c voltaTileCase) (*ir.Kernel, [4]int, error) {
	aT := ir.TileType{Rows: c.m, Cols: c.k, Elem: ir.TileF16, Use: ir.TileA}
	bT := ir.TileType{Rows: c.k, Cols: c.n, Elem: ir.TileF16, Use: ir.TileB}
	cT := ir.TileType{Rows: c.m, Cols: c.n, Elem: ir.TileF32, Use: ir.TileAcc}
	stride := func(rows, cols int, col bool) int {
		if col {
			return rows + 8
		}
		return cols + 8
	}
	st := [4]int{stride(c.m, c.k, c.aCol), stride(c.k, c.n, c.bCol), stride(c.m, c.n, false), stride(c.m, c.n, c.oCol)}
	b := ir.New("voltatile", [3]int{32, 1, 1})
	pA := b.ParamTile("pA", ir.TileF16)
	pB := b.ParamTile("pB", ir.TileF16)
	pC := b.ParamTile("pC", ir.TileF32)
	pO := b.ParamTile("pO", ir.TileF32)
	pA2 := b.ParamTile("pA2", ir.TileF16)
	pB2 := b.ParamTile("pB2", ir.TileF16)
	k := func(v int) ir.Value { return b.Const(ir.U32, int64(v)) }
	av := b.TileLoad(aT, pA, k(3), k(st[0]), c.aCol)
	bv := b.TileLoad(bT, pB, k(5), k(st[1]), c.bCol)
	var acc ir.Value
	if c.loadC {
		acc = b.TileLoad(cT, pC, k(0), k(st[2]), false)
	} else {
		acc = b.TileSplat(cT, b.ConstF32(1.5))
	}
	acc = b.TileMMA(av, bv, acc)
	acc = b.TileMMA(av, bv, acc)
	b.TileStore(pO, k(1), k(st[3]), acc, c.oCol)
	b.TileStore(pA2, k(3), k(st[0]), av, c.aCol)
	b.TileStore(pB2, k(5), k(st[1]), bv, c.bCol)
	kk, err := b.Done()
	return kk, st, err
}

func voltaTileCaseRun(t *testing.T, d backend.Device, c voltaTileCase) {
	t.Helper()
	kk, st, err := voltaTileKernel(c)
	if err != nil {
		t.Fatalf("%+v: %v", c, err)
	}
	kern, err := d.Compile(kk)
	if err != nil {
		t.Fatalf("%s sm_70 %+v: %v", d.API(), c, err)
	}
	defer kern.Close()
	rng := rand.New(rand.NewSource(int64(c.m*1000 + c.n*10 + c.k)))
	at := func(r, col, stride int, colMajor bool) int {
		if colMajor {
			return col*stride + r
		}
		return r*stride + col
	}
	extent := func(rows, cols, stride int, colMajor bool) int {
		if colMajor {
			return cols*stride + 16
		}
		return rows*stride + 16
	}
	A := make([]float32, c.m*c.k)
	B := make([]float32, c.k*c.n)
	C := make([]float32, c.m*c.n)
	for i := range A {
		A[i] = float32(rng.Intn(9) - 4)
	}
	for i := range B {
		B[i] = float32(rng.Intn(9) - 4)
	}
	for i := range C {
		C[i] = float32(rng.Intn(17) - 8)
	}
	// The views' gaps hold a sentinel an access must never read into the tile.
	aMem := make([]uint16, extent(c.m, c.k, st[0], c.aCol))
	bMem := make([]uint16, extent(c.k, c.n, st[1], c.bCol))
	for i := range aMem {
		aMem[i] = f16bits(1000)
	}
	for i := range bMem {
		bMem[i] = f16bits(1000)
	}
	cMem := make([]float32, extent(c.m, c.n, st[2], false))
	for r := 0; r < c.m; r++ {
		for j := 0; j < c.k; j++ {
			aMem[3+at(r, j, st[0], c.aCol)] = f16bits(A[r*c.k+j])
		}
		for j := 0; j < c.n; j++ {
			cMem[at(r, j, st[2], false)] = C[r*c.n+j]
		}
	}
	for r := 0; r < c.k; r++ {
		for j := 0; j < c.n; j++ {
			bMem[5+at(r, j, st[1], c.bCol)] = f16bits(B[r*c.n+j])
		}
	}
	g := newGPU(t, d)
	defer g.free()
	oN := extent(c.m, c.n, st[3], c.oCol)
	const sentinel = 0x7E00 // a binary16 NaN, never a tile's value
	a2 := make([]uint16, len(aMem))
	b2 := make([]uint16, len(bMem))
	for i := range a2 {
		a2[i] = sentinel
	}
	for i := range b2 {
		b2[i] = sentinel
	}
	bA, bB, bC := g.up(u16bytes(aMem)), g.up(u16bytes(bMem)), g.up(f32bytes(cMem))
	bO, bA2, bB2 := g.up(f32bytes(nanFill(oN))), g.up(u16bytes(a2)), g.up(u16bytes(b2))
	if err := kern.Launch(1, 32, bA, bB, bC, bO, bA2, bB2); err != nil {
		t.Fatal(err)
	}
	out := make([]byte, 4*oN)
	if err := bO.Read(out); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < oN; i++ {
		got := math.Float32frombits(binary.LittleEndian.Uint32(out[4*i:]))
		r, j, in := -1, -1, false
		if i >= 1 {
			x := i - 1
			if c.oCol {
				j, r = x/st[3], x%st[3]
			} else {
				r, j = x/st[3], x%st[3]
			}
			in = r < c.m && j < c.n
		}
		if !in {
			if !math.IsNaN(float64(got)) {
				t.Fatalf("%+v: element %d outside the tile written (%g)", c, i, got)
			}
			continue
		}
		want := 1.5
		if c.loadC {
			want = float64(C[r*c.n+j])
		}
		for x := 0; x < c.k; x++ {
			want += 2 * float64(A[r*c.k+x]) * float64(B[x*c.n+j])
		}
		if float64(got) != want {
			t.Fatalf("%+v: D[%d][%d] = %g, want %g", c, r, j, got, want)
		}
	}
	for _, v := range []struct {
		buf  backend.Buf
		mem  []uint16
		name string
	}{{bA2, aMem, "A"}, {bB2, bMem, "B"}} {
		raw := make([]byte, 2*len(v.mem))
		if err := v.buf.Read(raw); err != nil {
			t.Fatal(err)
		}
		for i := range v.mem {
			got := binary.LittleEndian.Uint16(raw[2*i:])
			want := v.mem[i]
			if want == f16bits(1000) {
				want = sentinel
			}
			if got != want {
				t.Fatalf("%+v: stored %s element %d = %#04x, want %#04x", c, v.name, i, got, want)
			}
		}
	}
}

func u16bytes(v []uint16) []byte {
	out := make([]byte, 2*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint16(out[2*i:], x)
	}
	return out
}
