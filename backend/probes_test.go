package backend_test

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/ir"
)

// The kernels the backend gates run as workloads and probes: IR built here,
// the way a gate builds a kernel to exercise one thing a backend lowers
// (a grid barrier, a subgroup collective, a matrix fragment, a strided copy).

// Restride copies a rows x cols block of 32-bit words from a source laid out at
// srcStride words a row into a destination at dstStride:
// dst[r*dstStride+c] = src[r*srcStride+c]. It is how a KV cache changes its
// capacity on the device (a transposed K is kvDim rows of cap+1 positions)
// without a round trip through the host.
func Restride(rows, cols, srcStride, dstStride int) (*ir.Kernel, error) {
	if rows < 1 || cols < 1 || srcStride < cols || dstStride < cols {
		return nil, fmt.Errorf("kernels: Restride: %dx%d at strides %d -> %d", rows, cols, srcStride, dstStride)
	}
	b := ir.New("restride", [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.F32)
	pDst := b.Param("pDst", ir.F32)
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(rows*cols-1)))
	c := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(cols)))
	r := b.Div(ir.U32, flat, b.Const(ir.U32, int64(cols)))
	src := b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(srcStride))), c)
	dst := b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(dstStride))), c)
	b.Store(pDst, dst, b.Load(ir.F32, pSrc, src, 0), 0)
	return b.Done()
}

// Add is the residual, three-operand: out[i] = a[i] + b[i].
//
// Not dst += src: surplus threads clamp onto the last element and would
// accumulate twice, and ir.Validate refuses the aliased form. The forward
// pass ping-pongs two residual buffers instead.
func Add(n int) (*ir.Kernel, error) {
	b := ir.New("add", [3]int{128, 1, 1})
	pA := b.Param("pA", ir.F32)
	pB := b.Param("pB", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(n-1)))
	b.Store(pOut, i, b.Add(ir.F32, b.Load(ir.F32, pA, i, 0), b.Load(ir.F32, pB, i, 0)), 0)
	return b.Done()
}

// AddScaled is the residual add of a block output that is scaled first
// (Granite's residual_scale): out[i] = a[i] + alpha*b[i], with alpha read from
// a one-element buffer for Scale's reason. Three-operand for Add's.
//
// One rounding, as the host's axpy rounds it (VFMADD231PS on amd64, FMLA on
// arm64): ir.Fma. A multiply and an add written apart are fused or not at the
// compiler's whim -- ptxas and both Vulkan drivers here fused them, Metal did
// not, and its scaled residual then parted from the host's in the last bit of
// 2.8% of every block's elements.
func AddScaled(n int) (*ir.Kernel, error) {
	b := ir.New("add_scaled", [3]int{128, 1, 1})
	pA := b.Param("pA", ir.F32)
	pB := b.Param("pB", ir.F32)
	pAlpha := b.Param("pAlpha", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	zero := b.Const(ir.U32, 0)
	i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(n-1)))
	b.Store(pOut, i, b.Fma(b.Load(ir.F32, pAlpha, zero, 0), b.Load(ir.F32, pB, i, 0), b.Load(ir.F32, pA, i, 0)), 0)
	return b.Done()
}

// GridProbePhases is how many phases GridProbe runs.
const GridProbePhases = 8

// GridProbe is the grid barrier's gate kernel (ir/gridsync.go): groups
// workgroups of width threads run GridProbePhases phases separated by grid
// barriers. Phase 0 writes out0[i] = i for its own thread i; phase p writes
// out_p[i] = out_{p-1}[GridProbeSrc(p, i, groups, width)] + 1, a word a
// DIFFERENT workgroup wrote in the phase before. Before each write a workgroup
// spins for a time that differs by workgroup and phase (the sum goes to pSink
// so it is not removed), so a barrier that lets one workgroup through early has
// it read a word not yet written: the outputs start poisoned, and the probe's
// answer is then the poison plus something.
//
// The parameters are out0..out7, pSink, pSync.
func GridProbe(groups, width int) (*ir.Kernel, error) {
	n := groups * width
	b := ir.New("gridprobe", [3]int{width, 1, 1})
	var out [GridProbePhases]ir.Value
	for p := range out {
		out[p] = b.Param("out"+string(rune('0'+p)), ir.U32)
	}
	pSink := b.Param("pSink", ir.F32)
	pSync := b.Param("pSync", ir.U32)
	c := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	g := b.CTAID()
	i := b.Add(ir.U32, b.Mul(ir.U32, g, c(int64(width))), b.TID())
	b.Store(out[0], i, i, 0)
	acc0 := b.ConstF32(0)
	for p := 1; p < GridProbePhases; p++ {
		b.GridBarrier(pSync)
		// The read comes straight after the barrier and the write after the
		// skew, so a workgroup through early reads a word its writer is still
		// spinning before.
		v := b.Load(ir.U32, out[p-1], gridProbeSrcIR(b, p, i, groups, width), 0)
		// Skew: workgroup g spins ((g*7 + p*13) mod groups) * 256 steps.
		spin := b.Mul(ir.U32, b.Rem(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, g, c(7)), c(int64(p*13))), c(int64(groups))), c(256))
		b.LoopN(spin)
		acc := b.Phi(ir.F32, acc0)
		b.SetPhi(acc, b.Fma(acc, b.ConstF32(0.999), b.ConstF32(1)))
		b.EndLoop()
		b.Store(pSink, b.Add(ir.U32, i, c(int64((p-1)*n))), acc, 0)
		b.Store(out[p], i, b.Add(ir.U32, v, c(1)), 0)
	}
	return b.Done()
}

// gridProbeShift is how many workgroups away phase p reads: never its own
// when there are two or more.
func gridProbeShift(p, groups int) int {
	if groups > 1 {
		return 1 + p%(groups-1)
	}
	return 1
}

// GridProbeSrc is the thread whose phase p-1 word thread i reads in phase p.
func GridProbeSrc(p, i, groups, width int) int {
	return (i + gridProbeShift(p, groups)*width + p) % (groups * width)
}

func gridProbeSrcIR(b *ir.Builder, p int, i ir.Value, groups, width int) ir.Value {
	return b.Rem(ir.U32, b.Add(ir.U32, i, b.Const(ir.U32, int64(gridProbeShift(p, groups)*width+p))),
		b.Const(ir.U32, int64(groups*width)))
}

// GridProbeWant is what GridProbe leaves in out0..out7.
func GridProbeWant(groups, width int) [][]uint32 {
	n := groups * width
	out := make([][]uint32, GridProbePhases)
	out[0] = make([]uint32, n)
	for i := range out[0] {
		out[0][i] = uint32(i)
	}
	for p := 1; p < GridProbePhases; p++ {
		out[p] = make([]uint32, n)
		for i := 0; i < n; i++ {
			out[p][i] = out[p-1][GridProbeSrc(p, i, groups, width)] + 1
		}
	}
	return out
}

// MMAProbe is a bare int8 GEMM through one warp-level matrix instruction, used
// to pin down the fragment layout on real hardware before any real kernel
// depends on it.
//
// The layout is the part that fails silently: a wrong lane mapping transposes
// a block or swaps rows without faulting. So the mapping is written once here,
// checked against a float64 reference on the device, and every real MMA kernel
// reuses these index expressions. The operands are in natural order in memory
// (A row-major [M][K], B column-major [K][N], as .row.col means) and the kernel
// does the gather; a probe that pre-arranged them would only test itself.
func MMAProbe(sh ir.MMAShape) (*ir.Kernel, error) {
	if sh.Kind == ir.MMAF16 {
		return mmaProbeF16(sh)
	}
	na, nb, nc := sh.Frags()
	if na == 0 || nb == 0 || nc == 0 {
		return nil, fmt.Errorf("kernels: MMAProbe: bad shape %dx%dx%d", sh.M, sh.N, sh.K)
	}
	b := ir.New("mmaprobe", [3]int{32, 1, 1})
	pA := b.Param("pA", ir.U32)
	pB := b.Param("pB", ir.U32)
	pOut := b.Param("pOut", ir.I32)

	lane := b.And(ir.U32, b.TID(), b.Const(ir.U32, 31))
	g := b.Shr(ir.U32, lane, b.Const(ir.U32, 2)) // 0..7
	t := b.And(ir.U32, lane, b.Const(ir.U32, 3)) // 0..3
	kw := sh.K / 4                               // u32 per row of A / column of B

	// A[m][k], row-major: lane holds rows g and g+8, four consecutive k at
	// 4t, and (for K=32) another four at 16+4t.
	af := make([]ir.Value, na)
	for i := 0; i < na; i++ {
		row, half := g, 0
		if i%2 == 1 {
			row = b.Add(ir.U32, g, b.Const(ir.U32, 8))
		}
		if i >= 2 {
			half = 4 // 16 elements on, in u32 units
		}
		af[i] = b.Load(ir.U32, pA, b.Add(ir.U32, b.Mul(ir.U32, row, b.Const(ir.U32, int64(kw))), t),
			int64(half))
	}
	// B[k][n], column-major: lane holds column g, the same k slices.
	bf := make([]ir.Value, nb)
	for i := 0; i < nb; i++ {
		half := 0
		if i >= 1 {
			half = 4
		}
		bf[i] = b.Load(ir.U32, pB, b.Add(ir.U32, b.Mul(ir.U32, g, b.Const(ir.U32, int64(kw))), t),
			int64(half))
	}
	zero := b.Const(ir.I32, 0)
	cf := make([]ir.Value, nc)
	for i := range cf {
		cf[i] = zero
	}
	d := b.MMA(sh, af, bf, cf)

	// D[m][n], row-major out: components 0,1 are row g at columns 2t and 2t+1;
	// components 2,3 are row g+8.
	for i, v := range d {
		row := g
		if i >= 2 {
			row = b.Add(ir.U32, g, b.Const(ir.U32, 8))
		}
		col := b.Add(ir.U32, b.Mul(ir.U32, t, b.Const(ir.U32, 2)), b.Const(ir.U32, int64(i%2)))
		b.Store(pOut, b.Add(ir.U32, b.Mul(ir.U32, row, b.Const(ir.U32, int64(sh.N))), col), v, 0)
	}
	return b.Done()
}

// mmaProbeF16 is MMAProbe for the binary16 shape. A and B arrive as float32 in
// natural order and the kernel packs them, as the real kernels do: only the
// fragment is half precision.
//
// The fragment layout is not the int8 one with different arithmetic: a lane
// holds two halves per register rather than four bytes, so it gets two k at
// 2t and two more at 2t+8. Reading the s8 table instead transposes a block.
func mmaProbeF16(sh ir.MMAShape) (*ir.Kernel, error) {
	na, nb, nc := sh.Frags()
	b := ir.New("mmaprobef16", [3]int{32, 1, 1})
	pA := b.Param("pA", ir.F32)
	pB := b.Param("pB", ir.F32)
	pOut := b.Param("pOut", ir.F32)

	lane := b.And(ir.U32, b.TID(), b.Const(ir.U32, 31))
	g := b.Shr(ir.U32, lane, b.Const(ir.U32, 2)) // 0..7
	t := b.And(ir.U32, lane, b.Const(ir.U32, 3)) // 0..3
	two := b.Mul(ir.U32, t, b.Const(ir.U32, 2))

	// A[m][k], row-major: register i covers row g (+8 for odd i) at k = 2t,
	// plus 8 more for the second pair of registers.
	af := make([]ir.Value, na)
	for i := 0; i < na; i++ {
		row := g
		if i%2 == 1 {
			row = b.Add(ir.U32, g, b.Const(ir.U32, 8))
		}
		off := int64(0)
		if i >= 2 {
			off = 8
		}
		idx := b.Add(ir.U32, b.Mul(ir.U32, row, b.Const(ir.U32, int64(sh.K))), two)
		af[i] = b.PackF16(b.Load(ir.F32, pA, idx, off), b.Load(ir.F32, pA, idx, off+1))
	}
	// B[k][n], column-major: column g, the same k pairs.
	bf := make([]ir.Value, nb)
	for i := 0; i < nb; i++ {
		off := int64(0)
		if i >= 1 {
			off = 8
		}
		idx := b.Add(ir.U32, b.Mul(ir.U32, g, b.Const(ir.U32, int64(sh.K))), two)
		bf[i] = b.PackF16(b.Load(ir.F32, pB, idx, off), b.Load(ir.F32, pB, idx, off+1))
	}
	zero := b.ConstF32(0)
	cf := make([]ir.Value, nc)
	for i := range cf {
		cf[i] = zero
	}
	d := b.MMA(sh, af, bf, cf)
	for i, v := range d {
		row := g
		if i >= 2 {
			row = b.Add(ir.U32, g, b.Const(ir.U32, 8))
		}
		col := b.Add(ir.U32, two, b.Const(ir.U32, int64(i%2)))
		b.Store(pOut, b.Add(ir.U32, b.Mul(ir.U32, row, b.Const(ir.U32, int64(sh.N))), col), v, 0)
	}
	return b.Done()
}

// voltaLane is where lane l sits in ir.MMAVolta(): its quad-pair q, the A row /
// B column r it holds, and the D row and column of accumulator i (see
// ir.MMAVolta()); every kernel on this shape derives its indices here.
type voltaLane struct {
	b    *ir.Builder
	lane ir.Value
}

// quad is the lane's quad-pair, 0..3.
func (v voltaLane) quad() ir.Value {
	b := v.b
	return b.And(ir.U32, b.Shr(ir.U32, v.lane, b.Const(ir.U32, 2)), b.Const(ir.U32, 3))
}

// row is the A row (and B column) the lane holds, 0..7.
func (v voltaLane) row() ir.Value {
	b := v.b
	return b.Add(ir.U32, b.And(ir.U32, v.lane, b.Const(ir.U32, 3)),
		b.Shl(ir.U32, b.Shr(ir.U32, v.lane, b.Const(ir.U32, 4)), b.Const(ir.U32, 2)))
}

// dRow is the D row of accumulator i: (l&1) + 2*((i>>1)&1) + 4*(l>>4).
func (v voltaLane) dRow(i int) ir.Value {
	b := v.b
	r := b.Add(ir.U32, b.And(ir.U32, v.lane, b.Const(ir.U32, 1)),
		b.Shl(ir.U32, b.Shr(ir.U32, v.lane, b.Const(ir.U32, 4)), b.Const(ir.U32, 2)))
	if o := 2 * ((i >> 1) & 1); o != 0 {
		r = b.Add(ir.U32, r, b.Const(ir.U32, int64(o)))
	}
	return r
}

// dCol is the D column of accumulator i: (i&1) + 2*((l>>1)&1) + 4*(i>>2).
func (v voltaLane) dCol(i int) ir.Value {
	b := v.b
	c := b.Shl(ir.U32, b.And(ir.U32, b.Shr(ir.U32, v.lane, b.Const(ir.U32, 1)), b.Const(ir.U32, 1)),
		b.Const(ir.U32, 1))
	if o := (i & 1) + 4*(i>>2); o != 0 {
		c = b.Add(ir.U32, c, b.Const(ir.U32, int64(o)))
	}
	return c
}

// MMAVoltaProbe multiplies, per quad-pair q, A rows 8q..8q+7 (row-major, 32x4
// float) by B columns 8q..8q+7 (B is 4x32, row-major float) and stores the
// 8x8 product at out[(8q+row)*8 + col]. It exists to hold the lowering and the
// lane layout in voltaLane to a CPU reference.
func MMAVoltaProbe() (*ir.Kernel, error) {
	b := ir.New("mmavoltaprobe", [3]int{32, 1, 1})
	pA := b.Param("pA", ir.F32)
	pB := b.Param("pB", ir.F32)
	pOut := b.Param("pOut", ir.F32)
	v := voltaLane{b, b.And(ir.U32, b.TID(), b.Const(ir.U32, 31))}
	q8 := b.Shl(ir.U32, v.quad(), b.Const(ir.U32, 3))
	r := b.Add(ir.U32, q8, v.row())
	aIdx := b.Shl(ir.U32, r, b.Const(ir.U32, 2))
	a := []ir.Value{
		b.PackF16(b.Load(ir.F32, pA, aIdx, 0), b.Load(ir.F32, pA, aIdx, 1)),
		b.PackF16(b.Load(ir.F32, pA, aIdx, 2), b.Load(ir.F32, pA, aIdx, 3)),
	}
	bb := []ir.Value{
		b.PackF16(b.Load(ir.F32, pB, r, 0), b.Load(ir.F32, pB, r, 32)),
		b.PackF16(b.Load(ir.F32, pB, r, 64), b.Load(ir.F32, pB, r, 96)),
	}
	zero := b.ConstF32(0)
	c := make([]ir.Value, 8)
	for i := range c {
		c[i] = zero
	}
	d := b.MMA(ir.MMAVolta(), a, bb, c)
	for i, x := range d {
		row := b.Add(ir.U32, q8, v.dRow(i))
		b.Store(pOut, b.Add(ir.U32, b.Shl(ir.U32, row, b.Const(ir.U32, 3)), v.dCol(i)), x, 0)
	}
	return b.Done()
}

// SubgroupProbeOuts is how many t-typed results SubgroupProbe writes per
// thread, and SubgroupProbeMeta how many u32 ones.
const (
	SubgroupProbeOuts = 5
	SubgroupProbeMeta = 2
)

// SubgroupProbe exercises every subgroup collective (ir/subgroup.go) in one
// kernel, for a gate that holds each target's lowering to a host reference.
//
// Workgroups of two subgroups, so a collective that leaked across the
// subgroup boundary reads the wrong half. Thread i of the launch (group*64 +
// tid) reads pIn[i] and pSrc[i] and writes, at pOut[i*5 ..]: the subgroup's
// sum, maximum and minimum of pIn, pIn broadcast from lane (7*sg + 3) & 31
// where sg is the subgroup's index in the workgroup (uniform), and pIn
// shuffled from lane pSrc[i] (per lane, any value: the op masks it to 31); and
// at pMeta[i*2 ..]: the lane index and the subgroup width.
func SubgroupProbe(t ir.Type) (*ir.Kernel, error) {
	switch t {
	case ir.F32, ir.I32, ir.U32:
	default:
		return nil, fmt.Errorf("kernels: SubgroupProbe: %s is not a 32-bit scalar", t)
	}
	const width = 2 * ir.SubgroupLanes
	b := ir.New("subgroupprobe_"+t.String(), [3]int{width, 1, 1})
	pIn := b.Param("pIn", t)
	pSrc := b.Param("pSrc", ir.U32)
	pOut := b.Param("pOut", t)
	pMeta := b.Param("pMeta", ir.U32)
	c := func(v int64) ir.Value { return b.Const(ir.U32, v) }
	i := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), c(width)), b.TID())
	x := b.Load(t, pIn, i, 0)
	src := b.Load(ir.U32, pSrc, i, 0)
	sg := b.SubgroupIndex()
	from := b.Add(ir.U32, b.Mul(ir.U32, sg, c(7)), c(3))

	o := b.Mul(ir.U32, i, c(SubgroupProbeOuts))
	b.Store(pOut, o, b.SubgroupReduce(t, x, ir.ReduceAdd), 0)
	b.Store(pOut, o, b.SubgroupReduce(t, x, ir.ReduceMax), 1)
	b.Store(pOut, o, b.SubgroupReduce(t, x, ir.ReduceMin), 2)
	b.Store(pOut, o, b.SubgroupBroadcast(t, x, from), 3)
	b.Store(pOut, o, b.Shuffle(t, x, src), 4)

	m := b.Mul(ir.U32, i, c(SubgroupProbeMeta))
	b.Store(pMeta, m, b.LaneID(), 0)
	b.Store(pMeta, m, b.SubgroupSize(), 1)
	k, err := b.Done()
	if err != nil {
		return nil, err
	}
	if err := k.Validate(); err != nil {
		return nil, err
	}
	return k, nil
}

// TileProbe is the smallest kernel that exercises a collective matrix end to
// end: load A and B, zero an accumulator, chain several multiply-accumulates,
// store the result.
//
// The chain is what matters: a cooperative value must survive from one MulAdd
// to the next without a memory round trip, which a single MMA would not test.
// The device's shapes are a list it enumerates
// (backend/vulkan.CooperativeMatrixShapes) and operand, accumulator and
// storage types must all match, so one numeric probe checks them all.
//
// A and B are row-major, k tiles of them laid end to end. The reference is
// plain: out = sum over t of A[t] * B[t].
func TileProbe(m, n, k, chain int, elem ir.TileElem) (*ir.Kernel, error) {
	if chain < 1 {
		return nil, fmt.Errorf("kernels: TileProbe: chain=%d, want at least one MulAdd", chain)
	}
	acc := ir.TileF32
	if elem == ir.TileS8 {
		acc = ir.TileI32
	}
	aT := ir.TileType{Rows: m, Cols: k, Elem: elem, Use: ir.TileA}
	bT := ir.TileType{Rows: k, Cols: n, Elem: elem, Use: ir.TileB}
	cT := ir.TileType{Rows: m, Cols: n, Elem: acc, Use: ir.TileAcc}
	for _, t := range []ir.TileType{aT, bT, cT} {
		if !t.Valid() {
			return nil, fmt.Errorf("kernels: TileProbe: %v is not a well-formed tile type", t)
		}
	}

	b := ir.New("tileprobe", [3]int{32, 1, 1})
	pA := b.ParamTile("pA", elem)
	pB := b.ParamTile("pB", elem)
	pOut := b.ParamTile("pOut", acc)

	// Strides are in elements of the component type: Vulkan counts them in
	// the matrix's own components, not bytes and not the buffer's scalars.
	aStride := b.Const(ir.U32, int64(k))
	bStride := b.Const(ir.U32, int64(n))
	outStride := b.Const(ir.U32, int64(n))

	c := b.TileSplat(cT, b.Const(acc2scalar(acc), 0))
	for t := 0; t < chain; t++ {
		av := b.TileLoad(aT, pA, b.Const(ir.U32, int64(t*m*k)), aStride, false)
		bv := b.TileLoad(bT, pB, b.Const(ir.U32, int64(t*k*n)), bStride, false)
		c = b.TileMMA(av, bv, c)
	}
	b.TileStore(pOut, b.Const(ir.U32, 0), outStride, c, false)
	return b.Done()
}

// acc2scalar is the scalar type a splat's seed carries. The accumulator's
// component type is the tile's; the value handed to the splat is an ordinary
// register, and those are still only 32-bit here.
func acc2scalar(e ir.TileElem) ir.Type {
	if e == ir.TileI32 {
		return ir.I32
	}
	return ir.F32
}
