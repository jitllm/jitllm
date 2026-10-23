package kernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/ir"
)

// Attention kernels. The score and accumulate loops run pos+1 times and pos
// advances every token, so their bound arrives at run time in a one-element
// u32 buffer read with LoopN; nothing is re-JITed per token. headDim is a
// constant from the file, so the head loop stays a baked Loop.

// RoPERows rotates each head's first nRot elements, reading pSrc and writing pDst
// at a runtime element offset, so k is rotated straight into its cache slot.
//
// It must not be in place: surplus threads are clamped to the last work item,
// which is benign only when the work is a pure function of read-only inputs.
// In place, the surplus threads re-rotate values their neighbours already
// rotated.
//
// The cos/sin table pCS is built on the device by RopeTable (or uploaded); it
// uses the host's own range reduction and polynomials rather than driver
// transcendentals, so every backend matches the host to about one ulp.
//
// When nRot < headDim the untouched tail is not copied; run CopyAt first.
//
// It runs over `rows` consecutive positions, as a prefill chunk needs. pCS holds one table per row and pOff one offset per row; a shared
// table would rotate every token by the first token's position.
func RoPERows(nHeads, headDim, nRot int, neox bool, rows int) (*ir.Kernel, error) {
	return RoPERowsT(nHeads, headDim, nRot, neox, rows, 0)
}

// RoPERowsT is RoPERows with a transposed destination: element (position, e) of
// the K cache lands at e*dstStride + position rather than position*kvDim + e.
// dstStride == 0 keeps the row-major destination, which is what q wants.
//
// The device K cache is transposed ([head][dim][position]) because the scores
// kernel reads K with one thread per position: row-major, a warp's lanes are
// kvDim*4 bytes apart; transposed, they read consecutive floats. The cost is a
// scattered write in the rotation. The host cache stays row-major, so
// MigrateKV transposes on the way across; V stays row-major because AttnAcc
// reads it with one thread per dimension, already coalesced.
func RoPERowsT(nHeads, headDim, nRot int, neox bool, rows, dstStride int) (*ir.Kernel, error) {
	return ropeRows(nHeads, headDim, nRot, neox, false, rows, dstStride)
}

// RoPERowsSplit is XD-RoPE's rotation (HunyuanVL): NEOX pairs whose halves
// turn by tables of their own -- pCS holds, per row, table A then table B,
// nRot floats each -- so a pair's first element takes A's angle and its second
// B's. With A equal to B it is RoPERowsT's NEOX rotation. dstStride as
// RoPERowsT's.
func RoPERowsSplit(nHeads, headDim, nRot, rows, dstStride int) (*ir.Kernel, error) {
	return ropeRows(nHeads, headDim, nRot, true, true, rows, dstStride)
}

func ropeRows(nHeads, headDim, nRot int, neox, split bool, rows, dstStride int) (*ir.Kernel, error) {
	if nRot%2 != 0 || nRot <= 0 || nRot > headDim {
		return nil, fmt.Errorf("kernels: RoPE: nRot=%d must be even and in (0,%d]", nRot, headDim)
	}
	if rows < 1 {
		rows = 1
	}
	pairs := nRot / 2
	name := "roperowst"
	if split {
		name = "roperowsx"
	}
	b := ir.New(name, [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.F32)
	pCS := b.Param("pCS", ir.F32) // cos,sin interleaved, one pair per rotated pair
	pOff := b.Param("pOff", ir.U32)
	pDst := b.Param("pDst", ir.F32)
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(nHeads*pairs*rows-1)))
	ropeBody(b, nHeads, headDim, nRot, neox, split, rows, dstStride, pSrc, pCS, pOff, pDst, flat)
	return b.Done()
}

// ropeBody is ropeRows after its thread index: the pair flat (clamped by the
// caller) of row-major q rotated from pSrc into pDst. RoPEQKVPaged runs it
// beside the paged K and V writers.
func ropeBody(b *ir.Builder, nHeads, headDim, nRot int, neox, split bool, rows, dstStride int,
	pSrc, pCS, pOff, pDst, flat ir.Value) {
	pairs := nRot / 2
	t, row := flat, ir.Value(0)
	if rows > 1 {
		perRow := b.Const(ir.U32, int64(nHeads*pairs))
		t = b.Rem(ir.U32, flat, perRow)
		row = b.Div(ir.U32, flat, perRow)
	}
	p := b.Rem(ir.U32, t, b.Const(ir.U32, int64(pairs)))
	h := b.Div(ir.U32, t, b.Const(ir.U32, int64(pairs)))
	var base ir.Value
	// The two pair layouts: NORM rotates neighbours (2p, 2p+1), NEOX rotates
	// across the half boundary (p, p+nRot/2). See nn.RoPENeox.
	head := b.Mul(ir.U32, h, b.Const(ir.U32, int64(headDim)))
	stride := int64(1)
	if neox {
		base = b.Add(ir.U32, head, p)
		stride = int64(pairs)
	} else {
		base = b.Add(ir.U32, head, b.Mul(ir.U32, p, b.Const(ir.U32, 2)))
	}
	cs := b.Mul(ir.U32, p, b.Const(ir.U32, 2))
	src := base
	tabW := int64(nRot)
	if split {
		tabW *= 2 // table A then table B
	}
	if rows > 1 {
		cs = b.Add(ir.U32, cs, b.Mul(ir.U32, row, b.Const(ir.U32, tabW)))
		src = b.Add(ir.U32, base, b.Mul(ir.U32, row, b.Const(ir.U32, int64(nHeads*headDim))))
	}
	c := b.Load(ir.F32, pCS, cs, 0)
	s := b.Load(ir.F32, pCS, cs, 1)
	c2, s2 := c, s
	if split {
		c2 = b.Load(ir.F32, pCS, cs, int64(nRot))
		s2 = b.Load(ir.F32, pCS, cs, int64(nRot)+1)
	}
	x0 := b.Load(ir.F32, pSrc, src, 0)
	x1 := b.Load(ir.F32, pSrc, src, stride)
	// Emitted here, after the loads: hoisting it reorders the IR and changes
	// the rows == 1 kernel the goldens hold.
	offIdx := row
	if rows == 1 {
		offIdx = b.Const(ir.U32, 0)
	}
	off := b.Load(ir.U32, pOff, offIdx, 0)
	out, outStride := b.Add(ir.U32, off, base), stride
	if dstStride > 0 {
		// Transposed: the pair (base, base+stride) is dstStride apart, and pOff
		// carries the position rather than position*kvDim.
		out = b.Add(ir.U32, b.Mul(ir.U32, base, b.Const(ir.U32, int64(dstStride))), off)
		outStride = stride * int64(dstStride)
	}
	b.Store(pDst, out, b.Sub(ir.F32, b.Mul(ir.F32, x0, c), b.Mul(ir.F32, x1, s)), 0)
	b.Store(pDst, out, b.Add(ir.F32, b.Mul(ir.F32, x0, s2), b.Mul(ir.F32, x1, c2)), outStride)
}

// CopyAt writes n floats into a destination at a runtime element offset, which
// is how v reaches its slot in the cache without a trip through the host.
func CopyAt(n int) (*ir.Kernel, error) { return copyAt(n, false, 1) }

// CopyAtRows writes `rows` consecutive vectors, each to its own offset.
func CopyAtRows(n, rows int) (*ir.Kernel, error) { return copyAt(n, false, rows) }

// CopyAtRowsT writes `rows` vectors into a transposed cache: element i of row r
// lands at i*dstStride + pos[r], which is where RoPERowsT puts it. A block
// without rotary (a vision tower) still needs its K in that layout; a plain
// CopyAtRows would write [pos][dim] into a [dim][pos] cache without faulting.
// pOff carries the position, not position*n.
func CopyAtRowsT(n, rows, dstStride int) (*ir.Kernel, error) {
	if dstStride <= 0 {
		return nil, fmt.Errorf("kernels: CopyAtRowsT: dstStride=%d must be positive", dstStride)
	}
	if rows < 1 {
		rows = 1
	}
	b := ir.New("copyatrowst", [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.F32)
	pDst := b.Param("pDst", ir.F32)
	pOff := b.Param("pOff", ir.U32)
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(n*rows-1)))
	i, row := flat, ir.Value(b.Const(ir.U32, 0))
	if rows > 1 {
		i = b.Rem(ir.U32, flat, b.Const(ir.U32, int64(n)))
		row = b.Div(ir.U32, flat, b.Const(ir.U32, int64(n)))
	}
	off := b.Load(ir.U32, pOff, row, 0)
	dst := b.Add(ir.U32, b.Mul(ir.U32, i, b.Const(ir.U32, int64(dstStride))), off)
	b.Store(pDst, dst, b.Load(ir.F32, pSrc, flat, 0), 0)
	return b.Done()
}

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

// CopyAtF16 writes the V cache as packed binary16, two elements per u32.
//
// V packs trivially because a thread owns two adjacent source elements. K
// does not: RoPE writes it from a pair that under NEOX is half a head apart,
// landing in different words.
//
// n must be even (every kvDim is), and so is the offset pos*kvDim.
func CopyAtF16(n int) (*ir.Kernel, error) { return copyAt(n, true, 1) }

func copyAt(n int, f16 bool, rows int) (*ir.Kernel, error) {
	if rows < 1 {
		rows = 1
	}
	if f16 && n%2 != 0 {
		return nil, fmt.Errorf("kernels: CopyAtF16: n=%d must be even", n)
	}
	b := ir.New("copyat", [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.F32)
	pDst := b.Param("pDst", ir.F32)
	pOff := b.Param("pOff", ir.U32)
	last := n - 1
	if f16 {
		last = n/2 - 1
	}
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64((last+1)*rows-1)))
	i, crow := flat, ir.Value(0)
	if rows > 1 {
		per := b.Const(ir.U32, int64(last+1))
		i = b.Rem(ir.U32, flat, per)
		crow = b.Div(ir.U32, flat, per)
	} else {
		crow = b.Const(ir.U32, 0)
	}
	off := b.Load(ir.U32, pOff, crow, 0)
	if rows > 1 {
		// The source advances by a whole vector per row; the destination does
		// not, because each row's offset already names where it goes.
		i = b.Add(ir.U32, i, b.Mul(ir.U32, crow, b.Const(ir.U32, int64(last+1))))
		off = b.Sub(ir.U32, off, b.Mul(ir.U32, crow, b.Const(ir.U32, int64(last+1))))
	}
	if !f16 {
		b.Store(pDst, b.Add(ir.U32, off, i), b.Load(ir.F32, pSrc, i, 0), 0)
		return b.Done()
	}
	two := b.Mul(ir.U32, i, b.Const(ir.U32, 2))
	w := b.PackF16(b.Load(ir.F32, pSrc, two, 0), b.Load(ir.F32, pSrc, two, 1))
	// pOff counts elements; the packed cache is indexed in words.
	wOff := b.Shr(ir.U32, off, b.Const(ir.U32, 1))
	b.Store(pDst, b.Add(ir.U32, wOff, i), b.Bitcast(ir.F32, w), 0)
	return b.Done()
}

// windowed masks a score to -inf when its key lies before the row's sliding
// window: a row whose causal count is cnt attends keys [cnt-min(cnt,W), cnt).
//
// Masking the scores alone is exact: softmax gives -inf zero weight, and the
// V it then multiplies was written earlier in the same sequence, so 0*V is 0.
// window 0 emits nothing, keeping unwindowed kernels byte-identical.
//
// A negative window is a chunk of -window keys (Llama 4's iRoPE, see
// ChunkWindow): the row attends [floor((cnt-1)/C)*C, cnt). It is a different
// mask, not a different width, carried by the same argument.
func windowed(b *ir.Builder, v, pos, cnt ir.Value, window int, negInf ir.Value) ir.Value {
	if window == 0 {
		return v
	}
	var lo ir.Value
	if window < 0 {
		c := b.Const(ir.U32, int64(-window))
		last := b.Sub(ir.U32, b.Max(ir.U32, cnt, b.Const(ir.U32, 1)), b.Const(ir.U32, 1))
		lo = b.Mul(ir.U32, b.Div(ir.U32, last, c), c)
	} else {
		lo = b.Sub(ir.U32, cnt, b.Min(ir.U32, cnt, b.Const(ir.U32, int64(window))))
	}
	return b.Select(ir.F32, b.Lt(ir.U32, pos, lo), negInf, v)
}

// ChunkWindow is the window argument that makes a windowed score kernel attend
// within an aligned chunk of c keys rather than the last c. See windowed.
func ChunkWindow(c int) int { return -c }

func attnScoresOne(nHeads, headDim, kvDim, gqa, maxSeq int, scale float32, kStride, window int) (*ir.Kernel, error) {
	if gqa <= 0 || nHeads%gqa != 0 {
		return nil, fmt.Errorf("kernels: AttnScores: gqa=%d does not divide nHeads=%d", gqa, nHeads)
	}
	b := ir.New("attnscoresone", [3]int{128, 1, 1})
	pQ := b.Param("pQ", ir.F32)
	pK := b.Param("pK", ir.F32)
	pN := b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	n := b.Load(ir.U32, pN, b.Const(ir.U32, 0), 0)
	tid := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	last := b.Sub(ir.U32, b.Mul(ir.U32, n, b.Const(ir.U32, int64(nHeads))), b.Const(ir.U32, 1))
	tid = b.Min(ir.U32, tid, last)
	pos := b.Rem(ir.U32, tid, n)
	h := b.Div(ir.U32, tid, n)
	// The row-major expression is emitted verbatim rather than as a special
	// case of the transposed one; reordering it changes the decode golden.
	kbase, kStep := ir.Value(0), int64(1)
	if kStride > 0 {
		kvh := b.Mul(ir.U32, b.Div(ir.U32, h, b.Const(ir.U32, int64(gqa))), b.Const(ir.U32, int64(headDim)))
		kbase, kStep = b.Add(ir.U32, b.Mul(ir.U32, kvh, b.Const(ir.U32, int64(kStride))), pos), int64(kStride)
	} else {
		kbase = b.Add(ir.U32, b.Mul(ir.U32, pos, b.Const(ir.U32, int64(kvDim))),
			b.Mul(ir.U32, b.Div(ir.U32, h, b.Const(ir.U32, int64(gqa))), b.Const(ir.U32, int64(headDim))))
	}
	qbase := b.Mul(ir.U32, h, b.Const(ir.U32, int64(headDim)))
	acc := b.ConstF32(0)
	for i := 0; i < headDim; i++ {
		acc = b.Fma(b.Load(ir.F32, pQ, qbase, int64(i)), b.Load(ir.F32, pK, kbase, int64(i)*kStep), acc)
	}
	out := b.Add(ir.U32, b.Mul(ir.U32, h, b.Const(ir.U32, int64(maxSeq))), pos)
	v := b.Mul(ir.F32, acc, b.ConstF32(scale))
	if window != 0 {
		v = windowed(b, v, pos, n, window, b.Bitcast(ir.F32, b.Const(ir.U32, 0xFF800000)))
	}
	b.Store(pOut, out, v, 0)
	return b.Done()
}

// AttnScoresWarp is AttnScores with one warp per (head, position): the 32 lanes
// stride the head dimension, so a warp's loads of a key row are one coalesced
// run, and a five-step ShuffleXor butterfly sums the lanes. Launch nHeads*n
// workgroups (n, or a capacity above it -- surplus groups clamp onto the last
// pair and store the same value) of ir.SubgroupLanes threads, on a device that
// guarantees that subgroup width. Row-major K only.
//
// It is for wide heads such as MLA's 576-wide latent row, where a thread per
// dot product means hundreds of dependent FMAs and rows far apart per lane.
func AttnScoresWarp(nHeads, headDim, kvDim, gqa, maxSeq int, scale float32) (*ir.Kernel, error) {
	if gqa <= 0 || nHeads%gqa != 0 || headDim < 1 {
		return nil, fmt.Errorf("kernels: AttnScoresWarp: gqa=%d does not divide nHeads=%d", gqa, nHeads)
	}
	const lanes = ir.SubgroupLanes
	b := ir.New("attnscoreswarp", [3]int{lanes, 1, 1})
	pQ := b.Param("pQ", ir.F32)
	pK := b.Param("pK", ir.F32)
	pN := b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	n := b.Load(ir.U32, pN, b.Const(ir.U32, 0), 0)
	last := b.Sub(ir.U32, b.Mul(ir.U32, n, b.Const(ir.U32, int64(nHeads))), b.Const(ir.U32, 1))
	w := b.Min(ir.U32, b.CTAID(), last)
	pos := b.Rem(ir.U32, w, n)
	h := b.Div(ir.U32, w, n)
	lane := b.TID()
	kbase := b.Add(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, pos, b.Const(ir.U32, int64(kvDim))),
		b.Mul(ir.U32, b.Div(ir.U32, h, b.Const(ir.U32, int64(gqa))), b.Const(ir.U32, int64(headDim)))), lane)
	qbase := b.Add(ir.U32, b.Mul(ir.U32, h, b.Const(ir.U32, int64(headDim))), lane)
	acc := b.ConstF32(0)
	full := headDim / lanes
	for i := 0; i < full; i++ {
		acc = b.Fma(b.Load(ir.F32, pQ, qbase, int64(i*lanes)), b.Load(ir.F32, pK, kbase, int64(i*lanes)), acc)
	}
	if rem := headDim % lanes; rem > 0 {
		// The ragged tail: lanes past it load a clamped element and add zero
		// (no conditional outside a loop, RULE 13).
		live := b.Lt(ir.U32, lane, b.Const(ir.U32, int64(rem)))
		off := b.Min(ir.U32, lane, b.Const(ir.U32, int64(rem-1)))
		qi := b.Add(ir.U32, b.Mul(ir.U32, h, b.Const(ir.U32, int64(headDim))), off)
		ki := b.Add(ir.U32, b.Sub(ir.U32, kbase, lane), off)
		prod := b.Mul(ir.F32, b.Load(ir.F32, pQ, qi, int64(full*lanes)), b.Load(ir.F32, pK, ki, int64(full*lanes)))
		acc = b.Add(ir.F32, acc, b.Select(ir.F32, live, prod, b.ConstF32(0)))
	}
	for m := lanes / 2; m >= 1; m /= 2 {
		acc = b.Add(ir.F32, acc, b.ShuffleXor(ir.F32, acc, int64(m)))
	}
	out := b.Add(ir.U32, b.Mul(ir.U32, h, b.Const(ir.U32, int64(maxSeq))), pos)
	b.Store(pOut, out, b.Mul(ir.F32, acc, b.ConstF32(scale)), 0)
	return b.Done()
}

// SoftmaxRows normalises each head's score row. `lanes` is how many threads
// cooperate on one row: 1 is one thread per head, 32 is one warp per head
// with a ShuffleXor butterfly for the max and the sum.
//
// A 32-lane butterfly is correct only where the subgroup is at least 32 wide
// and lane == tid mod 32. Vulkan permits 8 or 16, and an out-of-subgroup
// shuffle is undefined (it can even read right on an idle device), so the
// width is requested, not assumed: the shuffle sets ir.Kernel.Lanes and
// tier.pickSoftmax takes 32 only on a device that guarantees it (pinned via
// VK_EXT_subgroup_size_control, or fixed on CUDA and Metal).
//
// The clamp invariant survives because a butterfly leaves the result in every
// lane, so a surplus warp clamped onto the last head repeats identical stores.
//
// Over `rows` query positions each row is normalised over its own causal
// count. Deriving the query row from the
// group index and reading pN[1+row] stops each row where its causality does,
// rather than normalising every row over the uniform (masked) width.
func SoftmaxRows(nHeads, maxSeq, lanes, rows, qt int) (*ir.Kernel, error) {
	return SoftmaxRowsSink(nHeads, maxSeq, lanes, rows, qt, false)
}

// SoftmaxRowsSink is SoftmaxRows with, when sink is set, a learned logit per
// head joining the softmax -- gpt-oss's attention sinks. It takes a fourth
// buffer of nHeads floats; the sink is in the maximum and in the denominator
// and is not written, so a row's probabilities sum to less than one. That is
// ggml's soft_max_ext with sinks and transformers' cat-then-drop.
func SoftmaxRowsSink(nHeads, maxSeq, lanes, rows, qt int, sink bool) (*ir.Kernel, error) {
	if rows <= 1 {
		if lanes != 1 && lanes != 32 {
			return nil, fmt.Errorf("kernels: Softmax: lanes=%d, want 1 or 32", lanes)
		}
		if lanes == 1 {
			return softmaxScalar(nHeads, maxSeq, sinkOf(sink, nHeads))
		}
		return softmaxWarp(nHeads, maxSeq, 0, 1, sinkOf(sink, nHeads))
	}
	if lanes != 32 {
		// The scalar kernel handles many heads per group, so a per-row width
		// would vary within a group. It is the fallback for a device that will
		// not guarantee a 32-lane subgroup; it keeps the uniform width and
		// relies on the -inf mask.
		return softmaxScalar(nHeads*rows, maxSeq, sinkOf(sink, nHeads))
	}
	return softmaxWarp(nHeads*rows, maxSeq, nHeads, qt, sinkOf(sink, nHeads))
}

// sinkOf is the head count a sink array is indexed by, or 0 for no sink.
func sinkOf(sink bool, heads int) int {
	if sink {
		return heads
	}
	return 0
}

// softmaxWarp is the 32-lane kernel. perQuery is 0 for one query position, and
// the head count when the launch covers several -- then row r's own key count
// comes from pN[1+r] instead of the uniform pN[0].
//
// sinkHeads > 0 adds the sink buffer, indexed by the row's head modulo it.
func softmaxWarp(nHeads, maxSeq, perQuery, qt, sinkHeads int) (*ir.Kernel, error) {
	// One workgroup per head, exactly one warp wide: the group is the warp, so
	// lane == tid.
	b := ir.New("softmaxwarp", [3]int{32, 1, 1})
	pA := b.Param("pA", ir.F32)
	pN := b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	var pSink ir.Value
	if sinkHeads > 0 {
		pSink = b.Param("pSink", ir.F32)
	}
	// With one query the one-query order is kept: emitting the batched form's
	// Min first unconditionally reordered the decode kernel's golden.
	var n, h ir.Value
	if perQuery > 0 {
		h = b.Min(ir.U32, b.CTAID(), b.Const(ir.U32, int64(nHeads-1)))
		// A tiled AttnAcc walks its group's widest count, so every row of the
		// group must be normalised that far; round the row index up to the
		// group's last row.
		qr := b.Div(ir.U32, h, b.Const(ir.U32, int64(perQuery)))
		if qt > 1 {
			c := b.Const(ir.U32, int64(qt))
			qr = b.Add(ir.U32, b.Mul(ir.U32, b.Div(ir.U32, qr, c), c), b.Sub(ir.U32, c, b.Const(ir.U32, 1)))
		}
		n = b.Load(ir.U32, pN, b.Add(ir.U32, qr, b.Const(ir.U32, 1)), 0)
	} else {
		n = b.Load(ir.U32, pN, b.Const(ir.U32, 0), 0)
		h = b.Min(ir.U32, b.CTAID(), b.Const(ir.U32, int64(nHeads-1)))
	}
	// tid AND 31, not tid: a wider group then puts several warps on the same
	// row storing the same values, rather than walking off its end.
	lane := b.And(ir.U32, b.TID(), b.Const(ir.U32, 31))
	base := b.Add(ir.U32, b.Mul(ir.U32, h, b.Const(ir.U32, int64(maxSeq))), lane)

	// How many of the row's positions belong to this lane: ceil((n-lane)/32),
	// and zero past the end; the Max stops the subtraction wrapping, since
	// there is no conditional outside a loop to mask a lane. Unequal trip
	// counts diverge the warp, which the shuffle reconverges.
	cnt := b.Shr(ir.U32, b.Add(ir.U32, b.Sub(ir.U32, b.Max(ir.U32, n, lane), lane),
		b.Const(ir.U32, 31)), b.Const(ir.U32, 5))
	step := b.Const(ir.U32, 32)

	// The largest score, subtracted before exponentiating so exp never overflows.
	negInf := b.Bitcast(ir.F32, b.Const(ir.U32, 0xFF800000))
	b.LoopN(cnt)
	mx := b.Phi(ir.F32, negInf)
	mi := b.Phi(ir.U32, base)
	b.SetPhi(mx, b.Max(ir.F32, mx, b.Load(ir.F32, pA, mi, 0)))
	b.SetPhi(mi, b.Add(ir.U32, mi, step))
	b.EndLoop()
	rmx := butterfly(b, mx, func(x, y ir.Value) ir.Value { return b.Max(ir.F32, x, y) })
	var sv ir.Value
	if sinkHeads > 0 {
		sv = b.Load(ir.F32, pSink, b.Rem(ir.U32, h, b.Const(ir.U32, int64(sinkHeads))), 0)
		rmx = b.Max(ir.F32, rmx, sv)
	}

	zero := b.ConstF32(0)
	b.LoopN(cnt)
	sum := b.Phi(ir.F32, zero)
	si := b.Phi(ir.U32, base)
	b.SetPhi(sum, b.Add(ir.F32, sum, b.Exp(b.Sub(ir.F32, b.Load(ir.F32, pA, si, 0), rmx))))
	b.SetPhi(si, b.Add(ir.U32, si, step))
	b.EndLoop()
	rsum := butterfly(b, sum, func(x, y ir.Value) ir.Value { return b.Add(ir.F32, x, y) })
	if sinkHeads > 0 {
		rsum = b.Add(ir.F32, rsum, b.Exp(b.Sub(ir.F32, sv, rmx)))
	}

	// The third pass exponentiates again rather than storing the second
	// pass's values, which would make pA both read and written.
	inv := b.Div(ir.F32, b.ConstF32(1), rsum)
	b.LoopN(cnt)
	ni := b.Phi(ir.U32, base)
	e := b.Exp(b.Sub(ir.F32, b.Load(ir.F32, pA, ni, 0), rmx))
	b.Store(pOut, ni, b.Mul(ir.F32, e, inv), 0)
	b.SetPhi(ni, b.Add(ir.U32, ni, step))
	b.EndLoop()
	return b.Done()
}

// butterfly reduces v across the 32 lanes of a warp, leaving the answer in every
// one of them. Five shuffles, no shared memory, no barrier. The sum is a tree
// where the host's is sequential, so low bits differ; the gate is token ids.
func butterfly(b *ir.Builder, v ir.Value, comb func(x, y ir.Value) ir.Value) ir.Value {
	for m := int64(1); m < 32; m *= 2 {
		v = comb(v, b.ShuffleXor(ir.F32, v, m))
	}
	return v
}

// softmaxScalar is one thread per head and three sequential passes over the
// row: the fallback for a device whose subgroup will not carry a shuffle.
func softmaxScalar(nHeads, maxSeq, sinkHeads int) (*ir.Kernel, error) {
	b := ir.New("softmaxscalar", [3]int{64, 1, 1})
	pA := b.Param("pA", ir.F32)
	pN := b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	var pSink ir.Value
	if sinkHeads > 0 {
		pSink = b.Param("pSink", ir.F32)
	}
	n := b.Load(ir.U32, pN, b.Const(ir.U32, 0), 0)
	h := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(nHeads-1)))
	row := b.Mul(ir.U32, h, b.Const(ir.U32, int64(maxSeq)))
	one := b.Const(ir.U32, 1)

	// The largest score, subtracted before exponentiating so exp never overflows.
	negInf := b.Bitcast(ir.F32, b.Const(ir.U32, 0xFF800000))
	b.LoopN(n)
	mx := b.Phi(ir.F32, negInf)
	mi := b.Phi(ir.U32, row)
	b.SetPhi(mx, b.Max(ir.F32, mx, b.Load(ir.F32, pA, mi, 0)))
	b.SetPhi(mi, b.Add(ir.U32, mi, one))
	b.EndLoop()
	m := mx
	var sv ir.Value
	if sinkHeads > 0 {
		sv = b.Load(ir.F32, pSink, b.Rem(ir.U32, h, b.Const(ir.U32, int64(sinkHeads))), 0)
		m = b.Max(ir.F32, mx, sv)
	}

	zero := b.ConstF32(0)
	b.LoopN(n)
	sum := b.Phi(ir.F32, zero)
	si := b.Phi(ir.U32, row)
	b.SetPhi(sum, b.Add(ir.F32, sum, b.Exp(b.Sub(ir.F32, b.Load(ir.F32, pA, si, 0), m))))
	b.SetPhi(si, b.Add(ir.U32, si, one))
	b.EndLoop()
	tot := sum
	if sinkHeads > 0 {
		tot = b.Add(ir.F32, sum, b.Exp(b.Sub(ir.F32, sv, m)))
	}

	// The third pass exponentiates again rather than reusing the second pass's
	// values: storing them would make pA both read and written, which
	// ir.Validate refuses (most threads here are surplus and clamp onto the
	// last row).
	inv := b.Div(ir.F32, b.ConstF32(1), tot)
	b.LoopN(n)
	ni := b.Phi(ir.U32, row)
	e := b.Exp(b.Sub(ir.F32, b.Load(ir.F32, pA, ni, 0), m))
	b.Store(pOut, ni, b.Mul(ir.F32, e, inv), 0)
	b.SetPhi(ni, b.Add(ir.U32, ni, one))
	b.EndLoop()
	return b.Done()
}

// AttnAccSplitF16 is AttnAccSplit against a V cache packed as binary16, two
// elements per u32.
//
// Each thread owns two adjacent outputs: a thread's element index has
// constant parity down the whole cache, so a one-output thread would need a
// conditional to pick its half of every word. Owning both uses both halves
// of the loaded word and shares one probability load. Thread count is half
// AttnAccSplit's for the same split.
func AttnAccSplitF16(nHeads, headDim, kvDim, gqa, maxSeq, split int) (*ir.Kernel, error) {
	if headDim%2 != 0 || kvDim%2 != 0 {
		return nil, fmt.Errorf("kernels: AttnAccSplitF16: headDim=%d kvDim=%d must both be even", headDim, kvDim)
	}
	if split < 1 {
		split = 1
	}
	half := headDim / 2
	outPairs := nHeads * half
	b := ir.New("attnaccsplitf16", [3]int{128, 1, 1})
	pA := b.Param("pA", ir.F32)
	pV := b.Param("pV", ir.F32) // packed: two binary16 per word
	pN := b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	n := b.Load(ir.U32, pN, b.Const(ir.U32, 0), 0)

	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(outPairs*split-1)))
	tid := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(outPairs)))
	sp := b.Div(ir.U32, flat, b.Const(ir.U32, int64(outPairs)))
	j := b.Rem(ir.U32, tid, b.Const(ir.U32, int64(half)))
	h := b.Div(ir.U32, tid, b.Const(ir.U32, int64(half)))
	row := b.Mul(ir.U32, h, b.Const(ir.U32, int64(maxSeq)))
	// V words per position is kvDim/2; this thread's word within a position.
	vcol := b.Add(ir.U32, b.Mul(ir.U32, b.Div(ir.U32, h, b.Const(ir.U32, int64(gqa))),
		b.Const(ir.U32, int64(half))), j)
	kvw := b.Const(ir.U32, int64(kvDim/2))
	S := b.Const(ir.U32, int64(split))

	zeroF := b.ConstF32(0)
	a0 := b.Add(ir.U32, row, sp)
	v0 := b.Add(ir.U32, vcol, b.Mul(ir.U32, sp, kvw))
	sKv := b.Mul(ir.U32, S, kvw)
	cnt := b.Div(ir.U32, b.Add(ir.U32, b.Sub(ir.U32, b.Max(ir.U32, n, sp), sp),
		b.Const(ir.U32, int64(split-1))), S)

	b.LoopN(cnt)
	acc0 := b.Phi(ir.F32, zeroF)
	acc1 := b.Phi(ir.F32, zeroF)
	ai := b.Phi(ir.U32, a0)
	vi := b.Phi(ir.U32, v0)
	pw := b.Load(ir.F32, pA, ai, 0)
	w := b.Bitcast(ir.U32, b.Load(ir.F32, pV, vi, 0))
	b.SetPhi(acc0, b.Fma(pw, b.CvtF16H(w), acc0))
	b.SetPhi(acc1, b.Fma(pw, b.CvtF16H(b.Shr(ir.U32, w, b.Const(ir.U32, 16))), acc1))
	b.SetPhi(ai, b.Add(ir.U32, ai, S))
	b.SetPhi(vi, b.Add(ir.U32, vi, sKv))
	b.EndLoop()

	// Partials stay laid out [split][nHeads*headDim] so kernels.Reduce is
	// unchanged; a thread's two outputs are adjacent within that.
	base := b.Add(ir.U32, b.Mul(ir.U32, sp, b.Const(ir.U32, int64(nHeads*headDim))),
		b.Mul(ir.U32, tid, b.Const(ir.U32, 2)))
	b.Store(pOut, base, acc0, 0)
	b.Store(pOut, base, acc1, 1)
	return b.Done()
}

// AttnAccSplit is AttnAcc with the position loop divided among Split threads
// per output, writing Split partial sums that kernels.Reduce then adds.
//
// AttnAcc launches only nHeads*headDim threads, each walking the whole cache
// serially, so it falls behind with context depth; splitting multiplies the
// threads and shortens each chain. Positions are strided, not blocked: with
// a fixed split index adjacent outputs read adjacent addresses, and a stride
// needs only a count where a block would need a start and an end.
func AttnAccSplit(nHeads, headDim, kvDim, gqa, maxSeq, split int) (*ir.Kernel, error) {
	if split < 2 {
		return nil, fmt.Errorf("kernels: AttnAccSplit: split=%d, want >= 2", split)
	}
	b := ir.New("attnaccsplit", [3]int{128, 1, 1})
	pA := b.Param("pA", ir.F32)
	pV := b.Param("pV", ir.F32)
	pN := b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	n := b.Load(ir.U32, pN, b.Const(ir.U32, 0), 0)
	out := nHeads * headDim
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(out*split-1)))
	tid := b.Rem(ir.U32, flat, b.Const(ir.U32, int64(out)))
	sp := b.Div(ir.U32, flat, b.Const(ir.U32, int64(out))) // which split
	i := b.Rem(ir.U32, tid, b.Const(ir.U32, int64(headDim)))
	h := b.Div(ir.U32, tid, b.Const(ir.U32, int64(headDim)))
	row := b.Mul(ir.U32, h, b.Const(ir.U32, int64(maxSeq)))
	vcol := b.Add(ir.U32, b.Mul(ir.U32, b.Div(ir.U32, h, b.Const(ir.U32, int64(gqa))),
		b.Const(ir.U32, int64(headDim))), i)
	kvd := b.Const(ir.U32, int64(kvDim))
	S := b.Const(ir.U32, int64(split))

	// Positions this split owns: ceil((n - sp)/split), 0 once sp >= n. The Max
	// stops the unsigned subtraction wrapping.
	cnt := b.Div(ir.U32, b.Add(ir.U32, b.Sub(ir.U32, b.Max(ir.U32, n, sp), sp),
		b.Const(ir.U32, int64(split-1))), S)
	// Every Phi init is hoisted above the loop: an init emitted after LoopN is
	// defined inside the loop, which ir.Validate refuses (and a refused kernel
	// silently sends the block back to the host).
	zeroF := b.ConstF32(0)
	a0 := b.Add(ir.U32, row, sp)
	v0 := b.Add(ir.U32, vcol, b.Mul(ir.U32, sp, kvd))
	sKv := b.Mul(ir.U32, S, kvd)

	b.LoopN(cnt)
	acc := b.Phi(ir.F32, zeroF)
	ai := b.Phi(ir.U32, a0)
	vi := b.Phi(ir.U32, v0)
	b.SetPhi(acc, b.Fma(b.Load(ir.F32, pA, ai, 0), b.Load(ir.F32, pV, vi, 0), acc))
	b.SetPhi(ai, b.Add(ir.U32, ai, S))
	b.SetPhi(vi, b.Add(ir.U32, vi, sKv))
	b.EndLoop()
	b.Store(pOut, flat, acc, 0)
	return b.Done()
}

// AttnAcc sums the value vectors weighted by the normalised scores.
func AttnAcc(nHeads, headDim, kvDim, gqa, maxSeq int) (*ir.Kernel, error) {
	b := ir.New("attnacc", [3]int{128, 1, 1})
	pA := b.Param("pA", ir.F32)
	pV := b.Param("pV", ir.F32)
	pN := b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	n := b.Load(ir.U32, pN, b.Const(ir.U32, 0), 0)
	tid := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(nHeads*headDim-1)))
	i := b.Rem(ir.U32, tid, b.Const(ir.U32, int64(headDim)))
	h := b.Div(ir.U32, tid, b.Const(ir.U32, int64(headDim)))
	row := b.Mul(ir.U32, h, b.Const(ir.U32, int64(maxSeq)))
	vcol := b.Add(ir.U32, b.Mul(ir.U32, b.Div(ir.U32, h, b.Const(ir.U32, int64(gqa))),
		b.Const(ir.U32, int64(headDim))), i)
	kvd := b.Const(ir.U32, int64(kvDim))
	zero := b.ConstF32(0)

	b.LoopN(n)
	acc := b.Phi(ir.F32, zero)
	ai := b.Phi(ir.U32, row)
	vi := b.Phi(ir.U32, vcol)
	b.SetPhi(acc, b.Fma(b.Load(ir.F32, pA, ai, 0), b.Load(ir.F32, pV, vi, 0), acc))
	b.SetPhi(ai, b.Add(ir.U32, ai, b.Const(ir.U32, 1)))
	b.SetPhi(vi, b.Add(ir.U32, vi, kvd))
	b.EndLoop()
	b.Store(pOut, tid, acc, 0)
	return b.Done()
}

// HeadNorm RMS-normalises each head of a q or k vector independently, with a
// head_dim-wide weight (qwen3's attn_q_norm and attn_k_norm). It is nHeads
// norms over headDim elements, not one over the whole vector, and it comes
// before RoPE; either mistake still runs and produces fluent nonsense.
//
// `lanes` is how many threads cooperate on one head, as for Softmax: 32 is one
// subgroup per head (a ShuffleXor butterfly; headDim must be a multiple of 32)
// and 1 is one thread per head. The caller passes 32 only when the device
// guarantees that subgroup width (backend.GuaranteedLanes); ir.Kernel.Lanes
// carries the requirement.
//
// Out of place, because ir.Validate refuses in-place kernels: clamped surplus
// threads would scale the last head twice.
func HeadNorm(nHeads, headDim int, eps float64, lanes int) (*ir.Kernel, error) {
	if headDim <= 0 {
		return nil, fmt.Errorf("kernels: HeadNorm: headDim=%d", headDim)
	}
	if nHeads <= 0 {
		return nil, fmt.Errorf("kernels: HeadNorm: nHeads=%d", nHeads)
	}
	switch lanes {
	case 1:
		return headNormScalar(nHeads, headDim, eps)
	case 32:
	default:
		return nil, fmt.Errorf("kernels: HeadNorm: lanes=%d, want 32 (one subgroup per head) "+
			"or 1 (one thread per head)", lanes)
	}
	if headDim%32 != 0 {
		return nil, fmt.Errorf("kernels: HeadNorm: headDim=%d must be a multiple of 32 at "+
			"lanes=32; lanes=1 takes any width", headDim)
	}
	per := headDim / 32
	b := ir.New("headnorm", [3]int{32, 1, 1})
	pIn := b.Param("pIn", ir.F32)
	pW := b.Param("pW", ir.F32)
	pOut := b.Param("pOut", ir.F32)

	h := b.Min(ir.U32, b.CTAID(), b.Const(ir.U32, int64(nHeads-1)))
	lane := b.And(ir.U32, b.TID(), b.Const(ir.U32, 31))
	base := b.Add(ir.U32, b.Mul(ir.U32, h, b.Const(ir.U32, int64(headDim))), lane)

	// Sum of squares over this lane's elements.
	acc := b.ConstF32(0)
	for k := 0; k < per; k++ {
		v := b.Load(ir.F32, pIn, base, int64(k*32))
		acc = b.Fma(v, v, acc)
	}
	for m := 1; m < 32; m <<= 1 {
		acc = b.Add(ir.F32, acc, b.ShuffleXor(ir.F32, acc, int64(m)))
	}
	// scale = 1 / sqrt(sum/headDim + eps), folded with the per-element weight on
	// the way out.
	mean := b.Mul(ir.F32, acc, b.ConstF32(float32(1)/float32(headDim)))
	inv := b.Div(ir.F32, b.ConstF32(1), b.Sqrt(b.Add(ir.F32, mean, b.ConstF32(float32(eps)))))
	for k := 0; k < per; k++ {
		v := b.Load(ir.F32, pIn, base, int64(k*32))
		w := b.Load(ir.F32, pW, lane, int64(k*32))
		b.Store(pOut, base, b.Mul(ir.F32, b.Mul(ir.F32, v, inv), w), int64(k*32))
	}
	return b.Done()
}

// headNormScalar is one thread per head and two sequential passes: the twin of
// HeadNorm for a device that will not guarantee a 32-lane subgroup, needing no
// cross-lane communication at all. The passes are compile-time loops rather
// than an unroll because headDim can be 2048 (olmoe's whole-projection norm).
func headNormScalar(nHeads, headDim int, eps float64) (*ir.Kernel, error) {
	b := ir.New("headnormscalar", [3]int{64, 1, 1})
	pIn := b.Param("pIn", ir.F32)
	pW := b.Param("pW", ir.F32)
	pOut := b.Param("pOut", ir.F32)

	h := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(nHeads-1)))
	base := b.Mul(ir.U32, h, b.Const(ir.U32, int64(headDim)))
	one := b.Const(ir.U32, 1)
	zero := b.ConstF32(0)
	// Hoisted: a phi's init arrives from the preamble, so a constant emitted
	// inside the body would be refused by ir.Validate.
	izero := b.Const(ir.U32, 0)

	b.Loop(int64(headDim))
	acc := b.Phi(ir.F32, zero)
	si := b.Phi(ir.U32, base)
	v := b.Load(ir.F32, pIn, si, 0)
	b.SetPhi(acc, b.Fma(v, v, acc))
	b.SetPhi(si, b.Add(ir.U32, si, one))
	b.EndLoop()

	mean := b.Mul(ir.F32, acc, b.ConstF32(float32(1)/float32(headDim)))
	inv := b.Div(ir.F32, b.ConstF32(1), b.Sqrt(b.Add(ir.F32, mean, b.ConstF32(float32(eps)))))

	b.Loop(int64(headDim))
	oi := b.Phi(ir.U32, base)
	wi := b.Phi(ir.U32, izero)
	x := b.Load(ir.F32, pIn, oi, 0)
	w := b.Load(ir.F32, pW, wi, 0)
	b.Store(pOut, oi, b.Mul(ir.F32, b.Mul(ir.F32, x, inv), w), 0)
	b.SetPhi(oi, b.Add(ir.U32, oi, one))
	b.SetPhi(wi, b.Add(ir.U32, wi, one))
	b.EndLoop()
	return b.Done()
}

// AttnScoresTiled computes q . k[t] for `rows` consecutive query positions --
// one prefill chunk rather than one token -- with one thread carrying qt query
// rows and kt key positions.
//
// pN carries a header: element 0 is the uniform key count that maps a thread
// to (row, head, position), and elements 1..rows are the per-row causal
// counts. Surplus positions are masked to -inf rather than skipped (no
// conditional outside a loop), so the softmax and the weighted sum can run at
// the uniform width. kStride > 0 reads a TRANSPOSED K cache: element
// (position, e) lives at e*kStride + position (see RoPERowsT for why).
//
// An untiled thread issues two loads per FMA; a 2-D tile loads qt+kt vectors
// for qt*kt scores. A thread's key positions are strided, not adjacent (thread
// j owns j, j+G, j+2G ... with G = ceil(cap/kt) computed from pN), so
// neighbouring threads stay on neighbouring positions and keep coalescing.
//
// maxSeq here is the score row stride and the caller must pad it: G*kt can
// exceed cap by up to kt-1, and an unpadded stride writes into the next row.
// tier passes MaxSeq + maxKeyTile to this, Softmax and AttnAcc alike.
func AttnScoresTiled(nHeads, headDim, kvDim, gqa, maxSeq int, scale float32, rows, qt, kt, kStride int) (*ir.Kernel, error) {
	return AttnScoresTiledW(nHeads, headDim, kvDim, gqa, maxSeq, scale, rows, qt, kt, kStride, 0)
}

// AttnScoresTiledW is AttnScoresTiled for a SLIDING-WINDOW layer: each row
// attends only the last `window` keys of its causal count. 0 is no window, and
// a negative one is an aligned chunk (ChunkWindow).
func AttnScoresTiledW(nHeads, headDim, kvDim, gqa, maxSeq int, scale float32, rows, qt, kt, kStride, window int) (*ir.Kernel, error) {
	if rows <= 1 && qt <= 1 && kt <= 1 {
		return attnScoresOne(nHeads, headDim, kvDim, gqa, maxSeq, scale, kStride, window)
	}
	if gqa <= 0 || nHeads%gqa != 0 {
		return nil, fmt.Errorf("kernels: AttnScoresTiled: gqa=%d does not divide nHeads=%d", gqa, nHeads)
	}
	if qt < 1 || kt < 1 || rows%qt != 0 {
		return nil, fmt.Errorf("kernels: AttnScoresTiled: qt=%d kt=%d does not tile rows=%d", qt, kt, rows)
	}
	b := ir.New("attnscorestiledw", [3]int{128, 1, 1})
	pQ := b.Param("pQ", ir.F32)
	pK := b.Param("pK", ir.F32)
	pN := b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	cap := b.Load(ir.U32, pN, b.Const(ir.U32, 0), 0)
	// G = ceil(cap/kt): how many threads cover the key axis.
	G := b.Div(ir.U32, b.Add(ir.U32, cap, b.Const(ir.U32, int64(kt-1))), b.Const(ir.U32, int64(kt)))
	qgroups := rows / qt
	tid := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	last := b.Sub(ir.U32, b.Mul(ir.U32, G, b.Const(ir.U32, int64(nHeads*qgroups))), b.Const(ir.U32, 1))
	tid = b.Min(ir.U32, tid, last)
	j := b.Rem(ir.U32, tid, G)
	rest := b.Div(ir.U32, tid, G)
	h := b.Rem(ir.U32, rest, b.Const(ir.U32, int64(nHeads)))
	rg := b.Div(ir.U32, rest, b.Const(ir.U32, int64(nHeads)))

	kvh := b.Mul(ir.U32, b.Div(ir.U32, h, b.Const(ir.U32, int64(gqa))), b.Const(ir.U32, int64(headDim)))
	kvd := b.Const(ir.U32, int64(kvDim))
	kStep := int64(1)
	if kStride > 0 {
		kStep = int64(kStride)
	}
	// The kt key positions and the qt query rows this thread owns.
	poss := make([]ir.Value, kt)
	kbase := make([]ir.Value, kt)
	for x := 0; x < kt; x++ {
		poss[x] = j
		if x > 0 {
			poss[x] = b.Add(ir.U32, j, b.Mul(ir.U32, G, b.Const(ir.U32, int64(x))))
		}
		if kStride > 0 {
			kbase[x] = b.Add(ir.U32, b.Mul(ir.U32, kvh, b.Const(ir.U32, int64(kStride))), poss[x])
		} else {
			kbase[x] = b.Add(ir.U32, b.Mul(ir.U32, poss[x], kvd), kvh)
		}
	}
	qrow := b.Mul(ir.U32, rg, b.Const(ir.U32, int64(qt)))
	qbase := make([]ir.Value, qt)
	for t := 0; t < qt; t++ {
		r := qrow
		if t > 0 {
			r = b.Add(ir.U32, qrow, b.Const(ir.U32, int64(t)))
		}
		qbase[t] = b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(nHeads))), h),
			b.Const(ir.U32, int64(headDim)))
	}
	accs := make([]ir.Value, qt*kt)
	for i := range accs {
		accs[i] = b.ConstF32(0)
	}
	// qt + kt loads for qt*kt FMAs.
	for i := 0; i < headDim; i++ {
		qv := make([]ir.Value, qt)
		for t := 0; t < qt; t++ {
			qv[t] = b.Load(ir.F32, pQ, qbase[t], int64(i))
		}
		for x := 0; x < kt; x++ {
			kv := b.Load(ir.F32, pK, kbase[x], int64(i)*kStep)
			for t := 0; t < qt; t++ {
				accs[t*kt+x] = b.Fma(qv[t], kv, accs[t*kt+x])
			}
		}
	}
	negInf := b.Bitcast(ir.F32, b.Const(ir.U32, 0xFF800000))
	sc := b.ConstF32(scale)
	for t := 0; t < qt; t++ {
		r := qrow
		if t > 0 {
			r = b.Add(ir.U32, qrow, b.Const(ir.U32, int64(t)))
		}
		cnt := b.Load(ir.U32, pN, b.Add(ir.U32, r, b.Const(ir.U32, 1)), 0)
		row := b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(nHeads))), h),
			b.Const(ir.U32, int64(maxSeq)))
		for x := 0; x < kt; x++ {
			v := b.Select(ir.F32, b.Lt(ir.U32, poss[x], cnt),
				b.Mul(ir.F32, accs[t*kt+x], sc), negInf)
			v = windowed(b, v, poss[x], cnt, window, negInf)
			b.Store(pOut, b.Add(ir.U32, row, poss[x]), v, 0)
		}
	}
	return b.Done()
}

// AttnAccTiled is AttnAcc over `rows` query positions, each walking only its
// own causal count (pN[1+row]), with qt query rows to a thread so one V load
// serves all of them. Walking the uniform width would multiply zero
// probabilities by V past the prompt, which nothing has written: 0*NaN is NaN.
// A warp never straddles two rows, so the varying count does not diverge.
//
// A thread walks its group's widest count, so the lower rows read
// probabilities past their own bound; those are zero only because SoftmaxRows
// takes the same qt and normalises that far. Get the two out of step and the
// reads are stale probabilities against real V.
func AttnAccTiled(nHeads, headDim, kvDim, gqa, maxSeq, rows, qt int) (*ir.Kernel, error) {
	if rows <= 1 {
		return AttnAcc(nHeads, headDim, kvDim, gqa, maxSeq)
	}
	if qt < 1 || rows%qt != 0 {
		return nil, fmt.Errorf("kernels: AttnAccTiled: qt=%d does not divide rows=%d", qt, rows)
	}
	b := ir.New("attnacctiled", [3]int{128, 1, 1})
	pA := b.Param("pA", ir.F32)
	pV := b.Param("pV", ir.F32)
	pN := b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(nHeads*headDim*(rows/qt)-1)))
	perRow := b.Const(ir.U32, int64(nHeads*headDim))
	tid := b.Rem(ir.U32, flat, perRow)
	rg := b.Div(ir.U32, flat, perRow)
	row := rg
	if qt > 1 {
		row = b.Mul(ir.U32, rg, b.Const(ir.U32, int64(qt)))
	}
	// The group's widest causal count: the last row of the tile.
	n := b.Load(ir.U32, pN, b.Add(ir.U32, row, b.Const(ir.U32, int64(qt))), 0)
	i := b.Rem(ir.U32, tid, b.Const(ir.U32, int64(headDim)))
	h := b.Div(ir.U32, tid, b.Const(ir.U32, int64(headDim)))
	// The score row is (row*nHeads + h); one query row on is nHeads*maxSeq
	// further along, a compile-time displacement.
	arow := b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, row, b.Const(ir.U32, int64(nHeads))), h),
		b.Const(ir.U32, int64(maxSeq)))
	vcol := b.Add(ir.U32, b.Mul(ir.U32, b.Div(ir.U32, h, b.Const(ir.U32, int64(gqa))),
		b.Const(ir.U32, int64(headDim))), i)
	kvd := b.Const(ir.U32, int64(kvDim))
	zero := b.ConstF32(0)

	b.LoopN(n)
	accs := make([]ir.Value, qt)
	for u := range accs {
		accs[u] = b.Phi(ir.F32, zero)
	}
	ai := b.Phi(ir.U32, arow)
	vi := b.Phi(ir.U32, vcol)
	v := b.Load(ir.F32, pV, vi, 0)
	for u := 0; u < qt; u++ {
		b.SetPhi(accs[u], b.Fma(b.Load(ir.F32, pA, ai, int64(u*nHeads*maxSeq)), v, accs[u]))
	}
	b.SetPhi(ai, b.Add(ir.U32, ai, b.Const(ir.U32, 1)))
	b.SetPhi(vi, b.Add(ir.U32, vi, kvd))
	b.EndLoop()
	// The output index is built from the row, not from the thread: rg counts
	// thread groups, and at qt > 1 the row it serves is rg*qt.
	outBase := b.Add(ir.U32, b.Mul(ir.U32, row, perRow), tid)
	for u := 0; u < qt; u++ {
		b.Store(pOut, outBase, accs[u], int64(u*nHeads*headDim))
	}
	return b.Done()
}

// HeadNormRows applies HeadNorm to `rows` consecutive vectors. At lanes=32 that
// is one subgroup per (row, head), so the launch is rows*nHeads groups; at
// lanes=1 it is one thread per (row, head) and the caller divides by the group
// width. See HeadNorm for what lanes means and who chooses it.
func HeadNormRows(nHeads, headDim int, eps float64, rows, lanes int) (*ir.Kernel, error) {
	if rows <= 1 {
		return HeadNorm(nHeads, headDim, eps, lanes)
	}
	return HeadNorm(nHeads*rows, headDim, eps, lanes)
}

// AttnScoresMMA computes a prefill chunk's attention scores with the warp
// matrix instruction. The tile is 16 query rows by 8 key positions,
// accumulated over headDim in steps of 16:
//
//	A = q[row][k]      row-major, 16 query rows x 16 elements of the head
//	B = k[k][position] column-major, which is how the transposed K cache
//	                   already stores it
//	D = 16 x 8 float32 scores
//
// The operands drop to binary16 and the accumulator stays float32, so the gate
// is NMSE and token ids rather than bit equality. The per-row causal mask is
// applied to the result.
func AttnScoresMMA(nHeads, headDim, kvDim, gqa, maxSeq int, scale float32, rows, kStride, nt int) (*ir.Kernel, error) {
	return AttnScoresMMAW(nHeads, headDim, kvDim, gqa, maxSeq, scale, rows, kStride, nt, 0)
}

// AttnScoresMMAW is AttnScoresMMA for a sliding-window layer; see windowed.
func AttnScoresMMAW(nHeads, headDim, kvDim, gqa, maxSeq int, scale float32, rows, kStride, nt, window int) (*ir.Kernel, error) {
	if gqa <= 0 || nHeads%gqa != 0 {
		return nil, fmt.Errorf("kernels: AttnScoresMMA: gqa=%d does not divide nHeads=%d", gqa, nHeads)
	}
	if kStride <= 0 {
		return nil, fmt.Errorf("kernels: AttnScoresMMA: needs the transposed K cache")
	}
	if headDim%16 != 0 || rows%16 != 0 {
		return nil, fmt.Errorf("kernels: AttnScoresMMA: headDim=%d rows=%d must be multiples of 16", headDim, rows)
	}
	if nt < 1 {
		nt = 1
	}
	sh := ir.MMAShape{M: 16, N: 8, K: 16, Kind: ir.MMAF16}
	na, nb, nc := sh.Frags()

	const group = 128
	b := ir.New("attnscoresmmaw", [3]int{group, 1, 1})
	pQ := b.Param("pQ", ir.F32)
	pK := b.Param("pK", ir.F32)
	pN := b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.F32)

	cap := b.Load(ir.U32, pN, b.Const(ir.U32, 0), 0)
	// Key tiles: 8 positions each, nt of them to a warp.
	ktiles := b.Div(ir.U32, b.Add(ir.U32, cap, b.Const(ir.U32, int64(8*nt-1))),
		b.Const(ir.U32, int64(8*nt)))
	qgroups := rows / 16
	tid := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	warp := b.Div(ir.U32, tid, b.Const(ir.U32, 32))
	lane := b.And(ir.U32, tid, b.Const(ir.U32, 31))
	g := b.Shr(ir.U32, lane, b.Const(ir.U32, 2))
	t := b.And(ir.U32, lane, b.Const(ir.U32, 3))
	two := b.Mul(ir.U32, t, b.Const(ir.U32, 2))

	last := b.Sub(ir.U32, b.Mul(ir.U32, ktiles, b.Const(ir.U32, int64(nHeads*qgroups))),
		b.Const(ir.U32, 1))
	warp = b.Min(ir.U32, warp, last)
	// The key tile varies fastest so neighbouring warps walk neighbouring
	// positions, which is what the transposed cache coalesces on.
	kt := b.Rem(ir.U32, warp, ktiles)
	rest := b.Div(ir.U32, warp, ktiles)
	h := b.Rem(ir.U32, rest, b.Const(ir.U32, int64(nHeads)))
	qg := b.Div(ir.U32, rest, b.Const(ir.U32, int64(nHeads)))

	qrow := b.Mul(ir.U32, qg, b.Const(ir.U32, 16))
	// A's rows: this lane holds query rows qrow+g and qrow+g+8.
	qbase := b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, qrow, g),
		b.Const(ir.U32, int64(nHeads))), h), b.Const(ir.U32, int64(headDim)))
	qbase = b.Add(ir.U32, qbase, two)
	// B's columns: key positions kt*8*nt + j*8 + g, at element rows 2t and 2t+8.
	kvh := b.Mul(ir.U32, b.Div(ir.U32, h, b.Const(ir.U32, int64(gqa))), b.Const(ir.U32, int64(headDim)))
	pos0 := b.Add(ir.U32, b.Mul(ir.U32, kt, b.Const(ir.U32, int64(8*nt))), g)
	kbase := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32, kvh, two),
		b.Const(ir.U32, int64(kStride))), pos0)

	zero := b.ConstF32(0)
	accs := make([][]ir.Value, nt)
	for j := range accs {
		accs[j] = make([]ir.Value, nc)
		for c := range accs[j] {
			accs[j][c] = zero
		}
	}
	// headDim is fully unrolled, so every offset is an immediate on one base.
	for kk := 0; kk < headDim; kk += 16 {
		af := make([]ir.Value, na)
		for i := 0; i < na; i++ {
			// register i: row g (+8 when odd), element 2t (+8 when i >= 2).
			off := int64(kk)
			if i >= 2 {
				off += 8
			}
			row := int64(0)
			if i%2 == 1 {
				row = 8 * int64(nHeads) * int64(headDim)
			}
			af[i] = b.PackF16(b.Load(ir.F32, pQ, qbase, off+row),
				b.Load(ir.F32, pQ, qbase, off+row+1))
		}
		for j := 0; j < nt; j++ {
			bf := make([]ir.Value, nb)
			for i := 0; i < nb; i++ {
				// register i: element 2t (+8 when i >= 1), position column j*8.
				e := int64(kk)
				if i >= 1 {
					e += 8
				}
				o := e*int64(kStride) + int64(j*8)
				bf[i] = b.PackF16(b.Load(ir.F32, pK, kbase, o),
					b.Load(ir.F32, pK, kbase, o+int64(kStride)))
			}
			accs[j] = b.MMA(sh, af, bf, accs[j])
		}
	}

	negInf := b.Bitcast(ir.F32, b.Const(ir.U32, 0xFF800000))
	// The scale is applied to the float32 result, not folded into q, which
	// would round it into binary16 twice.
	sc := b.ConstF32(scale)
	for j := 0; j < nt; j++ {
		for c := 0; c < nc; c++ {
			// D component c: query row qrow+g (+8 when c >= 2), key column
			// kt*8*nt + j*8 + 2t + c%2.
			rr := b.Add(ir.U32, qrow, g)
			if c >= 2 {
				rr = b.Add(ir.U32, rr, b.Const(ir.U32, 8))
			}
			pp := b.Add(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, kt, b.Const(ir.U32, int64(8*nt))),
				two), b.Const(ir.U32, int64(j*8+c%2)))
			cnt := b.Load(ir.U32, pN, b.Add(ir.U32, rr, b.Const(ir.U32, 1)), 0)
			v := b.Select(ir.F32, b.Lt(ir.U32, pp, cnt), b.Mul(ir.F32, accs[j][c], sc), negInf)
			v = windowed(b, v, pp, cnt, window, negInf)
			out := b.Add(ir.U32, b.Mul(ir.U32, b.Add(ir.U32,
				b.Mul(ir.U32, rr, b.Const(ir.U32, int64(nHeads))), h),
				b.Const(ir.U32, int64(maxSeq))), pp)
			b.Store(pOut, out, v, 0)
		}
	}
	return b.Done()
}

// AttnScoresRagged is AttnScoresTiledW at one query row per tile for a batch
// whose rows are different sequences: row r's keys are positions
// pBase[r] .. pBase[r]+pN[1+r]-1 of a cache that holds every sequence in its
// own region, and its scores land at 0 .. pN[1+r]-1 of its own score row, so
// the softmax and the accumulate see exactly what one sequence's decode sees.
// pN[0] is the widest count, which sizes the grid, as in AttnScoresTiledW.
//
// It is a separate kernel rather than a flag on the tiled one because a query
// tile only works when its rows share a key range, and a batch of sequences
// shares none.
func AttnScoresRagged(nHeads, headDim, kvDim, gqa, maxSeq int, scale float32, rows, kt, kStride, window int) (*ir.Kernel, error) {
	if gqa <= 0 || nHeads%gqa != 0 || kt < 1 || rows < 1 {
		return nil, fmt.Errorf("kernels: AttnScoresRagged: gqa=%d nHeads=%d kt=%d rows=%d", gqa, nHeads, kt, rows)
	}
	b := ir.New("attnscoresragged", [3]int{128, 1, 1})
	pQ := b.Param("pQ", ir.F32)
	pK := b.Param("pK", ir.F32)
	pN := b.Param("pN", ir.U32)
	pBase := b.Param("pBase", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	cap := b.Load(ir.U32, pN, b.Const(ir.U32, 0), 0)
	G := b.Div(ir.U32, b.Add(ir.U32, cap, b.Const(ir.U32, int64(kt-1))), b.Const(ir.U32, int64(kt)))
	tid := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	last := b.Sub(ir.U32, b.Mul(ir.U32, G, b.Const(ir.U32, int64(nHeads*rows))), b.Const(ir.U32, 1))
	tid = b.Min(ir.U32, tid, last)
	j := b.Rem(ir.U32, tid, G)
	rest := b.Div(ir.U32, tid, G)
	h := b.Rem(ir.U32, rest, b.Const(ir.U32, int64(nHeads)))
	r := b.Div(ir.U32, rest, b.Const(ir.U32, int64(nHeads)))
	base := b.Load(ir.U32, pBase, r, 0)
	cnt := b.Load(ir.U32, pN, b.Add(ir.U32, r, b.Const(ir.U32, 1)), 0)
	// A position past this row's count reads the row's own last key rather
	// than whatever lies past it; its score is masked below.
	lastKey := b.Sub(ir.U32, b.Max(ir.U32, cnt, b.Const(ir.U32, 1)), b.Const(ir.U32, 1))

	kvh := b.Mul(ir.U32, b.Div(ir.U32, h, b.Const(ir.U32, int64(gqa))), b.Const(ir.U32, int64(headDim)))
	kvd := b.Const(ir.U32, int64(kvDim))
	kStep := int64(1)
	if kStride > 0 {
		kStep = int64(kStride)
	}
	poss := make([]ir.Value, kt)
	kbase := make([]ir.Value, kt)
	for x := 0; x < kt; x++ {
		poss[x] = j
		if x > 0 {
			poss[x] = b.Add(ir.U32, j, b.Mul(ir.U32, G, b.Const(ir.U32, int64(x))))
		}
		key := b.Add(ir.U32, base, b.Min(ir.U32, poss[x], lastKey))
		if kStride > 0 {
			kbase[x] = b.Add(ir.U32, b.Mul(ir.U32, kvh, b.Const(ir.U32, int64(kStride))), key)
		} else {
			kbase[x] = b.Add(ir.U32, b.Mul(ir.U32, key, kvd), kvh)
		}
	}
	qbase := b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(nHeads))), h),
		b.Const(ir.U32, int64(headDim)))
	accs := make([]ir.Value, kt)
	for i := range accs {
		accs[i] = b.ConstF32(0)
	}
	for i := 0; i < headDim; i++ {
		qv := b.Load(ir.F32, pQ, qbase, int64(i))
		for x := 0; x < kt; x++ {
			accs[x] = b.Fma(qv, b.Load(ir.F32, pK, kbase[x], int64(i)*kStep), accs[x])
		}
	}
	negInf := b.Bitcast(ir.F32, b.Const(ir.U32, 0xFF800000))
	sc := b.ConstF32(scale)
	row := b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, r, b.Const(ir.U32, int64(nHeads))), h),
		b.Const(ir.U32, int64(maxSeq)))
	for x := 0; x < kt; x++ {
		v := b.Select(ir.F32, b.Lt(ir.U32, poss[x], cnt), b.Mul(ir.F32, accs[x], sc), negInf)
		v = windowed(b, v, poss[x], cnt, window, negInf)
		b.Store(pOut, b.Add(ir.U32, row, poss[x]), v, 0)
	}
	return b.Done()
}

// AttnAccRagged is AttnAccTiled for a batch of different sequences: row r
// accumulates values pBase[r] .. pBase[r]+pN[1+r]-1 of the cache under its
// own probabilities. See AttnScoresRagged.
func AttnAccRagged(nHeads, headDim, kvDim, gqa, maxSeq, rows int) (*ir.Kernel, error) {
	if gqa <= 0 || nHeads%gqa != 0 || rows < 1 {
		return nil, fmt.Errorf("kernels: AttnAccRagged: gqa=%d nHeads=%d rows=%d", gqa, nHeads, rows)
	}
	b := ir.New("attnaccragged", [3]int{128, 1, 1})
	pA := b.Param("pA", ir.F32)
	pV := b.Param("pV", ir.F32)
	pN := b.Param("pN", ir.U32)
	pBase := b.Param("pBase", ir.U32)
	pOut := b.Param("pOut", ir.F32)
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		b.Const(ir.U32, int64(nHeads*headDim*rows-1)))
	perRow := b.Const(ir.U32, int64(nHeads*headDim))
	tid := b.Rem(ir.U32, flat, perRow)
	row := b.Div(ir.U32, flat, perRow)
	n := b.Load(ir.U32, pN, b.Add(ir.U32, row, b.Const(ir.U32, 1)), 0)
	base := b.Load(ir.U32, pBase, row, 0)
	i := b.Rem(ir.U32, tid, b.Const(ir.U32, int64(headDim)))
	h := b.Div(ir.U32, tid, b.Const(ir.U32, int64(headDim)))
	arow := b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, row, b.Const(ir.U32, int64(nHeads))), h),
		b.Const(ir.U32, int64(maxSeq)))
	kvd := b.Const(ir.U32, int64(kvDim))
	vcol := b.Add(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.Div(ir.U32, h, b.Const(ir.U32, int64(gqa))),
		b.Const(ir.U32, int64(headDim))), i), b.Mul(ir.U32, base, kvd))
	zero := b.ConstF32(0)

	b.LoopN(n)
	acc := b.Phi(ir.F32, zero)
	ai := b.Phi(ir.U32, arow)
	vi := b.Phi(ir.U32, vcol)
	b.SetPhi(acc, b.Fma(b.Load(ir.F32, pA, ai, 0), b.Load(ir.F32, pV, vi, 0), acc))
	b.SetPhi(ai, b.Add(ir.U32, ai, b.Const(ir.U32, 1)))
	b.SetPhi(vi, b.Add(ir.U32, vi, kvd))
	b.EndLoop()
	b.Store(pOut, b.Add(ir.U32, b.Mul(ir.U32, row, perRow), tid), acc, 0)
	return b.Done()
}
