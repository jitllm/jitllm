package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestMatVecMMAMatchesDot4 holds the tensor-core prefill GEMM to the dp4a one,
// bit for bit, on real hardware.
//
// Both kernels reduce each sub-block to an exact int32 dot product and apply
// the same float fold in the same order, so any difference is a defect (a
// transposed fragment, a row off by eight, a scale for the wrong sub-block)
// that an NMSE bar would swallow.
//
// The dp4a arm is itself gated against a float64 reference by
// TestMatVecMatchesReference, so this inherits that.
func TestMatVecMMAMatchesDot4(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
	}
	// Ask once per device rather than once per cell of the product; see
	// mmaDevices and lowertest's TestMMACoversExactlyWhatItClaims.
	// sm_75's m8n8k16 is the widest net: there MatVecMMA declines and only
	// GemmInt8's M8 tiles run (int8M8Tiles); from sm_80 every arm runs.
	for _, d := range mmaDevices(t, devs, ir.MMAS8Turing()) {
		t.Run(d.API(), func(t *testing.T) {
			for _, q := range []kernels.Quant{kernels.Q4_0, kernels.Q8_0, kernels.Q4_K,
				kernels.Q3_K, kernels.Q5_K, kernels.Q6_K, kernels.MXFP4, kernels.Q5_0, kernels.Q5_1, kernels.TQ2, kernels.TQ2F} {
				for _, sp := range []struct{ rows, k, ntok, mt, nt int }{
					{16, 256, 8, 1, 1},
					{64, 256, 16, 2, 2},
					{256, 2048, 128, 4, 4},
					{2048, 2048, 128, 4, 4},
					// A shape whose rows are not a multiple of the widest tile,
					// so the fallback to a narrower one is exercised.
					{5632, 2048, 128, 2, 4},
					// One token group, which is where an n-tile bug hides.
					{128, 512, 8, 4, 1},
				} {
					name := fmt.Sprintf("%s/%dx%d/ntok%d/%dx%d", q, sp.rows, sp.k, sp.ntok, sp.mt, sp.nt)
					t.Run(name, func(t *testing.T) {
						mmaMVCase(t, d, q, sp.rows, sp.k, sp.ntok, sp.mt, sp.nt)
					})
				}
			}
		})
	}
}

func mmaMVCase(t *testing.T, d backend.Device, q kernels.Quant, nrows, k, ntok, mt, nt int) {
	t.Helper()
	// Both activation windows: ActWin 256 lets the k-quant fold accumulate in
	// integer across a super-block (Q3_K and Q6_K only), 32 keeps the
	// per-sub-block float fold. Either must give the same answer.
	for _, win := range []int{32, 256} {
		t.Run(fmt.Sprintf("win%d", win), func(t *testing.T) {
			mmaMVCaseWin(t, d, q, nrows, k, ntok, mt, nt, win)
		})
	}
}

func mmaMVCaseWin(t *testing.T, d backend.Device, q kernels.Quant, nrows, k, ntok, mt, nt, win int) {
	t.Helper()
	shape := kernels.MatVecShape{T: q, K: k, Rows: nrows, NTok: ntok, MT: mt, NT: nt, ActWin: win}
	mk, err := kernels.MatVecMMA(shape)
	if err != nil {
		t.Skipf("shape rejected: %v", err)
	}
	// A declined MMA does not skip the case: the dp4a centering check below is
	// the only gate Metal and Vulkan have for their prefill matvec.
	mkern, mmaErr := d.Compile(mk)
	if mmaErr == nil {
		defer mkern.Close()
	} else {
		// Metal has no integer matrix instruction and Vulkan's is an opaque
		// extension; both decline in their lowerers, by name and with a reason.
		mkern = nil
	}
	// The dp4a arm at Tok=1, the plainest form of the batched kernel.
	dk, err := kernels.MatVec(kernels.MatVecShape{T: q, K: k, Rows: nrows, NTok: ntok, Tok: 1})
	if err != nil {
		t.Fatal(err)
	}
	dkern, err := d.Compile(dk)
	if err != nil {
		t.Fatal(err)
	}
	defer dkern.Close()

	rng := rand.New(rand.NewSource(int64(nrows*7 + k*13 + ntok + int(q)*101)))
	nsuper := k / q.Elems()
	raw := make([]byte, nrows*nsuper*q.BlockBytes())
	for i := range raw {
		raw[i] = byte(rng.Intn(256))
	}
	// The f16 scales are normalised, or random bits land on infinities and the
	// oracle is NaN. Every format's scale offsets are covered.
	setF16 := func(off int) {
		binary.LittleEndian.PutUint16(raw[off:], uint16(0x2000|rng.Intn(0x0C00)))
	}
	for r := 0; r < nrows; r++ {
		for sb := 0; sb < nsuper; sb++ {
			blk := (r*nsuper + sb) * q.BlockBytes()
			switch q {
			case kernels.Q4_0, kernels.Q8_0, kernels.Q5_0:
				setF16(blk)
			case kernels.Q4_K, kernels.Q5_K, kernels.Q5_1:
				setF16(blk)
				setF16(blk + 2)
			case kernels.Q3_K:
				setF16(blk + 108)
			case kernels.Q6_K:
				setF16(blk + 208)
			case kernels.TQ2:
				setF16(blk + 64)
			case kernels.TQ2F:
				binary.LittleEndian.PutUint32(raw[blk+64:], math.Float32bits(0.5+rng.Float32()))
			case kernels.MXFP4:
				// An E8M0 exponent the f16 plane holds exactly, or the packer
				// refuses the tensor.
				raw[blk] = byte(120 + rng.Intn(10))
			default:
				// A format with no arm fails rather than leaving its scales random.
				t.Fatalf("%s: no scale planter here", q)
			}
		}
	}
	x := make([]float32, ntok*k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	qs, dw, scw, err := kernels.PackWeights(q, raw, nrows, k)
	if err != nil {
		t.Fatal(err)
	}
	// The activations must be packed at the window the kernel was built for:
	// the integer fold assumes one activation scale covers the super-block,
	// and PackActivations defaults to a 32-element window.
	nbq := len(x) / 32
	av := make([]uint32, len(x)/4)
	as := make([]float32, nbq)
	asum := make([]float32, 2*nbq)
	if err := kernels.PackActivationsInto(av, as, asum, x, win); err != nil {
		t.Fatal(err)
	}

	var bufs []backend.Buf
	defer func() {
		for _, b := range bufs {
			b.Free()
		}
	}()
	up := func(n int, p []byte) backend.Buf {
		b, err := d.Alloc(n)
		if err != nil {
			t.Fatal(err)
		}
		if p != nil {
			if err := b.Write(p); err != nil {
				t.Fatal(err)
			}
		}
		bufs = append(bufs, b)
		return b
	}
	bQS := up(len(qs)*4, u32bytes(qs))
	bD := up(len(dw)*4, u32bytes(dw))
	bSC := up(4, nil)
	if len(scw) > 0 {
		bSC = up(len(scw)*4, u32bytes(scw))
	}
	bA := up(len(av)*4, u32bytes(av))
	bAX := up((len(as)+len(asum))*4, f32bytes(append(append([]float32{}, as...), asum...)))
	bMMA := up(nrows*ntok*4, nil)
	bDot := up(nrows*ntok*4, nil)
	bCen := up(nrows*ntok*4, nil)

	run := func(kern backend.Kernel, threads int, out backend.Buf) []byte {
		w := 128
		if err := kern.Launch((threads+w-1)/w, w, bQS, bD, bSC, bA, bAX, out); err != nil {
			t.Fatal(err)
		}
		p := make([]byte, nrows*ntok*4)
		if err := out.Read(p); err != nil {
			t.Fatal(err)
		}
		return p
	}
	gotD := run(dkern, nrows*ntok, bDot)

	// The dp4a centering gate. Weight centering turns on at Tok >= 8, which
	// Metal and Vulkan run for every prefill matvec. The centered dot is the
	// corrected dot exactly, so Tok=1 against Tok>=8 must be bit-identical per
	// column: Tok changes how many columns a thread carries, not the summation
	// order within one.
	for _, tk := range []int{8, 16} {
		if ntok%tk != 0 {
			continue
		}
		ck, err := kernels.MatVec(kernels.MatVecShape{T: q, K: k, Rows: nrows, NTok: ntok, Tok: tk})
		if err != nil {
			continue
		}
		ckern, err := d.Compile(ck)
		if err != nil {
			t.Fatalf("centered Tok=%d declined: %v", tk, err)
		}
		gotC := run(ckern, nrows*(ntok/tk), bCen)
		ckern.Close()
		diff, cworst, csse, csy2 := 0, 0.0, 0.0, 0.0
		for i := 0; i < nrows*ntok; i++ {
			c := math.Float32frombits(binary.LittleEndian.Uint32(gotC[i*4:]))
			dv := math.Float32frombits(binary.LittleEndian.Uint32(gotD[i*4:]))
			e := float64(c) - float64(dv)
			csse, csy2 = csse+e*e, csy2+float64(dv)*float64(dv)
			if c != dv {
				diff++
				if a := math.Abs(e); a > cworst {
					cworst = a
				}
			}
		}
		cnmse := 0.0
		if csy2 > 0 {
			cnmse = csse / csy2
		}
		if !math.IsNaN(csy2) && csy2 > 0 && diff != 0 {
			// Two bars, chosen by whether centering is live for this format,
			// read from the same Layout() the kernel reads.
			if _, _, bk, barr := kernels.Layout(q); bk != 0 && !barr {
				t.Fatalf("centering at Tok=%d: %d/%d elements differ, worst |delta| %g, "+
					"NMSE %.3e -- the centered dot must equal the uncentered one exactly",
					tk, diff, nrows*ntok, cworst, cnmse)
			} else if cnmse > 1e-8 {
				// The uncentered formats (Q4_K, Q5_K, with a per-sub-block min
				// array) are not Tok-invariant on SPIR-V, though PTX is exact:
				// a lowering difference in the min term's fold, bounded at
				// NMSE 1e-8 and undiagnosed (see AGENTS.md, Open questions).
				t.Fatalf("Tok=%d on an uncentered format: NMSE %.3e over %d/%d elements, "+
					"worst |delta| %g -- above the reassociation band",
					tk, cnmse, diff, nrows*ntok, cworst)
			}
		}
	}

	// GemmInt8 against the same dp4a arm: the staged int8 GEMM applies the
	// identical fold, so it is held bit for bit unsplit. Without the m16n8
	// integer shapes (sm_75) its m8n8k16 tiles still run.
	tiles := int8Tiles
	if mkern == nil {
		if d.API() != "ptx" {
			t.Skipf("no matrix instruction: %v", mmaErr)
		}
		tiles = int8M8Tiles
	}
	gemmInt8Case(t, d, q, nrows, k, ntok, gotD, tiles, func(kern backend.Kernel, groups, w, n int) []byte {
		out := up(n*4, f32bytes(nanFill(n)))
		if err := kern.Launch(groups, w, bQS, bD, bSC, bA, bAX, out); err != nil {
			t.Fatal(err)
		}
		p := make([]byte, n*4)
		if err := out.Read(p); err != nil {
			t.Fatal(err)
		}
		return p
	})
	if mkern == nil {
		t.Skipf("no m16n8 integer instruction (%v): GemmInt8's m8n8k16 tiles held, MatVecMMA not run", mmaErr)
	}
	// One warp per (m,n) tile, 128 threads to a group.
	warps := (nrows / (16 * mt)) * (ntok / (8 * nt))
	gotM := run(mkern, warps*32, bMMA)

	// The two kernels apply the identical float fold per sub-block over
	// integer-exact dots, so equality is achievable and anything else is a
	// defect. `exact` is a named constant because an integer fold, if it is
	// rebuilt, is more accurate and would need an NMSE bar instead -- see
	// MatVecMMA.
	const exact = true
	bad, nz := 0, 0
	var worst, sse, sy2 float64
	for i := 0; i < nrows*ntok; i++ {
		m := math.Float32frombits(binary.LittleEndian.Uint32(gotM[i*4:]))
		dv := math.Float32frombits(binary.LittleEndian.Uint32(gotD[i*4:]))
		if dv != 0 {
			nz++
		}
		if math.IsNaN(float64(dv)) || math.IsInf(float64(dv), 0) {
			t.Fatalf("the dp4a arm is not finite at %d (%v); the oracle is broken", i, dv)
		}
		e := float64(m) - float64(dv)
		sse += e * e
		sy2 += float64(dv) * float64(dv)
		if m != dv {
			if math.Abs(e) > worst {
				worst = math.Abs(e)
			}
			if exact && bad < 5 {
				t.Errorf("out[tok %d][row %d]: mma %v, dp4a %v", i/nrows, i%nrows, m, dv)
			}
			bad++
		}
	}
	if nz < nrows*ntok/2 {
		t.Fatalf("only %d/%d reference values are non-zero; the oracle is degenerate", nz, nrows*ntok)
	}
	if exact {
		if bad > 0 {
			t.Fatalf("%d/%d elements differ, worst |delta| %g -- this path must be bit-identical",
				bad, nrows*ntok, worst)
		}
		return
	}
	if nmse := sse / sy2; nmse > 1e-10 || math.IsNaN(nmse) {
		t.Fatalf("integer fold: NMSE %.3e (worst |delta| %g over %d/%d elements)",
			nmse, worst, bad, nrows*ntok)
	}
}

// TestAttnScoresMMAMatchesFMA holds the tensor-core scores kernel to the FMA one
// on real hardware.
//
// The bar is NMSE, not equality: the matrix instruction multiplies binary16
// operands into a float32 accumulator, where the FMA kernel keeps float32
// throughout. binary16's 11 bits give an NMSE near 2.5e-7 on a 64-element dot;
// 1e-5 leaves margin and is still far below a transposed fragment or a wrong
// causal count (~1e-1).
func TestAttnScoresMMAMatchesFMA(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
	}
	// Attention's tile is the FLOAT one, and a backend can lower one kind and
	// not the other -- so this probes MMAF16 rather than reusing the s8 answer.
	for _, d := range mmaDevices(t, devs, ir.MMAShape{M: 16, N: 8, K: 16, Kind: ir.MMAF16}) {
		t.Run(d.API(), func(t *testing.T) {
			for _, c := range []struct{ nHeads, headDim, nKV, rows, cap int }{
				{32, 64, 4, 128, 128},  // tinyllama, first chunk
				{32, 64, 4, 128, 384},  // and a later one
				{8, 256, 1, 32, 64},    // gemma: one kv head, wide
				{16, 128, 8, 64, 200},  // qwen3, a cap that is not a tile multiple
				{32, 96, 32, 128, 129}, // phi3, no GQA, ragged cap
			} {
				name := fmt.Sprintf("h%d/d%d/kv%d/rows%d/cap%d", c.nHeads, c.headDim, c.nKV, c.rows, c.cap)
				t.Run(name, func(t *testing.T) {
					scoresMMACase(t, d, c.nHeads, c.headDim, c.nKV, c.rows, c.cap)
				})
			}
		})
	}
}

func scoresMMACase(t *testing.T, d backend.Device, nHeads, headDim, nKV, rows, cap int) {
	t.Helper()
	const maxSeq = 1024
	kvDim, gqa := nKV*headDim, nHeads/nKV
	sstride, kStride := maxSeq+8, maxSeq+1
	scale := float32(1) / float32(math.Sqrt(float64(headDim)))

	mk, err := kernels.AttnScoresMMA(nHeads, headDim, kvDim, gqa, sstride, scale, rows, kStride, 1)
	if err != nil {
		t.Skipf("shape rejected: %v", err)
	}
	mkern, err := d.Compile(mk)
	if err != nil {
		t.Skipf("declined: %v", err)
	}
	defer mkern.Close()
	fk, err := kernels.AttnScoresTiled(nHeads, headDim, kvDim, gqa, sstride, scale, rows, 4, 2, kStride)
	if err != nil {
		t.Fatal(err)
	}
	fkern, err := d.Compile(fk)
	if err != nil {
		t.Fatal(err)
	}
	defer fkern.Close()

	rng := rand.New(rand.NewSource(int64(nHeads*17 + headDim + rows + cap)))
	q := make([]float32, rows*nHeads*headDim)
	for i := range q {
		q[i] = float32(rng.NormFloat64())
	}
	// K in the transposed device layout: element e of position p at e*kStride+p.
	kc := make([]float32, kvDim*kStride)
	for i := range kc {
		kc[i] = float32(rng.NormFloat64())
	}
	// pN: element 0 the uniform width, then each row's causal count.
	pn := make([]uint32, 1+rows)
	pn[0] = uint32(cap)
	for r := 0; r < rows; r++ {
		n := cap - rows + 1 + r
		if n < 1 {
			n = 1
		}
		pn[1+r] = uint32(n)
	}

	var bufs []backend.Buf
	defer func() {
		for _, b := range bufs {
			b.Free()
		}
	}()
	up := func(n int, p []byte) backend.Buf {
		b, err := d.Alloc(n)
		if err != nil {
			t.Fatal(err)
		}
		if p != nil {
			if err := b.Write(p); err != nil {
				t.Fatal(err)
			}
		}
		bufs = append(bufs, b)
		return b
	}
	bQ := up(len(q)*4, f32bytes(q))
	bK := up(len(kc)*4, f32bytes(kc))
	bN := up(len(pn)*4, u32bytes(pn))
	nOut := rows * nHeads * sstride
	bM, bF := up(nOut*4, nil), up(nOut*4, nil)

	run := func(kern backend.Kernel, threads int, out backend.Buf) []float32 {
		if err := kern.Launch((threads+127)/128, 128, bQ, bK, bN, out); err != nil {
			t.Fatal(err)
		}
		raw := make([]byte, nOut*4)
		if err := out.Read(raw); err != nil {
			t.Fatal(err)
		}
		v := make([]float32, nOut)
		for i := range v {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return v
	}
	kt := (cap + 7) / 8
	gotM := run(mkern, (rows/16)*nHeads*kt*32, bM)
	gotF := run(fkern, (rows/4)*nHeads*((cap+1)/2), bF)

	var sse, sy2 float64
	nz, bad := 0, 0
	for r := 0; r < rows; r++ {
		for h := 0; h < nHeads; h++ {
			for p := 0; p < int(pn[1+r]); p++ {
				i := (r*nHeads+h)*sstride + p
				m, f := gotM[i], gotF[i]
				if f != 0 {
					nz++
				}
				if math.IsInf(float64(f), 0) || math.IsNaN(float64(f)) {
					t.Fatalf("the FMA arm is not finite at row %d head %d pos %d", r, h, p)
				}
				e := float64(m) - float64(f)
				sse += e * e
				sy2 += float64(f) * float64(f)
				if math.IsNaN(float64(m)) || math.IsInf(float64(m), 0) {
					bad++
				}
			}
		}
	}
	// The mask, which the loop above cannot see: it compares only positions
	// inside each row's causal count. Everything past a row's bound must be
	// -inf, or the softmax normalises over the future.
	for r := 0; r < rows; r++ {
		for h := 0; h < nHeads; h++ {
			for p := int(pn[1+r]); p < cap; p++ {
				if v := gotM[(r*nHeads+h)*sstride+p]; !math.IsInf(float64(v), -1) {
					t.Fatalf("row %d head %d pos %d is %v past a causal count of %d, want -inf",
						r, h, p, v, pn[1+r])
				}
			}
		}
	}
	if nz < rows*nHeads/2 {
		t.Fatalf("only %d reference scores are non-zero; the oracle is degenerate", nz)
	}
	if bad > 0 {
		t.Fatalf("%d scores are not finite", bad)
	}
	if nmse := sse / sy2; nmse > 1e-5 || math.IsNaN(nmse) {
		t.Fatalf("NMSE %.3e against the FMA kernel", nmse)
	}
}
