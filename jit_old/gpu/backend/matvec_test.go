package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestMatVecMatchesReference holds the GPU decode matvec to jitllm's CPU bar:
// NMSE < 1e-10 against a float64 evaluation of the same quantities the kernel
// computes -- the same int8 activations, the same per-block scales. The only
// difference permitted is float32-vs-float64 summation order.
//
// Comparing against unquantized activations instead would need a tolerance near
// 1e-2, which hides real bugs. It runs on every backend present.
func TestMatVecMatchesReference(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			t.Logf("device: %s", d.Name())
			for _, q := range []mainkernels.Quant{mainkernels.Q4_0, mainkernels.Q8_0, mainkernels.Q4_K, mainkernels.MXFP4,
				mainkernels.Q5_0, mainkernels.Q5_1, mainkernels.TQ2, mainkernels.TQ2F, mainkernels.Float32, mainkernels.Float16, mainkernels.BFloat16} {
				// Splits are covered explicitly: the end-to-end token check
				// exercises them too, but a wrong split produces a plausible
				// number rather than a crash, and this isolates it.
				for _, shape := range []struct{ rows, k, split int }{
					{1, 32, 1}, {128, 256, 1}, {300, 2048, 1}, {2048, 2048, 1}, {37, 512, 1},
					{128, 256, 2}, {300, 2048, 4}, {2048, 2048, 16}, {37, 512, 8}, {256, 2048, 32},
					// llama-2-7B's real shapes, which once failed where every
					// smaller case passed.
					{4096, 4096, 16}, {11008, 4096, 8}, {4096, 11008, 8}, {1024, 4096, 64},
				} {
					// A shape a format cannot encode (k not a multiple of its
					// super-block) is filtered here, where the cross product is
					// formed, rather than registered as a SKIP.
					if shape.k%q.Elems() != 0 {
						continue
					}
					name := q.String() + "/" + itoa(shape.rows) + "x" + itoa(shape.k) +
						"/split" + itoa(shape.split)
					t.Run(name, func(t *testing.T) {
						for _, withBias := range []bool{false, true} {
							matvecCase(t, d, q, shape.rows, shape.k, shape.split, withBias)
						}
						// The batched prefill shape at several Tok, the decode
						// row tile at several Rowt, and the in-group split
						// reduction at every split the group divides: each has
						// per-column, per-tile or per-slot offsets that are
						// invisible at a width of 1 (or Split == 2).
						if mainkernels.GroupSplitOK(shape.rows, shape.split) {
							t.Run("groupsplit", func(t *testing.T) {
								for _, withBias := range []bool{false, true} {
									matvecCaseGroup(t, d, q, shape.rows, shape.k, shape.split, withBias)
								}
							})
						}
						// The tile at this shape's own split: tile rows are
						// adjacent and compose with the k-split, whose base
						// row, partial-sum row and bias gating are invisible
						// at Split == 1 and at Rowt == 1.
						for _, rt := range []int{2, 4} {
							if shape.rows%rt != 0 {
								continue
							}
							t.Run("rowt"+itoa(rt), func(t *testing.T) {
								for _, withBias := range []bool{false, true} {
									matvecCaseRowt(t, d, q, shape.rows, shape.k, shape.split, withBias, rt)
								}
							})
						}
						if shape.split == 1 {
							// 8 of 8 is a ragged decode step's shape at 8 rows:
							// one token group, every column in one thread.
							for _, tk := range []int{1, 2, 4, 8} {
								ntokCase(t, d, q, shape.rows, shape.k, 8, tk)
							}
							// Two sequences decoding together, unpadded.
							ntokCase(t, d, q, shape.rows, shape.k, 2, 2)
							ntokCase(t, d, q, shape.rows, shape.k, 4, 4)
							// Tok >= 8 is where weight centering turns on
							// (s.Tok >= 8), which Metal and Vulkan run for
							// every prefill matvec at BatchTok 16.
							for _, tk := range []int{8, 16} {
								ntokCase(t, d, q, shape.rows, shape.k, 16, tk)
							}
						}
					})
				}
			}
		})
	}
}

func matvecCase(t *testing.T, d backend.Device, q mainkernels.Quant, nrows, k, split int, withBias bool) {
	matvecCaseRowt(t, d, q, nrows, k, split, withBias, 1)
}

// matvecCaseGroup is matvecCaseRowt with the in-group split reduction: the
// Split segments of a row share a threadgroup and meet in threadgroup memory,
// so there is no partial buffer and no Reduce launch.
func matvecCaseGroup(t *testing.T, d backend.Device, q mainkernels.Quant, nrows, k, split int, withBias bool) {
	matvecCaseFull(t, d, q, nrows, k, split, withBias, 1, true)
}

// matvecCaseRowt is matvecCase with a decode row tile: rowt adjacent rows per
// thread, so a lane's payload words are contiguous.
func matvecCaseRowt(t *testing.T, d backend.Device, q mainkernels.Quant, nrows, k, split int, withBias bool, rowt int) {
	matvecCaseFull(t, d, q, nrows, k, split, withBias, rowt, false)
}

func matvecCaseFull(t *testing.T, d backend.Device, q mainkernels.Quant, nrows, k, split int, withBias bool, rowt int, groupSplit bool) {
	if k%q.Elems() != 0 {
		t.Skipf("k=%d is not a multiple of %d", k, q.Elems())
	}
	rng := rand.New(rand.NewSource(int64(nrows*7919 + k + int(q))))
	nb := k / 32
	bb := q.BlockBytes()
	nsuper := k / q.Elems()

	// GGUF-format weights, and the exact float values they decode to.
	raw := make([]byte, nrows*nsuper*bb)
	exact := make([][]float64, nrows)
	for r := range exact {
		exact[r] = make([]float64, k)
	}
	switch {
	case mainkernels.IsFloat(q):
		floatWeights(q, raw, exact, nrows, k, rng)
	case q == mainkernels.Q4_K:
		buildQ4K(raw, exact, nrows, nsuper, rng)
	case q == mainkernels.MXFP4:
		// An E8M0 exponent and sixteen bytes of e2m1 codes; the value is the
		// doubled e2m1 magnitude times 2^(e-128). Every code is drawn, so all
		// sixteen table entries are read in every row.
		for r := 0; r < nrows; r++ {
			for b := 0; b < nb; b++ {
				blk := raw[(r*nb+b)*bb:]
				blk[0] = byte(120 + rng.Intn(10))
				sc := math.Ldexp(1, int(blk[0])-128)
				for j := 0; j < 16; j++ {
					lo, hi := rng.Intn(16), rng.Intn(16)
					blk[1+j] = byte(lo | hi<<4)
					exact[r][b*32+j] = float64(quant.MXFP4Values[lo]) * sc
					exact[r][b*32+j+16] = float64(quant.MXFP4Values[hi]) * sc
				}
			}
		}
	case q == mainkernels.TQ2 || q == mainkernels.TQ2F:
		// Random 2-bit codes (3 included, which the format reads as 2) under
		// a sane scale, an f16 or an f32 one, decoded by the reference.
		rng.Read(raw)
		ref := make([]float64, 256)
		for i := 0; i < nrows*nsuper; i++ {
			blk := raw[i*bb : (i+1)*bb]
			if q == mainkernels.TQ2 {
				binary.LittleEndian.PutUint16(blk[64:], uint16(0x2000|rng.Intn(0x0C00)))
			} else {
				binary.LittleEndian.PutUint32(blk[64:], math.Float32bits(0.01+rng.Float32()))
			}
			if err := quant.Dequant(ggufOf[q], blk, ref); err != nil {
				t.Fatal(err)
			}
			copy(exact[i/nsuper][(i%nsuper)*256:], ref)
		}
	case q == mainkernels.Q5_0 || q == mainkernels.Q5_1:
		// Random planes under a sane scale (and, for Q5_1, a minimum drawn
		// the way its quantizer writes one: -d times up to 31, or a little
		// above zero), decoded by the reference dequantizer itself.
		rng.Read(raw)
		ref := make([]float64, 32)
		for i := 0; i < nrows*nb; i++ {
			blk := raw[i*bb : (i+1)*bb]
			h := uint16(0x2000 | rng.Intn(0x0C00))
			binary.LittleEndian.PutUint16(blk, h)
			if q == mainkernels.Q5_1 {
				m := float32(mainkernels.F16(h)) * float32(rng.Intn(36)-31)
				binary.LittleEndian.PutUint16(blk[2:], quant.EncodeHalf(m))
			}
			if err := quant.Dequant(ggufOf[q], blk, ref); err != nil {
				t.Fatal(err)
			}
			copy(exact[i/nb][(i%nb)*32:], ref)
		}
	default:
		for r := 0; r < nrows; r++ {
			for b := 0; b < nb; b++ {
				blk := raw[(r*nb+b)*bb:]
				// A sane f16 scale: exponent near 1.0 so products stay in range.
				h := uint16(0x2000 | rng.Intn(0x0C00))
				binary.LittleEndian.PutUint16(blk, h)
				dw := float64(mainkernels.F16(h))
				if q == mainkernels.Q4_0 {
					for j := 0; j < 16; j++ {
						lo, hi := rng.Intn(16), rng.Intn(16)
						blk[2+j] = byte(lo | hi<<4)
						exact[r][b*32+j] = float64(lo-8) * dw
						exact[r][b*32+j+16] = float64(hi-8) * dw
					}
				} else {
					for j := 0; j < 32; j++ {
						v := int8(rng.Intn(256) - 128)
						blk[2+j] = byte(v)
						exact[r][b*32+j] = float64(v) * dw
					}
				}
			}
		}
	}

	// Heavy-tailed activations: uniform alone hides the interesting failures.
	x := make([]float32, k)
	for i := range x {
		switch i % 4 {
		case 0:
			x[i] = float32(rng.NormFloat64())
		case 1:
			x[i] = float32(rng.NormFloat64() / (0.1 + math.Abs(rng.NormFloat64())))
		case 2:
			x[i] = float32(rng.Float64()*2 - 1)
		default:
			x[i] = float32(1 - 2*float64(i%2))
		}
	}

	qs, dw, scw, err := mainkernels.PackWeights(q, raw, nrows, k)
	if err != nil {
		t.Fatal(err)
	}
	av, as, asum, err := mainkernels.PackActivations(x)
	if err != nil {
		t.Fatal(err)
	}

	// Every shape runs with and without a bias. At Split > 1 the bias is gated
	// to segment 0 with ir.Select so it lands once; that gating is invisible at
	// Split == 1.
	bias := make([]float32, nrows)
	if withBias {
		for i := range bias {
			bias[i] = float32(rng.NormFloat64() * 4)
		}
	}
	kk, err := mainkernels.MatVec(mainkernels.MatVecShape{T: q, K: k, Rows: nrows, Split: split, Bias: withBias, Rowt: rowt, GroupSplit: groupSplit})
	if err != nil {
		t.Skipf("shape rejected: %v", err)
	}
	kern, err := d.Compile(kk)
	if err != nil {
		t.Fatal(err)
	}
	defer kern.Close()

	bufs := make([]backend.Buf, 0, 7)
	up := func(n int, p []byte) backend.Buf {
		b, err := d.Alloc(n)
		if err != nil {
			t.Fatal(err)
		}
		if err := b.Write(p); err != nil {
			t.Fatal(err)
		}
		bufs = append(bufs, b)
		return b
	}
	defer func() {
		for _, b := range bufs {
			b.Free()
		}
	}()

	if len(scw) == 0 {
		scw = []uint32{0}
	}
	bQS := up(len(qs)*4, u32bytes(qs))
	// A float format has no scale plane; its slot carries the float
	// activation (mainkernels.MatVec).
	var bD backend.Buf
	if mainkernels.IsFloat(q) {
		bD = up(len(x)*4, f32bytes(x))
	} else {
		bD = up(len(dw)*4, u32bytes(dw))
	}
	bSC := up(len(scw)*4, u32bytes(scw))
	bA := up(len(av)*4, u32bytes(av))
	// scales then per-16 sums, one f32 array matching the kernel's pAX param
	ax := append(append([]float32{}, as...), asum...)
	bAX := up(len(ax)*4, f32bytes(ax))
	// The in-group arm writes the final row with no partial buffer, so pOut is
	// sized at nrows there: a kernel that still wrote partials would fault.
	outN := nrows * split
	if groupSplit || split == 1 {
		outN = nrows
	}
	// Poisoned, not zeroed (RULE 13): a row the kernel never writes would
	// otherwise read as a plausible 0.0, or a quietly short sum after Reduce.
	// The check below refuses a non-finite result before the bound.
	bOut := up(outN*4, f32bytes(nanFill(outN)))
	bFinal := bOut
	if split > 1 && !groupSplit {
		bFinal = up(nrows*4, f32bytes(nanFill(nrows)))
	}

	width := 128
	if groupSplit {
		width = mainkernels.GroupSplitWidth(split)
	}
	groups := (nrows + width - 1) / width
	launchArgs := []backend.Buf{bQS, bD, bSC, bA, bAX, bOut}
	if withBias {
		bB := up(len(bias)*4, f32bytes(bias))
		launchArgs = append(launchArgs, bB)
	}
	// The grid shrinks with Rowt. Getting it wrong is silent: extra threads are
	// clamped to the last work item and recompute the last tile.
	if err := kern.Launch(((nrows/rowt)*split+width-1)/width, width, launchArgs...); err != nil {
		t.Fatal(err)
	}
	if split > 1 && !groupSplit {
		rk, err := mainkernels.Reduce(nrows, split)
		if err != nil {
			t.Fatal(err)
		}
		rkern, err := d.Compile(rk)
		if err != nil {
			t.Fatal(err)
		}
		defer rkern.Close()
		if err := rkern.Launch(groups, width, bOut, bFinal); err != nil {
			t.Fatal(err)
		}
	}
	rawOut := make([]byte, nrows*4)
	if err := bFinal.Read(rawOut); err != nil {
		t.Fatal(err)
	}

	// Reference: the same arithmetic in float64, over the same quantized
	// activations the kernel was given.
	var sse, sy2 float64
	for r := 0; r < nrows; r++ {
		var want float64
		if withBias {
			want += float64(bias[r])
		}
		for b := 0; b < nb; b++ {
			da := float64(as[b])
			for i := 0; i < 32; i++ {
				if mainkernels.IsFloat(q) {
					want += exact[r][b*32+i] * float64(x[b*32+i])
					continue
				}
				qi := int8(av[b*8+i/4] >> uint(8*(i%4)))
				want += exact[r][b*32+i] * float64(qi) * da
			}
		}
		got := float64(math.Float32frombits(binary.LittleEndian.Uint32(rawOut[r*4:])))
		// The poison is checked before the bound, and by row, so the failure
		// names the unwritten row; NaN > bound is false, so a bound test alone
		// would pass.
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

// nanFill is the poison RULE 13 asks for: a buffer nothing has written must be
// distinguishable from one written with zeros, and a fresh device allocation is
// usually already zero.
func nanFill(n int) []float32 {
	v := make([]float32, n)
	nan := float32(math.NaN())
	for i := range v {
		v[i] = nan
	}
	return v
}

func u32bytes(v []uint32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}

func f32bytes(v []float32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// buildQ4K writes synthetic Q4_K super-blocks and records the exact float
// weights they decode to, using gguf/dequant.go's formula rather than the
// packer's -- so the test cannot agree with itself.
func buildQ4K(raw []byte, exact [][]float64, nrows, nsuper int, rng *rand.Rand) {
	for r := 0; r < nrows; r++ {
		for sb := 0; sb < nsuper; sb++ {
			blk := raw[(r*nsuper+sb)*144:]
			hd := uint16(0x2000 | rng.Intn(0x0400))
			hm := uint16(0x1C00 | rng.Intn(0x0400))
			binary.LittleEndian.PutUint16(blk[0:], hd)
			binary.LittleEndian.PutUint16(blk[2:], hm)
			d := float64(mainkernels.F16(hd))
			dmin := float64(mainkernels.F16(hm))
			for i := 0; i < 12; i++ {
				blk[4+i] = byte(rng.Intn(256))
			}
			for i := 0; i < 128; i++ {
				blk[16+i] = byte(rng.Intn(256))
			}
			// Decode exactly as gguf/dequant.go does.
			sct, qs := blk[4:16], blk[16:144]
			o, is := 0, 0
			for j := 0; j < 256; j += 64 {
				qq := qs[j/2:]
				sc1, m1 := refScaleMinK4(is+0, sct)
				sc2, m2 := refScaleMinK4(is+1, sct)
				d1, min1 := d*float64(sc1), dmin*float64(m1)
				d2, min2 := d*float64(sc2), dmin*float64(m2)
				for l := 0; l < 32; l++ {
					exact[r][sb*256+o] = d1*float64(qq[l]&0xF) - min1
					o++
				}
				for l := 0; l < 32; l++ {
					exact[r][sb*256+o] = d2*float64(qq[l]>>4) - min2
					o++
				}
				is += 2
			}
		}
	}
}

func refScaleMinK4(j int, q []byte) (sc, m uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j] >> 6) << 4)
}

// TestFewSequenceDecodeKQuants runs the decode step of a few sequences
// (every token in one thread, at decode's own split and form) on a Q4_K_M
// model's own formats, shapes and splits: Llama-3.1-8B's ffn_gate (14336 x
// 4096) and ffn_down (4096 x 14336) reduced in the group at the splits its
// tuner chooses, and a Q6_K head-like projection unsplit; two and eight
// tokens, alone and at the head of a scratch padded to eight.
func TestFewSequenceDecodeKQuants(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			for _, c := range []struct {
				q       mainkernels.Quant
				rows, k int
				splits  []int
			}{
				{mainkernels.Q4_K, 14336, 4096, []int{8, 16}},
				{mainkernels.Q4_K, 4096, 14336, []int{14, 28}},
				{mainkernels.Q6_K, 4096, 14336, []int{14, 28}},
				{mainkernels.Q6_K, 2048, 4096, []int{1}},
			} {
				for _, split := range c.splits {
					for _, n := range []int{2, 8} {
						t.Run(fmt.Sprintf("%v/%dx%d", c.q, c.rows, c.k), func(t *testing.T) {
							ntokCaseRowt(t, d, c.q, c.rows, c.k, n, n, 1, split, split > 1, 0)
							ntokCaseRowt(t, d, c.q, c.rows, c.k, n, n, 1, split, split > 1, 8)
						})
					}
					// The decode matvec on the first of eight rows quantized
					// together: a ragged step whose one wanted row takes the
					// head alone (tier's one-row head), at decode's split,
					// reduced after and in the group.
					t.Run(fmt.Sprintf("%v/%dx%d/one", c.q, c.rows, c.k), func(t *testing.T) {
						ntokCaseRowt(t, d, c.q, c.rows, c.k, 1, 1, 1, split, false, 8)
						if split > 1 {
							ntokCaseRowt(t, d, c.q, c.rows, c.k, 1, 1, 1, split, true, 8)
						}
					})
				}
			}
		})
	}
}

// ntokCase runs the batched prefill shape against the same float64 reference.
//
// The point is Tok: one weight word consumed by Tok dot products. Every
// per-token offset (activation words, scale, sum halves, output row) is
// invisible at Tok == 1, so Tok is swept. Q4_0 only: Tok's offsets are
// format-independent, and the other cases cover each format's unpack.
//
// It also sweeps Rowt, because batchMV ships a row tile (tier.defBatchRowt) combined
// with NTok and Tok.
func ntokCase(t *testing.T, d backend.Device, q mainkernels.Quant, nrows, k, ntok, tok int) {
	for _, rowt := range []int{1, 2, 4} {
		if nrows%rowt != 0 {
			continue
		}
		// Split 4 is a decode batch's shape: partials per segment, summed
		// after (the tier's Reduce; here, the host).
		for _, split := range []int{1, 4} {
			ntokCaseRowt(t, d, q, nrows, k, ntok, tok, rowt, split, false, 0)
		}
	}
	// A decode step of a few sequences: every token in one thread and the
	// split reduced inside the group (tier.groupMV), so the kernel writes the
	// final rows and the host sums nothing.
	if tok == ntok && !mainkernels.IsFloat(q) {
		// 14 and 28 are what the tuner gives a Q4_K_M 8B's ffn matvecs.
		for _, split := range []int{4, 14, 16, 28} {
			if mainkernels.GroupSplitOK(nrows, split) && k/32%split == 0 {
				ntokCaseRowt(t, d, q, nrows, k, ntok, tok, 1, split, true, 0)
				// The step's live rows at the head of a scratch padded to
				// 8: the sums sit after all 8 rows' scales.
				ntokCaseRowt(t, d, q, nrows, k, ntok, tok, 1, split, true, 8)
			}
		}
	}
}

func ntokCaseRowt(t *testing.T, d backend.Device, q mainkernels.Quant, nrows, k, ntok, tok, rowt, split int, grp bool, actRows int) {
	// Q4_0 for the offsets, which are format-independent; Q4_K and Q6_K
	// because the in-group form is what a Q4_K_M model's decode step runs, and
	// a k-quant's sub-block sums and Q6_K's second plane are not Q4_0's.
	kq := q == mainkernels.Q4_K || q == mainkernels.Q6_K
	if q != mainkernels.Q4_0 && !mainkernels.IsFloat(q) && !(kq && tok == ntok) {
		return
	}
	g := ""
	if grp {
		g = "g"
	}
	if actRows > ntok {
		g += fmt.Sprintf("of%d", actRows)
	}
	vecs := max(ntok, actRows) // activation vectors quantized together
	t.Run(fmt.Sprintf("ntok%d/tok%d/rowt%d/split%d%s", ntok, tok, rowt, split, g), func(t *testing.T) {
		kk, err := mainkernels.MatVec(mainkernels.MatVecShape{T: q, K: k, Rows: nrows, NTok: ntok, Tok: tok, Rowt: rowt,
			Split: split, GroupSplit: grp, ActRows: actRows})
		if err != nil {
			t.Skipf("shape rejected: %v", err)
		}
		kern, err := d.Compile(kk)
		if err != nil {
			t.Fatal(err)
		}
		defer kern.Close()

		rng := rand.New(rand.NewSource(int64(nrows*31 + k*7 + ntok*3 + tok)))
		nb, bb := k/32, q.BlockBytes()
		raw := make([]byte, nrows*(k/q.Elems())*bb)
		exact := make([][]float64, nrows)
		for r := range exact {
			exact[r] = make([]float64, k)
		}
		switch {
		case mainkernels.IsFloat(q):
			floatWeights(q, raw, exact, nrows, k, rng)
		case q == mainkernels.Q4_K:
			buildQ4K(raw, exact, nrows, k/q.Elems(), rng)
		case q == mainkernels.Q6_K:
			// Random planes and int8 scales under a sane f16 d, which is the
			// block's last two bytes; the reference dequantizer is the oracle.
			rng.Read(raw)
			ref := make([]float64, q.Elems())
			for i := 0; i < nrows*k/q.Elems(); i++ {
				blk := raw[i*bb : (i+1)*bb]
				binary.LittleEndian.PutUint16(blk[bb-2:], uint16(0x1800|rng.Intn(0x0400)))
				if err := quant.Dequant(ggufOf[q], blk, ref); err != nil {
					t.Fatal(err)
				}
				copy(exact[i/(k/q.Elems())][(i%(k/q.Elems()))*q.Elems():], ref)
			}
		}
		for r := 0; r < nrows && q == mainkernels.Q4_0; r++ {
			for b := 0; b < nb; b++ {
				blk := raw[(r*nb+b)*bb:]
				h := uint16(0x2000 | rng.Intn(0x0C00))
				binary.LittleEndian.PutUint16(blk, h)
				dw := float64(mainkernels.F16(h))
				for j := 0; j < 16; j++ {
					lo, hi := rng.Intn(16), rng.Intn(16)
					blk[2+j] = byte(lo | hi<<4)
					exact[r][b*32+j] = float64(lo-8) * dw
					exact[r][b*32+j+16] = float64(hi-8) * dw
				}
			}
		}
		qs, dw, scw, err := mainkernels.PackWeights(q, raw, nrows, k)
		if err != nil {
			t.Fatal(err)
		}

		// All ntok columns in ONE vector, which is how the device sees them:
		// PackActivations over ntok*k writes every scale then every sum, the
		// layout the slot addressing already expects.
		x := make([]float32, vecs*k)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		av, as, asum, err := mainkernels.PackActivations(x)
		if err != nil {
			t.Fatal(err)
		}

		var bufs []backend.Buf
		up := func(n int, p []byte) backend.Buf {
			b2, err := d.Alloc(n)
			if err != nil {
				t.Fatal(err)
			}
			if p != nil {
				if err := b2.Write(p); err != nil {
					t.Fatal(err)
				}
			}
			bufs = append(bufs, b2)
			return b2
		}
		defer func() {
			for _, b2 := range bufs {
				b2.Free()
			}
		}()
		bQS := up(len(qs)*4, u32bytes(qs))
		var bD backend.Buf
		if mainkernels.IsFloat(q) { // the float activation, in the scale plane's slot
			bD = up(len(x)*4, f32bytes(x))
		} else {
			bD = up(len(dw)*4, u32bytes(dw))
		}
		bSC := up(4, nil)
		if len(scw) > 0 {
			bSC = up(len(scw)*4, u32bytes(scw))
		}
		bA := up(len(av)*4, u32bytes(av))
		bAX := up((len(as)+len(asum))*4, f32bytes(append(append([]float32{}, as...), asum...)))
		// RULE 13: poisoned, not zeroed -- a column or a tile row the kernel
		// never writes is otherwise a plausible 0.0.
		parts := split // segments the host sums: none when the group did
		if grp {
			parts = 1
		}
		bOut := up(parts*nrows*ntok*4, f32bytes(nanFill(parts*nrows*ntok)))

		// The kernel's own group width, not a guessed one: a mismatch is invisible
		// on PTX, skips rows on Metal (which bakes it) and over-launches on Vulkan.
		threads := (nrows / rowt) * (ntok / tok) * split
		width := kk.Group[0]
		if err := kern.Launch((threads+width-1)/width, width, bQS, bD, bSC, bA, bAX, bOut); err != nil {
			t.Fatal(err)
		}
		pb := make([]byte, parts*nrows*ntok*4)
		if err := bOut.Read(pb); err != nil {
			t.Fatal(err)
		}
		rawOut := make([]byte, nrows*ntok*4)
		for i := 0; i < nrows*ntok; i++ {
			var v float32
			for sg := 0; sg < parts; sg++ {
				v += math.Float32frombits(binary.LittleEndian.Uint32(pb[4*(sg*nrows*ntok+i):]))
			}
			binary.LittleEndian.PutUint32(rawOut[4*i:], math.Float32bits(v))
		}

		var sse, sy2 float64
		for col := 0; col < ntok; col++ {
			for r := 0; r < nrows; r++ {
				var want float64
				for b := 0; b < nb; b++ {
					da := float64(as[col*nb+b])
					for i := 0; i < 32; i++ {
						ki := b*32 + i
						if mainkernels.IsFloat(q) {
							want += exact[r][ki] * float64(x[col*k+ki])
							continue
						}
						qi := int8(av[(col*k+ki)/4] >> uint(8*(ki%4)))
						want += exact[r][ki] * float64(qi) * da
					}
				}
				o := (col*nrows + r) * 4
				gotv := float64(math.Float32frombits(binary.LittleEndian.Uint32(rawOut[o:])))
				if math.IsNaN(gotv) || math.IsInf(gotv, 0) {
					t.Fatalf("ntok=%d tok=%d rowt=%d: column %d row %d came back %v -- "+
						"the kernel never wrote it (output was poisoned, not zeroed)",
						ntok, tok, rowt, col, r, gotv)
				}
				dv := gotv - want
				sse += dv * dv
				sy2 += want * want
			}
		}
		if nmse := sse / sy2; nmse > 1e-10 || math.IsNaN(nmse) {
			t.Errorf("ntok=%d tok=%d rowt=%d: NMSE %.3e", ntok, tok, rowt, nmse)
		}
	})
}

// floatWeights fills raw with row-major float weights of format q and exact
// with the values they hold after rounding to it.
func floatWeights(q mainkernels.Quant, raw []byte, exact [][]float64, nrows, k int, rng *rand.Rand) {
	for r := 0; r < nrows; r++ {
		for i := 0; i < k; i++ {
			v, o := float32(rng.NormFloat64()), r*k+i
			switch q {
			case mainkernels.Float32:
				binary.LittleEndian.PutUint32(raw[4*o:], math.Float32bits(v))
			case mainkernels.Float16:
				h := quant.EncodeHalf(v)
				binary.LittleEndian.PutUint16(raw[2*o:], h)
				v = mainkernels.F16(h)
			default:
				h := uint16(math.Float32bits(v) >> 16)
				binary.LittleEndian.PutUint16(raw[2*o:], h)
				v = math.Float32frombits(uint32(h) << 16)
			}
			exact[r][i] = float64(v)
		}
	}
}
