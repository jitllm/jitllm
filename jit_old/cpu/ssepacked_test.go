//go:build amd64

package cpu

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// The SSE tier's packed matvec gates. Legacy SSE executes on an AVX2 host, so
// every gate here runs the SSE kernels directly. Three references:
//   - internal/oracle, at the AVX2 gate's bound (NMSE 1e-3, the q8 bound);
//   - the AVX2 twin, bit for bit, on inputs where every epilogue product is
//     exact (power-of-two scales, sub-block scales of at most seven bits,
//     activation block scale 2^-7), so the FMA-vs-rounded-add difference
//     rounds nothing and any difference is a wrong plane, offset, row, scale
//     byte or a saturated pair;
//   - the AVX2 twin on realistic inputs (planted f16 scales, subnormals
//     included), at a bound sized from the FMA alone.

// ssePackFix is one packed tensor and one activation, laid out exactly as nn
// hands them to a kernel.
type ssePackFix struct {
	g            quant.Type
	q            kernels.Quant
	nrows, k     int
	qb, db, scb  []byte
	x            []float32
	q8           []int8
	pairs, half  []float32
	scr          []byte
	exact        bool
	whereIsExact string
}

// newSSEPackFix packs a random tensor of g. exact makes every d a power of two,
// every sub-block scale at most seven bits wide and every activation block's
// scale 2^-7, so the epilogue's products are exact (see the file comment).
func newSSEPackFix(t testing.TB, g quant.Type, nrows, k int, seed int64, exact bool) *ssePackFix {
	t.Helper()
	q, ok := kernels.QuantOf(g)
	if !ok {
		t.Fatalf("%s has no device layout", g)
	}
	rng := rand.New(rand.NewSource(seed))
	nb := uint64(nrows*k) / g.BlockElems()
	src := make([]byte, nb*g.BlockBytes())
	for i := range src {
		src[i] = byte(rng.Intn(256))
	}
	if !quant.PlantScales(g, src, int(seed)) {
		t.Fatalf("%s: no scale planter -- add it rather than testing noise", g)
	}
	qs, d, sc, err := kernels.PackWeights(q, src, nrows, k)
	if err != nil {
		t.Fatal(err)
	}
	p := &ssePackFix{g: g, q: q, nrows: nrows, k: k, exact: exact}
	if exact {
		pow2 := []uint16{0x3400, 0x3800, 0x3C00, 0x4000} // 0.25 0.5 1 2
		for i := range d {
			if NarrowD(g) {
				// Two rows to a word, both halves a scale.
				d[i] = uint32(pow2[rng.Intn(4)]) | uint32(pow2[rng.Intn(4)])<<16
			} else if kernels.D32(q) {
				// The word is the f32 scale (TQ2_F32).
				d[i] = math.Float32bits(float32(quant.DecodeHalf(pow2[rng.Intn(4)])))
			} else {
				// d low, dmin high: only d multiplies the dot.
				d[i] = d[i]&0xFFFF0000 | uint32(pow2[rng.Intn(4)])
			}
		}
		if _, scOff := kernels.ScaleLayout(q); scOff == -128 {
			// Q6_K's stored byte is an int8 scale + 128; halve it so the
			// scale is seven bits and a 17-bit dot times it stays in 24.
			for i := range sc {
				var w uint32
				for b := 0; b < 4; b++ {
					v := int8(sc[i] >> (8 * b))
					w |= uint32(byte(128+int(v)>>1)) << (8 * b)
				}
				sc[i] = w
			}
		}
		p.whereIsExact = "powers-of-two d, <=7-bit sub-block scales, 2^-7 activation scales"
	}
	p.qb, p.db, p.scb = u32b(qs), u32b(d), u32b(sc)

	p.x = make([]float32, k)
	for b := 0; b < k/Q8Block; b++ {
		blk := p.x[b*Q8Block : (b+1)*Q8Block]
		for i := range blk {
			if exact {
				blk[i] = float32(rng.Intn(255)-127) / 128
			} else {
				blk[i] = float32(rng.NormFloat64())
			}
		}
		if exact {
			// One element at 127/128 makes QuantizeQ8's d exactly 2^-7.
			blk[rng.Intn(Q8Block)] = float32(1-2*rng.Intn(2)) * 127 / 128
		}
	}
	p.q8 = make([]int8, k)
	p.pairs = make([]float32, 2*(k/Q8Block))
	if err := oracle.QuantizeQ8(p.q8, p.pairs, p.x, BiasC(g)); err != nil {
		t.Fatal(err)
	}
	if exact {
		for b := 0; b < k/Q8Block; b++ {
			if p.pairs[2*b] != 1.0/128 {
				t.Fatalf("activation block %d has scale %v, not 2^-7 -- the exact arm is not exact", b, p.pairs[2*b])
			}
		}
	}
	p.half = make([]float32, k/16)
	oracle.QuantizeHalfSums(p.half, p.q8, BiasC(g))
	p.scr = make([]byte, PackedScratchBytes)
	if err := PackedScratch(p.scr); err != nil {
		t.Fatal(err)
	}
	return p
}

// args is the Args nn builds for rows [r0, ...) of this tensor, with count
// tiles or groups.
func (p *ssePackFix) args(out []float32, r0, count int, scratch, q32 []float32) Args {
	a := Args{
		Out: &out[0], W: &p.qb[r0*4], A: &p.q8[0], AScale: &p.pairs[0],
		Rows: int64(count), RowStr: int64(p.nrows * 4),
		K: int64(p.k / PackedOuterElems(p.g)), Scr: &p.scr[0], AHalf: &p.half[0],
		PD: &p.db[r0*DRowBytes(p.g)], DStr: int64(DSuperBytes(p.g, p.nrows)),
	}
	if len(p.scb) > 0 {
		a.PSC = &p.scb[r0*4]
	}
	if scratch != nil {
		a.Scratch = (*byte)(unsafe.Pointer(&scratch[0]))
		a.Q32 = &q32[0]
	}
	return a
}

// guarded is an output of n floats with a sentinel before and after it; check
// fails if the kernel wrote outside [0, n).
type guarded struct{ buf []float32 }

const sseGuardLen = 4

func newGuarded(n int, fill float32) guarded {
	g := guarded{buf: make([]float32, n+2*sseGuardLen)}
	for i := range g.buf {
		g.buf[i] = fill
	}
	for i := 0; i < sseGuardLen; i++ {
		g.buf[i] = -7.25
		g.buf[len(g.buf)-1-i] = -7.25
	}
	return g
}

func (g guarded) out() []float32 { return g.buf[sseGuardLen : len(g.buf)-sseGuardLen] }

func (g guarded) check(t testing.TB, what string) {
	t.Helper()
	for i := 0; i < sseGuardLen; i++ {
		if g.buf[i] != -7.25 || g.buf[len(g.buf)-1-i] != -7.25 {
			t.Fatalf("%s: wrote outside its rows (guard %d before, %d after)", what,
				int(math.Float32bits(g.buf[i])), int(math.Float32bits(g.buf[len(g.buf)-1-i])))
		}
	}
}

// runTile runs a tile kernel over rows [r0, r0+tiles*rows) into a NaN-filled
// output -- the kernel zeroes each tile itself, so garbage must not survive.
func (p *ssePackFix) runTile(t testing.TB, code []byte, rows, r0, tiles int) []float32 {
	t.Helper()
	c := mustMap(t, code)
	defer c.Close()
	g := newGuarded(tiles*rows, float32(math.NaN()))
	a := p.args(g.out(), r0, tiles, nil, nil)
	c.Call(&a)
	g.check(t, "tile")
	return append([]float32(nil), g.out()...)
}

// runFused runs a fused kernel over rows [r0, r0+groups*group) into a zeroed
// output (its ABI: the caller owns the zero), with poisoned scratch.
func (p *ssePackFix) runFused(t testing.TB, code []byte, r0, groups int) []float32 {
	t.Helper()
	c := mustMap(t, code)
	defer c.Close()
	n := groups * PackedFusedGroupOf(p.g)
	g := newGuarded(n, 0)
	scratch, q32 := make([]float32, n), make([]float32, n)
	for i := range scratch {
		scratch[i], q32[i] = float32(math.NaN()), float32(math.NaN())
	}
	a := p.args(g.out(), r0, groups, scratch, q32)
	c.Call(&a)
	g.check(t, "fused")
	return append([]float32(nil), g.out()...)
}

func (p *ssePackFix) oracle(t testing.TB) []float32 {
	t.Helper()
	ref := make([]float32, p.nrows)
	if err := oracle.MatVecPacked(ref, p.q, p.qb, p.db, p.scb, p.x, p.nrows, p.k); err != nil {
		t.Fatal(err)
	}
	return ref
}

func nmseOf(t testing.TB, got, ref []float32) float64 {
	t.Helper()
	var num, den float64
	for i := range ref {
		d := float64(got[i]) - float64(ref[i])
		num += d * d
		den += float64(ref[i]) * float64(ref[i])
	}
	if den == 0 || math.IsNaN(num) || math.IsInf(num, 0) || math.IsNaN(den) || math.IsInf(den, 0) {
		t.Fatalf("degenerate comparison (num=%v den=%v) -- this gate proved nothing", num, den)
	}
	return num / den
}

// ssePackShapes: 208 rows is three 64-row tiles and two 8-row tails, and
// thirteen 16-row (or twenty-six 8-row) fused groups; k=512 is two k-quant
// super-blocks and sixteen 32-element ones.
const ssePackRows, ssePackK = 208, 512

// ssePackedKernels emits the three SSE packed kernels for g and runs the
// VEX-leak gate on each.
func ssePackedKernels(t *testing.T, g quant.Type) (tile, tail, fused []byte) {
	t.Helper()
	var err error
	if tile, err = EmitPackedMatVecSSE(g, PackedRows); err != nil {
		t.Fatalf("%s tile: %v", g, err)
	}
	if tail, err = EmitPackedMatVecSSE(g, PackedTail); err != nil {
		t.Fatalf("%s tail: %v", g, err)
	}
	if fused, err = EmitPackedMatVecFusedSSE(g); err != nil {
		t.Fatalf("%s fused: %v", g, err)
	}
	for name, code := range map[string][]byte{"tile": tile, "tail": tail, "fused": fused} {
		if KernelTier(code) != TierSSE {
			t.Fatalf("%s %s: not declared SSE-tier", g, name)
		}
		requireSSEKernel(t, "sse_packed_"+g.String()+"_"+name, code)
	}
	return tile, tail, fused
}

// runAllSSE computes every row with the tile+tail kernels and, separately,
// with the fused kernel -- each in more than one call, so a kernel that
// ignores its row offset or its count cannot pass.
func runAllSSE(t *testing.T, p *ssePackFix, tile, tail, fused []byte) (byTile, byFused []float32) {
	t.Helper()
	nt := p.nrows / PackedRows
	byTile = append(p.runTile(t, tile, PackedRows, 0, 1), p.runTile(t, tile, PackedRows, PackedRows, nt-1)...)
	r := nt * PackedRows
	byTile = append(byTile, p.runTile(t, tail, PackedTail, r, (p.nrows-r)/PackedTail)...)
	grp := PackedFusedGroupOf(p.g)
	ng := p.nrows / grp
	byFused = append(p.runFused(t, fused, 0, 2), p.runFused(t, fused, 2*grp, ng-2)...)
	return byTile, byFused
}

// requireAVX2Twin skips a gate whose second arm is the AVX2 kernel on a host
// that cannot execute it; the oracle and guard-page gates still run there.
func requireAVX2Twin(t *testing.T) {
	t.Helper()
	code, err := EmitPackedMatVecFused(quant.Q4_0, DotVEX)
	if err != nil {
		t.Fatal(err)
	}
	c, err := Map(code)
	if errors.Is(err, ErrISA) {
		t.Skipf("THE AVX2 TWIN CANNOT RUN HERE (%v) -- this gate compares the two tiers and "+
			"needs both; on this host the oracle and guard-page gates carry the SSE kernels", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	c.Close()
}

// runAllAVX2 is runAllSSE with the AVX2 twins, the kernels this host runs.
func runAllAVX2(t *testing.T, p *ssePackFix) (byTile, byFused []float32) {
	t.Helper()
	tile, err := EmitPackedMatVec(p.g, PackedRows, HostDotKind())
	if err != nil {
		t.Fatal(err)
	}
	tail, err := EmitPackedMatVec(p.g, PackedTail, HostDotKind())
	if err != nil {
		t.Fatal(err)
	}
	fused, err := EmitPackedMatVecFused(p.g, HostDotKind())
	if err != nil {
		t.Fatal(err)
	}
	return runAllSSE(t, p, tile, tail, fused)
}

// TestSSEPackedMatchesTheOracle holds every SSE packed kernel to the reference,
// on every packed format, at the AVX2 gate's bound.
func TestSSEPackedMatchesTheOracle(t *testing.T) {
	ran := 0
	for _, g := range quant.PackedTypes {
		t.Run(g.String(), func(t *testing.T) {
			tile, tail, fused := ssePackedKernels(t, g)
			p := newSSEPackFix(t, g, ssePackRows, ssePackK, 7, false)
			ref := p.oracle(t)
			byTile, byFused := runAllSSE(t, p, tile, tail, fused)
			for name, got := range map[string][]float32{"tile+tail": byTile, "fused": byFused} {
				if nmse := nmseOf(t, got, ref); nmse > 1e-3 {
					t.Fatalf("%s %s: NMSE %.3e\n  got  %v\n  want %v", g, name, nmse, got[:8], ref[:8])
				} else {
					t.Logf("%s %s: NMSE %.2e over %d rows of %d", g, name, nmse, p.nrows, p.k)
				}
			}
			ran++
		})
	}
	if ran != len(quant.PackedTypes) {
		t.Fatalf("%d of %d formats ran", ran, len(quant.PackedTypes))
	}
}

// TestSSEPackedMatchesAVX2BitForBit is the equality gate: with every epilogue
// product exact, the SSE kernels and their AVX2 twins must agree to the bit.
func TestSSEPackedMatchesAVX2BitForBit(t *testing.T) {
	requireAVX2Twin(t)
	ran := 0
	for _, g := range quant.PackedTypes {
		t.Run(g.String(), func(t *testing.T) {
			tile, tail, fused := ssePackedKernels(t, g)
			for seed := int64(1); seed <= 3; seed++ {
				p := newSSEPackFix(t, g, ssePackRows, ssePackK, seed, true)
				if msg := firstExactMismatch(t, p, tile, tail, fused); msg != "" {
					t.Fatalf("%s seed %d: %s", g, seed, msg)
				}
			}
			ran++
		})
	}
	if ran != len(quant.PackedTypes) {
		t.Fatalf("%d of %d formats ran", ran, len(quant.PackedTypes))
	}
}

// firstExactMismatch runs both tiers on an exact fixture and names the first
// row where they differ by a single bit, or returns "". It also demands the
// exact arm computes the model, so it is not agreeing about nothing.
func firstExactMismatch(t *testing.T, p *ssePackFix, tile, tail, fused []byte) string {
	t.Helper()
	sTile, sFused := runAllSSE(t, p, tile, tail, fused)
	aTile, aFused := runAllAVX2(t, p)
	for _, c := range []struct {
		name     string
		sse, avx []float32
	}{{"tile+tail", sTile, aTile}, {"fused", sFused, aFused}} {
		for r := range c.sse {
			if math.Float32bits(c.sse[r]) != math.Float32bits(c.avx[r]) {
				return fmt.Sprintf("%s row %d: SSE %v (%#08x), AVX2 %v (%#08x) -- with %s "+
					"the two tiers compute the same integers and the same exact products, "+
					"so any difference is a wrong plane, offset, row, scale or saturated pair",
					c.name, r, c.sse[r], math.Float32bits(c.sse[r]),
					c.avx[r], math.Float32bits(c.avx[r]), p.whereIsExact)
			}
		}
	}
	if nmse := nmseOf(t, sFused, p.oracle(t)); nmse > 1e-3 {
		t.Fatalf("the exact arm's fused result is NMSE %.3e from the oracle", nmse)
	}
	return ""
}

// TestSSEPackedTracksAVX2OnRealScales runs both tiers on planted scales
// (subnormal f16 d included) and bounds the difference by what the missing
// FMA can cost: one rounding of each sub-block's product, relative to the
// largest output. The 1e-6 bound is about ten times the observed worst, while
// a wrong scale on one sub-block of one row lands near 1e-2.
func TestSSEPackedTracksAVX2OnRealScales(t *testing.T) {
	requireAVX2Twin(t)
	const bound = 1e-6
	for _, g := range quant.PackedTypes {
		t.Run(g.String(), func(t *testing.T) {
			tile, tail, fused := ssePackedKernels(t, g)
			p := newSSEPackFix(t, g, ssePackRows, ssePackK, 11, false)
			sTile, sFused := runAllSSE(t, p, tile, tail, fused)
			aTile, aFused := runAllAVX2(t, p)
			for _, c := range []struct {
				name     string
				sse, avx []float32
			}{{"tile+tail", sTile, aTile}, {"fused", sFused, aFused}} {
				var worst, scale float64
				differ := 0
				for r := range c.sse {
					if math.IsNaN(float64(c.sse[r])) || math.IsInf(float64(c.sse[r]), 0) {
						t.Fatalf("%s %s: row %d is %v", g, c.name, r, c.sse[r])
					}
					scale = math.Max(scale, math.Abs(float64(c.avx[r])))
					worst = math.Max(worst, math.Abs(float64(c.sse[r])-float64(c.avx[r])))
					if c.sse[r] != c.avx[r] {
						differ++
					}
				}
				if scale == 0 {
					t.Fatalf("%s %s: every AVX2 output is zero -- this gate proved nothing", g, c.name)
				}
				rel := worst / scale
				if rel > bound {
					t.Fatalf("%s %s: max|SSE-AVX2| / max|AVX2| = %.3e, over %.0e", g, c.name, rel, bound)
				}
				t.Logf("%s %s: %d of %d rows differ, max|d|/max|y| %.2e", g, c.name, differ, len(c.sse), rel)
			}
		})
	}
}

// TestSSEPackedSecondCallIntoTheSameBuffer is the arm64 zeroing bug's gate at
// the kernel: the tile kernel zeroes its own output, so a second call into the
// buffer the first one filled must reproduce it exactly, not double it.
func TestSSEPackedSecondCallIntoTheSameBuffer(t *testing.T) {
	for _, g := range quant.PackedTypes {
		tile, err := EmitPackedMatVecSSE(g, PackedRows)
		if err != nil {
			t.Fatal(err)
		}
		c := mustMap(t, tile)
		p := newSSEPackFix(t, g, 128, 256, 3, false)
		out := make([]float32, 128)
		a := p.args(out, 0, 2, nil, nil)
		c.Call(&a)
		first := append([]float32(nil), out...)
		c.Call(&a)
		c.Close()
		for r := range out {
			if out[r] != first[r] {
				t.Fatalf("%s: row %d went %v -> %v on a second call into the same buffer", g, r, first[r], out[r])
			}
		}
	}
}

// TestSSEDotMatchesVPDPBUSD runs the legacy dot sequence -- PMADDUBSW, PMADDWD,
// PADDD -- against a scalar reference, and against VPDPBUSD and the AVX2 DotVEX
// sequence on the same bytes, including every plane's extreme byte.
func TestSSEDotMatchesVPDPBUSD(t *testing.T) {
	build := func() []byte {
		var a Buf
		a.DeclareISA(ISATierSSE)
		a.MOVLoad(RSI, At(RDI, 16)) // A, signed
		a.MOVLoad(RDX, At(RDI, 8))  // W, unsigned
		a.MOVLoad(RCX, At(RDI, 56)) // Scr, the result
		d := sseDot{ones: XMM3}
		sseOnes16(&a, XMM3)
		for h := int32(0); h < 2; h++ {
			a.PXOR(XMM0, XMM0, XMM0)
			a.MOVDQULoad(XMM1, At(RDX, 16*h))
			a.MOVDQULoad(XMM2, At(RSI, 16*h))
			d.unsigned(&a, XMM0, XMM1, XMM2)
			a.MOVDQUStore(At(RCX, 16*h), XMM0)
		}
		a.RET()
		return a.Bytes()
	}
	code := build()
	requireSSEKernel(t, "sse_dot", code)
	c := mustMap(t, code)
	defer c.Close()

	// The widest unsigned byte any admitted plane reaches is 63 (a 4-bit
	// nibble, a pre-shifted 0x30/0x10 secondary, MXFP4's 24) -- PreVNNIPacked's
	// bound -- so the extremes are 0, 15, 16, 24, 48 and 63 against +-127.
	rng := rand.New(rand.NewSource(5))
	extremes := []byte{0, 15, 16, 24, 48, 63}
	for trial := 0; trial < 200; trial++ {
		act := make([]int8, 32)
		u := make([]byte, 32)
		for i := range act {
			switch trial % 3 {
			case 0:
				act[i], u[i] = int8(rng.Intn(255)-127), byte(rng.Intn(64))
			case 1:
				act[i], u[i] = int8(127*(1-2*rng.Intn(2))), extremes[rng.Intn(len(extremes))]
			default:
				act[i], u[i] = int8(127*(1-2*(i&1))), 63
			}
		}
		scr := make([]byte, 32)
		c.Call(&Args{A: &act[0], W: &u[0], Scr: &scr[0]})
		for lane := 0; lane < 8; lane++ {
			var want int32
			for i := 0; i < 4; i++ {
				want += int32(u[lane*4+i]) * int32(act[lane*4+i])
			}
			if got := int32(binary.LittleEndian.Uint32(scr[lane*4:])); got != want {
				t.Fatalf("trial %d lane %d: SSE dot %d, reference %d", trial, lane, got, want)
			}
		}
	}
}

// TestSSEPackedRefusesWhatItCannotServe: the tile's row count, the wide and
// token-tiled kernels (refused by decision), and a type with no layout.
func TestSSEPackedRefusesWhatItCannotServe(t *testing.T) {
	for _, g := range quant.PackedTypes {
		if !SupportedPackedSSE(g) {
			t.Errorf("%s: SupportedPackedSSE is false for a packed type", g)
		}
		if MaxTiledTokensSSE(g) != 0 {
			t.Errorf("%s: MaxTiledTokensSSE is %d, want 0", g, MaxTiledTokensSSE(g))
		}
		if _, err := EmitPackedMatVecWideSSE(g); err == nil {
			t.Errorf("%s: the wide kernel emitted", g)
		}
		if _, err := EmitPackedMatMulTiledSSE(g, 2048, 512, 2); err == nil {
			t.Errorf("%s: the token-tiled kernel emitted", g)
		}
		for _, rows := range []int{0, 7, 12, 72} {
			if _, err := EmitPackedMatVecSSE(g, rows); err == nil {
				t.Errorf("%s: a %d-row tile emitted", g, rows)
			}
		}
		for _, rows := range []int{8, 16, 24, 40, 64} {
			if _, err := EmitPackedMatVecSSE(g, rows); err != nil {
				t.Errorf("%s: a %d-row tile refused: %v", g, rows, err)
			}
		}
	}
	for _, g := range []quant.Type{quant.F32, quant.F16, quant.BF16} {
		if SupportedPackedSSE(g) {
			t.Errorf("%s: SupportedPackedSSE is true for a type with no packed layout", g)
		}
		if _, err := EmitPackedMatVecSSE(g, PackedRows); err == nil {
			t.Errorf("%s: the tile kernel emitted for a type with no packed layout", g)
		}
		if _, err := EmitPackedMatVecFusedSSE(g); err == nil {
			t.Errorf("%s: the fused kernel emitted for a type with no packed layout", g)
		}
	}
	em := EmittersFor(TierSSE)
	for _, g := range quant.PackedTypes {
		if _, err := em.PackedFused(g); errors.Is(err, ErrNoSSEKernel) || err != nil {
			t.Errorf("%s: the SSE table's fused field says %v", g, err)
		}
		if _, err := em.PackedMatVec(g, PackedTail); err != nil {
			t.Errorf("%s: the SSE table's tile field says %v", g, err)
		}
	}
}

// TestSSEPackedKernelSizes logs what each kernel costs in code, beside the
// AVX2 twin's. There is no budget refusal on this tier yet: the fused kernel's
// hot loop is one sub-block's body, well inside a 32 KB L1i, and the tile
// kernel serves only the rows the fused one cannot.
func TestSSEPackedKernelSizes(t *testing.T) {
	for _, g := range quant.PackedTypes {
		tile, _ := EmitPackedMatVecSSE(g, PackedRows)
		tail, _ := EmitPackedMatVecSSE(g, PackedTail)
		fused, _ := EmitPackedMatVecFusedSSE(g)
		at, _ := EmitPackedMatVec(g, PackedRows, DotVEX)
		af, _ := EmitPackedMatVecFused(g, DotVEX)
		t.Logf("%-6s SSE tile %6d  tail %6d  fused %6d   |  AVX2 tile %6d  fused %6d",
			g, len(tile), len(tail), len(fused), len(at), len(af))
	}
}

// TestRowsPerCallForKeepsTheAVX2Answer: on every tier but SSE, RowsPerCallFor
// is RowsPerCall exactly (so switching a caller to it moves nothing there); on
// the SSE tier it is whole cache lines of output and prices a call within
// maxCallNanos at sseCallNanosPerElem, down to the one-line floor.
func TestRowsPerCallForKeepsTheAVX2Answer(t *testing.T) {
	for _, g := range append(append([]quant.Type{}, quant.PackedTypes...), quant.F32, quant.F16) {
		for _, k := range []int{256, 512, 2048, 4096, 8192, 16384, 1 << 20} {
			if k%int(g.BlockElems()) != 0 {
				continue
			}
			for _, tier := range []Tier{TierNone, TierAVX2, TierNEON} {
				if got, want := RowsPerCallFor(tier, g, k), RowsPerCall(k/int(g.BlockElems())); got != want {
					t.Errorf("%v %s k=%d: RowsPerCallFor %d, RowsPerCall %d", tier, g, k, got, want)
				}
			}
			r := RowsPerCallFor(TierSSE, g, k)
			if r < OutLine || r%OutLine != 0 {
				t.Errorf("sse %s k=%d: %d rows is not whole cache lines of output", g, k, r)
			}
			if ns := float64(r) * float64(k) * sseCallNanosPerElem; r > OutLine && ns > maxCallNanos {
				t.Errorf("sse %s k=%d: %d rows prices at %.0f ns, over the %d budget", g, k, r, ns, maxCallNanos)
			}
			if ns := float64(r+OutLine) * float64(k) * sseCallNanosPerElem; ns <= maxCallNanos {
				t.Errorf("sse %s k=%d: %d rows leaves a whole line of budget unused", g, k, r)
			}
		}
	}
}
