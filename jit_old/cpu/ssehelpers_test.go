//go:build amd64

package cpu

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
)

// The SSE tier's shared helpers, each gated alone before any family builds on
// it (a softmax built on a broken exp still sums to 1). Every kernel built here
// also passes the VEX-leak gate.

// expKernelSSE exps 8 floats (two halves): Out = exp(A), Scr = the constants.
func expKernelSSE() []byte {
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RSI, At(RDI, 16)) // A -> x
	a.MOVLoad(RBX, At(RDI, 56)) // Scr
	loadExpConstsSSE(&a)
	for _, off := range []int32{0, 16} {
		a.MOVUPSLoad(XMM0, At(RSI, off))
		emitExpSSE(&a, XMM0, XMM1, XMM2)
		a.MOVUPSStore(At(RCX, off), XMM0)
	}
	a.RET()
	return a.Bytes()
}

// expKernelAVX2 is TestEmitExpPS's kernel, for the side-by-side error report.
func expKernelAVX2() []byte {
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))
	a.MOVLoad(RSI, At(RDI, 16))
	a.MOVLoad(RBX, At(RDI, 56))
	loadExpConsts(&a)
	a.VMOVDQULoad(Y0, At(RSI, 0))
	emitExpPS(&a, Y0, Y1, Y2)
	a.VMOVDQUStore(At(RCX, 0), Y0)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// TestEmitExpSSE holds the FMA-free exp to TestEmitExpPS's bound on the same
// fixed points, and on a dense sweep of the whole clamped domain to the AVX2
// kernel's own error measured in the same run. The sweep cannot use the 3e-6
// bound: TestEmitExpPS's points are all <= 0, and over [-87, 88] the degree-5
// polynomial exceeds it on both tiers, so the bound is relative to AVX2.
func TestEmitExpSSE(t *testing.T) {
	code := expKernelSSE()
	requireSSEKernel(t, "exp_sse", code)
	sse := mustMap(t, code)
	defer sse.Close()
	var avx *Code
	if HostTier() == TierAVX2 {
		avx = mustMap(t, expKernelAVX2())
		defer avx.Close()
	}
	consts := expConsts()
	call := func(c *Code, x []float32) []float32 {
		got := make([]float32, 8)
		c.Call(&Args{Out: &got[0], A: (*int8)(unsafe.Pointer(&x[0])), Scr: (*byte)(unsafe.Pointer(&consts[0]))})
		return got
	}
	xs := []float32{0, -0.5, -1, -2, -5, -10, -20, -40,
		-0.001, -0.125, -0.347, -0.694, -1.386, -50, -80, -87,
		-88, -100, -200, -1000, -3, -7, -15, -30}
	for v := float32(-87); v <= 88; v += 0.0137 {
		xs = append(xs, v)
	}
	for len(xs)%8 != 0 {
		xs = append(xs, 1)
	}
	var worstSSE, worstAVX float64
	for i := 0; i < len(xs); i += 8 {
		x := xs[i : i+8]
		got := call(sse, x)
		var ref []float32
		if avx != nil {
			ref = call(avx, x)
		}
		for j, v := range x {
			want := math.Exp(float64(v))
			if want < 1e-38 {
				if got[j] > 1e-30 {
					t.Errorf("exp(%g) = %g, want ~0", v, got[j])
				}
				continue
			}
			rel := math.Abs(float64(got[j])-want) / want
			worstSSE = math.Max(worstSSE, rel)
			if bound := map[bool]float64{true: 3e-6, false: 4e-6}[i < 24]; rel > bound {
				t.Errorf("exp(%g) = %g, want %g (rel %.2e, bound %.0e)", v, got[j], want, rel, bound)
			}
			if ref != nil {
				worstAVX = math.Max(worstAVX, math.Abs(float64(ref[j])-want)/want)
			}
		}
	}
	if v := call(sse, []float32{0, 0, 0, 0, 0, 0, 0, 0})[0]; v != 1 {
		t.Errorf("exp(0) = %v, want exactly 1", v)
	}
	if avx != nil && worstSSE > 1.1*worstAVX {
		t.Errorf("worst relative error %.2e is more than 10%% over the AVX2 kernel's %.2e", worstSSE, worstAVX)
	}
	if avx == nil {
		t.Logf("%d points: worst relative error SSE %.2e (no AVX2 on this host to compare)", len(xs), worstSSE)
		return
	}
	t.Logf("%d points: worst relative error SSE %.2e, AVX2 (FMA) %.2e", len(xs), worstSSE, worstAVX)
}

// halfKernelSSE converts Args.K groups of four binary16 at W into float32 at
// Out through halfToFloatSSE.
func halfKernelSSE() []byte {
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RSI, At(RDI, 8))  // W -> halves
	a.MOVLoad(R10, At(RDI, 40)) // K -> groups of four
	a.PXOR(XMM7, XMM7, XMM7)
	loadHalfMagicSSE(&a, XMM6, RAX)
	lp := a.Label()
	a.Bind(lp)
	loadU16x4SSE(&a, XMM0, At(RSI, 0), XMM7)
	halfToFloatSSE(&a, XMM1, XMM0, XMM2, XMM3, XMM6)
	a.MOVUPSStore(At(RCX, 0), XMM1)
	a.ADDimm(RSI, 8)
	a.ADDimm(RCX, 16)
	a.DEC(R10)
	a.JNZ(lp)
	a.RET()
	return a.Bytes()
}

// halfKernelF16C is the same conversion through VCVTPH2PS, eight at a time.
func halfKernelF16C() []byte {
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))
	a.MOVLoad(RSI, At(RDI, 8))
	a.MOVLoad(R10, At(RDI, 40)) // groups of EIGHT here
	lp := a.Label()
	a.Bind(lp)
	a.VCVTPH2PS(Y0, At(RSI, 0))
	a.VMOVDQUStore(At(RCX, 0), Y0)
	a.ADDimm(RSI, 16)
	a.ADDimm(RCX, 32)
	a.DEC(R10)
	a.JNZ(lp)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// TestSoftwareHalfMatchesF16C runs every one of the 65536 binary16 values
// through the software convert and compares it bit for bit with the hardware
// instruction the AVX2 tier uses and with quant's own decoder.
//
// All of them, because the cases that break (subnormals, infinities, NaN
// payloads, zeros) are rare. The one allowed difference is NaN quieting: F16C
// sets the quiet bit of a signalling NaN and the software convert does not, so
// for NaNs the quiet bit is masked.
func TestSoftwareHalfMatchesF16C(t *testing.T) {
	code := halfKernelSSE()
	requireSSEKernel(t, "half_sse", code)
	c := mustMap(t, code)
	defer c.Close()
	in := make([]uint16, 65536)
	for i := range in {
		in[i] = uint16(i)
	}
	got := make([]float32, 65536)
	c.Call(&Args{Out: &got[0], W: (*byte)(unsafe.Pointer(&in[0])), K: 65536 / 4})

	var hw []float32
	if CPU().F16C && CPU().UsableAVX2() {
		h := mustMap(t, halfKernelF16C())
		defer h.Close()
		hw = make([]float32, 65536)
		h.Call(&Args{Out: &hw[0], W: (*byte)(unsafe.Pointer(&in[0])), K: 65536 / 8})
	} else {
		t.Log("no F16C on this host: comparing against quant.DecodeHalf alone")
	}
	bad, nans, subs := 0, 0, 0
	for u := 0; u < 65536; u++ {
		g := math.Float32bits(got[u])
		ref := float32(quant.DecodeHalf(uint16(u)))
		isNaN := u&0x7C00 == 0x7C00 && u&0x3FF != 0
		if u&0x7C00 == 0 && u&0x3FF != 0 {
			subs++
		}
		if isNaN {
			nans++
			if !math.IsNaN(float64(got[u])) {
				t.Errorf("half %#04x is a NaN and converted to %v", u, got[u])
				bad++
			}
			want := uint32(u&0x8000)<<16 | 0x7F800000 | uint32(u&0x3FF)<<13
			if g != want {
				t.Errorf("half %#04x: NaN payload %#08x, want %#08x", u, g, want)
				bad++
			}
		} else if g != math.Float32bits(ref) {
			t.Errorf("half %#04x: %#08x (%g), quant.DecodeHalf says %#08x (%g)", u, g, got[u], math.Float32bits(ref), ref)
			bad++
		}
		if hw != nil {
			h := math.Float32bits(hw[u])
			if isNaN {
				g |= 1 << 22 // the quiet bit F16C sets
			}
			if g != h {
				t.Errorf("half %#04x: software %#08x, VCVTPH2PS %#08x", u, g, h)
				bad++
			}
		}
		if bad > 10 {
			t.Fatal("too many mismatches")
		}
	}
	t.Logf("65536 halves (%d subnormals, %d NaNs) bit-identical to %s", subs, nans,
		map[bool]string{true: "VCVTPH2PS and quant.DecodeHalf", false: "quant.DecodeHalf"}[hw != nil])
}

// TestBF16WidenSSE runs every bfloat16 through both halves of the widen.
func TestBF16WidenSSE(t *testing.T) {
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))
	a.MOVLoad(RSI, At(RDI, 8))
	a.MOVLoad(R10, At(RDI, 40)) // groups of eight
	lp := a.Label()
	a.Bind(lp)
	a.MOVDQULoad(XMM0, At(RSI, 0))
	bf16ToFloatSSE(&a, XMM1, XMM0)
	bf16HiToFloatSSE(&a, XMM2, XMM0)
	a.MOVUPSStore(At(RCX, 0), XMM1)
	a.MOVUPSStore(At(RCX, 16), XMM2)
	a.ADDimm(RSI, 16)
	a.ADDimm(RCX, 32)
	a.DEC(R10)
	a.JNZ(lp)
	a.RET()
	requireSSEKernel(t, "bf16_sse", a.Bytes())
	c := mustMap(t, a.Bytes())
	defer c.Close()
	in := make([]uint16, 65536)
	for i := range in {
		in[i] = uint16(i)
	}
	got := make([]float32, 65536)
	c.Call(&Args{Out: &got[0], W: (*byte)(unsafe.Pointer(&in[0])), K: 65536 / 8})
	for u := 0; u < 65536; u++ {
		if g, w := math.Float32bits(got[u]), uint32(u)<<16; g != w {
			t.Fatalf("bf16 %#04x widened to %#08x, want %#08x", u, g, w)
		}
	}
}

// TestReductionsMatchTheAVX2Fold checks hsum8SSE and hmax8SSE against the
// exact AVX2 sequences they mirror (EmitRMSNorm's sum fold, EmitSoftmax's max
// fold), bit for bit, on data that includes the values where fold order
// shows: NaNs (MAXPS returns its second operand), signed zeros, infinities
// and denormals.
func TestReductionsMatchTheAVX2Fold(t *testing.T) {
	if HostTier() != TierAVX2 {
		t.Skip("the AVX2 fold is the reference; this host cannot run it")
	}
	var s Buf
	s.DeclareISA(ISATierSSE)
	s.MOVLoad(RCX, At(RDI, 0))
	s.MOVLoad(RSI, At(RDI, 16))
	s.MOVUPSLoad(XMM0, At(RSI, 0))
	s.MOVUPSLoad(XMM1, At(RSI, 16))
	hsum8SSE(&s, XMM0, XMM1)
	s.MOVUPSStore(At(RCX, 0), XMM0)
	s.MOVUPSLoad(XMM2, At(RSI, 0))
	s.MOVUPSLoad(XMM3, At(RSI, 16))
	hmax8SSE(&s, XMM2, XMM3, XMM4)
	s.MOVUPSStore(At(RCX, 16), XMM2)
	s.RET()
	requireSSEKernel(t, "reduce_sse", s.Bytes())

	var v Buf
	v.MOVLoad(RCX, At(RDI, 0))
	v.MOVLoad(RSI, At(RDI, 16))
	v.VMOVDQULoad(Y0, At(RSI, 0))
	v.VEXTRACTF128(Y1, Y0, 1)
	v.VADDPSx(Y0, Y0, Y1)
	v.VHADDPSx(Y0, Y0, Y0)
	v.VHADDPSx(Y0, Y0, Y0)
	v.VMOVDQUStorex(At(RCX, 0), Y0)
	v.VMOVDQULoad(Y3, At(RSI, 0))
	v.VEXTRACTF128(Y1, Y3, 1)
	v.VMAXPS(Y3, Y3, Y1)
	v.VSHUFPS(Y1, Y3, Y3, 0x4E)
	v.VMAXPS(Y3, Y3, Y1)
	v.VSHUFPS(Y1, Y3, Y3, 0xB1)
	v.VMAXPS(Y3, Y3, Y1)
	v.VBROADCASTSSReg(Y3, Y3)
	v.VMOVDQUStorex(At(RCX, 16), Y3)
	v.VZEROUPPER()
	v.RET()

	cs, cv := mustMap(t, s.Bytes()), mustMap(t, v.Bytes())
	defer cs.Close()
	defer cv.Close()
	special := []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)),
		float32(math.Copysign(0, -1)), 0, 1e-40, -1e-40, math.MaxFloat32, -math.MaxFloat32}
	r := rand.New(rand.NewSource(1))
	for it := 0; it < 20000; it++ {
		x := make([]float32, 8)
		for i := range x {
			if r.Intn(4) == 0 {
				x[i] = special[r.Intn(len(special))]
			} else {
				x[i] = float32(r.NormFloat64() * math.Pow(10, float64(r.Intn(8)-4)))
			}
		}
		gs, gv := make([]float32, 8), make([]float32, 8)
		cs.Call(&Args{Out: &gs[0], A: (*int8)(unsafe.Pointer(&x[0]))})
		cv.Call(&Args{Out: &gv[0], A: (*int8)(unsafe.Pointer(&x[0]))})
		for i := range gs {
			if math.Float32bits(gs[i]) != math.Float32bits(gv[i]) {
				what := "sum"
				if i >= 4 {
					what = "max"
				}
				t.Fatalf("%s lane %d of %v: SSE %#08x, AVX2 %#08x", what, i%4, x,
					math.Float32bits(gs[i]), math.Float32bits(gv[i]))
			}
		}
	}
}

// TestElemLoopSSEEveryWidth runs a two-cursor body through elemLoopSSE at
// every width from 0 to five units plus a tail, on misaligned buffers with
// NaN-payload guards on both sides: dst[i] += src[i]. A legacy m128 operand
// #GPs on an unaligned address and Go does not align a []float32 to 16, so the
// buffers start 4 bytes past a 16-byte boundary.
func TestElemLoopSSEEveryWidth(t *testing.T) {
	var a Buf
	a.DeclareISA(ISATierSSE)
	a.MOVLoad(RCX, At(RDI, 0))  // Out -> dst
	a.MOVLoad(RSI, At(RDI, 24)) // AScale -> src
	a.MOVLoad(R10, At(RDI, 40)) // K
	a.MOVLoad(R11, At(RDI, 32)) // Rows
	elemLoopSSE(&a, R10, R11, []Reg{RCX, RSI}, func(ld func(dst, p Reg), st func(p, src Reg)) {
		ld(XMM0, RCX)
		ld(XMM1, RSI)
		a.ADDPS(XMM0, XMM0, XMM1)
		st(RCX, XMM0)
	})
	a.RET()
	requireSSEKernel(t, "elemloop_sse", a.Bytes())
	c := mustMap(t, a.Bytes())
	defer c.Close()

	guard := math.Float32frombits(0x7FC0BEEF)
	misaligned := func(n int) []float32 {
		raw := make([]float32, n+16)
		off := 0
		for uintptr(unsafe.Pointer(&raw[off]))%16 != 4 {
			off++
		}
		return raw[off : off+n+8]
	}
	for n := 0; n <= 5*ElemLanes+ElemLanes-1; n++ {
		dst, src := misaligned(n+2), misaligned(n+2)
		for i := range dst {
			dst[i], src[i] = guard, guard
		}
		for i := 0; i < n; i++ {
			dst[1+i], src[1+i] = float32(i), float32(100*i)
		}
		if n > 0 {
			c.Call(&Args{Out: &dst[1], AScale: &src[1], K: int64(n / ElemLanes), Rows: int64(n % ElemLanes)})
		}
		for i := range dst {
			want := math.Float32bits(guard)
			if i >= 1 && i <= n {
				want = math.Float32bits(float32(101 * (i - 1)))
			}
			if got := math.Float32bits(dst[i]); got != want {
				t.Fatalf("n=%d: dst[%d] = %#08x, want %#08x (a guard at 0 or past %d means a stray read or write)", n, i, got, want, n)
			}
		}
	}
}
