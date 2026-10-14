//go:build amd64

package cpu

import (
	"errors"
	"math"
	"math/rand"
	"strings"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/internal/oracle"
)

// The SSE tier's elementwise family (sse_elem.go), gated like the AVX2
// family, plus what only this tier needs:
//
//   - every kernel comes from EmittersFor(TierSSE), so a miswired table field
//     fails a numeric gate;
//   - every kernel passes the VEX-leak gate and carries an SSE-tier
//     declaration;
//   - every kernel executes on buffers 4 bytes past a 16-byte boundary with
//     NaN guards, because a legacy m128 operand faults on exactly the
//     misalignment a Go slice has;
//   - on an AVX2 host, each kernel's worst error is held to the AVX2 kernel's
//     worst error on the same inputs.
//
// Map failing is fatal, never mustMap's skip: an SSE-tier kernel must map on
// every host at or above the SSE floor.

const sseElemGuard = 0x7fc0beef

// sseElemBuf returns n floats that start 4 bytes past a 16-byte boundary,
// inside a backing slice with 2*ElemLanes NaN guards on each side.
func sseElemBuf(n int) (full, v []float32) {
	const g = 2 * ElemLanes
	raw := make([]float32, n+2*g+4)
	off := 0
	for uintptr(unsafe.Pointer(&raw[off+g]))%16 != 4 {
		off++
	}
	full = raw[off : off+n+2*g]
	for i := range full {
		full[i] = math.Float32frombits(sseElemGuard)
	}
	return full, full[g : g+n]
}

// sseElemGuardsHold fails t if anything before or after the n-element view of
// full was written.
func sseElemGuardsHold(t *testing.T, what string, full []float32, n int) {
	t.Helper()
	const g = 2 * ElemLanes
	for i := range full {
		if i >= g && i < g+n {
			continue
		}
		if b := math.Float32bits(full[i]); b != sseElemGuard {
			t.Fatalf("%s n=%d: guard %d (element %d relative to the start) was written: %#08x", what, n, i, i-g, b)
		}
	}
}

// sseElemMap emits one kernel through the SSE table, holds it to the tier's
// declaration and the VEX-leak gate, and maps it.
func sseElemMap(t *testing.T, name string, emit func() ([]byte, error)) *Code {
	t.Helper()
	code, err := emit()
	if err != nil {
		t.Fatalf("%s: the SSE table refused: %v", name, err)
	}
	if KernelTier(code) != TierSSE {
		t.Fatalf("%s: the SSE table returned a kernel that is not declared SSE-tier", name)
	}
	requireSSEKernel(t, "sse_"+strings.NewReplacer("/", "_", "-", "_").Replace(name), code)
	c, err := Map(code)
	if err != nil {
		t.Fatalf("%s: Map: %v -- an SSE-tier kernel must map on every host at or above the SSE floor", name, err)
	}
	return c
}

// sseElemAVX2 maps an AVX2 twin when the host can run one, and returns nil on
// a host that cannot (an Atom), where the twin comparisons are simply absent.
func sseElemAVX2(t *testing.T, code []byte) *Code {
	t.Helper()
	if HostTier() != TierAVX2 {
		return nil
	}
	c, err := Map(code)
	if err != nil {
		t.Fatalf("AVX2 twin: %v", err)
	}
	return c
}

// sseElemKern is one elementwise kernel of the family, as the width, twin and
// inventory gates need it: how to emit it on each tier, whether it reads a
// second vector, and the float64 definition it computes.
type sseElemKern struct {
	name string
	sse  func() ([]byte, error)
	avx2 func() []byte
	src  bool   // reads AScale as a second vector
	scr  []byte // what Scr points at (ActConsts or &alpha)
	w    []byte // what W points at (softcap's {2/c, c}), or nil
	ref  func(x, u float64) float64
	// twoRoundings marks a kernel whose AVX2 twin FUSES its final
	// multiply-add, which this tier rounds twice: its accuracy is not the
	// AVX2 kernel's by construction, and its gate is bit-identity with the
	// unfused loop instead (TestEmitAxpySSEMatchesReference).
	twoRoundings bool
}

func sseElemF32Bytes(v ...float32) []byte {
	b := make([]byte, 4*len(v))
	copy(b, unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(b)))
	return b
}

func refQuickGELUMul(g, u float64) float64 { return quickGELURef(g) * u }

// sseElemKerns is the family's kernel list at the configurations the engine
// runs: alpha != 1 for axpy and scale (alpha == 1 cannot tell a kernel that
// ignores Scr from a correct one), a cap small enough that the inputs reach
// saturation, every gated and every ungated kind, and the two combinations
// the AVX2 emitter also serves outside those lists.
func sseElemKerns() []sseElemKern {
	sse := EmittersFor(TierSSE)
	consts := sseElemF32Bytes(ActConsts()...)
	alpha := float32(-0.625)
	capC := float32(3)
	ks := []sseElemKern{
		{"axpy", sse.Axpy, EmitAxpy, true, sseElemF32Bytes(alpha), nil,
			func(x, u float64) float64 { return x + float64(alpha)*u }, true},
		{"scale", sse.Scale, EmitScale, false, sseElemF32Bytes(alpha), nil,
			func(x, _ float64) float64 { return x * float64(alpha) }, false},
		{"softcap", sse.Softcap, EmitSoftcap, false, consts, sseElemF32Bytes(2/capC, capC),
			func(x, _ float64) float64 { return float64(capC) * math.Tanh(x/float64(capC)) }, false},
		{"sigmoidmul", sse.SigmoidMul, EmitSigmoidMul, true, consts, nil,
			func(g, u float64) float64 { return refSigmoid(g) * u }, false},
		{"clamp", sse.Clamp, EmitClamp, false, sseElemF32Bytes(-2.5, 4), nil,
			func(x, _ float64) float64 { return math.Min(math.Max(x, -2.5), 4) }, false},
	}
	gated := map[ActKind]func(g, u float64) float64{
		ActSiLU:        func(g, u float64) float64 { return oracle.SiLU(g) * u },
		ActGELU:        func(g, u float64) float64 { return oracle.GELUTanh(g) * u },
		ActGELUErf:     func(g, u float64) float64 { return oracle.GELUErf(g) * u },
		ActSwiGLUOAI:   oracle.SwiGLUOAI,
		ActSwiGLUClamp: oracle.SwiGLUClamp,
		ActSitu:        oracle.Situ,
		ActIdentity:    func(g, u float64) float64 { return g * u },
		ActQuickGELU:   refQuickGELUMul,
	}
	for _, k := range append(Gated[:], ActQuickGELU) {
		ks = append(ks, sseElemKern{"actmul/" + k.String(), func() ([]byte, error) { return sse.ActMul(k) },
			func() []byte { return EmitActMul(k) }, true, consts, nil, gated[k], false})
	}
	ungated := map[ActKind]func(x, _ float64) float64{
		ActSiLU:        func(x, _ float64) float64 { return oracle.SiLU(x) },
		ActGELU:        func(x, _ float64) float64 { return oracle.GELUTanh(x) },
		ActGELUErf:     func(x, _ float64) float64 { return oracle.GELUErf(x) },
		ActSqrtSigmoid: func(x, _ float64) float64 { return oracle.SqrtSigmoid(x) },
		ActQuickGELU:   func(x, _ float64) float64 { return quickGELURef(x) },
		ActReLU2:       func(x, _ float64) float64 { r := math.Max(x, 0); return r * r },
		ActReLU:        func(x, _ float64) float64 { return math.Max(x, 0) },
		actSigmoid:     func(x, _ float64) float64 { return refSigmoid(x) },
		ActSqrtSoftplus: func(x, _ float64) float64 {
			return oracle.SqrtSoftplus(x)
		},
	}
	for _, k := range append(Ungated[:], actSigmoid) {
		ks = append(ks, sseElemKern{"act/" + k.String(), func() ([]byte, error) { return sse.Act(k) },
			func() []byte { return EmitAct(k) }, false, consts, nil, ungated[k], false})
	}
	return ks
}

// run calls c over dst[:n] (and src[:n]).
func (k sseElemKern) run(c *Code, dst, src []float32, n int) {
	args := Args{
		Out:  &dst[0],
		Scr:  &k.scr[0],
		K:    int64(n / ElemLanes),
		Rows: int64(n % ElemLanes),
	}
	if k.src {
		args.AScale = &src[0]
	}
	if k.w != nil {
		args.W = &k.w[0]
	}
	c.Call(&args)
}

// sseElemInputs fills a width-n input: gates and values spanning both signs,
// past where every activation saturates and past swiglu-oai's +-7 clamps.
func sseElemInputs(dst, src []float32) {
	for i := range dst {
		dst[i] = float32(math.Sin(float64(i)*0.37) * float64(1+i%17))
		src[i] = float32(math.Cos(float64(i)*0.23) * float64(1+i%13))
	}
}

// TestElementwiseSSEEveryWidth is TestElementwiseEveryWidth for the SSE tier:
// every kernel at widths 1..5*ElemLanes, each element bit-identical to a
// whole-unit run (the ops are lane-independent), nothing outside the slice
// written. An 8-float unit is two XMM bodies at offsets 0 and 16, so widths
// 5..8 and past 12 catch a body that loads the high half from the wrong
// offset. The whole-unit run is itself held to the float64 definition.
func TestElementwiseSSEEveryWidth(t *testing.T) {
	const max = 5 * ElemLanes
	for _, k := range sseElemKerns() {
		c := sseElemMap(t, k.name, k.sse)
		_, whole := sseElemBuf(max)
		_, wsrc := sseElemBuf(max)
		sseElemInputs(whole, wsrc)
		in := append([]float32(nil), whole...)
		k.run(c, whole, wsrc, max)
		var sse, sy2 float64
		for i := range whole {
			want := k.ref(float64(in[i]), float64(wsrc[i]))
			d := float64(whole[i]) - want
			sse, sy2 = sse+d*d, sy2+want*want
		}
		if nmse := sse / sy2; !(nmse <= 1e-9) {
			t.Fatalf("%s: the whole-unit run is NMSE %.3e from the float64 definition", k.name, nmse)
		}
		for n := 1; n <= max; n++ {
			full, dst := sseElemBuf(n)
			sfull, src := sseElemBuf(n)
			sseElemInputs(dst, src)
			k.run(c, dst, src, n)
			for i := 0; i < n; i++ {
				if math.Float32bits(dst[i]) != math.Float32bits(whole[i]) {
					t.Fatalf("%s width %d: element %d is %v, the whole-unit run gives %v", k.name, n, i, dst[i], whole[i])
				}
			}
			sseElemGuardsHold(t, k.name, full, n)
			sseElemGuardsHold(t, k.name+" (src)", sfull, n)
		}
		c.Close()
	}
}

// TestSSEElementwiseIsAsAccurateAsAVX2 holds every kernel's worst error
// against its float64 definition to the AVX2 kernel's worst error on the same
// inputs, measured in the same run.
//
// The metric is |error| / max(|want|, 1): GELU-tanh cancels catastrophically
// for very negative x on both tiers, so a pure relative bar would measure that
// shared cancellation. A worst-element bar also catches errors where outputs
// are small, which an NMSE (dominated by the largest outputs) would miss.
func TestSSEElementwiseIsAsAccurateAsAVX2(t *testing.T) {
	const n = 4099
	for _, k := range sseElemKerns() {
		c := sseElemMap(t, k.name, k.sse)
		avx := sseElemAVX2(t, k.avx2())
		worst := func(c *Code) (float64, int) {
			_, dst := sseElemBuf(n)
			_, src := sseElemBuf(n)
			sseElemInputs(dst, src)
			// And a dense sweep of the part of the range where the
			// activations bend, so a wrong constant cannot hide in the gaps.
			for i := 0; i < 1024; i++ {
				dst[i] = float32(-12 + 24*float64(i)/1023)
			}
			in := append([]float32(nil), dst...)
			k.run(c, dst, src, n)
			w, wi := 0.0, 0
			for i := range dst {
				want := k.ref(float64(in[i]), float64(src[i]))
				if e := math.Abs(float64(dst[i])-want) / math.Max(math.Abs(want), 1); !(e <= w) {
					w, wi = e, i
				}
			}
			return w, wi
		}
		ws, wi := worst(c)
		c.Close()
		if !(ws <= 1e-5) {
			t.Errorf("%s: worst error %.3e at element %d, over the 1e-5 every elementwise gate here allows", k.name, ws, wi)
		}
		if avx == nil || k.twoRoundings {
			if avx != nil {
				avx.Close()
			}
			t.Logf("%s: worst error %.3e (not compared with the AVX2 kernel: %s)", k.name, ws,
				map[bool]string{true: "it fuses the multiply-add this tier rounds twice", false: "no AVX2 on this host"}[k.twoRoundings])
			continue
		}
		wa, _ := worst(avx)
		avx.Close()
		// An FMA-free exp is the same accuracy class but not the same bits,
		// so the bar is the AVX2 kernel's worst plus a quarter, floored at
		// one f32 ulp of 1 for kernels that round exactly on both tiers.
		if bound := math.Max(1.25*wa, 1.2e-7); ws > bound {
			t.Errorf("%s: worst error %.3e, against the AVX2 kernel's %.3e on the same inputs (bound %.3e)", k.name, ws, wa, bound)
		}
		t.Logf("%s: worst error SSE %.3e, AVX2 %.3e", k.name, ws, wa)
	}
}

// TestEmitActMulSSEMatchesReference is TestEmitActMulMatchesReference for the
// SSE tier: every gated kind (and quick-GELU gated, which the emitter also
// serves) against internal/oracle, with swiglu-oai's clamps reached and the
// two controls that make "GELU works" a statement about GELU.
func TestEmitActMulSSEMatchesReference(t *testing.T) {
	consts := ActConsts()
	sse := EmittersFor(TierSSE)
	ref := func(k ActKind, g, u float64) float64 {
		switch k {
		case ActGELU:
			return oracle.GELUTanh(g) * u
		case ActGELUErf:
			return oracle.GELUErf(g) * u
		case ActIdentity:
			return g * u
		case ActSwiGLUOAI:
			return oracle.SwiGLUOAI(g, u)
		case ActSwiGLUClamp:
			return oracle.SwiGLUClamp(g, u)
		case ActSitu:
			return oracle.Situ(g, u)
		case ActQuickGELU:
			return quickGELURef(g) * u
		}
		return oracle.SiLU(g) * u
	}
	for _, k := range append(Gated[:], ActQuickGELU) {
		c := sseElemMap(t, "actmul/"+k.String(), func() ([]byte, error) { return sse.ActMul(k) })
		for _, n := range []int{1, 3, 4, 5, 7, 8, 9, 12, 13, 16, 17, 203, 1024, 1027} {
			full, dst := sseElemBuf(n)
			ufull, up := sseElemBuf(n)
			want := make([]float64, n)
			gHigh, uOut := 0, 0
			for i := range dst {
				// Phase-shifted by one from the AVX2 gate's inputs, whose
				// element 0 is sin(0) = 0: at n=1 that is an all-zero
				// reference, and an NMSE of 0/0 is a degenerate oracle.
				v := math.Sin(float64(i+1)*0.31) * float64(1+i%17)
				u := math.Sin(float64(i+1)*0.13) * float64(1+i%13)
				dst[i], up[i] = float32(v), float32(u)
				want[i] = ref(k, float64(dst[i]), float64(up[i]))
				if v > 7 {
					gHigh++
				}
				if math.Abs(u) > 7 {
					uOut++
				}
			}
			if n == 1024 && (gHigh == 0 || uOut == 0) {
				t.Fatalf("%s: %d gates above 7 and %d ups outside +-7 -- the clamps are not exercised", k, gHigh, uOut)
			}
			c.Call(&Args{Out: &dst[0], AScale: &up[0], Scr: (*byte)(unsafe.Pointer(&consts[0])),
				K: int64(n / ElemLanes), Rows: int64(n % ElemLanes)})
			var se, sy2 float64
			for i := range want {
				d := float64(dst[i]) - want[i]
				se, sy2 = se+d*d, sy2+want[i]*want[i]
			}
			if nmse := se / sy2; !(nmse <= 1e-9) {
				t.Errorf("%s n=%d: NMSE %.3e against internal/oracle", k, n, nmse)
			}
			sseElemGuardsHold(t, "actmul/"+k.String(), full, n)
			sseElemGuardsHold(t, "actmul/"+k.String()+" (up)", ufull, n)
		}
		c.Close()
	}

	// swiglu-oai is not SiLU times up: the control for its three differences.
	var d2 float64
	for i := -40; i <= 40; i++ {
		g, u := float64(i)*0.25, float64(-i)*0.3
		d2 += math.Abs(ref(ActSwiGLUOAI, g, u) - ref(ActSiLU, g, u))
	}
	if d2 < 1e-3 {
		t.Fatalf("swiglu-oai and SiLU*up agree to %g; this test cannot tell them apart", d2)
	}
	var diff float64
	for i := -40; i <= 40; i++ {
		x := float64(i) * 0.25
		diff += math.Abs(oracle.SiLU(x) - oracle.GELUTanh(x))
	}
	if diff < 1e-3 {
		t.Fatalf("SiLU and GELU agree to %g over the sampled range; this test cannot tell them apart", diff)
	}
}

// TestEmitSigmoidMulSSEMatchesReference is TestEmitSigmoidMulMatchesReference
// for the SSE tier, with its control: sigma(g)*v is NOT SiLU(g)*v, and a
// kernel that emitted the SiLU body here would be right exactly where |g| is
// near 1 and wrong everywhere a gate saturates.
func TestEmitSigmoidMulSSEMatchesReference(t *testing.T) {
	consts := ActConsts()
	c := sseElemMap(t, "sigmoidmul", EmittersFor(TierSSE).SigmoidMul)
	defer c.Close()
	for _, n := range []int{1, 6, 8, 11, 1024, 1029} {
		full, gate := sseElemBuf(n)
		_, val := sseElemBuf(n)
		want := make([]float64, n)
		silu := make([]float64, n)
		for i := range gate {
			g := math.Sin(float64(i)*0.37) * float64(1+i%23)
			gate[i] = float32(g)
			val[i] = float32(0.5 + float64(i%11)*0.2)
			want[i] = refSigmoid(float64(gate[i])) * float64(val[i])
			silu[i] = oracle.SiLU(float64(gate[i])) * float64(val[i])
		}
		c.Call(&Args{Out: &gate[0], AScale: &val[0], Scr: (*byte)(unsafe.Pointer(&consts[0])),
			K: int64(n / ElemLanes), Rows: int64(n % ElemLanes)})
		var se, sy2, sep float64
		for i := range want {
			d := float64(gate[i]) - want[i]
			se, sy2 = se+d*d, sy2+want[i]*want[i]
			sep += math.Abs(float64(gate[i]) - silu[i])
		}
		if nmse := se / sy2; !(nmse <= 1e-9) {
			t.Errorf("n=%d: NMSE %.3e against the float64 reference", n, nmse)
		}
		if n >= 1024 && sep < 1 {
			t.Errorf("n=%d: the kernel is within %g of SiLU(g)*v in total -- the control cannot tell the two apart", n, sep)
		}
		sseElemGuardsHold(t, "sigmoidmul", full, n)
	}
}

// TestEmitActSSEMatchesActMulSSE is TestEmitActMatchesActMul for the SSE
// tier: the ungated kernel is the gated one with `up` = 1, bit for bit (a
// multiply by 1.0 is exact), and the control is that a real `up` makes them
// disagree -- otherwise an ungated kernel that read `up` anyway would pass.
func TestEmitActSSEMatchesActMulSSE(t *testing.T) {
	consts := ActConsts()
	sse := EmittersFor(TierSSE)
	const n = 1029
	for _, k := range append(Ungated[:], actSigmoid) {
		gated := sseElemMap(t, "actmul/"+k.String(), func() ([]byte, error) { return sse.ActMul(k) })
		plain := sseElemMap(t, "act/"+k.String(), func() ([]byte, error) { return sse.Act(k) })
		mk := func() []float32 {
			_, d := sseElemBuf(n)
			for i := range d {
				d[i] = float32(math.Sin(float64(i)*0.31) * float64(1+i%17))
			}
			return d
		}
		a, b := mk(), mk()
		_, ones := sseElemBuf(n)
		for i := range ones {
			ones[i] = 1
		}
		args := Args{Out: &a[0], AScale: &ones[0], Scr: (*byte)(unsafe.Pointer(&consts[0])),
			K: int64(n / ElemLanes), Rows: int64(n % ElemLanes)}
		gated.Call(&args)
		plain.Call(&Args{Out: &b[0], Scr: (*byte)(unsafe.Pointer(&consts[0])),
			K: int64(n / ElemLanes), Rows: int64(n % ElemLanes)})
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				t.Fatalf("%s: ungated differs from gated-by-ones at %d: %v vs %v", k, i, b[i], a[i])
			}
		}
		c := mk()
		_, u := sseElemBuf(n)
		for i := range u {
			u[i] = float32(0.5 + float64(i%9)*0.1)
		}
		args.Out, args.AScale = &c[0], &u[0]
		gated.Call(&args)
		same := true
		for i := range c {
			same = same && c[i] == b[i]
		}
		if same {
			t.Fatalf("%s: gated and ungated agree on a non-unit `up`; the control proves nothing", k)
		}
		gated.Close()
		plain.Close()
	}
}

// TestQuickGELUSSEMatchesItsDefinition is TestQuickGELUMatchesItsDefinition
// for the SSE tier, including its separation check: quick-GELU must be far
// from SiLU and from GELU-tanh as the SSE kernels compute them, so the gate
// cannot be satisfied by either of the wrong two.
func TestQuickGELUSSEMatchesItsDefinition(t *testing.T) {
	sse := EmittersFor(TierSSE)
	consts := ActConsts()
	const n = 4096
	_, x := sseElemBuf(n)
	for i := range x {
		x[i] = float32(-12 + 24*float64(i)/float64(n-1))
	}
	runAct := func(k ActKind) []float32 {
		c := sseElemMap(t, "act/"+k.String(), func() ([]byte, error) { return sse.Act(k) })
		defer c.Close()
		full, buf := sseElemBuf(n)
		copy(buf, x)
		c.Call(&Args{Out: &buf[0], Scr: (*byte)(unsafe.Pointer(&consts[0])), K: n / ElemLanes, Rows: n % ElemLanes})
		sseElemGuardsHold(t, "act/"+k.String(), full, n)
		return buf
	}
	got := runAct(ActQuickGELU)
	worst, wi := 0.0, 0
	for i := range x {
		if d := math.Abs(float64(got[i]) - quickGELURef(float64(x[i]))); !(d <= worst) {
			worst, wi = d, i
		}
	}
	if worst > 1e-5 {
		t.Errorf("quick-GELU worst |error| %.3e at x=%v (kernel %v, want %v)", worst, x[wi], got[wi], quickGELURef(float64(x[wi])))
	}
	for _, other := range []struct {
		k   ActKind
		min float64
	}{{ActSiLU, 0.15}, {ActGELU, 0.02}} {
		buf := runAct(other.k)
		sep := 0.0
		for i := range buf {
			sep = math.Max(sep, math.Abs(float64(buf[i])-float64(got[i])))
		}
		if sep < other.min {
			t.Errorf("quick-GELU and %s differ by at most %.4f over [-12,12]; the gate cannot tell them apart", other.k, sep)
		}
	}
	t.Logf("quick-GELU worst |error| %.3e over %d points in [-12,12]", worst, n)
}

// TestUngatedSSEActivationsAreAllDistinct is TestUngatedActivationsAreAllDistinct
// for the SSE tier: three kinds, three different kernels.
func TestUngatedSSEActivationsAreAllDistinct(t *testing.T) {
	sse := EmittersFor(TierSSE)
	consts := ActConsts()
	const n = 512
	outs := make([][]float32, len(Ungated))
	for i, k := range Ungated {
		c := sseElemMap(t, "act/"+k.String(), func() ([]byte, error) { return sse.Act(k) })
		_, buf := sseElemBuf(n)
		for j := range buf {
			buf[j] = float32(-6 + 12*float64(j)/float64(n-1))
		}
		c.Call(&Args{Out: &buf[0], Scr: (*byte)(unsafe.Pointer(&consts[0])), K: n / ElemLanes})
		c.Close()
		outs[i] = buf
	}
	for i := range outs {
		for j := i + 1; j < len(outs); j++ {
			same := true
			for e := range outs[i] {
				same = same && outs[i][e] == outs[j][e]
			}
			if same {
				t.Errorf("%v and %v emit identical results: two names for one kernel", Ungated[i], Ungated[j])
			}
		}
	}
}

// TestActSSERefusesWhatItCannotBake: an ungated swiglu-oai would read an `up`
// the caller never set, and an unknown code must not become GELU by default.
// The two combinations outside Gated/Ungated that ARE well defined (a gated
// quick-GELU, an ungated sigmoid) emit and are covered by the width gate.
func TestActSSERefusesWhatItCannotBake(t *testing.T) {
	sse := EmittersFor(TierSSE)
	if _, err := sse.Act(ActSwiGLUOAI); err == nil || !strings.Contains(err.Error(), "only gated") {
		t.Errorf("act/swiglu-oai: %v, want a refusal saying it exists only gated", err)
	}
	for _, f := range []func(ActKind) ([]byte, error){sse.Act, sse.ActMul} {
		if _, err := f(ActKind(99)); err == nil || errors.Is(err, ErrNoSSEKernel) {
			t.Errorf("an unknown activation code: %v, want a refusal (and not the pending-family error)", err)
		}
	}
	for _, f := range []func() ([]byte, error){
		func() ([]byte, error) { return sse.ActMul(ActQuickGELU) },
		func() ([]byte, error) { return sse.Act(actSigmoid) },
	} {
		if _, err := f(); err != nil {
			t.Errorf("a well-defined combination was refused: %v", err)
		}
	}
}

// TestEmitAxpySSEMatchesReference: with no FMA, dst + alpha*src is two
// roundings, exactly what the unfused scalar loop computes, so the kernel is
// held bit-identical to it (the explicit float32 conversion below stops Go
// fusing it under GOAMD64=v3), and to oracle.AxpyF32 by NMSE.
func TestEmitAxpySSEMatchesReference(t *testing.T) {
	c := sseElemMap(t, "axpy", EmittersFor(TierSSE).Axpy)
	defer c.Close()
	for _, alpha := range []float32{1, 0.37, -2.5} {
		for _, n := range []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 64, 2048, 2051} {
			full, dst := sseElemBuf(n)
			_, src := sseElemBuf(n)
			for i := range dst {
				dst[i] = float32(math.Sin(float64(i) * 0.7))
				src[i] = float32(math.Cos(float64(i)*0.3) * 3)
			}
			orc := append([]float32(nil), dst...)
			oracle.AxpyF32(orc, alpha, src)
			exact := make([]float32, n)
			for i := range exact {
				exact[i] = dst[i] + float32(alpha*src[i])
			}
			al := alpha
			c.Call(&Args{Out: &dst[0], AScale: &src[0], Scr: (*byte)(unsafe.Pointer(&al)),
				K: int64(n / ElemLanes), Rows: int64(n % ElemLanes)})
			var se, sy2 float64
			for i := range dst {
				if math.Float32bits(dst[i]) != math.Float32bits(exact[i]) {
					t.Fatalf("alpha=%v n=%d: dst[%d] = %v, the unfused product-then-sum gives %v", alpha, n, i, dst[i], exact[i])
				}
				d := float64(dst[i]) - float64(orc[i])
				se, sy2 = se+d*d, sy2+float64(orc[i])*float64(orc[i])
			}
			if nmse := se / sy2; !(nmse <= 1e-12) {
				t.Errorf("alpha=%v n=%d: NMSE %.3e against oracle.AxpyF32", alpha, n, nmse)
			}
			sseElemGuardsHold(t, "axpy", full, n)
		}
	}
}

// TestEmitScaleSSEIsTheAVX2KernelBitForBit: scale is one multiply on both
// tiers in the same operand order, so the SSE kernel must reproduce the AVX2
// kernel's bits exactly -- NaN payloads, infinities, signed zeros and
// denormals included (Go leaves MXCSR at its default, no FTZ/DAZ, on both).
// On a host with no AVX2 it is held to Go's own float32 product instead,
// which is the same MULSS.
func TestEmitScaleSSEIsTheAVX2KernelBitForBit(t *testing.T) {
	c := sseElemMap(t, "scale", EmittersFor(TierSSE).Scale)
	defer c.Close()
	avx := sseElemAVX2(t, EmitScale())
	if avx != nil {
		defer avx.Close()
	}
	rng := rand.New(rand.NewSource(7))
	specials := []uint32{0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00001, 0xffa00002,
		0x00000001, 0x807fffff, 0x00400000, 0x7f7fffff, 0x3f800000}
	for _, alpha := range []float32{-0.625, 1, 0, float32(math.Inf(1)), 1e-30, 3.4e38,
		math.Float32frombits(0x00000003), math.Float32frombits(0x7fc0dead)} {
		for _, n := range []int{1, 7, 8, 9, 16, 1027} {
			in := make([]float32, n)
			for i := range in {
				if i < len(specials) {
					in[i] = math.Float32frombits(specials[i])
				} else {
					in[i] = math.Float32frombits(rng.Uint32())
				}
			}
			run := func(c *Code) []float32 {
				full, d := sseElemBuf(n)
				copy(d, in)
				al := alpha
				c.Call(&Args{Out: &d[0], Scr: (*byte)(unsafe.Pointer(&al)), K: int64(n / ElemLanes), Rows: int64(n % ElemLanes)})
				sseElemGuardsHold(t, "scale", full, n)
				return d
			}
			got := run(c)
			for i := range got {
				want := in[i] * alpha
				if math.Float32bits(got[i]) != math.Float32bits(want) {
					t.Fatalf("alpha=%#08x n=%d: %#08x * alpha = %#08x, Go's float32 product is %#08x",
						math.Float32bits(alpha), n, math.Float32bits(in[i]), math.Float32bits(got[i]), math.Float32bits(want))
				}
			}
			if avx == nil {
				continue
			}
			ref := run(avx)
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(ref[i]) {
					t.Fatalf("alpha=%#08x n=%d: element %d is %#08x on SSE and %#08x on AVX2",
						math.Float32bits(alpha), n, i, math.Float32bits(got[i]), math.Float32bits(ref[i]))
				}
			}
		}
	}
	if avx == nil {
		t.Log("no AVX2 on this host: held to Go's float32 product only")
	}
}

// TestEmitSoftcapSSEMatchesReference is TestEmitSoftcapMatchesReference for
// the SSE tier: gemma2's caps, over a range that saturates both ways.
func TestEmitSoftcapSSEMatchesReference(t *testing.T) {
	c := sseElemMap(t, "softcap", EmittersFor(TierSSE).Softcap)
	defer c.Close()
	consts := ActConsts()
	for _, cap := range []float32{30, 50} {
		kc := [2]float32{2 / cap, cap}
		for _, n := range []int{1, 5, 8, 13, 203} {
			full, x := sseElemBuf(n)
			want := make([]float64, n)
			sat := 0
			for i := range x {
				v := float32(math.Sin(float64(i)*0.37) * float64(i) * 20) // |x| to ~4000
				x[i] = v
				want[i] = float64(cap) * math.Tanh(float64(v)/float64(cap))
				if math.Abs(float64(v)) > 20*float64(cap) {
					sat++
				}
			}
			if n == 203 && sat == 0 {
				t.Fatalf("cap %v: no input saturates, so the clamps are untested", cap)
			}
			c.Call(&Args{Out: &x[0], W: (*byte)(unsafe.Pointer(&kc[0])),
				Scr: (*byte)(unsafe.Pointer(&consts[0])), K: int64(n / ElemLanes), Rows: int64(n % ElemLanes)})
			for i := range x {
				if d := math.Abs(float64(x[i]) - want[i]); !(d <= 2e-5*float64(cap)) {
					t.Fatalf("cap %v n=%d: x[%d] = %v, want %v", cap, n, i, x[i], want[i])
				}
			}
			sseElemGuardsHold(t, "softcap", full, n)
		}
	}
}

// sseSoftmax runs the SSE softmax over a fresh misaligned, guarded copy of in
// and returns the result.
func sseSoftmax(t *testing.T, c *Code, in []float32) []float32 {
	t.Helper()
	consts := ActConsts() // what nn passes; it begins with the exp block
	n := len(in)
	full, row := sseElemBuf(n)
	copy(row, in)
	c.Call(&Args{Out: &row[0], Scr: (*byte)(unsafe.Pointer(&consts[0])),
		K: int64(n / ElemLanes), Rows: int64(n % ElemLanes)})
	sseElemGuardsHold(t, "softmax", full, n)
	return row
}

// TestEmitSoftmaxSSEMatchesReference is TestEmitSoftmaxMatchesReference for
// the SSE tier, against internal/oracle.Softmax32 and a float64 three-pass
// reference, over the non-round lengths a decode produces -- every one of
// which exercises the tail, and every length past 4 the high half.
func TestEmitSoftmaxSSEMatchesReference(t *testing.T) {
	c := sseElemMap(t, "softmax", EmittersFor(TierSSE).Softmax)
	defer c.Close()
	for _, n := range []int{1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 31, 32, 33, 64, 127, 512, 1537} {
		in := make([]float32, n)
		want := make([]float64, n)
		for i := range in {
			v := math.Sin(float64(i)*0.9) * float64(3+i%23)
			in[i] = float32(v)
			want[i] = float64(in[i])
		}
		mx := want[0]
		for _, v := range want {
			mx = math.Max(mx, v)
		}
		var sum float64
		for i, v := range want {
			want[i] = math.Exp(v - mx)
			sum += want[i]
		}
		for i := range want {
			want[i] /= sum
		}
		orc := append([]float32(nil), in...)
		oracle.Softmax32(orc)

		row := sseSoftmax(t, c, in)
		var se, sy2, so, tot float64
		for i := 0; i < n; i++ {
			d := float64(row[i]) - want[i]
			se, sy2 = se+d*d, sy2+want[i]*want[i]
			do := float64(row[i]) - float64(orc[i])
			so += do * do
			tot += float64(row[i])
		}
		if nmse := se / sy2; !(nmse <= 1e-10) {
			t.Errorf("n=%d: NMSE %.3e against the float64 reference", n, nmse)
		}
		if nmse := so / sy2; !(nmse <= 1e-10) {
			t.Errorf("n=%d: NMSE %.3e against oracle.Softmax32", n, nmse)
		}
		if math.Abs(tot-1) > 1e-5 {
			t.Errorf("n=%d: probabilities sum to %.6f, not 1 -- a tail lane leaked into the sum", n, tot)
		}
	}
}

// TestEmitSoftmaxSSEMaxIsTheRealMax is TestEmitSoftmaxMaxIsTheRealMax for the
// SSE tier: two saturating peaks, 50 apart, in every pair of positions of a
// 16- and a 21-float row, so each peak lands in every lane of both halves and
// of the tail. A maximum that missed a half (or folded only one of its two
// accumulators) clamps both peaks and ties them.
func TestEmitSoftmaxSSEMaxIsTheRealMax(t *testing.T) {
	c := sseElemMap(t, "softmax", EmittersFor(TierSSE).Softmax)
	defer c.Close()
	for _, n := range []int{16, 21} {
		for hi := 0; hi < n; hi++ {
			lo := (hi + 5) % n
			in := make([]float32, n)
			for i := range in {
				in[i] = float32(i)
			}
			in[hi], in[lo] = 200, 150
			row := sseSoftmax(t, c, in)
			if row[hi] < 0.999 {
				t.Errorf("n=%d peaks at %d/%d: the larger holds %.6g of the mass, want ~1", n, hi, lo, row[hi])
			}
			if ratio := float64(row[lo]) / float64(row[hi]); ratio > 1e-20 {
				t.Errorf("n=%d peaks at %d/%d: ratio %.3g, want ~%.3g -- the max reduction missed one of those lanes",
					n, hi, lo, ratio, math.Exp(-50))
			}
		}
	}
}

// TestEmitSoftmaxSSEDeepNegativeRow is TestEmitSoftmaxDeepNegativeRow for the
// SSE tier: rows entirely below exp's low clamp (qwen2's -872), where a
// maximum seeded from a constant, or a tail lane that is not neutral, turns a
// one-element softmax into 1e-39.
func TestEmitSoftmaxSSEDeepNegativeRow(t *testing.T) {
	c := sseElemMap(t, "softmax", EmittersFor(TierSSE).Softmax)
	defer c.Close()
	for _, n := range []int{1, 2, 5, 9, 17} {
		for _, base := range []float64{-872.46, -1e5, -3.4e30} {
			in := make([]float32, n)
			want := make([]float64, n)
			for i := range in {
				in[i] = float32(base + float64(i)*0.5)
				want[i] = base + float64(i)*0.5
			}
			mx := want[n-1]
			var sum float64
			for i, v := range want {
				want[i] = math.Exp(v - mx)
				sum += want[i]
			}
			for i := range want {
				want[i] /= sum
			}
			row := sseSoftmax(t, c, in)
			var se, sy2 float64
			for i := 0; i < n; i++ {
				d := float64(row[i]) - want[i]
				se, sy2 = se+d*d, sy2+want[i]*want[i]
			}
			if nmse := se / sy2; !(nmse <= 1e-10) {
				t.Errorf("n=%d base=%g: NMSE %.3e -- got %v, want %v", n, base, nmse, row, want)
			}
		}
	}
}

// TestEveryElementwiseOpIsGeneratedSSE is TestEveryElementwiseOpIsGenerated
// for the SSE tier's family-1 rows of elementwiseInventory. It calls every
// kernel rather than only mapping it, since mapping says nothing about whether
// this CPU can execute the bytes: each runs once on a ragged width and must
// return finite values without touching its guards.
func TestEveryElementwiseOpIsGeneratedSSE(t *testing.T) {
	sse := EmittersFor(TierSSE)
	rows := []struct {
		name string
		emit func() ([]byte, error)
	}{
		{"softmax", sse.Softmax},
		{"sigmoidmul", sse.SigmoidMul},
		{"axpy", sse.Axpy},
		{"scale", sse.Scale},
		{"softcap", sse.Softcap},
		{"clamp", sse.Clamp},
	}
	for _, k := range Gated {
		rows = append(rows, struct {
			name string
			emit func() ([]byte, error)
		}{"actmul/" + k.String(), func() ([]byte, error) { return sse.ActMul(k) }})
	}
	for _, k := range Ungated {
		rows = append(rows, struct {
			name string
			emit func() ([]byte, error)
		}{"act/" + k.String(), func() ([]byte, error) { return sse.Act(k) }})
	}
	// The AVX2 inventory's family-1 rows, counted, so a row added there and
	// not here is a failure rather than an absence. The two delta gates are
	// called on SSE by their own gates (ssehybrid_test.go, k3decay_test.go).
	fam1 := 0
	for _, e := range elementwiseInventory {
		if !strings.HasPrefix(e.name, "rmsnorm") && !strings.HasPrefix(e.name, "layernorm") && e.name != "deltagate" &&
			e.name != "deltadecaybound" {
			fam1++
		}
	}
	if fam1 != len(rows) {
		t.Fatalf("elementwiseInventory has %d family-1 rows and this gate %d -- a row was added on one side only", fam1, len(rows))
	}
	consts := ActConsts()
	alpha := float32(0.5)
	kc := [2]float32{2.0 / 30, 30}
	clampLH := [2]float32{-2.5, 4}
	const n = 21
	for _, r := range rows {
		t.Run(r.name, func(t *testing.T) {
			c := sseElemMap(t, r.name, r.emit)
			defer c.Close()
			full, dst := sseElemBuf(n)
			sfull, src := sseElemBuf(n)
			sseElemInputs(dst, src)
			args := Args{Out: &dst[0], AScale: &src[0], Scr: (*byte)(unsafe.Pointer(&consts[0])),
				W: (*byte)(unsafe.Pointer(&kc[0])), K: n / ElemLanes, Rows: n % ElemLanes}
			if r.name == "axpy" || r.name == "scale" {
				args.Scr = (*byte)(unsafe.Pointer(&alpha))
			}
			if r.name == "clamp" {
				args.Scr = (*byte)(unsafe.Pointer(&clampLH[0]))
			}
			c.Call(&args)
			for i, v := range dst {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("element %d is %v", i, v)
				}
			}
			sseElemGuardsHold(t, r.name, full, n)
			sseElemGuardsHold(t, r.name+" (src)", sfull, n)
		})
	}
	t.Logf("%d SSE-tier elementwise ops generated, gated and executed on this host", len(rows))
}
