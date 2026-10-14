//go:build amd64

package cpu

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// Gates for the SSE tier's family 2 (sse_norm.go): RMSNorm, LayerNorm, RoPE,
// the packed embedding row and the widen.
//
// Every kernel is taken from EmittersFor(TierSSE), the table nn uses, so a
// miswired field fails here. Each passes the VEX-leak gate and is called
// directly (legacy SSE executes on an AVX2 host too).
//
// The oracle is internal/oracle; where the AVX2 kernel uses no FMA it is also
// a bit-for-bit oracle (LayerNorm, the packed row, the widen, NORM rotation's
// whole groups). RMSNorm and NEOX are fused on AVX2, so they are held to the
// float64 oracle only. Without AVX2 only the oracle half runs.

// f2SSEKernel emits one SSE-tier kernel through the table, checks it is
// declared SSE-tier and legacy-only, and maps it. A Map error is a failure,
// never a skip: an SSE kernel must map on any host that runs this suite.
func f2SSEKernel(t *testing.T, name string, code []byte, err error) *Code {
	t.Helper()
	if err != nil {
		t.Fatalf("%s: the SSE table refused: %v", name, err)
	}
	if KernelTier(code) != TierSSE {
		t.Fatalf("%s: the SSE table returned a kernel that is not declared SSE-tier", name)
	}
	requireSSEKernel(t, name, code)
	c, err := Map(code)
	if err != nil {
		t.Fatalf("%s: Map: %v", name, err)
	}
	return c
}

// f2AVX2Twin maps an AVX2-tier kernel for a bit-for-bit comparison, or returns
// nil on a host that cannot run it.
func f2AVX2Twin(t *testing.T, code []byte) *Code {
	t.Helper()
	c, err := Map(code)
	if errors.Is(err, ErrISA) {
		return nil
	}
	if err != nil {
		t.Fatalf("Map of the AVX2 twin: %v", err)
	}
	return c
}

// f2NMSE is the normalised squared error of got against want, and fails on a
// degenerate reference (NaN > bound is false, so a poisoned oracle would pass).
func f2NMSE(t *testing.T, what string, got []float32, want []float64) float64 {
	t.Helper()
	var sse, sy2 float64
	for i := range want {
		d := float64(got[i]) - want[i]
		sse, sy2 = sse+d*d, sy2+want[i]*want[i]
	}
	if sy2 == 0 || math.IsNaN(sy2) || math.IsInf(sy2, 0) {
		t.Fatalf("%s: the reference is degenerate (sum of squares %v), so this proves nothing", what, sy2)
	}
	return sse / sy2
}

// f2GuardIntact fails when a kernel wrote past n: sentinelRow's NaN payload.
func f2GuardIntact(t *testing.T, what string, row []float32, n int) {
	t.Helper()
	for i := n; i < len(row); i++ {
		if math.Float32bits(row[i]) != softmaxSentinel {
			t.Fatalf("%s: wrote element %d past the end", what, i)
		}
	}
}

// f2Widths is every width from 1 to 40 -- no vector, a vector and a tail
// of every residue, a whole 32-float block and its tails -- then the widths a
// model uses and the ragged ones near them.
func f2Widths() []int {
	var ns []int
	for n := 1; n <= 40; n++ {
		ns = append(ns, n)
	}
	return append(ns, 63, 64, 65, 100, 127, 128, 129, 576, 768, 771, 1152, 2047, 2048, 2304, 4096, 8192)
}

// TestRMSNormSSEMatchesTheOracle is TestRMSNormKernelMatchesReference for the
// SSE tier: the float64 oracle, 1e-10 NMSE, a guard past n, at every ragged
// width. Heavy-tailed input, since a uniform vector hides a dropped tail.
func TestRMSNormSSEMatchesTheOracle(t *testing.T) {
	em := EmittersFor(TierSSE)
	worst := 0.0
	for _, n := range f2Widths() {
		b, err := em.RMSNorm(n)
		c := f2SSEKernel(t, fmt.Sprintf("rmsnorm_sse_%d", n), b, err)
		x := make([]float32, n)
		w := make([]float32, n)
		for i := range x {
			x[i] = float32(math.Sin(float64(i)*0.7+0.3)) * float32(1+i%13)
			w[i] = 1 + float32(i%7)*0.03
		}
		const eps = 1e-5
		ref := make([]float32, n)
		oracle.RMSNorm32(ref, x, w, eps) // f64 accumulation, one rounding
		want := make([]float64, n)
		for i, v := range ref {
			want[i] = float64(v)
		}

		got := sentinelRow(n)
		kc := []float32{1 / float32(n), eps, 1}
		c.Call(&Args{Out: &got[0], A: (*int8)(unsafe.Pointer(&x[0])), AScale: &w[0],
			Scr: (*byte)(unsafe.Pointer(&kc[0])), K: int64(n)})
		c.Close()
		f2GuardIntact(t, fmt.Sprintf("rmsnorm n=%d", n), got, n)
		nmse := f2NMSE(t, fmt.Sprintf("rmsnorm n=%d", n), got, want)
		if !(nmse <= 1e-10) {
			t.Errorf("n=%d: NMSE %.3e against the float64 oracle", n, nmse)
		}
		worst = math.Max(worst, nmse)
	}
	t.Logf("%d widths, worst NMSE %.2e against the float64 oracle", len(f2Widths()), worst)
}

// TestLayerNormSSEIsTheAVX2KernelBitForBit holds the SSE LayerNorm to the
// float64 oracle and to the AVX2 kernel's exact bits, with and without a bias,
// on input with a large mean (where the one-pass identity cancels). The bit
// half catches a wrong fold order, which moves the mean by an ulp.
func TestLayerNormSSEIsTheAVX2KernelBitForBit(t *testing.T) {
	em := EmittersFor(TierSSE)
	identical, compared := 0, 0
	for _, n := range append(f2Widths(), 1024, 3072) {
		for _, bias := range []bool{false, true} {
			name := fmt.Sprintf("layernorm_sse_%d_bias=%v", n, bias)
			b, err := em.LayerNorm(n, bias)
			c := f2SSEKernel(t, name, b, err)
			x := make([]float32, n)
			w := make([]float32, n)
			bv := make([]float32, n)
			for i := range x {
				x[i] = float32(40 + math.Sin(float64(i)*0.37)*3)
				w[i] = float32(0.5 + float64(i%11)*0.07)
				bv[i] = float32(math.Cos(float64(i) * 0.11))
			}
			const eps = 1e-6
			var bp []float32
			if bias {
				bp = bv
			}
			ref := make([]float32, n)
			oracle.LayerNorm32(ref, x, w, bp, eps) // f64 throughout, one rounding
			want := make([]float64, n)
			for i, v := range ref {
				want[i] = float64(v)
			}

			kc := []float32{1 / float32(n), eps, 1}
			run := func(c *Code) []float32 {
				y := sentinelRow(n)
				args := Args{Out: &y[0], A: (*int8)(unsafe.Pointer(&x[0])), AScale: &w[0],
					Scr: (*byte)(unsafe.Pointer(&kc[0])), K: int64(n)}
				if bias {
					args.W = (*byte)(unsafe.Pointer(&bv[0]))
				}
				c.Call(&args)
				f2GuardIntact(t, name, y, n)
				return y
			}
			got := run(c)
			c.Close()
			zero := true // n=1 without a bias: x - mean is exactly 0
			for _, v := range want {
				zero = zero && v == 0
			}
			if zero {
				for i := 0; i < n; i++ {
					if got[i] != 0 {
						t.Errorf("%s: element %d is %v where the oracle is exactly 0", name, i, got[i])
					}
				}
			} else if nmse := f2NMSE(t, name, got, want); !(nmse <= 1e-10) {
				t.Errorf("%s: NMSE %.3e against the float64 oracle", name, nmse)
			}
			if tw := f2AVX2Twin(t, EmitLayerNorm(n, bias)); tw != nil {
				compared++
				twin := run(tw)
				tw.Close()
				same := true
				for i := 0; i < n; i++ {
					if math.Float32bits(got[i]) != math.Float32bits(twin[i]) {
						t.Errorf("%s: element %d is %#08x, the AVX2 kernel's is %#08x",
							name, i, math.Float32bits(got[i]), math.Float32bits(twin[i]))
						same = false
						break
					}
				}
				if same {
					identical++
				}
			}
		}
	}
	if compared == 0 {
		t.Logf("no AVX2 on this host: the SSE LayerNorm was held to the float64 oracle only")
	} else {
		t.Logf("%d of %d (width, bias) cases bit-identical to the AVX2 kernel", identical, compared)
	}
}

// TestRoPESSEMatchesTheOracle is TestEmitRoPEMatchesReference for the SSE
// tier: every pair count 1..33, a partial rotary tail, both layouts, three
// heads and a fourth that must come back untouched.
//
// The NORM layout must match the AVX2 kernel's bits over its whole groups
// (both compute two rounded products and one ADDSUB); the AVX2 kernel's last
// nrot/2 % 4 pairs are fused, so those and all of NEOX are held to the oracle
// only.
func TestRoPESSEMatchesTheOracle(t *testing.T) {
	em := EmittersFor(TierSSE)
	ran, bitPairs := 0, 0
	for _, neox := range []bool{false, true} {
		for pairs := 1; pairs <= 33; pairs++ {
			for _, extra := range []int{0, 6} {
				nrot := 2 * pairs
				hd := nrot + extra
				bitPairs += f2CheckRoPE(t, em, hd, nrot, neox, 3)
				ran++
			}
		}
	}
	// No heads is no work.
	b, err := em.RoPE(16, 16, false)
	c := f2SSEKernel(t, "rope_sse_zero_heads", b, err)
	x := ropeInput(16, 0)
	keep := append([]float32(nil), x...)
	cs := ropeTable(16)
	c.Call(&Args{Out: &x[0], AScale: &cs[0], Rows: 0})
	c.Close()
	for i := range x {
		if math.Float32bits(x[i]) != math.Float32bits(keep[i]) {
			t.Fatalf("Rows=0 wrote dimension %d", i)
		}
	}
	// The two layouts are different rotations.
	a1, a2 := f2RopeRun(t, em, 16, 16, false, 1), f2RopeRun(t, em, 16, 16, true, 1)
	same := true
	for i := range a1 {
		if a1[i] != a2[i] {
			same = false
		}
	}
	if same {
		t.Fatal("NORM and NEOX produced the same head: one layout was emitted for both")
	}
	t.Logf("%d (layout, pairs, partial) cases; %d NORM pairs bit-identical to the AVX2 kernel", ran, bitPairs)
}

func f2RopeRun(t *testing.T, em *Emitters, hd, nrot int, neox bool, heads int) []float32 {
	t.Helper()
	b, err := em.RoPE(hd, nrot, neox)
	c := f2SSEKernel(t, fmt.Sprintf("rope_sse_%d_%d_neox=%v", hd, nrot, neox), b, err)
	defer c.Close()
	x := ropeInput(hd, heads)
	cs := ropeTable(nrot)
	c.Call(&Args{Out: &x[0], AScale: &cs[0], Rows: int64(heads)})
	return x
}

// f2CheckRoPE checks one shape and returns how many pairs it held to the AVX2
// kernel's exact bits.
func f2CheckRoPE(t *testing.T, em *Emitters, hd, nrot int, neox bool, heads int) int {
	t.Helper()
	got := f2RopeRun(t, em, hd, nrot, neox, heads)
	in := ropeInput(hd, heads)
	cs := ropeTable(nrot)
	half := nrot / 2
	for h := 0; h <= heads; h++ {
		x := in[h*hd : (h+1)*hd]
		want := make([]float32, hd)
		copy(want, x)
		if h < heads {
			oracle.RopeApply(want[:nrot], cs, neox)
		}
		for i := 0; i < hd; i++ {
			g := got[h*hd+i]
			if h == heads || i >= nrot {
				if math.Float32bits(g) != math.Float32bits(x[i]) {
					t.Fatalf("hd=%d nrot=%d neox=%v head %d dim %d: %v should be untouched (%v)",
						hd, nrot, neox, h, i, g, x[i])
				}
				continue
			}
			// Two rounded products and a rounded sum, against the oracle's
			// f64 evaluation rounded once: a few ulps of the inputs' scale.
			if d := math.Abs(float64(g) - float64(want[i])); d > 2e-6*(1+math.Abs(float64(want[i]))) {
				t.Fatalf("hd=%d nrot=%d neox=%v head %d dim %d: %v, want %v",
					hd, nrot, neox, h, i, g, want[i])
			}
		}
	}
	if neox {
		return 0
	}
	tb, err := EmitRoPE(hd, nrot, neox)
	if err != nil {
		t.Fatal(err)
	}
	tw := f2AVX2Twin(t, tb)
	if tw == nil {
		return 0
	}
	defer tw.Close()
	twin := ropeInput(hd, heads)
	tw.Call(&Args{Out: &twin[0], AScale: &cs[0], Rows: int64(heads)})
	whole := 4 * (half / 4) // the AVX2 kernel's whole groups, in pairs
	for h := 0; h < heads; h++ {
		for i := 0; i < 2*whole; i++ {
			if math.Float32bits(got[h*hd+i]) != math.Float32bits(twin[h*hd+i]) {
				t.Fatalf("hd=%d nrot=%d NORM head %d dim %d: %#08x, the AVX2 kernel's is %#08x",
					hd, nrot, h, i, math.Float32bits(got[h*hd+i]), math.Float32bits(twin[h*hd+i]))
			}
		}
	}
	return heads * whole
}

// TestPackedRowSSEIsTheAVX2KernelBitForBit is TestPackedRowMatchesTheLayout
// for the SSE tier: every packed format, every row of an odd-row-count tensor
// (the narrow formats pair rows in their d plane), varying and subnormal
// scales, three super-blocks a row where the format has them -- against the
// oracle, and bit for bit against the AVX2 kernel.
//
// Subnormal scales are counted and must be nonzero per format: the software
// f16 convert is exact on them only while MXCSR.DAZ is off (sse_helpers.go).
// The scratch has a guard, since the kernel owns only PackedRowScratch bytes.
func TestPackedRowSSEIsTheAVX2KernelBitForBit(t *testing.T) {
	em := EmittersFor(TierSSE)
	ran, compared := 0, 0
	for _, gt := range quant.PackedTypes {
		q, ok := kernels.QuantOf(gt)
		if !ok {
			t.Fatalf("%s has no device layout", gt)
		}
		b, err := em.PackedRow(q)
		c := f2SSEKernel(t, "packed_row_sse_"+gt.String(), b, err)
		var tw *Code
		if tb, err := EmitPackedRow(q); err != nil {
			t.Fatalf("%s: the AVX2 emitter: %v", gt, err)
		} else {
			tw = f2AVX2Twin(t, tb)
		}
		sub, _, _, _ := kernels.Layout(q)
		perSuper, _ := kernels.ScaleLayout(q)
		superE := sub * perSuper
		k := 3 * superE
		if k < 96 {
			k = 96 // three 32-element super-blocks is too few to see a stride
		}
		if be := int(gt.BlockElems()); k%be != 0 {
			k = (k/be + 1) * be // TQ2_0's source block is 256 wide, its sub-block 32
		}
		const nrows = 13
		src := make([]byte, uint64(nrows*k)/gt.BlockElems()*gt.BlockBytes())
		rand.New(rand.NewSource(int64(gt) + 7)).Read(src)
		if !quant.PlantScales(gt, src, 0) {
			t.Fatalf("%s: no scale planter", gt)
		}
		qs32, d32, sc32, err := kernels.PackWeights(q, src, nrows, k)
		if err != nil {
			t.Fatal(err)
		}
		u8 := func(v []uint32) []byte {
			if len(v) == 0 {
				return nil
			}
			return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
		}
		qs, d, sc := u8(qs32), u8(d32), u8(sc32)
		subnormal := 0
		for i := 0; i+1 < len(d); i += 2 {
			if h := uint16(d[i]) | uint16(d[i+1])<<8; h&0x7C00 == 0 && h&0x03FF != 0 {
				subnormal++
			}
		}
		if subnormal == 0 && !kernels.D32(q) { // an f32 scale has no f16 to be subnormal
			t.Fatalf("%s: no subnormal f16 scale in the d plane -- the path this gate exists for was not exercised", gt)
		}
		konst := PackedRowConsts(q)
		for r := 0; r < nrows; r++ {
			want := make([]float32, k)
			if err := oracle.RowPacked(want, q, qs, d, sc, r, nrows, k); err != nil {
				t.Fatal(err)
			}
			di, slot := kernels.DIndex(q, 0, nrows, r)
			pd := di*4 + slot*DRowBytes(gt)
			run := func(c *Code) []float32 {
				got := make([]float32, k+8)
				for i := k; i < len(got); i++ {
					got[i] = -3
				}
				scratch := make([]byte, PackedRowScratch+32)
				for i := PackedRowScratch; i < len(scratch); i++ {
					scratch[i] = 0xA5
				}
				args := Args{
					Out: &got[0], W: &qs[r*4], RowStr: int64(nrows * 4),
					PD: &d[pd], DStr: int64(DSuperBytes(gt, nrows)),
					K:   int64(k / superE),
					Scr: &konst[0], Scratch: &scratch[0],
				}
				if len(sc) > 0 {
					args.PSC = &sc[r*4]
				}
				c.Call(&args)
				for i := k; i < len(got); i++ {
					if got[i] != -3 {
						t.Fatalf("%s row %d: wrote element %d past k", gt, r, i)
					}
				}
				for i := PackedRowScratch; i < len(scratch); i++ {
					if scratch[i] != 0xA5 {
						t.Fatalf("%s row %d: wrote scratch byte %d, past the %d it owns", gt, r, i, PackedRowScratch)
					}
				}
				return got
			}
			got := run(c)
			for i := 0; i < k; i++ {
				if dd := math.Abs(float64(got[i] - want[i])); !(dd <= 1e-6*(1+math.Abs(float64(want[i])))) {
					t.Fatalf("%s row %d element %d: %v, want %v", gt, r, i, got[i], want[i])
				}
			}
			if tw != nil {
				twin := run(tw)
				for i := 0; i < k; i++ {
					if math.Float32bits(got[i]) != math.Float32bits(twin[i]) {
						t.Fatalf("%s row %d element %d: %#08x, the AVX2 kernel's is %#08x",
							gt, r, i, math.Float32bits(got[i]), math.Float32bits(twin[i]))
					}
				}
				compared++
			}
			ran++
		}
		c.Close()
		if tw != nil {
			tw.Close()
		}
	}
	t.Logf("%d rows over %d formats equal to the layout's own reading; %d also bit-identical to the AVX2 kernel",
		ran, len(quant.PackedTypes), compared)
}

// TestWidenSSEIsTheAVX2KernelBitForBit converts every binary16 and bfloat16
// bit pattern (a software f16 convert has four regimes: zero, subnormal,
// normal, Inf/NaN), then the lengths that exercise the vector loop, the
// single-element loop, both and neither, each with a guard after dst. Every
// non-NaN half must equal quant.DecodeHalf exactly; a NaN keeps its payload
// (the software convert does not quiet), differing from F16C only in the
// quiet bit.
func TestWidenSSEIsTheAVX2KernelBitForBit(t *testing.T) {
	em := EmittersFor(TierSSE)
	for _, bf := range []bool{false, true} {
		name := map[bool]string{false: "widen_sse_f16", true: "widen_sse_bf16"}[bf]
		b, err := em.Widen(bf)
		c := f2SSEKernel(t, name, b, err)
		tw := f2AVX2Twin(t, EmitWiden(bf))
		call := func(c *Code, src []uint16) []float32 {
			n := len(src)
			got := make([]float32, n+ElemLanes)
			for i := n; i < len(got); i++ {
				got[i] = -9
			}
			var sp *byte
			if n > 0 {
				sp = (*byte)(unsafe.Pointer(&src[0]))
			}
			c.Call(&Args{Out: &got[0], W: sp, K: int64(n / ElemLanes), Rows: int64(n % ElemLanes)})
			for i := n; i < len(got); i++ {
				if got[i] != -9 {
					t.Fatalf("%s n=%d: wrote element %d past the end", name, n, i)
				}
			}
			return got[:n]
		}
		want := func(h uint16) uint32 {
			if bf {
				return uint32(h) << 16
			}
			if h&0x7C00 == 0x7C00 && h&0x03FF != 0 { // NaN: payload kept
				return uint32(h&0x8000)<<16 | 0x7F800000 | uint32(h&0x03FF)<<13
			}
			return math.Float32bits(float32(quant.DecodeHalf(h)))
		}

		// Every bit pattern, offset by 3 so the last three take the single-
		// element loop.
		all := make([]uint16, 65536+3)
		for i := range all {
			all[i] = uint16(i)
		}
		got := call(c, all)
		regime := map[string]int{}
		for i, h := range all {
			if g := math.Float32bits(got[i]); g != want(h) {
				t.Fatalf("%s: %#04x widened to %#08x, want %#08x", name, h, g, want(h))
			}
			if !bf {
				switch e, m := h&0x7C00, h&0x03FF; {
				case e == 0 && m == 0:
					regime["zero"]++
				case e == 0:
					regime["subnormal"]++
				case e == 0x7C00 && m == 0:
					regime["inf"]++
				case e == 0x7C00:
					regime["nan"]++
				default:
					regime["normal"]++
				}
			}
		}
		if tw != nil {
			twin := call(tw, all)
			for i, h := range all {
				g, w := math.Float32bits(got[i]), math.Float32bits(twin[i])
				if !bf && h&0x7C00 == 0x7C00 && h&0x03FF != 0 {
					g, w = g|0x00400000, w|0x00400000 // F16C quiets a signalling NaN
				}
				if g != w {
					t.Fatalf("%s: %#04x is %#08x here and %#08x from the AVX2 kernel", name, h, g, w)
				}
			}
		}
		for _, n := range []int{0, 1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 24, 26, 40, 1023} {
			src := make([]uint16, n)
			rng := rand.New(rand.NewSource(int64(n)))
			for i := range src {
				v := float32(rng.NormFloat64() * 10)
				if bf {
					src[i] = uint16(math.Float32bits(v) >> 16)
				} else {
					src[i] = quant.EncodeHalf(v)
				}
			}
			got := call(c, src)
			for i, h := range src {
				if g := math.Float32bits(got[i]); g != want(h) {
					t.Fatalf("%s n=%d element %d: %#08x, want %#08x", name, n, i, g, want(h))
				}
			}
		}
		c.Close()
		if tw != nil {
			tw.Close()
		}
		if !bf {
			t.Logf("f16: %v, every one exact", regime)
		}
	}
}

// TestSSEFamily2TableServesItsEmitters is the SSE half of
// TestPrimaryTableIsTodaysEmitters for this family: each table field returns
// exactly what the emitter it names returns.
func TestSSEFamily2TableServesItsEmitters(t *testing.T) {
	em := EmittersFor(TierSSE)
	check := func(name string, got []byte, gerr error, want []byte, werr error) {
		t.Helper()
		if gerr != nil || werr != nil {
			t.Fatalf("%s: table error %v, direct error %v", name, gerr, werr)
		}
		if !bytes.Equal(got, want) {
			t.Errorf("%s: the SSE table emits %d bytes that differ from the direct call's %d", name, len(got), len(want))
		}
	}
	for _, n := range []int{1, 7, 64, 771} {
		g, ge := em.RMSNorm(n)
		w, we := EmitRMSNormSSE(n)
		check(fmt.Sprintf("rmsnorm/%d", n), g, ge, w, we)
		for _, b := range []bool{false, true} {
			g, ge := em.LayerNorm(n, b)
			w, we := EmitLayerNormSSE(n, b)
			check(fmt.Sprintf("layernorm/%d/%v", n, b), g, ge, w, we)
		}
	}
	for _, neox := range []bool{false, true} {
		g, ge := em.RoPE(80, 32, neox)
		w, we := EmitRoPESSE(80, 32, neox)
		check(fmt.Sprintf("rope/%v", neox), g, ge, w, we)
	}
	for _, gt := range quant.PackedTypes {
		q, _ := kernels.QuantOf(gt)
		g, ge := em.PackedRow(q)
		w, we := EmitPackedRowSSE(q)
		check("packed_row/"+gt.String(), g, ge, w, we)
	}
	for _, b := range []bool{false, true} {
		g, ge := em.Widen(b)
		w, we := EmitWidenSSE(b)
		check(fmt.Sprintf("widen/%v", b), g, ge, w, we)
	}
	// The two layouts, and the two widths, must not be one kernel.
	a1, _ := em.RoPE(80, 32, false)
	a2, _ := em.RoPE(80, 32, true)
	w1, _ := em.Widen(false)
	w2, _ := em.Widen(true)
	if bytes.Equal(a1, a2) || bytes.Equal(w1, w2) {
		t.Error("a baked flag did not reach the SSE emitter: two configurations emitted one kernel")
	}
}

// TestSSEFamily2RefusesWhatTheAVX2TwinRefuses: the shape checks are the
// contract, not a detail of one tier.
func TestSSEFamily2RefusesWhatTheAVX2TwinRefuses(t *testing.T) {
	for _, r := range []struct{ hd, nrot int }{{64, 0}, {64, 3}, {64, 66}, {8, -2}} {
		_, e1 := EmitRoPE(r.hd, r.nrot, false)
		_, e2 := EmitRoPESSE(r.hd, r.nrot, false)
		if e1 == nil || e2 == nil {
			t.Errorf("rope hd=%d nrot=%d: AVX2 error %v, SSE error %v -- both must refuse", r.hd, r.nrot, e1, e2)
		}
	}
	if _, err := EmitRMSNormSSE(0); err == nil {
		t.Error("rmsnorm n=0 was emitted")
	}
	if _, err := EmitLayerNormSSE(0, true); err == nil {
		t.Error("layernorm n=0 was emitted")
	}
}
