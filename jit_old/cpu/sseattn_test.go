//go:build amd64

package cpu

import (
	"crypto/sha256"
	"math"
	"math/rand"
	"sync"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
)

// The SSE tier's attention kernels (sse_attn.go), each gated three ways:
//
//   - bit for bit against a Go model of the kernel's own arithmetic
//     (attnScoreModelSSE, attnAxpyModelSSE: same products, chains and
//     reduction tree, every step rounded to float32 so Go cannot fuse it);
//   - against internal/oracle (Dot32, AxpyF32) and a float64 evaluation, to
//     1e-12 NMSE;
//   - through the VEX-leak gate, at every shape and both KV widths.
//
// Every out-of-bounds element (KV padding between hd and kvStride, past the
// last row, the query past hd, weights past npos) is a NaN guard, and every
// output is followed by a sentinel. They execute directly; legacy SSE runs on
// an AVX2 host.

const attnHalfNaN = 0x7E00

// attnSSEData is one attention problem: a KV cache (as float32 and as the
// same values in binary16), two queries and two weight rows, guarded.
type attnSSEData struct {
	hd, stride, npos int
	kv32             []float32
	kv16             []uint16
	q, q2            []float32
	w, w2            []float32
}

const attnGuard = 8

// newAttnSSEData fills a problem. With rawHalves the cache values are
// uniformly random binary16 bit patterns (every exponent, subnormals
// included, no Inf/NaN) -- the widening's whole domain; otherwise they are
// N(0,1) rounded to the nearest half, so both caches still hold the same
// numbers and the sums are well conditioned for an NMSE.
func newAttnSSEData(hd, stride, npos int, seed int64, rawHalves bool) *attnSSEData {
	rng := rand.New(rand.NewSource(seed))
	d := &attnSSEData{hd: hd, stride: stride, npos: npos}
	n := npos*stride + attnGuard
	d.kv32 = make([]float32, n)
	d.kv16 = make([]uint16, n)
	for i := range d.kv32 {
		if i < npos*stride && i%stride < hd {
			var h uint16
			if rawHalves {
				h = uint16(rng.Intn(1 << 16))
				if (h>>10)&0x1F == 0x1F {
					h &^= 1 << 14
				}
			} else {
				h = quant.EncodeHalf(float32(rng.NormFloat64()))
			}
			d.kv16[i], d.kv32[i] = h, float32(quant.DecodeHalf(h))
		} else {
			d.kv16[i], d.kv32[i] = attnHalfNaN, float32(math.NaN())
		}
	}
	vec := func(n, valid int, f func() float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(math.NaN())
			if i < valid {
				v[i] = float32(f())
			}
		}
		return v
	}
	d.q = vec(hd+attnGuard, hd, rng.NormFloat64)
	d.q2 = vec(hd+attnGuard, hd, rng.NormFloat64)
	d.w = vec(npos+attnGuard, npos, rng.Float64)
	d.w2 = vec(npos+attnGuard, npos, rng.Float64)
	return d
}

// cache returns the cache's base pointer at position pos.
func (d *attnSSEData) cache(f16 bool, pos int) *byte {
	if f16 {
		return (*byte)(unsafe.Pointer(&d.kv16[pos*d.stride]))
	}
	return (*byte)(unsafe.Pointer(&d.kv32[pos*d.stride]))
}

// row is position t's hd real elements, as float32.
func (d *attnSSEData) row(t int) []float32 { return d.kv32[t*d.stride : t*d.stride+d.hd] }

// attnScoreModelSSE is the SSE score kernel's arithmetic in Go: chains
// accumulators of four lanes, vector i into chain i%chains, the tail into chain
// 0 lane 0, then ((c0+c1)+(c2+c3)) lane-wise and (l0+l1)+(l2+l3) -- the two
// HADDPS folds.
func attnScoreModelSSE(q, k []float32, hd, chains int) float32 {
	var c [4][4]float32
	nv := hd / 4
	for i := 0; i < nv; i++ {
		for l := 0; l < 4; l++ {
			c[i%chains][l] = c[i%chains][l] + float32(q[4*i+l]*k[4*i+l])
		}
	}
	for d := nv * 4; d < hd; d++ {
		c[0][0] = c[0][0] + float32(q[d]*k[d])
	}
	x := c[0]
	if chains == 4 {
		for l := range x {
			x[l] = (c[0][l] + c[1][l]) + (c[2][l] + c[3][l])
		}
	}
	return (x[0] + x[1]) + (x[2] + x[3])
}

// attnAxpyModelSSE is one position of the SSE accumulate: out = out + (V * w),
// the product rounded before the add (MULPS then ADDPS; no FMA on this tier).
func attnAxpyModelSSE(out []float32, w float32, v []float32) {
	for i := range out {
		out[i] = out[i] + float32(v[i]*w)
	}
}

var (
	attnGateMu   sync.Mutex
	attnGateSeen = map[[32]byte]bool{}
)

// sseAttnKernel emits, runs the VEX-leak gate on (once per distinct byte
// string), and maps an SSE kernel. It fails rather than skips when Map
// refuses: an SSE-tier kernel must run on every host at or above the floor.
func sseAttnKernel(t *testing.T, name string, b []byte, err error) *Code {
	t.Helper()
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if KernelTier(b) != TierSSE {
		t.Fatalf("%s: the kernel is not declared SSE-tier", name)
	}
	h := sha256.Sum256(b)
	attnGateMu.Lock()
	seen := attnGateSeen[h]
	attnGateSeen[h] = true
	attnGateMu.Unlock()
	if !seen {
		requireSSEKernel(t, name, b)
	}
	c, err := Map(b)
	if err != nil {
		t.Fatalf("%s: Map: %v", name, err)
	}
	t.Cleanup(func() { c.Close() })
	return c
}

// sseAttnGuarded reports whether row[n:] still holds sentinelRow's guard.
func sseAttnGuarded(row []float32, n int) bool {
	for _, v := range row[n:] {
		if math.Float32bits(v) != softmaxSentinel {
			return false
		}
	}
	return true
}

func sseAttnNMSE(got []float32, want []float64) float64 {
	var e, s float64
	for i, w := range want {
		d := float64(got[i]) - w
		e += d * d
		s += w * w
	}
	if s == 0 {
		return e
	}
	return e / s
}

func sseAttnSame(a, b float32) bool { return math.Float32bits(a) == math.Float32bits(b) }

// attnSSEShapes are TestAttnKernels' shapes plus every hd%4 residue at small
// and at real widths.
var attnSSEShapes = []struct{ hd, stride, npos int }{
	{64, 256, 1}, {64, 256, 7}, {64, 256, 821}, // tinyllama
	{256, 256, 1}, {256, 256, 33}, {256, 256, 500}, // gemma
	{128, 512, 64}, {80, 80, 9}, {8, 8, 3},
	{1, 1, 5}, {2, 3, 4}, {3, 5, 9}, {4, 4, 1}, {5, 9, 2}, {6, 6, 3}, {7, 7, 11},
	{12, 20, 7}, {17, 17, 33}, {33, 40, 100}, {63, 64, 5}, {66, 70, 3},
}

// TestAttnKernelsSSE is TestAttnKernels for the SSE tier, at both KV widths:
// scores, the accumulate and the accumulating twin, against the model bit for
// bit and against the oracle and float64 to 1e-12.
func TestAttnKernelsSSE(t *testing.T) {
	for _, s := range attnSSEShapes {
		for _, f16 := range []bool{false, true} {
			d := newAttnSSEData(s.hd, s.stride, s.npos, int64(s.hd*7919+s.npos), false)
			tag := func(k string) string {
				return k + "_hd" + itoa(s.hd) + "_s" + itoa(s.stride) + map[bool]string{true: "_f16"}[f16]
			}
			args := func(out []float32) Args {
				return Args{Out: &out[0], W: d.cache(f16, 0), Rows: int64(s.npos), Q32: &d.q[0], AScale: &d.w[0]}
			}

			// scores
			b, err := EmitAttnScoresSSE(s.hd, s.stride, KVOf(f16))
			sc := sseAttnKernel(t, tag("scores"), b, err)
			got := sentinelRow(s.npos)
			a := args(got)
			sc.Call(&a)
			if !sseAttnGuarded(got, s.npos) {
				t.Fatalf("%s npos=%d: wrote past the score row", tag("scores"), s.npos)
			}
			wantF, wantO := make([]float64, s.npos), make([]float64, s.npos)
			for p := 0; p < s.npos; p++ {
				if m := attnScoreModelSSE(d.q, d.row(p), s.hd, 4); !sseAttnSame(got[p], m) {
					t.Fatalf("%s npos=%d: score %d is %v (%#x), the kernel's own arithmetic gives %v (%#x)",
						tag("scores"), s.npos, p, got[p], math.Float32bits(got[p]), m, math.Float32bits(m))
				}
				wantO[p] = float64(oracle.Dot32(d.q[:s.hd], d.row(p)))
				for i := 0; i < s.hd; i++ {
					wantF[p] += float64(d.q[i]) * float64(d.row(p)[i])
				}
			}
			if e := sseAttnNMSE(got, wantO); e > 1e-12 || math.IsNaN(e) {
				t.Errorf("%s npos=%d: NMSE %.3e against oracle.Dot32", tag("scores"), s.npos, e)
			}
			if e := sseAttnNMSE(got, wantF); e > 1e-12 || math.IsNaN(e) {
				t.Errorf("%s npos=%d: NMSE %.3e against float64", tag("scores"), s.npos, e)
			}

			// the accumulate, from zero and into a seeded output
			for _, into := range []bool{false, true} {
				emit, name := EmitAttnAccSSE, "acc"
				if into {
					emit, name = EmitAttnAccIntoSSE, "acc_into"
				}
				b, err := emit(s.hd, s.stride, KVOf(f16))
				ac := sseAttnKernel(t, tag(name), b, err)
				out := sentinelRow(s.hd)
				model, orc := make([]float32, s.hd), make([]float32, s.hd)
				wantF := make([]float64, s.hd)
				if into {
					for i := range model {
						out[i] = float32(math.Sin(float64(i) * 0.7))
						model[i], orc[i], wantF[i] = out[i], out[i], float64(out[i])
					}
				}
				a := args(out)
				ac.Call(&a)
				if !sseAttnGuarded(out, s.hd) {
					t.Fatalf("%s npos=%d: wrote past the head", tag(name), s.npos)
				}
				for p := 0; p < s.npos; p++ {
					attnAxpyModelSSE(model, d.w[p], d.row(p))
					oracle.AxpyF32(orc, d.w[p], d.row(p))
					for i := range wantF {
						wantF[i] += float64(d.w[p]) * float64(d.row(p)[i])
					}
				}
				for i := range model {
					if !sseAttnSame(out[i], model[i]) {
						t.Fatalf("%s npos=%d: element %d is %v, the kernel's own arithmetic gives %v",
							tag(name), s.npos, i, out[i], model[i])
					}
				}
				wantO := make([]float64, s.hd)
				for i, v := range orc {
					wantO[i] = float64(v)
				}
				if e := sseAttnNMSE(out, wantO); e > 1e-12 || math.IsNaN(e) {
					t.Errorf("%s npos=%d: NMSE %.3e against oracle.AxpyF32", tag(name), s.npos, e)
				}
				if e := sseAttnNMSE(out, wantF); e > 1e-12 || math.IsNaN(e) {
					t.Errorf("%s npos=%d: NMSE %.3e against float64", tag(name), s.npos, e)
				}
			}
		}
	}
}

// TestAttnEmptyWindowSSE: Rows == 0 runs no position. The AVX2 kernels loop
// on a wrapped counter there and nn never asks; the SSE ones guard it, so an
// empty window is an empty sum -- scores untouched, Acc zero, AccInto its seed.
func TestAttnEmptyWindowSSE(t *testing.T) {
	d := newAttnSSEData(17, 17, 1, 1, false)
	for _, f16 := range []bool{false, true} {
		for _, k := range []struct {
			name string
			emit func(int, int, KVFmt) ([]byte, error)
			want func(i int, seed float32) float32
		}{
			{"scores", EmitAttnScoresSSE, func(_ int, seed float32) float32 { return seed }},
			{"scores2", EmitAttnScores2SSE, func(_ int, seed float32) float32 { return seed }},
			{"acc", EmitAttnAccSSE, func(int, float32) float32 { return 0 }},
			{"acc2", EmitAttnAcc2SSE, func(int, float32) float32 { return 0 }},
			{"acc_into", EmitAttnAccIntoSSE, func(_ int, seed float32) float32 { return seed }},
			{"acc2_into", EmitAttnAcc2IntoSSE, func(_ int, seed float32) float32 { return seed }},
		} {
			b, err := k.emit(17, 17, KVOf(f16))
			c := sseAttnKernel(t, "empty_"+k.name, b, err)
			o1, o2 := sentinelRow(17), sentinelRow(17)
			for i := 0; i < 17; i++ {
				o1[i], o2[i] = float32(i)+0.5, float32(i)-0.5
			}
			a := Args{Out: &o1[0], Out2: &o2[0], W: d.cache(f16, 0), Rows: 0,
				Q32: &d.q[0], Q2: &d.q2[0], AScale: &d.w[0], AScale2: &d.w2[0]}
			c.Call(&a)
			paired := k.name == "scores2" || k.name == "acc2" || k.name == "acc2_into"
			for i := 0; i < 17; i++ {
				if w := k.want(i, float32(i)+0.5); !sseAttnSame(o1[i], w) {
					t.Fatalf("%s f16=%v: Rows=0 left element %d at %v, want %v", k.name, f16, i, o1[i], w)
				}
				w2 := float32(i) - 0.5 // a single-head kernel must not touch Out2 at all
				if paired {
					w2 = k.want(i, w2)
				}
				if !sseAttnSame(o2[i], w2) {
					t.Fatalf("%s f16=%v: Rows=0 left Out2 element %d at %v, want %v", k.name, f16, i, o2[i], w2)
				}
			}
			if !sseAttnGuarded(o1, 17) || !sseAttnGuarded(o2, 17) {
				t.Fatalf("%s f16=%v: Rows=0 wrote past the row", k.name, KVOf(f16))
			}
		}
	}
}

// TestAttnPairedIsBitIdenticalSSE is TestAttnPairedIsBitIdentical for the SSE
// tier: each paired kernel is two calls of its single-head twin, bit for bit,
// at both KV widths: the paired kernel shares a load and changes no
// arithmetic.
func TestAttnPairedIsBitIdenticalSSE(t *testing.T) {
	for _, s := range []struct{ hd, stride, npos int }{
		{64, 256, 1}, {64, 256, 2}, {64, 256, 7},
		{64, 256, 512}, {64, 256, 1537},
		{256, 256, 33}, {128, 512, 64}, {80, 80, 5}, {8, 8, 3},
		{1, 1, 3}, {2, 2, 4}, {3, 5, 9}, {12, 20, 7}, {17, 17, 33}, {27, 30, 6},
	} {
		for _, f16 := range []bool{false, true} {
			d := newAttnSSEData(s.hd, s.stride, s.npos, int64(s.hd*31+s.npos), false)
			tag := func(k string) string {
				return k + "_hd" + itoa(s.hd) + "_s" + itoa(s.stride) + map[bool]string{true: "_f16"}[f16]
			}
			emit := func(name string, f func(int, int, KVFmt) ([]byte, error)) *Code {
				b, err := f(s.hd, s.stride, KVOf(f16))
				return sseAttnKernel(t, tag(name), b, err)
			}
			base := Args{W: d.cache(f16, 0), Rows: int64(s.npos)}

			// scores: two single calls, then one paired call
			single, paired := emit("scores", EmitAttnScoresSSE), emit("scores2", EmitAttnScores2SSE)
			w0, w1 := sentinelRow(s.npos), sentinelRow(s.npos)
			a := base
			a.Out, a.Q32 = &w0[0], &d.q[0]
			single.Call(&a)
			a.Out, a.Q32 = &w1[0], &d.q2[0]
			single.Call(&a)
			g0, g1 := sentinelRow(s.npos), sentinelRow(s.npos)
			a = base
			a.Out, a.Q32, a.Out2, a.Q2 = &g0[0], &d.q[0], &g1[0], &d.q2[0]
			paired.Call(&a)
			sseAttnPairBits(t, tag("scores2")+" head0", s.npos, w0, g0)
			sseAttnPairBits(t, tag("scores2")+" head1", s.npos, w1, g1)

			// the accumulate and its Into twin
			for _, into := range []bool{false, true} {
				sName, pName := "acc", "acc2"
				sf, pf := EmitAttnAccSSE, EmitAttnAcc2SSE
				if into {
					sName, pName = "acc_into", "acc2_into"
					sf, pf = EmitAttnAccIntoSSE, EmitAttnAcc2IntoSSE
				}
				single, paired := emit(sName, sf), emit(pName, pf)
				seed := func() []float32 {
					r := sentinelRow(s.hd)
					if into {
						for i := 0; i < s.hd; i++ {
							r[i] = float32(math.Cos(float64(i) * 1.3))
						}
					}
					return r
				}
				w0, w1, g0, g1 := seed(), seed(), seed(), seed()
				a := base
				a.Out, a.AScale = &w0[0], &d.w[0]
				single.Call(&a)
				a.Out, a.AScale = &w1[0], &d.w2[0]
				single.Call(&a)
				a = base
				a.Out, a.AScale, a.Out2, a.AScale2 = &g0[0], &d.w[0], &g1[0], &d.w2[0]
				paired.Call(&a)
				sseAttnPairBits(t, tag(pName)+" head0", s.hd, w0, g0)
				sseAttnPairBits(t, tag(pName)+" head1", s.hd, w1, g1)
			}
		}
	}
}

// pairBits requires got == want bit for bit over n elements and the guard
// after them intact.
func sseAttnPairBits(t *testing.T, what string, n int, want, got []float32) {
	t.Helper()
	if !sseAttnGuarded(got, n) {
		t.Fatalf("%s: the paired kernel wrote past element %d", what, n)
	}
	for i := 0; i < n; i++ {
		if !sseAttnSame(want[i], got[i]) {
			t.Fatalf("%s: element %d is %v, two single-head calls give %v -- the paired "+
				"kernel must be bit-identical, not close", what, i, got[i], want[i])
		}
	}
}

// TestAttnIntoSplitsBitIdenticallySSE is nn's TestAttnAccIntoSplitsBitIdentically
// and its paired twin at the kernel, for the SSE tier and both KV widths: a
// window accumulated in pieces (Acc, then AccInto, as a page walk calls them)
// gives exactly the bits of one call over the whole window.
func TestAttnIntoSplitsBitIdenticallySSE(t *testing.T) {
	splits := 0
	for _, hd := range []int{64, 128, 256, 12, 17, 3} {
		for _, npos := range []int{16, 129, 512} {
			for _, f16 := range []bool{false, true} {
				d := newAttnSSEData(hd, hd, npos, int64(hd*1000+npos), false)
				tag := "hd" + itoa(hd) + map[bool]string{true: "_f16"}[f16]
				k := func(name string, f func(int, int, KVFmt) ([]byte, error)) *Code {
					b, err := f(hd, hd, KVOf(f16))
					return sseAttnKernel(t, name+"_"+tag, b, err)
				}
				acc, into := k("acc", EmitAttnAccSSE), k("acc_into", EmitAttnAccIntoSSE)
				acc2, into2 := k("acc2", EmitAttnAcc2SSE), k("acc2_into", EmitAttnAcc2IntoSSE)
				run := func(c *Code, o0, o1 []float32, lo, n int) {
					a := Args{Out: &o0[0], Out2: &o1[0], W: d.cache(f16, lo), Rows: int64(n),
						AScale: &d.w[lo], AScale2: &d.w2[lo]}
					c.Call(&a)
				}
				whole, whole2 := make([]float32, hd), make([]float32, hd)
				run(acc2, whole, whole2, 0, npos)
				for _, P := range []int{1, 8, 16, 64, 100} {
					if P >= npos {
						continue
					}
					single := make([]float32, hd)
					p0, p1 := make([]float32, hd), make([]float32, hd)
					calls := 0
					for lo := 0; lo < npos; lo += P {
						n := min(P, npos-lo)
						if lo == 0 {
							run(acc, single, single, lo, n)
							run(acc2, p0, p1, lo, n)
						} else {
							run(into, single, single, lo, n)
							run(into2, p0, p1, lo, n)
						}
						calls++
					}
					if calls < 2 {
						t.Fatalf("%s npos=%d P=%d: %d call(s) -- the gate never split", tag, npos, P, calls)
					}
					for i := range whole {
						if !sseAttnSame(single[i], whole[i]) || !sseAttnSame(p0[i], whole[i]) || !sseAttnSame(p1[i], whole2[i]) {
							t.Fatalf("%s npos=%d P=%d: element %d is %v (single) and (%v, %v) (paired) split, "+
								"(%v, %v) whole -- the page walk does not reproduce the contiguous call",
								tag, npos, P, i, single[i], p0[i], p1[i], whole[i], whole2[i])
						}
					}
					splits++
				}
			}
		}
	}
	t.Logf("%d split walks reproduced the contiguous call exactly", splits)
}

// TestAttnF16MatchesF32OnExactHalvesSSE is TestAttnF16MatchesF32OnExactHalves
// for the SSE tier, over all seven kernels, on caches whose every value is a
// uniformly random binary16 -- subnormals and the largest normals included --
// so f32 and f16 hold the same numbers and only the software widening and the
// f16 addressing can differ. They must not: the widening is exact, and the
// arithmetic after it is the f32 kernel's.
func TestAttnF16MatchesF32OnExactHalvesSSE(t *testing.T) {
	type kern struct {
		name string
		emit func(hd, stride int, fm KVFmt) ([]byte, error)
		acc  bool
	}
	kerns := []kern{
		{"scores", EmitAttnScoresSSE, false}, {"scores2", EmitAttnScores2SSE, false},
		{"acc", EmitAttnAccSSE, true}, {"acc_into", EmitAttnAccIntoSSE, true},
		{"acc2", EmitAttnAcc2SSE, true}, {"acc2_into", EmitAttnAcc2IntoSSE, true},
		{"scores_t1", func(hd, stride int, fm KVFmt) ([]byte, error) {
			return EmitAttnScoresTiledSSE(hd, stride, 0, 0, 1, fm)
		}, false},
	}
	for _, hd := range []int{1, 2, 3, 4, 5, 8, 12, 17, 64, 128, 256} {
		for _, npos := range []int{1, 2, 7, 33} {
			stride := hd*3 + 1
			d := newAttnSSEData(hd, stride, npos, int64(hd*9176+npos), true)
			for _, k := range kerns {
				run := func(f16 bool) ([]float32, []float32) {
					b, err := k.emit(hd, stride, KVOf(f16))
					c := sseAttnKernel(t, k.name+"_exact_hd"+itoa(hd)+map[bool]string{true: "_f16"}[f16], b, err)
					n := npos
					if k.acc {
						n = hd
					}
					o0, o1 := sentinelRow(n), sentinelRow(n)
					if k.name == "acc_into" || k.name == "acc2_into" {
						for i := 0; i < n; i++ {
							o0[i], o1[i] = float32(i), -float32(i)
						}
					}
					a := Args{Out: &o0[0], Out2: &o1[0], W: d.cache(f16, 0), Rows: int64(npos),
						Q32: &d.q[0], Q2: &d.q2[0], AScale: &d.w[0], AScale2: &d.w2[0]}
					c.Call(&a)
					if !sseAttnGuarded(o0, n) || !sseAttnGuarded(o1, n) {
						t.Fatalf("%s hd=%d f16=%v: wrote past the row", k.name, hd, KVOf(f16))
					}
					return o0[:n], o1[:n]
				}
				a0, a1 := run(false)
				b0, b1 := run(true)
				for i := range a0 {
					if !sseAttnSame(a0[i], b0[i]) || !sseAttnSame(a1[i], b1[i]) {
						t.Fatalf("%s hd=%d npos=%d: element %d f32 (%v, %v) f16 (%v, %v)",
							k.name, hd, npos, i, a0[i], a1[i], b0[i], b1[i])
					}
					if math.IsNaN(float64(a0[i])) {
						t.Fatalf("%s hd=%d npos=%d: element %d is NaN on both widths -- a guard was read",
							k.name, hd, npos, i)
					}
				}
			}
		}
	}
}

// TestAttnF16KernelSizeSSE: with no F16C an f16 cache costs PMOVZXWD and a
// software widening per four elements, where f32 is one MOVUPS. The
// scores+acc pair must grow by more than the 5% nn.JIT.KVWidthPaysOff allows,
// so the SSE tier keeps an f32 cache unless the caller forces f16.
func TestAttnF16KernelSizeSSE(t *testing.T) {
	for _, hd := range []int{64, 80, 128, 256} {
		size := func(f func(int, int, KVFmt) ([]byte, error), f16 bool) int {
			b, err := f(hd, hd*4, KVOf(f16))
			if err != nil {
				t.Fatal(err)
			}
			return len(b)
		}
		s32, s16 := size(EmitAttnScoresSSE, false), size(EmitAttnScoresSSE, true)
		a32, a16 := size(EmitAttnAccSSE, false), size(EmitAttnAccSSE, true)
		grow := float64(s16+a16) / float64(s32+a32)
		t.Logf("sse hd=%3d  scores %d -> %d  acc %d -> %d  pair %+.0f%%",
			hd, s32, s16, a32, a16, 100*(grow-1))
		if grow <= 1.05 {
			t.Errorf("hd=%d: the SSE f16 pair grows %.3fx, inside KVWidthPaysOff's 5%% band -- "+
				"it would pick an f16 cache on a host with no F16C", hd, grow)
		}
	}
}

// TestAttnScoresTiledSSE is TestAttnScoresTiledMatchesReference for the SSE
// tier at every query tile the registers allow, both KV widths: bit for bit
// against the model, and -- one chain per query being oracle.Dot32's order --
// against the oracle itself.
func TestAttnScoresTiledSSE(t *testing.T) {
	const (
		stride      = 768 // kv and q rows both step by the residual width
		scoreStride = 136 // deliberately not a round number
		npos        = 37
	)
	for _, hd := range []int{64, 17, 12, 3, 1} {
		for _, f16 := range []bool{false, true} {
			qts := []int{1, 2, 4, 8, 11, 14}
			if f16 {
				qts = []int{1, 2, 8, 11}
			}
			for _, qt := range qts {
				rng := rand.New(rand.NewSource(int64(hd*131 + qt)))
				d := newAttnSSEData(hd, stride, npos, int64(hd+qt), false)
				q := make([]float32, qt*stride+attnGuard)
				for i := range q {
					q[i] = float32(math.NaN())
					if i < qt*stride && i%stride < hd {
						q[i] = float32(rng.NormFloat64())
					}
				}
				tag := "scores_t" + itoa(qt) + "_hd" + itoa(hd) + map[bool]string{true: "_f16"}[f16]
				b, err := EmitAttnScoresTiledSSE(hd, stride, stride, scoreStride, qt, KVOf(f16))
				c := sseAttnKernel(t, tag, b, err)
				scores := make([]float32, qt*scoreStride+attnGuard)
				for i := range scores {
					scores[i] = math.Float32frombits(softmaxSentinel)
				}
				a := Args{Out: &scores[0], W: d.cache(f16, 0), Rows: npos, Q32: &q[0]}
				c.Call(&a)
				same := 0
				for j := 0; j < qt; j++ {
					qj := q[j*stride : j*stride+hd]
					for p := 0; p < npos; p++ {
						got := scores[j*scoreStride+p]
						if m := attnScoreModelSSE(qj, d.row(p), hd, 1); !sseAttnSame(got, m) {
							t.Fatalf("%s: query %d key %d is %v, the kernel's own arithmetic gives %v",
								tag, j, p, got, m)
						}
						o := oracle.Dot32(qj, d.row(p))
						if rel := math.Abs(float64(got-o)) / (math.Abs(float64(o)) + 1e-9); rel > 1e-5 || math.IsNaN(rel) {
							t.Fatalf("%s: query %d key %d is %v, oracle.Dot32 gives %v", tag, j, p, got, o)
						}
						if sseAttnSame(got, o) {
							same++
						}
					}
				}
				for i, v := range scores {
					if i%scoreStride >= npos || i >= qt*scoreStride {
						if math.Float32bits(v) != softmaxSentinel {
							t.Fatalf("%s: wrote element %d, outside every row's npos prefix", tag, i)
						}
					}
				}
				if same != qt*npos {
					t.Logf("%s: %d of %d scores are bit-identical to oracle.Dot32 (Go fused its "+
						"multiply-adds on this build)", tag, same, qt*npos)
				}
			}
			// One past the register file is refused by name, not emitted wrong.
			over := 15
			if f16 {
				over = 12
			}
			if _, err := EmitAttnScoresTiledSSE(hd, stride, stride, scoreStride, over, KVOf(f16)); err == nil {
				t.Errorf("hd=%d f16=%v: qt=%d emitted, and it needs more than sixteen XMM registers", hd, f16, over)
			}
		}
	}
}
