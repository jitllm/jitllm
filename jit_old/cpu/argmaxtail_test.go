//go:build amd64 && linux

package cpu

import (
	"errors"
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// The greedy argmax's gate, for every length and both x86 tiers.
//
// The bar is equality with model.Greedy's scan: an argmax has no tolerance,
// the index is either right or a different token. It compares the index and
// the value's bits over every length 1..40, the shipping vocabularies and the
// ragged lengths around them, and the data shapes where a lane-parallel fold
// goes wrong: ascending (maximum in the tail), descending, flat (all tie),
// all-negative (a zero leaking into a lane shows only here) and random. Both
// tiers do no arithmetic, so anything but exact agreement is a bug.

// argmaxScanGo is model.Greedy's former scan, copied rather than re-derived.
// It returns the value too, because `best` is what the kernel writes to Out.
func argmaxScanGo(x []float32) (int32, float32) {
	best, bi := float32(math.Inf(-1)), int32(0)
	for i, v := range x {
		if v > best {
			best, bi = v, int32(i)
		}
	}
	return bi, best
}

// argmaxKernel is one tier's mapped kernel.
type argmaxKernel struct {
	name string
	c    *Code
}

var argmaxTestConsts = ArgmaxConsts()

// call runs the kernel over the whole of x, in the ABI nn uses: Args.K is the
// element count.
func (k argmaxKernel) call(x []float32) (int32, float32) {
	var idx int32
	var val float32
	args := Args{
		K:    int64(len(x)),
		Out:  &val,
		ASum: &idx,
		Scr:  (*byte)(unsafe.Pointer(&argmaxTestConsts[0])),
	}
	if len(x) > 0 {
		args.Q32 = &x[0]
	}
	k.c.Call(&args)
	return idx, val
}

// argmaxKernels maps both tiers' kernels, and checks each is the one the
// emitter table hands nn, so a right kernel behind a miswired table fails.
func argmaxKernels(t *testing.T) []argmaxKernel {
	t.Helper()
	var ks []argmaxKernel

	avx := EmitArgmax()
	if len(avx) == 0 {
		t.Fatal("EmitArgmax produced no code on amd64")
	}
	if b, err := EmittersFor(TierAVX2).Argmax(); err != nil {
		t.Fatalf("the AVX2 table refused its own argmax: %v", err)
	} else if string(b) != string(avx) {
		t.Fatal("EmittersFor(TierAVX2).Argmax() is not EmitArgmax's bytes")
	}
	switch c, err := Map(avx); {
	case err == nil:
		t.Cleanup(func() { c.Close() })
		ks = append(ks, argmaxKernel{"avx2", c})
	case errors.Is(err, ErrISA):
		t.Logf("NOT RUNNABLE HERE: %v -- the SSE half of this gate still runs, "+
			"which on an Atom is the half that matters", err)
	default:
		t.Fatalf("Map(argmax avx2): %v", err)
	}

	sse := emitArgmaxSSEBody()
	if KernelTier(sse) != TierSSE {
		t.Fatal("the SSE argmax is not declared SSE-tier")
	}
	requireSSEKernel(t, "argmax_sse", sse)
	// The table's entry is checked only when it has one; when wired it must
	// return exactly these bytes.
	if b, err := EmittersFor(TierSSE).Argmax(); err == nil && string(b) != string(sse) {
		t.Fatal("EmittersFor(TierSSE).Argmax() is wired to something other than emitArgmaxSSEBody")
	} else if err != nil {
		t.Logf("the SSE table still returns %q -- this gate drives emitArgmaxSSEBody directly", err)
	}
	c, err := Map(sse)
	if err != nil {
		// An SSE kernel maps on any host that runs this suite: a part below
		// SSE4.1 is refused by the tier probe long before a kernel is emitted.
		t.Fatalf("Map(argmax sse): %v", err)
	}
	t.Cleanup(func() { c.Close() })
	return append(ks, argmaxKernel{"sse", c})
}

// argmaxLens is every length from 1 to 40 -- no vector at all, one vector,
// a vector and a tail of every residue -- and then the vocabularies that ship
// with the ragged lengths around them. 50257 (GPT-2) is the only shipping
// vocabulary eight does not divide, so it is what reaches the tail.
func argmaxLens() []int {
	var ns []int
	for n := 1; n <= 40; n++ {
		ns = append(ns, n)
	}
	for _, v := range []int{32000, 49152, 50257, 128256, 151936, 201088, 256000} {
		ns = append(ns, v-7, v-1, v, v+1, v+7)
	}
	return ns
}

var argmaxShapes = []string{"ascending", "descending", "flat", "negative", "random", "mixed"}

func argmaxData(shape string, n int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	x := make([]float32, n)
	for i := range x {
		switch shape {
		case "ascending": // the maximum is the LAST element, i.e. in the tail
			x[i] = float32(i) * 1e-3
		case "descending": // and here it is the first
			x[i] = float32(n-i) * 1e-3
		case "flat": // every element ties; index 0 must win
			x[i] = 1.25
		case "negative": // a zero leaking into a lane would win this one
			x[i] = -1 - float32(r.Float64())*40
		case "mixed": // both zeros and a plateau of equal maxima
			switch i % 5 {
			case 0:
				x[i] = 0
			case 1:
				x[i] = float32(math.Copysign(0, -1))
			case 2:
				x[i] = 7.5
			default:
				x[i] = float32(r.NormFloat64())
			}
		default:
			x[i] = float32(r.NormFloat64()) * 8
		}
	}
	return x
}

func TestArgmaxServesEveryLengthOnBothTiers(t *testing.T) {
	ks := argmaxKernels(t)
	ran := 0
	for _, n := range argmaxLens() {
		for _, shape := range argmaxShapes {
			x := argmaxData(shape, n, int64(n*31+len(shape)))
			wi, wv := argmaxScanGo(x)
			for _, k := range ks {
				gi, gv := k.call(x)
				if gi != wi {
					// The index may be out of range (0x7FFFFFFF when no lane
					// ties), so it is reported rather than dereferenced.
					t.Fatalf("%s n=%d %s: kernel index %d, scan %d (x[%d]=%v)",
						k.name, n, shape, gi, wi, wi, x[wi])
				}
				if math.Float32bits(gv) != math.Float32bits(wv) {
					t.Fatalf("%s n=%d %s: kernel value %v (%#08x), scan %v (%#08x)",
						k.name, n, shape, gv, math.Float32bits(gv), wv, math.Float32bits(wv))
				}
				ran++
			}
		}
	}
	t.Logf("%d (length, shape, tier) cases agree with model.Greedy's scan, index and value bits",
		ran)
}

// TestArgmaxKeepsTheLowestIndexOfATie is the property greedy decoding rests on,
// tested where a lane-parallel fold is most likely to lose it: exhaustively at
// every short length -- so the two maxima land in every pair of lanes, in every
// pair of groups, and across the seam between the vector loop and the scalar
// tail -- and then at the placements that matter in a real vocabulary.
func TestArgmaxKeepsTheLowestIndexOfATie(t *testing.T) {
	ks := argmaxKernels(t)
	pairs := 0
	tie := func(n, i, j int) {
		t.Helper()
		x := make([]float32, n)
		for e := range x {
			x[e] = -1
		}
		x[i], x[j] = 9, 9
		for _, k := range ks {
			gi, _ := k.call(x)
			if gi != int32(i) {
				t.Fatalf("%s n=%d: maxima at %d and %d, kernel chose %d -- the LOWER "+
					"index must win, or a greedy chain diverges from llama.cpp exactly "+
					"where two logits are equal", k.name, n, i, j, gi)
			}
		}
		pairs++
	}
	for n := 1; n <= 40; n++ {
		for i := 0; i < n; i++ {
			for j := i; j < n; j++ {
				tie(n, i, j)
			}
		}
	}
	for _, n := range []int{32000, 50257, 128256, 256000 + 3} {
		v := n - n%8 // the first element of the tail
		for _, p := range [][2]int{
			{0, 1}, {0, n - 1}, {5, 13}, {7, 8}, {n - 9, n - 1},
			{1000, 24000}, {v - 1, v}, {v, n - 1}, {0, v}, {v - 8, v + 1},
		} {
			if p[0] >= 0 && p[1] < n && p[0] <= p[1] {
				tie(n, p[0], p[1])
			}
		}
	}
	t.Logf("%d tie placements keep the lower index on every tier", pairs)
}

// TestArgmaxReadsNothingPastTheLastLogit is why the tail is scalar: the logits
// end right before a PROT_NONE page, so a kernel that loads a whole vector to
// use one element faults here, where a Go heap slice would hide the over-read.
func TestArgmaxReadsNothingPastTheLastLogit(t *testing.T) {
	ks := argmaxKernels(t)
	for _, n := range argmaxLens() {
		if n > 1000 && n%8 == 0 {
			continue // the interesting lengths are the ragged ones
		}
		x := guardF32(t, argmaxData("random", n, int64(n)))
		wi, _ := argmaxScanGo(x)
		for _, k := range ks {
			if gi, _ := k.call(x); gi != wi {
				t.Fatalf("%s n=%d at a guard page: index %d, scan %d", k.name, n, gi, wi)
			}
		}
	}
}

// TestArgmaxNeverLetsANaNWin is the claim EmitArgmax's comment makes about its
// VMAXPS operand order.
func TestArgmaxNeverLetsANaNWin(t *testing.T) {
	ks := argmaxKernels(t)
	nan := float32(math.NaN())
	for _, n := range []int{1, 3, 8, 9, 16, 23, 32, 4096, 4099} {
		for _, place := range []string{"first", "last", "all", "scattered"} {
			x := argmaxData("random", n, int64(n))
			switch place {
			case "first":
				x[0] = nan
			case "last":
				x[n-1] = nan
			case "all":
				for i := range x {
					x[i] = nan
				}
			default:
				for i := range x {
					if i%3 == 0 {
						x[i] = nan
					}
				}
			}
			wi, _ := argmaxScanGo(x)
			for _, k := range ks {
				gi, _ := k.call(x)
				if gi != wi {
					t.Fatalf("%s n=%d NaN %s: kernel %d, scan %d -- a NaN must never "+
						"win an argmax and must never take the running maximum with it",
						k.name, n, place, gi, wi)
				}
			}
		}
	}
}

// argmaxPredecessor is the previous kernel: whole vectors only (Args.K counted
// vectors), and VMAXPS with the running maximum as SRC1. It lives in the test
// as the violation the gates are run against.
func argmaxPredecessor() []byte {
	var a Buf
	a.MOVLoad(RSI, At(RDI, 112))
	a.MOVLoad(RBX, At(RDI, 56))
	a.MOVLoad(R10, At(RDI, 40)) // K -> VECTORS of eight
	a.VBROADCASTSS(Y0, At(RBX, 40))
	a.VMOVDQULoad(Y1, At(RBX, 0))
	a.VMOVDQULoad(Y2, At(RBX, 0))
	a.VPBROADCASTD(Y3, At(RBX, 32))
	lp := a.Label()
	a.Bind(lp)
	a.VMOVDQULoad(Y4, At(RSI, 0))
	a.VCMPPS(Y5, Y4, Y0, 14)
	a.VMAXPS(Y0, Y0, Y4) // the old order: a NaN element becomes the maximum
	a.VPAND(Y6, Y2, Y5)
	a.VPANDN(Y7, Y5, Y1)
	a.VPOR(Y1, Y6, Y7)
	a.VPADDD(Y2, Y2, Y3)
	a.ADDimm(RSI, 32)
	a.DEC(R10)
	a.JNZ(lp)
	a.VMAXPS(Y8, Y0, Y0)
	a.VEXTRACTF128(Y4, Y0, 1)
	a.VMAXPS(Y0, Y0, Y4)
	a.VSHUFPS(Y4, Y0, Y0, 0x4E)
	a.VMAXPS(Y0, Y0, Y4)
	a.VSHUFPS(Y4, Y0, Y0, 0xB1)
	a.VMAXPS(Y0, Y0, Y4)
	a.VBROADCASTSSReg(Y0, Y0)
	a.VCMPPS(Y5, Y8, Y0, 0)
	a.VPBROADCASTD(Y6, At(RBX, 36))
	a.VPAND(Y7, Y1, Y5)
	a.VPANDN(Y6, Y5, Y6)
	a.VPOR(Y1, Y7, Y6)
	a.VEXTRACTI128(Y4, Y1, 1)
	a.VPMINSD(Y1, Y1, Y4)
	a.VSHUFPS(Y4, Y1, Y1, 0x4E)
	a.VPMINSD(Y1, Y1, Y4)
	a.VSHUFPS(Y4, Y1, Y1, 0xB1)
	a.VPMINSD(Y1, Y1, Y4)
	a.MOVLoad(R8, At(RDI, 96))
	a.VMOVSSStore(At(R8, 0), Y1)
	a.MOVLoad(R9, At(RDI, 0))
	a.VMOVSSStore(At(R9, 0), Y0)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// TestArgmaxGateSeesItsViolations runs the comparisons above against each
// defect they exist to catch; each leaves a kernel that answers plausibly.
func TestArgmaxGateSeesItsViolations(t *testing.T) {
	ks := argmaxKernels(t)

	// Violation 1: the tie rule, mutated in the oracle. A scan that keeps the
	// last of a tie must disagree with the kernel.
	lastOfTie := func(x []float32) int32 {
		best, bi := float32(math.Inf(-1)), int32(0)
		for i, v := range x {
			if v >= best {
				best, bi = v, int32(i)
			}
		}
		return bi
	}
	for _, n := range []int{9, 17, 33, 256, 50257} {
		x := make([]float32, n)
		for i := range x {
			x[i] = -1
		}
		x[3], x[n-1] = 5, 5
		for _, k := range ks {
			gi, _ := k.call(x)
			if gi != 3 {
				t.Fatalf("%s n=%d: kernel chose %d, want 3", k.name, n, gi)
			}
			if gi == lastOfTie(x) {
				t.Fatalf("%s n=%d: the kernel agrees with a last-of-tie scan, so this "+
					"gate cannot tell the two rules apart", k.name, n)
			}
		}
	}

	// Violation 2: no tail. Passing Args.K as a count of whole vectors makes
	// the kernel ignore the last n%8 elements. n=1 is left out: scanning
	// nothing returns index 0, which is right for a single element.
	for _, n := range []int{7, 9, 23, 50257} {
		x := argmaxData("ascending", n, 5) // the maximum is the last element
		wi, _ := argmaxScanGo(x)
		for _, k := range ks {
			var idx int32
			var val float32
			args := Args{
				K:    int64(len(x) / ArgmaxLanes), // the old meaning: whole vectors
				Q32:  &x[0],
				Out:  &val,
				ASum: &idx,
				Scr:  (*byte)(unsafe.Pointer(&argmaxTestConsts[0])),
			}
			k.c.Call(&args)
			if idx == wi {
				t.Fatalf("%s n=%d: the kernel returned the right index %d from a K that "+
					"names %d elements -- this gate cannot see a missing tail",
					k.name, n, idx, len(x)/ArgmaxLanes)
			}
			t.Logf("%s n=%d: the vector-count contract answers %d where the scan says %d",
				k.name, n, idx, wi)
		}
	}

	// Violation 3: the VMAXPS operand order of the predecessor. A NaN logit
	// becomes its running maximum, the equality fold matches no lane, and it
	// returns the INT_MAX sentinel.
	c, err := Map(argmaxPredecessor())
	if errors.Is(err, ErrISA) {
		t.Skip("no AVX2 on this host: the predecessor cannot be run here")
	}
	if err != nil {
		t.Fatalf("Map(predecessor): %v", err)
	}
	defer c.Close()
	// Whether it fails depends on the NaN's lane (the fold's VMAXPS also
	// returns its second operand on a NaN), so several placements are tried.
	bad := 0
	for _, place := range []int{0, 1, 2, 3, 4, 5, 6, 7, 13, -1} {
		x := argmaxData("random", 64, 1)
		if place < 0 {
			for i := range x {
				x[i] = float32(math.NaN())
			}
		} else {
			x[place] = float32(math.NaN())
		}
		wi, _ := argmaxScanGo(x)
		var idx int32
		var val float32
		c.Call(&Args{
			K: int64(len(x) / ArgmaxLanes), Q32: &x[0], Out: &val, ASum: &idx,
			Scr: (*byte)(unsafe.Pointer(&argmaxTestConsts[0])),
		})
		if idx != wi {
			bad++
			t.Logf("predecessor: a NaN at %d of 64 gives index %d (value %v) where the "+
				"scan says %d", place, idx, val, wi)
		}
	}
	if bad == 0 {
		t.Fatal("the predecessor agrees with the scan on every NaN placement, so the " +
			"operand-order claim in EmitArgmax's comment is not measured by anything")
	}
	t.Logf("the predecessor disagrees with the scan on %d of 10 NaN placements; today's "+
		"kernel is held to the scan on the same shapes in TestArgmaxNeverLetsANaNWin", bad)
}
