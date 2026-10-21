//go:build jitllmtest

package nn

import (
	"runtime"
	"testing"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// forcedPreVNNI builds a JIT as though this host had no AVX-VNNI, and reports
// whether the forcing actually took.
//
// It checks the probe rather than assuming the hook took:
// cpu.ForceNoVNNIForTest is a no-op on architectures with no VNNI to pretend
// away, and a hook that reaches nothing leaves the gate green.
func forcedPreVNNI(t *testing.T, maxK, maxRows int, ts []quant.Type) *JIT {
	t.Helper()
	old := cpu.ForceNoVNNIForTest(true)
	if cpu.HostDotKind() != cpu.DotVEX {
		cpu.ForceNoVNNIForTest(old)
		t.Fatal("ForceNoVNNIForTest did not make HostDotKind report DotVEX, so " +
			"this arm would be the VNNI kernel compared with itself")
	}
	j := NewJIT(maxK, maxRows, ts)
	cpu.ForceNoVNNIForTest(old)
	if cpu.HostDotKind() != cpu.DotVNNI {
		t.Fatal("ForceNoVNNIForTest did not restore the probe")
	}
	return j
}

// needPreVNNI decides whether this host can run both sequences, and skips
// naming why when it cannot. On arm64 there is no second sequence (the packed
// kernels use SDOT; see nn.TestPackedFamilyIsBuiltOnThisHost). A host with no
// VNNI has no VNNI arm, and a host on the SSE tier runs neither.
func needPreVNNI(t *testing.T) {
	t.Helper()
	if tier := cpu.HostTier(); tier != cpu.TierAVX2 && tier != cpu.TierNEON {
		t.Skipf("tier %v runs neither VPDPBUSD nor its VEX sequence; its packed kernels are "+
			"gated by the SSE tier's own gates (cpu.TestSSEPackedMatchesAVX2BitForBit)", tier)
	}
	native := cpu.HostDotKind()
	old := cpu.ForceNoVNNIForTest(true)
	got := cpu.HostDotKind()
	cpu.ForceNoVNNIForTest(old)
	if got != cpu.DotVEX {
		t.Skipf("%s has no pre-VNNI sequence: HostDotKind stays %s under forcing, "+
			"so there is nothing here to compare (see nn.TestPackedFamilyIsBuiltOnThisHost)",
			runtime.GOARCH, got)
	}
	if native != cpu.DotVNNI {
		t.Skipf("this host has no AVX-VNNI (HostDotKind %s unforced), so there is no VNNI arm "+
			"to hold the pre-VNNI sequence to", native)
	}
}

// TestPreVNNIMatchesVNNIBitForBit holds the pre-VNNI kernels to the VNNI ones
// with float32 equality. The sequence is built to be arithmetically identical:
// VPMADDUBSW+VPMADDWD+VPADDD equals VPDPBUSD wherever the int16 pair cannot
// saturate, and for Q8_0, where it can, VPSIGNB moves the sign onto the
// activation and signedFix restores VPDPBUSD's scale before the epilogue. An
// NMSE bound would pass a subtly wrong kernel.
func TestPreVNNIMatchesVNNIBitForBit(t *testing.T) {
	needPreVNNI(t)
	ran := 0
	for _, gt := range quant.PackedTypes {
		if !cpu.PackedSupported(gt) {
			t.Logf("%s: no packed emitter on this architecture", gt)
			continue
		}
		for _, sh := range prevnniShapes {
			if sh.k%int(gt.BlockElems()) != 0 {
				continue
			}
			pk := packOne(t, gt, sh.rows, sh.k)
			x := prevnniAct(sh.k)

			jv := NewJIT(sh.k, sh.rows, []quant.Type{gt})
			if jv == nil {
				t.Fatalf("%s: no VNNI JIT", gt)
			}
			vnni := make([]float32, sh.rows)
			if !jv.MatVecPacked(vnni, gt, pk, x, sh.rows, sh.k) {
				jv.Close()
				t.Fatalf("%s %dx%d: the VNNI arm declined", gt, sh.rows, sh.k)
			}
			jv.Close()

			jx := forcedPreVNNI(t, sh.k, sh.rows, []quant.Type{gt})
			if jx == nil {
				t.Fatalf("%s: no pre-VNNI JIT", gt)
			}
			vex := make([]float32, sh.rows)
			if !jx.MatVecPacked(vex, gt, pk, x, sh.rows, sh.k) {
				jx.Close()
				t.Fatalf("%s %dx%d: the pre-VNNI arm declined -- a pre-VNNI host "+
					"would decode this weight through the float64 oracle", gt, sh.rows, sh.k)
			}
			jx.Close()

			for r := 0; r < sh.rows; r++ {
				if vnni[r] != vex[r] {
					t.Fatalf("%s %dx%d: row %d vnni %v vex %v -- the two sequences "+
						"are supposed to be bit-identical", gt, sh.rows, sh.k, r, vnni[r], vex[r])
				}
			}
			ran++
		}
	}
	if ran == 0 {
		t.Fatal("no (format, shape) pair ran -- this gate proved nothing")
	}
	t.Logf("%d pair(s) bit-identical between vnni and vex", ran)
}

// TestPreVNNISignedViolationIsCaught runs the gate above against the mistake a
// mechanical port actually makes, and shows it failing: keeping the VNNI
// kernel's +128 centring and swapping in the three-instruction sequence
// saturates VPMADDUBSW on Q8_0 (2*255*127 is 198% of int16), leaving the dot
// products slightly wrong. Only bit equality sees it.
func TestPreVNNISignedViolationIsCaught(t *testing.T) {
	needPreVNNI(t)
	const gt = quant.Q8_0
	sh := prevnniShapes[0]
	pk := packOne(t, gt, sh.rows, sh.k)
	x := prevnniAct(sh.k)

	jv := NewJIT(sh.k, sh.rows, []quant.Type{gt})
	vnni := make([]float32, sh.rows)
	if !jv.MatVecPacked(vnni, gt, pk, x, sh.rows, sh.k) {
		t.Fatal("the VNNI arm declined")
	}
	jv.Close()

	restore := cpu.SetNaiveSignedQ8ForTest(true)
	// Deferred as well: a Fatal inside forcedPreVNNI must not leave the naive
	// sequence armed for every later test in the process.
	defer cpu.SetNaiveSignedQ8ForTest(restore)
	jx := forcedPreVNNI(t, sh.k, sh.rows, []quant.Type{gt})
	cpu.SetNaiveSignedQ8ForTest(restore)
	bad := make([]float32, sh.rows)
	if !jx.MatVecPacked(bad, gt, pk, x, sh.rows, sh.k) {
		t.Fatal("the violated arm declined, so the violation was not exercised")
	}
	jx.Close()

	diff := 0
	for r := 0; r < sh.rows; r++ {
		if vnni[r] != bad[r] {
			diff++
		}
	}
	if diff == 0 {
		t.Fatal("the +128-centred pre-VNNI kernel agreed with VNNI on every row: " +
			"either the violation hook is not wired or the equality gate above " +
			"cannot see a saturating kernel, and both make it decorative")
	}
	t.Logf("VIOLATION: naive +128 centring under VPMADDUBSW disagrees on %d of %d rows "+
		"(first: vnni %v, naive %v)", diff, sh.rows, vnni[0], bad[0])
}
