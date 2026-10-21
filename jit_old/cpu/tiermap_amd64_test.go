//go:build amd64

package cpu

import (
	"errors"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// sseDoubleKernel is the smallest real SSE-tier kernel: x[i] += x[i] over
// Args.K units of 8 and Args.Rows tail elements, through elemLoopSSE, ending
// in a plain RET: SSE by declaration and by construction.
func sseDoubleKernel(mask ISA) []byte {
	var a Buf
	a.DeclareISA(mask)
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(R10, At(RDI, 40)) // K
	a.MOVLoad(R11, At(RDI, 32)) // Rows
	elemLoopSSE(&a, R10, R11, []Reg{RCX}, func(ld func(dst, p Reg), st func(p, src Reg)) {
		ld(XMM0, RCX)
		a.ADDPS(XMM0, XMM0, XMM0)
		st(RCX, XMM0)
	})
	a.RET()
	return a.Bytes()
}

// TestMapAdmitsWhatTheHostCanRunAndNamesWhatItCannot is Map's ISA gate in
// both directions on this host, then on a forced SSE host and a forced
// below-floor host. Under the SSE force the SSE kernel must map and run (a
// byte scan tripping over a legal C5 ModRM byte would refuse it), and the AVX2
// kernel must be refused naming the missing extensions.
func TestMapAdmitsWhatTheHostCanRunAndNamesWhatItCannot(t *testing.T) {
	avx := EmitRMSNorm(64)
	sse := sseDoubleKernel(ISATierSSE)

	run := func(t *testing.T) {
		t.Helper()
		c, err := Map(sse)
		if err != nil {
			t.Fatalf("the SSE kernel did not map: %v", err)
		}
		defer c.Close()
		x := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
		c.Call(&Args{Out: &x[0], K: 1, Rows: 3})
		for i, v := range x {
			if want := float32(2 * (i + 1)); v != want {
				t.Fatalf("x[%d] = %v, want %v (the SSE kernel mapped and computed wrong)", i, v, want)
			}
		}
	}

	refusedByName := func(t *testing.T) {
		t.Helper()
		_, err := Map(avx)
		if err == nil {
			t.Fatal("Map accepted a VEX kernel on an SSE host -- this is the Atom's SIGILL")
		}
		if !errors.Is(err, ErrISA) {
			t.Fatalf("refusal is %v, want an ErrISA", err)
		}
		for _, n := range []string{"avx2", "fma", "f16c"} {
			if !strings.Contains(err.Error(), n) {
				t.Errorf("refusal %q does not name %s", err, n)
			}
		}
		if _, err := MapNamed(avx, "rmsnorm"); !errors.Is(err, ErrISA) {
			t.Errorf("MapNamed is not gated the same way: %v", err)
		}
	}

	host := HostTier()
	switch host {
	case TierAVX2:
		// Unforced: this host runs both tiers' kernels.
		if c, err := Map(avx); err != nil {
			t.Fatalf("an AVX2 kernel did not map on an AVX2 host: %v", err)
		} else {
			c.Close()
		}
		run(t)
	case TierSSE:
		// The real host is an SSE-tier machine: the probe is the answer.
		refusedByName(t)
		run(t)
	default:
		t.Skipf("this host is tier %v: below every floor this gate can exercise", HostTier())
	}

	t.Run("forced sse", func(t *testing.T) {
		if host != TierAVX2 {
			t.Skip("the host is itself SSE-tier; the unforced arm above covered it")
		}
		withTier(t, TierSSE)
		refusedByName(t)
		run(t)
		// Baseline passes an SSE host exactly when the SSE table is complete,
		// and otherwise names the pending ops (TestBaselineAnswersFromTheSSETable
		// owns the detail).
		if err, pending := Baseline(), SSEPending(); (err == nil) != (len(pending) == 0) {
			t.Errorf("Baseline on a forced-SSE host = %v with %d SSE ops pending", err, len(pending))
		}
		// The probes still run with every VEX unit forced away: they go
		// through mapExec, not the gate.
		if f := CPU(); !f.SSE2 || !f.SSSE3 || f.AVX2 || f.FMA || f.F16C || f.AVXVNNI {
			t.Errorf("forced-SSE feature set is wrong: %+v", f)
		}
	})

	t.Run("forced below the floor", func(t *testing.T) {
		withTier(t, TierNone)
		_, err := Map(sse)
		if err == nil {
			t.Fatal("Map accepted an SSSE3/SSE4.1 kernel on a host without them")
		}
		for _, n := range []string{"ssse3", "sse4.1"} {
			if !strings.Contains(err.Error(), n) {
				t.Errorf("refusal %q does not name %s", err, n)
			}
		}
		// A kernel that declares only SSE2 is still runnable there: the
		// declaration is per kernel, not per tier.
		if c, err := Map(sseDoubleKernel(ISASSE2)); err != nil {
			t.Errorf("an SSE2-only kernel was refused below the SSE floor: %v", err)
		} else {
			c.Close()
		}
		if err := Baseline(); err == nil || !strings.Contains(err.Error(), "missing ssse3, sse4.1") {
			t.Errorf("Baseline below the floor = %v, want the missing SSE-floor extensions named", err)
		}
	})

	// And the force is scoped: the host is itself again.
	if HostTier() != host {
		t.Fatalf("after the forced arms the host reads %v, was %v -- the force leaked", HostTier(), host)
	}
	if err, want := Baseline(), host == TierAVX2 || (host == TierSSE && len(SSEPending()) == 0); (err == nil) != want {
		t.Fatalf("after the forced arms Baseline says %v on a %v host -- its cache kept the forced verdict", err, host)
	}
}

// TestMappedByTierCountsTheDeclaration is the selection check's own gate: a
// Map of an SSE kernel counts under sse and a Map of an AVX2 kernel under avx2.
// A counter that miscounts would let a forced run that silently used AVX2
// kernels read as an SSE run.
func TestMappedByTierCountsTheDeclaration(t *testing.T) {
	if HostTier() != TierAVX2 {
		t.Skip("needs a host that can map both tiers")
	}
	before := MappedByTier()
	c1 := mustMap(t, sseDoubleKernel(ISATierSSE))
	c1.Close()
	c2 := mustMap(t, EmitRMSNorm(64))
	c2.Close()
	after := MappedByTier()
	if d := after[TierSSE] - before[TierSSE]; d != 1 {
		t.Errorf("mapping one SSE kernel moved the sse count by %d", d)
	}
	if d := after[TierAVX2] - before[TierAVX2]; d != 1 {
		t.Errorf("mapping one AVX2 kernel moved the avx2 count by %d", d)
	}
	// The probes are not counted: they are not kernels.
	b := MappedByTier()
	retier(TierAVX2)
	if a := MappedByTier(); a[TierAVX2] != b[TierAVX2] || a[TierSSE] != b[TierSSE] {
		t.Errorf("re-probing moved the map counters: %v -> %v", b, a)
	}
}

// TestForcedTierResetsEveryDerivedAnswer gates the force itself: every
// question derived from the probe must answer for the forced host, including
// the ones cached behind their own Once.
func TestForcedTierResetsEveryDerivedAnswer(t *testing.T) {
	if HostTier() != TierAVX2 {
		t.Skip("needs an AVX2 host to force down from")
	}
	// Prime every cached answer with the real host's verdict first, so a
	// force that fails to reset one is seen here and not only by whichever
	// test happens to run after it.
	if err := Baseline(); err != nil {
		t.Fatalf("an AVX2 host fails its own floor: %v", err)
	}
	// Baseline is probed below the floor: an SSE host passes it like the AVX2
	// host primed above, so only a forced TierNone host can show a cache that
	// was not reset.
	withTier(t, TierNone)
	if Baseline() == nil {
		t.Error("forced below the floor, Baseline still reports the pre-force verdict -- its Once was not reset")
	}
	withTier(t, TierSSE)
	if CPU().ISA()&ISATierAVX2 != 0 {
		t.Errorf("forced SSE, ISA still reports %v", CPU().ISA())
	}
	if HostDotKind() != DotVEX {
		t.Error("forced SSE, HostDotKind still reports VNNI")
	}
	if Native().Tier != TierSSE {
		t.Errorf("forced SSE, Native() is the %v table", Native().Tier)
	}
	// The row-major and packed capability questions follow the tier: the SSE
	// tier answers for itself, and until the matvec family lands it answers no.
	for _, q := range []quant.Type{quant.Q4_K, quant.Q8_0, quant.F32} {
		if got, want := SupportedPackedNative(q), EmittersFor(TierSSE).PackedSupported(q); got != want {
			t.Errorf("%v: SupportedPackedNative did not dispatch to the SSE tier (%v vs %v)", q, got, want)
		}
		if got, want := SupportedNative(q), EmittersFor(TierSSE).RowMajorSupported(q); got != want {
			t.Errorf("%v: SupportedNative did not dispatch to the SSE tier (%v vs %v)", q, got, want)
		}
	}
}
