//go:build amd64

// amd64 only: every assertion here is about CPUID.

package cpu

import (
	"strings"
	"testing"
)

// TestISADeclarationRoundTrips checks that the declaration written at the head
// of a kernel is the one KernelISA reads back, and that an undeclared kernel
// reads as the AVX2 tier.
func TestISADeclarationRoundTrips(t *testing.T) {
	for _, mask := range []ISA{ISASSE2, ISATierSSE, ISATierSSE | ISASSE42 | ISAPOPCNT} {
		var a Buf
		a.DeclareISA(mask)
		a.PXOR(XMM0, XMM0, XMM0)
		a.RET()
		code := a.Bytes()
		got, declared := KernelISA(code)
		if !declared || got != mask {
			t.Errorf("declared %v, read back %v (declared=%v) from % x", mask, got, declared, code[:ISAMarkerLen])
		}
		if KernelTier(code) != TierSSE {
			t.Errorf("%v: KernelTier = %v, want sse", mask, KernelTier(code))
		}
	}
	// An AVX2-tier kernel carries no declaration, and every one of them must
	// read as needing exactly the old baseline.
	code := EmitRMSNorm(64)
	if need, declared := KernelISA(code); declared || need != ISATierAVX2 {
		t.Errorf("EmitRMSNorm reads as %v declared=%v, want %v undeclared", need, declared, ISATierAVX2)
	}
	if KernelTier(code) != TierAVX2 && KernelTier(code) != primaryTier {
		t.Errorf("EmitRMSNorm's tier = %v", KernelTier(code))
	}
}

// TestDeclaredKernelsCannotExceedTheirDeclaration is the emit-time half of the
// ISA check: a declared buffer panics on an instruction its declaration does
// not cover. Each case is a violation that must panic, and the last is the
// control that must not.
func TestDeclaredKernelsCannotExceedTheirDeclaration(t *testing.T) {
	mustPanic := func(name, want string, f func()) {
		t.Helper()
		defer func() {
			r := recover()
			if r == nil {
				t.Errorf("%s: did NOT panic -- the declaration is not enforced at emit time", name)
				return
			}
			if msg, _ := r.(string); !strings.Contains(msg, want) {
				t.Errorf("%s: panicked with %q, want it to mention %q", name, r, want)
			}
		}()
		f()
	}
	sse := func() *Buf {
		var a Buf
		a.DeclareISA(ISATierSSE)
		return &a
	}
	// A VEX instruction in an SSE kernel: the SIGILL this tier exists to stop.
	mustPanic("vex op", "VEX", func() { sse().VPXOR(Y0, Y0, Y0) })
	mustPanic("vex memory op", "VEX", func() { sse().VMOVDQULoad(Y0, At(RSI, 0)) })
	mustPanic("vzeroupper", "VZEROUPPER", func() { sse().VZEROUPPER() })
	// An extension the kernel did not declare.
	sse2 := func() *Buf {
		var a Buf
		a.DeclareISA(ISASSE2)
		return &a
	}
	mustPanic("pshufb in sse2", "ssse3", func() { sse2().PSHUFB(XMM0, XMM0, XMM1) })
	mustPanic("pmaddubsw in sse2", "ssse3", func() { sse2().PMADDUBSW(XMM0, XMM0, XMM1) })
	mustPanic("pmovzxbd in sse2", "sse4.1", func() { sse2().PMOVZXBDLoad(XMM0, At(RSI, 0)) })
	mustPanic("haddps in sse2", "sse3", func() { sse2().HADDPS(XMM0, XMM0, XMM1) })
	mustPanic("popcnt in sse tier", "popcnt", func() { sse().POPCNT(RAX, RCX) })
	mustPanic("pcmpgtq in sse tier", "sse4.2", func() { sse().PCMPGTQ(XMM0, XMM0, XMM1) })
	// The declaration itself.
	mustPanic("declare late", "first", func() {
		var a Buf
		a.RET()
		a.DeclareISA(ISATierSSE)
	})
	mustPanic("declare twice", "twice", func() { sse().DeclareISA(ISATierSSE) })
	mustPanic("declare without sse2", "sse2", func() {
		var a Buf
		a.DeclareISA(ISASSSE3)
	})

	// The control: everything the SSE floor covers emits without complaint,
	// and an undeclared buffer (an AVX2 kernel, or an encoder test) is not
	// checked at all.
	a := sse()
	a.PSHUFB(XMM0, XMM0, XMM1)
	a.PMOVZXBDLoad(XMM2, At(RSI, 0))
	a.HADDPS(XMM3, XMM3, XMM4)
	a.RET()
	var u Buf
	u.VPXOR(Y0, Y0, Y0)
	u.PSHUFB(XMM0, XMM0, XMM1)
	u.VZEROUPPER()
}

// TestISANamesAreTheProbesNames pins the names an ISAError and Baseline print,
// because a refusal the user reads has to name the extension they would look
// up.
func TestISANamesAreTheProbesNames(t *testing.T) {
	if got := ISATierSSE.String(); got != "sse2, sse3, ssse3, sse4.1" {
		t.Errorf("ISATierSSE = %q", got)
	}
	if got := ISATierAVX2.String(); got != "avx2, fma, f16c" {
		t.Errorf("ISATierAVX2 = %q", got)
	}
	f := Features{SSE2: true, SSE3: true, AVX2: true, FMA: true, F16C: true}
	if got := missingNames(f, ISATierAVX2|ISATierSSE); got != "ssse3, sse4.1, avx2 (present, but the OS does not save YMM state), fma (present, but unusable without OS YMM state), f16c (present, but unusable without OS YMM state)" {
		t.Errorf("missingNames = %q", got)
	}
	for tier, want := range map[Tier]string{TierNone: "none", TierSSE: "sse", TierAVX2: "avx2", TierNEON: "neon"} {
		if tier.String() != want {
			t.Errorf("%d.String() = %q, want %q", tier, tier.String(), want)
		}
	}
	// tierOf is the one rule, and it is checked on synthetic hosts because this
	// box can only ever be one of them.
	sse := Features{SSE2: true, SSE3: true, SSSE3: true, SSE41: true, SSE42: true, POPCNT: true}
	if tierOf(sse) != TierSSE {
		t.Errorf("an Atom's feature set is tier %v, want sse", tierOf(sse))
	}
	avx := sse
	avx.AVX2, avx.FMA, avx.F16C, avx.OSAVX = true, true, true, true
	if tierOf(avx) != TierAVX2 {
		t.Errorf("an AVX2 host is tier %v", tierOf(avx))
	}
	noOS := avx
	noOS.OSAVX = false
	if tierOf(noOS) != TierSSE {
		t.Errorf("AVX2 without OS YMM state is tier %v, want sse: the unit is unusable", tierOf(noOS))
	}
	old := sse
	old.SSSE3 = false
	if tierOf(old) != TierNone {
		t.Errorf("a host without SSSE3 is tier %v, want none", tierOf(old))
	}
}
