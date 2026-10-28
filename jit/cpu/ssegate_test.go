//go:build amd64

package cpu

import (
	"os/exec"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/ssegate"
)

// The SSE tier's VEX-leak gate: every family's kernel gates call
// requireSSEKernel on what their emitters produce.
//
// DeclareISA's emit-time panics (isa.go) are the encoder checking itself; this
// reads the bytes back through objdump, which shares no code with the encoder,
// and fails on anything a pre-AVX host would fault on: a VEX or EVEX prefix
// (C4, C5, 62 in the opcode position, including VEX-encoded BMI ops), a ymm or
// zmm register, an instruction from an undeclared extension, and "(bad)".
//
// It decodes rather than byte-scans (C5 can be a ModRM byte, C4 is PINSRW's
// opcode), so the VEX test is exact. It also checks the trampoline's rules: no
// push, pop or call (RSP must be identical on every path), and no write to R14
// (Go's g), RBP (callee-saved and not saved by callKernel) or RSP.

// sseGateProblems returns every way code fails to be a legal SSE-tier
// kernel (internal/ssegate, shared with the composer's x86 tests).
func sseGateProblems(t testing.TB, name string, code []byte) []string {
	return ssegate.Problems(t, name, code)
}

// writesFirstOperand is false for the instructions whose first operand is
// only read.
func writesFirstOperand(mn string) bool { return ssegate.WritesFirstOperand(mn) }

// requireSSEKernel fails t for every problem sseGateProblems finds in code.
// Every SSE-tier family gate calls it on every kernel it builds, at every
// shape it tests.
func requireSSEKernel(t testing.TB, name string, code []byte) {
	t.Helper()
	ssegate.Require(t, name, code)
}

// TestVEXGateCatchesItsViolations runs the gate against each thing it exists
// to catch, and against the legal bytes a byte scan would get wrong.
func TestVEXGateCatchesItsViolations(t *testing.T) {
	if _, err := exec.LookPath("objdump"); err != nil {
		t.Skip("objdump not installed")
	}
	clean := func() *Buf {
		var a Buf
		a.DeclareISA(ISATierSSE)
		a.MOVLoad(RCX, At(RDI, 0))
		a.MOVUPSLoad(XMM0, At(RCX, 0))
		// The bytes a scan would misread: C5 as a ModRM byte, C4 as an
		// opcode, 62 as a ModRM. All legal legacy SSE.
		a.PXOR(XMM0, XMM0, XMM5)                // 66 0F EF C5
		a.PINSRWLoad(XMM1, XMM1, At(RSI, 0), 1) // 66 0F C4 0E 01
		a.PUNPCKLDQ(XMM4, XMM4, XMM2)           // 66 0F 62 E2
		a.PSHUFB(XMM2, XMM2, XMM3)
		a.PMOVZXBD(XMM3, XMM4)
		a.MOVUPSStore(At(RCX, 0), XMM0)
		a.RET()
		return &a
	}
	if p := sseGateProblems(t, "clean", clean().Bytes()); len(p) != 0 {
		t.Fatalf("the gate rejects a legal SSE kernel -- a false positive here is the "+
			"only kernel an Atom has:\n  %s", strings.Join(p, "\n  "))
	}

	splice := func(mask ISA, tail []byte) []byte {
		var a Buf
		a.DeclareISA(mask)
		a.MOVLoad(RCX, At(RDI, 0))
		return append(append([]byte(nil), a.Bytes()...), append(tail, 0xC3)...)
	}
	legacy := func(f func(a *Buf)) []byte {
		var a Buf
		f(&a)
		return a.Bytes()
	}
	cases := []struct {
		name string
		code []byte
		want string
	}{
		{"vzeroupper", splice(ISATierSSE, []byte{0xC5, 0xF8, 0x77}), "VEX"},
		{"vex xmm op", splice(ISATierSSE, legacy(func(a *Buf) { a.VADDPSx(Y0, Y1, Y2) })), "VEX"},
		{"vex ymm op", splice(ISATierSSE, legacy(func(a *Buf) { a.VPXOR(Y0, Y0, Y0) })), "256/512"},
		{"bmi2 shlx", splice(ISATierSSE, []byte{0xC4, 0xE2, 0xF9, 0xF7, 0xC0}), "VEX"},
		{"evex", splice(ISATierSSE, []byte{0x62, 0xF1, 0x7C, 0x48, 0x58, 0xC1}), "EVEX"},
		{"ssse3 undeclared", splice(ISASSE2, legacy(func(a *Buf) { a.PSHUFB(XMM0, XMM0, XMM1) })), "ssse3"},
		{"sse4.1 undeclared", splice(ISASSE2|ISASSSE3, legacy(func(a *Buf) { a.PMOVZXBD(XMM0, XMM1) })), "sse4.1"},
		{"push", splice(ISATierSSE, []byte{0x53}), "RSP"},
		{"writes r14", splice(ISATierSSE, legacy(func(a *Buf) { a.MOVQ(R14, RAX) })), "r14"},
		{"writes rbp", splice(ISATierSSE, legacy(func(a *Buf) { a.MOVLoad(RBP, At(RDI, 0)) })), "rbp"},
		{"undeclared avx2 kernel", EmitRMSNorm(64), "no ISA declaration"},
		{"declares avx2", func() []byte {
			var a Buf
			a.DeclareISA(ISATierSSE | ISAAVX2)
			a.RET()
			return a.Bytes()
		}(), "VEX extension"},
	}
	for _, c := range cases {
		p := sseGateProblems(t, strings.ReplaceAll(c.name, " ", "_"), c.code)
		hit := false
		for _, s := range p {
			hit = hit || strings.Contains(s, c.want)
		}
		if !hit {
			t.Errorf("%s: the gate did not report %q; it said %q", c.name, c.want, p)
		}
	}
}
