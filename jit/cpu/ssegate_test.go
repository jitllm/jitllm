//go:build amd64

package cpu

import (
	"fmt"
	"os/exec"
	"strings"
	"testing"
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

// sseMnemonicISA classifies every mnemonic an SSE-tier kernel may contain by
// the extension that carries it; 0 is base x86-64 (the GP instructions). An
// unclassified mnemonic is a gate failure, so an instruction nobody thought
// about cannot pass by default -- add it here with its extension.
var sseMnemonicISA = func() map[string]ISA {
	m := map[string]ISA{}
	add := func(isa ISA, names string) {
		for _, n := range strings.Fields(names) {
			m[n] = isa
		}
	}
	add(0, `mov movabs movzx movsx movsxd add sub and or xor inc dec neg not test cmp
		lea imul div shl shr sar sal rol ror jmp je jne jz jnz ja jae jb jbe jg jge jl jle
		js jns ret nop prefetcht0 prefetcht1 prefetcht2 prefetchnta
		cmove cmovne cmovz cmovnz cmova cmovae cmovb cmovbe cmovg cmovge cmovl cmovle`)
	add(ISASSE2, `movaps movups movss movd movq movdqu movdqa movhlps movlhps movmskps pmovmskb
		addps subps mulps divps maxps minps andps andnps orps xorps unpcklps unpckhps shufps
		sqrtps rcpps rsqrtps cmpeqps cmpltps cmpleps cmpunordps cmpneqps cmpnltps cmpnleps cmpordps
		addss subss mulss divss maxss minss sqrtss cvtss2sd cvtsd2ss ucomiss comiss
		addpd subpd mulpd divpd
		cvtdq2ps cvtps2dq cvttps2dq cvtps2pd cvtpd2ps cvtsi2ss cvttss2si cvtss2si
		paddb paddw paddd paddq psubb psubw psubd psubq pmullw pmulhw pmulhuw pmuludq pmaddwd
		pand pandn por pxor pcmpeqb pcmpeqw pcmpeqd pcmpgtb pcmpgtw pcmpgtd
		packsswb packssdw packuswb punpcklbw punpcklwd punpckldq punpcklqdq
		punpckhbw punpckhwd punpckhdq punpckhqdq pminub pmaxub pminsw pmaxsw pavgb pavgw psadbw
		pshufd pshuflw pshufhw psrlw psrld psrlq psraw psrad psllw pslld psllq psrldq pslldq
		pinsrw pextrw lfence sfence mfence`)
	add(ISASSE3, `haddps hsubps addsubps movsldup movshdup movddup lddqu`)
	add(ISASSSE3, `pshufb phaddw phaddd phsubw phsubd pmaddubsw psignb psignw psignd pmulhrsw
		pabsb pabsw pabsd palignr`)
	add(ISASSE41, `pmovsxbw pmovsxbd pmovsxbq pmovsxwd pmovsxwq pmovsxdq
		pmovzxbw pmovzxbd pmovzxbq pmovzxwd pmovzxwq pmovzxdq
		pmulld pmuldq pcmpeqq packusdw pminsb pminsd pminuw pminud pmaxsb pmaxsd pmaxuw pmaxud
		ptest pblendvb blendvps blendps pblendw insertps extractps pinsrb pinsrd pinsrq
		pextrb pextrd pextrq roundps roundss roundpd dpps`)
	add(ISASSE42, `pcmpgtq crc32`)
	add(ISAPOPCNT, `popcnt`)
	return m
}()

// sseGateProblems returns every way code fails to be a legal SSE-tier kernel.
// It returns rather than failing, so the gate's own self-test can run it
// against violations.
func sseGateProblems(t testing.TB, name string, code []byte) []string {
	t.Helper()
	var bad []string
	add := func(f string, a ...any) { bad = append(bad, fmt.Sprintf(f, a...)) }

	declared, ok := KernelISA(code)
	if !ok {
		add("no ISA declaration at code[0] (DeclareISA): Map would treat it as an AVX2 kernel")
	}
	if declared&(ISAAVX2|ISAFMA|ISAF16C|ISAVNNI) != 0 {
		add("declares %v: an SSE-tier kernel may not need any VEX extension", declared)
	}
	ins := disasmWide(t, t.TempDir(), name, code)
	if len(ins) == 0 {
		add("objdump found no instructions")
	}
	for i, in := range ins {
		where := fmt.Sprintf("insn %d (% x) %q", i, in.bytes, in.text)
		if strings.HasPrefix(in.text, "(bad)") {
			add("%s: objdump cannot decode it", where)
			continue
		}
		// The opcode's first byte, past legacy prefixes and REX.
		j := 0
		for j < len(in.bytes) && isLegacyPrefix(in.bytes[j]) {
			j++
		}
		if j < len(in.bytes) && in.bytes[j]&0xF0 == 0x40 {
			j++
		}
		if j < len(in.bytes) {
			switch in.bytes[j] {
			case 0xC4, 0xC5:
				add("%s: VEX-encoded -- faults on a host with no AVX", where)
			case 0x62:
				add("%s: EVEX-encoded -- faults on a host with no AVX-512", where)
			}
		}
		fields := strings.Fields(in.text)
		mn := fields[0]
		ops := strings.Join(fields[1:], " ")
		if strings.HasPrefix(mn, "v") {
			add("%s: a VEX mnemonic", where)
		}
		if strings.Contains(ops, "ymm") || strings.Contains(ops, "zmm") {
			add("%s: a 256/512-bit register", where)
		}
		switch mn {
		case "push", "pop", "call", "enter", "leave", "pushf", "popf", "pushfq", "popfq":
			add("%s: moves RSP; the trampoline requires RSP identical on every path", where)
			continue
		}
		if strings.HasPrefix(mn, "rex") || mn == "data16" || mn == "addr32" || mn == "lock" || mn == "rep" {
			add("%s: a stray prefix objdump could not attach to an instruction", where)
			continue
		}
		isa, known := sseMnemonicISA[mn]
		if mn == "pextrw" && strings.Contains(ops, "PTR") {
			isa = ISASSE41 // the memory-destination form is SSE4.1's
		}
		if !known {
			add("%s: unclassified mnemonic %q -- add it to sseMnemonicISA with its extension", where, mn)
		} else if isa&^declared != 0 {
			add("%s: needs %v, which the kernel did not declare (%v)", where, isa, declared)
		}
		if dst := strings.SplitN(ops, ",", 2)[0]; writesFirstOperand(mn) && !strings.Contains(dst, "[") {
			switch dst {
			case "r14", "r14d", "r14w", "r14b", "rbp", "ebp", "bp", "bpl", "rsp", "esp", "sp", "spl":
				add("%s: writes %s (R14 is Go's g; RBP is not saved by callKernel; RSP must not move)", where, dst)
			}
		}
	}
	if len(ins) > 0 && ok {
		if first := ins[0]; len(first.bytes) != ISAMarkerLen || !strings.HasPrefix(first.text, "nop") {
			add("the first instruction is %q (% x), not the ISA declaration's NOP", first.text, first.bytes)
		}
	}
	return bad
}

// isLegacyPrefix reports the one-byte prefixes that may precede an opcode.
func isLegacyPrefix(b byte) bool {
	switch b {
	case 0x66, 0x67, 0xF0, 0xF2, 0xF3, 0x2E, 0x36, 0x3E, 0x26, 0x64, 0x65:
		return true
	}
	return false
}

// writesFirstOperand is false for the instructions whose first operand is
// only read.
func writesFirstOperand(mn string) bool {
	switch mn {
	case "cmp", "test", "ptest", "ucomiss", "comiss", "prefetcht0", "prefetcht1",
		"prefetcht2", "prefetchnta", "jmp", "ret", "nop":
		return false
	}
	return !strings.HasPrefix(mn, "j")
}

// requireSSEKernel fails t for every problem sseGateProblems finds in code.
// Every SSE-tier family gate calls it on every kernel it builds, at every
// shape it tests.
//
// Where objdump is missing it logs and returns rather than skipping, so the
// calling gate's execution half still runs on a host without objdump.
func requireSSEKernel(t testing.TB, name string, code []byte) {
	t.Helper()
	if _, err := exec.LookPath("objdump"); err != nil {
		t.Logf("%s: objdump not installed -- the VEX-leak half of this gate did not run here", name)
		return
	}
	for _, p := range sseGateProblems(t, name, code) {
		t.Errorf("%s: %s", name, p)
	}
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
