//go:build amd64

package cpu

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestSSEOpsMatchObjdump pins every legacy-SSE method in sse_ops.go against a
// disassembler: register-register with low and high registers (REX.R and
// REX.B), every addressing mode the memory path can take, the imm8 forms and
// the shift group.
//
// The expected text is written by hand beside each call, not derived from the
// opcode table, so a method wired to the wrong descriptor fails. A legacy
// encoding mistake is usually a valid instruction (a missing 0x66 turns PXOR
// into an MMX op, a wrong REX bit names another register), which disassembles
// cleanly into something else.
func TestSSEOpsMatchObjdump(t *testing.T) {
	if _, err := exec.LookPath("objdump"); err != nil {
		t.Skip("objdump not installed")
	}
	type tc struct {
		want string
		emit func(a *Buf)
	}
	var cases []tc
	add := func(want string, emit func(a *Buf)) { cases = append(cases, tc{want, emit}) }

	// rr adds a three-operand op in its no-copy shape (dst == src1), twice:
	// low registers, then high ones for REX.R and REX.B.
	rr := func(mn string, f func(a *Buf, d, s1, s2 Reg)) {
		add(mn+" xmm1,xmm2", func(a *Buf) { f(a, XMM1, XMM1, XMM2) })
		add(mn+" xmm9,xmm12", func(a *Buf) { f(a, XMM9, XMM9, XMM12) })
		add(mn+" xmm3,xmm15", func(a *Buf) { f(a, XMM3, XMM3, XMM15) })
	}
	// un adds a unary op, dst = f(src).
	un := func(mn string, f func(a *Buf, d, s Reg)) {
		add(mn+" xmm1,xmm2", func(a *Buf) { f(a, XMM1, XMM2) })
		add(mn+" xmm10,xmm7", func(a *Buf) { f(a, XMM10, XMM7) })
		add(mn+" xmm4,xmm13", func(a *Buf) { f(a, XMM4, XMM13) })
	}
	rri := func(mn string, imm byte, f func(a *Buf, d, s1, s2 Reg, imm byte)) {
		add(fmt.Sprintf("%s xmm1,xmm2,0x%x", mn, imm), func(a *Buf) { f(a, XMM1, XMM1, XMM2, imm) })
		add(fmt.Sprintf("%s xmm11,xmm14,0x%x", mn, imm), func(a *Buf) { f(a, XMM11, XMM11, XMM14, imm) })
	}
	uni := func(mn string, imm byte, f func(a *Buf, d, s Reg, imm byte)) {
		add(fmt.Sprintf("%s xmm1,xmm2,0x%x", mn, imm), func(a *Buf) { f(a, XMM1, XMM2, imm) })
		add(fmt.Sprintf("%s xmm12,xmm5,0x%x", mn, imm), func(a *Buf) { f(a, XMM12, XMM5, imm) })
	}
	sh := func(mn string, f func(a *Buf, d, s Reg, imm byte)) {
		add(mn+" xmm1,0x5", func(a *Buf) { f(a, XMM1, XMM1, 5) })
		add(mn+" xmm13,0x7", func(a *Buf) { f(a, XMM13, XMM13, 7) })
	}

	// ---- the memory path, every addressing-mode quirk modrmMem handles ----
	modes := []struct {
		m    Mem
		text string
	}{
		{At(RAX, 0), "[rax]"},
		{At(RSI, 0x20), "[rsi+0x20]"},
		{At(RDX, -8), "[rdx-0x8]"},
		{At(RCX, 0x1000), "[rcx+0x1000]"},
		{At(R12, 0), "[r12]"},     // needs a SIB byte
		{At(R13, 0), "[r13+0x0]"}, // mod=00 would be RIP-relative
		{At(RSP, 4), "[rsp+0x4]"}, // SIB, no index
		{At(RBP, 0), "[rbp+0x0]"},
		{At(R9, 0x40), "[r9+0x40]"},
		{Idx(RAX, RCX, 4, 8), "[rax+rcx*4+0x8]"},
		{Idx(R8, R9, 8, 0x100), "[r8+r9*8+0x100]"},
		{Idx(RDI, R11, 2, 0), "[rdi+r11*2]"},
	}
	for _, md := range modes {
		m, txt := md.m, md.text
		add("movups xmm0,XMMWORD PTR "+txt, func(a *Buf) { a.MOVUPSLoad(XMM0, m) })
		add("movups XMMWORD PTR "+txt+",xmm9", func(a *Buf) { a.MOVUPSStore(m, XMM9) })
		add("movdqu xmm14,XMMWORD PTR "+txt, func(a *Buf) { a.MOVDQULoad(XMM14, m) })
		add("movd xmm6,DWORD PTR "+txt, func(a *Buf) { a.MOVDLoad(XMM6, m) })
		add("pinsrw xmm10,WORD PTR "+txt+",0x2", func(a *Buf) { a.PINSRWLoad(XMM10, XMM10, m, 2) })
	}

	// ---- moves ----
	add("movaps xmm13,xmm14", func(a *Buf) { a.MOVAPS(XMM13, XMM14) })
	add("movups xmm3,XMMWORD PTR [rsi+0x10]", func(a *Buf) { a.MOVUPSLoad(XMM3, At(RSI, 16)) })
	add("movdqu XMMWORD PTR [rcx+0x10],xmm2", func(a *Buf) { a.MOVDQUStore(At(RCX, 16), XMM2) })
	add("lddqu xmm5,[rsi]", func(a *Buf) { a.LDDQU(XMM5, At(RSI, 0)) })
	add("movss xmm1,DWORD PTR [rsi+0x4]", func(a *Buf) { a.MOVSSLoad(XMM1, At(RSI, 4)) })
	add("movss DWORD PTR [rcx+0x8],xmm11", func(a *Buf) { a.MOVSSStore(At(RCX, 8), XMM11) })
	rr("movss", func(a *Buf, d, s1, s2 Reg) { a.MOVSS(d, s1, s2) })
	add("movd DWORD PTR [rcx],xmm12", func(a *Buf) { a.MOVDStore(At(RCX, 0), XMM12) })
	add("movd xmm3,eax", func(a *Buf) { a.MOVDToX(XMM3, RAX) })
	add("movd xmm11,r10d", func(a *Buf) { a.MOVDToX(XMM11, R10) })
	add("movd ecx,xmm2", func(a *Buf) { a.MOVDFromX(RCX, XMM2) })
	add("movd r9d,xmm14", func(a *Buf) { a.MOVDFromX(R9, XMM14) })
	add("movq xmm7,QWORD PTR [rdx+0x18]", func(a *Buf) { a.MOVQLoad(XMM7, At(RDX, 24)) })
	add("movq xmm8,QWORD PTR [r11]", func(a *Buf) { a.MOVQLoad(XMM8, At(R11, 0)) })
	add("movq QWORD PTR [rcx+0x8],xmm1", func(a *Buf) { a.MOVQStore(At(RCX, 8), XMM1) })
	add("movq QWORD PTR [r8],xmm10", func(a *Buf) { a.MOVQStore(At(R8, 0), XMM10) })
	add("movq xmm2,rax", func(a *Buf) { a.MOVQToX(XMM2, RAX) })
	add("movq xmm12,r15", func(a *Buf) { a.MOVQToX(XMM12, R15) })
	add("movq rdx,xmm3", func(a *Buf) { a.MOVQFromX(RDX, XMM3) })
	add("movq r13,xmm9", func(a *Buf) { a.MOVQFromX(R13, XMM9) })
	un("movq", func(a *Buf, d, s Reg) { a.MOVQX(d, s) })
	rr("movhlps", func(a *Buf, d, s1, s2 Reg) { a.MOVHLPS(d, s1, s2) })
	rr("movlhps", func(a *Buf, d, s1, s2 Reg) { a.MOVLHPS(d, s1, s2) })
	un("movsldup", func(a *Buf, d, s Reg) { a.MOVSLDUP(d, s) })
	un("movshdup", func(a *Buf, d, s Reg) { a.MOVSHDUP(d, s) })
	un("movddup", func(a *Buf, d, s Reg) { a.MOVDDUP(d, s) })
	add("movmskps eax,xmm1", func(a *Buf) { a.MOVMSKPS(RAX, XMM1) })
	add("movmskps r10d,xmm13", func(a *Buf) { a.MOVMSKPS(R10, XMM13) })
	add("pmovmskb edx,xmm4", func(a *Buf) { a.PMOVMSKB(RDX, XMM4) })
	add("pmovmskb r8d,xmm11", func(a *Buf) { a.PMOVMSKB(R8, XMM11) })
	add("pinsrw xmm1,eax,0x3", func(a *Buf) { a.PINSRW(XMM1, XMM1, RAX, 3) })
	add("pinsrw xmm9,r12d,0x7", func(a *Buf) { a.PINSRW(XMM9, XMM9, R12, 7) })
	add("pextrw eax,xmm1,0x3", func(a *Buf) { a.PEXTRW(RAX, XMM1, 3) })
	add("pextrw r11d,xmm10,0x1", func(a *Buf) { a.PEXTRW(R11, XMM10, 1) })
	add("pinsrb xmm2,ecx,0x9", func(a *Buf) { a.PINSRB(XMM2, XMM2, RCX, 9) })
	add("pinsrd xmm3,edx,0x2", func(a *Buf) { a.PINSRD(XMM3, XMM3, RDX, 2) })
	add("pinsrd xmm13,r9d,0x1", func(a *Buf) { a.PINSRD(XMM13, XMM13, R9, 1) })
	add("pinsrd xmm4,DWORD PTR [rsi+0x8],0x3", func(a *Buf) { a.PINSRDLoad(XMM4, XMM4, At(RSI, 8), 3) })
	add("pextrb eax,xmm5,0xf", func(a *Buf) { a.PEXTRB(RAX, XMM5, 15) })
	add("pextrd ecx,xmm6,0x2", func(a *Buf) { a.PEXTRD(RCX, XMM6, 2) })
	add("pextrd r14d,xmm15,0x3", func(a *Buf) { a.PEXTRD(R14, XMM15, 3) })
	add("pextrd DWORD PTR [rcx+0x4],xmm7,0x1", func(a *Buf) { a.PEXTRDStore(At(RCX, 4), XMM7, 1) })
	rri("insertps", 0x10, func(a *Buf, d, s1, s2 Reg, imm byte) { a.INSERTPS(d, s1, s2, imm) })

	// ---- packed float ----
	rr("addps", func(a *Buf, d, s1, s2 Reg) { a.ADDPS(d, s1, s2) })
	rr("subps", func(a *Buf, d, s1, s2 Reg) { a.SUBPS(d, s1, s2) })
	rr("mulps", func(a *Buf, d, s1, s2 Reg) { a.MULPS(d, s1, s2) })
	rr("divps", func(a *Buf, d, s1, s2 Reg) { a.DIVPS(d, s1, s2) })
	rr("maxps", func(a *Buf, d, s1, s2 Reg) { a.MAXPS(d, s1, s2) })
	rr("minps", func(a *Buf, d, s1, s2 Reg) { a.MINPS(d, s1, s2) })
	rr("andps", func(a *Buf, d, s1, s2 Reg) { a.ANDPS(d, s1, s2) })
	rr("andnps", func(a *Buf, d, s1, s2 Reg) { a.ANDNPS(d, s1, s2) })
	rr("orps", func(a *Buf, d, s1, s2 Reg) { a.ORPS(d, s1, s2) })
	rr("xorps", func(a *Buf, d, s1, s2 Reg) { a.XORPS(d, s1, s2) })
	rr("unpcklps", func(a *Buf, d, s1, s2 Reg) { a.UNPCKLPS(d, s1, s2) })
	rr("unpckhps", func(a *Buf, d, s1, s2 Reg) { a.UNPCKHPS(d, s1, s2) })
	rr("haddps", func(a *Buf, d, s1, s2 Reg) { a.HADDPS(d, s1, s2) })
	rr("hsubps", func(a *Buf, d, s1, s2 Reg) { a.HSUBPS(d, s1, s2) })
	rr("addsubps", func(a *Buf, d, s1, s2 Reg) { a.ADDSUBPS(d, s1, s2) })
	// objdump renders the eight legacy CMPPS predicates as pseudo-mnemonics.
	for imm, mn := range []string{"cmpeqps", "cmpltps", "cmpleps", "cmpunordps", "cmpneqps", "cmpnltps", "cmpnleps", "cmpordps"} {
		imm := byte(imm)
		add(mn+" xmm1,xmm2", func(a *Buf) { a.CMPPS(XMM1, XMM1, XMM2, imm) })
	}
	add("cmpltps xmm10,xmm11", func(a *Buf) { a.CMPPS(XMM10, XMM10, XMM11, 1) })
	rri("shufps", 0x4e, func(a *Buf, d, s1, s2 Reg, imm byte) { a.SHUFPS(d, s1, s2, imm) })
	rri("shufps", 0x0, func(a *Buf, d, s1, s2 Reg, imm byte) { a.SHUFPS(d, s1, s2, imm) })
	rri("blendps", 0x1, func(a *Buf, d, s1, s2 Reg, imm byte) { a.BLENDPS(d, s1, s2, imm) })
	add("blendvps xmm1,xmm2,xmm0", func(a *Buf) { a.BLENDVPS(XMM1, XMM1, XMM2) })
	add("blendvps xmm9,xmm12,xmm0", func(a *Buf) { a.BLENDVPS(XMM9, XMM9, XMM12) })
	un("sqrtps", func(a *Buf, d, s Reg) { a.SQRTPS(d, s) })
	un("rcpps", func(a *Buf, d, s Reg) { a.RCPPS(d, s) })
	un("rsqrtps", func(a *Buf, d, s Reg) { a.RSQRTPS(d, s) })
	uni("roundps", 0x1, func(a *Buf, d, s Reg, imm byte) { a.ROUNDPS(d, s, imm) })

	// ---- scalar float ----
	rr("addss", func(a *Buf, d, s1, s2 Reg) { a.ADDSS(d, s1, s2) })
	rr("subss", func(a *Buf, d, s1, s2 Reg) { a.SUBSS(d, s1, s2) })
	rr("mulss", func(a *Buf, d, s1, s2 Reg) { a.MULSS(d, s1, s2) })
	rr("divss", func(a *Buf, d, s1, s2 Reg) { a.DIVSS(d, s1, s2) })
	rr("maxss", func(a *Buf, d, s1, s2 Reg) { a.MAXSS(d, s1, s2) })
	rr("minss", func(a *Buf, d, s1, s2 Reg) { a.MINSS(d, s1, s2) })
	rr("sqrtss", func(a *Buf, d, s1, s2 Reg) { a.SQRTSS(d, s1, s2) })
	rr("cvtss2sd", func(a *Buf, d, s1, s2 Reg) { a.CVTSS2SD(d, s1, s2) })
	rr("cvtsd2ss", func(a *Buf, d, s1, s2 Reg) { a.CVTSD2SS(d, s1, s2) })
	rri("roundss", 0x3, func(a *Buf, d, s1, s2 Reg, imm byte) { a.ROUNDSS(d, s1, s2, imm) })
	add("addss xmm1,DWORD PTR [rbx+0x1c]", func(a *Buf) { a.ADDSSMem(XMM1, XMM1, At(RBX, 28)) })
	add("subss xmm9,DWORD PTR [r10]", func(a *Buf) { a.SUBSSMem(XMM9, XMM9, At(R10, 0)) })
	add("mulss xmm2,DWORD PTR [rbx+0x4]", func(a *Buf) { a.MULSSMem(XMM2, XMM2, At(RBX, 4)) })
	add("divss xmm3,DWORD PTR [rsi]", func(a *Buf) { a.DIVSSMem(XMM3, XMM3, At(RSI, 0)) })
	add("maxss xmm4,DWORD PTR [rsi+0x20]", func(a *Buf) { a.MAXSSMem(XMM4, XMM4, At(RSI, 32)) })
	add("minss xmm15,DWORD PTR [r12]", func(a *Buf) { a.MINSSMem(XMM15, XMM15, At(R12, 0)) })

	// ---- conversions ----
	un("cvtdq2ps", func(a *Buf, d, s Reg) { a.CVTDQ2PS(d, s) })
	un("cvtps2dq", func(a *Buf, d, s Reg) { a.CVTPS2DQ(d, s) })
	un("cvttps2dq", func(a *Buf, d, s Reg) { a.CVTTPS2DQ(d, s) })
	un("cvtps2pd", func(a *Buf, d, s Reg) { a.CVTPS2PD(d, s) })
	un("cvtpd2ps", func(a *Buf, d, s Reg) { a.CVTPD2PS(d, s) })
	add("cvtsi2ss xmm1,rax", func(a *Buf) { a.CVTSI2SS(XMM1, XMM1, RAX) })
	add("cvtsi2ss xmm12,r9", func(a *Buf) { a.CVTSI2SS(XMM12, XMM12, R9) })
	add("cvttss2si rax,xmm1", func(a *Buf) { a.CVTTSS2SI(RAX, XMM1) })
	add("cvttss2si r11,xmm13", func(a *Buf) { a.CVTTSS2SI(R11, XMM13) })
	add("cvtss2si rdx,xmm2", func(a *Buf) { a.CVTSS2SI(RDX, XMM2) })

	// ---- integer, SSE2 ----
	for _, c := range []struct {
		mn string
		f  func(a *Buf, d, s1, s2 Reg)
	}{
		{"paddb", (*Buf).PADDB}, {"paddw", (*Buf).PADDW}, {"paddd", (*Buf).PADDD}, {"paddq", (*Buf).PADDQ},
		{"psubb", (*Buf).PSUBB}, {"psubw", (*Buf).PSUBW}, {"psubd", (*Buf).PSUBD}, {"psubq", (*Buf).PSUBQ},
		{"pmullw", (*Buf).PMULLW}, {"pmulhw", (*Buf).PMULHW}, {"pmulhuw", (*Buf).PMULHUW}, {"pmuludq", (*Buf).PMULUDQ},
		{"pmaddwd", (*Buf).PMADDWD},
		{"pand", (*Buf).PAND}, {"pandn", (*Buf).PANDN}, {"por", (*Buf).POR}, {"pxor", (*Buf).PXOR},
		{"pcmpeqb", (*Buf).PCMPEQB}, {"pcmpeqw", (*Buf).PCMPEQW}, {"pcmpeqd", (*Buf).PCMPEQD},
		{"pcmpgtb", (*Buf).PCMPGTB}, {"pcmpgtw", (*Buf).PCMPGTW}, {"pcmpgtd", (*Buf).PCMPGTD},
		{"packsswb", (*Buf).PACKSSWB}, {"packssdw", (*Buf).PACKSSDW}, {"packuswb", (*Buf).PACKUSWB},
		{"punpcklbw", (*Buf).PUNPCKLBW}, {"punpcklwd", (*Buf).PUNPCKLWD}, {"punpckldq", (*Buf).PUNPCKLDQ},
		{"punpcklqdq", (*Buf).PUNPCKLQDQ}, {"punpckhbw", (*Buf).PUNPCKHBW}, {"punpckhwd", (*Buf).PUNPCKHWD},
		{"punpckhdq", (*Buf).PUNPCKHDQ}, {"punpckhqdq", (*Buf).PUNPCKHQDQ},
		{"pminub", (*Buf).PMINUB}, {"pmaxub", (*Buf).PMAXUB}, {"pminsw", (*Buf).PMINSW}, {"pmaxsw", (*Buf).PMAXSW},
		{"pavgb", (*Buf).PAVGB}, {"pavgw", (*Buf).PAVGW}, {"psadbw", (*Buf).PSADBW},
		{"psrlw", (*Buf).PSRLWBy}, {"psrld", (*Buf).PSRLDBy}, {"psrlq", (*Buf).PSRLQBy},
		{"psraw", (*Buf).PSRAWBy}, {"psrad", (*Buf).PSRADBy},
		{"psllw", (*Buf).PSLLWBy}, {"pslld", (*Buf).PSLLDBy}, {"psllq", (*Buf).PSLLQBy},
	} {
		rr(c.mn, c.f)
	}
	uni("pshufd", 0xb1, func(a *Buf, d, s Reg, imm byte) { a.PSHUFD(d, s, imm) })
	uni("pshufd", 0x0, func(a *Buf, d, s Reg, imm byte) { a.PSHUFD(d, s, imm) })
	uni("pshuflw", 0x1b, func(a *Buf, d, s Reg, imm byte) { a.PSHUFLW(d, s, imm) })
	uni("pshufhw", 0x1b, func(a *Buf, d, s Reg, imm byte) { a.PSHUFHW(d, s, imm) })
	sh("psrlw", (*Buf).PSRLW)
	sh("psraw", (*Buf).PSRAW)
	sh("psllw", (*Buf).PSLLW)
	sh("psrld", (*Buf).PSRLD)
	sh("psrad", (*Buf).PSRAD)
	sh("pslld", (*Buf).PSLLD)
	sh("psrlq", (*Buf).PSRLQ)
	sh("psllq", (*Buf).PSLLQ)
	sh("psrldq", (*Buf).PSRLDQ)
	sh("pslldq", (*Buf).PSLLDQ)

	// ---- SSSE3 ----
	for _, c := range []struct {
		mn string
		f  func(a *Buf, d, s1, s2 Reg)
	}{
		{"pshufb", (*Buf).PSHUFB}, {"phaddw", (*Buf).PHADDW}, {"phaddd", (*Buf).PHADDD},
		{"phsubw", (*Buf).PHSUBW}, {"phsubd", (*Buf).PHSUBD}, {"pmaddubsw", (*Buf).PMADDUBSW},
		{"psignb", (*Buf).PSIGNB}, {"psignw", (*Buf).PSIGNW}, {"psignd", (*Buf).PSIGND},
		{"pmulhrsw", (*Buf).PMULHRSW},
	} {
		rr(c.mn, c.f)
	}
	un("pabsb", (*Buf).PABSB)
	un("pabsw", (*Buf).PABSW)
	un("pabsd", (*Buf).PABSD)
	rri("palignr", 0x4, func(a *Buf, d, s1, s2 Reg, imm byte) { a.PALIGNR(d, s1, s2, imm) })

	// ---- SSE4.1 / SSE4.2 ----
	for _, c := range []struct {
		mn   string
		f    func(a *Buf, d, s Reg)
		ld   func(a *Buf, d Reg, m Mem)
		size string
	}{
		{"pmovsxbw", (*Buf).PMOVSXBW, (*Buf).PMOVSXBWLoad, "QWORD"},
		{"pmovsxbd", (*Buf).PMOVSXBD, (*Buf).PMOVSXBDLoad, "DWORD"},
		{"pmovsxbq", (*Buf).PMOVSXBQ, (*Buf).PMOVSXBQLoad, "WORD"},
		{"pmovsxwd", (*Buf).PMOVSXWD, (*Buf).PMOVSXWDLoad, "QWORD"},
		{"pmovsxwq", (*Buf).PMOVSXWQ, (*Buf).PMOVSXWQLoad, "DWORD"},
		{"pmovsxdq", (*Buf).PMOVSXDQ, (*Buf).PMOVSXDQLoad, "QWORD"},
		{"pmovzxbw", (*Buf).PMOVZXBW, (*Buf).PMOVZXBWLoad, "QWORD"},
		{"pmovzxbd", (*Buf).PMOVZXBD, (*Buf).PMOVZXBDLoad, "DWORD"},
		{"pmovzxbq", (*Buf).PMOVZXBQ, (*Buf).PMOVZXBQLoad, "WORD"},
		{"pmovzxwd", (*Buf).PMOVZXWD, (*Buf).PMOVZXWDLoad, "QWORD"},
		{"pmovzxwq", (*Buf).PMOVZXWQ, (*Buf).PMOVZXWQLoad, "DWORD"},
		{"pmovzxdq", (*Buf).PMOVZXDQ, (*Buf).PMOVZXDQLoad, "QWORD"},
	} {
		un(c.mn, c.f)
		ld, size, mn := c.ld, c.size, c.mn
		add(mn+" xmm3,"+size+" PTR [rsi+0xc]", func(a *Buf) { ld(a, XMM3, At(RSI, 12)) })
		add(mn+" xmm11,"+size+" PTR [r8+rax*1]", func(a *Buf) { ld(a, XMM11, Idx(R8, RAX, 1, 0)) })
	}
	for _, c := range []struct {
		mn string
		f  func(a *Buf, d, s1, s2 Reg)
	}{
		{"pmulld", (*Buf).PMULLD}, {"pmuldq", (*Buf).PMULDQ}, {"pcmpeqq", (*Buf).PCMPEQQ},
		{"packusdw", (*Buf).PACKUSDW},
		{"pminsb", (*Buf).PMINSB}, {"pminsd", (*Buf).PMINSD}, {"pminuw", (*Buf).PMINUW}, {"pminud", (*Buf).PMINUD},
		{"pmaxsb", (*Buf).PMAXSB}, {"pmaxsd", (*Buf).PMAXSD}, {"pmaxuw", (*Buf).PMAXUW}, {"pmaxud", (*Buf).PMAXUD},
		{"pcmpgtq", (*Buf).PCMPGTQ},
	} {
		rr(c.mn, c.f)
	}
	add("ptest xmm1,xmm2", func(a *Buf) { a.PTEST(XMM1, XMM2) })
	add("ptest xmm9,xmm14", func(a *Buf) { a.PTEST(XMM9, XMM14) })
	add("pblendvb xmm3,xmm4,xmm0", func(a *Buf) { a.PBLENDVB(XMM3, XMM3, XMM4) })
	add("pblendvb xmm10,xmm12,xmm0", func(a *Buf) { a.PBLENDVB(XMM10, XMM10, XMM12) })
	rri("pblendw", 0xf0, func(a *Buf, d, s1, s2 Reg, imm byte) { a.PBLENDW(d, s1, s2, imm) })
	add("popcnt rax,rcx", func(a *Buf) { a.POPCNT(RAX, RCX) })
	add("popcnt r10,r15", func(a *Buf) { a.POPCNT(R10, R15) })

	// ---- the lowering copies, where the op is NOT in its no-copy shape ----
	add("movaps xmm1,xmm2", func(a *Buf) { a.SUBPS(XMM1, XMM2, XMM3) })
	add("subps xmm1,xmm3", func(*Buf) {}) // the second half of the case above
	add("movaps xmm8,xmm3", func(a *Buf) { a.PSRLD(XMM8, XMM3, 4) })
	add("psrld xmm8,0x4", func(*Buf) {})
	add("movaps xmm5,xmm6", func(a *Buf) { a.PINSRWLoad(XMM5, XMM6, At(RSI, 2), 0) })
	add("pinsrw xmm5,WORD PTR [rsi+0x2],0x0", func(*Buf) {})
	add("movaps xmm2,xmm4", func(a *Buf) { a.MULSSMem(XMM2, XMM4, At(RBX, 0)) })
	add("mulss xmm2,DWORD PTR [rbx]", func(*Buf) {})

	// One buffer, one objdump. Each case above emits exactly one instruction
	// (the lowering pairs split it across two rows), so the disassembly lines
	// up with the table one to one, and a wrong length shows up as every row
	// after it disagreeing -- the first mismatch is the one to read.
	var a Buf
	for _, c := range cases {
		c.emit(&a)
	}
	got := disasmWide(t, t.TempDir(), "sseops", a.Bytes())
	if len(got) != len(cases) {
		t.Errorf("objdump found %d instructions for %d cases", len(got), len(cases))
	}
	bad := 0
	for i := 0; i < len(cases) && i < len(got); i++ {
		if got[i].text != cases[i].want {
			t.Errorf("case %d: emitted % x\n  objdump: %q\n  want:    %q", i, got[i].bytes, got[i].text, cases[i].want)
			if bad++; bad > 10 {
				t.Fatal("too many mismatches; the first one is the one to read")
			}
		}
	}
	t.Logf("%d legacy-SSE encodings pinned against objdump", len(cases))
}

// insn is one disassembled instruction: its bytes and its normalised text.
type insn struct {
	bytes []byte
	text  string
}

// disasmWide disassembles code with objdump, one row per instruction and the
// whole encoding on that row (--insn-width=16, so a long instruction's bytes
// are not wrapped onto a continuation line). It keeps "(bad)" rows, because a
// gate must see them.
func disasmWide(t testing.TB, dir, name string, code []byte) []insn {
	t.Helper()
	p := filepath.Join(dir, name+".bin")
	if err := os.WriteFile(p, code, 0o600); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command("objdump", "-D", "-b", "binary", "-m", "i386:x86-64",
		"-M", "intel", "--insn-width=16", p).Output()
	if err != nil {
		t.Fatalf("objdump: %v", err)
	}
	var got []insn
	for _, ln := range strings.Split(string(out), "\n") {
		f := strings.SplitN(ln, "\t", 3)
		if len(f) < 3 || !strings.HasSuffix(strings.TrimSpace(f[0]), ":") {
			continue
		}
		var bs []byte
		for _, h := range strings.Fields(f[1]) {
			var v byte
			if _, err := fmt.Sscanf(h, "%02x", &v); err != nil {
				t.Fatalf("objdump byte %q in %q: %v", h, ln, err)
			}
			bs = append(bs, v)
		}
		txt := strings.Join(strings.Fields(f[2]), " ")
		if i := strings.Index(txt, " #"); i >= 0 {
			txt = strings.TrimSpace(txt[:i]) // objdump's trailing comments
		}
		got = append(got, insn{bytes: bs, text: txt})
	}
	return got
}

// TestSSELoweringRefusesWhatItCannotServe extends TestSSEThreeOperandLowering
// to the named ops: every non-commutative one must panic when dst aliases src2
// (a copy there would compute src1 op src1), and the commutative ones must
// swap rather than copy.
//
// The list is the ops whose operand order matters: MAXPS/MINPS return their
// second operand on a NaN and on a +0/-0 tie (the exp clamp and softmax fold
// rely on it), PSHUFB's destination is the table, PMADDUBSW's is the unsigned
// operand, and the scalar SS ops take their upper lanes from src1.
func TestSSELoweringRefusesWhatItCannotServe(t *testing.T) {
	nonComm := map[string]func(a *Buf, d, s1, s2 Reg){
		"maxps": (*Buf).MAXPS, "minps": (*Buf).MINPS, "subps": (*Buf).SUBPS, "divps": (*Buf).DIVPS,
		"andnps": (*Buf).ANDNPS, "unpcklps": (*Buf).UNPCKLPS, "haddps": (*Buf).HADDPS,
		"addsubps": (*Buf).ADDSUBPS, "addss": (*Buf).ADDSS, "mulss": (*Buf).MULSS, "movss": (*Buf).MOVSS,
		"pshufb": (*Buf).PSHUFB, "pmaddubsw": (*Buf).PMADDUBSW, "psignb": (*Buf).PSIGNB,
		"psubd": (*Buf).PSUBD, "pandn": (*Buf).PANDN, "pcmpgtd": (*Buf).PCMPGTD,
		"packssdw": (*Buf).PACKSSDW, "punpcklwd": (*Buf).PUNPCKLWD, "phaddd": (*Buf).PHADDD,
		"movhlps": (*Buf).MOVHLPS, "psrldby": (*Buf).PSRLDBy, "packusdw": (*Buf).PACKUSDW,
		"shufps":  func(a *Buf, d, s1, s2 Reg) { a.SHUFPS(d, s1, s2, 0x4E) },
		"cmpps":   func(a *Buf, d, s1, s2 Reg) { a.CMPPS(d, s1, s2, 1) },
		"blendps": func(a *Buf, d, s1, s2 Reg) { a.BLENDPS(d, s1, s2, 1) },
	}
	for name, f := range nonComm {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: dst == src2 did NOT panic; a copy computes src1 op src1 and a swap reverses the operands", name)
				}
			}()
			var a Buf
			f(&a, XMM1, XMM2, XMM1)
		}()
	}
	// The commutative ones swap: one instruction, no MOVAPS.
	for name, f := range map[string]func(a *Buf, d, s1, s2 Reg){
		"addps": (*Buf).ADDPS, "mulps": (*Buf).MULPS, "pxor": (*Buf).PXOR, "paddd": (*Buf).PADDD,
		"pmaddwd": (*Buf).PMADDWD, "pmulld": (*Buf).PMULLD, "pcmpeqd": (*Buf).PCMPEQD,
	} {
		var a Buf
		f(&a, XMM1, XMM2, XMM1)
		var b Buf
		f(&b, XMM1, XMM1, XMM2)
		if !bytes.Equal(a.Bytes(), b.Bytes()) {
			t.Errorf("%s: dst == src2 emitted % x, want the swapped single instruction % x", name, a.Bytes(), b.Bytes())
		}
	}
	// And the implicit-XMM0 blends refuse to overwrite their own mask.
	for name, f := range map[string]func(){
		"blendvps": func() { var a Buf; a.BLENDVPS(XMM0, XMM1, XMM2) },
		"pblendvb": func() { var a Buf; a.PBLENDVB(XMM0, XMM1, XMM2) },
		"cmpps 8":  func() { var a Buf; a.CMPPS(XMM1, XMM1, XMM2, 8) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s did not panic", name)
				}
			}()
			f()
		}()
	}
}
