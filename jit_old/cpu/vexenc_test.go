//go:build amd64

package cpu

import (
	"testing"
)

// TestVEXEncodingMatchesObjdump pins the instructions the tail loops and the
// rotary kernel added against a disassembler, for TestSSEEncodingMatchesObjdump's
// reason: a wrong prefix or map is often a valid instruction that computes
// something else.
func TestVEXEncodingMatchesObjdump(t *testing.T) {
	cases := []struct {
		name string
		emit func(a *Buf)
		want string
	}{
		{"testq", func(a *Buf) { a.TESTQ(R10, R10) }, "test r10,r10"},
		{"testq lo", func(a *Buf) { a.TESTQ(RAX, RDX) }, "test rax,rdx"},
		{"vmovss load", func(a *Buf) { a.VMOVSSLoad(Y4, At(R11, 12)) }, "vmovss xmm4,DWORD PTR [r11+0xc]"},
		{"vmovss load hi", func(a *Buf) { a.VMOVSSLoad(Y13, At(RCX, 0)) }, "vmovss xmm13,DWORD PTR [rcx]"},
		{"vmovsldup", func(a *Buf) { a.VMOVSLDUP(Y1, Y9) }, "vmovsldup ymm1,ymm9"},
		{"vmovshdup", func(a *Buf) { a.VMOVSHDUP(Y10, Y2) }, "vmovshdup ymm10,ymm2"},
		{"vaddsubps", func(a *Buf) { a.VADDSUBPS(Y0, Y1, Y12) }, "vaddsubps ymm0,ymm1,ymm12"},
		{"vfnmadd231ps", func(a *Buf) { a.VFNMADD231PS(Y3, Y4, Y11) }, "vfnmadd231ps ymm3,ymm4,ymm11"},
		{"vpermilps", func(a *Buf) { a.VPERMILPS(Y5, Y14, 0xB1) }, "vpermilps ymm5,ymm14,0xb1"},
		{"vpinsrw load", func(a *Buf) { a.VPINSRWLoad(Y4, Y4, At(RDX, 14), 0) }, "vpinsrw xmm4,xmm4,WORD PTR [rdx+0xe],0x0"},
		{"vpinsrw load hi", func(a *Buf) { a.VPINSRWLoad(Y14, Y14, At(R11, 0), 0) }, "vpinsrw xmm14,xmm14,WORD PTR [r11],0x0"},
		{"vpmovzxwd", func(a *Buf) { a.VPMOVZXWD(Y5, At(RDX, 16)) }, "vpmovzxwd ymm5,XMMWORD PTR [rdx+0x10]"},
		{"vpsrlvd", func(a *Buf) { a.VPSRLVD(Y2, Y2, Y3) }, "vpsrlvd ymm2,ymm2,ymm3"},
		{"vpsrlvd hi", func(a *Buf) { a.VPSRLVD(Y12, Y9, Y14) }, "vpsrlvd ymm12,ymm9,ymm14"},
		{"and imm8", func(a *Buf) { a.ANDimm8(R12, 7) }, "and r12,0x7"},
		{"and imm8 lo", func(a *Buf) { a.ANDimm8(RAX, 3) }, "and rax,0x3"},
		{"vcvtph2ps reg", func(a *Buf) { a.VCVTPH2PSReg(Y14, Y14) }, "vcvtph2ps ymm14,xmm14"},
		// VPSRAD is the rotary table's quadrant mask, and it shares a base word
		// with VPSRLD and VPSLLD; the opcode extension is the only difference,
		// so a transposition is a valid instruction with the wrong shift kind.
		{"vpsrad", func(a *Buf) { a.VPSRAD(Y15, Y12, 31) }, "vpsrad ymm15,ymm12,0x1f"},
		{"vpsrad lo", func(a *Buf) { a.VPSRAD(Y2, Y3, 1) }, "vpsrad ymm2,ymm3,0x1"},
		{"vpsrld beside it", func(a *Buf) { a.VPSRLD(Y2, Y3, 1) }, "vpsrld ymm2,ymm3,0x1"},
		{"vpslld beside it", func(a *Buf) { a.VPSLLD(Y2, Y3, 1) }, "vpslld ymm2,ymm3,0x1"},
		// The image path's stores and its legacy additions (imgproc_sse.go),
		// each with a high register or an index where a REX/VEX bit shows.
		{"vpunpckldq x", func(a *Buf) { a.VPUNPCKLDQx(Y1, Y1, Y5) }, "vpunpckldq xmm1,xmm1,xmm5"},
		{"vpunpckldq x hi", func(a *Buf) { a.VPUNPCKLDQx(Y9, Y12, Y13) }, "vpunpckldq xmm9,xmm12,xmm13"},
		{"vmovq store", func(a *Buf) { a.VMOVQStore(At(RCX, 0), Y1) }, "vmovq QWORD PTR [rcx],xmm1"},
		{"vmovq store hi", func(a *Buf) { a.VMOVQStore(At(R9, 8), Y12) }, "vmovq QWORD PTR [r9+0x8],xmm12"},
		{"vpextrb store", func(a *Buf) { a.VPEXTRBStore(At(RCX, 0), Y1, 0) }, "vpextrb BYTE PTR [rcx],xmm1,0x0"},
		{"vpextrb store hi", func(a *Buf) { a.VPEXTRBStore(At(R11, 3), Y10, 2) }, "vpextrb BYTE PTR [r11+0x3],xmm10,0x2"},
		{"addpd", func(a *Buf) { a.ADDPD(Y1, Y1, Y2) }, "addpd xmm1,xmm2"},
		{"subpd hi", func(a *Buf) { a.SUBPD(Y9, Y9, Y12) }, "subpd xmm9,xmm12"},
		{"mulpd", func(a *Buf) { a.MULPD(Y3, Y3, Y14) }, "mulpd xmm3,xmm14"},
		{"divpd", func(a *Buf) { a.DIVPD(Y10, Y10, Y4) }, "divpd xmm10,xmm4"},
		{"roundpd", func(a *Buf) { a.ROUNDPD(Y2, Y11, 1) }, "roundpd xmm2,xmm11,0x1"},
		{"movddup load", func(a *Buf) { a.MOVDDUPLoad(Y13, At(RBX, 72)) }, "movddup xmm13,QWORD PTR [rbx+0x48]"},
		{"pextrb store", func(a *Buf) { a.PEXTRBStore(At(RCX, 2), Y3, 2) }, "pextrb BYTE PTR [rcx+0x2],xmm3,0x2"},
		{"pextrb store hi", func(a *Buf) { a.PEXTRBStore(At(R12, 0), Y9, 0) }, "pextrb BYTE PTR [r12],xmm9,0x0"},
		{"pextrw store", func(a *Buf) { a.PEXTRWStore(At(RCX, 0), Y3, 0) }, "pextrw WORD PTR [rcx],xmm3,0x0"},
		{"pextrw store hi", func(a *Buf) { a.PEXTRWStore(At(R13, 4), Y10, 1) }, "pextrw WORD PTR [r13+0x4],xmm10,0x1"},
		{"movzx byte", func(a *Buf) { a.MOVZXBLoad(RAX, At(RDX, 0)) }, "movzx eax,BYTE PTR [rdx]"},
		{"movzx byte hi", func(a *Buf) { a.MOVZXBLoad(R10, Idx(R9, R11, 1, 2)) }, "movzx r10d,BYTE PTR [r9+r11*1+0x2]"},
	}
	dir := t.TempDir()
	for _, c := range cases {
		var a Buf
		c.emit(&a)
		b := a.Bytes()
		if got := disasmIntel(t, dir, "enc", b); got != c.want {
			t.Errorf("%s: emitted % x\n  objdump: %q\n  want:    %q", c.name, b, got, c.want)
		}
	}
	// A forward JZ over a body: objdump prints the target, which must be the
	// byte after the skipped instruction.
	var a Buf
	l := a.Label()
	a.JZ(l)
	a.TESTQ(RAX, RAX)
	a.Bind(l)
	if got := disasmIntel(t, dir, "jz", a.Bytes()); got == "" {
		t.Error("jz: nothing disassembled")
	}
}
