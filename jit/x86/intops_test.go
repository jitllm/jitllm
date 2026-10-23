package x86

import (
	"encoding/hex"
	"testing"
)

// TestGPREncodings pins the general-register encoders the composer's CPU
// lowering uses to bytes objdump decodes as the named instruction.
func TestGPREncodings(t *testing.T) {
	for _, c := range []struct {
		name string
		emit func(*Buf)
		want string
	}{
		{"and %r9,%rax", func(a *Buf) { a.ANDQ(RAX, R9) }, "4c21c8"},
		{"or %rdx,%r11", func(a *Buf) { a.ORQ(R11, RDX) }, "4909d3"},
		{"and $0xfff,%r10", func(a *Buf) { a.ANDimm32(R10, 0x0FFF) }, "4981e2ff0f0000"},
		{"imul $0x28,%rbx,%r12", func(a *Buf) { a.IMULimm(R12, RBX, 40) }, "4c69e328000000"},
		{"shrx %rcx,%r13,%r8", func(a *Buf) { a.SHRX(R8, R13, RCX) }, "c442f3f7c5"},
		{"shlx %r9,%r15,%rax", func(a *Buf) { a.SHLX(RAX, R15, R9) }, "c4c2b1f7c7"},
		{"mov (%r8,%r13,4),%r11d", func(a *Buf) { a.MOVLLoad(R11, Idx(R8, R13, 4, 0)) }, "478b1ca8"},
		{"mov %r9d,%eax", func(a *Buf) { a.MOVL(RAX, R9) }, "4489c8"},
		{"movslq %r10d,%rbx", func(a *Buf) { a.MOVSXD(RBX, R10) }, "4963da"},
		{"neg %r12", func(a *Buf) { a.NEGQ(R12) }, "49f7dc"},
		{"xor $0x80808080,%rbx", func(a *Buf) { a.XORimm32(RBX, -0x7f7f7f80) }, "4881f380808080"},
		{"or $0x10,%r13", func(a *Buf) { a.ORimm32(R13, 0x10) }, "4981cd10000000"},
		{"vpsrlvd %ymm2,%ymm1,%ymm3", func(a *Buf) { a.VPSRLVD(Y3, Y1, Y2) }, "c4e27545da"},
		{"cmp %r10d,%ebx", func(a *Buf) { a.CMPL(RBX, R10) }, "4439d3"},
		{"movq $0x0,-0x8(%rsp)", func(a *Buf) { a.MOVMemImm(At(RSP, -8), 0) }, "48c74424f800000000"},
		{"addq $0x1,-0x10(%rsp)", func(a *Buf) { a.ADDMemImm8(At(RSP, -16), 1) }, "48834424f001"},
		{"addq $0x400,-0x18(%rsp)", func(a *Buf) { a.ADDMemImm32(At(RSP, -24), 1024) }, "48814424e800040000"},
		{"mov $0x80808080,%r15d", func(a *Buf) { a.MOVLimm(R15, 0x80808080) }, "41bf80808080"},
		{"cmp $0x3f,%r9d", func(a *Buf) { a.CMPLimm(R9, 63) }, "4181f93f000000"},
		{"vpsllvd %ymm2,%ymm1,%ymm3", func(a *Buf) { a.VPSLLVD(Y3, Y1, Y2) }, "c4e27547da"},
		{"vpsravd %ymm2,%ymm1,%ymm3", func(a *Buf) { a.VPSRAVD(Y3, Y1, Y2) }, "c4e27546da"},
	} {
		var a Buf
		c.emit(&a)
		b, err := a.Bytes()
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if got := hex.EncodeToString(b); got != c.want {
			t.Errorf("%s: %s, want %s", c.name, got, c.want)
		}
	}
}
