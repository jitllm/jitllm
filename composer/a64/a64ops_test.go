package a64

import "testing"

// TestA64OpsMatchLLVM pins the composer lowering's instructions to the words
// llvm-mc -triple=aarch64 -show-encoding gives for them.
func TestA64OpsMatchLLVM(t *testing.T) {
	acc := func(op A64Access, rt uint8, rn XReg, off int32) func(a *A64) {
		return func(a *A64) {
			if !a.Access(op, rt, rn, off) {
				t.Errorf("access %d at %d does not encode", op, off)
			}
		}
	}
	for _, c := range []struct {
		what string
		emit func(a *A64)
		want uint32
	}{
		{"eor v1.16b, v2.16b, v31.16b", func(a *A64) { a.EOR16b(1, 2, 31) }, 0x6e3f1c41},
		{"bsl v3.16b, v4.16b, v5.16b", func(a *A64) { a.BSL16b(3, 4, 5) }, 0x6e651c83},
		{"bif v3.16b, v4.16b, v5.16b", func(a *A64) { a.BIF16b(3, 4, 5) }, 0x6ee51c83},
		{"trn1 v1.4s, v2.4s, v3.4s", func(a *A64) { a.TRN1s4(1, 2, 3) }, 0x4e832841},
		{"trn2 v1.4s, v2.4s, v3.4s", func(a *A64) { a.TRN2s4(1, 2, 3) }, 0x4e836841},
		{"zip1 v1.4s, v2.4s, v3.4s", func(a *A64) { a.ZIP1s4(1, 2, 3) }, 0x4e833841},
		{"zip2 v1.4s, v2.4s, v3.4s", func(a *A64) { a.ZIP2s4(1, 2, 3) }, 0x4e837841},
		{"cmgt v1.8h, v2.8h, v3.8h", func(a *A64) { a.CMGT8h(1, 2, 3) }, 0x4e633441},
		{"smax v1.4s, v2.4s, v3.4s", func(a *A64) { a.SMAX4s(1, 2, 3) }, 0x4ea36441},
		{"umax v1.4s, v2.4s, v3.4s", func(a *A64) { a.UMAX4s(1, 2, 3) }, 0x6ea36441},
		{"umin v1.4s, v2.4s, v3.4s", func(a *A64) { a.UMIN4s(1, 2, 3) }, 0x6ea36c41},
		{"sminp v1.4s, v2.4s, v3.4s", func(a *A64) { a.SMINP4s(1, 2, 3) }, 0x4ea3ac41},
		{"fmaxp v1.4s, v2.4s, v3.4s", func(a *A64) { a.FMAXP4s(1, 2, 3) }, 0x6e23f441},
		{"sshl v1.4s, v2.4s, v3.4s", func(a *A64) { a.SSHL4s(1, 2, 3) }, 0x4ea34441},
		{"rev64 v1.4s, v30.4s", func(a *A64) { a.REV64s4(1, 30) }, 0x4ea00bc1},
		{"neg v1.4s, v30.4s", func(a *A64) { a.NEG4s(1, 30) }, 0x6ea0bbc1},
		{"xtn v1.4h, v30.4s", func(a *A64) { a.XTN4h(1, 30) }, 0x0e612bc1},
		{"fcvtn v1.4h, v30.4s", func(a *A64) { a.FCVTN4h(1, 30) }, 0x0e216bc1},
		{"fcvtn2 v1.8h, v30.4s", func(a *A64) { a.FCVTN2h8(1, 30) }, 0x4e216bc1},
		{"dup v1.4s, w30", func(a *A64) { a.DUPsW(1, X30) }, 0x4e040fc1},
		{"ld1r {v1.4s}, [x17]", func(a *A64) { a.LD1R4s(1, X17) }, 0x4d40ca21},
		{"ld1 {v1.h}[0], [x17]", func(a *A64) { a.LD1h0(1, X17) }, 0x0d404221},
		{"mov w17, v30.s[0]", func(a *A64) { a.UMOVs0(X17, 30) }, 0x0e043fd1},
		{"fmov s1, w17", func(a *A64) { a.FMOVsW(1, X17) }, 0x1e270221},
		{"scvtf s1, w17", func(a *A64) { a.SCVTFsW(1, X17) }, 0x1e220221},
		{"scvtf s1, x17", func(a *A64) { a.SCVTFsX(1, X17) }, 0x9e220221},
		{"add x1, x2, x3, lsl #2", func(a *A64) { a.ADDlsl(X1, X2, X3, 2) }, 0x8b030841},
		{"sub x1, x2, x3", func(a *A64) { a.SUBreg(X1, X2, X3) }, 0xcb030041},
		{"and x1, x2, x3", func(a *A64) { a.ANDreg(X1, X2, X3) }, 0x8a030041},
		{"orr x1, x2, x3", func(a *A64) { a.ORRreg(X1, X2, X3) }, 0xaa030041},
		{"eor x1, x2, x3", func(a *A64) { a.EORreg(X1, X2, X3) }, 0xca030041},
		{"bic x1, x2, x3", func(a *A64) { a.BICreg(X1, X2, X3) }, 0x8a230041},
		{"lsl x1, x2, x3", func(a *A64) { a.LSLV(X1, X2, X3) }, 0x9ac32041},
		{"lsr x1, x2, x3", func(a *A64) { a.LSRV(X1, X2, X3) }, 0x9ac32441},
		{"mul x1, x2, x3", func(a *A64) { a.MUL(X1, X2, X3) }, 0x9b037c41},
		{"udiv x1, x2, x3", func(a *A64) { a.UDIV(X1, X2, X3) }, 0x9ac30841},
		{"mvn x1, x3", func(a *A64) { a.MVN(X1, X3) }, 0xaa2303e1},
		{"neg x1, x3", func(a *A64) { a.NEG(X1, X3) }, 0xcb0303e1},
		{"mov w1, w3", func(a *A64) { a.MOVw(X1, X3) }, 0x2a0303e1},
		{"lsl x1, x2, #5", func(a *A64) { a.LSLimm(X1, X2, 5) }, 0xd37be841},
		{"lsl x1, x2, #0", func(a *A64) { a.LSLimm(X1, X2, 0) }, 0xd340fc41},
		{"cmp w2, w3", func(a *A64) { a.CMPw(X2, X3) }, 0x6b03005f},
		{"csel x1, x2, x3, le", func(a *A64) { a.CSEL(X1, X2, X3, CondLE) }, 0x9a83d041},
		{"csetm w1, lt", func(a *A64) { a.CSETMw(X1, CondLT) }, 0x5a9fa3e1},
		{"ldr q1, [x2, #32]", acc(A64LdrQ, 1, X2, 32), 0x3dc00841},
		{"ldur q1, [x2, #-32]", acc(A64LdrQ, 1, X2, -32), 0x3cde0041},
		{"stur q1, [sp, #-176]", acc(A64StrQ, 1, SP, -176), 0x3c9503e1},
		{"ldr w1, [x2, #8]", acc(A64LdrW, 1, X2, 8), 0xb9400841},
		{"stur w1, [x2, #-4]", acc(A64StrW, 1, X2, -4), 0xb81fc041},
		{"ldur x1, [sp, #-8]", acc(A64LdrX, 1, SP, -8), 0xf85f83e1},
		{"stur xzr, [sp, #-16]", acc(A64StrX, 31, SP, -16), 0xf81f03ff},
		{"str h1, [x2, #6]", acc(A64StrH, 1, X2, 6), 0x7d000c41},
		{"ldr d1, [x2, #8]", acc(A64LdrD, 1, X2, 8), 0xfd400441},
		{"str s1, [x2, #4]", acc(A64StrS, 1, X2, 4), 0xbd000441},
		{"mov x1, #-1", func(a *A64) { a.MOVconst(X1, ^uint64(0)) }, 0x92800001},
		{"mov x1, #0x12340000", func(a *A64) { a.MOVconst(X1, 0x12340000) }, 0xd2a24681},
	} {
		var a A64
		c.emit(&a)
		b := bytesOf(t, &a)
		if len(b) != 4 {
			t.Errorf("%s: %d bytes", c.what, len(b))
			continue
		}
		if got := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24; got != c.want {
			t.Errorf("%s: %#08x, want %#08x", c.what, got, c.want)
		}
	}
}

// TestA64MOVconstMakesTheValue checks MOVconst's words decode to its value
// for constants of every half pattern.
func TestA64MOVconstMakesTheValue(t *testing.T) {
	for _, v := range []uint64{0, 1, 0xFFFF, 0x10000, 0xFFFFFFFF, 0x80000000, ^uint64(0), 0xFFFFFFFF_FFFF0000, 0xFFFFFFFF_FFFF1234,
		0x1234_0000_5678_0000, 0xFFFF_0000_FFFF_0001, 0x8000_0000_0000_0000} {
		var a A64
		a.MOVconst(X1, v)
		b := bytesOf(t, &a)
		var x uint64
		for i := 0; i < len(b); i += 4 {
			w := uint32(b[i]) | uint32(b[i+1])<<8 | uint32(b[i+2])<<16 | uint32(b[i+3])<<24
			sh := 16 * uint64(w>>21&3)
			h := uint64(w >> 5 & 0xFFFF)
			switch w & 0xFF800000 {
			case 0xD2800000:
				x = h << sh
			case 0x92800000:
				x = ^(h << sh)
			case 0xF2800000:
				x = x&^(0xFFFF<<sh) | h<<sh
			default:
				t.Fatalf("%#x: word %#08x is no move", v, w)
			}
		}
		if x != v {
			t.Errorf("MOVconst(%#x) makes %#x in %d words", v, x, len(b)/4)
		}
	}
}
