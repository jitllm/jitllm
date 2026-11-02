package cpu

import "testing"

// TestA64NewEncodingsMatchLLVM pins the structure loads and FMLS the rotary
// kernel uses, and the widen and mask the row-major float kernel uses.
// Reference words from llvm-mc -triple=aarch64 -show-encoding.
//
// A wrong size field on LD2 is still a valid LD2 -- of bytes or halves -- and
// deinterleaves the table into nonsense that disassembles perfectly.
func TestA64NewEncodingsMatchLLVM(t *testing.T) {
	for _, c := range []struct {
		what string
		emit func(a *A64)
		want uint32
	}{
		{"ld2 {v2.4s, v3.4s}, [x5], #32", func(a *A64) { a.LD2Post(2, X5) }, 0x4cdf88a2},
		{"ld2 {v4.4s, v5.4s}, [x6]", func(a *A64) { a.LD2(4, X6) }, 0x4c4088c4},
		{"st2 {v6.4s, v7.4s}, [x6], #32", func(a *A64) { a.ST2Post(6, X6) }, 0x4c9f88c6},
		{"ld2 {v30.4s, v31.4s}, [x28], #32", func(a *A64) { a.LD2Post(30, XReg(28)) }, 0x4cdf8b9e},
		{"st2 {v31.4s, v0.4s}, [x1]", func(a *A64) { a.ST2(31, X1) }, 0x4c00883f},
		{"fmls v6.4s, v5.4s, v3.4s", func(a *A64) { a.FMLS4s(6, 5, 3) }, 0x4ea3cca6},
		{"fmls v31.4s, v0.4s, v30.4s", func(a *A64) { a.FMLS4s(31, 0, 30) }, 0x4ebecc1f},
		{"shll v4.4s, v20.4h, #16", func(a *A64) { a.SHLL(4, 20) }, 0x2e613a84},
		{"shll2 v5.4s, v20.8h, #16", func(a *A64) { a.SHLL2(5, 20) }, 0x6e613a85},
		{"shll v31.4s, v0.4h, #16", func(a *A64) { a.SHLL(31, 0) }, 0x2e61381f},
		{"ushl v2.4s, v2.4s, v5.4s", func(a *A64) { a.USHL4s(2, 2, 5) }, 0x6ea54442},
		{"ushl v31.4s, v0.4s, v17.4s", func(a *A64) { a.USHL4s(31, 0, 17) }, 0x6eb1441f},
		{"str d1, [x13, #32]", func(a *A64) { a.STRd(1, XReg(13), 32) }, 0xfd0011a1},
		{"str d2, [x13, #40]", func(a *A64) { a.STRd(2, XReg(13), 40) }, 0xfd0015a2},
		{"and x10, x10, #7", func(a *A64) { a.ANDlow(X10, X10, 3) }, 0x9240094a},
		{"and x11, x6, #3", func(a *A64) { a.ANDlow(XReg(11), X6, 2) }, 0x924004cb},
		{"uzp1 v22.8h, v22.8h, v22.8h", func(a *A64) { a.UZP1h8(22, 22, 22) }, 0x4e561ad6},
		{"uzp2 v23.8h, v22.8h, v22.8h", func(a *A64) { a.UZP2h8(23, 22, 22) }, 0x4e565ad7},
		{"uzp1 v31.8h, v0.8h, v17.8h", func(a *A64) { a.UZP1h8(31, 0, 17) }, 0x4e51181f},
		{"uzp2 v1.8h, v30.8h, v2.8h", func(a *A64) { a.UZP2h8(1, 30, 2) }, 0x4e425bc1},
		{"prfm pldl1keep, [x12, #512]", func(a *A64) { a.PRFM(XReg(12), 512) }, 0xf9810180},
		{"prfm pldl1keep, [x3]", func(a *A64) { a.PRFM(X3, 0) }, 0xf9800060},
		{"prfm pldl2keep, [x1]", func(a *A64) { a.PRFMop(prfmPLDL2KEEP, X1, 0) }, 0xf9800022},
		{"prfm pldl1keep, [x11, #32760]", func(a *A64) { a.PRFM(XReg(11), 32760) }, 0xf9bffd60},
	} {
		var a A64
		c.emit(&a)
		b := a.Bytes()
		got := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
		if got != c.want {
			t.Errorf("%s: emitted %08x, llvm says %08x", c.what, got, c.want)
		}
	}
}
