package cpu

import "testing"

// TestA64ShiftEncodingsMatchClang pins the shift encodings against words
// produced by clang's own assembler on Apple hardware.
//
// The field holds (2*esize - shift) for a right shift and (esize + shift) for
// a left one, and its width selects the element size, so a wrong encoding
// still disassembles and runs, shifting by the wrong amount or lane width.
//
// Generated on an arm64 host with:
//
//	clang -c -o probe.o probe.s && otool -t probe.o
func TestA64ShiftEncodingsMatchClang(t *testing.T) {
	for _, c := range []struct {
		what string
		got  func(*A64)
		want uint32
	}{
		{"ushr v0.4s, v1.4s, #4", func(a *A64) { a.USHR4s(0, 1, 4) }, 0x6f3c0420},
		{"ushr v2.4s, v3.4s, #8", func(a *A64) { a.USHR4s(2, 3, 8) }, 0x6f380462},
		{"ushr v20.4s, v21.4s, #16", func(a *A64) { a.USHR4s(20, 21, 16) }, 0x6f3006b4},
		{"ushr v5.4s, v6.4s, #24", func(a *A64) { a.USHR4s(5, 6, 24) }, 0x6f2804c5},
		{"ushr v7.4s, v8.4s, #1", func(a *A64) { a.USHR4s(7, 8, 1) }, 0x6f3f0507},
		{"ushr v9.4s, v10.4s, #32", func(a *A64) { a.USHR4s(9, 10, 32) }, 0x6f200549},
		{"sshr v1.4s, v1.4s, #22", func(a *A64) { a.SSHR4s(1, 1, 22) }, 0x4f2a0421},
		{"sshr v3.4s, v1.4s, #31", func(a *A64) { a.SSHR4s(3, 1, 31) }, 0x4f210423},
		{"movi v0.4s, #32", func(a *A64) { a.MOVI4s(0, 32) }, 0x4f010400},
		{"movi v20.4s, #1", func(a *A64) { a.MOVI4s(20, 1) }, 0x4f000434},
		{"movi v5.4s, #63", func(a *A64) { a.MOVI4s(5, 63) }, 0x4f0107e5},
		{"sub v0.4s, v1.4s, v2.4s", func(a *A64) { a.SUB4s(0, 1, 2) }, 0x6ea28420},
		{"sub v20.4s, v21.4s, v22.4s", func(a *A64) { a.SUB4s(20, 21, 22) }, 0x6eb686b4},
		{"movk x19, #1, lsl #16", func(a *A64) { a.MOVK(X19, 1, 1) }, 0xf2a00033},
		{"movk x1, #2, lsl #16", func(a *A64) { a.MOVK(X1, 2, 1) }, 0xf2a00041},
		{"movk x26, #43981, lsl #48", func(a *A64) { a.MOVK(X26, 43981, 3) }, 0xf2f579ba},
	} {
		var a A64
		c.got(&a)
		b := a.Bytes()
		if len(b) != 4 {
			t.Fatalf("%s: %d bytes", c.what, len(b))
		}
		got := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
		if got != c.want {
			t.Errorf("%s: emitted %08x, clang says %08x", c.what, got, c.want)
		}
	}
}

// TestA64MOVimmReachesEveryConstant covers MOVimm's multi-field path. 0x18000
// is a conv1d constant and 0x10000 the packed tiled matmul's output token
// stride at a 16384-row FFN.
func TestA64MOVimmReachesEveryConstant(t *testing.T) {
	for _, v := range []int64{0, 1, 0xFFFF, 0x10000, 0x18000, 0x8000_0000, 0x1234_5678, 0x1_0001, 0xDEAD_BEEF_CAFE} {
		var a A64
		a.MOVimm(X19, v) // must not panic
		if n := len(a.Bytes()); n == 0 || n%4 != 0 || n > 16 {
			t.Errorf("MOVimm %#x emitted %d bytes", v, n)
		}
	}
}
