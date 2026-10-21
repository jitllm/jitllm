//go:build arm64

package cpu

// EmitPackAct is the NEON twin of the amd64 activation packer. See packact.go
// for the contract and for why the unit is an amax window. FABS replaces the
// amd64 absolute-value mask.
//
// x1 scale, x2 dst, x3 amax src, x4 blocks, x5 consts, x6 amax count,
// x7 sum, x8 half, x9 quantize src, x10 cursor.
func EmitPackAct(tok int, half bool) []byte {
	var a A64
	a.LDRx(X1, X0, 0)   // Out  -> scale
	a.LDRx(X2, X0, 8)   // W    -> dst
	a.LDRx(X3, X0, 16)  // A    -> amax source
	a.LDRx(X4, X0, 32)  // Rows -> blocks
	a.LDRx(X5, X0, 56)  // Scr  -> constants
	a.LDRx(X6, X0, 40)  // K    -> elements to scan
	a.LDRx(X7, X0, 96)  // ASum
	a.LDRx(X9, X0, 112) // Q32  -> quantize source
	if half {
		a.LDRx(X8, X0, 104) // AHalfSum
	}

	// ---- the window's amax ----
	a.MOVIzero(V0)
	a.MOVreg(X10, X3)
	a.LSRimm(X11, X6, 2) // vectors of four
	mx := a.Label()
	a.Bind(mx)
	a.LDRq(V1, X10, 0)
	a.FABS4s(V1, V1)
	a.FMAX4s(V0, V0, V1)
	a.ADDimm(X10, X10, 16)
	a.SUBimm(X11, X11, 1)
	a.CBNZ(X11, mx)
	a.FMAXV(V0, V0) // one instruction; amd64 needs a three-fold ladder
	a.DUPs4(V0, V0)

	// ---- d = amax/127, inv = 1/d with inv zeroed when d is zero ----
	//
	// Two divisions, reproducing `d := amax/127` then `1/d` exactly: a single
	// 127/amax differs in the last bit and would move a quantized value at the
	// boundary. The zero case is the padding rows of a ragged tile, which
	// arrive on every prompt whose length is not a multiple of the tile.
	a.LDRs(V2, X5, 0) // 127
	a.DUPs4(V2, V2)
	a.FDIV4s(V3, V0, V2) // d
	a.LDRs(V4, X5, 4)    // 1.0
	a.DUPs4(V4, V4)
	a.FDIV4s(V5, V4, V3) // 1/d
	a.FCMEQzero(V6, V3)  // all ones where d == 0
	a.BIC16b(V5, V5, V6) // inv
	a.LDRs(V12, X5, 8)   // 0.5
	a.DUPs4(V12, V12)
	a.LDRs(V11, X5, 16) // 0x80000000
	a.DUPs4(V11, V11)
	a.LDRs(V7, X5, 20) // bias
	a.DUPs4(V7, V7)

	// ---- the blocks ----
	blk := a.Label()
	a.Bind(blk)
	a.MOVIzero(V8) // elements 0..15
	a.MOVIzero(V9) // elements 16..31
	for j := 0; j < 8; j++ {
		acc := V8
		if j >= 4 {
			acc = V9
		}
		a.LDRq(V1, X9, int32(j)*16)
		a.FMUL4s(V1, V1, V5)
		a.AND16b(V13, V1, V11)  // sign bits
		a.ORR16b(V13, V13, V12) // copysign(0.5, x)
		a.FADD4s(V1, V1, V13)
		a.FCVTZS4s(V1, V1)
		// The sum uses the unsaturated integers, which is exact: inv is
		// 1/(amax/127) and amax dominates its own window, so |v*inv| <= 127 and
		// the saturating narrow below never bites. See packact.go.
		a.ADD4s(acc, acc, V1)
		a.SQXTN4h(V14, V1)
		a.SQXTN8b(V14, V14)
		a.STRs(V14, X2, int32(j)*int32(tok)*4)
	}
	a.ADD4s(V15, V8, V9)
	a.ADDV(V15, V15)
	a.MUL4s(V15, V15, V7)
	a.STRs(V15, X7, 0)
	if half {
		a.ADDV(V16, V8)
		a.MUL4s(V16, V16, V7)
		a.STRs(V16, X8, 0)
		a.ADDV(V17, V9)
		a.MUL4s(V17, V17, V7)
		a.STRs(V17, X8, int32(tok)*4)
	}
	a.STRs(V3, X1, 0) // scale = d

	a.ADDimm(X9, X9, 128)
	a.addBig(X2, int32(tok)*4*8, X11)
	a.ADDimm(X1, X1, int32(tok)*4)
	a.ADDimm(X7, X7, int32(tok)*4)
	if half {
		a.ADDimm(X8, X8, int32(tok)*4*2)
	}
	a.SUBimm(X4, X4, 1)
	a.CBNZ(X4, blk)
	a.RET()
	return a.Bytes()
}

// EmitQuantAct is the matvec pair layout's packer on NEON (quantact_a64.go):
// the same body as above with a different epilogue -- {d, -sum*biasC/8} as an
// interleaved float32 pair per block, and each sixteen summed on its own.
func EmitQuantAct(half bool) []byte { return emitQuantActA64(half) }
