package cpu

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// Every expected byte string below came out of an assembler (clang-18
// --target=aarch64-linux-gnu, read back with llvm-objdump-18).
// TestA64Assembler regenerates them from the `name` column and diffs, so the
// table cannot drift from the assembler.
//
// The traps on A64 are all field packing:
//
//   - USHR's immediate is INVERTED: the field holds (2*esize - shift). A wrong
//     one is a valid instruction with a silently wrong shift count.
//   - Scaled load/store offsets are divided by the access size, and a bad one
//     is silently a different instruction (the assembler rewrites ldr->ldur).
//     Emitting (off/16)<<10 for off=8 in Go integer arithmetic gives offset 0.
//   - The by-element index of FMLA is split across L (bit 21) and H (bit 11),
//     and its Rm field is five bits wide even though the manual documents it as
//     four plus a separate M bit. Both mistakes silently pick another lane or
//     another register.
//
// So each has an entry with a high register number and a non-zero index or a
// boundary offset, where a narrow field shows itself.
var a64Encodings = []struct {
	name string
	want string
	emit func(*A64)
}{
	// The quantizer's and the argmax's NEON additions, pinned here too so the
	// amd64 host checks them against an assembler.
	{"fcmgt v9.4s, v7.4s, v0.4s", "e9e4a06e", func(a *A64) { a.FCMGT4s(VReg(9), VReg(7), VReg(0)) }},
	{"fcmeq v12.4s, v0.4s, v11.4s", "0ce42b4e", func(a *A64) { a.FCMEQ4s(VReg(12), VReg(0), VReg(11)) }},
	{"smin v15.4s, v15.4s, v17.4s", "ef6db14e", func(a *A64) { a.SMIN4s(VReg(15), VReg(15), VReg(17)) }},
	{"sminv s15, v15.4s", "efa9b14e", func(a *A64) { a.SMINV(VReg(15), VReg(15)) }},
	// Scalar, enough to drive a loop.
	{"mla v1.4s, v2.4s, v3.4s", "4194a34e", func(a *A64) { a.MLA4s(VReg(1), VReg(2), VReg(3)) }},
	{"mla v31.4s, v30.4s, v29.4s", "df97bd4e", func(a *A64) { a.MLA4s(VReg(31), VReg(30), VReg(29)) }},
	{"fadd v0.4s, v0.4s, v26.4s", "00d43a4e", func(a *A64) { a.FADD4s(VReg(0), VReg(0), VReg(26)) }},
	{"fadd v28.4s, v27.4s, v31.4s", "7cd73f4e", func(a *A64) { a.FADD4s(VReg(28), VReg(27), VReg(31)) }},
	{"ldr x1, [x0]", "010040f9", func(a *A64) { a.LDRx(X1, X0, 0) }},
	{"ldr x7, [x0, #48]", "071840f9", func(a *A64) { a.LDRx(X7, X0, 48) }},
	{"ldr x20, [x21, #32760]", "b4fe7ff9", func(a *A64) { a.LDRx(X20, X21, 32760) }},
	// LDP q: the pair load. Offsets chosen to cover zero, a positive multiple,
	// the negative half of the signed field, and both ends of the range.
	{"ldp q0, q1, [x2]", "400440ad", func(a *A64) { a.LDPq(VReg(0), VReg(1), X2, 0) }},
	{"ldp q4, q5, [x6, #32]", "c41441ad", func(a *A64) { a.LDPq(VReg(4), VReg(5), X6, 32) }},
	{"ldp q30, q31, [x9, #-32]", "3e7d7fad", func(a *A64) { a.LDPq(VReg(30), VReg(31), X9, -32) }},
	{"ldp q2, q3, [x4, #1008]", "828c5fad", func(a *A64) { a.LDPq(VReg(2), VReg(3), X4, 1008) }},
	{"ldp q2, q3, [x4, #-1024]", "820c60ad", func(a *A64) { a.LDPq(VReg(2), VReg(3), X4, -1024) }},
	{"str x1, [x2]", "410000f9", func(a *A64) { a.STRx(X1, X2, 0) }},
	{"str x9, [x10, #8]", "490500f9", func(a *A64) { a.STRx(X9, X10, 8) }},
	{"str x20, [x21, #32760]", "b4fe3ff9", func(a *A64) { a.STRx(X20, X21, 32760) }},
	{"add x2, x2, #34", "42880091", func(a *A64) { a.ADDimm(X2, X2, 34) }},
	{"add x9, x9, #4095", "29fd3f91", func(a *A64) { a.ADDimm(X9, X9, 4095) }},
	{"sub x5, x5, #1", "a50400d1", func(a *A64) { a.SUBimm(X5, X5, 1) }},
	{"sub x20, x21, #4095", "b4fe3fd1", func(a *A64) { a.SUBimm(X20, X21, 4095) }},
	{"mov x7, x3", "e70303aa", func(a *A64) { a.MOVreg(X7, X3) }},
	{"mov x25, x19", "f90313aa", func(a *A64) { a.MOVreg(X25, X19) }},
	{"ret", "c0035fd6", func(a *A64) { a.RET() }},

	// Vector load/store. LDUR is the load-bearing one: a Q4_0 block is 18 bytes
	// and a Q8_0 block 34, so the payload never lands on a multiple of 16 and
	// the scaled LDR form cannot address it at all.
	{"ldr q0, [x1]", "2000c03d", func(a *A64) { a.LDRq(V0, X1, 0) }},
	{"ldr q3, [x2, #16]", "4304c03d", func(a *A64) { a.LDRq(V3, X2, 16) }},
	{"ldr q31, [x30, #65520]", "dfffff3d", func(a *A64) { a.LDRq(V31, X30, 65520) }},
	{"ldr q5, [x2, x6]", "4568e63c", func(a *A64) { a.LDRqIdx(V5, X2, X6) }},
	{"str q0, [x1]", "2000803d", func(a *A64) { a.STRq(V0, X1, 0) }},
	{"str q31, [x30, #65520]", "dfffbf3d", func(a *A64) { a.STRq(V31, X30, 65520) }},
	{"ldur q2, [x2, #2]", "4220c03c", func(a *A64) { a.LDURq(V2, X2, 2) }},
	{"ldur q3, [x2, #18]", "4320c13c", func(a *A64) { a.LDURq(V3, X2, 18) }},
	{"ldur q0, [x0, #-16]", "0000df3c", func(a *A64) { a.LDURq(V0, X0, -16) }},
	{"ldur q31, [x30, #255]", "dff3cf3c", func(a *A64) { a.LDURq(V31, X30, 255) }},
	{"ldur q17, [x9, #-256]", "3101d03c", func(a *A64) { a.LDURq(V17, X9, -256) }},
	{"ldr h1, [x2]", "4100407d", func(a *A64) { a.LDRh(V1, X2, 0) }},
	{"ldr b1, [x2]", "4100403d", func(a *A64) { a.LDRb(V1, X2, 0) }},
	{"ldr b17, [x9, #4095]", "31fd7f3d", func(a *A64) { a.LDRb(V17, X9, 4095) }},
	{"ldr h17, [x9, #8190]", "31fd7f7d", func(a *A64) { a.LDRh(V17, X9, 8190) }},
	{"ldr s7, [x8]", "070140bd", func(a *A64) { a.LDRs(V7, X8, 0) }},
	{"ldr s13, [x3, #8]", "6d0840bd", func(a *A64) { a.LDRs(V13, X3, 8) }},
	{"str s0, [x1]", "200000bd", func(a *A64) { a.STRs(V0, X1, 0) }},
	{"str s31, [x30, #16380]", "dfff3fbd", func(a *A64) { a.STRs(V31, X30, 16380) }},

	// Constants. The split imm8 puts its high three bits at [18:16] and its low
	// five at [9:5], so 0xff is the entry that would catch a naive imm8<<5.
	{"movi v0.4s, #0", "0004004f", func(a *A64) { a.MOVIzero(V0) }},
	{"movi v31.4s, #0", "1f04004f", func(a *A64) { a.MOVIzero(V31) }},
	{"movi v29.16b, #15", "fde5004f", func(a *A64) { a.MOVI16b(V29, 0x0F) }},
	{"movi v30.16b, #8", "1ee5004f", func(a *A64) { a.MOVI16b(V30, 8) }},
	{"movi v31.16b, #255", "ffe7074f", func(a *A64) { a.MOVI16b(V31, 0xFF) }},

	// SDOT.
	{"sdot v0.4s, v1.16b, v2.16b", "2094824e", func(a *A64) { a.SDOT(V0, V1, V2) }},
	{"sdot v6.4s, v2.16b, v4.16b", "4694844e", func(a *A64) { a.SDOT(V6, V2, V4) }},
	{"sdot v31.4s, v30.16b, v29.16b", "df979d4e", func(a *A64) { a.SDOT(V31, V30, V29) }},

	// SDOT widened for a chip without FEAT_DotProd (sdotemu.go).
	{"smull v8.8h, v1.8b, v2.8b", "28c0220e", func(a *A64) { a.SMULL8b(V8, V1, V2) }},
	{"smull v31.8h, v30.8b, v29.8b", "dfc33d0e", func(a *A64) { a.SMULL8b(V31, V30, V29) }},
	{"smull2 v9.8h, v1.16b, v2.16b", "29c0224e", func(a *A64) { a.SMULL16b(V9, V1, V2) }},
	{"smull2 v31.8h, v30.16b, v29.16b", "dfc33d4e", func(a *A64) { a.SMULL16b(V31, V30, V29) }},
	{"saddlp v8.4s, v8.8h", "0829604e", func(a *A64) { a.SADDLP8h(V8, V8) }},
	{"saddlp v31.4s, v30.8h", "df2b604e", func(a *A64) { a.SADDLP8h(V31, V30) }},
	{"addp v8.4s, v8.4s, v9.4s", "08bda94e", func(a *A64) { a.ADDP4s(V8, V8, V9) }},
	{"addp v31.4s, v30.4s, v29.4s", "dfbfbd4e", func(a *A64) { a.ADDP4s(V31, V30, V29) }},
	{"smlal v1.8h, v2.8b, v3.8b", "4180230e", func(a *A64) { a.SMLAL8b(V1, V2, V3) }},
	{"smlal v31.8h, v30.8b, v29.8b", "df833d0e", func(a *A64) { a.SMLAL8b(V31, V30, V29) }},
	{"smlal2 v1.8h, v2.16b, v3.16b", "4180234e", func(a *A64) { a.SMLAL16b(V1, V2, V3) }},
	{"smlal2 v31.8h, v30.16b, v29.16b", "df833d4e", func(a *A64) { a.SMLAL16b(V31, V30, V29) }},
	{"addp v1.8h, v2.8h, v3.8h", "41bc634e", func(a *A64) { a.ADDP8h(V1, V2, V3) }},
	{"addp v31.8h, v30.8h, v29.8h", "dfbf7d4e", func(a *A64) { a.ADDP8h(V31, V30, V29) }},
	{"sadalp v1.4s, v2.8h", "4168604e", func(a *A64) { a.SADALP8h(V1, V2) }},
	{"sadalp v31.4s, v30.8h", "df6b604e", func(a *A64) { a.SADALP8h(V31, V30) }},

	// The Q4_0 nibble unpack.
	{"and v3.16b, v2.16b, v29.16b", "431c3d4e", func(a *A64) { a.AND16b(V3, V2, V29) }},
	{"and v31.16b, v30.16b, v29.16b", "df1f3d4e", func(a *A64) { a.AND16b(V31, V30, V29) }},
	{"sub v3.16b, v3.16b, v30.16b", "63843e6e", func(a *A64) { a.SUB16b(V3, V3, V30) }},
	{"sub v31.16b, v30.16b, v29.16b", "df873d6e", func(a *A64) { a.SUB16b(V31, V30, V29) }},
	{"ushr v4.16b, v2.16b, #4", "44040c6f", func(a *A64) { a.USHR16b(V4, V2, 4) }},
	{"ushr v31.16b, v30.16b, #1", "df070f6f", func(a *A64) { a.USHR16b(V31, V30, 1) }},
	{"ushr v0.16b, v1.16b, #8", "2004086f", func(a *A64) { a.USHR16b(V0, V1, 8) }},

	// Q5_0's fifth-bit plane: TBL fans the four qh bytes out to sixteen lanes,
	// USHL16b gives each lane its own shift so bit (j%8) lands at position 4,
	// and ORR merges it into the nibble. USHL4s is pinned beside its byte form
	// because the two differ only in bits 23:22.
	{"ushl v19.16b, v19.16b, v27.16b", "73463b6e", func(a *A64) { a.USHL16b(V19, V19, V27) }},
	{"ushl v0.16b, v1.16b, v2.16b", "2044226e", func(a *A64) { a.USHL16b(V0, V1, V2) }},
	{"ushl v31.16b, v30.16b, v29.16b", "df473d6e", func(a *A64) { a.USHL16b(V31, V30, V29) }},
	{"ushl v4.4s, v5.4s, v6.4s", "a444a66e", func(a *A64) { a.USHL4s(V4, V5, V6) }},
	{"tbl v19.16b, { v17.16b }, v26.16b", "33021a4e", func(a *A64) { a.TBL(V19, V17, V26) }},
	{"tbl v31.16b, { v30.16b }, v29.16b", "df031d4e", func(a *A64) { a.TBL(V31, V30, V29) }},
	{"orr v3.16b, v3.16b, v19.16b", "631cb34e", func(a *A64) { a.ORR16b(V3, V3, V19) }},
	{"orr v31.16b, v30.16b, v29.16b", "df1fbd4e", func(a *A64) { a.ORR16b(V31, V30, V29) }},
	{"movi v28.16b, #16", "1ce6004f", func(a *A64) { a.MOVI16b(V28, 0x10) }},

	// Convert, scale, accumulate.
	{"scvtf v6.4s, v6.4s", "c6d8214e", func(a *A64) { a.SCVTF4s(V6, V6) }},
	{"scvtf v31.4s, v30.4s", "dfdb214e", func(a *A64) { a.SCVTF4s(V31, V30) }},
	{"fcvt s1, h1", "2140e21e", func(a *A64) { a.FCVTsh(V1, V1) }},
	{"fcvt s31, h30", "df43e21e", func(a *A64) { a.FCVTsh(V31, V30) }},
	{"fmul s1, s1, s7", "2108271e", func(a *A64) { a.FMULs(V1, V1, V7) }},
	{"fmul s31, s30, s29", "df0b3d1e", func(a *A64) { a.FMULs(V31, V30, V29) }},
	{"ld1 { v4.16b, v5.16b, v6.16b, v7.16b }, [x9]", "2421404c", func(a *A64) { a.LD1x4(V4, X9) }},
	{"ld1 { v4.16b, v5.16b, v6.16b, v7.16b }, [x9], #64", "2421df4c", func(a *A64) { a.LD1x4Post(V4, X9) }},
	{"ld1 { v16.16b, v17.16b, v18.16b, v19.16b }, [x22], #64", "d022df4c", func(a *A64) { a.LD1x4Post(V16, X22) }},
	{"bic v0.16b, v1.16b, v2.16b", "201c624e", func(a *A64) { a.BIC16b(V0, V1, V2) }},
	{"bic v29.16b, v30.16b, v31.16b", "dd1f7f4e", func(a *A64) { a.BIC16b(V29, V30, V31) }},
	{"mla v0.4s, v6.4s, v1.s[0]", "c000816f", func(a *A64) { a.MLAelem(V0, V6, V1, 0) }},
	{"mla v0.4s, v11.4s, v31.s[3]", "6009bf6f", func(a *A64) { a.MLAelem(V0, V11, V31, 3) }},
	{"mla v17.4s, v18.4s, v19.s[2]", "510a936f", func(a *A64) { a.MLAelem(V17, V18, V19, 2) }},
	{"add v0.4s, v1.4s, v2.4s", "2084a24e", func(a *A64) { a.ADD4s(V0, V1, V2) }},
	{"add v28.4s, v27.4s, v31.4s", "7c87bf4e", func(a *A64) { a.ADD4s(V28, V27, V31) }},
	{"fmla v0.4s, v6.4s, v1.s[0]", "c010814f", func(a *A64) { a.FMLAelem(V0, V6, V1, 0) }},
	{"fmla v0.4s, v11.4s, v31.s[3]", "6019bf4f", func(a *A64) { a.FMLAelem(V0, V11, V31, 3) }},
	{"fmla v17.4s, v18.4s, v19.s[2]", "511a934f", func(a *A64) { a.FMLAelem(V17, V18, V19, 2) }},
	{"fmla v31.4s, v30.4s, v29.s[1]", "df13bd4f", func(a *A64) { a.FMLAelem(V31, V30, V29, 1) }},

	// Horizontal reduction.
	{"faddp v0.4s, v0.4s, v0.4s", "00d4206e", func(a *A64) { a.FADDP4s(V0, V0, V0) }},
	{"faddp v31.4s, v30.4s, v29.4s", "dfd73d6e", func(a *A64) { a.FADDP4s(V31, V30, V29) }},
	{"faddp s0, v0.2s", "00d8307e", func(a *A64) { a.FADDPs(V0, V0) }},
	{"faddp s31, v30.2s", "dfdb307e", func(a *A64) { a.FADDPs(V31, V30) }},

	// The image path's additions (a64img.go): float64 lanes, the narrowing
	// chain, the narrow stores and the table and pixel loads.
	{"fadd v1.2d, v2.2d, v3.2d", "41d4634e", func(a *A64) { a.FADD2d(V1, V2, V3) }},
	{"fadd v31.2d, v30.2d, v29.2d", "dfd77d4e", func(a *A64) { a.FADD2d(V31, V30, V29) }},
	{"fsub v1.2d, v2.2d, v3.2d", "41d4e34e", func(a *A64) { a.FSUB2d(V1, V2, V3) }},
	{"fsub v17.2d, v30.2d, v29.2d", "d1d7fd4e", func(a *A64) { a.FSUB2d(V17, V30, V29) }},
	{"fmul v1.2d, v2.2d, v3.2d", "41dc636e", func(a *A64) { a.FMUL2d(V1, V2, V3) }},
	{"fmul v31.2d, v16.2d, v29.2d", "1fde7d6e", func(a *A64) { a.FMUL2d(V31, V16, V29) }},
	{"fdiv v1.2d, v2.2d, v3.2d", "41fc636e", func(a *A64) { a.FDIV2d(V1, V2, V3) }},
	{"fdiv v20.2d, v21.2d, v22.2d", "b4fe766e", func(a *A64) { a.FDIV2d(V20, V21, V22) }},
	{"frintn v1.2d, v2.2d", "4188614e", func(a *A64) { a.FRINTN2d(V1, V2) }},
	{"frintn v30.2d, v19.2d", "7e8a614e", func(a *A64) { a.FRINTN2d(V30, V19) }},
	{"frintm v1.2d, v2.2d", "4198614e", func(a *A64) { a.FRINTM2d(V1, V2) }},
	{"frintm v30.2d, v19.2d", "7e9a614e", func(a *A64) { a.FRINTM2d(V30, V19) }},
	{"frintm v1.4s, v2.4s", "4198214e", func(a *A64) { a.FRINTM4s(V1, V2) }},
	{"frintm v30.4s, v19.4s", "7e9a214e", func(a *A64) { a.FRINTM4s(V30, V19) }},
	{"fcvtl v1.2d, v2.2s", "4178610e", func(a *A64) { a.FCVTL2d(V1, V2) }},
	{"fcvtl v28.2d, v17.2s", "3c7a610e", func(a *A64) { a.FCVTL2d(V28, V17) }},
	{"fcvtn v1.2s, v2.2d", "4168610e", func(a *A64) { a.FCVTN2s(V1, V2) }},
	{"fcvtn v28.2s, v17.2d", "3c6a610e", func(a *A64) { a.FCVTN2s(V28, V17) }},
	{"shl v1.2d, v2.2d, #52", "4154744f", func(a *A64) { a.SHL2d(V1, V2, 52) }},
	{"shl v30.2d, v31.2d, #1", "fe57414f", func(a *A64) { a.SHL2d(V30, V31, 1) }},
	{"dup v1.2d, v2.d[0]", "4104084e", func(a *A64) { a.DUPd2(V1, V2) }},
	{"dup v29.2d, v18.d[0]", "5d06084e", func(a *A64) { a.DUPd2(V29, V18) }},
	{"dup v3.4s, v1.s[3]", "23041c4e", func(a *A64) { a.DUPs4lane(V3, V1, 3) }},
	{"dup v30.4s, v17.s[1]", "3e060c4e", func(a *A64) { a.DUPs4lane(V30, V17, 1) }},
	{"dup v1.4s, w2", "410c044e", func(a *A64) { a.DUPgp(V1, X2) }},
	{"dup v30.4s, w13", "be0d044e", func(a *A64) { a.DUPgp(V30, X13) }},
	{"rev16 v1.8b, v2.8b", "4118200e", func(a *A64) { a.REV16_8b(V1, V2) }},
	{"rev16 v29.8b, v18.8b", "5d1a200e", func(a *A64) { a.REV16_8b(V29, V18) }},
	{"sub x1, x2, x3", "410003cb", func(a *A64) { a.SUBreg(X1, X2, X3) }},
	{"sub x20, x13, x9", "b40109cb", func(a *A64) { a.SUBreg(X20, X13, X9) }},
	{"smax v1.4s, v2.4s, v3.4s", "4164a34e", func(a *A64) { a.SMAX4s(V1, V2, V3) }},
	{"smax v31.4s, v30.4s, v16.4s", "df67b04e", func(a *A64) { a.SMAX4s(V31, V30, V16) }},
	{"xtn v1.4h, v2.4s", "4128610e", func(a *A64) { a.XTN4h(V1, V2) }},
	{"xtn v29.4h, v18.4s", "5d2a610e", func(a *A64) { a.XTN4h(V29, V18) }},
	{"xtn v1.8b, v2.8h", "4128210e", func(a *A64) { a.XTN8b(V1, V2) }},
	{"xtn v29.8b, v18.8h", "5d2a210e", func(a *A64) { a.XTN8b(V29, V18) }},
	{"str b1, [x2]", "4100003d", func(a *A64) { a.STRb(V1, X2, 0) }},
	{"str b30, [x9, #4095]", "3efd3f3d", func(a *A64) { a.STRb(V30, X9, 4095) }},
	{"str h1, [x2]", "4100007d", func(a *A64) { a.STRh(V1, X2, 0) }},
	{"str h30, [x9, #8190]", "3efd3f7d", func(a *A64) { a.STRh(V30, X9, 8190) }},
	{"ldrb w1, [x2]", "41004039", func(a *A64) { a.LDRBw(X1, X2, 0) }},
	{"ldrb w20, [x9, #4095]", "34fd7f39", func(a *A64) { a.LDRBw(X20, X9, 4095) }},
	{"ldr s1, [x2, x3, lsl #2]", "417863bc", func(a *A64) { a.LDRsIdx(V1, X2, X3) }},
	{"ldr s30, [x19, x20, lsl #2]", "7e7a74bc", func(a *A64) { a.LDRsIdx(V30, X19, X20) }},
	{"ld1 { v1.b }[2], [x2]", "4108400d", func(a *A64) { a.LD1b(V1, 2, X2) }},
	{"ld1 { v30.b }[13], [x9]", "3e15404d", func(a *A64) { a.LD1b(V30, 13, X9) }},
	{"st1 { v1.b }[2], [x2]", "4108000d", func(a *A64) { a.ST1b(V1, 2, X2) }},
	{"st1 { v30.b }[13], [x9]", "3e15004d", func(a *A64) { a.ST1b(V30, 13, X9) }},
}

// TestA64Encodings is host-independent: it builds byte strings and never
// executes them. That is also why the emitter lives in a64.go rather than a
// *_arm64.go file, whose implicit GOARCH constraint would confine this test.
func TestA64Encodings(t *testing.T) {
	for _, tc := range a64Encodings {
		var a A64
		tc.emit(&a)
		if got := hex.EncodeToString(a.Bytes()); got != tc.want {
			t.Errorf("%s\n got  %s\n want %s", tc.name, got, tc.want)
		}
		if len(a.Bytes()) != 4 {
			t.Errorf("%s emitted %d bytes; A64 instructions are always 4",
				tc.name, len(a.Bytes()))
		}
	}
	t.Logf("%d A64 encodings byte-exact against clang's aarch64 assembler", len(a64Encodings))
}

// TestA64Assembler regenerates the whole table from an assembler and diffs,
// catching a `want` mistyped in a way the emitter happens to reproduce. It is
// the arm64 analogue of scripts/vexref.sh.
func TestA64Assembler(t *testing.T) {
	as := lookAny("clang-18", "clang", "clang-17", "clang-19")
	od := lookAny("llvm-objdump-18", "llvm-objdump", "llvm-objdump-17", "llvm-objdump-19")
	if as == "" || od == "" {
		t.Skip("no clang + llvm-objdump to cross-assemble aarch64")
	}

	var src strings.Builder
	// Without +dotprod clang rejects sdot outright.
	src.WriteString(".arch armv8.2-a+dotprod+fp16\n.text\n")
	for _, tc := range a64Encodings {
		src.WriteString(tc.name)
		src.WriteByte('\n')
	}
	dir := t.TempDir()
	sp := filepath.Join(dir, "a.s")
	op := filepath.Join(dir, "a.o")
	if err := os.WriteFile(sp, []byte(src.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(as, "--target=aarch64-linux-gnu", "-c", sp, "-o", op).CombinedOutput(); err != nil {
		t.Skipf("%s cannot cross-assemble aarch64: %v\n%s", as, err, out)
	}
	out, err := exec.Command(od, "-d", op).CombinedOutput()
	if err != nil {
		t.Fatalf("%s: %v\n%s", od, err, out)
	}

	// "       4: f9401807     \tldr\tx7, [x0, #0x30]"
	re := regexp.MustCompile(`(?m)^\s*[0-9a-f]+:\s+([0-9a-f]{8})\s`)
	ms := re.FindAllStringSubmatch(string(out), -1)
	if len(ms) != len(a64Encodings) {
		t.Fatalf("assembled %d instructions, table has %d\n%s", len(ms), len(a64Encodings), out)
	}
	for i, m := range ms {
		// objdump prints the 32-bit word; the file holds it little-endian.
		w := m[1]
		want := w[6:8] + w[4:6] + w[2:4] + w[0:2]
		if want != a64Encodings[i].want {
			t.Errorf("%s: table says %s, assembler says %s",
				a64Encodings[i].name, a64Encodings[i].want, want)
		}
	}
	t.Logf("%d encodings regenerated from %s and matched", len(ms), as)
}

// a64FloatEncodings pins the elementwise float ops byte-for-byte against
// llvm-mc -show-encoding. A wrong opcode does not fault; it decodes to some
// other NEON instruction.
var a64FloatEncodings = []struct {
	name string
	emit func(*A64)
	want uint32
}{
	{"fsub v1.4s, v3.4s, v4.4s", func(a *A64) { a.FSUB4s(1, 3, 4) }, 0x4EA4D461},
	{"fmax v1.4s, v3.4s, v4.4s", func(a *A64) { a.FMAX4s(1, 3, 4) }, 0x4E24F461},
	{"fmin v1.4s, v3.4s, v4.4s", func(a *A64) { a.FMIN4s(1, 3, 4) }, 0x4EA4F461},
	{"fdiv v1.4s, v3.4s, v4.4s", func(a *A64) { a.FDIV4s(1, 3, 4) }, 0x6E24FC61},
	{"fsqrt v1.4s, v3.4s", func(a *A64) { a.FSQRT4s(1, 3) }, 0x6EA1F861},
	{"fneg v1.4s, v3.4s", func(a *A64) { a.FNEG4s(1, 3) }, 0x6EA0F861},
	{"fcvtns v1.4s, v3.4s", func(a *A64) { a.FCVTNS4s(1, 3) }, 0x4E21A861},
	{"shl v1.4s, v3.4s, #23", func(a *A64) { a.SHL4s(1, 3, 23) }, 0x4F375461},
	{"fmaxv s1, v3.4s", func(a *A64) { a.FMAXV(1, 3) }, 0x6E30F861},
	{"dup v1.4s, v3.s[0]", func(a *A64) { a.DUPs4(1, 3) }, 0x4E040461},
}

func TestA64FloatEncodings(t *testing.T) {
	for _, tc := range a64FloatEncodings {
		var a A64
		tc.emit(&a)
		b := a.Bytes()
		if len(b) != 4 {
			t.Fatalf("%s: %d bytes", tc.name, len(b))
		}
		got := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
		if got != tc.want {
			t.Errorf("%s\n got  %08x\n want %08x", tc.name, got, tc.want)
		}
	}
}

// TestA64Disassembles round-trips the table and the real kernels through a
// disassembler, catching a valid-but-wrong encoding. It is the only structural
// check the kernels get on a machine that cannot run them.
func TestA64Disassembles(t *testing.T) {
	mc := lookAny("llvm-mc-18", "llvm-mc", "llvm-mc-17", "llvm-mc-19")
	if mc == "" {
		t.Skip("no llvm-mc to disassemble aarch64")
	}
	var a A64
	for _, tc := range a64Encodings {
		tc.emit(&a)
	}
	stream := a.Bytes()
	for _, wt := range a64KernelTypes {
		code, err := EmitA64(Spec{W: wt, Rows: 1, Accs: 1, Cols: 1})
		if err != nil {
			t.Fatal(err)
		}
		stream = append(stream, code...)
	}

	var in strings.Builder
	for _, b := range stream {
		fmt.Fprintf(&in, "0x%02x ", b)
	}
	cmd := exec.Command(mc, "--disassemble", "-triple=aarch64", "-mattr=+dotprod,+fullfp16")
	cmd.Stdin = strings.NewReader(in.String())
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%s: %v\n%s", mc, err, out)
	}
	text := string(out)
	// llvm-mc prints an undecodable word as a warning plus a .byte directive.
	for _, bad := range []string{"warning:", "error:", ".byte", "invalid"} {
		if strings.Contains(text, bad) {
			t.Errorf("disassembler rejected part of the stream (%q):\n%s", bad, text)
		}
	}
	for _, want := range []string{
		"sdot", "ldur", "movi", "ushr", "scvtf", "fcvt", "fmul",
		"fmla", "faddp", "cbnz", "ret",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("disassembler never saw %q in the emitted stream", want)
		}
	}
	t.Logf("llvm-mc round-trip clean over %d instructions", len(stream)/4)
}

// TestA64Labels: a backward loop branch must land exactly on its target, and a
// dangling label must be impossible to get bytes out of.
func TestA64Labels(t *testing.T) {
	var a A64
	top := a.Label()
	a.Bind(top)
	a.SDOT(V0, V1, V2)  // 4e829420
	a.SUBimm(X9, X9, 1) // d1000529
	a.CBNZ(X9, top)     // b5ffffc9, imm19 = -2 instructions
	a.RET()             // d65f03c0
	if got := hex.EncodeToString(a.Bytes()); got != "2094824e290500d1c9ffffb5c0035fd6" {
		t.Errorf("loop encoded as %s", got)
	}

	// Forward branch: same four bytes reserved, patched in place. There is no
	// short-vs-near choice to make on A64, so code size never changes at Bind.
	var f A64
	done := f.Label()
	f.CBZ(X5, done)
	f.SDOT(V0, V1, V2)
	f.SDOT(V0, V1, V2)
	f.Bind(done)
	f.RET()
	b := f.Bytes()
	if len(b) != 16 {
		t.Fatalf("got %d bytes, want 16", len(b))
	}
	w := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
	if d := (w >> 5) & 0x7FFFF; d != 3 {
		t.Errorf("forward branch displacement is %d instructions, want 3", d)
	}

	var d A64
	d.CBNZ(X0, d.Label()) // never bound
	func() {
		defer func() {
			if recover() == nil {
				t.Error("Bytes() returned code with an unbound label")
			}
		}()
		d.Bytes()
	}()
}

// TestA64RejectsBadOffsets: a scaled offset that is not a multiple of its
// access size would silently truncate to a different valid address ((8/16)<<10
// is 0), so the emitter must panic instead.
func TestA64RejectsBadOffsets(t *testing.T) {
	for _, tc := range []struct {
		name string
		emit func(*A64)
	}{
		{"ldr q misaligned", func(a *A64) { a.LDRq(V0, X0, 8) }},
		{"ldr q too far", func(a *A64) { a.LDRq(V0, X0, 65536) }},
		{"ldr q negative", func(a *A64) { a.LDRq(V0, X0, -16) }},
		{"ldr x misaligned", func(a *A64) { a.LDRx(X1, X0, 4) }},
		{"ldr x too far", func(a *A64) { a.LDRx(X1, X0, 32768) }},
		{"ldr h misaligned", func(a *A64) { a.LDRh(V1, X0, 1) }},
		{"ldr s misaligned", func(a *A64) { a.LDRs(V1, X0, 2) }},
		{"ldur past +255", func(a *A64) { a.LDURq(V0, X0, 256) }},
		{"ldur past -256", func(a *A64) { a.LDURq(V0, X0, -257) }},
		{"add imm too large", func(a *A64) { a.ADDimm(X0, X0, 4096) }},
		{"ushr shift 0", func(a *A64) { a.USHR16b(V0, V1, 0) }},
		{"ushr shift 9", func(a *A64) { a.USHR16b(V0, V1, 9) }},
		{"fmla element 4", func(a *A64) { a.FMLAelem(V0, V1, V2, 4) }},
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: emitted silently instead of panicking", tc.name)
				}
			}()
			var a A64
			tc.emit(&a)
		}()
	}
}

func lookAny(names ...string) string {
	for _, n := range names {
		if p, err := exec.LookPath(n); err == nil {
			return p
		}
	}
	return ""
}

// TestMOVimmShifted pins the shifted MOVZ encodings. MOVZ's hw field (bits
// 22:21) selects a 0/16/32/48 shift, and a wrong one loads a value off by a
// factor of 65536 without faulting. Expected words are from the ARM manual.
func TestMOVimmShifted(t *testing.T) {
	for _, c := range []struct {
		imm  int64
		want uint32
	}{
		{0, 0xD2800000},           // movz x0, #0
		{1, 0xD2800020},           // movz x0, #1
		{0xFFFF, 0xD29FFFE0},      // movz x0, #65535
		{0x41000000, 0xD2A82000},  // movz x0, #0x4100, lsl #16   -- 8.0f
		{0x10000, 0xD2A00020},     // movz x0, #1, lsl #16
		{0x100000000, 0xD2C00020}, // movz x0, #1, lsl #32
	} {
		var a A64
		a.MOVimm(X0, c.imm)
		got := binary.LittleEndian.Uint32(a.Bytes())
		if got != c.want {
			t.Errorf("MOVimm(x0, %#x) = %#08x, want %#08x", c.imm, got, c.want)
		}
	}
	// A two-field immediate is built as MOVZ then MOVK per non-zero field
	// (conv1d and the packed tiled matmul need them).
	func() {
		var a A64
		a.MOVimm(X0, 0x12345)
		if n := len(a.Bytes()); n != 8 {
			t.Errorf("MOVimm(x0, 0x12345) emitted %d bytes, want 8 (movz + movk)", n)
		}
	}()
}
