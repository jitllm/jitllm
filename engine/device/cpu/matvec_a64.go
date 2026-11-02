package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"
)

// a64KernelTypes is the quantized formats with a NEON kernel. Tests iterate it
// assuming block-sized weights, int8 activations and a KernelConst, so F32 and
// the other float formats are handled separately in SupportedA64.
var a64KernelTypes = []quant.Type{quant.Q4_0, quant.Q5_0, quant.Q8_0, quant.Q3_K, quant.Q4_K, quant.Q5_K, quant.Q6_K}

// SupportedA64 reports whether EmitA64 has a kernel for t.
func SupportedA64(t quant.Type) bool {
	// The float formats are deliberately not in a64KernelTypes (see there).
	if t == quant.F32 || t == quant.F16 || t == quant.BF16 {
		// Baseline ARMv8-A: LDRq, FMLA4s, FADDP4s, FADDPs, UBFM, FCVTL for
		// the half form and SHLL for bfloat16. No feature to probe.
		return true
	}
	for _, k := range a64KernelTypes {
		if k == t {
			// The FEAT_DotProd check is in SupportedNative, not here: EmitA64
			// is host-pure so that NEON can be emitted and disassembled from an
			// x86 box.
			return true
		}
	}
	return false
}

// EmitA64 generates an arm64/NEON decode matvec kernel:
// out[r] = dot(row r of quantized weights, int8 activations).
//
// Host-pure, like Emit. It takes the same Spec and Args as the x86 kernels and
// consumes the same QuantizeQ8 output, but never reads the bias half of the
// AScale pairs (see emitA64MatVec).
func EmitA64(s Spec) ([]byte, error) {
	if !SupportedA64(s.W) {
		return nil, fmt.Errorf("jit: EmitA64: %s has no arm64 kernel", s.W)
	}
	if s.Cols != 1 {
		return nil, fmt.Errorf("jit: EmitA64: Cols=%d unsupported (decode matvec only)", s.Cols)
	}
	if s.Rows != 1 {
		return nil, fmt.Errorf("jit: EmitA64: Rows=%d unsupported (one row per iteration; "+
			"interleaving is not ported yet)", s.Rows)
	}
	var code []byte
	if s.W == quant.F16 {
		code = emitA64F16MatVec()
		if err := checkBudget(s, len(code)); err != nil {
			return nil, err
		}
		return code, nil
	}
	if s.W == quant.BF16 {
		code = emitA64BF16MatVec()
		if err := checkBudget(s, len(code)); err != nil {
			return nil, err
		}
		return code, nil
	}
	if s.W == quant.F32 {
		code = emitA64F32MatVec()
		if err := checkBudget(s, len(code)); err != nil {
			return nil, err
		}
		return code, nil
	}
	if s.W.BlockElems() == 256 {
		var err error
		if code, err = emitA64KMatVec(s.W, int(s.ActWin)); err != nil {
			return nil, err
		}
	} else {
		code = emitA64MatVec(s.W)
	}
	if err := checkBudget(s, len(code)); err != nil {
		return nil, err
	}
	return code, nil
}

// Kernel register assignment, fixed by hand. With 32 V registers nothing
// spills, and nothing stages below SP: AAPCS64 has no red zone (signal
// delivery clobbers it), so the amd64 kernels' red-zone spills must not be
// ported.
//
//	x0   *Args on entry; dead after the prologue
//	x1   Out cursor            x2  weight cursor
//	x3   A base                x4  AScale base
//	x5   row counter           x6  blocks per row
//	x7   activation cursor     x8  scale-pair cursor
//	x9   block counter
//
//	v0   float32 accumulator, 4 lanes
//	v1   d_w, then d_w*d_x, in lane 0
//	v2-v6  weights and activations
//	v7   int32 dot (Q4_0 and Q5_0; Q8_0 uses v6)
//	v16  d_x
//	v17-v20  the qh bytes and their fanned-out bit planes  (Q5_0 only)
//	v26  qh fan-out table, v27 per-lane shift, v28 the constant 0x10
//	                                            (Q5_0 only, loop-invariant)
//	v29  0x0F nibble mask, v30 the zero point -- 8 for Q4_0, 16 for Q5_0
//	                                      (Q4_0 and Q5_0, loop-invariant)
//
// v8-v15 hold nothing but v8/v9, a widened SDOT's scratch on a chip without
// FEAT_DotProd (sdotemu.go). Go's arm64 ABI saves no V register
// (trampoline_arm64.s), so using them costs nothing. x18/x27/x28/x29/x30/sp are
// never written.
//
// Three x86 mechanisms do not exist here:
//
//  1. No bias correction. SDOT is signed x signed, so Q8_0 weights are used as
//     loaded and Q4_0 needs one SUB #8. The kernel never reads pairs[2b+1],
//     which makes it correct for any biasC the caller passed to QuantizeQ8
//     (the amd64 value is divided by 8 for eight AVX2 lanes; NEON has four,
//     so reusing it would apply half a correction).
//
//  2. Uniform constants are built with MOVI, so Q4_0 and Q8_0 read no constant
//     block. Q5_0 needs two per-lane vectors MOVI cannot express (a TBL table
//     and a USHL count), read once from KernelConst(Q5_0) through Args.Scr.
//
//  3. No broadcasts: LDR S lands d_x in lane 0 and the by-element FMLA reads
//     it from there.
//
// Q4_0 stores elements 0..15 as the low nibbles and 16..31 as the high
// nibbles of the same sixteen bytes, not interleaved; AND and USHR of one
// register land the two halves in the two registers the SDOTs want.
func emitA64MatVec(t quant.Type) []byte {
	bb := int32(t.BlockBytes()) // 18 for Q4_0, 22 for Q5_0, 34 for Q8_0
	var a A64
	// See the register map above.
	a.DotScratch(V8, V9)

	// Prologue. The single argument *Args arrives in x0 (AAPCS64's first
	// integer argument, the analogue of amd64's RDI).
	a.LDRx(X1, X0, 0)  // Out
	a.LDRx(X2, X0, 8)  // W
	a.LDRx(X3, X0, 16) // A
	a.LDRx(X4, X0, 24) // AScale -> {d_x, unused} pairs
	a.LDRx(X5, X0, 32) // Rows
	a.LDRx(X6, X0, 40) // K, in BLOCKS per row
	switch t {
	case quant.Q4_0:
		a.MOVI16b(V29, 0x0F) // nibble mask
		a.MOVI16b(V30, 8)    // the Q4_0 zero point
	case quant.Q5_0:
		// The two non-uniform vectors, which MOVI cannot build, come from the
		// constant block through Args.Scr. x10 is dead after this: the block is
		// read once and the registers are loop-invariant.
		a.LDRx(X10, X0, 56)  // Scr -> KernelConst(Q5_0)
		a.LDRq(V26, X10, 0)  // qh fan-out table, {0 x8, 1 x8}
		a.LDRq(V27, X10, 16) // per-lane shift, {4,3,2,1,0,-1,-2,-3} twice
		a.MOVI16b(V28, 0x10) // the fifth bit, once USHL has aligned it
		a.MOVI16b(V29, 0x0F) // nibble mask
		a.MOVI16b(V30, 16)   // the Q5_0 zero point -- 16, NOT Q4_0's 8
	}

	row := a.Label()
	a.Bind(row)
	a.MOVIzero(V0)   // float32 accumulator
	a.MOVreg(X7, X3) // activation cursor
	a.MOVreg(X8, X4) // pair cursor
	a.MOVreg(X9, X6) // blocks remaining

	blk := a.Label()
	a.Bind(blk)
	a.LDRh(V1, X2, 0) // d_w, the f16 at the head of the block

	// A quant block is 32 elements and SDOT consumes 16 bytes, so every block
	// takes two dot products into the same four int32 lanes.
	dot := V6
	switch t {
	case quant.Q4_0:
		dot = V7
		// The payload is at +2, which is not a multiple of 16, so LDR Q cannot
		// address it and LDUR must be used; that is also why the amd64 baked
		// row-stride trick is unencodable here.
		a.LDURq(V2, X2, 2)
		a.AND16b(V3, V2, V29) // low nibbles  = elements 0..15
		a.USHR16b(V4, V2, 4)  // high nibbles = elements 16..31
		a.SUB16b(V3, V3, V30) // unsigned 0..15 -> signed -8..7
		a.SUB16b(V4, V4, V30)
		a.LDRq(V5, X7, 0)
		a.LDRq(V6, X7, 16)
		a.MOVIzero(dot)
		a.SDOT(dot, V3, V5)
		a.SDOT(dot, V4, V6)
	case quant.Q5_0:
		// Q5_0: Q4_0's unpack plus a fifth bit per element. A block is
		// { f16 d; u32 qh; u8 qs[16] } (22 bytes) and element i is
		// (qs_nibble_i | ((qh>>i)&1)<<4) - 16, times d, with the nibbles laid
		// out as Q4_0's. The zero point is 16, not 8.
		//
		// The bit index is the element's own, so no immediate shift reaches it:
		// TBL fans one qh byte across eight lanes, then USHL16b moves lane j's
		// bit (j%8) to position 4, so one AND with 0x10 yields the fifth bit
		// already weighted and one ORR merges it into the nibble.
		dot = V7
		a.LDURq(V2, X2, 6) // the 16 nibble bytes, at +6 and so LDUR again
		a.LDRh(V17, X2, 2) // qh bytes 0,1 -> the fifth bits of elements 0..15
		a.LDRh(V18, X2, 4) // qh bytes 2,3 -> elements 16..31
		a.AND16b(V3, V2, V29)
		a.USHR16b(V4, V2, 4)
		a.TBL(V19, V17, V26)     // lane j = the qh byte holding bit j
		a.TBL(V20, V18, V26)     // ... and bit j+16
		a.USHL16b(V19, V19, V27) // lane j: bit (j%8) -> bit 4
		a.USHL16b(V20, V20, V27)
		a.AND16b(V19, V19, V28) // 0 or 16, no further shift needed
		a.AND16b(V20, V20, V28)
		a.ORR16b(V3, V3, V19) // q = nibble | fifth bit, unsigned 0..31
		a.ORR16b(V4, V4, V20)
		a.SUB16b(V3, V3, V30) // -> signed -16..15
		a.SUB16b(V4, V4, V30)
		a.LDRq(V5, X7, 0)
		a.LDRq(V6, X7, 16)
		a.MOVIzero(dot)
		a.SDOT(dot, V3, V5)
		a.SDOT(dot, V4, V6)
	case quant.Q8_0:
		// Q8_0: 32 signed int8 weights, used exactly as stored. No unpack and no
		// bias, which is why this format sits closest to the memory wall.
		a.LDURq(V2, X2, 2)
		a.LDURq(V3, X2, 18)
		a.LDRq(V4, X7, 0)
		a.LDRq(V5, X7, 16)
		a.MOVIzero(dot)
		a.SDOT(dot, V2, V4)
		a.SDOT(dot, V3, V5)
	default:
		// Named arms and a refusal: a format falling into another format's arm
		// runs at full speed and is silently wrong.
		panic(fmt.Sprintf("jit: emitA64MatVec: %s is in a64KernelTypes and has no "+
			"32-element arm -- add one rather than letting it decode as Q8_0", t))
	}

	a.SCVTF4s(dot, dot)        // int32 lanes -> float32
	a.FCVTsh(V1, V1)           // d_w: f16 -> f32, in lane 0
	a.LDRs(V16, X8, 0)         // d_x, straight into lane 0
	a.FMULs(V1, V1, V16)       // combined scale d_w*d_x, still scalar
	a.FMLAelem(V0, dot, V1, 0) // acc += dot * scale, broadcast from lane 0

	a.ADDimm(X2, X2, bb)      // one weight block
	a.ADDimm(X7, X7, Q8Block) // 32 activation bytes
	a.ADDimm(X8, X8, 8)       // one {d_x, unused} pair
	a.SUBimm(X9, X9, 1)
	a.CBNZ(X9, blk)

	// Horizontal sum of four float32 lanes. Base ARMv8 has no FADDV (SVE only),
	// and ADDV is integer: pointed at a float accumulator it yields garbage.
	a.FADDP4s(V0, V0, V0)
	a.FADDPs(V0, V0)
	a.STRs(V0, X1, 0)

	a.ADDimm(X1, X1, 4)
	a.SUBimm(X5, X5, 1)
	a.CBNZ(X5, row)

	// No VZEROUPPER analogue: NEON has no AVX-SSE transition penalty, and no
	// register needs restoring because Go's arm64 ABI has no callee-saved set.
	a.RET()
	return a.Bytes()
}

// emitA64F16MatVec is emitA64FloatMatVec for f16 weights: one LDRq brings
// eight halves and FCVTL/FCVTL2 widen them into two 4S vectors (Mixtral's
// router is F16).
func emitA64F16MatVec() []byte { return emitA64FloatMatVec(quant.F16) }

func emitA64F32MatVec() []byte  { return emitA64FloatMatVec(quant.F32) }
func emitA64BF16MatVec() []byte { return emitA64FloatMatVec(quant.BF16) }

// emitA64FloatMatVec is the NEON twin of emitFloatMatVec: row-major F32, F16 or
// BF16 against f32 activations at any k -- groups of 32 in four independent
// chains, then whole quads into chain 0, then single elements. arm64 has no
// HADDPS, so the chains reduce pairwise with FADDP, in a different (still
// deterministic) order from the x86 kernel's.
func emitA64FloatMatVec(t quant.Type) []byte {
	var a A64
	esz := int32(4)
	if t != quant.F32 {
		esz = 2
	}
	// quad loads four weights as f32 into w, using raw as the 16-bit staging
	// register; one loads a single weight into lane 0 with the rest zero.
	quad := func(w, raw VReg, off int32) {
		switch t {
		case quant.F32:
			a.LDRq(w, X2, off)
		case quant.F16:
			a.LDRd(raw, X2, off)
			a.FCVTL(w, raw)
		default:
			a.LDRd(raw, X2, off)
			a.SHLL(w, raw)
		}
	}
	one := func(w VReg) {
		switch t {
		case quant.F32:
			a.LDRs(w, X2, 0)
		case quant.F16:
			a.LDRh(w, X2, 0)
			a.FCVTL(w, w)
		default:
			a.LDRh(w, X2, 0)
			a.SHLL(w, w)
		}
	}

	a.LDRx(X1, X0, 0)  // Out
	a.LDRx(X2, X0, 8)  // W, walked continuously across rows
	a.LDRx(X3, X0, 16) // A, reinterpreted as float32
	a.LDRx(X5, X0, 32) // Rows
	a.LDRx(X6, X0, 40) // K, in elements
	a.LSRimm(X10, X6, 2)
	a.ANDlow(X10, X10, 3) // whole quads past the groups of 32
	a.ANDlow(XReg(11), X6, 2)
	a.LSRimm(X6, X6, 5) // groups of 32

	row := a.Label()
	a.Bind(row)
	a.MOVIzero(V0)
	a.MOVIzero(V1)
	a.MOVIzero(V2)
	a.MOVIzero(V3)
	a.MOVreg(X7, X3) // activation cursor, rewound every row

	loop := func(n XReg, body func(), wstep, xstep int32) {
		skip := a.Label()
		a.MOVreg(X9, n)
		a.CBZ(X9, skip)
		lp := a.Label()
		a.Bind(lp)
		body()
		a.ADDimm(X2, X2, wstep)
		a.ADDimm(X7, X7, xstep)
		a.SUBimm(X9, X9, 1)
		a.CBNZ(X9, lp)
		a.Bind(skip)
	}
	loop(X6, func() {
		// 32 elements: eight quads, four accumulators, so each chain carries
		// two independent FMLAs per iteration.
		for j := 0; j < 8; j++ {
			w := VReg(int(V4) + j)  // V4..V11
			x := VReg(int(V12) + j) // V12..V19
			if t == quant.F32 {
				a.LDRq(w, X2, int32(16*j))
			} else if j%2 == 0 {
				raw := VReg(int(V20) + j/2) // eight halves per load
				a.LDRq(raw, X2, int32(8*j))
				if t == quant.F16 {
					a.FCVTL(w, raw)
					a.FCVTL2(VReg(int(w)+1), raw)
				} else {
					a.SHLL(w, raw)
					a.SHLL2(VReg(int(w)+1), raw)
				}
			}
			a.LDRq(x, X7, int32(16*j))
		}
		for j := 0; j < 8; j++ {
			a.FMLA4s(VReg(int(V0)+j%4), VReg(int(V4)+j), VReg(int(V12)+j))
		}
	}, 32*esz, 128)
	loop(X10, func() {
		quad(V4, V20, 0)
		a.LDRq(V12, X7, 0)
		a.FMLA4s(V0, V4, V12)
	}, 4*esz, 16)
	loop(XReg(11), func() {
		one(V4)
		a.LDRs(V12, X7, 0)
		a.FMLA4s(V0, V4, V12)
	}, esz, 4)

	a.FADDP4s(V0, V0, V1)
	a.FADDP4s(V2, V2, V3)
	a.FADDP4s(V0, V0, V2) // [sum(V0), sum(V1), sum(V2), sum(V3)]
	a.FADDP4s(V0, V0, V0)
	a.FADDPs(V0, V0)
	a.STRs(V0, X1, 0)

	a.ADDimm(X1, X1, 4)
	a.SUBimm(X5, X5, 1)
	a.CBNZ(X5, row)
	a.RET()
	return a.Bytes()
}
