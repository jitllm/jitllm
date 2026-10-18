package cpu

import (
	"fmt"
	"runtime"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// Spec is the entire intermediate representation: a few fields, no graph, no
// instruction stream, no register allocator. Register allocation for tensor
// kernels is template instantiation; the only real decision is the tile shape,
// which is a parameter.
//
// Adding a field requires a paired measurement and a statement of what it buys
// (AGENTS.md RULE 6); this is where a tensor compiler would start growing. The
// row count is deliberately absent; nb, the blocks per row, is a codegen
// parameter of EmitInterleaved, which is why nn.JIT keys kernels on (type, nb).
type Spec struct {
	W    quant.Type
	Rows int8 // output rows per inner iteration
	Accs int8 // accumulator chains
	Cols int8 // activation columns: 1 = decode matvec
	// ActWin is how many activation elements share one quantization scale: 0 or
	// Q8Block for the usual per-32 scales, 256 when the model is all k-quants
	// and nn.NewJIT widened the amax. With one scale per super-block the kernel
	// hoists a load and a multiply out of its sub-block loop.
	ActWin int16
}

// Q8Block is the activation quantization group. It matches Q4_0's 32 elements so
// one weight block pairs with exactly one activation block.
const Q8Block = 32

// WideActWindow is the activation amax window to use for a model whose matvec
// types are these. It is a property of the kernel that will be generated, not
// of the target, so it is decided here at runtime.
//
// Two conditions: every matvec type must be a k-quant (a wider amax is a
// coarser scale, and narrow-block formats depend on the narrow one), and the
// native kernel must hoist the scale out of its sub-block loop, or the coarser
// scale costs accuracy for nothing. The second is asserted rather than
// measured. Narrow blocks are read off the descriptor, not a list, so every
// 32-wide format (Q5_0, MXFP4, ...) keeps the narrow window.
func WideActWindow(types []quant.Type) int {
	any := false
	for _, t := range types {
		q, ok := kernels.QuantOf(t)
		if !ok {
			continue
		}
		if kernels.NarrowScales(q) {
			return Q8Block // a 32-wide weight block wants a 32-wide scale
		}
		any = true
	}
	if !any || !nativeHoistsActScale() {
		return Q8Block
	}
	return 256
}

// nativeHoistsActScale reports whether a k-quant kernel this process generates
// hoists the activation scale out of its sub-block loop, so that a wider window
// buys something rather than only costing accuracy.
//
// True on both architectures, for different kernels: arm64's decode matvec
// hoists it (emitA64KMatVec); on amd64 the payer is the Q3_K prefill kernel,
// whose fold becomes integer once d_a is constant over a super-block
// (emitGEMMQ3K). The window is one decision per session because prefill and
// decode write the same KV cache and must quantize identically.
func nativeHoistsActScale() bool {
	return runtime.GOARCH == "arm64" || runtime.GOARCH == "amd64"
}

// BiasC is the unsigned-bias constant for a format, the value QuantizeQ8 needs.
func BiasC(t quant.Type) float64 {
	// Derived from the format's descriptor, not a case per format: a zero here
	// is not a refusal, it silently drops the -bias*scale*sum(a) term (Q5_0
	// once fell through a switch that way). The three cases are the whole
	// rule:
	//
	//	a format with its own per-sub-block MINIMUM supplies dmin*m_g itself
	//	and needs only -sum/8 here, so the constant is 1 (Q4_K, Q5_K);
	//	a SIGNED payload is centred by flipping the sign bit, which is an
	//	offset of 128 (Q8_0);
	//	otherwise the bias is the descriptor's biasK -- 8 for Q4_0, 4 for
	//	Q3_K, 16 for Q5_0, 32 for Q6_K.
	//
	// The 8 in every case cancels against the row-major kernel's eight-lane
	// horizontal reduction; see PackedScratch.
	q, ok := kernels.QuantOf(t)
	if !ok {
		return 0
	}
	_, _, biasK, biasArray := kernels.Layout(q)
	switch {
	case biasArray:
		return 1
	case kernels.SignedPayload(q):
		return 128
	}
	return float64(biasK)
}

// KernelConst is the constant block a format's kernel loads at entry.
//
// Q4_0 and Q8_0 need one 16-byte value each: a nibble mask, and a sign-flip
// (XOR 0x80 maps int8 to the unsigned biased form VPDPBUSD's first operand
// requires).
//
// Q4_K needs five, because its eight 6-bit sub-block scales and eight 6-bit
// mins are packed across twelve bytes in an order no arithmetic reaches:
//
//	sc[0..3] = b0..b3 & 63
//	sc[4..7] = (b8..b11 & 15) | ((b0..b3 >> 6) << 4)
//	m [0..3] = b4..b7 & 63
//	m [4..7] = (b8..b11 >> 4) | ((b4..b7 >> 6) << 4)
//
// Two shuffles gather the operands so all four cases become the same three
// expressions on different dword lanes, which VPBLENDD then selects between.
// Note (x >> 6) << 4 collapses to (x >> 2) & 0x30 — one shift, not two.
func KernelConst(t quant.Type) []byte {
	switch t {
	case quant.Q4_0:
		return rep16(0x0F)
	case quant.Q8_0:
		return rep16(0x80)
	case quant.Q6_K:
		// 0x0F for the low nibble, 0x03 for the two high bits from qh, then a
		// 32-byte 0x30 for the prefill GEMM, which folds the shift and the mask
		// into one step. APPENDED: the decode kernel reads offsets 0 and 16 by
		// baked displacement, so nothing before this may move.
		out := append(rep16(0x0F), rep16(0x03)...)
		return append(out, rep32(0x30)...)
	case quant.Q3_K:
		// Q3_K packs its sixteen 6-bit scales across twelve bytes in a THIRD
		// arrangement, different again from Q4_K's:
		//
		//	sc[0:4]   = (b0..b3  & 0x0F) | ((b8..b11 >> 0) & 3) << 4
		//	sc[4:8]   = (b4..b7  & 0x0F) | ((b8..b11 >> 2) & 3) << 4
		//	sc[8:12]  = ((b0..b3 >> 4)  ) | ((b8..b11 >> 4) & 3) << 4
		//	sc[12:16] = ((b4..b7 >> 4)  ) | ((b8..b11 >> 6) & 3) << 4
		//
		// Every dword wants a DIFFERENT shift of the same donor bytes, so the
		// donors are broadcast once and four shifted copies blended together.
		// All constants are 32 bytes so they serve both as VBROADCASTI128
		// sources and as 256-bit memory operands -- the format needs more
		// distinct masks than the register file has room for.
		out := make([]byte, 0, 224)
		out = append(out, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7) // shufA
		out = append(out, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7)
		out = append(out, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11) // shufB
		out = append(out, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11)
		// 0x04 is appended for the prefill GEMM, which folds the high-mask bit
		// straight into bit 2. Appended: the decode kernel addresses every
		// constant before it by baked displacement.
		for _, v := range []byte{0x0F, 0x30, 0x03, 0x01, 0x20, 0x04} {
			out = append(out, rep32(v)...)
		}
		return out
	case quant.Q5_K:
		// Q4_K's constants plus a 0x10 mask at offset 160 for the fifth bit.
		out := append(KernelConst(quant.Q4_K), rep32(0x10)...)
		return out
	case quant.Q5_0:
		// Two vectors the arm64 kernel reads. Uniform masks (0x0F, 0x10, the
		// zero point 16) are built with MOVI; these two are non-uniform across
		// lanes, which MOVI cannot express:
		//
		//	offset 0   a TBL table that fans one qh byte out to eight lanes,
		//	           {0 x8, 1 x8}, so lane j holds the qh byte holding bit j
		//	offset 16  a per-lane USHL count, {4,3,2,1,0,-1,-2,-3} twice, which
		//	           moves lane j's own bit (j%8) to position 4 -- so a single
		//	           AND with 0x10 yields the fifth bit already weighted
		//
		// TestEveryA64KernelTypeHasAConstantBlock is the invariant on the size.
		out := make([]byte, 0, 32)
		for j := 0; j < 16; j++ {
			out = append(out, byte(j/8))
		}
		for j := 0; j < 16; j++ {
			out = append(out, byte(int8(4-j%8)))
		}
		return out
	case quant.Q4_K:
		// Every constant is 32 bytes, following Q3_K: that lets the same bytes
		// serve as a VBROADCASTI128 source and as a 256-bit memory operand, so
		// the two masks used only in the per-super-block prologue (0x3F, 0x30)
		// need no register at all. The two registers that frees are what the
		// accumulator chains in emitKQuantMatVec are built from.
		out := make([]byte, 0, 160)
		// shufA gathers [b0-3 | b8-11 | b4-7 | b8-11]
		shufA := []byte{0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11}
		// shufB gathers the donors of the high two bits: [_ | b0-3 | _ | b4-7]
		shufB := []byte{0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 7}
		out = append(append(out, shufA...), shufA...)
		out = append(append(out, shufB...), shufB...)
		out = append(out, rep32(0x3F)...)
		out = append(out, rep32(0x0F)...)
		out = append(out, rep32(0x30)...)
		return out
	}
	return nil
}

func rep32(v byte) []byte {
	m := make([]byte, 32)
	for i := range m {
		m[i] = v
	}
	return m
}

func rep16(v byte) []byte {
	m := make([]byte, 16)
	for i := range m {
		m[i] = v
	}
	return m
}

// BlocksPerRow is the loop count a kernel expects in Args.K: how many WEIGHT
// blocks make up one row. For Q4_0 and Q8_0 that is a 32-element block; for the
// k-quants it is a 256-element super-block, which spans eight activation blocks.
func BlocksPerRow(t quant.Type, k int) int { return k / int(t.BlockElems()) }

// RowBytes is the on-disk size of one weight row.
func RowBytes(t quant.Type, k int) int {
	return k / int(t.BlockElems()) * int(t.BlockBytes())
}

// NeedsHalfSums reports whether a format's sub-blocks are 16 elements wide, so
// its kernel needs per-16 activation sums in Args.AHalf rather than only the
// per-32 sums in AScale.
func NeedsHalfSums(t quant.Type) bool { return t == quant.Q6_K || t == quant.Q3_K }

// Supported reports whether Emit has a kernel for t.
func Supported(t quant.Type) bool {
	switch t {
	case quant.Q4_0, quant.Q8_0, quant.Q4_K, quant.Q5_K, quant.Q6_K, quant.Q3_K:
		// No CPU-feature check, deliberately: Emit is host-pure so golden
		// disassembly tests can cross-emit. The runtime gate is SupportedNative.
		return true
	case quant.F16:
		// Mixtral's router (ffn_gate_inp) is F16.
		return true
	case quant.BF16:
		// Qwen3-Next's shared-expert gate, the one BF16 tensor a model
		// carries; see emitBF16MatVec.
		return true
	case quant.F32:
		// Mixture routers (ffn_gate_inp) are F32 in most files; the device
		// generates a RouterMatVec for them and so does the host.
		return true
	}
	return false
}

// Emit generates a kernel for s. Host-pure: it depends on nothing but s, which
// is what makes golden-disassembly tests possible at all.
func Emit(s Spec) ([]byte, error) { return emitPF(s, 0) }

// emitPF is Emit with the k-quant kernels' PREFETCHT0 distance in super-blocks
// (EmitOpts.Prefetch); 0 emits none.
func emitPF(s Spec, pf int) ([]byte, error) {
	if !Supported(s.W) {
		return nil, fmt.Errorf("jit: Emit: %s has no kernel yet (Q4_0 and Q8_0 only)", s.W)
	}
	if s.Cols != 1 {
		return nil, fmt.Errorf("jit: Emit: Cols=%d unsupported (decode matvec only)", s.Cols)
	}
	if s.Rows != 1 && s.Rows != 4 {
		return nil, fmt.Errorf("jit: Emit: Rows=%d unsupported (1 or 4)", s.Rows)
	}
	if s.W == quant.F16 {
		if s.Rows != 1 {
			return nil, fmt.Errorf("jit: Emit: F16 has no interleaved kernel")
		}
		code := emitF16MatVec()
		if err := checkBudget(s, len(code)); err != nil {
			return nil, err
		}
		return code, nil
	}
	if s.W == quant.BF16 {
		if s.Rows != 1 {
			return nil, fmt.Errorf("jit: Emit: BF16 has no interleaved kernel")
		}
		code := emitBF16MatVec()
		if err := checkBudget(s, len(code)); err != nil {
			return nil, err
		}
		return code, nil
	}
	if s.W == quant.F32 {
		if s.Rows != 1 {
			return nil, fmt.Errorf("jit: Emit: F32 has no interleaved kernel")
		}
		code := emitF32MatVec()
		if err := checkBudget(s, len(code)); err != nil {
			return nil, err
		}
		return code, nil
	}
	if s.W == quant.Q3_K {
		if s.Rows != 1 {
			return nil, fmt.Errorf("jit: Emit: Q3_K has no interleaved kernel yet")
		}
		code := emitQ3KMatVec(pf)
		if err := checkBudget(s, len(code)); err != nil {
			return nil, err
		}
		return code, nil
	}
	if s.W == quant.Q6_K {
		if s.Rows != 1 {
			return nil, fmt.Errorf("jit: Emit: Q6_K has no interleaved kernel yet")
		}
		code := emitQ6KMatVec()
		if err := checkBudget(s, len(code)); err != nil {
			return nil, err
		}
		return code, nil
	}
	if s.W == quant.Q4_K || s.W == quant.Q5_K {
		// There is no interleaved k-quant kernel; emitKQuantMatVec is
		// single-row, so the guard must stay.
		if s.Rows != 1 {
			return nil, fmt.Errorf("jit: Emit: %s has no interleaved kernel yet", s.W)
		}
		code := emitKQuantMatVec(s.W, int(s.Accs), pf)
		if err := checkBudget(s, len(code)); err != nil {
			return nil, err
		}
		return code, nil
	}
	code := emitMatVec(s.W, int(s.Rows))
	if err := checkBudget(s, len(code)); err != nil {
		return nil, err
	}
	return code, nil
}

// emitMatVec generates out[r] = dot(row r of quantized weights, int8 activations).
//
// Register assignment, fixed by hand (AVX2 has 16 YMM; a tile that spills
// loses badly, so the budget is respected rather than tested):
//
//	Y0 accumulator (float32)   Y5 dot as float32
//	Y1 nibble scratch          Y6 combined scale d_w*d_x
//	Y2 shifted scratch         Y7 broadcast temp
//	Y3 unpacked nibbles        Y8 0x0F mask, loop-invariant
//	Y4 integer dot
//
// GP: RDX weights, RCX out, RSI activation base, R8 pair base, RBX constants,
// RAX row counter, R10 block counter, R11 activation cursor, RDI pair cursor.
// RBX is free because the trampoline saves it; R12-R15 are not touched (R14 is
// the current goroutine under Go's register ABI).
func emitMatVec(t quant.Type, rows int) []byte {
	if rows == 4 {
		return emitMatVec4(t)
	}
	blockBytes := int32(t.BlockBytes())
	var a Buf

	// Prologue. RDI holds *Args on entry and is recycled as a cursor after.
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 8))  // W
	a.MOVLoad(RSI, At(RDI, 16)) // A
	a.MOVLoad(R8, At(RDI, 24))  // AScale -> {scale, negSum} pairs
	a.MOVLoad(RAX, At(RDI, 32)) // Rows
	a.MOVLoad(R9, At(RDI, 40))  // K, in BLOCKS per row
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> 16-byte nibble mask
	a.VBROADCASTI128(Y8, At(RBX, 0))

	row := a.Label()
	a.Bind(row)
	a.VPXOR(Y0, Y0, Y0)
	a.MOVQ(R11, RSI) // activation cursor
	a.MOVQ(RDI, R8)  // pair cursor
	a.MOVQ(R10, R9)  // blocks remaining

	blk := a.Label()
	a.Bind(blk)
	// Unpack 16 packed bytes into 32 nibbles in element order. Q4_0 stores elements 0..15 as the low nibbles and 16..31 as the high
	// nibbles of the same 16 bytes — not interleaved 2j/2j+1. Broadcasting those
	// 16 bytes to both 128-bit lanes and then taking the raw lane 0 alongside the
	// shifted lane 1 lands both halves in the right order with no shuffle.
	if t == quant.Q4_0 {
		a.VBROADCASTI128(Y1, At(RDX, 2))
		a.VPSRLW(Y2, Y1, 4)
		a.VPBLENDD(Y1, Y1, Y2, 0xF0)
		a.VPAND(Y3, Y1, Y8)
	} else {
		// Q8_0: 32 signed int8 weights, biased to unsigned with one XOR.
		a.VMOVDQULoad(Y1, At(RDX, 2))
		a.VPXOR(Y3, Y1, Y8)
	}

	// Integer dot. Nibbles (0..15) are the unsigned operand and the int8
	// activations the signed one, which is the order VPDPBUSD requires and
	// the reason the -8 bias is handled on the activation side instead.
	a.VPXOR(Y4, Y4, Y4)
	a.VPDPBUSDMem(Y4, Y3, At(R11, 0))
	a.VCVTDQ2PS(Y5, Y4)

	// scale = d_w (f16, at the head of the block) * d_x (from the pair array).
	a.VCVTPH2PSx(Y6, At(RDX, 0))
	a.VBROADCASTSSReg(Y6, Y6)
	a.VBROADCASTSS(Y7, At(RDI, 0))
	a.VMULPS(Y6, Y6, Y7)

	a.VFMADD231PS(Y0, Y5, Y6)      // acc += dot * scale
	a.VBROADCASTSS(Y7, At(RDI, 4)) // -sum(q)
	a.VFMADD231PS(Y0, Y6, Y7)      // acc += scale * -sum(q), x8 via the reduction

	a.ADDimm(RDX, blockBytes) // one weight block
	a.ADDimm(R11, Q8Block)
	a.ADDimm(RDI, 8)
	a.DEC(R10)
	a.JNZ(blk)

	// Horizontal sum of the 8 float32 lanes, then store one result.
	a.VEXTRACTF128(Y1, Y0, 1)
	a.VADDPSx(Y0, Y0, Y1)
	a.VHADDPSx(Y0, Y0, Y0)
	a.VHADDPSx(Y0, Y0, Y0)
	a.VMOVSSStore(At(RCX, 0), Y0)

	a.ADDimm(RCX, 4)
	a.DEC(RAX)
	a.JNZ(row)

	// Mandatory: leaving the YMM upper halves dirty taxes every later SSE
	// instruction in the process, so the symptom shows up in code that never
	// called a kernel.
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// Interleave is how many rows emitMatVec4 processes per iteration.
const Interleave = 4

// emitMatVec4 is the interleaved kernel: four output rows advance together
// through the same activation blocks.
//
// The point is instruction count: six of the ~17 instructions per block depend
// only on the activation side and are shared by every row, so four rows
// amortize them 4x (17 -> 12.5 per row-block). Eight rows would need eight
// weight pointers, and the general-purpose file is the scarce resource here.
//
// Registers. GP: RCX out, RDX/R8/R9/R10 the four weight cursors, RSI activation
// cursor, RDI scale-pair cursor, R11 block counter, RAX group counter, RBX
// scratch. YMM: Y0-Y3 accumulators, Y4 activation scale, Y5 activation bias,
// Y6 weights, Y7 dot, Y8 the format constant, Y9 combined scale.
//
// The loop-invariant bases live in the RED ZONE ([RSP-128, RSP)), which System V
// reserves for leaf functions. This kernel makes no calls, so it may use it
// without touching RSP — which is what keeps the "kernels perform no pushes"
// property that lets RSP be identical on every path.
func emitMatVec4(t quant.Type) []byte {
	bb := int32(t.BlockBytes())
	var a Buf

	// Prologue. Rows is the GROUP count here: the caller passes rows/4 and
	// handles any remainder with the single-row kernel.
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 8))  // W, becomes w0
	a.MOVLoad(RSI, At(RDI, 16)) // A
	a.MOVLoad(RAX, At(RDI, 24)) // AScale
	a.MOVLoad(R11, At(RDI, 32)) // Rows, in groups of 4
	a.MOVStore(At(RSP, -8), RSI)
	a.MOVStore(At(RSP, -16), RAX)
	a.MOVLoad(RAX, At(RDI, 40)) // K, in blocks
	a.MOVStore(At(RSP, -24), RAX)
	a.MOVLoad(RBX, At(RDI, 48)) // RowStr
	a.MOVStore(At(RSP, -32), RBX)

	// w1..w3 = w0 + n*RowStr, with RBX still holding RowStr.
	a.MOVQ(R8, RDX)
	a.ADDQ(R8, RBX)
	a.MOVQ(R9, R8)
	a.ADDQ(R9, RBX)
	a.MOVQ(R10, R9)
	a.ADDQ(R10, RBX)

	a.MOVLoad(RBX, At(RDI, 56)) // Scr, the format constant
	a.VBROADCASTI128(Y8, At(RBX, 0))
	a.MOVQ(RAX, R11) // group counter

	ws := [4]Reg{RDX, R8, R9, R10}
	accs := [4]Reg{Y0, Y1, Y2, Y3}

	group := a.Label()
	a.Bind(group)
	a.MOVLoad(RSI, At(RSP, -8))  // activation cursor
	a.MOVLoad(RDI, At(RSP, -16)) // scale-pair cursor
	a.MOVLoad(R11, At(RSP, -24)) // blocks remaining
	for _, acc := range accs {
		a.VPXOR(acc, acc, acc)
	}

	blk := a.Label()
	a.Bind(blk)
	// Shared across all four rows: this is the whole point of interleaving.
	a.VBROADCASTSS(Y4, At(RDI, 0)) // activation scale
	a.VBROADCASTSS(Y5, At(RDI, 4)) // -sum(q)*biasC/8

	for r := 0; r < 4; r++ {
		w, acc := ws[r], accs[r]
		if t == quant.Q4_0 {
			a.VBROADCASTI128(Y6, At(w, 2))
			a.VPSRLW(Y7, Y6, 4)
			a.VPBLENDD(Y6, Y6, Y7, 0xF0)
			a.VPAND(Y6, Y6, Y8)
		} else {
			a.VMOVDQULoad(Y6, At(w, 2))
			a.VPXOR(Y6, Y6, Y8)
		}
		a.VPXOR(Y7, Y7, Y7)
		a.VPDPBUSDMem(Y7, Y6, At(RSI, 0))
		a.VCVTDQ2PS(Y7, Y7)
		a.VCVTPH2PSx(Y9, At(w, 0))
		a.VBROADCASTSSReg(Y9, Y9)
		a.VMULPS(Y9, Y9, Y4) // scale = d_w * d_x
		a.VFMADD231PS(acc, Y7, Y9)
		a.VFMADD231PS(acc, Y9, Y5)
	}
	for _, w := range ws {
		a.ADDimm(w, bb)
	}
	a.ADDimm(RSI, Q8Block)
	a.ADDimm(RDI, 8)
	a.DEC(R11)
	a.JNZ(blk)

	// Four horizontal sums, one per row.
	for r := 0; r < 4; r++ {
		acc := accs[r]
		a.VEXTRACTF128(Y4, acc, 1)
		a.VADDPSx(acc, acc, Y4)
		a.VHADDPSx(acc, acc, acc)
		a.VHADDPSx(acc, acc, acc)
		a.VMOVSSStore(At(RCX, int32(r*4)), acc)
	}
	a.ADDimm(RCX, 16)

	// Next group. Every cursor advanced by exactly RowStr during the block loop,
	// so w3 now points at the next group's first row and the others rebuild from
	// it. That avoids keeping a separate base pointer alive.
	a.MOVLoad(RBX, At(RSP, -32))
	a.MOVQ(RDX, R10)
	a.MOVQ(R8, RDX)
	a.ADDQ(R8, RBX)
	a.MOVQ(R9, R8)
	a.ADDQ(R9, RBX)
	a.MOVQ(R10, R9)
	a.ADDQ(R10, RBX)

	a.DEC(RAX)
	a.JNZ(group)

	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// prefetchAhead emits a PREFETCHT0 for the weight stream d super-blocks past
// the cursor. d is EmitOpts.Prefetch; 0 disables, and is the default until a
// distance is measured on the host rather than tabulated.
func (a *Buf) prefetchAhead(cur Reg, blockBytes int32, d int) {
	if d > 0 {
		a.PREFETCHT0(At(cur, int32(d)*blockBytes))
	}
}

// emitKQuantMatVec generates the Q4_K/Q5_K decode matvec.
//
// block_q4_K is 144 bytes for 256 elements: f16 d, f16 dmin, 12 bytes of packed
// 6-bit scales and mins, then 128 bytes of nibbles, in eight groups of 32:
//
//	value = d*sc_g*nibble - dmin*m_g
//
// A group is exactly one 32-element activation block, so each group's dot is one
// VPDPBUSD and the min term is weight-independent:
//
//	sum_g [ d*sc_g*dx_g*dot_g - dmin*m_g*dx_g*sum(q_g) ]
//
// qs is consumed 32 bytes at a time, giving 32 low-nibble elements and then 32
// high-nibble elements from the same bytes: even groups read low nibbles and
// odd groups high nibbles of the preceding 32-byte run. The scales are unpacked
// once per super-block, spilled to the red zone and broadcast per group.
//
// accs is the number of independent accumulator chains: with one chain a
// super-block carries 16 serially dependent FMAs, which makes the kernel
// latency-bound. The registers come from KernelConst's 32-byte constants: the
// prologue-only 0x3F and 0x30 masks are memory operands, freeing Y12 and Y14.
// Y15 is free too on Q4_K; Q5_K needs it for qh, so it caps one chain lower.
func emitKQuantMatVec(t quant.Type, accs, pf int) []byte {
	q5 := t == quant.Q5_K
	bb := int32(t.BlockBytes())
	qsBase := int32(16) // Q4_K: d, dmin, scales[12], then qs
	if q5 {
		qsBase = 48 // Q5_K inserts qh[32] before qs
	}
	var a Buf

	const (
		stage = -32 // 16 B: gathered sc|m bytes
		scVec = -64 // 32 B: d * sc[0..7] * d_a[0..7]
		// The min term is consumed in the prologue and needs no slot.
	)

	a.MOVLoad(RCX, At(RDI, 0))         // Out
	a.MOVLoad(RDX, At(RDI, 8))         // W
	a.MOVLoad(R8, At(RDI, 16))         // A base
	a.MOVLoad(R9, At(RDI, 24))         // AScale base
	a.MOVLoad(RAX, At(RDI, 32))        // Rows
	a.MOVLoad(R10, At(RDI, 40))        // K, in 256-element super-blocks
	a.MOVLoad(RBX, At(RDI, 56))        // Scr
	a.VBROADCASTI128(Y10, At(RBX, 0))  // shufA
	a.VBROADCASTI128(Y11, At(RBX, 32)) // shufB
	a.VBROADCASTI128(Y13, At(RBX, 96)) // 0x0F, the only mask the group loop needs
	// 0x3F at 64 and 0x30 at 128 stay in memory: prologue-only, once per 144 B.

	// Accumulators, in the order chains are handed out. Y0 always; then the two
	// registers the memory-operand masks freed; then Y15, which Q5_K spends on
	// qh instead.
	accRegs := []Reg{Y0, Y12, Y14}
	if !q5 {
		accRegs = append(accRegs, Y15)
	}
	if accs < 1 {
		accs = 1
	}
	if accs > len(accRegs) {
		accs = len(accRegs)
	}
	accRegs = accRegs[:accs]

	row := a.Label()
	a.Bind(row)
	for _, r := range accRegs {
		a.VPXOR(r, r, r)
	}
	a.MOVQ(RSI, R8)  // activation cursor
	a.MOVQ(RDI, R9)  // scale-pair cursor
	a.MOVQ(R11, R10) // super-blocks remaining

	blk := a.Label()
	a.Bind(blk)
	a.prefetchAhead(RDX, bb, pf)

	// --- unpack the eight (sc, m) pairs, once per 144-byte super-block ---
	a.VBROADCASTI128(Y1, At(RDX, 4)) // the 12 packed scale bytes
	a.VPSHUFB(Y2, Y1, Y10)           // A
	a.VPSHUFB(Y3, Y1, Y11)           // B, the high-bit donors
	a.VPSRLW(Y4, Y3, 2)
	a.VPANDMem(Y4, Y4, At(RBX, 128)) // P = (B >> 2) & 0x30, i.e. (B >> 6) << 4
	a.VPANDMem(Y5, Y2, At(RBX, 64))  // X = A & 0x3F          -> sc[0..3], m[0..3]
	a.VPAND(Y6, Y2, Y13)
	a.VPOR(Y6, Y6, Y4) // Y = (A & 0x0F) | P    -> sc[4..7]
	a.VPSRLW(Y7, Y2, 4)
	a.VPAND(Y7, Y7, Y13)
	a.VPOR(Y7, Y7, Y4) // Z = ((A >> 4) & 0x0F) | P -> m[4..7]
	// dword0,2 from X; dword1 from Y; dword3 from Z.
	a.VPBLENDD(Y5, Y5, Y6, 0x02)
	a.VPBLENDD(Y5, Y5, Y7, 0x08)
	a.VMOVDQUStorex(At(RSP, stage), Y5)

	a.VPMOVZXBD(Y6, At(RSP, stage))   // sc[0..7] as int32
	a.VPMOVZXBD(Y7, At(RSP, stage+8)) // m[0..7]
	// x8, as an integer shift, to undo the /8 the activation sums carry
	// (QuantizeQ8 stores -sum(a)/8 for kernels whose horizontal add multiplies
	// it back). The min term here is summed lane-wise, so the eight is
	// reapplied to the 6-bit m[] before conversion: one exact instruction.
	a.VPSLLD(Y7, Y7, 3)
	a.VCVTDQ2PS(Y6, Y6)
	a.VCVTDQ2PS(Y7, Y7)
	a.VCVTPH2PSx(Y8, At(RDX, 0)) // d
	a.VBROADCASTSSReg(Y8, Y8)
	a.VMULPS(Y6, Y6, Y8)
	a.VCVTPH2PSx(Y9, At(RDX, 2)) // dmin
	a.VBROADCASTSSReg(Y9, Y9)
	a.VMULPS(Y7, Y7, Y9)

	// Fold the activation scale in here, once per super-block: the eight groups
	// line up with eight consecutive activation blocks, so their scales are
	// gathered and multiplied in one go instead of per group.
	//
	// The pairs array interleaves {scale, bias}, so the scales are the even
	// lanes of two loads. VSHUFPS 0x88 gathers them but cannot cross the 128-bit
	// halves, leaving [dx0 dx1 dx4 dx5 | dx2 dx3 dx6 dx7]; VPERMQ 0xD8 swaps the
	// middle two 64-bit lanes back into group order.
	a.VMOVDQULoad(Y1, At(RDI, 0))
	a.VMOVDQULoad(Y2, At(RDI, 32))
	// The ODD lanes are the sums, gathered exactly as the even lanes are: the
	// same pair of loads serves both, so this costs one shuffle and one permute.
	a.VSHUFPS(Y5, Y1, Y2, 0xDD)
	a.VPERMQ(Y5, Y5, 0xD8)
	a.VSHUFPS(Y1, Y1, Y2, 0x88)
	a.VPERMQ(Y1, Y1, 0xD8)
	a.VMULPS(Y6, Y6, Y1)
	a.VMULPS(Y7, Y7, Y1)

	// The whole min term, once per super-block, lane-wise:
	// -dmin*m[g]*d_a[g]*sum(a)[g] over the eight groups is a dot of two vectors
	// already in registers. The lanes land in accumulator 0 and the row's
	// horizontal reduce sums them, so it needs no extra register or epilogue.
	a.VMULPS(Y7, Y7, Y5)
	a.VADDPS(accRegs[0], accRegs[0], Y7)

	a.VMOVDQUStore(At(RSP, scVec), Y6)

	// Q5_K's fifth bits live in one 32-byte field shared by all eight groups:
	// qh does not advance. Group g reads bit g of every byte.
	if q5 {
		a.VMOVDQULoad(Y15, At(RDX, 16))
	}

	// --- eight groups of 32 ---
	for g := 0; g < 8; g++ {
		qsOff := qsBase + int32(32*(g/2))
		a.VMOVDQULoad(Y3, At(RDX, qsOff))
		if g%2 == 1 {
			a.VPSRLW(Y3, Y3, 4)
		}
		a.VPAND(Y3, Y3, Y13)
		if q5 {
			// ((qh >> g) & 1) << 4, folded into one shift: for g < 4 that is
			// (qh << (4-g)) & 0x10, otherwise (qh >> (g-4)) & 0x10.
			if g < 4 {
				a.VPSLLW(Y9, Y15, byte(4-g))
			} else {
				a.VPSRLW(Y9, Y15, byte(g-4))
			}
			a.VPANDMem(Y9, Y9, At(RBX, 160))
			a.VPOR(Y3, Y3, Y9)
		}
		a.VPXOR(Y4, Y4, Y4)
		a.VPDPBUSDMem(Y4, Y3, At(RSI, int32(32*g)))
		a.VCVTDQ2PS(Y4, Y4)

		// The scales already carry dx and the min correction is done, so each
		// group needs only its scale broadcast.
		a.VBROADCASTSS(Y7, At(RSP, scVec+int32(4*g)))
		acc := accRegs[g%len(accRegs)]
		a.VFMADD231PS(acc, Y4, Y7)
	}

	a.ADDimm(RDX, bb)
	a.ADDimm(RSI, 256)
	a.ADDimm(RDI, 64) // 8 activation blocks x 8 bytes
	a.DEC(R11)
	a.JNZ(blk)

	// Once per row, not per super-block: the chains are independent all the way
	// down the row and only meet here.
	for _, r := range accRegs[1:] {
		a.VADDPS(Y0, Y0, r)
	}
	a.VEXTRACTF128(Y1, Y0, 1)
	a.VADDPSx(Y0, Y0, Y1)
	a.VHADDPSx(Y0, Y0, Y0)
	a.VHADDPSx(Y0, Y0, Y0)
	a.VMOVSSStore(At(RCX, 0), Y0)
	a.ADDimm(RCX, 4)
	a.DEC(RAX)
	a.JNZ(row)

	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// Pack8 is the widest interleave EmitInterleaved supports, bounded by the
// register file: eight accumulators plus eight working registers is all sixteen.
const Pack8 = 8

// BestPack is the interleave width to use for a format: two. Wider wins on one
// core, but at the shipping thread count every interleaved row is another read
// stream competing for the prefetchers and the dTLB, and two wins end to end
// (see docs/engineering-history/cpu-kernels.md).
//
// PackWidth takes a pin for re-measuring. This is the host-default reporter
// (`jitllm hardware` pins through PackWidthNative); nn calls PackWidth with its
// own per-JIT pin.
func BestPack(t quant.Type) int { return PackWidth(0, t) }

// PackWidth is BestPack with the pin passed in rather than read from a package
// variable, which is what makes the width a property of ONE JIT.
func PackWidth(pin int, t quant.Type) int {
	if pin >= 1 && pin <= Pack8 {
		return pin
	}
	switch t {
	case quant.Q8_0, quant.Q4_0:
		return 2
	}
	return 1
}

// BestAccs is how many independent accumulator chains a kernel should carry.
//
// Only the k-quants have a choice: the single-chain Q4_K/Q5_K kernel is
// latency-bound, and two chains take most of the gain (TestABKQuantAccs;
// docs/engineering-history/cpu-kernels.md).
//
// AccChains takes a pin, for re-measuring rather than for tuning. See BestPack:
// this is the host-default reporter and nn calls AccChains with its own pin.
func BestAccs(t quant.Type) int8 { return AccChains(0, t) }

// AccChains is BestAccs with the pin passed in rather than read from a package
// variable.
func AccChains(pin int, t quant.Type) int8 {
	if pin >= 1 && pin <= 4 {
		return int8(pin)
	}
	switch t {
	case quant.Q4_K, quant.Q5_K:
		return 2
	}
	return 1
}

// EmitInterleaved generates a matvec that advances `rows` output rows together,
// with the row stride baked in as an immediate.
//
// Baking the stride is what makes eight rows fit: row r's block is
// [w + r*rowStride] off a single base, so another row costs one accumulator
// rather than one general-purpose pointer, and code size stays one block body.
//
// Registers. YMM: Y0-Y7 accumulators, Y8 the format constant, Y9-Y12 working,
// Y13 activation scale, Y14 activation bias, Y15 unpack temp. GP: RCX out, RDX
// weights, RSI activation cursor, RDI pair cursor, R8/R9 their bases, R11 block
// counter, RAX group counter.
func EmitInterleaved(t quant.Type, rows, nb int) ([]byte, error) {
	if t != quant.Q4_0 && t != quant.Q8_0 {
		return nil, fmt.Errorf("jit: EmitInterleaved: %s not supported", t)
	}
	if rows < 2 || rows > Pack8 {
		return nil, fmt.Errorf("jit: EmitInterleaved: rows=%d out of range", rows)
	}
	bb := int32(t.BlockBytes())
	stride := int32(nb) * bb
	if int64(rows-1)*int64(stride) > 1<<30 {
		return nil, fmt.Errorf("jit: EmitInterleaved: row stride %d too large to bake", stride)
	}
	var a Buf

	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 8))  // W
	a.MOVLoad(R8, At(RDI, 16))  // A base
	a.MOVLoad(R9, At(RDI, 24))  // AScale base
	a.MOVLoad(RAX, At(RDI, 32)) // Rows, in groups
	a.MOVLoad(R10, At(RDI, 40)) // K, in blocks
	a.MOVLoad(RBX, At(RDI, 56)) // Scr
	a.VBROADCASTI128(Y8, At(RBX, 0))

	accs := []Reg{Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7}[:rows]

	group := a.Label()
	a.Bind(group)
	for _, acc := range accs {
		a.VPXOR(acc, acc, acc)
	}
	a.MOVQ(RSI, R8)
	a.MOVQ(RDI, R9)
	a.MOVQ(R11, R10)

	blk := a.Label()
	a.Bind(blk)
	{
		const bOff = int32(0)
		a.VBROADCASTSS(Y13, At(RDI, bOff*8))
		a.VBROADCASTSS(Y14, At(RDI, bOff*8+4))
		// Hoist the activation load: every row multiplies the same 32 bytes, and
		// a VPDPBUSD memory operand is a real load each time, so this cuts the
		// loads per block against the load ports. The unpack borrows Y10, which
		// is zeroed immediately after.
		a.VMOVDQULoad(Y15, At(RSI, bOff*Q8Block))
		for r := 0; r < rows; r++ {
			off := int32(r)*stride + bOff*bb
			if t == quant.Q4_0 {
				a.VBROADCASTI128(Y9, At(RDX, off+2))
				a.VPSRLW(Y10, Y9, 4)
				a.VPBLENDD(Y9, Y9, Y10, 0xF0)
				a.VPAND(Y9, Y9, Y8)
			} else {
				a.VMOVDQULoad(Y9, At(RDX, off+2))
				a.VPXOR(Y9, Y9, Y8)
			}
			a.VPXOR(Y10, Y10, Y10)
			a.VPDPBUSD(Y10, Y9, Y15)
			a.VCVTDQ2PS(Y11, Y10)
			a.VCVTPH2PSx(Y12, At(RDX, off))
			a.VBROADCASTSSReg(Y12, Y12)
			a.VMULPS(Y12, Y12, Y13)
			// One accumulator update, not two: dot*scale + negsum*scale is
			// (dot + negsum)*scale, which halves the dependency chain into acc.
			a.VADDPS(Y11, Y11, Y14)
			a.VFMADD231PS(accs[r], Y11, Y12)
		}
	}
	a.ADDimm(RDX, bb)
	a.ADDimm(RSI, Q8Block)
	a.ADDimm(RDI, 8)
	a.DEC(R11)
	a.JNZ(blk)

	for r := 0; r < rows; r++ {
		a.VEXTRACTF128(Y13, accs[r], 1)
		a.VADDPSx(accs[r], accs[r], Y13)
		a.VHADDPSx(accs[r], accs[r], accs[r])
		a.VHADDPSx(accs[r], accs[r], accs[r])
		a.VMOVSSStore(At(RCX, int32(r*4)), accs[r])
	}
	a.ADDimm(RCX, int32(rows*4))
	// RDX advanced by one row during the block loop; skip the remaining rows-1.
	a.ADDimm(RDX, int32(rows-1)*stride)
	a.DEC(RAX)
	a.JNZ(group)

	a.VZEROUPPER()
	a.RET()
	return a.Bytes(), nil
}

// emitQ6KMatVec generates the Q6_K decode matvec.
//
// block_q6_K is 210 bytes for 256 elements: ql[128], qh[64], sixteen int8
// scales, then f16 d. Value is d*sc_g*(q-32), where q is six bits assembled
// from a nibble in ql and two bits in qh.
//
// Sub-blocks are sixteen elements, not 32, which is the structural difference
// from Q4_K and the reason this kernel exists in its own function. An activation
// block is 32, so each one spans TWO weight groups with different scales. That
// falls out for free: VPDPBUSD over 32 bytes leaves eight int32 lanes where
// lanes 0-3 sum elements 0-15 and lanes 4-7 sum elements 16-31 — exactly the two
// groups. A per-lane scale vector (four lanes of sc_A, four of sc_B) then scales
// both in one multiply.
//
// The bias correction needs per-16 activation sums for the same reason, which is
// what Args.AHalf carries.
//
// Element order is not sequential. For each 128-element half, with ql[0:64] and
// qh[0:32]: block 0 is ql[0:32] low nibbles with qh bits 0-1, block 1 is
// ql[32:64] low with bits 2-3, block 2 is ql[0:32] HIGH nibbles with bits 4-5,
// block 3 is ql[32:64] high with bits 6-7.
func emitQ6KMatVec() []byte {
	var a Buf
	const scVec = -64 // 64 B of red zone: sixteen float32 scales

	a.MOVLoad(RCX, At(RDI, 0))         // Out
	a.MOVLoad(RDX, At(RDI, 8))         // W
	a.MOVLoad(R8, At(RDI, 16))         // A base
	a.MOVLoad(R9, At(RDI, 24))         // AScale base
	a.MOVLoad(RAX, At(RDI, 32))        // Rows
	a.MOVLoad(R10, At(RDI, 40))        // K, in super-blocks
	a.MOVLoad(R12, At(RDI, 64))        // AHalf base
	a.MOVLoad(RBX, At(RDI, 56))        // Scr
	a.VBROADCASTI128(Y13, At(RBX, 0))  // 0x0F
	a.VBROADCASTI128(Y14, At(RBX, 16)) // 0x03

	row := a.Label()
	a.Bind(row)
	a.VPXOR(Y0, Y0, Y0)
	a.MOVQ(RSI, R8)
	a.MOVQ(RDI, R9)
	a.MOVQ(R13, R12)
	a.MOVQ(R11, R10)

	blk := a.Label()
	a.Bind(blk)
	// Sixteen int8 scales -> sixteen float32, scaled by d, spilled once.
	a.VPMOVSXBD(Y6, At(RDX, 192))
	a.VPMOVSXBD(Y7, At(RDX, 200))
	a.VCVTDQ2PS(Y6, Y6)
	a.VCVTDQ2PS(Y7, Y7)
	a.VCVTPH2PSx(Y8, At(RDX, 208)) // d
	a.VBROADCASTSSReg(Y8, Y8)
	a.VMULPS(Y6, Y6, Y8)
	a.VMULPS(Y7, Y7, Y8)
	a.VMOVDQUStore(At(RSP, scVec), Y6)
	a.VMOVDQUStore(At(RSP, scVec+32), Y7)

	for b := 0; b < 8; b++ {
		half, within := b/4, b%4
		qlOff := int32(half*64 + (within%2)*32)
		qhOff := int32(128 + half*32)

		a.VMOVDQULoad(Y3, At(RDX, qlOff))
		if within >= 2 {
			a.VPSRLW(Y3, Y3, 4)
		}
		a.VPAND(Y3, Y3, Y13)
		a.VMOVDQULoad(Y9, At(RDX, qhOff))
		if sh := byte(within * 2); sh != 0 {
			a.VPSRLW(Y9, Y9, sh)
		}
		a.VPAND(Y9, Y9, Y14)
		a.VPSLLW(Y9, Y9, 4)
		a.VPOR(Y3, Y3, Y9) // six-bit value, 0..63, unsigned

		a.VPXOR(Y4, Y4, Y4)
		a.VPDPBUSDMem(Y4, Y3, At(RSI, int32(32*b)))
		a.VCVTDQ2PS(Y4, Y4)

		// Per-lane scales: lanes 0-3 take sc[2b], lanes 4-7 take sc[2b+1].
		a.VBROADCASTSS(Y6, At(RSP, scVec+int32(8*b)))
		a.VBROADCASTSS(Y7, At(RSP, scVec+int32(8*b)+4))
		a.VPBLENDD(Y6, Y6, Y7, 0xF0)
		a.VBROADCASTSS(Y5, At(RDI, int32(8*b))) // activation scale
		a.VMULPS(Y6, Y6, Y5)

		// Same lane split for the -32*sum(q) correction, from AHalf.
		a.VBROADCASTSS(Y7, At(R13, int32(8*b)))
		a.VBROADCASTSS(Y8, At(R13, int32(8*b)+4))
		a.VPBLENDD(Y7, Y7, Y8, 0xF0)
		a.VADDPS(Y4, Y4, Y7)
		a.VFMADD231PS(Y0, Y4, Y6)
	}

	a.ADDimm(RDX, 210)
	a.ADDimm(RSI, 256)
	a.ADDimm(RDI, 64)
	a.ADDimm(R13, 64) // sixteen halves x 4 bytes
	a.DEC(R11)
	a.JNZ(blk)

	a.VEXTRACTF128(Y1, Y0, 1)
	a.VADDPSx(Y0, Y0, Y1)
	a.VHADDPSx(Y0, Y0, Y0)
	a.VHADDPSx(Y0, Y0, Y0)
	a.VMOVSSStore(At(RCX, 0), Y0)
	a.ADDimm(RCX, 4)
	a.DEC(RAX)
	a.JNZ(row)

	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// emitQ3KMatVec generates the Q3_K decode matvec.
//
// block_q3_K is 110 bytes for 256 elements: hmask[32], qs[64], twelve bytes of
// packed 6-bit scales, f16 d. Sub-blocks are sixteen elements with their own
// scale, like Q6_K, so the same per-lane trick applies.
//
// Value is d * (sc_g - 32) * (v + 4*hbit - 4), where v is two bits from qs and
// hbit is one bit from hmask. A set high-mask bit means do not subtract 4;
// inverting it yields a model that runs and emits fluent nonsense.
//
// For activation block b (0..7), covering weight groups 2b and 2b+1:
//
//	qs source  = qs[(b/4)*32 : +32]
//	shift      = 2*(b%4)
//	hmask bit  = b            -- hmask never advances; its 32 bytes carry
//	                             eight bits each, one per element of the 256
//
// Biasing to VPDPBUSD's unsigned operand: u = v + 4*hbit is 0..7, and the -4
// falls out as a -4*sum(q) correction per group, carried in Args.AHalf.
func emitQ3KMatVec(pf int) []byte {
	var a Buf
	const (
		stage  = -32
		scVec  = -64
		scVec2 = -96
		cShufA = 0
		cShufB = 32
		c0F    = 64
		c30    = 96
		c03    = 128
		c01    = 160
		c20    = 192
		c04    = 224
	)

	a.MOVLoad(RCX, At(RDI, 0))
	a.MOVLoad(RDX, At(RDI, 8))
	a.MOVLoad(R8, At(RDI, 16))
	a.MOVLoad(R9, At(RDI, 24))
	a.MOVLoad(RAX, At(RDI, 32))
	a.MOVLoad(R10, At(RDI, 40))
	a.MOVLoad(R12, At(RDI, 64)) // AHalf
	a.MOVLoad(RBX, At(RDI, 56)) // Scr, stays live for memory-operand masks
	a.VBROADCASTI128(Y10, At(RBX, cShufA))
	a.VBROADCASTI128(Y11, At(RBX, cShufB))
	a.VBROADCASTI128(Y12, At(RBX, c0F))
	a.VBROADCASTI128(Y13, At(RBX, c03))
	a.VBROADCASTI128(Y14, At(RBX, c20))

	row := a.Label()
	a.Bind(row)
	a.VPXOR(Y0, Y0, Y0)
	a.MOVQ(RSI, R8)
	a.MOVQ(RDI, R9)
	a.MOVQ(R13, R12)
	a.MOVQ(R11, R10)

	blk := a.Label()
	a.Bind(blk)
	a.prefetchAhead(RDX, 110, pf)

	// --- unpack the sixteen 6-bit scales ---
	a.VBROADCASTI128(Y1, At(RDX, 96))
	a.VPSHUFB(Y2, Y1, Y10) // [b0-3 | b4-7 | b0-3 | b4-7]
	a.VPSHUFB(Y3, Y1, Y11) // [b8-11 x4], the high-bit donors
	// low four bits: dwords 0,1 straight, dwords 2,3 shifted down
	a.VPAND(Y4, Y2, Y12)
	a.VPSRLW(Y5, Y2, 4)
	a.VPAND(Y5, Y5, Y12)
	a.VPBLENDD(Y4, Y4, Y5, 0b11001100)
	// high two bits: each dword wants a different shift of the same donors, and
	// ((x >> s) & 3) << 4 collapses to a single shift masked with 0x30.
	a.VPSLLW(Y5, Y3, 4) // s=0
	a.VPSLLW(Y6, Y3, 2) // s=2
	a.VPSRLW(Y7, Y3, 2) // s=6
	a.VPBLENDD(Y5, Y5, Y6, 0b00100010)
	a.VPBLENDD(Y5, Y5, Y3, 0b01000100) // s=4 needs no shift
	a.VPBLENDD(Y5, Y5, Y7, 0b10001000)
	a.VPANDMem(Y5, Y5, At(RBX, c30))
	a.VPOR(Y4, Y4, Y5) // sc, 0..63 unsigned
	// Bias by 32 in integer space, as bytes, so VPMOVSXBD can widen them
	// signed. Doing it in float would need a constant and a subtract.
	a.VPSUBB(Y4, Y4, Y14)
	a.VMOVDQUStorex(At(RSP, stage), Y4)
	a.VPMOVSXBD(Y6, At(RSP, stage))
	a.VPMOVSXBD(Y7, At(RSP, stage+8))
	a.VCVTDQ2PS(Y6, Y6)
	a.VCVTDQ2PS(Y7, Y7)
	a.VCVTPH2PSx(Y8, At(RDX, 108)) // d
	a.VBROADCASTSSReg(Y8, Y8)
	a.VMULPS(Y6, Y6, Y8)
	a.VMULPS(Y7, Y7, Y8)

	// d_a and the whole -4 correction are folded in here, once per super-block,
	// rather than per group. One d_a covers weight sub-blocks 2b and 2b+1, so
	// sc'[j] = d * sc[j] * d_a[j/2]: the d_a values duplicated pairwise, which
	// VPERMQ then VSHUFPS builds in two instructions per half.
	a.VMOVDQULoad(Y1, At(RDI, 0))
	a.VMOVDQULoad(Y2, At(RDI, 32))
	a.VSHUFPS(Y1, Y1, Y2, 0x88) // the even lanes are the scales
	a.VPERMQ(Y1, Y1, 0xD8)      // undo VSHUFPS's 128-bit split: d_a[0..7]
	a.VPERMQ(Y2, Y1, 0x50)      // qwords [0,0,1,1] -> [a0 a1 a0 a1 a2 a3 a2 a3]
	a.VSHUFPS(Y2, Y2, Y2, 0x50) // within each half: [a0 a0 a1 a1 | a2 a2 a3 a3]
	a.VPERMQ(Y5, Y1, 0xFA)      // qwords [2,2,3,3]
	a.VSHUFPS(Y5, Y5, Y5, 0x50)
	a.VMULPS(Y6, Y6, Y2)
	a.VMULPS(Y7, Y7, Y5)
	a.VMOVDQUStore(At(RSP, scVec), Y6)
	a.VMOVDQUStore(At(RSP, scVec2), Y7)

	// The -4 correction is a 16-element dot product: -4*sum(a) over each
	// sixteen elements is what AHalf holds, already in sub-block order. Summed
	// over the row's horizontal reduce the contribution is
	// 4 * SUM_j corr[j] * sc'[j] (each value occupies four lanes); the x4 is two
	// doublings, which needs no constant.
	a.VMOVDQULoad(Y2, At(R13, 0))
	a.VMOVDQULoad(Y5, At(R13, 32))
	a.VMULPS(Y2, Y2, Y6)
	a.VMULPS(Y5, Y5, Y7)
	a.VADDPS(Y2, Y2, Y5)
	a.VADDPS(Y2, Y2, Y2)
	a.VADDPS(Y2, Y2, Y2)
	a.VADDPS(Y0, Y0, Y2)

	// hmask is loop-invariant across all eight blocks: 32 bytes, eight bits each.
	a.VMOVDQULoad(Y15, At(RDX, 0))

	for b := 0; b < 8; b++ {
		a.VMOVDQULoad(Y3, At(RDX, int32(32+(b/4)*32)))
		if sh := byte(2 * (b % 4)); sh != 0 {
			a.VPSRLW(Y3, Y3, sh)
		}
		a.VPAND(Y3, Y3, Y13) // v = 0..3
		// hbit is bit b of every hmask byte and must end at bit 2 (the value is
		// v + 4*hbit), so one shift of |b-2| and a 0x04 mask put it there; b == 2
		// needs no shift because VPANDMem takes a separate source register.
		//
		// The shifts are word shifts on a byte mask, which is safe because every
		// bit that crosses a byte boundary lands outside bit 2 and the mask
		// removes it. Left by 1 or 2
		// puts byte 1's bit 2 at word bit 10, sourced from word bit 10-s, which
		// is still inside byte 1; right by 1..5 sources byte 0's bit 2 from bit
		// 2+s = b, inside byte 0.
		switch {
		case b < 2:
			a.VPSLLW(Y9, Y15, byte(2-b))
			a.VPANDMem(Y9, Y9, At(RBX, c04))
		case b > 2:
			a.VPSRLW(Y9, Y15, byte(b-2))
			a.VPANDMem(Y9, Y9, At(RBX, c04))
		default:
			a.VPANDMem(Y9, Y15, At(RBX, c04))
		}
		a.VPOR(Y3, Y3, Y9) // u = v + 4*hbit, 0..7

		a.VPXOR(Y4, Y4, Y4)
		a.VPDPBUSDMem(Y4, Y3, At(RSI, int32(32*b)))
		a.VCVTDQ2PS(Y4, Y4)

		// lanes 0-3 take sc[2b], lanes 4-7 take sc[2b+1]
		off := int32(scVec + 4*(2*b))
		if b >= 4 {
			off = int32(scVec2 + 4*(2*(b-4)))
		}
		// The scales already carry d and d_a, and the -4 correction was consumed
		// in the prologue; only the blend of the two half scales is left.
		a.VBROADCASTSS(Y6, At(RSP, off))
		a.VBROADCASTSS(Y7, At(RSP, off+4))
		a.VPBLENDD(Y6, Y6, Y7, 0xF0)
		a.VFMADD231PS(Y0, Y4, Y6)
	}

	a.ADDimm(RDX, 110)
	a.ADDimm(RSI, 256)
	a.ADDimm(RDI, 64)
	a.ADDimm(R13, 64)
	a.DEC(R11)
	a.JNZ(blk)

	a.VEXTRACTF128(Y1, Y0, 1)
	a.VADDPSx(Y0, Y0, Y1)
	a.VHADDPSx(Y0, Y0, Y0)
	a.VHADDPSx(Y0, Y0, Y0)
	a.VMOVSSStore(At(RCX, 0), Y0)
	a.ADDimm(RCX, 4)
	a.DEC(RAX)
	a.JNZ(row)

	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// emitF16MatVec is emitFloatMatVec for f16 weights: VCVTPH2PS widens them
// straight from memory. Accumulation stays f32, because a router's output feeds
// a softmax and a top-k where one ulp can select a different expert
// (TestF16MatVecMatchesReference). F16C ships with every AVX2 part this tier
// requires, so it needs no separate probe.
func emitF16MatVec() []byte { return emitFloatMatVec(quant.F16) }

func emitF32MatVec() []byte { return emitFloatMatVec(quant.F32) }

// emitBF16MatVec is the same kernel for bfloat16 weights: a bfloat16 is the top
// half of a float32, so widening is a zero-extend and a sixteen-bit shift.
func emitBF16MatVec() []byte { return emitFloatMatVec(quant.BF16) }

// emitFloatMatVec generates out[r] = dot(row r, x) for row-major F32, F16 or
// BF16 weights and f32 activations, at any k.
//
// Args.A carries float32 here, not int8: the caller passes a *float32 through
// unsafe.Pointer, which stays a real Go pointer for the GC (see Args). Args.K
// is the element count (BlocksPerRow for a block size of one), and rows are
// contiguous.
//
// The body is 32 elements in four independent FMA chains (one chain would be a
// latency-bound dependency); the k%32 remainder goes through whole vectors of
// eight into chain 0 and then single elements. The chains are summed pairwise,
// so the result is deterministic but not sequential-sum order.
func emitFloatMatVec(t quant.Type) []byte {
	var a Buf
	esz := int32(4)
	if t != quant.F32 {
		esz = 2
	}
	// wide loads eight weights as f32; one loads a single weight into lane 0
	// with the rest zero.
	wide := func(dst Reg, off int32) {
		switch t {
		case quant.F32:
			a.VMOVDQULoad(dst, At(RDX, off))
		case quant.F16:
			a.VCVTPH2PS(dst, At(RDX, off))
		default:
			a.VPMOVZXWD(dst, At(RDX, off))
			a.VPSLLD(dst, dst, 16)
		}
	}
	one := func(dst Reg) {
		switch t {
		case quant.F32:
			a.VMOVSSLoad(dst, At(RDX, 0))
		case quant.F16:
			a.VPXOR(dst, dst, dst)
			a.VPINSRWLoad(dst, dst, At(RDX, 0), 0)
			a.VCVTPH2PSReg(dst, dst)
		default:
			a.VPXOR(dst, dst, dst)
			a.VPINSRWLoad(dst, dst, At(RDX, 0), 0)
			a.VPSLLD(dst, dst, 16)
		}
	}

	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 8))  // W, walked continuously across rows
	a.MOVLoad(RSI, At(RDI, 16)) // A, reinterpreted as float32
	a.MOVLoad(RAX, At(RDI, 32)) // Rows
	a.MOVLoad(R9, At(RDI, 40))  // K, in elements
	a.MOVQ(R12, R9)
	a.SHRimm(R12, 3)
	a.ANDimm8(R12, 3) // whole vectors of eight past the groups of 32
	a.MOVQ(R13, R9)
	a.ANDimm8(R13, 7) // single elements past those
	a.SHRimm(R9, 5)   // groups of 32

	row := a.Label()
	a.Bind(row)
	a.VPXOR(Y0, Y0, Y0)
	a.VPXOR(Y1, Y1, Y1)
	a.VPXOR(Y2, Y2, Y2)
	a.VPXOR(Y3, Y3, Y3)
	a.MOVQ(R11, RSI) // activation cursor, rewound for every row

	loop := func(n Reg, body func(), wstep, xstep int32) {
		skip := a.Label()
		a.MOVQ(R10, n)
		a.TESTQ(R10, R10)
		a.JZ(skip)
		lp := a.Label()
		a.Bind(lp)
		body()
		a.ADDimm(RDX, wstep)
		a.ADDimm(R11, xstep)
		a.DEC(R10)
		a.JNZ(lp)
		a.Bind(skip)
	}
	loop(R9, func() {
		// 32 elements: four loads, one accumulator each so the four FMA
		// chains are independent.
		for i := int32(0); i < 4; i++ {
			wide(Reg(4+i), 8*esz*i)
			a.VFMADD231PSMem(Reg(i), Reg(4+i), At(R11, 32*i))
		}
	}, 32*esz, 128)
	loop(R12, func() {
		wide(Y4, 0)
		a.VFMADD231PSMem(Y0, Y4, At(R11, 0))
	}, 8*esz, 32)
	loop(R13, func() {
		one(Y4)
		a.VMOVSSLoad(Y5, At(R11, 0))
		a.VFMADD231PS(Y0, Y4, Y5)
	}, esz, 4)

	// Pairwise, then the usual lane reduction.
	a.VADDPS(Y0, Y0, Y1)
	a.VADDPS(Y2, Y2, Y3)
	a.VADDPS(Y0, Y0, Y2)
	a.VEXTRACTF128(Y1, Y0, 1)
	a.VADDPSx(Y0, Y0, Y1)
	a.VHADDPSx(Y0, Y0, Y0)
	a.VHADDPSx(Y0, Y0, Y0)
	a.VMOVSSStore(At(RCX, 0), Y0)

	a.ADDimm(RCX, 4)
	a.DEC(RAX)
	a.JNZ(row)

	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}
