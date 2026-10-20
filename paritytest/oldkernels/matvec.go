// Package kernels holds the NN kernels written against the IR.
package oldkernels

import (
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/jit/gpu/kernels"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// decodeE2M1 maps four MXFP4 codes, one nibble in each byte lane of x, to
// their mxfp4Codes entries. The IR has no byte shuffle, so the table is
// computed rather than looked up: with q = s|e1|e0|m (sign in bit 3), the
// doubled magnitudes 0,1,2,3,4,6,8,12 are
//
//	v = l + (l if b2) + 4*b2 + 2*(b0 b1 b2),   l = q&3
//
// and the biased code is (v+12) - (2v if s). Every byte stays in 0..24 at each
// step, so nothing carries into its neighbour.
func decodeE2M1(b *ir.Builder, x ir.Value) ir.Value {
	c := func(v uint32) ir.Value { return b.Const(ir.U32, int64(v)) }
	ones, ff := c(0x01010101), c(0xFF)
	l := b.And(ir.U32, x, c(0x03030303))
	b2 := b.And(ir.U32, b.Shr(ir.U32, x, c(2)), ones)
	v := b.Add(ir.U32, l, b.And(ir.U32, l, b.Mul(ir.U32, b2, ff)))
	v = b.Add(ir.U32, v, b.Shl(ir.U32, b2, c(2)))
	t := b.And(ir.U32, b.And(ir.U32, x, b.Shr(ir.U32, x, c(1))), b2)
	v = b.Add(ir.U32, v, b.Shl(ir.U32, t, c(1)))
	neg := b.Mul(ir.U32, b.And(ir.U32, b.Shr(ir.U32, x, c(3)), ones), ff)
	return b.Sub(ir.U32, b.Add(ir.U32, v, c(rep4(kernels.MXFP4Bias))),
		b.And(ir.U32, b.Shl(ir.U32, v, c(1)), neg))
}

// MatVecShape is a decode matvec: out[row] = dot(quantized row, activations).
//
// K and Rows are baked; shapes come from the container, so nothing is
// speculative.
type MatVecShape struct {
	T    Quant
	K    int // elements per row, a multiple of 32
	Rows int // output rows
	// PickOf, when set, makes output row r the weight's row pPick[r] of a
	// PickOf-row matrix: the label pick's projection (nn.Head.Pick), Rows
	// picked rows of a vocabulary head in place of all of them. pPick follows
	// pOut and pBias. Dense one-token decode shapes only.
	PickOf int
	// Bias adds a per-row vector to the result (qwen2-style projections),
	// folded into the store to save a launch per projection. With Split > 1 it
	// is gated to segment 0 with ir.Select, a data-flow select rather than a
	// branch, so the bias lands once.
	Bias bool
	// BiasRows says pBias holds one vector per token, token t's at t*Rows,
	// where Bias alone reads pBias[row] for every token. It is the residual a
	// decode step of several sequences adds in the bias slot (each sequence
	// has its own stream), as decode's one row adds its own. Only with every
	// token in one thread (Tok == NTok) and no experts.
	BiasRows bool
	// Gate multiplies the finished row by GateAct(pGate[row]): the up
	// projection writing act(gate)*up itself, so the gated FFN needs no
	// elementwise launch between it and the down projection's quantize. Only
	// where the matvec writes the FINAL row -- GroupSplit, or Split 1 -- and on a
	// plain decode shape, or a dense one carrying every token in one thread
	// (Tok == NTok), whose token t reads pGate at t*Rows; pGate follows pOut.
	Gate    bool
	GateAct ActKind
	// Gate2 is Gate with the gate's row computed in this launch rather than
	// read: a second weight stream (pQS2, pD2, pSC2, the gate's weight in up's
	// format and shape) summed beside up's over the one operand, reduced in
	// the same order as the gate's own launch, so the row written is the bits
	// act(gate)*up the separate launches write. One launch where there were
	// two, and each workgroup reads its rows of both matrices. Decode only:
	// one token, Rowt 1, no experts, a quantized format, a final-row split.
	Gate2 bool
	// NTok is how many token columns this launch computes and Tok how many one
	// thread carries. Both default to the decode form, which is then unchanged.
	//
	// This is the device prefill: without it a prompt read every weight once
	// per token. The reuse happens in registers: one thread takes a row and Tok
	// token columns and loads each weight word once for Tok dot products. NTok
	// reuses the slot machinery (SlotAct) for addressing.
	NTok int
	Tok  int
	// ActRows is how many vectors the quantized activation holds when the
	// launch reads only the first NTok of them -- a decode step of NTok live
	// sequences in a scratch padded wider, or (NTok 0 or 1) the decode matvec
	// reading the first of a step's rows, the one that wants the head. One
	// Quantize over all of them wrote every scale before any sum, so the sums
	// start ActRows scale blocks in; 0 means NTok.
	ActRows int
	// Rowt is how many rows one thread carries. With Tok alone the load:dot4
	// ratio stays 1:1 (Tok activation words per weight word) and the kernel is
	// load-unit bound; Rowt rows share one set of activation loads, making it
	// 1:Rowt, at the cost of Rowt*Tok accumulators. A thread's rows are
	// adjacent (see tileOff).
	Rowt int
	// GroupSplit puts the Split segments of a row in one threadgroup and
	// reduces them through threadgroup memory, so the kernel writes the final
	// output with no Reduce launch or partial buffer.
	//
	// It is for the small matvecs: they want a wider split, but the global
	// partial round trip grows with Split and cancels the gain. Adjacent lanes
	// still hold adjacent rows within a segment, so coalescing is unchanged.
	GroupSplit bool
	// MT and NT are the MMA path's tile: one warp owns MT*16 weight rows by
	// NT*8 token columns. Ignored by the dp4a kernel.
	MT, NT int
	// Center is the weight centering policy; see Center.
	Center Center
	// ActWin is how many activation elements share one quantization scale.
	// Carried for the MMA path's integer fold, which is not built (see
	// MatVecMMA); nothing reads it.
	//
	// Experts is how many experts the weight buffer holds; 0 or 1 is an
	// ordinary dense matvec and the kernel is unchanged. An indexed matvec
	// (llama.cpp's MUL_MAT_ID) needs no new IR op: the expert id loaded from
	// pSel is uniform, so it is address arithmetic, not divergent control flow.
	ActWin  int
	Experts int
	// SlotAct says each slot reads its own activation vector, laid out
	// contiguously at stride K, which an MoE down-projection needs because slot
	// j consumes expert e_j's FFN activations. It lets the down-projection be
	// one launch; gate and up leave it false and share one activation.
	SlotAct bool
	// Slots is how many selected experts one launch covers, so the kernel runs
	// Rows*Slots*Split threads and pSel holds Slots expert ids.
	Slots int
	// Split is how many threads share one row, each summing 1/Split of the
	// blocks. 0 or 1 is one thread per row; anything more writes Split partial
	// sums per row that Reduce (or the in-group reduction) adds.
	//
	// It exists because one thread per row cannot fill the card on the small
	// shapes. It changes the float summation order, so outputs differ from the
	// CPU in low bits and greedy tokens can diverge on deep models; it is gated
	// on NMSE, and tier.WithSplit(1) gives bit-stable output.
	Split int
	// Sub overrides the format's activation sub-block (the granule the k loop
	// walks) and is legal only for a float format; 0 keeps the format's own.
	//
	// It exists because K is not always a multiple of 32: MLA's absorbed
	// projections have k equal to the latent rank or the nope half. A quantized
	// format cannot go below its sub-block (scales and sums are laid out
	// against it); a float weight has no scale plane, so a smaller sub-block is
	// the same arithmetic in smaller steps. It must divide K, be a multiple of
	// 4 and be at most the format's own sub-block.
	Sub int
	// In builds the int8 operand in the kernel from the float row (ActIn):
	// parameters pIn [pNW] in place of pA and pAX.
	In ActIn
}

// FloatSub is the sub-block a float matvec of width k walks when k is not a
// whole number of 32-element blocks: k itself below 32, else the widest of
// 16, 8 and 4 that divides it; 0 for a quantized format, a whole k, or a k no
// multiple of 4 divides. MatVecShape.Sub takes it when the caller names none.
func FloatSub(t Quant, k int) int {
	if infoOf(t).float == 0 || k <= 0 || k%4 != 0 || k%32 == 0 {
		return 0
	}
	if k < 32 {
		return k
	}
	for _, sub := range []int{16, 8, 4} {
		if k%sub == 0 {
			return sub
		}
	}
	return 0
}

// Gate2Exact reports whether a Gate2 matvec writes the bits of the two
// launches it replaces on a backend: true where the lowering fixes every
// rounding (PTX's .rn arithmetic). SPIR-V lowers an Fma as an unfused pair
// with no NoContraction decoration and MSL compiles with fast math, so a
// driver may contract the one kernel's sums differently from the other's.
func Gate2Exact(api string) bool { return api == "ptx" }

// MatVec builds the decode matvec.
//
// One thread per row works because of the layout: weights are transposed at
// upload so element u of every row is contiguous,
//
//	qs[u*nrows + row]   instead of   qs[row*nu + u]
//
// and adjacent threads reading the same u coalesce. It is a permutation, so it
// costs no bytes, and it removes the need for a cross-thread reduction.
//
// Rows is baked because it is the layout's stride, needed on every load.
//
// The IR has no conditional execution outside a loop, so a thread past the
// last row is clamped to it with Min: it recomputes that row and writes the
// same value to the same address, which is benign.
func MatVec(s MatVecShape) (*ir.Kernel, error) { return matVec(s, nil, nil) }

// Epilogue is an elementwise chain a matvec applies to each finished output
// before its store, composed from the ops that read the output (a bias, a
// residual, a gate, a clamp...) rather than taken from MatVecShape's menu.
// Params declares its buffers after the matvec's own; Apply maps the output
// v, at element index at of the output (and row, its row of Rows), to the
// value stored, emitting any further stores of its own.
type Epilogue interface {
	Params(b *ir.Builder)
	Apply(b *ir.Builder, v, at, row ir.Value) ir.Value
}

// MatVecEpi is MatVec whose every output passes through e. It is for a dense
// matvec that reduces in one launch -- no split partial buffer, one expert,
// no menu epilogue of its own -- which is every shape the tier's decode
// matvec takes when its split reduces in the group.
func MatVecEpi(s MatVecShape, e Epilogue) (*ir.Kernel, error) {
	if e == nil {
		return MatVec(s)
	}
	if s.Split > 1 && !s.GroupSplit || s.Experts > 1 || s.Slots > 1 || s.Bias || s.Gate || s.Gate2 || s.PickOf > 0 {
		return nil, fmt.Errorf("kernels: MatVecEpi: a shape that reduces across launches or carries its own epilogue")
	}
	return matVec(s, nil, e)
}

// segCtx is where matVec emits when it is one SEGMENT of a MatVecSegments
// kernel instead of a kernel of its own: the shared builder, a suffix that keeps
// its parameters apart from the other segments', the activation every segment
// reads, the block index relative to the segment's first block, and the
// threadgroup buffer declared once before any segment's loop.
type segCtx struct {
	b       *ir.Builder
	suffix  string
	pA, pAX ir.Value
	cta     func() ir.Value
	shPart  ir.Value
	hasSh   bool
}

func matVec(s MatVecShape, into *segCtx, epi Epilogue) (*ir.Kernel, error) {
	// NTok reuses the slot machinery: n token columns are exactly SlotAct's
	// "each slot reads its own activation vector at stride K", and one thread
	// carries Tok of them.
	//
	// Rowt applies to decode too. The body is general in Rowt; what it buys in
	// decode is contiguity: in the transposed layout the Rowt rows of an
	// adjacent tile are Rowt consecutive u32, so a lane reads more bytes per
	// load. Split then puts the threads back. See
	// docs/engineering-history/gpu-kernels.md for the measurements.
	if s.Rowt > 1 {
		if s.Experts > 1 {
			return nil, fmt.Errorf("kernels: MatVec: Rowt=%d needs no experts", s.Rowt)
		}
		if s.Rows%s.Rowt != 0 {
			return nil, fmt.Errorf("kernels: MatVec: Rowt=%d does not divide Rows=%d", s.Rowt, s.Rows)
		}
	}
	if s.Rowt < 1 {
		s.Rowt = 1
	}
	if s.GroupSplit {
		// One row per thread: the group's threads are spent on segments. An
		// indexed (expert) matvec takes it too; expert matvecs are small and
		// want a wide split. Workgroups are laid out slot-major: Rows/rpg
		// groups per slot. Not with a bias: an expert bias here has no caller.
		//
		// A few tokens take it when one thread carries all of them (Tok ==
		// NTok): a decode step over several sequences, which is the decode
		// matvec reading each weight word once for every token, and the
		// shared array holds a word per token per thread.
		tokOK := s.NTok <= 1 || (s.Tok == s.NTok && s.Experts <= 1)
		if s.Split < 2 || !tokOK || s.Rowt > 1 || (s.Experts <= 1 && s.Slots > 1) {
			return nil, fmt.Errorf("kernels: MatVec: GroupSplit needs Split>1 and a plain "+
				"or indexed decode shape (got split=%d experts=%d slots=%d ntok=%d rowt=%d bias=%v)",
				s.Split, s.Experts, s.Slots, s.NTok, s.Rowt, s.Bias)
		}
		if gw := GroupSplitWidth(s.Split); gw%s.Split != 0 {
			return nil, fmt.Errorf("kernels: MatVec: GroupSplit=%d does not divide the %d-thread group",
				s.Split, gw)
		}
		if rpg := GroupSplitWidth(s.Split) / s.Split; s.Rows%rpg != 0 {
			return nil, fmt.Errorf("kernels: MatVec: GroupSplit leaves %d rows per group, "+
				"which does not divide Rows=%d", rpg, s.Rows)
		}
	}
	if s.Gate {
		// An indexed Gate is the mixture's up projection writing act(gate)*up
		// itself with its per-expert bias (llama.cpp's MUL_MAT_ID glu fusion).
		// pGate is the gate matvec's output at slot*Rows+row; swiglu-oai is
		// allowed because it clamps up as well as gate.
		dense := s.Experts <= 1
		okAct := s.GateAct == ActSiLU || s.GateAct == ActGELU ||
			(!dense && s.GateAct == ActSwiGLUOAI)
		allTok := s.NTok <= 1 || (dense && s.Tok == s.NTok)
		if (s.Split > 1 && !s.GroupSplit) || (dense && s.Slots > 1) || !allTok ||
			s.Rowt > 1 || (dense && s.Bias) || !okAct || s.SlotAct {
			return nil, fmt.Errorf("kernels: MatVec: Gate needs a final-row decode shape and "+
				"SiLU or GELU (split=%d group=%v experts=%d slots=%d ntok=%d rowt=%d bias=%v act=%v)",
				s.Split, s.GroupSplit, s.Experts, s.Slots, s.NTok, s.Rowt, s.Bias, s.GateAct)
		}
	}
	if s.Gate2 {
		ok := (s.GateAct == ActSiLU || s.GateAct == ActGELU) && !s.Gate && s.NTok <= 1 && s.Rowt <= 1 &&
			s.Experts <= 1 && s.Slots <= 1 && !s.Bias && (s.Split <= 1 || s.GroupSplit) && infoOf(s.T).float == 0 &&
			s.PickOf == 0 && s.ActRows <= 1 && into == nil
		if !ok {
			return nil, fmt.Errorf("kernels: MatVec: Gate2 needs a one-token dense quantized decode shape "+
				"writing its final row, SiLU or GELU (split=%d group=%v experts=%d ntok=%d rowt=%d bias=%v act=%v)",
				s.Split, s.GroupSplit, s.Experts, s.NTok, s.Rowt, s.Bias, s.GateAct)
		}
	}
	if s.BiasRows && (!s.Bias || s.NTok < 2 || s.Tok != s.NTok || s.Experts > 1) {
		return nil, fmt.Errorf("kernels: MatVec: BiasRows needs Bias and every token of NTok>1 "+
			"in one thread, dense (bias=%v ntok=%d tok=%d experts=%d)", s.Bias, s.NTok, s.Tok, s.Experts)
	}
	// NTok with Experts is a grouped mixture: pSel holds one expert per token
	// group (NTok/Tok of them) and every column of a group reads that expert's
	// sheet. The caller sorts (token, slot) pairs by expert and pads each run to
	// a whole group. No bias: pBias is indexed by row alone, and an expert bias
	// is added by IndexedBiasAdd afterwards.
	grouped := s.NTok > 1 && s.Experts > 1
	if s.NTok > 1 {
		if s.Slots > 1 || (grouped && (s.Bias || s.Rowt > 1 || s.Gate)) {
			return nil, fmt.Errorf("kernels: MatVec: NTok=%d with slots, or experts with a bias, "+
				"row tile or gate, is not supported", s.NTok)
		}
		// Split is allowed with NTok for decode batches: many sequences with every
		// token in one thread leave too few threads. Partials land at
		// splitSeg*Rows*Slots and a Reduce over Rows*NTok sums them, or, with
		// every token in one thread, the group reduces them (checked above).
		if s.Tok < 1 {
			s.Tok = 1
		}
		if s.NTok%s.Tok != 0 {
			return nil, fmt.Errorf("kernels: MatVec: NTok=%d is not a multiple of Tok=%d", s.NTok, s.Tok)
		}
		s.Slots, s.SlotAct = s.NTok, true
	} else {
		s.Tok = 1
	}
	qi := infoOf(s.T)
	if why := DeviceWhyNot(s.T); why != "" {
		return nil, fmt.Errorf("kernels: MatVec: %s", why)
	}
	// The sub-block is the format's unless a float caller narrows it; see
	// MatVecShape.Sub.
	sub := qi.sub
	if s.Sub == 0 {
		s.Sub = FloatSub(s.T, s.K)
	}
	if s.Sub > 0 {
		if qi.float == 0 {
			return nil, fmt.Errorf("kernels: MatVec: Sub=%d is float-only; %v stores %d-element "+
				"scales and cannot be walked below them", s.Sub, s.T, qi.sub)
		}
		if s.Sub%4 != 0 || s.Sub > qi.sub || s.K%s.Sub != 0 {
			return nil, fmt.Errorf("kernels: MatVec: Sub=%d must be a multiple of 4, at most %d, "+
				"and divide K=%d", s.Sub, qi.sub, s.K)
		}
		sub = s.Sub
	}
	if s.K <= 0 || (s.Sub == 0 && s.K%qi.blockE != 0) {
		return nil, fmt.Errorf("kernels: MatVec: K=%d must be a positive multiple of %d", s.K, qi.blockE)
	}
	if s.Rows <= 0 {
		return nil, fmt.Errorf("kernels: MatVec: Rows=%d must be positive", s.Rows)
	}
	if s.PickOf > 0 && (s.NTok > 1 || s.Experts > 1 || s.GroupSplit || s.Rowt > 1 || s.In.On || s.Gate ||
		s.BiasRows || s.ActRows > 1 || into != nil) {
		return nil, fmt.Errorf("kernels: MatVec: PickOf takes a dense one-token decode shape")
	}
	nsub := s.K / sub // sub-blocks per row, each with its own scale
	nb := s.K / 32    // 32-element activation blocks
	split := s.Split
	if split < 1 {
		split = 1
	}
	if nsub%split != 0 {
		return nil, fmt.Errorf("kernels: MatVec: Split=%d does not divide %d sub-blocks", split, nsub)
	}
	experts, slots := s.Experts, s.Slots
	if experts < 1 {
		experts = 1
	}
	// Slots means expert slots to the MoE caller, so a dense launch resets it,
	// but a token batch sets Slots = NTok with Experts = 1 and must keep it.
	if experts == 1 && s.NTok <= 1 {
		slots = 1
	}
	if slots < 1 {
		slots = 1
	}
	// A slot per activation (SlotAct) may name one expert many times, so only
	// the decode top-k -- one shared activation -- is bounded by the bank.
	if slots > experts && s.NTok <= 1 && !s.SlotAct {
		return nil, fmt.Errorf("kernels: MatVec: Slots=%d exceeds Experts=%d", slots, experts)
	}
	spg := nsub / split // sub-blocks per thread
	// pw primary words then hw secondary ones; see qinfo.hi and packSub.
	pw, hw := sub*qi.bits/32, sub*qi.hi/32
	words := pw + hw  // payload u32 per sub-block
	awords := sub / 4 // activation u32 per sub-block

	group := matvecGroup
	if s.GroupSplit {
		group = GroupSplitWidth(split)
	}
	// The shape is in the name so a profiler can tell projections apart; nothing
	// looks a matvec up by it.
	name := fmt.Sprintf("matvec_%s_%dx%d_s%d", strings.ToLower(s.T.String()), s.Rows, s.K, s.Split)
	if s.PickOf > 0 {
		name += fmt.Sprintf("_pick%d", s.PickOf)
	}
	if s.GroupSplit {
		name += "g"
	}
	if s.Experts > 1 {
		name += fmt.Sprintf("_e%dx%d", s.Experts, s.Slots)
	}
	if s.NTok > 1 {
		name += fmt.Sprintf("_t%d", s.NTok)
	}
	if s.ActRows > s.NTok {
		name += fmt.Sprintf("of%d", s.ActRows)
	}
	if s.Gate {
		name += "_act"
	}
	if s.Gate2 {
		name += "_gu"
	}
	if s.BiasRows {
		name += "_res"
	}
	if s.In.On {
		name += actInName(s.In)
	}
	var b *ir.Builder
	suffix := ""
	cta := func() ir.Value { return b.CTAID() }
	if into == nil {
		b = ir.New(name, [3]int{group, 1, 1})
	} else {
		b, suffix, cta = into.b, into.suffix, into.cta
	}
	done := func() (*ir.Kernel, error) {
		if into != nil {
			return nil, nil
		}
		return b.Done(), nil
	}
	pQS := b.Param("pQS"+suffix, ir.U32) // payload, transposed
	// A float format has no scale plane; its slot carries the float activation
	// instead, since a float weight keeps its precision only against a float
	// input. The launch keeps its arity.
	var pD, pX ir.Value
	if qi.float > 0 {
		pX = b.Param("pX"+suffix, ir.F32)
	} else {
		pD = b.Param("pD"+suffix, ir.U32) // super-block f16 scales (and minima)
	}
	pSC := b.Param("pSC"+suffix, ir.U32) // per-sub-block integer scales, packed bytes
	var pA ir.Value                      // int8 activations, 4 per u32
	// pAX: activation scale per 32 at [0,nb), activation sums per 16 at
	// [nb, nb+n16). Per-16 sums serve both sub-block sizes -- a 32-element
	// format adds two of them, which is one instruction and avoids a second
	// array.
	var pAX ir.Value
	var pIn actParamSet
	switch {
	case into == nil && s.In.On:
		// The operand prologue (ActIn): pIn [pNW] in pA/pAX's place.
		if s.NTok > 1 || s.Experts > 1 || s.ActRows > 1 || qi.float > 0 {
			return nil, fmt.Errorf("kernels: MatVec: ActIn needs a one-token dense quantized decode shape")
		}
		if err := s.In.check(s.K, group); err != nil {
			return nil, err
		}
		// Emitted just before the k loop, behind the weight prefetch.
		pIn = actParams(b, s.In)
	case into == nil:
		pA = b.Param("pA", ir.U32)
		pAX = b.Param("pAX", ir.F32)
	default:
		pA, pAX = into.pA, into.pAX
	}
	pOut := b.Param("pOut"+suffix, ir.F32)
	var pBias ir.Value
	if s.Bias {
		pBias = b.Param("pBias"+suffix, ir.F32)
	}
	var pGate ir.Value
	if s.Gate {
		pGate = b.Param("pGate"+suffix, ir.F32)
	}
	if epi != nil {
		epi.Params(b)
	}
	// Gate2: the gate's weight, the same format and shape as up's (pQS).
	var pQS2, pD2, pSC2 ir.Value
	if s.Gate2 {
		pQS2 = b.Param("pQS2"+suffix, ir.U32)
		pD2 = b.Param("pD2"+suffix, ir.U32)
		pSC2 = b.Param("pSC2"+suffix, ir.U32)
	}

	var pSel ir.Value
	if experts > 1 {
		pSel = b.Param("pSel"+suffix, ir.U32) // Slots expert ids, chosen by the router
	}
	var pPick ir.Value
	if s.PickOf > 0 {
		pPick = b.Param("pPick", ir.U32) // the weight row each output row reads
	}

	orows := b.Const(ir.U32, int64(s.Rows)) // rows of ONE expert; the output row count
	// nrow2 is how many threads cover the rows: Rows, or Rows/Rowt with a tile.
	// It is not the stride between tile rows; those are adjacent (see tileOff).
	nrow2 := s.Rows / s.Rowt
	trows := orows
	if s.Rowt > 1 {
		trows = b.Const(ir.U32, int64(nrow2))
	}
	one := b.Const(ir.U32, 1)
	// GroupSplit lays the group out as Split segments of rpg rows, row fastest,
	// so adjacent lanes read adjacent rows of the transposed layout.
	var shPart, segLocal, rowLocal ir.Value
	rpg := 0
	if s.GroupSplit {
		rpg = group / split
		tid := b.TID()
		rowLocal = b.Rem(ir.U32, tid, b.Const(ir.U32, int64(rpg)))
		segLocal = b.Div(ir.U32, tid, b.Const(ir.U32, int64(rpg)))
		// Declared before every Loop; ir.Validate refuses it anywhere else.
		if into != nil && into.hasSh {
			shPart = into.shPart
		} else {
			n := group * max(s.Tok, 1)
			if s.Gate2 {
				n = 2 * group
			}
			shPart = b.Shared("part", ir.F32, n)
		}
	}
	flat := b.Add(ir.U32, b.Mul(ir.U32, cta(), b.NTID()), b.TID())
	// A thread owns a token group, not a token: with Tok > 1 the grid is
	// Rows*(NTok/Tok), and clamping to Rows*NTok would let groups overlap.
	groups := slots / s.Tok
	flat = b.Min(ir.U32, flat, b.Const(ir.U32, int64(nrow2*groups*split-1)))
	// row varies fastest so consecutive threads hit consecutive rows, which the
	// transposed layout needs to coalesce.
	orow := b.Rem(ir.U32, flat, trows)
	rest := b.Div(ir.U32, flat, trows)
	// The tile's base row is the thread index times Rowt, emitted only when
	// Rowt > 1: even a Mul by Const(1) is an IR instruction and would change
	// every dense kernel's golden (TestDenseKernelsUnchanged).
	if s.Rowt > 1 {
		orow = b.Mul(ir.U32, orow, b.Const(ir.U32, int64(s.Rowt)))
	}

	// The split segment and the expert slot are different things and must never
	// share a value; they are decoded once here.
	//
	// slot is not materialised when there are no experts (a dead Const would
	// change the dense goldens), but it must be whenever pSel is read, even at
	// one slot: ir.Value is a bare uint32, so an unset one silently refers to
	// instruction 0 (it crashed the Vulkan driver).
	slot, splitSeg := ir.Value(0), rest
	grp := ir.Value(0) // the token group; pSel's index when grouped
	var gslot ir.Value // the indexed in-group split's slot, from the group index
	switch {
	case s.GroupSplit && experts > 1:
		// Slot-major groups: Rows/rpg of them per slot. Rows is a multiple of rpg,
		// so no row clamp is needed; the slot is clamped so a surplus group
		// recomputes the last one (RULE 13).
		rb := b.Const(ir.U32, int64(s.Rows/rpg))
		c := cta()
		gslot = b.Min(ir.U32, b.Div(ir.U32, c, rb), b.Const(ir.U32, int64(slots-1)))
		orow = b.Add(ir.U32, b.Mul(ir.U32, b.Rem(ir.U32, c, rb), b.Const(ir.U32, int64(rpg))), rowLocal)
		splitSeg = segLocal
	case s.GroupSplit:
		// ctaid covers rpg rows; the segment is the group-local one.
		orow = b.Add(ir.U32, b.Mul(ir.U32, cta(), b.Const(ir.U32, int64(rpg))), rowLocal)
		orow = b.Min(ir.U32, orow, b.Sub(ir.U32, orows, one))
		splitSeg = segLocal
	}
	switch {
	case s.GroupSplit && experts > 1:
		slot = gslot
	case s.GroupSplit && slots > 1:
		slot = b.Const(ir.U32, 0) // one token group: every token is in this thread
	case slots > 1:
		ngrp := b.Const(ir.U32, int64(groups))
		g := b.Rem(ir.U32, rest, ngrp)
		splitSeg = b.Div(ir.U32, rest, ngrp)
		// slot is this group's first token; the Tok loops address the rest as
		// immediate offsets. At Tok == 1 group and slot coincide.
		slot, grp = g, g
		if s.Tok > 1 {
			slot = b.Mul(ir.U32, g, b.Const(ir.U32, int64(s.Tok)))
		}
	case experts > 1:
		slot = b.Const(ir.U32, 0) // one slot: always pSel[0]
	}

	// An expert is a sheet: the container packs a bank as n_expert independent
	// packed matrices back to back (jlm.sheetsOf), so the stride is one expert's
	// row count and the expert adds a flat base. qs, d and sc hold different
	// word counts per sheet, so each array gets its own base; sharing one would
	// read another expert's scales against this expert's payload.
	//
	// The originals are named orow/orows so a weight expression still written in
	// terms of the bare row cannot silently compute a wrong address.
	wrow, wrowD, wrowSC, bankRows := orow, orow, orow, orows
	// wsel and wrows are the weight's row and row count where the narrow d
	// plane and a float row address them: the output row and Rows, or the
	// picked row and PickOf.
	wsel, wrows := orow, s.Rows
	if s.PickOf > 0 {
		// An id past the matrix (the pick's padding) reads its last row; the
		// caller reads only the ids it asked for.
		wsel = b.Min(ir.U32, b.Load(ir.U32, pPick, orow, 0), b.Const(ir.U32, int64(s.PickOf-1)))
		wrows = s.PickOf
		wrow, wrowD, wrowSC, bankRows = wsel, wsel, wsel, b.Const(ir.U32, int64(s.PickOf))
	}
	// dExpAdd adds the expert's base without the row, which the narrow d plane
	// needs: rows share a word there, so the row enters the index divided. A
	// function, not a value, so the dense path emits nothing extra.
	dExpAdd := func(v ir.Value) ir.Value { return v }
	var eSel ir.Value // the selected expert, for a float bank's row-major base
	if experts > 1 {
		nq, nd, nsc, err := PackedWords(s.T, s.Rows, s.K)
		if err != nil {
			return nil, err
		}
		selIdx := slot
		if grouped {
			selIdx = grp
		}
		e := b.Load(ir.U32, pSel, selIdx, 0)
		eSel = e
		base := func(words int) ir.Value {
			if words == 0 {
				return orow
			}
			return b.Add(ir.U32, b.Mul(ir.U32, e, b.Const(ir.U32, int64(words))), orow)
		}
		wrow, wrowD, wrowSC = base(nq), base(nd), base(nsc)
		dExpAdd = func(v ir.Value) ir.Value {
			return b.Add(ir.U32, b.Mul(ir.U32, e, b.Const(ir.U32, int64(nd))), v)
		}
		// Within a sheet the packed stride is the sheet height.
		bankRows = orows
	}
	// Row j of the tile is the base row plus j: a compile-time displacement, and
	// in the transposed layout Rowt consecutive u32, so a lane reads one
	// contiguous run. At Rowt == 1 nothing new is emitted.
	tileOff := func(j int) int64 { return int64(j) }

	// pAX is flat, not per-slot, because one Quantize launch over Slots*K writes
	// all the scales and then all the per-16 sums: slot j's scales are at j*nb
	// and its sums at Slots*nb + j*(K/16). A single base of j*(nb + K/16) would
	// read another slot's floats. pA is K/4 u32 per vector, contiguous.
	aBase, axS, axU := ir.Value(0), ir.Value(0), ir.Value(0)
	slotAct := s.SlotAct && slots > 1
	// One vector read out of ActRows quantized together (a ragged step's one
	// row that wants the head): its scales lead, and its sums follow every
	// vector's scales.
	sumOff := int64(max(s.ActRows, 1) * nb)
	if slotAct {
		aBase = b.Mul(ir.U32, slot, b.Const(ir.U32, int64(s.K/4)))
		axS = b.Mul(ir.U32, slot, b.Const(ir.U32, int64(nb)))
		axU = b.Mul(ir.U32, slot, b.Const(ir.U32, int64(s.K/16)))
		sumOff = int64(max(slots, s.ActRows) * nb)
	}

	mask := b.Const(ir.U32, 0x0F0F0F0F)
	four := b.Const(ir.U32, 4)
	zeroF := b.ConstF32(0)
	negBiasK := b.ConstF32(-qi.biasK)
	// Q8_0 is the one format whose weights reach -128.
	dot4 := b.Dot4Bounded
	if qi.signedQ {
		dot4 = b.Dot4
	}
	// The zero point can ride the weights instead of the activation sum: a packed
	// quant is unsigned and the true weight is q - biasK, so centering the bytes
	// makes the integer dot the corrected dot and removes the -biasK*sum(a) Fma.
	// Centering keeps every operand above -128, so Dot4Bounded still applies.
	//
	// Centering is a fixed cost per (word, row) while the Fma it removes is per
	// (row, token), so it only pays at larger Tok; decode (Tok=1) would get
	// longer. eligible is the correctness predicate (a constant non-zero bias);
	// Center.MinTok is the profitability policy, settable so it can be measured.
	eligible := qi.biasK != 0 && !qi.biasArray
	centered := !s.Center.NoCenter && eligible && s.Tok >= s.Center.Tokens()
	var cAdd, cXor ir.Value
	if centered {
		c := (0x80 - int64(qi.biasK)) & 0xFF
		cAdd = b.Const(ir.U32, c|c<<8|c<<16|c<<24)
		cXor = b.Const(ir.U32, int64(0x80808080))
	}
	center := func(v ir.Value) ir.Value {
		if !centered {
			return v
		}
		return b.Xor(ir.U32, b.Add(ir.U32, v, cAdd), cXor)
	}

	segSub := b.Mul(ir.U32, splitSeg, b.Const(ir.U32, int64(spg))) // first sub-block
	wStart := b.Add(ir.U32, wrow, b.Mul(ir.U32, segSub,
		b.Mul(ir.U32, b.Const(ir.U32, int64(words)), bankRows)))
	aStart := b.Mul(ir.U32, segSub, b.Const(ir.U32, int64(awords)))
	if slotAct {
		aStart = b.Add(ir.U32, aStart, aBase)
	}
	// A float weight is row-major, not transposed: row r's words are
	// r*fw .. r*fw+fw-1, and a bank's expert e starts at e*Rows rows. Everything
	// outside the weight read is the quantized kernel's.
	fw, fsub := 0, 0 // u32 words per row, and per 32-element sub-block
	if qi.float > 0 {
		fw, fsub = s.K*qi.float/4, sub*qi.float/4
		rowF := wsel
		if experts > 1 {
			rowF = b.Add(ir.U32, b.Mul(ir.U32, eSel, orows), orow)
		}
		wStart = b.Add(ir.U32, b.Mul(ir.U32, rowF, b.Const(ir.U32, int64(fw))),
			b.Mul(ir.U32, segSub, b.Const(ir.U32, int64(fsub))))
	}

	if into == nil && s.In.On {
		// The weight stream starts before the prologue: its reduction and
		// barriers otherwise hold every word of it back (ActIn.Prefetch).
		if n := s.In.Prefetch; n != 0 && qi.float == 0 {
			if n < 0 || n > spg*words {
				n = spg * words
			}
			for j := 0; j < n; j++ {
				b.Prefetch(pQS, b.Add(ir.U32, wStart, b.Mul(ir.U32, b.Const(ir.U32, int64(j)), bankRows)))
			}
		}
		pA, pAX = actPrologue(b, s.K, group, s.In, pIn)
	}
	b.Loop(int64(spg))
	// One accumulator per (tile row, token column): the weight word is loaded
	// once and consumed by Tok dot products, which is the weight reuse.
	accs := make([]ir.Value, s.Rowt*s.Tok)
	for t := range accs {
		accs[t] = b.Phi(ir.F32, zeroF)
	}
	acc := accs[0]
	wIdx := b.Phi(ir.U32, wStart)   // into pQS
	aIdx := b.Phi(ir.U32, aStart)   // into pA
	subIdx := b.Phi(ir.U32, segSub) // which sub-block
	iaccs := make([]ir.Value, s.Rowt*s.Tok)
	for t := range iaccs {
		iaccs[t] = b.Const(ir.I32, 0)
	}
	// Gate2's second stream: the gate row's own accumulators.
	var gaccs, giaccs []ir.Value
	if s.Gate2 {
		gaccs, giaccs = []ir.Value{b.Phi(ir.F32, zeroF)}, []ir.Value{b.Const(ir.I32, 0)}
	}

	stream := SubDot(b, &SubDotEnv{S: s, PA: pA, PAX: pAX, PX: pX,
		Mask: mask, Four: four, ZeroF: zeroF, One: one, NegBK: negBiasK,
		Center: center, Dot4: dot4, Centered: centered,
		HW: hw, PW: pw, AWords: awords, NB: nb, FW: fw, FSub: fsub,
		BankRows: bankRows, WSel: wsel, WRowD: wrowD, WRowSC: wrowSC, WRows: wrows,
		DExpAdd: dExpAdd, TileOff: tileOff, SlotAct: slotAct, AxS: axS, AxU: axU, SumOff: sumOff}, wIdx, aIdx, subIdx)
	w := stream(pQS, pD, pSC, accs, iaccs)
	if s.Gate2 {
		stream(pQS2, pD2, pSC2, gaccs, giaccs)
	}

	b.SetPhi(wIdx, w)
	b.SetPhi(aIdx, b.Add(ir.U32, aIdx, b.Const(ir.U32, int64(awords))))
	b.SetPhi(subIdx, b.Add(ir.U32, subIdx, one))
	b.EndLoop()

	// Partial sums land transposed too, so the reduction reads them coalesced.
	// The output is indexed by slot, never by expert id, at the launch's own
	// Rows*Slots stride.
	outIdx := b.Add(ir.U32, orow, b.Mul(ir.U32, splitSeg, orows)) // the dense form, verbatim
	if slots > 1 {
		outIdx = b.Add(ir.U32, b.Mul(ir.U32, slot, orows), orow)
		if split > 1 {
			outIdx = b.Add(ir.U32, outIdx,
				b.Mul(ir.U32, splitSeg, b.Const(ir.U32, int64(s.Rows*slots))))
		}
	}
	// The in-group reduction: every segment of a row is a thread of this group,
	// so partials meet in threadgroup memory rather than a global buffer.
	//
	// Every thread computes the sum and every thread stores it: the IR has no
	// conditional outside a loop (RULE 13), and after the barrier the sum is a
	// pure function of read-only memory, so the threads sharing a row write the
	// same bits to the same address.
	//
	// An expert's bias is at eSel*Rows+row and its gate at slot*Rows+row; a
	// dense shape keeps pBias[row] and pGate[row], token t of a step whose
	// tokens share the thread t*Rows further on (a gate always, a bias with
	// BiasRows).
	biasAt := func(j, t int) ir.Value {
		if experts > 1 {
			return b.Load(ir.F32, pBias, b.Add(ir.U32, b.Mul(ir.U32, eSel, orows), orow), tileOff(j))
		}
		off := tileOff(j)
		if s.BiasRows {
			off += int64(t * s.Rows)
		}
		return b.Load(ir.F32, pBias, orow, off)
	}
	gateMul := func(j, t int, u ir.Value, own bool, gv ir.Value) ir.Value {
		gi := orow
		if experts > 1 {
			gi = b.Add(ir.U32, b.Mul(ir.U32, slot, orows), orow)
		}
		var g ir.Value
		if own {
			g = gv
		} else {
			g = b.Load(ir.F32, pGate, gi, tileOff(j)+int64(t*s.Rows))
		}
		if s.GateAct == ActSwiGLUOAI {
			// ActMul's swiglu-oai, element for element: the gate clamped at
			// 7 through QuickGELU, the up clamped to [-7, 7] and offset by 1.
			lim := b.ConstF32(7)
			a := act(b, b.Min(ir.F32, g, lim), ActQuickGELU)
			u = b.Add(ir.F32, b.Max(ir.F32, b.Min(ir.F32, u, lim), b.ConstF32(-7)), b.ConstF32(1))
			return b.Mul(ir.F32, a, u)
		}
		return b.Mul(ir.F32, act(b, g, s.GateAct), u)
	}
	if s.GroupSplit {
		// Token t's partials sit t groups further along the shared array.
		for t := 0; t < s.Tok; t++ {
			b.Store(shPart, b.TID(), accs[t], int64(t*group))
		}
		if s.Gate2 {
			// The gate's partials one group further along: shared memory is
			// sized 2*group for it.
			b.Store(shPart, b.TID(), gaccs[0], int64(group))
		}
		b.Barrier()
		for t := 0; t < s.Tok; t++ {
			sum := b.Load(ir.F32, shPart, rowLocal, int64(t*group))
			for sg := 1; sg < split; sg++ {
				// rpg is compile-time, so a segment is an IMMEDIATE displacement.
				sum = b.Add(ir.F32, sum, b.Load(ir.F32, shPart, rowLocal, int64(t*group+sg*rpg)))
			}
			if s.Bias {
				// Once, after the reduction.
				sum = b.Add(ir.F32, sum, biasAt(0, t))
			}
			if s.Gate {
				sum = gateMul(0, t, sum, false, 0)
			}
			if s.Gate2 {
				// The gate's row reduced in the same order as up's, which is
				// the order the gate's own launch reduces it in.
				gs := b.Load(ir.F32, shPart, rowLocal, int64(group))
				for sg := 1; sg < split; sg++ {
					gs = b.Add(ir.F32, gs, b.Load(ir.F32, shPart, rowLocal, int64(group+sg*rpg)))
				}
				sum = gateMul(0, t, sum, true, gs)
			}
			if experts > 1 {
				// The slot's output rows at the launch's own stride.
				b.Store(pOut, b.Add(ir.U32, b.Mul(ir.U32, slot, orows), orow), sum, 0)
				return done()
			}
			// Token t is t*Rows further along the output, as below.
			if epi != nil {
				at := b.Add(ir.U32, orow, b.Const(ir.U32, int64(t*s.Rows)))
				b.Store(pOut, at, epi.Apply(b, sum, at, orow), 0)
				continue
			}
			b.Store(pOut, orow, sum, int64(t*s.Rows))
		}
		return done()
	}
	// Every accumulator is stored, not just the first; unwritten columns would
	// hold plausible garbage.
	for j := 0; j < s.Rowt; j++ {
		for t := 0; t < s.Tok; t++ {
			a := accs[j*s.Tok+t]
			if s.Bias {
				bv := biasAt(j, t)
				if split > 1 {
					first := b.Lt(ir.U32, splitSeg, b.Const(ir.U32, 1))
					bv = b.Select(ir.F32, first, bv, b.ConstF32(0))
				}
				a = b.Add(ir.F32, a, bv)
			}
			if s.Gate {
				a = gateMul(j, t, a, false, 0)
			}
			if s.Gate2 {
				a = gateMul(j, t, a, true, gaccs[0])
			}
			// Token t is t*Rows further along the output; tile row j is the next
			// row, so the tile's outputs are contiguous like its inputs.
			if epi != nil {
				at := b.Add(ir.U32, outIdx, b.Const(ir.U32, int64(t*s.Rows)+tileOff(j)))
				row := b.Add(ir.U32, outIdx, b.Const(ir.U32, tileOff(j)))
				b.Store(pOut, at, epi.Apply(b, a, at, row), 0)
				continue
			}
			b.Store(pOut, outIdx, a, int64(t*s.Rows)+tileOff(j))
		}
	}
	_ = acc
	return done()
}

// MatVecSegments is several decode matvecs over one activation in one launch:
// the attention's q, k and v, whose weights may be different formats. Segment
// i owns a contiguous run of workgroups and runs matVec's own body for its
// shape there, guarded by a loop that runs once in its workgroups and never in
// the others -- the IR has no conditional outside a loop, and a trip count
// that is uniform per workgroup is exactly that conditional.
//
// It works because every format is known when the kernel is emitted: nothing
// is repacked or copied and each segment writes its own output. A small k or
// v matvec cannot fill a large card alone; here it rides q's grid.
//
// Parameters: pA, pAX, then per segment pQS, pD, pSC, pOut and, when the
// segment has one, pBias. Every segment takes the same Split: in-group, or 1,
// so no segment needs a separate reduction.
//
// The segments may carry a decode step of a few sequences, every token in one
// thread (Tok == NTok, the same for all): the grid is decode's and each
// segment writes NTok rows of its own output.
func MatVecSegments(segs []MatVecShape, split int, grp bool) (*ir.Kernel, error) {
	return MatVecSegmentsIn(segs, split, grp, ActIn{})
}

// MatVecSegmentsIn is MatVecSegments with the operand prologue (ActIn) when
// in.On: parameters pIn [pNW] in pA/pAX's place, so the norm-and-quantize
// launch before q/k/v goes.
func MatVecSegmentsIn(segs []MatVecShape, split int, grp bool, in ActIn) (*ir.Kernel, error) {
	blocks, group, err := SegmentsGrid(segs, split, grp)
	if err != nil {
		return nil, err
	}
	name := "matvecseg"
	for _, sg := range segs {
		name += fmt.Sprintf("_%s_%d", strings.ToLower(sg.T.String()), sg.Rows)
	}
	name += fmt.Sprintf("x%d_s%d", segs[0].K, split)
	if segs[0].NTok > 1 {
		name += fmt.Sprintf("_t%d", segs[0].NTok)
	}
	if grp {
		name += "g"
	}
	var ctx segCtx
	if in.On {
		if segs[0].NTok > 1 {
			return nil, fmt.Errorf("kernels: MatVecSegments: ActIn needs one token")
		}
		if err := in.check(segs[0].K, group); err != nil {
			return nil, err
		}
		name += actInName(in)
		b := ir.New(name, [3]int{group, 1, 1})
		pA, pAX := actPrologue(b, segs[0].K, group, in, actParams(b, in))
		ctx = segCtx{b: b, pA: pA, pAX: pAX}
	} else {
		b := ir.New(name, [3]int{group, 1, 1})
		ctx = segCtx{b: b, pA: b.Param("pA", ir.U32), pAX: b.Param("pAX", ir.F32)}
	}
	b := ctx.b
	if grp {
		ctx.shPart, ctx.hasSh = b.Shared("part", ir.F32, group*max(segs[0].Tok, 1)), true
	}
	start := 0
	for i, sg := range segs {
		sg.Split, sg.GroupSplit = split, grp
		lo, hi := b.Const(ir.U32, int64(start)), b.Const(ir.U32, int64(start+blocks[i]))
		c := b.CTAID()
		zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
		b.LoopN(b.Select(ir.U32, b.Lt(ir.U32, c, hi), b.Select(ir.U32, b.Lt(ir.U32, c, lo), zero, one), zero))
		seg := ctx
		seg.suffix = fmt.Sprintf("%d", i)
		seg.cta = func() ir.Value { return b.Sub(ir.U32, b.CTAID(), lo) }
		if _, err := matVec(sg, &seg, nil); err != nil {
			return nil, fmt.Errorf("kernels: MatVecSegments: segment %d: %w", i, err)
		}
		b.EndLoop()
		start += blocks[i]
	}
	return b.Done(), nil
}

// SegmentsGrid is MatVecSegments' launch: the workgroups each segment owns, in
// order, and the workgroup width. It refuses what the kernel cannot express.
func SegmentsGrid(segs []MatVecShape, split int, grp bool) (blocks []int, group int, err error) {
	if len(segs) < 2 {
		return nil, 0, fmt.Errorf("kernels: MatVecSegments: %d segment(s)", len(segs))
	}
	if split > 1 && !grp {
		return nil, 0, fmt.Errorf("kernels: MatVecSegments: split %d needs the in-group reduction", split)
	}
	group = matvecGroup
	if grp {
		group = GroupSplitWidth(split)
	}
	for _, sg := range segs {
		switch {
		case sg.K != segs[0].K:
			return nil, 0, fmt.Errorf("kernels: MatVecSegments: k %d and %d share no activation", sg.K, segs[0].K)
		case sg.NTok != segs[0].NTok || sg.Tok != segs[0].Tok || sg.ActRows != segs[0].ActRows ||
			(sg.NTok > 1 && sg.Tok != sg.NTok):
			return nil, 0, fmt.Errorf("kernels: MatVecSegments: segments must carry the same tokens, "+
				"all in one thread (ntok %d tok %d)", sg.NTok, sg.Tok)
		case sg.Experts > 1 || sg.Slots > 1 || sg.Rowt > 1 || sg.Gate || sg.BiasRows || IsFloat(sg.T):
			return nil, 0, fmt.Errorf("kernels: MatVecSegments: %v %dx%d is not a plain quantized decode shape", sg.T, sg.Rows, sg.K)
		case grp && !GroupSplitOK(sg.Rows, split):
			return nil, 0, fmt.Errorf("kernels: MatVecSegments: %d rows at in-group split %d", sg.Rows, split)
		}
		blocks = append(blocks, (sg.Rows*split+group-1)/group)
	}
	return blocks, group, nil
}

// GroupSplitWidth is the workgroup width of an in-group split matvec: 32 rows
// by Split segments, so every warp is one segment across 32 contiguous rows,
// clamped to [128, 1024]. A fixed 128 scattered a warp's loads over several
// rows past Split 4.
func GroupSplitWidth(split int) int {
	return min(max(32*split, matvecGroup), 1024)
}

// GroupSplitOK reports whether MatVec accepts an in-group split of rows.
func GroupSplitOK(rows, split int) bool {
	w := GroupSplitWidth(split)
	return split >= 2 && w%split == 0 && rows%(w/split) == 0
}

// matvecGroup is the workgroup width of both the matvec and its reduction; a
// package constant because MatVec validates GroupSplit against it.
const matvecGroup = 128

// Reduce sums Split partial rows into the final output. It is the global
// alternative to MatVecShape.GroupSplit's in-group reduction.
func Reduce(rows, split int) (*ir.Kernel, error) {
	if rows <= 0 || split < 2 {
		return nil, fmt.Errorf("kernels: Reduce: rows=%d split=%d out of range", rows, split)
	}
	const group = matvecGroup
	b := ir.New("reduce", [3]int{group, 1, 1})
	pIn := b.Param("pIn", ir.F32)
	pOut := b.Param("pOut", ir.F32)

	nrows := b.Const(ir.U32, int64(rows))
	flat := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	row := b.Min(ir.U32, flat, b.Sub(ir.U32, nrows, b.Const(ir.U32, 1)))
	zeroF := b.ConstF32(0)

	b.Loop(int64(split))
	acc := b.Phi(ir.F32, zeroF)
	idx := b.Phi(ir.U32, row)
	b.SetPhi(acc, b.Add(ir.F32, acc, b.Load(ir.F32, pIn, idx, 0)))
	b.SetPhi(idx, b.Add(ir.U32, idx, nrows))
	b.EndLoop()

	b.Store(pOut, row, acc, 0)
	return b.Done(), nil
}

// StreamWide is a read-bandwidth probe: every thread sums per u32 with a grid
// stride and writes one result, reading w consecutive words per step, so the
// same bytes are read with 1/w the load instructions.
//
// It is a diagnostic, not a kernel the engine runs: the IR has no vector
// load, so this asks whether w consecutive scalar loads buy bandwidth. spread
// > 0 makes a lane's w words strided instead of adjacent, which keeps the
// request count and destroys contiguity, separating the two explanations.
func StreamWide(per, w int) (*ir.Kernel, error) { return StreamWideSpread(per, w, 0) }

func StreamWideSpread(per, w, spread int) (*ir.Kernel, error) {
	if per < 1 || w < 1 {
		return nil, fmt.Errorf("kernels: StreamWide: per=%d w=%d must be positive", per, w)
	}
	// per must be a whole number of lane-widths: the loop runs per/w times, so
	// otherwise it reads less than the caller times (overstating bandwidth), and
	// per < w makes Loop(0), which wraps on CUDA.
	if per%w != 0 {
		return nil, fmt.Errorf("kernels: StreamWide: per=%d is not a whole number of "+
			"%d-word lanes; the loop would run %d times and read %d of %d words while "+
			"the caller times all of them", per, w, per/w, (per/w)*w, per)
	}
	if spread > 0 {
		// Thread t owns word t of w tiles `spread` apart, so a SIMD group's lanes
		// stay contiguous within each tile and a lane's own w words do not.
		//
		// spread must be the thread count: otherwise the w tiles overlap, part of
		// the buffer is never read, and the rest is re-read from cache, which
		// reads above the roofline.
		const group = 256
		b := ir.New("streamwidespread", [3]int{group, 1, 1})
		pIn := b.Param("pIn", ir.U32)
		pN := b.Param("pN", ir.U32)
		pOut := b.Param("pOut", ir.U32)
		stride := b.Mul(ir.U32, b.Load(ir.U32, pN, b.Const(ir.U32, 0), 0), b.Const(ir.U32, int64(w)))
		tid := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
		zero := b.Const(ir.U32, 0)
		b.Loop(int64(per / w))
		acc := b.Phi(ir.U32, zero)
		idx := b.Phi(ir.U32, tid)
		sum := acc
		for i := 0; i < w; i++ {
			sum = b.Add(ir.U32, sum, b.Load(ir.U32, pIn, idx, int64(i*spread)))
		}
		b.SetPhi(acc, sum)
		b.SetPhi(idx, b.Add(ir.U32, idx, stride))
		b.EndLoop()
		b.Store(pOut, tid, acc, 0)
		return b.Done(), nil
	}
	const group = 256
	b := ir.New("streamwidespread", [3]int{group, 1, 1})
	pIn := b.Param("pIn", ir.U32)
	pN := b.Param("pN", ir.U32)
	pOut := b.Param("pOut", ir.U32)

	// The grid stride is w words per thread, so thread t owns [t*w, t*w+w) of
	// every stride-sized tile and the tile is contiguous across the group.
	stride := b.Mul(ir.U32, b.Load(ir.U32, pN, b.Const(ir.U32, 0), 0), b.Const(ir.U32, int64(w)))
	tid := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	base := b.Mul(ir.U32, tid, b.Const(ir.U32, int64(w)))

	zero := b.Const(ir.U32, 0)
	b.Loop(int64(per / w))
	acc := b.Phi(ir.U32, zero)
	idx := b.Phi(ir.U32, base)
	sum := acc
	for i := 0; i < w; i++ {
		sum = b.Add(ir.U32, sum, b.Load(ir.U32, pIn, idx, int64(i)))
	}
	b.SetPhi(acc, sum)
	b.SetPhi(idx, b.Add(ir.U32, idx, stride))
	b.EndLoop()

	b.Store(pOut, tid, acc, 0)
	return b.Done(), nil
}

func log2(n int) int {
	k := 0
	for n > 1 {
		n >>= 1
		k++
	}
	return k
}

// rep4 repeats a byte mask across a u32, which is how every per-byte constant
// in these kernels is spelled.
func rep4(m uint32) uint32 { return m&0xFF | (m&0xFF)<<8 | (m&0xFF)<<16 | (m&0xFF)<<24 }
