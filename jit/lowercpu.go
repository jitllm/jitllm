package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/ir"
	"github.com/jitllm/jitllm/jit/x86"
)

// cpuLowering lowers an ordered, bound kernel to x86 through jit/x86: each
// entry of its sequence the instructions that do it, the call's arguments
// addressed from RDI. Shuffles, where set, interleaves within lanes by a
// shuffle and a permute rather than an unpack.
type cpuLowering struct {
	Shuffles bool
}

func (l cpuLowering) Lower(k *Kernel, d Device) (Lowered, error) {
	if k.Seq == nil {
		return Lowered{}, fmt.Errorf("the x86 lowering of %s: the kernel is not ordered", k.Name)
	}
	e := &x86Lowerer{k: k, a: &x86.Buf{}, top: map[int]int{}, end: map[int]int{}, preset: map[int]bool{}, home: k.Homes(), words: k.VectorWords(), shuffles: l.Shuffles,
		isa: x86.ISA(d.ISA)}
	if r := x86.Reg(k.Ones - 1); k.Ones > 0 {
		e.a.VPCMPEQW(r, r, r)
		e.a.VPSRLW(r, r, 15)
	}
	for _, en := range k.Seq {
		if e.err != nil {
			break
		}
		e.cur = -1
		switch en.Kind {
		case EntryNode:
			e.cur = en.I
			e.node(en.I)
			if s := k.Nodes[en.I].Spill; s > 0 && k.Nodes[en.I].Reg > 0 {
				e.a.MOVStore(spillSlot(s), x86.Reg(k.Nodes[en.I].Reg-1))
			}
		case EntryStore:
			e.store(en.I)
		case EntryLoop:
			e.enter(en.I)
		case EntryEnd:
			e.leave(en.I)
		}
	}
	if e.err != nil {
		return Lowered{}, fmt.Errorf("the x86 lowering of %s: %w", k.Name, e.err)
	}
	e.a.VZEROUPPER()
	e.a.RET()
	code, err := e.a.Bytes()
	if err != nil {
		return Lowered{}, err
	}
	return Lowered{CPU: code}, nil
}

type x86Lowerer struct {
	k        *Kernel
	home     []int
	words    map[int]bool
	a        *x86.Buf
	top, end map[int]int
	// preset is the loops whose counter was set as an earlier loop began.
	preset   map[int]bool
	err      error
	shuffles bool
	// cur is the node being made, -1 between nodes; reloads counts the
	// spilled reads, which take the reload registers in turn.
	cur, reloads int
	// isa is the instruction sets the host executes.
	isa x86.ISA
}

// shiftCount is the red-zone word a vector shift by a general register's
// count reads it from: slot 16, which no spill takes.
var shiftCount = x86.At(x86.RSP, -128)

// spillSlot is red-zone slot s, 1-based, below RSP.
func spillSlot(s int) x86.Mem { return x86.At(x86.RSP, int32(-8*s)) }

// spillRegs are the registers a spilled value is loaded into at a read
// (Regs.Reloads in the CPU profile).
var spillRegs = [2]x86.Reg{x86.R13, x86.RBX}

func (e *x86Lowerer) fail(format string, args ...any) {
	if e.err == nil {
		e.err = fmt.Errorf(format, args...)
	}
}

func (e *x86Lowerer) reg(i int) x86.Reg {
	if nd := e.k.Nodes[i]; nd.Spill < 0 {
		r := spillRegs[e.reloads%2]
		e.reloads++
		e.a.MOVLoad(r, x86.At(x86.RDI, int32(nd.Imm)))
		return r
	}
	if s := e.k.Nodes[i].Spill; s > 0 && i != e.cur {
		r := spillRegs[e.reloads%2]
		e.reloads++
		e.a.MOVLoad(r, spillSlot(s))
		return r
	}
	r := e.k.Nodes[i].Reg
	if r == 0 {
		e.fail("node %d (%s) has no register", i, e.k.Nodes[i].Op)
	}
	return x86.Reg(max(r-1, 0))
}

// addr is the memory operand of buffer b at constant index node ix through
// pointer ptr: the buffer's address, or one of the kernel's copies.
func (e *x86Lowerer) addr(b, ix, ptr int) x86.Mem {
	k := e.k
	nd := k.Nodes[ix]
	if nd.Op != OpConst {
		// A computed index: the buffer's address plus the index's register
		// times an element.
		switch {
		case k.Bufs[b].Word > 0:
			e.fail("an argument word of %s at a computed index", k.Bufs[b].Operand.Name)
		case e.words[ix] || nd.Type == ir.F32:
			// The binder holds this index in a vector register: an address
			// needs it in a general one.
			e.fail("an index of %s held in a vector register", k.Bufs[b].Operand.Name)
		}
		return x86.Idx(e.ptrReg(b, ptr), e.reg(ix), uint8(ElemBytes(k.Bufs[b])), 0)
	}

	if w := k.Bufs[b].Word; w > 0 {
		return x86.At(x86.RDI, int32(w-1)+int32(nd.Imm)*int32(ElemBytes(k.Bufs[b])))
	}
	return x86.At(e.ptrReg(b, ptr), int32(nd.Imm)*int32(ElemBytes(k.Bufs[b])))
}

// ptrReg is the register of buffer b's pointer ptr.
func (e *x86Lowerer) ptrReg(b, ptr int) x86.Reg {
	r := e.k.Bufs[b].Reg
	if ptr > 0 {
		r = e.k.Ptrs[ptr-1].Reg
		if s := e.k.Ptrs[ptr-1].Spill; r == 0 && s > 0 {
			e.a.MOVLoad(reloadReg, spillSlot(s))
			return reloadReg
		}
	}
	if r == 0 && ptr == 0 && e.k.Bufs[b].Reload && e.k.Bufs[b].Base > 0 {
		e.a.MOVLoad(reloadReg, x86.At(x86.RDI, int32(e.k.Nodes[e.k.Bufs[b].Base-1].Imm)))
		return reloadReg
	}
	if r == 0 {
		e.fail("%s has no register", e.k.Bufs[b].Operand.Name)
	}
	return x86.Reg(max(r-1, 0))
}

// reloaded reports whether param i is a buffer address the code reloads at
// each access.
func (e *x86Lowerer) reloaded(i int) bool {
	for _, bf := range e.k.Bufs {
		if bf.Base == i+1 && bf.Reload {
			return true
		}
	}
	return false
}

// ElemBytes is the bytes of one of buffer b's elements.
func ElemBytes(b Buf) int {
	if b.Half {
		return 2
	}
	return 4
}

// lanes is how many lanes loop c's code runs, 0 outside any loop.
// feedsHalves reports whether every reader of node i is made from halves.
func (e *x86Lowerer) feedsHalves(i int) bool {
	n := 0
	for r, nd := range e.k.Nodes {
		for _, a := range e.k.Operands(&nd) {
			if *a == i {
				if !e.half(r) {
					return false
				}
				n++
			}
		}
	}
	return n > 0
}

// half reports whether node i is made from halves: a narrowed value's
// words, which sit in a vector's low 128 bits.
func (e *x86Lowerer) half(i int) bool {
	return e.k.computedFrom(i, OpHalf)
}

func (e *x86Lowerer) lanes(c int) int {
	if c < 0 {
		return 0
	}
	return max(e.k.Loops[c].Lanes, e.k.Loops[c].Wide)
}

func (e *x86Lowerer) node(i int) {
	k, a := e.k, e.a
	nd := k.Nodes[i]
	switch nd.Op {
	case OpParam:
		if nd.Reg == 0 && e.reloaded(i) || nd.Spill < 0 {
			return
		}
		a.MOVLoad(e.reg(i), x86.At(x86.RDI, int32(nd.Imm)))
	case OpFold:
		init := k.Nodes[nd.B]
		switch {
		case init.Op == OpFold:
			if nd.Reg != init.Reg {
				e.fail("a sum carried on out of its register")
			}
		case init.Op == OpConst && init.Imm == 0 && e.gpr(i):
			r := e.reg(i)
			a.XORQ(r, r)
		case init.Op == OpConst && init.Imm == 0:
			r := e.reg(i)
			a.VPXOR(r, r, r)
		case init.Op == OpLoad && e.home[nd.B] < 0 && init.Imm > 0:
			a.VMOVDQULoad(e.reg(i), e.addr(init.Buf, init.A, init.Ptr))
		case init.Op == OpLoad && e.home[nd.B] < 0:
			a.VBROADCASTSS(e.reg(i), e.addr(init.Buf, init.A, init.Ptr))
		case init.Op == OpConst && e.words[nd.B] && init.Reg > 0:
			a.VMOVAPSReg(e.reg(i), e.reg(nd.B))
		case init.Op != OpConst && (init.Reg > 0 || init.Spill != 0) && (e.home[nd.B] < 0 || !k.within(e.home[nd.B], nd.At)):
			// A sum that starts at a value made outside its loop: the value,
			// copied into the sum's register where it is not already there.
			r, from := e.reg(i), e.reg(nd.B)
			switch {
			case r == from:
			case e.gpr(i):
				a.MOVQ(r, from)
			default:
				a.VMOVAPSReg(r, from)
			}
		default:
			e.fail("a sum from %s", init.Op)
		}
	case OpLoad:
		if nd.Mem {
			return
		}
		home := e.home[i]
		m := e.addr(nd.Buf, nd.A, nd.Ptr)
		if e.gpr(i) && !k.Bufs[nd.Buf].Half {
			a.MOVLLoad(e.reg(i), m)
			return
		}
		if k.Bufs[nd.Buf].Half {
			if e.lanes(home) == 0 {
				a.VPINSRWLoad(e.reg(i), e.reg(i), m, 0)
				a.VCVTPH2PSReg(e.reg(i), e.reg(i))
				return
			}
			a.VCVTPH2PS(e.reg(i), m)
			return
		}
		switch {
		case home < 0 && nd.Type != ir.F32:
			a.VPBROADCASTD(e.reg(i), m)
		case home < 0:
			a.VBROADCASTSS(e.reg(i), m)
		case e.lanes(home) > 0:
			a.VMOVDQULoad(e.reg(i), m)
		default:
			a.VMOVSSLoad(e.reg(i), m)
		}
	case OpBitcast:
		switch {
		case e.gpr(nd.A) && !e.gpr(i):
			a.VMOVDToX(e.reg(i), e.reg(nd.A))
		case e.gpr(nd.A) && e.reg(i) != e.reg(nd.A):
			a.MOVQ(e.reg(i), e.reg(nd.A))
		case e.reg(i) != e.reg(nd.A):
			// The operand is read again after: its bits in a register of
			// their own.
			a.VMOVAPSReg(e.reg(i), e.reg(nd.A))
		}
	case OpDot4, OpDot4U:
		// Each lane's four byte products and C: A's magnitudes as unsigned
		// bytes times B with A's signs (B is never -128, so neither the
		// sign nor a pair's sum overflows), or A as it is where its bytes
		// are unsigned; the pairs summed by ones.
		if nd.Tmp == 0 || nd.Tmp2 == 0 || e.gpr(i) {
			e.fail("node %d: a byte dot outside vector lanes", i)
			return
		}
		t, u := x86.Reg(nd.Tmp-1), x86.Reg(nd.Tmp2-1)
		if nd.Op == OpDot4U || e.unsignedBytes(nd.A) {
			// A's bytes are already magnitudes: A times B as they are.
			u, t = e.reg(nd.A), e.reg(nd.B)
		} else {
			a.VPSIGNB(t, e.reg(nd.B), e.reg(nd.A))
			a.VPABSB(u, e.reg(nd.A))
		}
		if e.isa&x86.ISAVNNI != 0 {
			// VPDPBUSD sums each lane's four products into it at once.
			r := e.reg(i)
			switch {
			case r == e.reg(nd.C):
				a.VPDPBUSD(r, u, t)
			case r == u || r == t:
				// The result takes an operand's register: summed aside.
				s := x86.Reg(nd.Tmp - 1)
				a.VMOVAPSReg(s, e.reg(nd.C))
				a.VPDPBUSD(s, u, t)
				a.VMOVAPSReg(r, s)
			default:
				a.VMOVAPSReg(r, e.reg(nd.C))
				a.VPDPBUSD(r, u, t)
			}
			return
		}
		p := x86.Reg(nd.Tmp - 1)
		a.VPMADDUBSW(p, u, t)
		u, t = x86.Reg(nd.Tmp2-1), p
		ones := x86.Reg(e.k.Ones - 1)
		if e.k.Ones == 0 {
			a.VPCMPEQW(u, u, u)
			a.VPSRLW(u, u, 15)
			ones = u
		}
		a.VPMADDWD(t, t, ones)
		a.VPADDD(e.reg(i), e.reg(nd.C), t)
	case OpWidenHalf:
		switch {
		case e.gpr(nd.A):
			a.VMOVDToX(e.reg(i), e.reg(nd.A))
			a.VCVTPH2PSReg(e.reg(i), e.reg(i))
		case e.lanes(e.home[i]) == 0:
			// A word's low half in a vector register's first lane: the
			// conversion's first half.
			a.VCVTPH2PSReg(e.reg(i), e.reg(nd.A))
		default:
			// Each lane's low half: the high halves cleared, the dwords
			// packed to words within each 128-bit half, the halves' low
			// quadwords gathered, and the eight halves converted.
			r := e.reg(i)
			a.VPSLLD(r, e.reg(nd.A), 16)
			a.VPSRLD(r, r, 16)
			a.VPACKUSDW(r, r, r)
			a.VPERMQ(r, r, 0x08)
			a.VCVTPH2PSReg(r, r)
		}
	case OpConst:
		if nd.Type == ir.F32 && nd.Imm == 0 && nd.Reg > 0 {
			r := e.reg(i)
			a.VPXOR(r, r, r)
			return
		}
		if !e.words[i] {
			return
		}
		t := x86.Reg(nd.Tmp - 1)
		if nd.Tmp == 0 {
			e.fail("a splat with no scratch word")
		}
		a.MOVimm32(t, int32(uint32(nd.Imm)))
		a.VMOVDToX(e.reg(i), t)
		if e.feedsHalves(i) {
			a.VPSHUFDx(e.reg(i), e.reg(i), 0)
		} else {
			a.VPBROADCASTDReg(e.reg(i), e.reg(i))
		}
	case OpHalf:
		a.VCVTPS2PH(e.reg(i), e.reg(nd.A), e.lanes(e.home[i]) > 0, 0)
	case OpGtW:
		a.VPCMPGTWx(e.reg(i), e.reg(nd.A), e.reg(nd.B))
	case OpLess:
		if k.Nodes[nd.A].Type != ir.F32 {
			a.VPCMPGTD(e.reg(i), e.reg(nd.B), e.reg(nd.A))
			return
		}
		a.VCMPPS(e.reg(i), e.reg(nd.B), e.reg(nd.A), 14)
	case OpSelect:
		if nd.Tmp == 0 {
			e.fail("a select with no scratch register")
		}
		t := x86.Reg(nd.Tmp - 1)
		a.VPANDN(t, e.reg(nd.A), e.reg(nd.C))
		a.VPAND(e.reg(i), e.reg(nd.A), e.reg(nd.B))
		a.VPOR(e.reg(i), e.reg(i), t)
	case OpAndNot:
		if e.half(i) {
			a.VPANDNx(e.reg(i), e.reg(nd.B), e.reg(nd.A))
			return
		}
		a.VPANDN(e.reg(i), e.reg(nd.B), e.reg(nd.A))
	case OpOr:
		if e.gpr(i) {
			e.integer(i)
			return
		}
		a.VPOR(e.reg(i), e.reg(nd.A), e.reg(nd.B))
	case OpDupEven:
		a.VMOVSLDUP(e.reg(i), e.reg(nd.A))
	case OpDupOdd:
		a.VMOVSHDUP(e.reg(i), e.reg(nd.A))
	case OpSwapPairs:
		a.VPERMILPS(e.reg(i), e.reg(nd.A), 0xB1)
	case OpCopy:
		a.VMOVAPSReg(e.reg(i), e.reg(nd.A))
	case OpFirstLane:
		a.VPBLENDD(e.reg(i), e.reg(nd.B), e.reg(nd.A), 1)
	case OpAddSub:
		a.VADDSUBPS(e.reg(i), e.reg(nd.A), e.reg(nd.B))
	case OpSar:
		if k.Nodes[nd.B].Op != OpConst && !e.words[nd.B] {
			a.MOVStore(shiftCount, e.reg(nd.B))
			a.VPSRADMem(e.reg(i), e.reg(nd.A), shiftCount)
			return
		}
		if k.Nodes[nd.B].Op != OpConst {
			a.VPSRAVD(e.reg(i), e.reg(nd.A), e.reg(nd.B))
			return
		}
		a.VPSRAD(e.reg(i), e.reg(nd.A), byte(k.Nodes[nd.B].Imm))
	case OpEq:
		a.VCMPPS(e.reg(i), e.reg(nd.A), e.reg(nd.B), 0)
	case OpHMax, OpHMin:
		r, x := e.reg(i), e.reg(nd.A)
		if nd.Tmp == 0 {
			e.fail("a lane reduction with no scratch register")
		}
		t := x86.Reg(nd.Tmp - 1)
		step := a.VMAXPS
		if nd.Op == OpHMin {
			a.VEXTRACTI128(t, x, 1)
			step = a.VPMINSD
		} else {
			a.VEXTRACTF128(t, x, 1)
		}
		step(r, x, t)
		a.VSHUFPS(t, r, r, 0x4E)
		step(r, r, t)
		a.VSHUFPS(t, r, r, 0xB1)
		step(r, r, t)
	case OpMul, OpAdd, OpShr, OpAnd, OpShl:
		if b := k.Nodes[nd.B]; e.words[i] && b.Mem && (nd.Op == OpAdd || nd.Op == OpAnd) {
			m := e.addr(b.Buf, b.A, b.Ptr)
			if nd.Op == OpAdd {
				a.VPADDDMem(e.reg(i), e.reg(nd.A), m)
			} else {
				a.VPANDMem(e.reg(i), e.reg(nd.A), m)
			}
			return
		}
		if (e.words[i] || nd.Type == ir.F32) && nd.Op == OpAnd {
			x, y := e.reg(nd.A), e.reg(nd.B)
			if e.reg(i) == y {
				x, y = y, x
			}
			if e.half(i) {
				a.VPANDx(e.reg(i), x, y)
			} else {
				a.VPAND(e.reg(i), x, y)
			}
			return
		}
		if e.words[i] && nd.Op == OpAdd {
			a.VPADDD(e.reg(i), e.reg(nd.A), e.reg(nd.B))
			return
		}
		// A shift by a computed count shifts each lane by its own; only a
		// constant count is an immediate.
		if e.words[i] && (nd.Op == OpShl || nd.Op == OpShr) && k.Nodes[nd.B].Op != OpConst && !e.words[nd.B] {
			// One count for every lane, from a general register: through
			// the count's word below RSP.
			a.MOVStore(shiftCount, e.reg(nd.B))
			if nd.Op == OpShl {
				a.VPSLLDMem(e.reg(i), e.reg(nd.A), shiftCount)
			} else {
				a.VPSRLDMem(e.reg(i), e.reg(nd.A), shiftCount)
			}
			return
		}
		if e.words[i] && nd.Op == OpShl {
			if k.Nodes[nd.B].Op != OpConst {
				a.VPSLLVD(e.reg(i), e.reg(nd.A), e.reg(nd.B))
				return
			}
			a.VPSLLD(e.reg(i), e.reg(nd.A), byte(k.Nodes[nd.B].Imm))
			return
		}
		if e.words[i] && nd.Op == OpShr {
			if k.Nodes[nd.B].Op != OpConst {
				a.VPSRLVD(e.reg(i), e.reg(nd.A), e.reg(nd.B))
				return
			}
			a.VPSRLD(e.reg(i), e.reg(nd.A), byte(k.Nodes[nd.B].Imm))
			return
		}
		if e.words[i] && nd.Op == OpMul {
			a.VPMULLD(e.reg(i), e.reg(nd.A), e.reg(nd.B))
			return
		}
		if nd.Type != ir.F32 {
			e.integer(i)
			return
		}
		if b := k.Nodes[nd.B]; b.Mem {
			m := e.addr(b.Buf, b.A, b.Ptr)
			if nd.Op == OpMul {
				a.VMULPSMem(e.reg(i), e.reg(nd.A), m)
			} else {
				a.VADDPSMem(e.reg(i), e.reg(nd.A), m)
			}
			return
		}
		if nd.Op == OpMul {
			a.VMULPS(e.reg(i), e.reg(nd.A), e.reg(nd.B))
		} else {
			a.VADDPS(e.reg(i), e.reg(nd.A), e.reg(nd.B))
		}
	case OpSub, OpDiv, OpMin, OpMax:
		if e.gpr(i) {
			e.integer(i)
			return
		}
		x, y := e.reg(nd.A), e.reg(nd.B)
		if nd.Type != ir.F32 {
			// Integer lanes: signed for i32, unsigned for u32.
			signed := nd.Type == ir.I32
			switch {
			case nd.Op == OpSub:
				a.VPSUBD(e.reg(i), x, y)
			case nd.Op == OpMin && signed:
				a.VPMINSD(e.reg(i), x, y)
			case nd.Op == OpMin:
				a.VPMINUD(e.reg(i), x, y)
			case nd.Op == OpMax && signed:
				a.VPMAXSD(e.reg(i), x, y)
			case nd.Op == OpMax:
				a.VPMAXUD(e.reg(i), x, y)
			default:
				e.fail("node %d: %s of integer lanes", i, nd.Op)
			}
			return
		}
		switch nd.Op {
		case OpSub:
			a.VSUBPS(e.reg(i), x, y)
		case OpDiv:
			a.VDIVPS(e.reg(i), x, y)
		case OpMin:
			a.VMINPS(e.reg(i), x, y)
		case OpMax:
			a.VMAXPS(e.reg(i), x, y)
		}
	case OpSqrt:
		a.VSQRTPS(e.reg(i), e.reg(nd.A))
	case OpBroadcast:
		if x := k.Nodes[nd.A]; x.Mem {
			a.VBROADCASTSS(e.reg(i), e.addr(x.Buf, x.A, x.Ptr))
			return
		}
		a.VBROADCASTSSReg(e.reg(i), e.reg(nd.A))
	case OpZip:
		if nd.Tmp == 0 {
			e.fail("a zip with no scratch register")
		}
		t := x86.Reg(nd.Tmp - 1)
		a.VUNPCKLPS(t, e.reg(nd.A), e.reg(nd.B))
		a.VUNPCKHPS(e.reg(i), e.reg(nd.A), e.reg(nd.B))
		a.VPERM2F128(e.reg(i), t, e.reg(i), byte(0x20+0x11*nd.Imm))
	case OpUnpack:
		switch {
		case e.shuffles:
			a.VSHUFPS(e.reg(i), e.reg(nd.A), e.reg(nd.B), byte(0x44+0xAA*nd.Imm))
			a.VPERMILPS(e.reg(i), e.reg(i), 0xD8)
		case nd.Imm == 0:
			a.VUNPCKLPS(e.reg(i), e.reg(nd.A), e.reg(nd.B))
		default:
			a.VUNPCKHPS(e.reg(i), e.reg(nd.A), e.reg(nd.B))
		}
	case OpUpper:
		a.VEXTRACTF128(e.reg(i), e.reg(nd.A), 1)
	case OpRoundI32:
		a.VCVTPS2DQ(e.reg(i), e.reg(nd.A))
	case OpCvtI32:
		a.VCVTTPS2DQ(e.reg(i), e.reg(nd.A))
	case OpXor:
		if e.gpr(i) {
			e.integer(i)
			return
		}
		a.VPXOR(e.reg(i), e.reg(nd.A), e.reg(nd.B))
	case OpCvtF32:
		if src := e.reg(nd.A); e.gpr(nd.A) {
			// A word converted from a general register: an i32 is its low
			// half sign-extended, a u32 is the register as it is.
			if k.Nodes[nd.A].Type == ir.I32 {
				a.MOVSXD(src, src)
			}
			a.VCVTSI2SS(e.reg(i), e.reg(i), src)
			return
		}
		a.VCVTDQ2PS(e.reg(i), e.reg(nd.A))
	case OpFma:
		if c := k.Nodes[nd.C]; c.Op != OpLoad || !c.Mem {
			if r := e.reg(i); r != e.reg(nd.C) {
				switch r {
				case e.reg(nd.A):
					a.VFMADD213PS(r, e.reg(nd.B), e.reg(nd.C))
				case e.reg(nd.B):
					a.VFMADD213PS(r, e.reg(nd.A), e.reg(nd.C))
				default:
					a.VMOVAPSReg(r, e.reg(nd.A))
					a.VFMADD213PS(r, e.reg(nd.B), e.reg(nd.C))
				}
				return
			}
		}
		if e.reg(i) != e.reg(nd.C) {
			e.fail("an fma whose result is not its addend's register")
		}
		if b := k.Nodes[nd.B]; b.Mem {
			a.VFMADD231PSMem(e.reg(i), e.reg(nd.A), e.addr(b.Buf, b.A, b.Ptr))
			return
		}
		if x := k.Nodes[nd.A]; x.Mem {
			a.VFMADD231PSMem(e.reg(i), e.reg(nd.B), e.addr(x.Buf, x.A, x.Ptr))
			return
		}
		a.VFMADD231PS(e.reg(i), e.reg(nd.A), e.reg(nd.B))
	case OpHSum:
		r := e.reg(i)
		if r != e.reg(nd.A) || nd.Tmp == 0 {
			e.fail("a lane sum out of its operand's register")
		}
		t := x86.Reg(nd.Tmp - 1)
		a.VEXTRACTF128(t, r, 1)
		a.VADDPSx(r, r, t)
		a.VHADDPSx(r, r, r)
		a.VHADDPSx(r, r, r)
	default:
		e.fail("node %d: no x86 form of %s.%s", i, nd.Op, nd.Type)
	}
}

// unsignedBytes reports whether every byte of node i is below 0x80: a mask
// of bytes each of which is.
func (e *x86Lowerer) unsignedBytes(i int) bool {
	nd := e.k.Nodes[i]
	if nd.Op != OpAnd {
		return false
	}
	for _, c := range []int{nd.A, nd.B} {
		if v, ok := e.constWord(c); ok && v&0x80808080 == 0 {
			return true
		}
	}
	return false
}

// constWord is node i's word where it is fixed: a constant, or a constant
// block's word at a constant index.
func (e *x86Lowerer) constWord(i int) (uint32, bool) {
	nd := e.k.Nodes[i]
	switch {
	case nd.Op == OpConst:
		return uint32(nd.Imm), true
	case nd.Op == OpLoad && e.k.Bufs[nd.Buf].Words != nil && e.k.Nodes[nd.A].Op == OpConst:
		w := e.k.Bufs[nd.Buf].Words
		if x := int(e.k.Nodes[nd.A].Imm); x >= 0 && x < len(w) {
			return w[x], true
		}
	}
	return 0, false
}

// keepIf jumps to skip where the compare just made leaves dst as it is: a
// minimum keeps a value at or below the other, a maximum one at or above,
// unsigned for u32 and signed for i32.
func (e *x86Lowerer) keepIf(nd Node, skip int) {
	signed := nd.Type == ir.I32
	switch {
	case nd.Op == OpMin && signed:
		e.a.JLE(skip)
	case nd.Op == OpMin:
		e.a.JBE(skip)
	case signed:
		e.a.JGE(skip)
	default:
		e.a.JAE(skip)
	}
}

// gpr reports whether node i is held in a general register: an integer a
// vector register does not hold.
func (e *x86Lowerer) gpr(i int) bool {
	nd := e.k.Nodes[i]
	return nd.Type != ir.F32 && nd.Type != ir.Pred && !nd.Mem && !e.words[i] && !e.k.needsVector(i)
}

// integer is an integer node in general registers: by a constant through
// its immediate form, else register to register, a shift by a register
// through SHRX or SHLX. A 32-bit result that can reach past 32 bits is cut
// back to them, so a later shift reads no bit above.
func (e *x86Lowerer) integer(i int) {
	k, a := e.k, e.a
	nd := k.Nodes[i]
	dst := e.reg(i)
	A, B := nd.A, nd.B
	commutes := nd.Op == OpAdd || nd.Op == OpMul || nd.Op == OpAnd || nd.Op == OpOr || nd.Op == OpXor
	if commutes && k.Nodes[A].Op == OpConst && k.Nodes[B].Op != OpConst {
		A, B = B, A
	}
	if c := k.Nodes[B]; c.Op == OpConst {
		src, imm := e.reg(A), int32(c.Imm)
		if nd.Op == OpMul {
			a.IMULimm(dst, src, imm)
		} else {
			if src != dst {
				a.MOVQ(dst, src)
			}
			switch nd.Op {
			case OpShr:
				a.SHRimm(dst, byte(c.Imm))
			case OpShl:
				a.SHLimm(dst, byte(c.Imm))
			case OpAnd:
				a.ANDimm32(dst, imm)
			case OpAdd:
				a.ADDimm(dst, imm)
			case OpSub:
				a.ADDimm(dst, -imm)
			case OpXor:
				a.XORimm32(dst, imm)
			case OpOr:
				a.ORimm32(dst, imm)
			case OpMin, OpMax:
				// dst is a; where a is past the constant, the constant.
				skip := a.Label()
				a.CMPLimm(dst, imm)
				e.keepIf(nd, skip)
				a.MOVLimm(dst, uint32(imm))
				a.Bind(skip)
			default:
				e.fail("node %d: no general-register form of %s by a constant", i, nd.Op)
				return
			}
		}
	} else {
		var x x86.Reg
		if c := k.Nodes[A]; c.Op == OpConst {
			// A constant first operand of an operation that has no form
			// taking it as an immediate: made in the reload register, free
			// between one access's address and the next.
			x = reloadReg
			a.MOVLimm(x, uint32(c.Imm))
		} else {
			x = e.reg(A)
		}
		y := e.reg(B)
		switch nd.Op {
		case OpShr:
			a.SHRX(dst, x, y)
		case OpShl:
			a.SHLX(dst, x, y)
		case OpSub:
			if dst == y {
				a.NEGQ(dst)
				a.ADDQ(dst, x)
			} else {
				if dst != x {
					a.MOVQ(dst, x)
				}
				a.SUBQ(dst, y)
			}
		case OpMin, OpMax:
			if dst == y {
				x, y = y, x
			}
			if dst != x {
				a.MOVQ(dst, x)
			}
			skip := a.Label()
			a.CMPL(dst, y)
			e.keepIf(nd, skip)
			a.MOVQ(dst, y)
			a.Bind(skip)
		case OpAdd, OpMul, OpAnd, OpOr, OpXor:
			if dst == y {
				x, y = y, x
			}
			if dst != x {
				a.MOVQ(dst, x)
			}
			switch nd.Op {
			case OpAdd:
				a.ADDQ(dst, y)
			case OpMul:
				a.IMUL(dst, y)
			case OpAnd:
				a.ANDQ(dst, y)
			case OpOr:
				a.ORQ(dst, y)
			case OpXor:
				a.XORQ(dst, y)
			}
		default:
			e.fail("node %d: no general-register form of %s", i, nd.Op)
			return
		}
	}
	switch nd.Op {
	case OpAdd, OpSub, OpMul, OpShl, OpXor, OpOr, OpMin, OpMax:
		// An immediate sign-extends: xor or or by one with its top bit set
		// sets the high half too.
		a.MOVL(dst, dst)
	}
}

func (e *x86Lowerer) store(i int) {
	s := e.k.Stores[i]
	m := e.addr(s.Buf, s.Ix, s.Ptr)
	wide := e.lanes(s.At) > 0
	if v := e.k.Nodes[s.Val]; v.Op == OpFold && e.k.Loops[v.At].Wide > 0 {
		wide = true
	}
	if s.Narrow {
		e.a.VMOVDQUStorex(m, e.reg(s.Val))
		return
	}
	if e.gpr(s.Val) {
		if e.k.Bufs[s.Buf].Half {
			e.fail("store %d: a half from a general register", i)
			return
		}
		e.a.MOVLStore(m, e.reg(s.Val))
		return
	}
	if e.k.Bufs[s.Buf].Half {
		if wide {
			e.a.VMOVDQUStorex(m, e.reg(s.Val))
		} else {
			e.a.VPEXTRWStore(m, e.reg(s.Val), 0)
		}
		return
	}
	if wide {
		e.a.VMOVDQUStore(m, e.reg(s.Val))
		return
	}
	e.a.VMOVSSStore(m, e.reg(s.Val))
}

func (e *x86Lowerer) counter(c int) x86.Reg {
	r := e.k.Loops[c].Reg
	if r == 0 {
		e.fail("loop %d has no counter", c)
	}
	return x86.Reg(max(r-1, 0))
}

func (e *x86Lowerer) enter(c int) {
	k, a := e.k, e.a
	l := k.Loops[c]
	if it := k.iterOf(c); it >= 0 && k.Nodes[it].Reg > 0 {
		r := e.reg(it)
		a.XORQ(r, r)
	} else if it >= 0 && k.Nodes[it].Spill > 0 {
		a.MOVMemImm(spillSlot(k.Nodes[it].Spill), 0)
	}
	for _, cu := range l.Cursors {
		if cu.Take && cu.Ptr > 0 && k.Ptrs[cu.Ptr-1].Spill > 0 {
			a.MOVStore(spillSlot(k.Ptrs[cu.Ptr-1].Spill), e.ptrReg(cu.Buf, 0))
		} else if cu.Take {
			a.MOVQ(e.ptrReg(cu.Buf, cu.Ptr), e.ptrReg(cu.Buf, 0))
		}
	}
	if l.once() {
		return
	}
	ctr := e.counter(c)
	runtime := true
	if l.Count > 0 {
		if n := k.Nodes[l.Count-1]; n.Op == OpConst {
			a.MOVimm(ctr, n.Imm)
			runtime = false
		} else if src := e.reg(l.Count - 1); src != ctr && !e.preset[c] {
			a.MOVQ(ctr, src)
		}
		if t := c + 1; l.Lanes > 0 && t < len(k.Loops) && k.Loops[t].Of == c+1 && k.passes() && k.Loops[t].Count > 0 {
			if tc, src := e.counter(t), e.reg(k.Loops[t].Count-1); tc != ctr && tc != src && k.Nodes[k.Loops[t].Count-1].Op != OpConst {
				a.MOVQ(tc, src)
				e.preset[t] = true
			}
		}
	} else {
		a.MOVimm(ctr, int64(l.Trip))
		runtime = false
	}
	e.end[c] = a.Label()
	if runtime && !l.Nonempty {
		a.TESTQ(ctr, ctr)
		a.JZ(e.end[c])
	}
	e.top[c] = a.Label()
	a.Bind(e.top[c])
}

func (e *x86Lowerer) leave(c int) {
	k, a := e.k, e.a
	l := k.Loops[c]
	for _, cu := range l.Cursors {
		step := int32(cu.Elems * ElemBytes(k.Bufs[cu.Buf]))
		switch {
		case cu.Elems == 0:
		case cu.Ptr > 0 && k.Ptrs[cu.Ptr-1].Spill > 0:
			a.ADDMemImm32(spillSlot(k.Ptrs[cu.Ptr-1].Spill), step)
		default:
			a.ADDimm(e.ptrReg(cu.Buf, cu.Ptr), step)
		}
	}
	// An iteration counts the loop's iterations, Lanes of them a vector
	// and Copies vectors a trip.
	step := int32(max(1, l.Lanes) * max(1, l.Copies))
	if it := k.iterOf(c); it >= 0 && k.Nodes[it].Reg > 0 {
		a.ADDimm(e.reg(it), step)
	} else if it >= 0 && k.Nodes[it].Spill > 0 {
		a.ADDMemImm32(spillSlot(k.Nodes[it].Spill), step)
	}
	if l.once() {
		return
	}
	a.DEC(e.counter(c))
	a.JNZ(e.top[c])
	a.Bind(e.end[c])
}
