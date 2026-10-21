package cpu

import (
	"fmt"
	"sync/atomic"
)

// The arm64 kernels without FEAT_DotProd. SDOT is ARMv8.2-A; a Cortex-A53,
// A57, A72 or A73 (a Raspberry Pi 3 or 4) has only ARMv8.0 NEON. RULE 8 makes
// a missing instruction a generated kernel, never a refusal, so on such a chip
// every SDOT and SDOTelem an emitter asks for is widened in place:
//
//	smull   t0.8h, vn.8b,  vm.8b      the low eight int8 x int8 products
//	smull2  t1.8h, vn.16b, vm.16b     the high eight
//	addp    t0.8h, t0.8h, t1.8h       adjacent pairs: lane k is bytes 2k, 2k+1
//	sadalp  vd.4s, t0.8h              pairs of pairs widened into vd: lane i
//	                                  gains bytes 4i..4i+3
//
// The pair sum in int16 is exact because one operand of every dot in this
// package is a quantized activation or query, |a| <= 127 (amax/127: packact*.go,
// gemmpack.go, kvfmt.go): two products are at most 2*128*127 = 32512, inside
// int16. Every later sum is taken at int32, so the four instructions compute
// SDOT's four lanes exactly: a widened kernel gives the SDOT kernel's bits, and
// the gates hold it to that. The by-element form first broadcasts vm.s[idx]
// into t1 (DUP), which SMULL2 may then overwrite since it reads before it
// writes.
//
// The decode kernels do better than one widening per SDOT (DotChain): the dots
// a kernel issues back to back into one accumulator share one ADDP and one
// SADALP, the later links taken by SMLAL/SMLAL2 into the same int16 pair, and
// the emitter hoists the broadcast out of its row-group loop. A chain's length
// is bounded by the weight's magnitude (DotChainCap).
//
// The activation quantization and the packed layouts are unchanged: the
// widening is below the emitters, which only name two scratch vectors
// (A64.DotScratch) and, for a chain, the registers its broadcasts live in.

// dotEmulated counts the SDOTs widened since the process started, so a gate can
// assert the no-dotprod path was actually emitted and not merely requested.
var dotEmulated atomic.Int64

// DotEmulated reports how many SDOT/SDOTelem instructions have been widened to
// the ARMv8.0 sequence in this process.
func DotEmulated() int64 { return dotEmulated.Load() }

// DotScratch names the two vectors an emulated SDOT may clobber and decides,
// from the probe, whether this kernel's SDOTs are widened. An emitter calls it
// once, before its first SDOT, with registers it never holds live across one;
// on a chip with FEAT_DotProd they go unused.
func (a *A64) DotScratch(t0, t1 VReg) {
	if t0 == t1 {
		panic(fmt.Sprintf("jit: DotScratch: v%d twice", t0))
	}
	a.dotT = [2]VReg{t0, t1}
	a.dotEmu = a64EmulateDot()
}

func (a *A64) sdotWide(vd, vn, vm VReg, elem bool, idx uint8) {
	if a.dotLinks != 0 {
		panic(fmt.Sprintf("jit: SDOT into v%d inside an open DotChain on v%d", vd, a.dotPend))
	}
	t0, t1 := a.dotT[0], a.dotT[1]
	for _, r := range []VReg{vd, vn, vm} {
		if r == t0 || r == t1 {
			panic(fmt.Sprintf("jit: emulated SDOT v%d, v%d, v%d overlaps its scratch v%d, v%d", vd, vn, vm, t0, t1))
		}
	}
	dotEmulated.Add(1)
	if elem {
		a.DUPs4lane(t1, vm, idx)
		vm = t1
	}
	a.SMULL8b(t0, vn, vm)
	a.SMULL16b(t1, vn, vm)
	a.ADDP8h(t0, t0, t1)
	a.SADALP8h(vd, t0)
}

// DotEmulating reports whether this kernel's SDOTs are widened (DotScratch
// decided it), so an emitter can choose the chained form and its broadcasts.
func (a *A64) DotEmulating() bool { return a.dotEmu }

// DotChainCap is the longest DotChain whose int16 sums stay exact when every
// weight byte is at most wmax in magnitude and every activation at most 127:
// a lane holds one product per link and ADDP adds two lanes, so
// 2*links*wmax*127 must not pass 32767. Any int8 weight fits one link
// (2*128*127 = 32512); a nibble (15) fits eight.
func DotChainCap(wmax int) int {
	if wmax < 1 {
		wmax = 1
	}
	n := 32767 / (2 * 127 * wmax)
	if n < 1 {
		panic(fmt.Sprintf("jit: DotChainCap: |w| <= %d does not fit one link", wmax))
	}
	return n
}

// DotChain is vd.4s += SDOT(vn.16b, vm.16b), one link of a chain into vd. With
// FEAT_DotProd it is that SDOT. Without it the first link multiplies into the
// scratch pair (SMULL, SMULL2), each later one accumulates there (SMLAL,
// SMLAL2), and the link marked last folds the pair into vd (ADDP, SADALP). The
// scratch is held between links, so nothing else may write it and no other
// SDOT may be issued while a chain is open; the caller bounds the chain with
// DotChainCap. vm is a whole vector: the caller broadcasts an element once for
// every chain that reads it.
func (a *A64) DotChain(vd, vn, vm VReg, last bool) {
	if !a.dotEmu {
		a.SDOT(vd, vn, vm)
		return
	}
	t0, t1 := a.dotT[0], a.dotT[1]
	for _, r := range []VReg{vd, vn, vm} {
		if r == t0 || r == t1 {
			panic(fmt.Sprintf("jit: DotChain v%d, v%d, v%d overlaps its scratch v%d, v%d", vd, vn, vm, t0, t1))
		}
	}
	if a.dotLinks != 0 && a.dotPend != vd {
		panic(fmt.Sprintf("jit: DotChain into v%d while v%d's is open", vd, a.dotPend))
	}
	dotEmulated.Add(1)
	if a.dotLinks == 0 {
		a.SMULL8b(t0, vn, vm)
		a.SMULL16b(t1, vn, vm)
	} else {
		a.SMLAL8b(t0, vn, vm)
		a.SMLAL16b(t1, vn, vm)
	}
	a.dotPend, a.dotLinks = vd, a.dotLinks+1
	if last {
		a.ADDP8h(t0, t0, t1)
		a.SADALP8h(vd, t0)
		a.dotLinks = 0
	}
}

// SMULL8b is smull vd.8h, vn.8b, vm.8b: the low eight signed byte products.
func (a *A64) SMULL8b(vd, vn, vm VReg) {
	a.w(0x0E20C000 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// SMULL16b is smull2 vd.8h, vn.16b, vm.16b: the high eight.
func (a *A64) SMULL16b(vd, vn, vm VReg) {
	a.w(0x4E20C000 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// SMLAL8b is smlal vd.8h, vn.8b, vm.8b: the low eight products accumulated.
func (a *A64) SMLAL8b(vd, vn, vm VReg) {
	a.w(0x0E208000 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// SMLAL16b is smlal2 vd.8h, vn.16b, vm.16b: the high eight accumulated.
func (a *A64) SMLAL16b(vd, vn, vm VReg) {
	a.w(0x4E208000 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// SADDLP8h is saddlp vd.4s, vn.8h: adjacent int16 pairs summed into int32.
func (a *A64) SADDLP8h(vd, vn VReg) {
	a.w(0x4E602800 | uint32(vn)<<5 | uint32(vd))
}

// SADALP8h is sadalp vd.4s, vn.8h: adjacent int16 pairs summed and added into
// vd's int32 lanes.
func (a *A64) SADALP8h(vd, vn VReg) {
	a.w(0x4E606800 | uint32(vn)<<5 | uint32(vd))
}

// ADDP8h is addp vd.8h, vn.8h, vm.8h: int16 pairwise add, vn's pairs in the
// low four lanes and vm's in the high four.
func (a *A64) ADDP8h(vd, vn, vm VReg) {
	a.w(0x4E60BC00 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}

// ADDP4s is addp vd.4s, vn.4s, vm.4s: integer pairwise add, vn's pairs in the
// low two lanes and vm's in the high two.
func (a *A64) ADDP4s(vd, vn, vm VReg) {
	a.w(0x4EA0BC00 | uint32(vm)<<16 | uint32(vn)<<5 | uint32(vd))
}
