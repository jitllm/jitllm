//go:build amd64

package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"
)

// liveAbove asserts that every register live across the inner loop sits above
// the accumulators.
//
// The integer accumulators occupy Y0..Y(n-1) where n is mr*nr, so at the widest
// tile Y10 and Y11 are accumulators. A format may use them only where the
// accumulators are dead (the per-super-block prologue); a mask kept live across
// the loop there corrupts results without crashing. A violation is a
// programming error: move the scratch register up, do not narrow the tile.
func liveAbove(n, mr, nr int, live ...Reg) error {
	for _, r := range live {
		if n > int(r) {
			return fmt.Errorf("jit: EmitGEMM: tile %dx%d has %d accumulators, which collide with live register Y%d -- move the scratch up", mr, nr, n, int(r))
		}
	}
	return nil
}

// kGEMM is the geometry the three row-major k-quant GEMMs share: an mr x nr
// tile of 8*nr tokens over rows of nb super-blocks of bb bytes, its int32
// accumulators in Y0..Y(n-1), and its per-worker Scratch:
//
//	faccOff   the float accumulators, n x 32 bytes, live across the row
//	scaleOff  the per-row unpacked scales, 64 bytes a row
//	stageOff  a staging slot for the byte-to-dword widening
//	wtOff     the super-block's unpacked weights, 256 bytes a row
//	totOff    the wide path's per-super-block totals
//
// Always in memory: the scale prologue needs ten temporaries and the register
// file has to hold the integer accumulators across a whole sub-block. What
// each format unpacks and how it folds stays in its emitter.
type kGEMM struct {
	mr, nr, n, nb int
	rowStr        int32
	aKG, sBlk     int32 // activation bytes per k-group; scale bytes per 32 elements
	outRow        int32
	faccOff       int32
	scaleOff      int32
	stageOff      int32
	wtOff         int32
	totOff        int32
	wide          bool // a 256-wide activation window: d_a is one per super-block
	iacc          []Reg
}

func newKGEMM(mr, nr, nb, window int, bb int32) (*kGEMM, error) {
	if mr < 1 || nr < 1 || mr*nr > 12 || mr > 8 {
		return nil, fmt.Errorf("jit: EmitGEMM: tile %dx%d out of range", mr, nr)
	}
	tok := int32(8 * nr)
	g := &kGEMM{mr: mr, nr: nr, n: mr * nr, nb: nb, rowStr: int32(nb) * bb,
		aKG: tok * 4, sBlk: tok * 4, outRow: tok * 4, wide: window >= 256}
	g.scaleOff = g.faccOff + int32(g.n)*32
	g.stageOff = g.scaleOff + int32(mr)*64
	g.wtOff = g.stageOff + 32
	g.totOff = g.wtOff + int32(mr)*256
	g.iacc = make([]Reg, g.n)
	for i := range g.iacc {
		g.iacc[i] = Reg(i)
	}
	return g, nil
}

// facc and tot are tile element (m, j)'s float accumulator and wide total.
func (g *kGEMM) facc(m, j int) int32 { return g.faccOff + int32(m*g.nr+j)*32 }
func (g *kGEMM) tot(m, j int) int32  { return g.totOff + int32(m*g.nr+j)*32 }

// args loads the pointers: Out, W, A, AScale, the row groups, Scr, Scratch
// into R15, and the activation sums at sumOff into R9 (never R14: Go's
// register ABI pins it to the goroutine), before Scratch when sumFirst.
func (g *kGEMM) args(a *Buf, sumOff int32, sumFirst bool) {
	a.MOVLoad(RCX, At(RDI, 0))  // Out
	a.MOVLoad(RDX, At(RDI, 8))  // W
	a.MOVLoad(RSI, At(RDI, 16)) // A
	a.MOVLoad(R8, At(RDI, 24))  // AScale
	a.MOVLoad(RAX, At(RDI, 32)) // row groups
	a.MOVLoad(RBX, At(RDI, 56)) // Scr: 32-byte constants
	if sumFirst {
		a.MOVLoad(R9, At(RDI, sumOff))
	}
	a.MOVLoad(R15, At(RDI, 72)) // Scratch
	if !sumFirst {
		a.MOVLoad(R9, At(RDI, sumOff))
	}
}

// rowHead opens the row-group loop -- fresh activation cursors, the
// super-block count, the float accumulators zeroed -- and then the
// super-block loop, and returns both labels.
func (g *kGEMM) rowHead(a *Buf) (row, blk int) {
	row = a.Label()
	a.Bind(row)
	a.MOVQ(R12, RSI)
	a.MOVQ(R13, R8)
	a.MOVQ(RDI, R9)
	a.MOVimm(R11, int64(g.nb))
	a.VPXOR(Y0, Y0, Y0)
	for i := 0; i < g.n; i++ {
		a.VMOVDQUStore(At(R15, g.faccOff+int32(i)*32), Y0)
	}
	blk = a.Label()
	a.Bind(blk)
	return row, blk
}

// wideFold is the one d_a multiply of the float wide path, once per (row,
// token group): the super-block's total times d_a, into the accumulator.
func (g *kGEMM) wideFold(a *Buf) {
	for m := 0; m < g.mr; m++ {
		for j := 0; j < g.nr; j++ {
			acc := g.iacc[m*g.nr+j]
			a.VMOVDQULoad(acc, At(R15, g.tot(m, j)))
			a.VMULPSMem(acc, acc, At(R13, int32(j)*32))
			a.VADDPSMem(acc, acc, At(R15, g.facc(m, j)))
			a.VMOVDQUStore(At(R15, g.facc(m, j)), acc)
		}
	}
}

// rowTail stores the row group's tile, steps W and Out one row group and
// closes the row loop and the kernel.
func (g *kGEMM) rowTail(a *Buf, row int) []byte {
	for m := 0; m < g.mr; m++ {
		for j := 0; j < g.nr; j++ {
			a.VMOVDQULoad(Y0, At(R15, g.facc(m, j)))
			a.VMOVDQUStore(At(RCX, int32(m)*g.outRow+int32(j)*32), Y0)
		}
	}
	a.ADDimm(RDX, (int32(g.mr)-1)*g.rowStr)
	a.ADDimm(RCX, int32(g.mr)*g.outRow)
	a.DEC(RAX)
	a.JNZ(row)
	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// emitGEMMQ4K generates the prefill kernel for Q4_K and Q5_K.
//
// A GEMM amortizes the unpack over all 8*nr tokens of the tile, so the formats
// with the most expensive unpack gain the most from batching.
//
// The structure follows Q4_0's: a super-block is 256 elements in eight
// 32-element sub-blocks; sub-block 2j reads the low nibbles of a 32-byte group
// and 2j+1 the high nibbles of the same bytes. What differs is the scale: d and
// dmin per super-block plus eight 6-bit (sc, m) pairs packed across twelve
// bytes, so the value is d*sc[i]*q - dmin*m[i], which is affine.
//
// The offset needs only sum(a) per (sub-block, token): -dmin*m[i]*sum(a).
// Args.ASum carries exactly that, since BiasC(Q4_K) is 1 (the nibbles are
// already unsigned and need no bias correction).
//
// The twelve packed scale bytes are unpacked once per super-block per row into
// Scratch, while the accumulators are dead, so their registers are the
// temporaries.
func emitGEMMQ4K(t quant.Type, mr, nr, nb, window int) ([]byte, error) {
	q5 := t == quant.Q5_K
	bb := int32(t.BlockBytes()) // 144 for Q4_K, 176 for Q5_K
	g, err := newKGEMM(mr, nr, nb, window, bb)
	if err != nil {
		return nil, err
	}
	rowStr, aKG, sBlk, n := g.rowStr, g.aKG, g.sBlk, g.n
	qsBase := int32(16) // d, dmin, scales[12]
	if q5 {
		qsBase = 48 // ... plus qh[32]
	}
	faccOff, scaleOff, stageOff, wtOff := g.faccOff, g.scaleOff, g.stageOff, g.wtOff
	// At window 256 d_a is constant over the super-block, so both terms factor
	// it out: d_a * sum_i(d*sc[i]*dot_i - dmin*m[i]*sumA_i). Q4_K/Q5_K cannot
	// take Q3_K's integer route (a 32-element sub-block reaches 32*15*127 =
	// 60960, past int16), but the hoist still removes loads and a multiply.
	ftotOff, wide := g.totOff, g.wide
	iacc := g.iacc
	const (
		shufA = Y10 // prologue only, where the accumulators are dead
		shufB = Y11 // ... likewise
		vSc   = Y12
		vMin  = Y13
		vAux  = Y14 // sum(a) in the fold; the inner loop no longer needs a mask
		wb    = Y15
		// Weight staging, prologue only.
		sQh = Y1
		sL  = Y2
		sW  = Y3
		sT  = Y4
	)
	if err := liveAbove(n, mr, nr, vSc, vMin, vAux, wb); err != nil {
		return nil, err
	}

	var a Buf
	g.args(&a, 96, true) // ASum
	row, blk := g.rowHead(&a)

	// --- once per super-block: the eight (sc, m) pairs for each row ---
	a.VBROADCASTI128(shufA, At(RBX, 0))
	a.VBROADCASTI128(shufB, At(RBX, 32))
	for m := 0; m < mr; m++ {
		base := int32(m) * rowStr
		a.VBROADCASTI128(Y1, At(RDX, base+4))
		a.VPSHUFB(Y2, Y1, shufA)
		a.VPSHUFB(Y3, Y1, shufB)
		a.VPSRLW(Y4, Y3, 2)
		a.VPANDMem(Y4, Y4, At(RBX, 128)) // 0x30
		a.VPANDMem(Y5, Y2, At(RBX, 64))  // 0x3F
		a.VPANDMem(Y6, Y2, At(RBX, 96))  // 0x0F
		a.VPOR(Y6, Y6, Y4)
		a.VPSRLW(Y7, Y2, 4)
		a.VPANDMem(Y7, Y7, At(RBX, 96))
		a.VPOR(Y7, Y7, Y4)
		a.VPBLENDD(Y5, Y5, Y6, 0x02)
		a.VPBLENDD(Y5, Y5, Y7, 0x08)
		a.VMOVDQUStore(At(R15, stageOff), Y5)
		a.VPMOVZXBD(Y6, At(R15, stageOff))   // sc[0..7]
		a.VPMOVZXBD(Y7, At(R15, stageOff+8)) // m[0..7]
		a.VCVTDQ2PS(Y6, Y6)
		a.VCVTDQ2PS(Y7, Y7)
		a.VCVTPH2PSx(Y8, At(RDX, base))
		a.VBROADCASTSSReg(Y8, Y8) // d
		a.VMULPS(Y6, Y6, Y8)
		a.VCVTPH2PSx(Y9, At(RDX, base+2))
		a.VBROADCASTSSReg(Y9, Y9) // dmin
		a.VMULPS(Y7, Y7, Y9)
		a.VMOVDQUStore(At(R15, scaleOff+int32(m)*64), Y6)
		a.VMOVDQUStore(At(R15, scaleOff+int32(m)*64+32), Y7)
	}

	// The super-block is unpacked once, 32 elements at a time (see Q3_K).
	// Element e = i*32 + g*4 with g in [0,8) takes its nibble from qs byte
	// qsBase + (i/2)*32 + g*4, so the eight k-groups of one sub-block are 32
	// contiguous bytes, and the same bytes serve sub-blocks 2h and 2h+1.
	//
	// qh does not advance: the same 32 high-bit bytes serve all four 64-element
	// groups (the byte is the k-group, the bit the sub-block), so one load
	// covers every g. Folded as in decode: ((qh >> i) & 1) << 4.
	for m := 0; m < mr; m++ {
		base := int32(m) * rowStr
		if q5 {
			a.VMOVDQULoad(sQh, At(RDX, base+16))
		}
		for h := int32(0); h < 4; h++ {
			a.VMOVDQULoad(sL, At(RDX, base+qsBase+h*32))
			for lo := int32(0); lo < 2; lo++ {
				i := h*2 + lo
				if lo == 1 {
					a.VPSRLW(sW, sL, 4)
					a.VPANDMem(sW, sW, At(RBX, 96)) // 0x0F
				} else {
					a.VPANDMem(sW, sL, At(RBX, 96))
				}
				if q5 {
					if i <= 4 {
						a.VPSLLW(sT, sQh, byte(4-i))
					} else {
						a.VPSRLW(sT, sQh, byte(i-4))
					}
					a.VPANDMem(sT, sT, At(RBX, 160)) // 0x10
					a.VPOR(sW, sW, sT)
				}
				a.VMOVDQUStore(At(R15, wtOff+int32(m)*256+i*32), sW)
			}
		}
	}

	for i := int32(0); i < 8; i++ {
		for _, r := range iacc {
			a.VPXOR(r, r, r)
		}
		for g := int32(0); g < 8; g++ {
			for m := 0; m < mr; m++ {
				a.VPBROADCASTD(wb, At(R15, wtOff+int32(m)*256+i*32+g*4))
				for j := 0; j < nr; j++ {
					a.VPDPBUSDMem(iacc[m*nr+j], wb, At(R12, (i*8+g)*aKG+int32(j)*32))
				}
			}
		}
		// Fold, j outer so sum(a) is loaded and widened once per token group.
		for j := 0; j < nr; j++ {
			a.VMOVDQULoad(vAux, At(RDI, i*sBlk+int32(j)*32))
			a.VCVTDQ2PS(vAux, vAux) // sum(a) as float, per token
			for m := 0; m < mr; m++ {
				acc := iacc[m*nr+j]
				a.VBROADCASTSS(vSc, At(R15, scaleOff+int32(m)*64+i*4))
				a.VBROADCASTSS(vMin, At(R15, scaleOff+int32(m)*64+32+i*4))
				a.VCVTDQ2PS(acc, acc)
				if wide {
					a.VMULPS(acc, acc, vSc)    // d*sc
					a.VMULPS(vMin, vMin, vAux) // dmin*m * sum(a)
					a.VSUBPS(acc, acc, vMin)
					if i > 0 { // i == 0 initialises rather than accumulates
						a.VADDPSMem(acc, acc, At(R15, ftotOff+int32(m*nr+j)*32))
					}
					a.VMOVDQUStore(At(R15, ftotOff+int32(m*nr+j)*32), acc)
					continue
				}
				a.VMULPSMem(wb, vSc, At(R13, i*sBlk+int32(j)*32)) // d*sc * d_a
				a.VMULPS(acc, acc, wb)
				a.VMULPS(vMin, vMin, vAux)                           // dmin*m * sum(a)
				a.VMULPSMem(vMin, vMin, At(R13, i*sBlk+int32(j)*32)) // * d_a
				a.VSUBPS(acc, acc, vMin)                             // ... subtracted, not added
				a.VADDPSMem(acc, acc, At(R15, faccOff+int32(m*nr+j)*32))
				a.VMOVDQUStore(At(R15, faccOff+int32(m*nr+j)*32), acc)
			}
		}
	}

	if wide {
		g.wideFold(&a)
	}

	a.ADDimm(RDX, bb)
	a.ADDimm(R12, aKG*64) // 64 k-groups per super-block
	a.ADDimm(R13, sBlk*8)
	a.ADDimm(RDI, sBlk*8)
	a.DEC(R11)
	a.JNZ(blk)
	return g.rowTail(&a, row), nil
}

// emitGEMMQ6K generates the prefill kernel for Q6_K.
//
// Q6_K carries sixteen scales per super-block, one per 16 elements, so the fold
// happens twice as often as Q4_K's and the bias correction is summed over
// sixteen activations (Args.AHalfSum). Folding at 32 would average two weight
// scales and produce plausible, wrong numbers.
//
// Within each 128-element group the quarters are strided: element l+0 takes
// the low nibble of ql[l] with qh bits 0-1, l+32 the low nibble of ql[l+32]
// with bits 2-3, l+64 the high nibble of ql[l] with bits 4-5, and l+96 the
// high nibble of ql[l+32] with bits 6-7. Four consecutive k still land on four
// consecutive bytes, so one broadcast serves a k-group.
//
// The two high bits are shifted and masked in one step: ((qh >> s) & 3) << 4 is
// (qh << (4-s)) & 0x30 for s <= 4 and (qh >> (s-4)) & 0x30 above it.
func emitGEMMQ6K(mr, nr, nb, window int) ([]byte, error) {
	const bb = int32(210) // ql[128] qh[64] scales[16] d
	g, err := newKGEMM(mr, nr, nb, window, bb)
	if err != nil {
		return nil, err
	}
	rowStr, aKG, sBlk, n := g.rowStr, g.aKG, g.sBlk, g.n
	hBlk := sBlk // half-sum stride, per 16 elements
	// scaleOff holds 16 floats a row.
	faccOff, scaleOff, stageOff, wtOff := g.faccOff, g.scaleOff, g.stageOff, g.wtOff
	// At window 256 d_a leaves the sub-block loop. Q6_K cannot take Q3_K's
	// integer route (its bracket reaches 16*32*127 = 65024, past int16), but
	// the same hoist works in float: ftot holds
	// sum_s d*sc[s]*(dot_s - 32*sumA_s) and d_a multiplies it once.
	ftotOff, wide := g.totOff, g.wide
	iacc := g.iacc
	const (
		vM0F = Y12
		vM30 = Y13
		vQh  = Y14
		wb   = Y15
		vSc  = Y14 // fold only, where vQh is dead
		// Staging registers, prologue only -- the accumulators are not zeroed
		// until the sub-block loop, which is why the scale staging above may
		// use Y6 and Y8 for the same reason.
		sQh = Y1
		sL0 = Y2
		sL1 = Y3
		sW  = Y4
		sT  = Y5
	)
	if err := liveAbove(n, mr, nr, vM0F, vM30, vQh, wb); err != nil {
		return nil, err
	}

	var a Buf
	g.args(&a, 104, false) // AHalfSum
	a.VBROADCASTI128(vM0F, At(RBX, 0))
	a.VBROADCASTI128(vM30, At(RBX, 32))
	row, blk := g.rowHead(&a)
	// Sixteen signed int8 scales, widened and scaled by d, once per super-block.
	for m := 0; m < mr; m++ {
		base := int32(m) * rowStr
		a.VCVTPH2PSx(Y8, At(RDX, base+208))
		a.VBROADCASTSSReg(Y8, Y8) // d
		for h := 0; h < 2; h++ {
			a.VPMOVSXBD(Y6, At(RDX, base+192+int32(h)*8))
			a.VCVTDQ2PS(Y6, Y6)
			a.VMULPS(Y6, Y6, Y8)
			a.VMOVDQUStore(At(R15, scaleOff+int32(m)*64+int32(h)*32), Y6)
		}
	}
	_ = stageOff

	// The super-block is unpacked once, 32 elements at a time (see Q3_K). For
	// element e = grp*128 + q*32 + l with l in [0,32):
	//
	//	low nibble   qs byte  grp*64 + (q&1)*32 + l, shifted 4 when q >= 2
	//	high 2 bits  qh byte  128 + grp*32 + l, at bit 2*q
	//
	// so l is contiguous across all three of qs, qh and the staged output, and
	// the two qs loads per grp each serve two quarters.
	for m := 0; m < mr; m++ {
		base := int32(m) * rowStr
		for grp := int32(0); grp < 2; grp++ {
			a.VMOVDQULoad(sQh, At(RDX, base+128+grp*32))
			a.VMOVDQULoad(sL0, At(RDX, base+grp*64))
			a.VMOVDQULoad(sL1, At(RDX, base+grp*64+32))
			for q := int32(0); q < 4; q++ {
				src := sL0
				if q&1 == 1 {
					src = sL1
				}
				if q >= 2 {
					a.VPSRLW(sW, src, 4)
					a.VPAND(sW, sW, vM0F)
				} else {
					a.VPAND(sW, src, vM0F)
				}
				if shift := q * 2; shift <= 4 {
					a.VPSLLW(sT, sQh, byte(4-shift))
				} else {
					a.VPSRLW(sT, sQh, byte(shift-4))
				}
				a.VPAND(sT, sT, vM30)
				a.VPOR(sW, sW, sT) // 0..63, unsigned; the -32 comes off in the fold
				a.VMOVDQUStore(At(R15, wtOff+int32(m)*256+grp*128+q*32), sW)
			}
		}
	}

	for s := int32(0); s < 16; s++ {
		for _, r := range iacc {
			a.VPXOR(r, r, r)
		}
		for gg := int32(0); gg < 4; gg++ { // four k-groups per 16-element scale
			e := s*16 + gg*4
			for m := 0; m < mr; m++ {
				a.VPBROADCASTD(wb, At(R15, wtOff+int32(m)*256+e))
				for j := 0; j < nr; j++ {
					a.VPDPBUSDMem(iacc[m*nr+j], wb, At(R12, (s*4+gg)*aKG+int32(j)*32))
				}
			}
		}
		// m outer, j inner: vSc is a function of (m, s) alone. See Q3_K.
		for m := 0; m < mr; m++ {
			a.VBROADCASTSS(vSc, At(R15, scaleOff+int32(m)*64+s*4))
			for j := 0; j < nr; j++ {
				acc := iacc[m*nr+j]
				// Bias off in int32, exactly: both terms are integers.
				a.VPSUBDMem(acc, acc, At(RDI, s*hBlk+int32(j)*32))
				a.VCVTDQ2PS(acc, acc)
				if wide {
					a.VMULPS(acc, acc, vSc)
					if s > 0 { // s == 0 initialises rather than accumulates
						a.VADDPSMem(acc, acc, At(R15, ftotOff+int32(m*nr+j)*32))
					}
					a.VMOVDQUStore(At(R15, ftotOff+int32(m*nr+j)*32), acc)
					continue
				}
				a.VMULPSMem(wb, vSc, At(R13, (s/2)*sBlk+int32(j)*32))
				a.VMULPS(acc, acc, wb)
				a.VADDPSMem(acc, acc, At(R15, faccOff+int32(m*nr+j)*32))
				a.VMOVDQUStore(At(R15, faccOff+int32(m*nr+j)*32), acc)
			}
		}
	}

	if wide {
		g.wideFold(&a)
	}

	a.ADDimm(RDX, bb)
	a.ADDimm(R12, aKG*64)
	a.ADDimm(R13, sBlk*8)
	a.ADDimm(RDI, hBlk*16)
	a.DEC(R11)
	a.JNZ(blk)
	return g.rowTail(&a, row), nil
}

// emitGEMMQ3K generates the prefill kernel for Q3_K.
//
// Three layout traps each produce fluent wrong text rather than a crash:
//
//  1. The high mask is inverted: a set bit means do not subtract 4. The kernel
//     builds u = v + 4*bit as an unsigned 0..7 and the fold removes a flat 4
//     via BiasC, once per sub-block and exactly in int32.
//
//  2. hmask is not re-based for the second 128-element group: qs advances by
//     32 bytes and hmask does not; the second group uses bits 4..7.
//
//  3. The sixteen 6-bit scales are packed in a third arrangement, different
//     from Q4_K's. That unpack is lifted verbatim from emitQ3KMatVec.
//
// At window 256 the whole fold is integer: d_a leaves the sub-block loop and
//
//	out = d * d_a * SUM_s sc[s] * (dot_s - 4*sumA_s)
//
// with every factor of the sum an integer. The bracket is at most
// 16*4*127 = 8128, so it fits int16 and one VPMADDWD against a dword whose high
// half is zeroed multiplies it by sc in a single uop; the total is at most
// 16*32*8128 = 4.2M. Only Q3_K qualifies: the other k-quants' brackets exceed
// int16 and would need VPMULLD, which is slower than the convert it replaces.
func emitGEMMQ3K(mr, nr, nb, window int) ([]byte, error) {
	const (
		bb     = int32(110) // hmask[32] qs[64] scales[12] d
		cShufA = 0
		cShufB = 32
		c0F    = 64
		c30    = 96
		c03    = 128
		c20    = 192
		c04    = 224
	)
	g, err := newKGEMM(mr, nr, nb, window, bb)
	if err != nil {
		return nil, err
	}
	rowStr, aKG, sBlk, n := g.rowStr, g.aKG, g.sBlk, g.n
	hBlk := sBlk
	// The super-block's 256 weights are unpacked once into wtOff and the inner
	// loop only broadcasts them; the wide path's totals are int32 here.
	faccOff, scaleOff, stageOff, wtOff := g.faccOff, g.scaleOff, g.stageOff, g.wtOff
	itotOff, wide := g.totOff, g.wide
	iacc := g.iacc
	const (
		vShA = Y10 // prologue only, where the accumulators are dead
		vShB = Y11 // ... likewise
		vSc  = Y12
		vHm  = Y13
		v03  = Y14
		wb   = Y15
		// The weight staging runs in the same prologue as the scale staging,
		// where Y1..Y4 are dead: the accumulators are not zeroed until the
		// sub-block loop below.
		sH = Y1 // the row's 32 high-mask bytes, loaded once
		sQ = Y2 // 32 packed qs bytes for one (grp)
		sW = Y3 // the unpacked 3-bit values being assembled
		sT = Y4 // the shifted high bit
	)
	if err := liveAbove(n, mr, nr, vSc, vHm, v03, wb); err != nil {
		return nil, err
	}

	var a Buf
	g.args(&a, 104, false) // AHalfSum
	a.VBROADCASTI128(v03, At(RBX, c03))
	row, blk := g.rowHead(&a)
	a.VBROADCASTI128(vShA, At(RBX, cShufA))
	a.VBROADCASTI128(vShB, At(RBX, cShufB))
	for m := 0; m < mr; m++ {
		base := int32(m) * rowStr
		a.VBROADCASTI128(Y1, At(RDX, base+96))
		a.VPSHUFB(Y2, Y1, vShA)
		a.VPSHUFB(Y3, Y1, vShB)
		a.VPANDMem(Y4, Y2, At(RBX, c0F))
		a.VPSRLW(Y5, Y2, 4)
		a.VPANDMem(Y5, Y5, At(RBX, c0F))
		a.VPBLENDD(Y4, Y4, Y5, 0b11001100)
		a.VPSLLW(Y5, Y3, 4)
		a.VPSLLW(Y6, Y3, 2)
		a.VPSRLW(Y7, Y3, 2)
		a.VPBLENDD(Y5, Y5, Y6, 0b00100010)
		a.VPBLENDD(Y5, Y5, Y3, 0b01000100)
		a.VPBLENDD(Y5, Y5, Y7, 0b10001000)
		a.VPANDMem(Y5, Y5, At(RBX, c30))
		a.VPOR(Y4, Y4, Y5)
		a.VPSUBBMem(Y4, Y4, At(RBX, c20)) // bias by 32 as bytes, so the widen is signed
		a.VMOVDQUStore(At(R15, stageOff), Y4)
		a.VPMOVSXBD(Y6, At(R15, stageOff))
		a.VPMOVSXBD(Y7, At(R15, stageOff+8))
		if wide {
			// Keep sc as an integer, and zero the dword's high half so
			// VPMADDWD sees (sc, 0) and multiplies the low int16 alone. The
			// shift pair is the mask; it needs no constant. d is applied once
			// per super-block in the epilogue instead of folded in here.
			a.VPSLLD(Y6, Y6, 16)
			a.VPSRLD(Y6, Y6, 16)
			a.VPSLLD(Y7, Y7, 16)
			a.VPSRLD(Y7, Y7, 16)
		} else {
			a.VCVTDQ2PS(Y6, Y6)
			a.VCVTDQ2PS(Y7, Y7)
			a.VCVTPH2PSx(Y8, At(RDX, base+108))
			a.VBROADCASTSSReg(Y8, Y8)
			a.VMULPS(Y6, Y6, Y8)
			a.VMULPS(Y7, Y7, Y8)
		}
		a.VMOVDQUStore(At(R15, scaleOff+int32(m)*64), Y6)
		a.VMOVDQUStore(At(R15, scaleOff+int32(m)*64+32), Y7)
	}

	// Unpack the whole super-block once, 32 elements at a time, rather than four
	// at a time inside the inner loop (which was bound by unpack, not by VNNI).
	//
	// The 32 elements of one (grp, j) are contiguous in all three places. For
	// element e = grp*128 + j*32 + b with b in [0,32):
	//
	//	qs byte      32 + grp*32 + b     -> 32 contiguous bytes
	//	hmask byte   b                   -> 32 contiguous bytes, and the same
	//	                                    32 for every j, so it loads once
	//	high bit     grp*4 + j           -> one bit, uniform across the group
	//
	// so one 256-bit load, shift, mask, shift, mask, or and store covers all 32.
	// The staged bytes are indexed by element, so the inner loop's four
	// consecutive elements are four consecutive scratch bytes and a single
	// VPBROADCASTD reads them.
	for m := 0; m < mr; m++ {
		base := int32(m) * rowStr
		a.VMOVDQULoad(sH, At(RDX, base)) // hmask[32], once for all eight groups
		for grp := int32(0); grp < 2; grp++ {
			a.VMOVDQULoad(sQ, At(RDX, base+32+grp*32))
			for j := int32(0); j < 4; j++ {
				if sh := byte(2 * j); sh != 0 {
					a.VPSRLW(sW, sQ, sh)
					a.VPAND(sW, sW, v03)
				} else {
					a.VPAND(sW, sQ, v03)
				}
				// ((hm >> bit) & 1) << 2, as one shift and one mask.
				if hbit := grp*4 + j; hbit <= 2 {
					a.VPSLLW(sT, sH, byte(2-hbit))
				} else {
					a.VPSRLW(sT, sH, byte(hbit-2))
				}
				a.VPANDMem(sT, sT, At(RBX, c04))
				a.VPOR(sW, sW, sT) // u = v + 4*bit, unsigned 0..7
				a.VMOVDQUStore(At(R15, wtOff+int32(m)*256+grp*128+j*32), sW)
			}
		}
	}

	for s := int32(0); s < 16; s++ {
		for _, r := range iacc {
			a.VPXOR(r, r, r)
		}
		for gg := int32(0); gg < 4; gg++ {
			e := s*16 + gg*4
			for m := 0; m < mr; m++ {
				a.VPBROADCASTD(wb, At(R15, wtOff+int32(m)*256+e))
				for jj := 0; jj < nr; jj++ {
					a.VPDPBUSDMem(iacc[m*nr+jj], wb, At(R12, (s*4+gg)*aKG+int32(jj)*32))
				}
			}
		}
		// m outer, jj inner, because vSc = d*sc[s] depends on (m, s) alone and
		// is broadcast once per m rather than per (m, jj).
		for m := 0; m < mr; m++ {
			a.VBROADCASTSS(vSc, At(R15, scaleOff+int32(m)*64+s*4))
			for jj := 0; jj < nr; jj++ {
				acc := iacc[m*nr+jj]
				a.VPSUBDMem(acc, acc, At(RDI, s*hBlk+int32(jj)*32))
				if wide {
					a.VPMADDWD(acc, acc, vSc)
					if s > 0 { // s == 0 initialises rather than accumulates
						a.VPADDDMem(acc, acc, At(R15, itotOff+int32(m*nr+jj)*32))
					}
					a.VMOVDQUStore(At(R15, itotOff+int32(m*nr+jj)*32), acc)
					continue
				}
				a.VCVTDQ2PS(acc, acc)
				a.VMULPSMem(wb, vSc, At(R13, (s/2)*sBlk+int32(jj)*32))
				a.VMULPS(acc, acc, wb)
				a.VADDPSMem(acc, acc, At(R15, faccOff+int32(m*nr+jj)*32))
				a.VMOVDQUStore(At(R15, faccOff+int32(m*nr+jj)*32), acc)
			}
		}
	}

	// The one float pass of the wide path: d per row, d_a per token group,
	// both constant over the super-block, applied to a total that is exact.
	if wide {
		for m := 0; m < mr; m++ {
			a.VCVTPH2PSx(vSc, At(RDX, int32(m)*rowStr+108))
			a.VBROADCASTSSReg(vSc, vSc)
			for jj := 0; jj < nr; jj++ {
				acc := iacc[m*nr+jj]
				a.VMOVDQULoad(acc, At(R15, itotOff+int32(m*nr+jj)*32))
				a.VCVTDQ2PS(acc, acc)
				a.VMULPS(acc, acc, vSc)
				a.VMULPSMem(acc, acc, At(R13, int32(jj)*32))
				a.VADDPSMem(acc, acc, At(R15, faccOff+int32(m*nr+jj)*32))
				a.VMOVDQUStore(At(R15, faccOff+int32(m*nr+jj)*32), acc)
			}
		}
	}

	a.ADDimm(RDX, bb)
	a.ADDimm(R12, aKG*64)
	a.ADDimm(R13, sBlk*8)
	a.ADDimm(RDI, hBlk*16)
	a.DEC(R11)
	a.JNZ(blk)

	return g.rowTail(&a, row), nil
}
