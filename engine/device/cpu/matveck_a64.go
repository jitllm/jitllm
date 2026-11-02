package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"
)

// emitA64KMatVec is the k-quant decode matvec: out[r] = dot(row r, activations).
//
// It is the GEMM at mr=1, nr=1: emitA64KScales is shared with emitGEMMA64K.
// Decode has a single activation stream, so it uses plain SDOT (sixteen bytes
// against sixteen) rather than the by-element form.
//
// The min term (Q4_K/Q5_K: w = d*sc*q - dmin*m) needs sum(a) per 32-element
// block, which comes out of the existing AScale pair: QuantizeQ8 stores
// -sum(q)*biasC/8 and BiasC(Q4_K) is 1, so sum(a) = -8*pairs[2b+1].
//
// AHalf is never read: SDOT is signed x signed, so Q6_K and Q3_K subtract their
// zero point from the weight bytes and need no correction array.
//
// The probes are Config fields, measurement scaffolding; 0 disables.
func scalarProbe() int { return cfg.ScalarProbe }
func loadProbe() int   { return cfg.LoadProbe }
func vectorProbe() int { return cfg.VectorProbe }

func emitA64KMatVec(t quant.Type, actWin int) ([]byte, error) {
	var (
		bb, qsBase, scBase, dOff int32
		sub                      int32
		zero                     byte
		hasMin                   bool
	)
	switch t {
	case quant.Q4_K:
		bb, qsBase, scBase, dOff, sub, hasMin = 144, 16, 4, 0, 32, true
	case quant.Q5_K:
		bb, qsBase, scBase, dOff, sub, hasMin = 176, 48, 4, 0, 32, true
	case quant.Q6_K:
		bb, qsBase, scBase, dOff, sub, zero = 210, 0, 192, 208, 16, 32
	case quant.Q3_K:
		bb, qsBase, scBase, dOff, sub, zero = 110, 32, 96, 108, 16, 4
	default:
		return nil, fmt.Errorf("jit: emitA64KMatVec: %s is not a k-quant", t)
	}
	nsub := int32(256) / sub // scales per super-block: 8 or 16
	wregs := int(sub / 16)   // 16-byte weight registers per sub-block
	perBlk := int32(32)      // activation elements per AScale pair
	aStride := int32(256)    // activation bytes per super-block
	pStride := aStride / perBlk * 8

	// Four accumulator chains, because a single FMLA chain across a row is
	// latency-bound and the core has no memory stall to hide it behind.
	// v26-v28 are otherwise unused here.
	//
	// When the whole super-block shares one activation scale (nn.NewJIT widens
	// the amax to 256 for an all-k-quant model), d_a is loaded once per
	// super-block, the sub-blocks accumulate unscaled, and one FMLA at the end
	// applies it. Same pair array and stride, only a wider amax.
	shared := actWin >= 256
	accs := 4
	faccs := [4]VReg{0, 26, 27, 28}
	// dax holds d_a for the super-block; taccs accumulate before it is applied.
	dax := VReg(28)
	taccs := [2]VReg{26, 27}
	if shared {
		// facc's chain is one FMLA per super-block, so it needs no
		// splitting and the registers go to the unscaled accumulators.
		accs = 1
	}
	const (
		facc  = VReg(0)
		iacc  = VReg(1)
		wreg0 = VReg(2) // v2..v3
		areg0 = VReg(4) // v4..v5
		eight = VReg(6)
		f0    = VReg(7)
	)
	// v8-v15 are free (Go's arm64 ABI saves no V register); v8 and v9 are a
	// widened SDOT's scratch (sdotemu.go) and the rest unused. Ten temporaries for the scale unpack, which is the
	// widest thing this kernel does.
	tmps := []VReg{16, 17, 18, 19, 20, 21, 22, 23, 24, 25}
	u1 := tmps[1]
	m0, m1, m2 := tmps[3], tmps[4], tmps[5]

	// For the 16-sub-block formats (Q3_K, Q6_K) four spare vectors hold the
	// sixteen int32 scales, so they are never stored and reloaded. The
	// destinations must come from outside emitA64KScales' tmp range (it uses
	// tmp[4] as widen16's source, so writing a quarter there corrupts the
	// next): wreg0+1 and areg0+1 are free because wregs is 1, and eight and f0
	// because there is no min term.
	var scDst []VReg
	if shared && nsub == 16 && wregs == 1 && !hasMin {
		scDst = []VReg{wreg0 + 1, areg0 + 1, eight, f0}
	}

	// The mask constants are built once for the whole kernel: every value the
	// unpack asks for is loop-invariant. v29-v31 are caller-saved and hold at
	// most three distinct values; no format asks for more.
	konsts := map[byte]VReg{}
	pool := []VReg{29, 30, 31}
	var pre []byte // the values to materialise, in first-asked order
	probe := func(imm byte) VReg {
		if r, ok := konsts[imm]; ok {
			return r
		}
		if len(konsts) >= len(pool) {
			return pool[0] // over budget: caller falls back (no format does)
		}
		r := pool[len(konsts)]
		konsts[imm] = r
		pre = append(pre, imm)
		return r
	}

	var a A64
	// v8 and v9: see the temporaries above.
	a.DotScratch(V8, V9)
	a.LDRx(X1, X0, 0)   // Out
	a.LDRx(X2, X0, 8)   // W
	a.LDRx(X3, X0, 16)  // A
	a.LDRx(X4, X0, 24)  // AScale
	a.LDRx(X5, X0, 32)  // Rows
	a.LDRx(X8, X0, 48)  // RowStr
	a.LDRx(X6, X0, 56)  // Scr, the unpack tables
	a.LDRx(X19, X0, 72) // Scratch, where the per-super-block scales land

	// Two hoists at different scopes: constants never change, so they are built
	// once before every loop; loads change per super-block, so they are issued
	// at the top of each block and reused across its sub-blocks. Which of each
	// the unpack wants is found by a throwaway pass over the same emitter calls,
	// which keeps the lists in step with the unpack.
	loadOff := []int32{}
	seenOff := map[int32]bool{}
	{
		var scan A64
		scanLd := func(off int32) VReg {
			if !seenOff[off] {
				seenOff[off] = true
				loadOff = append(loadOff, off)
			}
			return wreg0
		}
		for s := int32(0); s < nsub; s++ {
			scan.emitA64KDecodeUnpack(t, qsBase, s, zero, wreg0, u1, scanLd, probe)
		}
	}
	// Registers for the hoisted loads. tmps[3..5] are not free: m0, m1 and m2
	// are reloaded inside the sub-block loop for the float scales.
	ldPool := []VReg{16, 18, 22, 23, 24, 25}
	if len(loadOff) > len(ldPool) {
		loadOff = loadOff[:len(ldPool)] // the rest fall back to re-loading
	}
	ldReg := map[int32]VReg{}
	for i, off := range loadOff {
		ldReg[off] = ldPool[i]
	}
	for _, imm := range pre {
		a.MOVI16b(konsts[imm], imm)
	}

	// 8.0f, built once through Scratch: this emitter has no FMOV-immediate, so
	// it is written as an integer and read back as a float.
	a.MOVimm(X23, 0x41000000)
	a.STRx(X23, X19, nsub*8)
	a.LDRs(eight, X19, nsub*8)

	row := a.Label()
	a.Bind(row)
	a.MOVreg(X7, X2)  // weight cursor for this row
	a.MOVreg(X9, X3)  // activation cursor
	a.MOVreg(X10, X4) // AScale cursor
	a.LDRx(X22, X0, 40)
	for i := 0; i < accs; i++ {
		a.MOVIzero(faccs[i])
	}

	blk := a.Label()
	a.Bind(blk)
	a.emitA64KScales(t, X7, X19, 0, nsub, scBase, dOff, X6, tmps, shared, scDst)
	if shared {
		// One d_a for the whole super-block, and two accumulators that do not
		// have it applied yet.
		a.LDRs(dax, X10, 0)
		a.MOVIzero(taccs[0])
		a.MOVIzero(taccs[1])
	}
	for _, off := range loadOff {
		a.LDURq(ldReg[off], X7, off)
	}
	load := func(off int32) VReg {
		if r, ok := ldReg[off]; ok {
			return r
		}
		a.LDURq(u1, X7, off) // over budget: re-load, as before
		return u1
	}

	// In the shared path the sub-block tail is one integer instruction: dot and
	// scale are both integers, so Sum_s sc[s]*dot_s is exact in int32 and d and
	// d_a apply once at the end. SCVTF + LDR S + FMLA becomes one
	// `MLA vd.4s, vn.4s, vm.s[i]`.
	//
	// Two integer chains where there is no min term, one where there is: Q4_K
	// and Q5_K need a float accumulator for -dmin*m[s]*sum(a), and taccs[1] is
	// the only register left for it.
	intAcc := shared
	minAcc := taccs[1]
	for s := int32(0); s < nsub; s++ {
		acc := faccs[s%int32(accs)]
		if shared {
			acc = taccs[s%2]
		}
		if intAcc {
			acc = taccs[0]
			if !hasMin {
				acc = taccs[s%2]
			}
			if scDst == nil && s%4 == 0 {
				a.LDRq(f0, X19, s*4) // four int32 scales
			}
		}
		a.MOVIzero(iacc)
		a.emitA64KDecodeUnpack(t, qsBase, s, zero, wreg0, u1, load, probe)
		// Two adjacent activation quads are one LDP (a single uop, unlike LD1
		// multi-register, which is microcoded on Apple silicon).
		if wregs == 2 {
			a.LDPq(areg0, areg0+1, X9, s*sub)
		}
		for w := 0; w < wregs; w++ {
			if wregs != 2 {
				a.LDRq(areg0+VReg(w), X9, s*sub+int32(w)*16)
			}
			a.SDOT(iacc, areg0+VReg(w), wreg0+VReg(w))
		}
		if intAcc {
			sc := f0
			if scDst != nil {
				sc = scDst[s/4]
			}
			a.MLAelem(acc, iacc, sc, uint8(s%4))
			if hasMin {
				acc = minAcc
			}
		} else {
			a.SCVTF4s(iacc, iacc)
			a.LDRs(f0, X19, s*4) // d * sc[s]
			if shared {
				// d_a is common to the super-block and applied once, below.
				a.FMLAelem(acc, iacc, f0, 0)
			} else {
				a.LDRs(m0, X10, (s*sub/perBlk)*8) // d_a for the covering block
				a.FMULs(f0, f0, m0)
				a.FMLAelem(acc, iacc, f0, 0)
			}
		}
		if hasMin {
			// facc += 8 * dmin*m[s] * d_a * pairs[2b+1], which is
			// -dmin*m[s] * d_a * sum(a) with sum(a) = -8*pairs[2b+1].
			a.LDRs(m1, X19, nsub*4+s*4)         // dmin * m[s]
			a.LDRs(m2, X10, (s*sub/perBlk)*8+4) // -sum(a)/8
			a.FMULs(m1, m1, m2)
			// When the scale is shared it is applied once, at the end, to the
			// whole accumulator including this term; multiplying it in here as
			// well would square d_a on the min correction. The sums stay
			// per-32; only the scale is common.
			if !shared {
				a.FMULs(m1, m1, m0)
			}
			// m1's upper lanes are zero -- scalar FP writes lane 0 and clears
			// the rest -- so this adds to lane 0 alone.
			a.FMLAelem(acc, m1, eight, 0)
		}
	}

	switch {
	case intAcc:
		// One conversion and one scale for the whole super-block: d comes off
		// the block header here rather than through scratch, because intSc left
		// it unapplied and two instructions is cheaper than a store and a load.
		if !hasMin {
			a.ADD4s(taccs[0], taccs[0], taccs[1])
		}
		a.SCVTF4s(taccs[0], taccs[0])
		a.LDRh(m0, X7, dOff)
		a.FCVTsh(m0, m0)
		a.FMULs(m0, m0, dax) // d * d_a
		a.FMLAelem(faccs[0], taccs[0], m0, 0)
		if hasMin {
			// The min term is already float and already carries dmin; only the
			// activation scale is outstanding.
			a.FMLAelem(faccs[0], minAcc, dax, 0)
		}
	case shared:
		// Apply the super-block's activation scale once, to both accumulators.
		a.FMLAelem(faccs[0], taccs[0], dax, 0)
		a.FMLAelem(faccs[0], taccs[1], dax, 0)
	}
	// Probe (Config.ScalarProbe): N dummy scalar adds per block, to price
	// spare integer issue capacity against the NEON pipes. Three registers
	// round-robin so this measures throughput, not a dependency chain. X11-X13
	// are caller-saved and dead here.
	if n := scalarProbe(); n > 0 {
		for i := 0; i < n; i++ {
			a.ADDimm(XReg(11+i%3), XReg(11+i%3), 1)
		}
	}
	// The load probe: N dummy 128-bit loads per block from the constant block
	// (X6, always valid and L1-hot), pricing the load/store units. v29-v31 are
	// outside every allocation above, so this and the vector probe below
	// clobber nothing.
	if n := loadProbe(); n > 0 {
		for i := 0; i < n; i++ {
			a.LDRq(VReg(29+i%3), X6, 0)
		}
	}
	if n := vectorProbe(); n > 0 {
		for i := 0; i < n; i++ {
			v := VReg(29 + i%3)
			a.ADD4s(v, v, v)
		}
	}
	a.ADDimm(X7, X7, bb)
	a.addBig(X9, aStride, X23)
	a.addBig(X10, pStride, X23)
	a.SUBimm(X22, X22, 1)
	a.CBNZ(X22, blk)

	// Merge the chains as a tree, not a chain (a chain would reintroduce the
	// dependency on the row's critical path), then the horizontal sum: two
	// pairwise adds (see FADDPs).
	if accs == 4 {
		a.FADD4s(faccs[0], faccs[0], faccs[1])
		a.FADD4s(faccs[2], faccs[2], faccs[3])
		a.FADD4s(faccs[0], faccs[0], faccs[2])
	}
	a.FADDP4s(f0, facc, facc)
	a.FADDPs(f0, f0)
	a.STRs(f0, X1, 0)

	a.ADDreg(X2, X2, X8)
	a.ADDimm(X1, X1, 4)
	a.SUBimm(X5, X5, 1)
	a.CBNZ(X5, row)

	a.RET()
	return a.Bytes(), nil
}
