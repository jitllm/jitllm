//go:build amd64

package cpu

// EmitSoftmax generates an in-place softmax over a row whose length is a
// runtime value: a score row's length is pos+1 and changes every token, so
// Args.K carries whole vectors of eight and Args.Rows the single elements
// after them. The tail of each pass is written out: the maximum broadcasts its
// one element, and the exp pass clears lanes 1..7 before they reach the sum.
//
// Three passes, like nn.Softmax: the online form is the same speed on unsorted
// rows and far worse on ascending ones.
//
// Registers: RCX row, RBX consts, R9/RDX the two counts, R10/R8 their
// working copies, R11 cursor. Y7..Y15 are exp's
// constants; Y0..Y6 are free.
func EmitSoftmax() []byte { return emitSoftmax(false) }

// EmitLogSoftmax is EmitSoftmax's log form, in place over a row of runtime
// length with the same Args: x[i] - max - ln(sum exp(x - max)). Pass 2 sums
// without storing, the log of the sum is taken once in vector registers
// (emitLnSum), and pass 3 subtracts max + ln(sum) from the row.
func EmitLogSoftmax() []byte { return emitSoftmax(true) }

func emitSoftmax(log bool) []byte {
	var a Buf
	a.MOVLoad(RCX, At(RDI, 0))  // Out -> the row, read and written in place
	a.MOVLoad(RBX, At(RDI, 56)) // Scr -> expConsts
	a.MOVLoad(R9, At(RDI, 40))  // K -> whole vectors
	a.MOVLoad(RDX, At(RDI, 32)) // Rows -> tail elements
	loadExpConsts(&a)
	pass := func(vec, one func()) {
		a.MOVQ(R11, RCX)
		a.MOVQ(R10, R9)
		a.MOVQ(R8, RDX)
		elemLoops(&a, R10, R8, []Reg{R11}, vec, one)
	}

	// ---- pass 1: the maximum ----
	//
	// Seeded from element 0, never from a constant: seeding from exp's low
	// clamp assumes every score is above -87.3, which qwen2's are not.
	a.VBROADCASTSS(Y3, At(RCX, 0))
	pass(func() {
		a.VMOVDQULoad(Y0, At(R11, 0))
		a.VMAXPS(Y3, Y3, Y0)
	}, func() {
		a.VBROADCASTSS(Y0, At(R11, 0))
		a.VMAXPS(Y3, Y3, Y0)
	})
	// Lane reduction to a broadcast maximum. Three folds: 256 -> 128 with
	// VEXTRACTF128, then 64 and 32 bits within the low lane with VSHUFPS (not
	// VPERMQ, which is cross-lane and would repeat the first fold).
	//
	// Softmax is shift-invariant, so a wrong maximum only shows when the shift
	// pushes a real score past exp's +88 clamp and saturated entries tie;
	// TestEmitSoftmaxMaxIsTheRealMax is the gate that can see it.
	a.VEXTRACTF128(Y1, Y3, 1)
	a.VMAXPS(Y3, Y3, Y1)
	a.VSHUFPS(Y1, Y3, Y3, 0x4E)
	a.VMAXPS(Y3, Y3, Y1)
	a.VSHUFPS(Y1, Y3, Y3, 0xB1)
	a.VMAXPS(Y3, Y3, Y1)
	a.VBROADCASTSSReg(Y3, Y3)

	// ---- pass 2: exp(x - max), summed ----
	a.VPXOR(Y4, Y4, Y4)
	a.VPXOR(Y6, Y6, Y6)
	pass(func() {
		a.VMOVDQULoad(Y0, At(R11, 0))
		a.VSUBPS(Y0, Y0, Y3)
		emitExpPS(&a, Y0, Y1, Y2)
		if !log {
			a.VMOVDQUStore(At(R11, 0), Y0)
		}
		a.VADDPS(Y4, Y4, Y0)
	}, func() {
		a.VMOVSSLoad(Y0, At(R11, 0))
		a.VSUBPS(Y0, Y0, Y3)
		emitExpPS(&a, Y0, Y1, Y2)
		a.VPBLENDD(Y0, Y6, Y0, 0x01) // lanes 1..7 held exp(-max): clear them
		if !log {
			a.VMOVSSStore(At(R11, 0), Y0)
		}
		a.VADDPS(Y4, Y4, Y0)
	})

	// ---- pass 3: divide by the sum ----
	a.VEXTRACTF128(Y1, Y4, 1)
	a.VADDPSx(Y4, Y4, Y1)
	a.VHADDPSx(Y4, Y4, Y4)
	a.VHADDPSx(Y4, Y4, Y4)
	if log {
		// ---- or subtract max + ln(sum) ----
		a.VBROADCASTSSReg(Y4, Y4)
		emitLnSum(&a)
		pass(func() {
			a.VMOVDQULoad(Y0, At(R11, 0))
			a.VSUBPS(Y0, Y0, Y3)
			a.VMOVDQUStore(At(R11, 0), Y0)
		}, func() {
			a.VMOVSSLoad(Y0, At(R11, 0))
			a.VSUBPS(Y0, Y0, Y3)
			a.VMOVSSStore(At(R11, 0), Y0)
		})
		a.VZEROUPPER()
		a.RET()
		return a.Bytes()
	}
	a.VBROADCASTSS(Y5, At(RBX, 28)) // 1.0
	a.VDIVPS(Y4, Y5, Y4)
	a.VBROADCASTSSReg(Y4, Y4)
	pass(func() {
		a.VMOVDQULoad(Y0, At(R11, 0))
		a.VMULPS(Y0, Y0, Y4)
		a.VMOVDQUStore(At(R11, 0), Y0)
	}, func() {
		a.VMOVSSLoad(Y0, At(R11, 0))
		a.VMULPS(Y0, Y0, Y4)
		a.VMOVSSStore(At(R11, 0), Y0)
	})

	a.VZEROUPPER()
	a.RET()
	return a.Bytes()
}

// elemLanes: a YMM is 256 bits.
const elemLanes = 8

// emitLnSum adds ln(Y4) to Y3, every lane: Y4 is the softmax's sum, at least 1
// (the maximum's own term) and finite. With sum = m * 2^e, m in [1,2),
// ln(sum) = e*ln2 + ln(m), and ln(m) = log1p(u) on u = m-1 in [0,1) is the
// atanh series sqrt-softplus runs (actConsts 80..96): s = u/(2+u) in [0,1/3],
// five terms, ~1e-6 absolute. e and m come off the bits; the sum is positive,
// so the shift needs no sign. Clobbers Y0..Y2, Y5, Y6.
func emitLnSum(a *Buf) {
	a.VPSRLD(Y0, Y4, 23)
	a.VBROADCASTSS(Y1, At(RBX, 36)) // the exponent bias, as bits
	a.VPSUBD(Y0, Y0, Y1)
	a.VCVTDQ2PS(Y0, Y0)              // e
	a.VBROADCASTSS(Y1, At(RBX, 172)) // the mantissa mask
	a.VPAND(Y1, Y4, Y1)
	a.VBROADCASTSS(Y2, At(RBX, 28)) // 1.0
	a.VPOR(Y1, Y1, Y2)              // m
	a.VSUBPS(Y1, Y1, Y2)            // u
	a.VBROADCASTSS(Y5, At(RBX, 80)) // 2
	a.VADDPS(Y2, Y1, Y5)
	a.VDIVPS(Y2, Y1, Y2) // s = u/(2+u)
	a.VMULPS(Y1, Y2, Y2) // s^2
	a.VBROADCASTSS(Y6, At(RBX, 84))
	for _, off := range []int32{88, 92, 96, 28} {
		a.VBROADCASTSS(Y5, At(RBX, off))
		a.VFMADD213PS(Y6, Y1, Y5)
	}
	a.VMULPS(Y6, Y6, Y2)
	a.VBROADCASTSS(Y5, At(RBX, 80))
	a.VMULPS(Y6, Y6, Y5)             // ln(m)
	a.VBROADCASTSS(Y5, At(RBX, 176)) // ln2
	a.VFMADD213PS(Y0, Y5, Y6)        // e*ln2 + ln(m)
	a.VADDPS(Y3, Y3, Y0)
}
