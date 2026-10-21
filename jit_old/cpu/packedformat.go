//go:build arm64 || amd64

package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// packedFormat is a packed format's layout as the matrix emitters read it,
// decomposed once: the payload planes, the scale layout, the correction the
// payload needs, and the d plane's width. It decides no schedule; every
// emitter keeps its own register plan and instruction order.
type packedFormat struct {
	q         kernels.Quant
	sub, bits int // elements per sub-block, primary payload bits
	biasK     float32
	biasArray bool // a minimum per sub-block (Q4_K, Q5_K, Q5_1)
	perSuper  int  // sub-blocks per super-block
	scOff     int  // the offset a stored sub-block scale carries (Q3_K, Q6_K)
	signed    bool // the payload is int8 (Q8_0)
	minD      bool // Q5_1: dmin alone, no sc word to read m from
	hi        int  // secondary plane bits, 0 for none
	codes     bool // the payload indexes a code table (MXFP4)
	pw, hw    int  // primary and secondary payload words per sub-block
	words     int  // pw + hw
	halfSub   int
	// fields is how many primary codes a payload byte holds (8/bits: two
	// nibbles, four 2-bit codes, one byte) and run how far apart in the
	// sub-block a byte's fields are, sub/fields elements (kernels.Fields).
	fields, run int
	sub16       bool // a 16-wide sub-block, corrected through AHalf
	stream      bool // container v27's SC stream (scstream.go)
	narrow      bool // the d plane is narrower than a word a row (dplane.go)
	e8          bool // the d plane is an E8M0 byte
	d32         bool // the d word is the scale as an f32 (TQ2_F32)
}

// packedFormatOf decomposes t. who names the emitter in a refusal.
func packedFormatOf(t quant.Type, who string) (packedFormat, error) {
	q, ok := kernels.QuantOf(t)
	if !ok {
		return packedFormat{}, fmt.Errorf("jit: %s: %s has no device layout", who, t)
	}
	return packedFormatK(q, t)
}

// packedFormatK is packedFormatOf for an emitter that has resolved the quant.
func packedFormatK(q kernels.Quant, t quant.Type) (packedFormat, error) {
	f := packedFormat{q: q}
	f.sub, f.bits, f.biasK, f.biasArray = kernels.Layout(q)
	f.perSuper, f.scOff = kernels.ScaleLayout(q)
	f.signed = kernels.SignedPayload(q)
	f.minD = kernels.MinInD(q)
	f.hi = kernels.HiPlane(q)
	codes, err := hasCodes(q)
	if err != nil {
		return packedFormat{}, err
	}
	f.codes = codes
	f.pw, f.hw = f.sub*f.bits/32, f.sub*f.hi/32
	f.words = f.pw + f.hw
	f.halfSub = f.sub / 2
	f.fields = 8 / f.bits
	f.run = f.sub / f.fields
	f.sub16 = f.sub == Q8Block/2
	f.stream = kernels.ScStream(q)
	f.narrow = NarrowD(t)
	f.e8 = E8M0D(t)
	f.d32 = kernels.D32(q)
	return f, nil
}

// pairOff is sub-block sn's byte offset into a token's AScale: a (d_act,
// correction) pair per 32 activations.
func (f *packedFormat) pairOff(sn int) int32 { return int32(f.sub * sn / Q8Block * 8) }

// scShift is sub-block sn's bit position within its packed SC word.
func (f *packedFormat) scShift(sn int) byte {
	return byte((sn % perWordOf(f.biasArray)) * strideOf(f.biasArray))
}

// scWordDone reports whether sub-block sn is the last to read the SC word
// under the cursor, so the cursor steps one plane stride after it.
func (f *packedFormat) scWordDone(sn int) bool {
	if f.stream {
		return scStep(sn)
	}
	return f.perSuper > 1 && sn%perWordOf(f.biasArray) == perWordOf(f.biasArray)-1
}

// corrK is the correction's constant factor. QuantizeQ8Window stores
// -sum*BiasC/8 in the activation's pair, so a kernel multiplies it back up: 8
// for a fixed bias or a minimum array, 4 for the 16-wide formats whose sums
// are per half-block, and 0 for a signed payload, which needs no correction
// on a signed dot.
func (f *packedFormat) corrK() int {
	switch {
	case f.signed:
		return 0
	case f.biasArray:
		return 8
	case f.sub16:
		return 4
	case f.biasK != 0:
		return 8
	}
	return 0
}

// hiFields calls field for every primary chunk whose secondary bits live in
// secondary word hwi, in payload order (lo half then hi half): gi is the chunk,
// w its primary word and s its bit offset within each byte of the secondary
// word. packSub's pairing puts chunk gi in byte lane gi%lanes of word
// gi/lanes.
func (f *packedFormat) hiFields(hwi int, field func(half, w, s int)) {
	lanes := 8 / f.hi
	for half := 0; half < 2; half++ {
		for w := 0; w < f.pw; w++ {
			gi := w + half*f.pw
			if gi/lanes != hwi {
				continue
			}
			field(half, w, f.hi*(gi%lanes))
		}
	}
}

// perWordOf is how many sub-blocks share one packed SC word.
func perWordOf(biasArray bool) int {
	if biasArray {
		return 2
	}
	return 4
}

// strideOf is the bits one sub-block takes in a packed SC word.
func strideOf(biasArray bool) int {
	if biasArray {
		return 16
	}
	return 8
}

// fieldMask turns the 0x0F byte mask a kernel loads from its constants into a
// field's mask: 0x0F for a nibble, and for a 2-bit field 0x03, as
// mask & (mask >> 2) -- a dword shift moves bits across bytes, and the AND
// keeps only each byte's own. tmp is clobbered.
func fieldMask(a *Buf, mask, tmp Reg, bits int) {
	if bits != 2 {
		return
	}
	a.VPSRLD(tmp, mask, 2)
	a.VPAND(mask, mask, tmp)
}

// fieldOf leaves field f of every payload byte in dst, unshifted to bit 0 and
// masked: (pay >> bits*f) & mask.
func fieldOf(a *Buf, dst, pay, mask Reg, bits, f int) {
	if f == 0 {
		a.VPAND(dst, pay, mask)
		return
	}
	a.VPSRLD(dst, pay, byte(bits*f))
	a.VPAND(dst, dst, mask)
}
