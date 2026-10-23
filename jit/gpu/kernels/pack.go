package kernels

import (
	"encoding/binary"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
)

// BlockBytes and Elems describe the source (GGUF) block.
func (q Quant) BlockBytes() int { return q.info().blockB }
func (q Quant) Elems() int      { return q.info().blockE }

// PackWeights turns a GGUF tensor into the device layout: payload and scales,
// both transposed so element u of every row is contiguous.
//
// The transpose is a permutation, so it costs no bytes; it makes a warp's
// loads (one thread per row, element u of consecutive rows) coalesce.
//
// Every format is normalised to value = scale*q - bias here, once at load.
// PackedWords gives the resulting sizes; Q3_K (+34.5%) and Q6_K (+31.4%) are
// the formats still stored wider than their GGUF payload.
func PackWeights(t Quant, src []byte, nrows, k int) (qs, d, sc []uint32, err error) {
	nq, nd, nsc, err := PackedWords(t, nrows, k)
	if err != nil {
		return nil, nil, nil, err
	}
	qs, d = make([]uint32, nq), make([]uint32, nd)
	if nsc > 0 {
		sc = make([]uint32, nsc)
	}
	if err := PackWeightsInto(t, src, nrows, k, qs, d, sc); err != nil {
		return nil, nil, nil, err
	}
	return qs, d, sc, nil
}

// PackWeightsInto is PackWeights writing into caller-owned arrays.
// PackWeightsInto is PackWeights writing into caller-owned arrays, so the pack
// can land in memory that outlives the upload (Arena.Pack keeps it for later
// page-ins). TestArenaMatchesPackWeights holds the two forms to the same bytes.
//
// The destination lengths are PackedWords', and sc may be nil for a format that
// has no sub-block scale array.
//
// d and sc are cleared here: packRows ORs its scale words (rows and sub-blocks
// share words), so leftover contents would OR into the new scales.
func PackWeightsInto(t Quant, src []byte, nrows, k int, qs, d, sc []uint32) error {
	qi := qtab[t]
	// A float format is sized in elements, not blocks: its "block" is the
	// activation granule and raw rows have none. The quantized formats keep the
	// block checks, which catch a truncated tensor.
	if qi.float > 0 {
		if k <= 0 || nrows <= 0 {
			return fmt.Errorf("kernels: PackWeights: %dx%d", nrows, k)
		}
		if want := nrows * k * qi.float; len(src) < want {
			return fmt.Errorf("kernels: PackWeights: need %d bytes, have %d", want, len(src))
		}
	}
	nsuper := 0
	if qi.float == 0 {
		if qi.blockE == 0 || k%qi.blockE != 0 {
			return fmt.Errorf("kernels: PackWeights: k=%d is not a multiple of %d", k, qi.blockE)
		}
		nsuper = k / qi.blockE
		if want := nrows * nsuper * qi.blockB; len(src) < want {
			return fmt.Errorf("kernels: PackWeights: need %d bytes, have %d", want, len(src))
		}
	}
	nq, nd, nsc, err := PackedWords(t, nrows, k)
	if err != nil {
		return err
	}
	if len(qs) < nq || len(d) < nd || len(sc) < nsc {
		return fmt.Errorf("kernels: PackWeights: destination too small: qs %d/%d, d %d/%d, sc %d/%d",
			len(qs), nq, len(d), nd, len(sc), nsc)
	}
	qs, d = qs[:nq], d[:nd]
	// nil rather than empty: packRows tests sc for nil to decide whether the
	// format has a sub-block scale array.
	if nsc == 0 {
		sc = nil
	} else {
		sc = sc[:nsc]
	}
	clear(d)
	clear(sc)
	if qi.float > 0 {
		// Raw rows, row-major: the words are the source bytes (little-endian
		// on every host this builds for).
		copy(unsafe.Slice((*byte)(unsafe.Pointer(&qs[0])), 4*len(qs)), src)
		return nil
	}
	if t == MXFP4 {
		if err := mxfp4Refuse(src[:nrows*nsuper*qi.blockB]); err != nil {
			return err
		}
	}

	// Row-parallel: every payload write is indexed [... * nrows + r] and every
	// read comes from row r's own source bytes. Rows are not fully disjoint in
	// the d plane, though: DSlots rows share one word and packRows ORs into it,
	// so the split below keeps each word's rows in one worker. The output is
	// byte-identical for any worker count.
	n := runtime.GOMAXPROCS(0)
	if n > nrows {
		n = nrows
	}
	// A small tensor is not worth the goroutines.
	if nrows < 256 || n < 2 {
		packRows(t, qi, src, qs, d, sc, nrows, nsuper, 0, nrows)
		return nil
	}
	var wg sync.WaitGroup
	// The split must not cut a d word's group of DSlots rows (two for Q4_0/Q8_0,
	// four for MXFP4): two workers ORing into one word lose an update and a
	// scale silently reads back as zero. packPer rounds the share to whole
	// words; `go test -race` catches a violation.
	per := packPer(t, nrows, n)
	for w := 0; w < n; w++ {
		lo, hi := w*per, min(w*per+per, nrows)
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			packRows(t, qi, src, qs, d, sc, nrows, nsuper, lo, hi)
		}(lo, hi)
	}
	wg.Wait()
	return nil
}

// packPer is how many rows one packing worker takes: a share of nrows over n,
// rounded up to whole d words (DSlots rows each) so no two workers OR into the
// same word.
func packPer(t Quant, nrows, n int) int {
	per := (nrows + n - 1) / n
	if g := DSlots(t); per%g != 0 {
		per += g - per%g
	}
	return per
}

// packRows packs rows [lo, hi) of one tensor. Its q scratch is per call, which
// is what makes the fan-out above safe.
func packRows(t Quant, qi qinfo, src []byte, qs, d, sc []uint32, nrows, nsuper, lo, hi int) {
	q := make([]int32, qi.sub)
	for r := lo; r < hi; r++ {
		for sb := 0; sb < nsuper; sb++ {
			blk := src[(r*nsuper+sb)*qi.blockB:][:qi.blockB]
			perSuper := qi.blockE / qi.sub
			for s := 0; s < perSuper; s++ {
				dBits, dminBits, sci, mi := extract(t, blk, s, q)
				si := sb*perSuper + s
				packSub(qs, q, qi, si, nrows, r)

				// d is copied as raw f16 bits, not re-encoded.
				sup := si / qi.perSuper
				// OR, not assign: rows share a word on a narrow format, which is
				// why PackWeightsInto clears d on entry.
				di, slot := DIndex(t, sup, nrows, r)
				switch DSlots(t) {
				case 4:
					// One byte per row: extract returned the stored exponent.
					d[di] |= uint32(dBits) << (8 * uint(slot))
				case 2:
					d[di] |= uint32(dBits) << (16 * uint(slot))
				default:
					d[di] |= uint32(dBits) | uint32(dminBits)<<16
				}
				if sc != nil {
					if ScStream(t) {
						// The 96-bit stream (scstream.go): OR each field into its
						// word, and a straddling one into the next word too.
						var ws [ScStreamWords]uint32
						scPut(&ws, s, false, sci)
						scPut(&ws, s, true, mi)
						for w, v := range ws {
							sc[(sup*ScStreamWords+w)*nrows+r] |= v
						}
					} else if qi.biasArray {
						w, lane := si/2, si%2
						sc[w*nrows+r] |= uint32(sci)<<(uint(lane)*16) | uint32(mi)<<(uint(lane)*16+8)
					} else {
						w, lane := si/4, si%4
						sc[w*nrows+r] |= uint32(sci) << (uint(lane) * 8)
					}
				}
			}
		}
	}
}

// packSub writes one sub-block's integers in the device layout: for 4-bit, the
// first half in the low nibbles and the second half in the high nibbles of the
// same bytes; for 8-bit, straight signed bytes. A format with a secondary plane
// appends it after the primary words.
//
// The secondary plane's pairing is chosen for the kernel: byte j of a hi word
// holds every element with l%4 == j, ordered by l/4, so elements [4w,4w+4) are
// `(v >> hi*(w % lanes)) & rep(mask)` and line up with the four bytes the
// primary word w produced. The kernel never shuffles.
func packSub(qs []uint32, q []int32, qi qinfo, si, nrows, r int) {
	pw := qi.sub * qi.bits / 32
	hw := qi.sub * qi.hi / 32
	var out [32]byte
	if qi.bits < 8 {
		// Fields(qi) fields a byte: element l is field l/(sub/fields) of byte
		// l%(sub/fields), so a byte's fields are elements a field-run apart
		// -- the 4-bit formats' two nibbles half a sub-block apart, TQ2's four
		// 2-bit codes a quarter apart.
		n := qi.sub / Fields(qi)
		for l := 0; l < qi.sub; l++ {
			out[l%n] |= (byte(q[l]) & byte(1<<uint(qi.bits)-1)) << uint(qi.bits*(l/n))
		}
	} else {
		for l := 0; l < qi.sub; l++ {
			out[l] = byte(int8(q[l]))
		}
	}
	if hw > 0 {
		lanes := 8 / qi.hi // elements per byte of a hi word
		for l := 0; l < qi.sub; l++ {
			g := l / 4
			b := pw*4 + (g/lanes)*4 + l%4
			out[b] |= byte(q[l]>>uint(qi.bits)) << uint(qi.hi*(g%lanes))
		}
	}
	for w := 0; w < pw+hw; w++ {
		qs[(si*(pw+hw)+w)*nrows+r] = binary.LittleEndian.Uint32(out[4*w:])
	}
}

// Fields is how many primary codes one payload byte holds: two nibbles, four
// 2-bit codes, or one byte. A field-run, sub/Fields elements, is how far apart
// a byte's fields are (packSub).
func Fields(qi qinfo) int { return 8 / qi.bits }

// FieldsOf is Fields for a format, for an emitter.
func FieldsOf(q Quant) int { return Fields(q.info()) }

// extract pulls one sub-block's integer quants out of a GGUF block, along with
// the raw f16 bits of its super-scale and the integer per-sub-block scale.
// It is the only place any format's bit layout is known.
func extract(t Quant, blk []byte, s int, q []int32) (dBits, dminBits uint16, sc, m uint8) {
	switch t {
	case Q4_0:
		for l := 0; l < 16; l++ {
			q[l] = int32(blk[2+l] & 0xF)
			q[l+16] = int32(blk[2+l] >> 4)
		}
		return binary.LittleEndian.Uint16(blk), 0, 0, 0

	case Q5_0:
		// Q4_0's nibbles plus a 32-bit fifth-bit plane; element l takes bit l.
		qh := binary.LittleEndian.Uint32(blk[2:])
		for l := 0; l < 16; l++ {
			q[l] = int32(blk[6+l] & 0xF)
			q[l+16] = int32(blk[6+l] >> 4)
		}
		for l := 0; l < 32; l++ {
			if qh&(1<<uint(l)) != 0 {
				q[l] += 16
			}
		}
		return binary.LittleEndian.Uint16(blk), 0, 0, 0

	case Q5_1:
		// Q5_0's planes after two halves, d then m. The minimum goes out negated
		// (sign bit flipped, exact for every half) because every reader subtracts
		// dmin*m and Q5_1's m is one: see MinInD.
		qh := binary.LittleEndian.Uint32(blk[4:])
		for l := 0; l < 16; l++ {
			q[l] = int32(blk[8+l] & 0xF)
			q[l+16] = int32(blk[8+l] >> 4)
		}
		for l := 0; l < 32; l++ {
			if qh&(1<<uint(l)) != 0 {
				q[l] += 16
			}
		}
		return binary.LittleEndian.Uint16(blk), binary.LittleEndian.Uint16(blk[2:]) ^ 0x8000, 0, 0

	case Q8_0:
		for l := 0; l < 32; l++ {
			q[l] = int32(int8(blk[2+l]))
		}
		return binary.LittleEndian.Uint16(blk), 0, 0, 0

	case TQ2, TQ2F:
		// Sub-block s is elements 32s..32s+31: bits 2(s%4) of qs[32(s/4)+l]
		// (quant's TQ2_0 decoder). Every sub-block carries the block's d;
		// TQ2F's f32 goes out as the d word's two halves, whole.
		g, sh := blk[32*(s/4):], uint(2*(s%4))
		for l := 0; l < 32; l++ {
			q[l] = int32(g[l] >> sh & 3)
		}
		if t == TQ2F {
			return binary.LittleEndian.Uint16(blk[64:]), binary.LittleEndian.Uint16(blk[66:]), 0, 0
		}
		return binary.LittleEndian.Uint16(blk[64:]), 0, 0, 0

	case MXFP4:
		// Q4_0's nibble order with the raw e2m1 code kept; the kernels translate
		// it through qinfo.codes. mxfp4Refuse has already rejected exponents the
		// d plane cannot hold.
		for l := 0; l < 16; l++ {
			q[l] = int32(blk[1+l] & 0xF)
			q[l+16] = int32(blk[1+l] >> 4)
		}
		return uint16(e8m0Store(blk[0])), 0, 0, 0

	case Q4_K:
		sc, m = scaleMinK4(s, blk[4:16])
		grp := blk[16:][(s/2)*32:]
		for l := 0; l < 32; l++ {
			if s%2 == 0 {
				q[l] = int32(grp[l] & 0xF)
			} else {
				q[l] = int32(grp[l] >> 4)
			}
		}
		return binary.LittleEndian.Uint16(blk[0:]), binary.LittleEndian.Uint16(blk[2:]), sc, m

	case Q5_K:
		sc, m = scaleMinK4(s, blk[4:16])
		qh, ql := blk[16:48], blk[48:][(s/2)*32:]
		// ql advances 32 bytes per 64-element group, qh does not: the same 32
		// high-bit bytes serve all four groups, two bits each.
		for l := 0; l < 32; l++ {
			v := int32(ql[l] & 0xF)
			if s%2 == 1 {
				v = int32(ql[l] >> 4)
			}
			if qh[l]&(1<<uint(s)) != 0 {
				v += 16
			}
			q[l] = v
		}
		return binary.LittleEndian.Uint16(blk[0:]), binary.LittleEndian.Uint16(blk[2:]), sc, m

	case Q3_K:
		raw := q3kScaleRaw(blk[96:108], s) // 0..63, the -32 lives in scOff
		hm, qsb := blk[0:32], blk[32:96]
		half, j := s/8, (s%8)/2
		base := (s % 8 % 2) * 16
		shift := uint(2 * j)
		mbit := byte(1) << uint(j+4*half)
		for l := 0; l < 16; l++ {
			v := int32((qsb[half*32+base+l] >> shift) & 3)
			// A set high-mask bit means do not subtract 4; inverting this produces
			// plausible text, not a crash. The -4 is in biasK, so a set bit adds 4.
			if hm[base+l]&mbit != 0 {
				v += 4
			}
			q[l] = v
		}
		return binary.LittleEndian.Uint16(blk[108:]), 0, raw, 0

	case Q6_K:
		// Stored biased by 128 so the byte is unsigned; scOff undoes it.
		scv := uint8(int(int8(blk[192+s])) + 128)
		g, e0 := s/8, (s%8)*16
		ql, qh := blk[g*64:g*64+64], blk[128+g*32:128+g*32+32]
		for l := 0; l < 16; l++ {
			e := e0 + l
			lo, sh := e%32, uint(2*(e/32))
			nib := ql[lo+(e/32)%2*32]
			if e/32 >= 2 {
				nib >>= 4
			} else {
				nib &= 0xF
			}
			q[l] = int32(nib) | int32((qh[lo]>>sh)&3)<<4
		}
		return binary.LittleEndian.Uint16(blk[208:]), 0, scv, 0
	}
	return 0, 0, 0, 0
}

// e8m0Store is what the d plane holds for an MXFP4 block: the E8M0 exponent
// biased so that the stored byte shifted left 23 is the f32 scale.
//
// The block scale is 2^(e-128) (the exponent halved to match the doubled code
// values) and an f32's bits for 2^x are (x+127)<<23, so storing e-1 makes
// every reader's decode a single shift. e 0, 1 and 255 are refused by
// mxfp4Refuse.
func e8m0Store(e uint8) uint8 { return e - 1 }

// mxfp4Refuse checks every block's exponent against what the d plane can hold.
// The E8M0 byte holds every exponent the format defines; what is left is what
// an f32 cannot express as a normal scale:
//
//	e == 0    2^-128, below f32's smallest normal
//	e == 1    2^-127, likewise; e8m0Store would give it bit pattern 0 (0.0),
//	          so it is refused rather than silently flushed
//	e == 255  E8M0's NaN
//
// A block of zeros is exact at any scale, so it passes.
func mxfp4Refuse(src []byte) error {
	for b := 0; b+17 <= len(src); b += 17 {
		if e := src[b]; e >= 2 && e != 0xFF {
			continue
		}
		for _, c := range src[b+1 : b+17] {
			if c&0x77 != 0 {
				return fmt.Errorf("kernels: MXFP4 block %d has scale 2^%d, which an f32 "+
					"cannot hold as a normal scale", b/17, int(src[b])-128)
			}
		}
	}
	return nil
}

// scaleMinK4 unpacks Q4_K's and Q5_K's 6-bit scale/min pair for sub-block j.
func scaleMinK4(j int, q []byte) (sc, m uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j] >> 6) << 4)
}

// q3kScaleRaw unpacks Q3_K's raw six-bit sub-scale (0..63) for sub-block s.
func q3kScaleRaw(b []byte, s int) uint8 { return uint8(q3kScale(b, s)) & 63 }

// q3kScale unpacks Q3_K's 16 six-bit sub-scales from 12 bytes.
//
// Four low bits sit in place and two high bits are gathered from the last four
// bytes; aux[2] and aux[3] must be computed from the original aux[0] and
// aux[1] before either is overwritten.
func q3kScale(b []byte, s int) int8 {
	var aux [4]uint32
	aux[0] = binary.LittleEndian.Uint32(b[0:])
	aux[1] = binary.LittleEndian.Uint32(b[4:])
	aux[2] = binary.LittleEndian.Uint32(b[8:])
	const kmask1, kmask2 = 0x03030303, 0x0f0f0f0f
	tmp := aux[2]
	aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
	aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
	aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
	aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
	return int8(aux[s/4] >> uint(8*(s%4)))
}

// F16 decodes an IEEE half, delegating to quant.DecodeHalf (one decoder; a
// hand-written copy here once mishandled subnormals).
func F16(u uint16) float32 { return float32(quant.DecodeHalf(u)) }

func f16(u uint16) float32 { return F16(u) }

// NarrowScales reports whether d is the scale: one scale per 32-wide block and
// no per-sub-block one (Q4_0, Q5_0, Q5_1, Q8_0 and MXFP4). How many rows share
// a d word is DSlots' question. Its caller (cpu.WideActWindow) asks the block
// width.
func NarrowScales(q Quant) bool { return qtab[q].perSuper == 1 }

// DSlots is how many rows share one 32-bit d word: one for a format with a
// minimum beside its scale (it needs the whole word), two for a narrow f16
// scale, and four for MXFP4, whose E8M0 scale fits one byte (making the packed
// block 17 bytes, the GGUF's own size).
func DSlots(q Quant) int {
	switch {
	case qtab[q].e8m0:
		return 4
	case qtab[q].biasArray, qtab[q].d32:
		// A minimum beside the scale needs the whole word, whatever perSuper
		// is (Q5_1's d and m, a k-quant's d and dmin), and so does an f32
		// scale (TQ2F).
		return 1
	case qtab[q].perSuper == 1:
		return 2
	}
	return 1
}

// DIndex maps a (super-block, row) onto the d plane: the word index, and which
// slot of that word this row's scale occupies.
//
// Adjacent rows share a word, not adjacent super-blocks: every kernel walks
// rows in groups of four, eight or sixteen from an aligned base, so the slot is
// known at emit time; pairing super-blocks would put it on a runtime loop
// counter and force an unroll.
//
// One function for the packer and every reader; emitters bake the same
// arithmetic, and TestPackAgainstRealWeights holds an independent restatement
// against both. The device has two readers: kernels/matvec.go and
// kernels/mma.go.
func DIndex(q Quant, super, nrows, r int) (idx, slot int) {
	if n := DSlots(q); n > 1 {
		return super*((nrows+n-1)/n) + r/n, r % n
	}
	return super*nrows + r, 0
}

// PackedWords returns the length of the three arrays PackWeights would build,
// without building them, so a budget decision costs nothing.
// TestPackedWordsMatchesPack holds it to PackWeights format by format.
func PackedWords(t Quant, nrows, k int) (qs, d, sc int, err error) {
	qi := qtab[t]
	// A float format is answered before the block check because it has no
	// block: qtab's blockE 32 is the activation granule, and MLA's absorbed
	// per-head sheets can have k = 16 (see MatVecShape.Sub).
	if qi.float > 0 {
		if k <= 0 {
			return 0, 0, 0, fmt.Errorf("kernels: PackedWords: k=%d", k)
		}
		return nrows * k * qi.float / 4, 0, 0, nil
	}
	// The layout's granule is a super-block of sub-blocks, which is the source
	// block for every format but TQ2, whose 256-wide source block is eight
	// sub-blocks of their own scale: a k-slice of a packed row (superWords)
	// is whole sub-blocks, and packing checks the source block itself.
	if unit := qi.sub * qi.perSuper; qi.blockE == 0 || k <= 0 || k%unit != 0 {
		return 0, 0, 0, fmt.Errorf("kernels: PackedWords: k=%d is not a multiple of %d", k, unit)
	}
	nsub := k / qi.sub
	qs = nrows * nsub * (qi.sub * (qi.bits + qi.hi) / 32)
	// One d word per super-block per row, except that DSlots rows share a
	// word on the formats with one scale per 32 weights (see DIndex); the row
	// count rounds up so a ragged last group cannot start super-block s+1
	// part way through a word.
	nsuper := nsub / qi.perSuper
	if n := DSlots(t); n > 1 {
		d = nsuper * ((nrows + n - 1) / n)
	} else {
		d = nsuper * nrows
	}
	switch {
	case ScStream(t):
		sc = nrows * nsuper * ScStreamWords
	case qi.perSuper > 1:
		per := 4
		if qi.biasArray {
			per = 2
		}
		sc = nrows * ((nsub + per - 1) / per)
	}
	return qs, d, sc, nil
}

// RowWindow copies rows [lo, lo+n) of a packed tensor whose buffer holds nrows
// rows into dst tensors laid out for exactly `to` rows (n <= to), the rows past
// n zeroed. dst spans must be PackedWords(t, to, k) words each. It lets a host
// kernel serve a row count its tile does not divide.
//
// Every plane is [unit][row] words with the row fastest, so a window is one
// strided copy per unit, except the d plane, where DSlots rows share a word:
// lo must be a multiple of DSlots, which a tile boundary always is.
func RowWindow(t Quant, qs, d, sc []byte, nrows, k, lo, n, to int, dqs, dd, dsc []byte) error {
	nq, nd, nsc, err := PackedWords(t, nrows, k)
	if err != nil {
		return err
	}
	tq, td, tsc, _ := PackedWords(t, to, k)
	if n > to || lo < 0 || lo+n > nrows || len(dqs) < tq*4 || len(dd) < td*4 || len(dsc) < tsc*4 {
		return fmt.Errorf("kernels: RowWindow: rows [%d,%d) of %d into %d", lo, lo+n, nrows, to)
	}
	clear(dqs[:tq*4])
	clear(dd[:td*4])
	clear(dsc[:tsc*4])
	plane := func(dst, src []byte, units, srcRows, dstRows, first, count int) {
		for u := 0; u < units; u++ {
			copy(dst[(u*dstRows)*4:(u*dstRows+count)*4], src[(u*srcRows+first)*4:(u*srcRows+first+count)*4])
		}
	}
	plane(dqs, qs, nq/nrows, nrows, to, lo, n)
	if nsc > 0 {
		plane(dsc, sc, nsc/nrows, nrows, to, lo, n)
	}
	if sl := DSlots(t); sl > 1 {
		// The first row must start a word; tile boundaries always do.
		if lo%sl != 0 {
			return fmt.Errorf("kernels: RowWindow: a %d-row-per-word d plane needs a first "+
				"row that is a multiple of %d, not %d", sl, sl, lo)
		}
		sp, dp := (nrows+sl-1)/sl, (to+sl-1)/sl
		plane(dd, d, nd/sp, sp, dp, lo/sl, (n+sl-1)/sl)
		// A ragged last word carries rows past the window (the next tile's
		// scales), which must be zeroed.
		if rem := n % sl; rem != 0 && lo+n < nrows {
			w0 := (n / sl) * 4
			for u := 0; u < nd/sp; u++ {
				w := u*dp*4 + w0
				for b := rem * (4 / sl); b < 4; b++ {
					dd[w+b] = 0
				}
			}
		}
		return nil
	}
	plane(dd, d, nd/nrows, nrows, to, lo, n)
	return nil
}
