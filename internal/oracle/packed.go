package oracle

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// The packed layout, read in Go: the reference every generated packed matvec
// and row kernel is gated against. It reads the layout through the kernels
// package's accessors, so the definition still lives in one place (qtab and
// packSub) and this is only a reader of it.

// fmtInfo is the slice of a format's description a reader needs.
type fmtInfo struct {
	blockE, sub, bits, hi, perSuper, scOff int
	biasK                                  float32
	biasArray, signedQ                     bool
	codes                                  *[16]uint8
}

func info(q kernels.Quant) fmtInfo {
	sub, bits, biasK, biasArray := kernels.Layout(q)
	perSuper, scOff := kernels.ScaleLayout(q)
	return fmtInfo{
		blockE: q.Elems(), sub: sub, bits: bits, hi: kernels.HiPlane(q),
		perSuper: perSuper, scOff: scOff, biasK: biasK, biasArray: biasArray,
		signedQ: kernels.SignedPayload(q), codes: kernels.Codes(q),
	}
}

func half(u uint16) float32 { return float32(quant.DecodeHalf(u)) }

// hiPlane reads element l's secondary-plane code out of the words that follow
// the primary ones. base is the sub-block's first word for this row. One shift
// and one mask is the whole contract; packSub's pairing makes it a VPSRLD and
// a VPAND in the kernel.
func hiPlane(qs []byte, qi fmtInfo, base, nrows, pw, l int) uint32 {
	if qi.hi == 0 {
		return 0
	}
	lanes := 8 / qi.hi
	g := l / 4
	v := binary.LittleEndian.Uint32(qs[(base+(pw+g/lanes)*nrows)*4:])
	return (v >> (8*uint(l%4) + uint(qi.hi*(g%lanes)))) & (1<<uint(qi.hi) - 1)
}

// code is what primary code q means: itself, or its entry in the format's code
// table (MXFP4's e2m1 values). See qinfo.codes.
func code(qi fmtInfo, q uint32) uint32 {
	if qi.codes == nil {
		return q
	}
	return uint32(qi.codes[q])
}

// dScale reads row r's super-scale (and its minimum, where the format has one)
// out of the d plane.
//
// Three layouts: a format with a minimum gets the whole word; a narrow f16
// scale shares a word with the next row; and MXFP4's E8M0 exponent is one byte
// of four, biased so that the byte shifted left 23 is the f32, exactly.
func dScale(t kernels.Quant, d []byte, super, nrows, r int) (scale, dmin float32) {
	di, slot := kernels.DIndex(t, super, nrows, r)
	dw := binary.LittleEndian.Uint32(d[di*4:])
	switch kernels.DSlots(t) {
	case 4:
		return math.Float32frombits(uint32(uint8(dw>>(8*uint(slot)))) << 23), 0
	case 2:
		return half(uint16(dw >> (16 * uint(slot)))), 0
	}
	if kernels.D32(t) {
		return math.Float32frombits(dw), 0
	}
	return half(uint16(dw)), half(uint16(dw >> 16))
}

// MatVecPacked is MatVecPackedRows over every row.
func MatVecPacked(out []float32, t kernels.Quant, qs, d, sc []byte, x []float32, nrows, k int) error {
	return MatVecPackedRows(out, t, qs, d, sc, x, nrows, k, 0, nrows)
}

// MatVecPackedRows computes rows [lo, hi) only, so a caller can split the work.
func MatVecPackedRows(out []float32, t kernels.Quant, qs, d, sc []byte, x []float32, nrows, k, lo, hi int) error {
	qi := info(t)
	if qi.blockE == 0 || k%qi.blockE != 0 {
		return fmt.Errorf("oracle: MatVecPacked: k=%d is not a multiple of %s's %d", k, t, qi.blockE)
	}
	// The output is bounded by hi, not nrows: nrows is the buffer's row count
	// (the layout's stride), and a caller may ask for rows [lo, hi) of it, as
	// one expert of a mixture's bank does.
	if lo < 0 || hi < lo || hi > nrows {
		return fmt.Errorf("oracle: MatVecPacked: rows [%d,%d) of %d", lo, hi, nrows)
	}
	if len(x) < k || len(out) < hi {
		return fmt.Errorf("oracle: MatVecPacked: x=%d out=%d for rows [%d,%d) of %d",
			len(x), len(out), lo, hi, k)
	}
	nq, nd, nsc, err := kernels.PackedWords(t, nrows, k)
	if err != nil {
		return err
	}
	if len(qs) < nq*4 || len(d) < nd*4 || len(sc) < nsc*4 {
		return fmt.Errorf("oracle: MatVecPacked: spans qs=%d d=%d sc=%d, want %d/%d/%d bytes",
			len(qs), len(d), len(sc), nq*4, nd*4, nsc*4)
	}
	nsub := k / qi.sub
	// The payload is a primary plane of qi.bits and, for a format whose quants
	// do not fit it, a secondary plane of qi.hi appended after it. See packSub.
	pw, hw := qi.sub*qi.bits/32, qi.sub*qi.hi/32
	words := pw + hw
	// Two scale bytes per word when the format carries its own minimum, four
	// when it does not -- the same split packRows writes and the device kernel
	// reads. stride is the bit offset between lanes within the word.
	perWord, stride := 4, 8
	if qi.biasArray {
		perWord, stride = 2, 16
	}
	u32 := func(b []byte, i int) uint32 { return binary.LittleEndian.Uint32(b[i*4:]) }

	for r := lo; r < hi; r++ {
		var acc float32
		for si := 0; si < nsub; si++ {
			super := si / qi.perSuper
			// One word per super-block per row: d low, dmin high (zero for a
			// format that carries no minimum array).
			scale, dmin := dScale(t, d, super, nrows, r)
			// bias is subtracted from every dequantised value, and the two
			// families reach it differently: a k-quant with a minimum array
			// stores it per sub-block, everything else folds a constant into
			// the scale. Normalising here is what lets the payload loop below
			// be one loop rather than six.
			var bias float32
			if kernels.ScStream(t) {
				scv, mv := scStream(sc, si, nrows, r)
				scale *= float32(int32(scv) + int32(qi.scOff))
				bias = dmin * float32(mv)
			} else if qi.perSuper > 1 {
				sw := u32(sc, (si/perWord)*nrows+r)
				sh := uint(si%perWord) * uint(stride)
				scale *= float32(int32((sw>>sh)&0xFF) + int32(qi.scOff))
				if qi.biasArray {
					bias = dmin * float32((sw>>(sh+8))&0xFF)
				}
			} else if qi.biasArray {
				bias = dmin // Q5_1: no sc plane, so m is one (kernels.MinInD)
			}
			if !qi.biasArray {
				bias = scale * qi.biasK
			}

			base, e0 := si*words*nrows+r, si*qi.sub
			if qi.bits < 8 {
				// packSub: byte b holds element b in its low field and element
				// b + f*sub/fields in field f, so one byte serves activations a
				// field-run apart (two nibbles half a sub-block apart, TQ2's
				// four 2-bit codes a quarter apart).
				fields := 8 / qi.bits
				h := qi.sub / fields
				mask := uint32(1)<<uint(qi.bits) - 1
				for w := 0; w < pw; w++ {
					v := u32(qs, base+w*nrows)
					for b := 0; b < 4; b++ {
						by := v >> (8 * uint(b))
						for f := 0; f < fields; f++ {
							l := w*4 + b + f*h
							q := code(qi, (by>>uint(qi.bits*f))&mask) | hiPlane(qs, qi, base, nrows, pw, l)<<4
							acc += (scale*float32(q) - bias) * x[e0+l]
						}
					}
				}
				continue
			}
			for w := 0; w < pw; w++ {
				v := u32(qs, base+w*nrows)
				for b := 0; b < 4; b++ {
					by := byte(v >> (8 * uint(b)))
					q := float32(by)
					if qi.signedQ {
						q = float32(int8(by))
					}
					acc += (scale*q - bias) * x[e0+w*4+b]
				}
			}
		}
		out[r] = acc
	}
	return nil
}

// RowPacked dequantises row r of a weight in the device layout -- the embedding
// lookup, on bytes that are column-major across rows -- so the container needs
// no second layout for token_embd. The read is strided (one row's words sit
// nrows*4 bytes apart). The engine runs cpu.EmitPackedRow; this is its
// reference.
func RowPacked(dst []float32, t kernels.Quant, qs, d, sc []byte, r, nrows, k int) error {
	qi := info(t)
	if qi.blockE == 0 || k%qi.blockE != 0 {
		return fmt.Errorf("oracle: RowPacked: k=%d is not a multiple of %s's %d", k, t, qi.blockE)
	}
	if r < 0 || r >= nrows {
		return fmt.Errorf("oracle: RowPacked: row %d of %d", r, nrows)
	}
	if len(dst) < k {
		return fmt.Errorf("oracle: RowPacked: dst=%d, want %d", len(dst), k)
	}
	nq, nd, nsc, err := kernels.PackedWords(t, nrows, k)
	if err != nil {
		return err
	}
	if len(qs) < nq*4 || len(d) < nd*4 || len(sc) < nsc*4 {
		return fmt.Errorf("oracle: RowPacked: spans qs=%d d=%d sc=%d, want %d/%d/%d bytes",
			len(qs), len(d), len(sc), nq*4, nd*4, nsc*4)
	}
	nsub := k / qi.sub
	// The payload layout is as in MatVecPackedRows.
	pw, hw := qi.sub*qi.bits/32, qi.sub*qi.hi/32
	words := pw + hw
	perWord, stride := 4, 8
	if qi.biasArray {
		perWord, stride = 2, 16
	}
	u32 := func(b []byte, i int) uint32 { return binary.LittleEndian.Uint32(b[i*4:]) }

	for si := 0; si < nsub; si++ {
		super := si / qi.perSuper
		scale, dmin := dScale(t, d, super, nrows, r)
		var bias float32
		if kernels.ScStream(t) {
			scv, mv := scStream(sc, si, nrows, r)
			scale *= float32(int32(scv) + int32(qi.scOff))
			bias = dmin * float32(mv)
		} else if qi.perSuper > 1 {
			sw := u32(sc, (si/perWord)*nrows+r)
			sh := uint(si%perWord) * uint(stride)
			scale *= float32(int32((sw>>sh)&0xFF) + int32(qi.scOff))
			if qi.biasArray {
				bias = dmin * float32((sw>>(sh+8))&0xFF)
			}
		} else if qi.biasArray {
			bias = dmin // Q5_1: no sc plane, so m is one (kernels.MinInD)
		}
		if !qi.biasArray {
			bias = scale * qi.biasK
		}

		base, e0 := si*words*nrows+r, si*qi.sub
		if qi.bits < 8 {
			fields := 8 / qi.bits
			h := qi.sub / fields
			mask := uint32(1)<<uint(qi.bits) - 1
			for w := 0; w < pw; w++ {
				v := u32(qs, base+w*nrows)
				for b := 0; b < 4; b++ {
					by := v >> (8 * uint(b))
					for f := 0; f < fields; f++ {
						l := w*4 + b + f*h
						q := code(qi, (by>>uint(qi.bits*f))&mask) | hiPlane(qs, qi, base, nrows, pw, l)<<4
						dst[e0+l] = scale*float32(q) - bias
					}
				}
			}
			continue
		}
		for w := 0; w < pw; w++ {
			v := u32(qs, base+w*nrows)
			for b := 0; b < 4; b++ {
				by := byte(v >> (8 * uint(b)))
				q := float32(by)
				if qi.signedQ {
					q = float32(int8(by))
				}
				dst[e0+w*4+b] = scale*q - bias
			}
		}
	}
	return nil
}

// scStream reads sub-block si's scale and minimum out of a row's 96-bit SC
// stream (kernels.ScStream): three words a super-block, [word][row].
func scStream(sc []byte, si, nrows, r int) (scale, min uint8) {
	var ws [kernels.ScStreamWords]uint32
	sup := si / 8
	for w := range ws {
		ws[w] = binary.LittleEndian.Uint32(sc[((sup*kernels.ScStreamWords+w)*nrows+r)*4:])
	}
	return kernels.ScValue(ws, si%8, false), kernels.ScValue(ws, si%8, true)
}
