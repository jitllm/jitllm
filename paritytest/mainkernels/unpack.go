package mainkernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// Unpack turns raw GGUF bytes into the device weight layout, on the device.
//
// Paging a block onto a card otherwise costs a host repack (PackWeightsInto)
// plus an upload, and the repack dominates. It is a
// permutation plus a bit extraction, which a card does at memory rate, so
// uploading the raw bytes and running this kernel makes a page-in roughly one
// upload. It also needs no arena: the backing store is the source bytes.
// tier's resident() asks unpackInto before the arena; an uncovered format
// falls back to PackWeightsInto and is counted (Stats.NoUnpackKernel).
//
// Source and destination are distinct buffers (pSrc only loaded, pQS/pD/pSC
// only stored), so clamped surplus threads recompute from read-only input
// (RULE 13); ir.Validate enforces it and TestUnpackRefusesAnAliasedBuffer
// checks the refusal.
//
// The specification is PackWeightsInto (packRows, packSub, extract); every
// index below is the same expression with per-thread constants folded out,
// and the gate is byte equality against it. Read pack.go for the layout.
//
// One thread owns one (row, super-block), the smallest unit whose outputs are
// assigned rather than OR-ed, so each destination word is written once with a
// complete value (no read-modify-write). The thread index is row-fastest so
// every store coalesces; the reads are then permuted across a warp but still
// contiguous in total, and cached.
func Unpack(t Quant, nrows, k int) (*ir.Kernel, error) {
	qi := qtab[t]
	if qi.blockE == 0 {
		return nil, fmt.Errorf("kernels: Unpack: unknown format %d", t)
	}
	if nrows <= 0 {
		return nil, fmt.Errorf("kernels: Unpack: nrows=%d", nrows)
	}
	if k%qi.blockE != 0 {
		return nil, fmt.Errorf("kernels: Unpack: k=%d is not a multiple of %d", k, qi.blockE)
	}
	if t != Q4_K {
		return nil, fmt.Errorf("kernels: Unpack: %s has no device unpack; %s", t, UnpackWhyNot(t))
	}
	return unpackQ4K(nrows, k)
}

// UnpackSupported reports whether Unpack can emit a kernel for a format.
//
// It is a list, not a capability probe, so a test can sweep it and a format
// losing its kernel is a red line rather than a silent fallback to the host
// packer. TestUnpackCoversExactlyWhatItClaims in lowertest holds it to
// Unpack.
func UnpackSupported(t Quant) bool { return t == Q4_K }

// UnpackWhyNot is the refusal, per format, and it is a statement about the
// source block rather than about effort.
//
// The IR addresses buffers by element index and has no byte type, so a
// format whose block is not a multiple of 4 bytes has fields straddling u32
// boundaries at every offset mod 4, which needs a different kernel:
//
//	Q4_K  144 B  word-aligned   implemented
//	Q5_K  176 B  word-aligned   not yet (Q4_K plus the qh plane)
//	Q4_0   18 B  not aligned    needs the two-word merge
//	Q8_0   34 B  not aligned    needs the two-word merge
//	Q3_K  110 B  not aligned    needs the two-word merge
//	Q6_K  210 B  not aligned    needs the two-word merge
//
// It is exported so the tier reports the reason rather than keeping its own
// copy of this table.
func UnpackWhyNot(t Quant) string {
	if qtab[t].float > 0 {
		return "a float format's pack is a copy of its rows; there is nothing to unpack"
	}
	if qtab[t].blockB%4 != 0 {
		return fmt.Sprintf("its %d-byte block is not a multiple of 4, so the fields straddle "+
			"u32 boundaries and the IR addresses buffers by element index, never by byte",
			qtab[t].blockB)
	}
	return "its block is word-aligned and the kernel is simply not written yet"
}

// unpackQ4K is the Q4_K kernel. Its source block is 144 bytes = 36 u32 words:
//
//	word 0        d (low half) | dmin (high half)     -- already the packed d word
//	words 1..3    the twelve 6-bit scale/min bytes
//	words 4..35   the 128 payload bytes, four 64-element groups of 8 words
//
// and one thread emits, for its (row, super-block): 32 payload words, three sc
// words and one d word.
func unpackQ4K(nrows, k int) (*ir.Kernel, error) {
	const (
		blockWords  = 144 / 4 // u32 per source super-block
		subs        = 8       // sub-blocks per super-block
		wordsPerSub = 4       // payload u32 per sub-block, 32 elements at 4 bits
	)
	nsuper := k / 256
	total := nrows * nsuper

	b := ir.New("unpack_q4k", [3]int{128, 1, 1})
	pSrc := b.Param("pSrc", ir.U32) // raw GGUF tensor bytes, read only
	pQS := b.Param("pQS", ir.U32)   // payload, written only
	pD := b.Param("pD", ir.U32)     // d|dmin per super-block, written only
	pSC := b.Param("pSC", ir.U32)   // packed sc/m pairs, written only

	u32 := func(v int64) ir.Value { return b.Const(ir.U32, v) }

	// flat = super*nrows + row, so consecutive threads differ in the row and
	// every store below is coalesced; the clamp is safe because the kernel is a
	// pure function of pSrc (RULE 13).
	flat := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
		u32(int64(total-1)))
	row := b.Rem(ir.U32, flat, u32(int64(nrows)))
	sup := b.Div(ir.U32, flat, u32(int64(nrows)))

	// src[(row*nsuper + sup) * 36 + i] is packRows' src[(r*nsuper+sb)*blockB:]
	// counted in words instead of bytes.
	sbase := b.Mul(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, row, u32(int64(nsuper))), sup),
		u32(blockWords))
	w := make([]ir.Value, blockWords)
	for i := range w {
		w[i] = b.Load(ir.U32, pSrc, sbase, int64(i))
	}

	// d is the source word verbatim: GGUF stores d then dmin as two
	// little-endian halves of the block's first four bytes, which is exactly the
	// packed d word. Nothing is decoded or rounded.
	b.Store(pD, b.Add(ir.U32, b.Mul(ir.U32, sup, u32(int64(nrows))), row), w[0], 0)

	// The 6-bit scale/min table, byte for byte as scaleMinK4 reads it. s[j] is
	// block byte 4+j, so s[0..3] is word 1, s[4..7] word 2 and s[8..11] word 3.
	sByte := func(j int) ir.Value {
		return b.And(ir.U32, b.Shr(ir.U32, w[1+j/4], u32(int64(8*(j%4)))), u32(0xFF))
	}
	var sc, mn [subs]ir.Value
	for j := 0; j < subs; j++ {
		if j < 4 {
			sc[j] = b.And(ir.U32, sByte(j), u32(63))
			mn[j] = b.And(ir.U32, sByte(j+4), u32(63))
			continue
		}
		// (q[j+4]&0xF) | ((q[j-4]>>6)<<4) and (q[j+4]>>4) | ((q[j]>>6)<<4).
		hi := sByte(j + 4)
		sc[j] = b.Add(ir.U32, b.And(ir.U32, hi, u32(0x0F)),
			b.Shl(ir.U32, b.Shr(ir.U32, sByte(j-4), u32(6)), u32(4)))
		mn[j] = b.Add(ir.U32, b.Shr(ir.U32, hi, u32(4)),
			b.Shl(ir.U32, b.Shr(ir.U32, sByte(j), u32(6)), u32(4)))
	}
	// Container v27's 96-bit stream (scstream.go): the sixteen fields in
	// consumption order, three words a super-block at (sup*3+w)*nrows + r. Each
	// word is one complete value here; a straddling field's high bits open the
	// next word.
	var ws [ScStreamWords]ir.Value
	var have [ScStreamWords]bool
	put := func(w int, v ir.Value) {
		if !have[w] {
			ws[w], have[w] = v, true
			return
		}
		ws[w] = b.Add(ir.U32, ws[w], v) // disjoint bits: Add is the OR
	}
	for j := 0; j < subs; j++ {
		for _, fv := range []struct {
			min bool
			v   ir.Value
		}{{false, sc[j]}, {true, mn[j]}} {
			w, bit, hi := ScField(j, fv.min)
			put(w, b.Shl(ir.U32, fv.v, u32(int64(bit))))
			if hi > 0 {
				put(w+1, b.Shr(ir.U32, fv.v, u32(int64(6-hi))))
			}
		}
	}
	scBase := b.Add(ir.U32, b.Mul(ir.U32, sup, u32(int64(ScStreamWords*nrows))), row)
	for w, v := range ws {
		b.Store(pSC, scBase, v, int64(w*nrows))
	}

	// The payload. extract reads grp := blk[16:][(s/2)*32:] and takes the low
	// nibble of every byte for an even sub-block and the high nibble for an odd
	// one; packSub then puts element l<16 in byte l's low nibble and element
	// l+16 in byte l's high nibble. So for output word wi of sub-block s, with
	// g = s/2:
	//
	//	A = src word 4 + g*8 + wi        (elements  0..15 of the group's half)
	//	B = src word 4 + g*8 + 4 + wi    (elements 16..31)
	//	even: (A & 0x0F0F0F0F) | ((B & 0x0F0F0F0F) << 4)
	//	odd:  ((A >> 4) & 0x0F0F0F0F) | (B & 0xF0F0F0F0)
	//
	// The masks keep every term at most 0xF or 0xF0 per byte, so nothing carries
	// across a byte. ir.OpShr is logical; an arithmetic shift would smear a set
	// high bit through the mask.
	qBase := b.Add(ir.U32, b.Mul(ir.U32, sup, u32(int64(subs*wordsPerSub*nrows))), row)
	lo := u32(0x0F0F0F0F)
	hi := u32(0xF0F0F0F0)
	for s := 0; s < subs; s++ {
		g := s / 2
		for wi := 0; wi < wordsPerSub; wi++ {
			a, bb := w[4+g*8+wi], w[4+g*8+4+wi]
			var v ir.Value
			if s%2 == 0 {
				v = b.Add(ir.U32, b.And(ir.U32, a, lo),
					b.Shl(ir.U32, b.And(ir.U32, bb, lo), u32(4)))
			} else {
				v = b.Add(ir.U32, b.And(ir.U32, b.Shr(ir.U32, a, u32(4)), lo),
					b.And(ir.U32, bb, hi))
			}
			b.Store(pQS, qBase, v, int64((s*wordsPerSub+wi)*nrows))
		}
	}

	kk := b.Done()
	if err := kk.Validate(); err != nil {
		return nil, fmt.Errorf("kernels: Unpack: %w", err)
	}
	return kk, nil
}

// UnpackThreads is how many threads Unpack's kernel needs: one per
// (row, super-block). The caller sizes the launch with it rather than
// re-deriving the mapping.
func UnpackThreads(t Quant, nrows, k int) int {
	qi := qtab[t]
	if qi.blockE == 0 || k%qi.blockE != 0 {
		return 0
	}
	return nrows * (k / qi.blockE)
}
