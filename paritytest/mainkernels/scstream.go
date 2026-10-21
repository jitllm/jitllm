package mainkernels

import "github.com/jitllm/jitllm/jit_old/gpu/ir"

// ScStreamWords is how many SC words a row's super-block holds in a format
// whose sub-block scales and minima are 6-bit fields (ScStream): sixteen of
// them in 96 bits.
const ScStreamWords = 3

// ScStream reports whether q packs its sub-block scale/min pairs as the 96-bit
// stream: Q4_K and Q5_K, whose file already stores them as sixteen 6-bit
// fields in 12 bytes.
//
// Container v27 stores the fields in 12 bytes, as the file does; the earlier
// byte per field made Q4_K 148 bytes against the GGUF's 144, a cost on every
// bandwidth-bound decode (docs/engineering-history/cpu-kernels.md).
//
// The fields run in the order a kernel consumes them -- sub-block 0's scale,
// its minimum, sub-block 1's scale, ... -- at bit 6f of the stream, word f*6/32
// of the super-block's three. Fourteen sit inside one word and read as two
// shifts; two straddle a word boundary (sub-block 2's minimum, sub-block 5's
// scale) and read one more word, always the next one, so a cursor walking the
// plane forward never steps back.
func ScStream(q Quant) bool { qi := qtab[q]; return qi.biasArray && qi.perSuper == 8 }

// ScField places field (sub, min) of the stream: word w of the super-block's
// ScStreamWords, starting at bit b, with hi of its six bits in the low bits of
// word w+1 (0 when it does not straddle).
//
// A reader extracts it as (W[w] << (26-b)) >> 26 when hi is 0, and otherwise
// as (W[w] >> b) | ((W[w+1] << (32-hi)) >> 26).
func ScField(sub int, min bool) (w, b, hi int) {
	f := 2 * sub
	if min {
		f++
	}
	w, b = f*6/32, f*6%32
	if b > 26 {
		hi = b - 26
	}
	return w, b, hi
}

// ScValue is ScField's extraction in Go: the packer's inverse and the oracle's
// reader.
func ScValue(words [ScStreamWords]uint32, sub int, min bool) uint8 {
	w, b, hi := ScField(sub, min)
	if hi == 0 {
		return uint8(words[w] << uint(26-b) >> 26)
	}
	return uint8(words[w]>>uint(b) | words[w+1]<<uint(32-hi)>>26)
}

// scPut ORs a 6-bit field into a row's three stream words.
func scPut(words *[ScStreamWords]uint32, sub int, min bool, v uint8) {
	w, b, hi := ScField(sub, min)
	words[w] |= uint32(v&63) << uint(b)
	if hi > 0 {
		words[w+1] |= uint32(v&63) >> uint(6-hi)
	}
}

// scStreamAt locates sub-block subIdx's scale/min pair at run time, where the
// sub-block is not known when the kernel is built. A super-block is exactly
// ScStreamWords words (96 bits, eight 12-bit pairs), so a row's whole plane
// is one contiguous bit stream and sub-block i's pair starts at bit 12*i: word
// lo = 12*i/32 of the row, from bit `bit`, spilling into word hi = lo+1 only
// when it straddles (bit > 20). A pair that does not straddle reads hi = lo,
// whose bits scStreamPair shifts entirely away, and the tensor's last pair
// never straddles, so no index runs past the plane. (Splitting into
// super-block and word costs extra ops the issue-bound device matvec feels.)
func scStreamAt(b *ir.Builder, subIdx ir.Value) (lo, hi, bit ir.Value) {
	c := func(x int64) ir.Value { return b.Const(ir.U32, x) }
	pos := b.Mul(ir.U32, subIdx, c(12))
	lo, bit = b.Shr(ir.U32, pos, c(5)), b.And(ir.U32, pos, c(31))
	return lo, b.Add(ir.U32, lo, b.Select(ir.U32, b.Lt(ir.U32, c(20), bit), c(1), c(0))), bit
}

// scStreamPair is the funnel over the two words scStreamAt named: one shf.r on
// PTX (ir.OpFunnel).
func scStreamPair(b *ir.Builder, lo, hi, bit ir.Value) (sc, m ir.Value) {
	c := func(x int64) ir.Value { return b.Const(ir.U32, x) }
	pair := b.Funnel(lo, hi, bit)
	return b.And(ir.U32, pair, c(63)), b.And(ir.U32, b.Shr(ir.U32, pair, c(6)), c(63))
}
