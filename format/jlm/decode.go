package jlm

// The reading rules of a packed weight, stated once over an arithmetic the
// caller supplies: a Go evaluator's values are numbers (DecodeElement), the
// composer's are the nodes of a kernel, so the device's decode and the
// format's own reader are one definition. TestDecodeMatchesThePacker holds
// it to the packer's reference reader for every packed type.

// Plane is one of a packed tensor's three word arrays.
type Plane uint8

const (
	PlaneQS Plane = iota // the payload
	PlaneD               // the super-scales (and minimums)
	PlaneSC              // the per-sub-block scales (and minimums)
)

// Arith is the arithmetic the rules are written in. Integer operations are
// on u32 values, float operations on f32 ones; Word reads word i of a plane.
// A shift is always below 32. Bit fields are joined by addition: the rules
// only ever join fields that do not overlap, where it is or.
type Arith[V any] interface {
	U(uint32) V
	Add(a, b V) V
	Sub(a, b V) V
	Mul(a, b V) V
	Div(a, b V) V
	Rem(a, b V) V
	Min(a, b V) V
	Shr(a, b V) V
	Shl(a, b V) V
	And(a, b V) V
	Xor(a, b V) V
	Word(p Plane, i V) V
	// Code is entry q of a 16-entry code table.
	Code(t *[16]uint8, q V) V
	// Half is the low binary16 half of a u32 as a float, F32 its bits as a
	// float, Int it as a signed int32 converted to a float, Float it as an
	// unsigned one converted.
	Half(V) V
	F32(V) V
	Int(V) V
	Float(V) V
	FConst(float32) V
	FMul(a, b V) V
	FSub(a, b V) V
	FAdd(a, b V) V
}

// scStreamWords is the three words a super-block's six-bit scales and
// minimums take, eight sub-blocks of two fields each.
const scStreamWords = 3

// dSlots is how many rows' super-scales share a d word: four E8M0 bytes, two
// binary16 halves, or one row's whole word where a minimum or an f32 scale
// takes it.
func (l TypeLayout) dSlots() int {
	switch {
	case l.E8M0:
		return 4
	case l.HasMin || l.D32:
		return 1
	case l.SuperSub == 1:
		return 2
	}
	return 1
}

// scStream reports a format whose per-sub-block scales and minimums are six
// bits each in a stream of scStreamWords words a super-block.
func (l TypeLayout) scStream() bool { return l.HasMin && l.SuperSub == 8 }

// DecodeElement is element c of row r of a packed tensor of rows rows: its
// code, scaled, less its bias.
func DecodeElement[V any](a Arith[V], l TypeLayout, rows int, r, c V) V {
	u := a.U
	R := u(uint32(rows))
	sub, bits, hi := l.Sub, l.Planes[0], l.Planes[1]
	pw, hw := sub*bits/32, sub*hi/32
	si, e := a.Div(c, u(uint32(sub))), a.Rem(c, u(uint32(sub)))
	super := a.Div(si, u(uint32(l.SuperSub)))

	// The super-scale, and the minimum where the d word carries one.
	var scale, dmin V
	n := l.dSlots()
	slot := u(0)
	di := a.Add(a.Mul(super, R), r)
	if n > 1 {
		di = a.Add(a.Mul(super, u(uint32((rows+n-1)/n))), a.Div(r, u(uint32(n))))
		slot = a.Rem(r, u(uint32(n)))
	}
	dw := a.Word(PlaneD, di)
	switch {
	case n == 4:
		b := a.And(a.Shr(dw, a.Mul(slot, u(8))), u(0xff))
		scale = a.F32(a.Shl(b, u(23)))
	case n == 2:
		scale = a.Half(a.Shr(dw, a.Mul(slot, u(16))))
	case l.D32:
		scale = a.F32(dw)
	default:
		scale, dmin = a.Half(dw), a.Half(a.Shr(dw, u(16)))
	}

	// The sub-block's own scale and minimum, and the bias.
	var bias V
	switch {
	case l.scStream():
		s8 := a.Rem(si, u(8))
		field := func(min uint32) V {
			f := a.Add(a.Mul(s8, u(2)), u(min))
			bit := a.Mul(f, u(6))
			w, b := a.Div(bit, u(32)), a.Rem(bit, u(32))
			word := func(w V) V {
				return a.Word(PlaneSC, a.Add(a.Mul(a.Add(a.Mul(super, u(scStreamWords)), w), R), r))
			}
			lo := a.Shr(word(w), b)
			// A field past bit 26 takes its high bits from the next word;
			// shifted 32-b in two steps, so no shift reaches 32, and the
			// next word's index held inside the stream.
			next := word(a.Min(a.Add(w, u(1)), u(scStreamWords-1)))
			hiPart := a.Shl(a.Shl(next, u(1)), a.Sub(u(31), b))
			return a.And(a.Add(lo, hiPart), u(63))
		}
		scale = a.FMul(scale, a.Float(field(0)))
		bias = a.FMul(dmin, a.Float(field(1)))
	case l.SuperSub > 1:
		perWord, stride := 4, 8
		if l.HasMin {
			perWord, stride = 2, 16
		}
		sw := a.Word(PlaneSC, a.Add(a.Mul(a.Div(si, u(uint32(perWord))), R), r))
		sh := a.Mul(a.Rem(si, u(uint32(perWord))), u(uint32(stride)))
		sv := a.FAdd(a.Float(a.And(a.Shr(sw, sh), u(0xff))), a.FConst(float32(l.ScaleOff)))
		scale = a.FMul(scale, sv)
		if l.HasMin {
			bias = a.FMul(dmin, a.Float(a.And(a.Shr(sw, a.Add(sh, u(8))), u(0xff))))
		}
	case l.HasMin:
		bias = dmin
	}
	if !l.HasMin {
		bias = a.FMul(scale, a.FConst(l.Bias))
	}

	// The code: the primary plane's field, the secondary plane's bits above
	// it, through the code table where the format has one.
	base := a.Add(a.Mul(si, u(uint32((pw+hw)*rows))), r)
	var q V
	if bits < 8 {
		fields := 8 / bits
		h := sub / fields
		f, rem := a.Div(e, u(uint32(h))), a.Rem(e, u(uint32(h)))
		w, b := a.Div(rem, u(4)), a.Rem(rem, u(4))
		v := a.Word(PlaneQS, a.Add(base, a.Mul(w, R)))
		sh := a.Add(a.Mul(b, u(8)), a.Mul(f, u(uint32(bits))))
		q = a.And(a.Shr(v, sh), u(uint32(1)<<bits-1))
		if l.Codes != nil {
			q = a.Code(l.Codes, q)
		}
		if hi > 0 {
			lanes := 8 / hi
			g := a.Div(e, u(4))
			hv := a.Word(PlaneQS, a.Add(base, a.Mul(a.Add(u(uint32(pw)), a.Div(g, u(uint32(lanes)))), R)))
			hsh := a.Add(a.Mul(a.Rem(e, u(4)), u(8)), a.Mul(a.Rem(g, u(uint32(lanes))), u(uint32(hi))))
			hb := a.And(a.Shr(hv, hsh), u(uint32(1)<<hi-1))
			q = a.Add(q, a.Shl(hb, u(uint32(bits))))
		}
		return a.FSub(a.FMul(scale, a.Float(q)), bias)
	}
	v := a.Word(PlaneQS, a.Add(base, a.Mul(a.Div(e, u(4)), R)))
	by := a.And(a.Shr(v, a.Mul(a.Rem(e, u(4)), u(8))), u(0xff))
	if l.Signed {
		// The byte's sign extended: its top bit flipped, 128 taken away.
		return a.FSub(a.FMul(scale, a.Int(a.Sub(a.Xor(by, u(0x80)), u(0x80)))), bias)
	}
	return a.FSub(a.FMul(scale, a.Float(by)), bias)
}

// Words is how many words each plane of a packed tensor of rows rows of k
// holds: the payload, the super-scales, the per-sub-block scales.
func (l TypeLayout) Words(rows, k int) (qs, d, sc int) {
	nsub := k / l.Sub
	qs = rows * nsub * (l.Sub * (l.Planes[0] + l.Planes[1]) / 32)
	nsuper := nsub / l.SuperSub
	if n := l.dSlots(); n > 1 {
		d = nsuper * ((rows + n - 1) / n)
	} else {
		d = nsuper * rows
	}
	switch {
	case l.scStream():
		sc = rows * nsuper * scStreamWords
	case l.SuperSub > 1:
		per := 4
		if l.HasMin {
			per = 2
		}
		sc = rows * ((nsub + per - 1) / per)
	}
	return qs, d, sc
}

// DecodeVerbatim is element c of row r of a verbatim tensor of k columns:
// row-major values, an f32 a word, an f16 or a bf16 a half of one.
func DecodeVerbatim[V any](a Arith[V], t Type, k int, r, c V) V {
	u := a.U
	i := a.Add(a.Mul(r, u(uint32(k))), c)
	if t == TypeF32 {
		return a.F32(a.Word(PlaneQS, i))
	}
	h := a.Shr(a.Word(PlaneQS, a.Div(i, u(2))), a.Mul(a.Rem(i, u(2)), u(16)))
	if t == TypeBF16 {
		return a.F32(a.Shl(a.And(h, u(0xffff)), u(16)))
	}
	return a.Half(h)
}

// Verbatim reports a type DecodeVerbatim reads.
func Verbatim(t Type) bool { return t == TypeF32 || t == TypeF16 || t == TypeBF16 }
