//go:build amd64

package cpu

// elemLoop emits an elementwise kernel's loop: body over K whole vectors,
// then over R single elements, where K and R are registers holding n/8 and
// n%8. Either may be zero.
//
// Every width is generated code: the tail runs the same body on a register
// holding one element in lane 0, with ld and st as a scalar load (which zeroes
// the other lanes) and a scalar store, so nothing is read or written past the
// last element and the arithmetic is the vector path's own.
//
// body must do all its memory access through ld and st, never through a
// memory operand -- a 32-byte operand in the tail reads past the end.
func elemLoop(a *Buf, k, r Reg, cursors []Reg, body func(ld func(dst, p Reg), st func(p, src Reg))) {
	vecLd := func(dst, p Reg) { a.VMOVDQULoad(dst, At(p, 0)) }
	vecSt := func(p, src Reg) { a.VMOVDQUStore(At(p, 0), src) }
	oneLd := func(dst, p Reg) { a.VMOVSSLoad(dst, At(p, 0)) }
	oneSt := func(p, src Reg) { a.VMOVSSStore(At(p, 0), src) }
	elemLoops(a, k, r, cursors, func() { body(vecLd, vecSt) }, func() { body(oneLd, oneSt) })
}

// elemLoops is elemLoop with the two bodies written separately, for a kernel
// whose tail cannot be the vector body on one lane -- a reduction, where the
// other seven lanes must be neutral for the operation rather than zero.
func elemLoops(a *Buf, k, r Reg, cursors []Reg, vec, one func()) {
	tail, done := a.Label(), a.Label()
	a.TESTQ(k, k)
	a.JZ(tail)
	lp := a.Label()
	a.Bind(lp)
	vec()
	for _, c := range cursors {
		a.ADDimm(c, 32)
	}
	a.DEC(k)
	a.JNZ(lp)

	a.Bind(tail)
	a.TESTQ(r, r)
	a.JZ(done)
	lt := a.Label()
	a.Bind(lt)
	one()
	for _, c := range cursors {
		a.ADDimm(c, 4)
	}
	a.DEC(r)
	a.JNZ(lt)
	a.Bind(done)
}
