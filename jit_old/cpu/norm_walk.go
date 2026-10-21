package cpu

// rowWalk is the fixed-width walk every host row-reduction emitter shares: n is
// baked, so the trip counts and the tails are compile-time. It writes the
// control structure and nothing else -- the instructions of each step, and the
// registers they use, are the caller's.
//
//	body    block elements a trip, looped when n holds a whole block. The loop
//	        is guarded by the caller's count (enter is never reached for zero
//	        blocks): a do-while whose counter starts at zero would run the body
//	        once and then 2^64 times.
//	vec     step elements, baked once for each whole step the blocks leave.
//	scalar  one element, baked for each of the n%step the steps leave.
//
// advance moves every cursor the pass reads or writes by the given number of
// elements; run calls it after body (inside the loop) and after each vec, in
// the order the cursors were listed.
type rowWalk struct {
	n, block, step int
	enter          func(blocks int)
	leave          func()
	advance        func(elems int)
}

func (w rowWalk) run(body, vec func(), scalar func(i int)) {
	if blocks := w.n / w.block; blocks > 0 {
		w.enter(blocks)
		body()
		w.advance(w.block)
		w.leave()
	}
	for off := (w.n / w.block) * w.block; off+w.step <= w.n; off += w.step {
		vec()
		w.advance(w.step)
	}
	for i := 0; i < w.n%w.step; i++ {
		scalar(i)
	}
}

// newRowWalk walks n floats on amd64, AVX2 or SSE alike: eight floats a step,
// four steps a block, R10 the trip counter and cursors the registers advanced.
func newRowWalk(a *Buf, n int, cursors ...Reg) rowWalk {
	var lp int
	return rowWalk{
		n: n, block: 32, step: 8,
		enter: func(blocks int) {
			a.MOVimm(R10, int64(blocks))
			lp = a.Label()
			a.Bind(lp)
		},
		leave: func() {
			a.DEC(R10)
			a.JNZ(lp)
		},
		advance: func(elems int) {
			for _, c := range cursors {
				a.ADDimm(c, int32(4*elems))
			}
		},
	}
}

// newA64RowWalk walks n floats on arm64: four floats a step, four steps a
// block, X4 the trip counter and cursors the registers advanced.
func newA64RowWalk(a *A64, n int, cursors ...XReg) rowWalk {
	var lp int
	return rowWalk{
		n: n, block: 16, step: 4,
		enter: func(blocks int) {
			a.MOVimm(X4, int64(blocks))
			lp = a.Label()
			a.Bind(lp)
		},
		leave: func() {
			a.SUBimm(X4, X4, 1)
			a.CBNZ(X4, lp)
		},
		advance: func(elems int) {
			for _, c := range cursors {
				a.ADDimm(c, c, int32(4*elems))
			}
		},
	}
}
