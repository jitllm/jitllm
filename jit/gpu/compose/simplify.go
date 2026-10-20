package compose

import (
	"math"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// Simplify is the composer's IR pass on k: constants folded (integer and
// float arithmetic of constants, a binary16 pack of two, a conversion or
// reinterpretation of one), identities removed, an addend that is a
// multiple of a power-of-two modulus dropped from what the remainder reads,
// common subexpressions merged and what no effect reads removed.
func Simplify(k *ir.Kernel) *ir.Kernel { return simplify(k) }

// simplify is k after the IR passes the former leaves to them: integer
// constants folded, common subexpressions merged (cse) and identities removed (x+0, x*1, x*0, x%1, x-x, min(x, 0)), a split index
// recombined ((f/c)*c + f%c is f), an add of a constant whose one reader is
// an access taken as that access's offset, and what no effect reads removed.
func simplify(k *ir.Kernel) *ir.Kernel {
	ops := append([]ir.Op(nil), k.Ops...)
	for i := range ops {
		ops[i].Vargs = append([]ir.Value(nil), ops[i].Vargs...)
	}
	repl := make([]ir.Value, len(ops)+1)
	res := func(v ir.Value) ir.Value {
		for v != 0 && repl[v] != 0 {
			v = repl[v]
		}
		return v
	}
	intConst := func(v ir.Value) (uint32, bool) {
		o := &ops[v-1]
		return uint32(o.Imm), o.Kind == ir.OpConst && (o.Type == ir.U32 || o.Type == ir.I32)
	}
	backEdges := func() {
		for i := range ops {
			if o := &ops[i]; o.Kind == ir.OpPhi && o.Args[1] != 0 {
				o.Args[1] = res(o.Args[1])
			}
		}
	}
	fold := func() {
		for i := range ops {
			o := &ops[i]
			for j, a := range o.Args {
				if a != 0 && (o.Kind != ir.OpPhi || j == 0) {
					o.Args[j] = res(a)
				}
			}
			for j, a := range o.Vargs {
				o.Vargs[j] = res(a)
			}
			if c, ok := foldConst(ops, o); ok {
				*o = ir.Op{Kind: ir.OpConst, Type: o.Type, Imm: c}
				continue
			}
			if o.Type != ir.U32 && o.Type != ir.I32 {
				continue
			}
			peelMultiples(ops, o)
			if v, ok := foldInt(o, intConst); ok {
				if v != 0 {
					repl[i+1] = v
				} else {
					*o = ir.Op{Kind: ir.OpConst, Type: o.Type, Imm: o.Imm}
				}
				continue
			}
		}
		backEdges()
	}
	fold()
	cse(ops, repl, res)
	backEdges()
	fold()
	readers := make([]int, len(ops)+1)
	live := liveOf(ops)
	for i, o := range ops {
		if !live[i+1] {
			continue
		}
		for _, a := range o.Args {
			readers[a]++
		}
	}
	for i := range ops {
		o := &ops[i]
		for j, a := range o.Args {
			if a != 0 && (o.Kind != ir.OpPhi || j == 0) {
				o.Args[j] = res(a)
			}
		}
		if f, ok := recombined(ops, o, readers); ok {
			repl[i+1] = f
		}
	}
	backEdges()
	for i := range ops {
		o := &ops[i]
		for j, a := range o.Args {
			if a != 0 && (o.Kind != ir.OpPhi || j == 0) {
				o.Args[j] = res(a)
			}
		}
		for j, a := range o.Vargs {
			o.Vargs[j] = res(a)
		}
	}
	// An add of a constant whose one live reader is an access's index is that
	// access's offset; one several read, or a cursor read by its next step,
	// stays an add. Rounds until nothing moves: a sum of constants peels a
	// level a round.
	for moved := true; moved; {
		moved = false
		live := liveOf(ops)
		readers := make([]int, len(ops)+1)
		for i, o := range ops {
			if !live[i+1] {
				continue
			}
			for _, a := range o.Args {
				readers[a]++
			}
			for _, a := range o.Vargs {
				readers[a]++
			}
		}
		for i := range ops {
			o := &ops[i]
			if o.Kind != ir.OpLoad && o.Kind != ir.OpStore || !live[i+1] {
				continue
			}
			a := o.Args[1]
			ao := &ops[a-1]
			if ao.Kind != ir.OpAdd || ao.Type != ir.U32 || readers[a] != 1 {
				continue
			}
			x, c := ao.Args[0], ao.Args[1]
			if v, ok := intConst(c); ok && int32(v) >= 0 {
				o.Args[1], o.Imm, moved = x, o.Imm+int64(v), true
			} else if v, ok := intConst(x); ok && int32(v) >= 0 {
				o.Args[1], o.Imm, moved = c, o.Imm+int64(v), true
			}
		}
	}
	return compact(k, ops)
}

// foldInt is integer op o without its instruction: a value it equals
// (non-zero), or o's constant in o.Imm (zero).
func foldInt(o *ir.Op, intConst func(ir.Value) (uint32, bool)) (ir.Value, bool) {
	switch o.Kind {
	case ir.OpAdd, ir.OpSub, ir.OpMul, ir.OpDiv, ir.OpRem, ir.OpMin, ir.OpAnd, ir.OpXor, ir.OpShl, ir.OpShr:
	default:
		return 0, false
	}
	x, y := o.Args[0], o.Args[1]
	if x == y && (o.Kind == ir.OpSub || o.Kind == ir.OpXor) {
		o.Imm = 0
		return 0, true
	}
	cx, kx := intConst(x)
	cy, ky := intConst(y)
	signed := o.Type == ir.I32
	if kx && ky {
		var r uint32
		switch o.Kind {
		case ir.OpAdd:
			r = cx + cy
		case ir.OpSub:
			r = cx - cy
		case ir.OpMul:
			r = cx * cy
		case ir.OpDiv, ir.OpRem:
			if cy == 0 || signed {
				return 0, false
			}
			if r = cx / cy; o.Kind == ir.OpRem {
				r = cx % cy
			}
		case ir.OpMin:
			if signed {
				return 0, false
			}
			r = min(cx, cy)
		case ir.OpAnd:
			r = cx & cy
		case ir.OpXor:
			r = cx ^ cy
		case ir.OpShl:
			r = cx << (cy & 31)
		case ir.OpShr:
			if signed {
				return 0, false
			}
			r = cx >> (cy & 31)
		}
		o.Imm = int64(r)
		return 0, true
	}
	switch o.Kind {
	case ir.OpAdd, ir.OpXor:
		if ky && cy == 0 {
			return x, true
		}
		if kx && cx == 0 {
			return y, true
		}
	case ir.OpSub, ir.OpShl, ir.OpShr:
		if ky && cy == 0 {
			return x, true
		}
	case ir.OpMul:
		switch {
		case ky && cy == 1:
			return x, true
		case kx && cx == 1:
			return y, true
		case kx && cx == 0, ky && cy == 0:
			o.Imm = 0
			return 0, true
		}
	case ir.OpDiv:
		if ky && cy == 1 {
			return x, true
		}
	case ir.OpRem:
		if ky && cy == 1 {
			o.Imm = 0
			return 0, true
		}
	case ir.OpMin:
		if !signed && (kx && cx == 0 || ky && cy == 0) {
			o.Imm = 0
			return 0, true
		}
	}
	return 0, false
}

// recombined is f where o is (f/c)*c + f%c in either order, for a constant
// c: the index a row and column split was taken from, where the split
// serves nothing else (the quotient's one reader the product).
func recombined(ops []ir.Op, o *ir.Op, readers []int) (ir.Value, bool) {
	if o.Kind != ir.OpAdd || o.Type != ir.U32 {
		return 0, false
	}
	at := func(v ir.Value) *ir.Op { return &ops[v-1] }
	for _, ab := range [2][2]ir.Value{{o.Args[0], o.Args[1]}, {o.Args[1], o.Args[0]}} {
		m, r := at(ab[0]), at(ab[1])
		if m.Kind != ir.OpMul || r.Kind != ir.OpRem {
			continue
		}
		for _, dc := range [2][2]ir.Value{{m.Args[0], m.Args[1]}, {m.Args[1], m.Args[0]}} {
			d := at(dc[0])
			if d.Kind == ir.OpDiv && readers[dc[0]] == 1 && d.Args[0] == r.Args[0] && sameConst(ops, d.Args[1], dc[1]) && sameConst(ops, r.Args[1], dc[1]) {
				return r.Args[0], true
			}
		}
	}
	return 0, false
}

func sameConst(ops []ir.Op, a, b ir.Value) bool {
	x, y := &ops[a-1], &ops[b-1]
	return a == b || x.Kind == ir.OpConst && y.Kind == ir.OpConst && x.Type == y.Type && x.Imm == y.Imm
}

// liveOf is which ops an effect reads, transitively (by value, 1-based).
func liveOf(ops []ir.Op) []bool {
	live := make([]bool, len(ops)+1)
	var mark func(v ir.Value)
	mark = func(v ir.Value) {
		if v == 0 || live[v] {
			return
		}
		live[v] = true
		o := &ops[v-1]
		for _, a := range o.Args {
			mark(a)
		}
		for _, a := range o.Vargs {
			mark(a)
		}
	}
	for i, o := range ops {
		if hasEffect(o.Kind) {
			mark(ir.Value(i + 1))
		}
	}
	return live
}

// compact is k's ops without those no effect reads, renumbered.
func compact(k *ir.Kernel, ops []ir.Op) *ir.Kernel {
	live := liveOf(ops)
	at := make([]ir.Value, len(ops)+1)
	out := *k
	out.Ops = make([]ir.Op, 0, len(ops))
	for i, o := range ops {
		if !live[i+1] {
			continue
		}
		out.Ops = append(out.Ops, o)
		at[i+1] = ir.Value(len(out.Ops))
	}
	for i := range out.Ops {
		o := &out.Ops[i]
		for j, a := range o.Args {
			o.Args[j] = at[a]
		}
		for j, a := range o.Vargs {
			o.Vargs[j] = at[a]
		}
	}
	return &out
}

// hasEffect is whether an op of kind k does something beyond its value: it
// stays whatever reads it.
func hasEffect(k ir.Kind) bool {
	switch k {
	case ir.OpStore, ir.OpStoreV, ir.OpTileStore, ir.OpBarrier, ir.OpGridBarrier, ir.OpLoop, ir.OpEndLoop,
		ir.OpParam, ir.OpShared, ir.OpPrefetch, ir.OpDepTask, ir.OpDepEpoch, ir.OpDepWait, ir.OpDepSignal, ir.OpDepExit:
		return true
	}
	return false
}

// cseKey is a pure op by what it computes.
type cseKey struct {
	kind ir.Kind
	typ  ir.Type
	args [3]ir.Value
	imm  int64
}

// cse is common-subexpression elimination over k's pure ops: a later op
// computing what an earlier one visible to it does reads that one instead
// (repl), a load of a buffer the kernel does not write among them. A value made in a loop's body is visible until the loop ends.
func cse(ops []ir.Op, repl []ir.Value, res func(ir.Value) ir.Value) {
	// A buffer the kernel never stores to reads the same at every load
	// (RULE 13: no kernel reads what it writes within a phase).
	stored := map[ir.Value]bool{}
	for _, o := range ops {
		if o.Kind == ir.OpStore || o.Kind == ir.OpStoreV || o.Kind == ir.OpTileStore {
			stored[res(o.Args[0])] = true
		}
	}
	seen := map[cseKey]ir.Value{}
	var scopes [][]cseKey
	for i := range ops {
		o := &ops[i]
		for j, a := range o.Args {
			if a != 0 && (o.Kind != ir.OpPhi || j == 0) {
				o.Args[j] = res(a)
			}
		}
		for j, a := range o.Vargs {
			o.Vargs[j] = res(a)
		}
		switch o.Kind {
		case ir.OpLoop:
			scopes = append(scopes, nil)
			continue
		case ir.OpEndLoop:
			if n := len(scopes); n > 0 {
				for _, k := range scopes[n-1] {
					delete(seen, k)
				}
				scopes = scopes[:n-1]
			}
			continue
		}
		load := o.Kind == ir.OpLoad && ops[o.Args[0]-1].Kind == ir.OpParam && !stored[o.Args[0]]
		if !pure(o.Kind) && !load || len(o.Vargs) > 0 || o.Tile != nil {
			continue
		}
		k := cseKey{o.Kind, o.Type, o.Args, o.Imm}
		if v, ok := seen[k]; ok {
			repl[i+1] = v
			continue
		}
		seen[k] = ir.Value(i + 1)
		if n := len(scopes); n > 0 {
			scopes[n-1] = append(scopes[n-1], k)
		}
	}
}

// pure is whether an op of kind k computes from its operands alone.
func pure(k ir.Kind) bool {
	switch k {
	case ir.OpConst, ir.OpTID, ir.OpCTAID, ir.OpNTID, ir.OpAdd, ir.OpSub, ir.OpMul, ir.OpDiv, ir.OpRem, ir.OpShl,
		ir.OpShr, ir.OpAnd, ir.OpXor, ir.OpMin, ir.OpMax, ir.OpExp, ir.OpSqrt, ir.OpCvtF32, ir.OpCvtF16H,
		ir.OpPackF16, ir.OpCvtI32, ir.OpBitcast, ir.OpDot4, ir.OpFma, ir.OpLt, ir.OpSelect, ir.OpFunnel, ir.OpSubF16x2, ir.OpFmaF16x2:
		return true
	}
	return false
}

// foldConst is op o's constant where every operand is one and o is
// float arithmetic, a binary16 pack, a conversion to float or a
// reinterpretation: its value's bits, exactly as the device computes it.
func foldConst(ops []ir.Op, o *ir.Op) (int64, bool) {
	arg := func(j int) (ir.Op, bool) {
		a := o.Args[j]
		if a == 0 {
			return ir.Op{}, false
		}
		x := ops[a-1]
		return x, x.Kind == ir.OpConst
	}
	f := func(x ir.Op) float32 { return math.Float32frombits(uint32(x.Imm)) }
	bits := func(v float32) int64 { return int64(math.Float32bits(v)) }
	x, okx := arg(0)
	switch o.Kind {
	case ir.OpBitcast:
		if okx {
			return int64(uint32(x.Imm)), true
		}
	case ir.OpCvtF32:
		if okx && o.Type == ir.F32 {
			if x.Type == ir.I32 {
				return bits(float32(int32(x.Imm))), true
			}
			if x.Type == ir.U32 {
				return bits(float32(uint32(x.Imm))), true
			}
		}
	case ir.OpPackF16:
		y, oky := arg(1)
		if okx && oky && x.Type == ir.F32 && y.Type == ir.F32 {
			return int64(uint32(quant.EncodeHalf(f(x))) | uint32(quant.EncodeHalf(f(y)))<<16), true
		}
	case ir.OpAdd, ir.OpSub, ir.OpMul:
		y, oky := arg(1)
		if !okx || !oky || o.Type != ir.F32 || x.Type != ir.F32 || y.Type != ir.F32 {
			return 0, false
		}
		a, b := f(x), f(y)
		switch o.Kind {
		case ir.OpAdd:
			return bits(a + b), true
		case ir.OpSub:
			return bits(a - b), true
		}
		return bits(a * b), true
	}
	return 0, false
}

// peelMultiples drops from what an unsigned remainder by a power of two
// (or a mask of one less) reads each addend that is a multiple of it.
func peelMultiples(ops []ir.Op, o *ir.Op) {
	if o.Type != ir.U32 || o.Args[0] == 0 || o.Args[1] == 0 {
		return
	}
	c := ops[o.Args[1]-1]
	if c.Kind != ir.OpConst {
		return
	}
	m := uint64(0)
	switch v := uint64(uint32(c.Imm)); {
	case o.Kind == ir.OpRem && v > 0 && v&(v-1) == 0:
		m = v
	case o.Kind == ir.OpAnd && v != 0 && (v+1)&v == 0:
		m = v + 1
	default:
		return
	}
	multiple := func(a ir.Value) bool {
		x := ops[a-1]
		k := func(j int) (uint64, bool) {
			y := ops[x.Args[j]-1]
			return uint64(uint32(y.Imm)), x.Args[j] != 0 && y.Kind == ir.OpConst
		}
		switch x.Kind {
		case ir.OpConst:
			return uint64(uint32(x.Imm))%m == 0
		case ir.OpMul:
			a, oka := k(0)
			b, okb := k(1)
			return oka && a%m == 0 || okb && b%m == 0
		case ir.OpShl:
			b, okb := k(1)
			return okb && b < 32 && (uint64(1)<<b)%m == 0
		}
		return false
	}
	for x := ops[o.Args[0]-1]; x.Kind == ir.OpAdd && x.Type == ir.U32; x = ops[o.Args[0]-1] {
		switch {
		case multiple(x.Args[1]):
			o.Args[0] = x.Args[0]
		case multiple(x.Args[0]):
			o.Args[0] = x.Args[1]
		default:
			return
		}
	}
}
