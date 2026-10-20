// Package paritytest holds main's kernel generators and tier (mainkernels,
// maintier) as the reference the composer converges to, and the comparison
// of the two: Compare holds main's kernels' instructions to the composer's by
// what each computes. Nothing outside this directory but tests imports it.
package paritytest

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// Miss is one of main's instructions the composer's kernels do not hold:
// the kernel and the instruction's index in it.
type Miss struct {
	K  *ir.Kernel
	Op int
}

// Compare looks for each value main's kernels compute among the composer's
// by its dataflow (Dataflow): each of a kernel's live instructions (liveOps),
// a value computed twice in one kernel counted once, and an instruction of
// the composer's matching one of main's at most. Constants are not counted
// (constant): a constant op or constant-valued arithmetic is an immediate. It returns how many of
// main's values there are (parameters aside), how many match, and the rest.
func Compare(main, composed []*ir.Kernel) (total, same int, missing []Miss) {
	in := NewInterner()
	have := map[int32]int{}
	for _, k := range composed {
		for _, id := range in.Dataflow(k) {
			have[id]++
		}
	}
	for _, k := range main {
		live := liveOps(k)
		// A value a kernel computes twice is one value: main's builder
		// eliminates no common subexpression, and recomputing is no part of
		// what it computes.
		seen := map[int32]bool{}
		for i, id := range in.Dataflow(k) {
			if k.Ops[i].Kind == ir.OpParam || !live[i] || seen[id] || in.constant(k, i, id) {
				continue
			}
			seen[id] = true
			total++
			if have[id] > 0 {
				have[id]--
				same++
				continue
			}
			missing = append(missing, Miss{K: k, Op: i})
		}
	}
	return total, same, missing
}

// liveOps is which of k's instructions an effect reaches: the void ones
// (stores, barriers, the loops) and every operand they read, transitively.
// What nothing reads computes nothing, so it is not held to.
func liveOps(k *ir.Kernel) []bool {
	live := make([]bool, len(k.Ops))
	var mark func(v ir.Value)
	mark = func(v ir.Value) {
		if v == 0 || int(v) > len(k.Ops) || live[v-1] {
			return
		}
		live[v-1] = true
		o := k.Ops[v-1]
		for _, a := range o.Args {
			mark(a)
		}
		for _, a := range o.Vargs {
			mark(a)
		}
	}
	for i, o := range k.Ops {
		if o.Type == ir.Void {
			mark(ir.Value(i + 1))
		}
	}
	return live
}

// Interner names each distinct canonical form by a small integer, so equal
// forms in any of the kernels it saw get equal names.
type Interner struct {
	ids map[form]int32
	// fields is each field atom's parts.
	fields map[int32]field
	// desc is each atom's description, for Explain.
	desc map[int32]string
	// affs is each named affine form, by its name.
	affs map[int32]affine
}

// NewInterner is an empty interner.
func NewInterner() *Interner {
	return &Interner{ids: map[form]int32{}, desc: map[int32]string{}, fields: map[int32]field{}, affs: map[int32]affine{}}
}

// form is one canonical form: an instruction by its kind, type, loop depth,
// immediate, tile and its operands' names; or a piece of an affine index.
type form struct {
	tag  uint8 // fInstr, fAffine, fTerm, fAtom
	kind ir.Kind
	typ  ir.Type
	d    int16
	imm  int64
	tile ir.TileType
	a    [4]int32
}

const (
	fInstr uint8 = iota
	fAffine
	fTerm
	fAtom
	fVargs
	fBack
)

func (in *Interner) name(f form) int32 {
	if id, ok := in.ids[f]; ok {
		return id
	}
	id := int32(len(in.ids)) + 1
	in.ids[f] = id
	return id
}

// The atoms an index is affine over.
const (
	atomTID = iota + 1
	atomCTAID
	atomNTID
	atomIter // a loop's iteration, by its depth and trip count
	atomProd // a product of two atoms
	atomOpaque
	// atomField is floor((x mod m) / d) of an index x, m 0 for none: a digit
	// of x in mixed radix, which nested divisions and remainders by
	// constants all reduce to.
	atomField
	// atomCarried stands for a phi while its back edge is evaluated: the
	// edge as carried + c makes the phi an induction of step c.
	atomCarried
)

// field is a field atom's x (an affine's name), d and m.
type field struct {
	x    int32
	d, m int64
}

// Dataflow is each of k's instructions' name by what it computes: its kind,
// type, tile and loop depth and its operands' names, with integer arithmetic
// canonical -- an index is an affine form over the thread's id, its group's,
// the group width and each loop's iteration (a phi stepped by a constant is
// its start plus the step times the iteration), a load's or store's
// immediate folded into its address, a quotient or remainder a bound proves
// trivial simplified -- so two kernels agree where they read the same
// elements, compute the same floats and write the same places, whatever
// index arithmetic got them there and in whatever order. A float carried
// value enters by its kind alone. A parameter is its element type.
func (in *Interner) Dataflow(k *ir.Kernel) []int32 {
	c := newCanon(k, in)
	out := make([]int32, len(k.Ops))
	for i := range k.Ops {
		out[i] = c.name(i)
	}
	return out
}

// constant is whether value id is a constant: a constant op, or integer
// arithmetic whose canonical form is one -- a value any compiler folds, an
// immediate rather than an instruction.
func (in *Interner) constant(k *ir.Kernel, i int, id int32) bool {
	if k.Ops[i].Kind == ir.OpConst {
		return true
	}
	a, ok := in.affs[id]
	return ok && a.isConst()
}

// affine is c + sum coef*atom, its terms sorted by atom.
type affine struct {
	c     int64
	terms []term
}

type term struct {
	atom int32
	coef int64
}

func (a affine) add(b affine, sign int64) affine {
	o := affine{c: a.c + sign*b.c}
	i, j := 0, 0
	for i < len(a.terms) || j < len(b.terms) {
		switch {
		case j == len(b.terms) || i < len(a.terms) && a.terms[i].atom < b.terms[j].atom:
			o.terms = append(o.terms, a.terms[i])
			i++
		case i == len(a.terms) || b.terms[j].atom < a.terms[i].atom:
			o.terms = append(o.terms, term{b.terms[j].atom, sign * b.terms[j].coef})
			j++
		default:
			if v := a.terms[i].coef + sign*b.terms[j].coef; v != 0 {
				o.terms = append(o.terms, term{a.terms[i].atom, v})
			}
			i++
			j++
		}
	}
	return o
}

func (a affine) scale(m int64) affine {
	o := affine{c: a.c * m}
	if m != 0 {
		for _, t := range a.terms {
			o.terms = append(o.terms, term{t.atom, t.coef * m})
		}
	}
	return o
}

func (a affine) isConst() bool { return len(a.terms) == 0 }

type canon struct {
	in    *Interner
	k     *ir.Kernel
	depth []int16
	loop  []int // the loop op each op sits in, -1 outside
	aff   []*affine
	ids   []int32
	busy  []bool
}

func newCanon(k *ir.Kernel, in *Interner) *canon {
	n := len(k.Ops)
	c := &canon{in: in, k: k, depth: make([]int16, n), loop: make([]int, n), aff: make([]*affine, n),
		ids: make([]int32, n), busy: make([]bool, n)}
	var stack []int
	for i, o := range k.Ops {
		if o.Kind == ir.OpEndLoop && len(stack) > 0 {
			stack = stack[:len(stack)-1]
		}
		c.depth[i] = int16(len(stack))
		c.loop[i] = -1
		if len(stack) > 0 {
			c.loop[i] = stack[len(stack)-1]
		}
		if o.Kind == ir.OpLoop {
			stack = append(stack, i)
		}
	}
	return c
}

func (c *canon) atom(kind int64, a, b int32) affine {
	id := c.in.name(form{tag: fAtom, imm: kind, a: [4]int32{a, b}})
	if _, ok := c.in.desc[id]; !ok {
		switch kind {
		case atomTID:
			c.in.desc[id] = "tid"
		case atomCTAID:
			c.in.desc[id] = "ctaid"
		case atomNTID:
			c.in.desc[id] = "ntid"
		case atomIter:
			c.in.desc[id] = fmt.Sprintf("iter(d%d,t%d)", a, b)
		case atomProd:
			c.in.desc[id] = c.in.desc[a] + "*" + c.in.desc[b]
		default:
			c.in.desc[id] = fmt.Sprintf("op%d", a)
		}
	}
	return affine{terms: []term{{id, 1}}}
}

func (c *canon) fieldAtom(f field) affine {
	if f.m == 1 {
		return affine{}
	}
	if f.d == 1 && f.m == 0 {
		if x, ok := c.in.affs[f.x]; ok {
			return x
		}
	}
	id := c.in.name(form{tag: fAtom, imm: atomField, a: [4]int32{f.x, int32(f.d), int32(f.m)}})
	if _, ok := c.in.fields[id]; !ok {
		c.in.fields[id] = f
		c.in.desc[id] = fmt.Sprintf("field(v%d,/%d,%%%d)", f.x, f.d, f.m)
	}
	return affine{terms: []term{{id, 1}}}
}

func (c *canon) affineName(a affine) int32 {
	id := c.in.name(form{tag: fAffine, imm: a.c})
	for _, t := range a.terms {
		id = c.in.name(form{tag: fTerm, imm: t.coef, a: [4]int32{id, t.atom}})
	}
	c.in.affs[id] = a
	return id
}

// merge is a with its digits of one dividend recombined: coef*field(x,/d,%m)
// beside coef*(m/d)*field(x,/m,%M) is coef*field(x,/d,%M), and a field
// x/1 with no modulus is x itself.
func (c *canon) merge(a affine) affine {
	for changed := true; changed; {
		changed = false
		for i := 0; i < len(a.terms) && !changed; i++ {
			f1, ok1 := c.in.fields[a.terms[i].atom]
			if !ok1 || f1.m == 0 {
				continue
			}
			for j := 0; j < len(a.terms) && !changed; j++ {
				f2, ok2 := c.in.fields[a.terms[j].atom]
				if i == j || !ok2 || f2.x != f1.x || f2.d != f1.m || f1.m%f1.d != 0 ||
					a.terms[j].coef != a.terms[i].coef*(f1.m/f1.d) {
					continue
				}
				coef := a.terms[i].coef
				var rest affine
				rest.c = a.c
				for k, t := range a.terms {
					if k != i && k != j {
						rest.terms = append(rest.terms, t)
					}
				}
				a = rest.add(c.fieldAtom(field{x: f1.x, d: f1.d, m: f2.m}).scale(coef), 1)
				changed = true
			}
		}
		for i, t := range a.terms {
			f, ok := c.in.fields[t.atom]
			if changed || !ok || f.m != 0 || f.d <= 1 || t.coef%f.d != 0 {
				continue
			}
			x, ok := c.in.affs[f.x]
			k := -t.coef / f.d
			if !ok || x.isConst() {
				continue
			}
			var rest affine
			rest.c = a.c
			for j, u := range a.terms {
				if j != i {
					rest.terms = append(rest.terms, u)
				}
			}
			r := rest.add(x.scale(k), -1)
			if len(r.terms) > len(rest.terms)-len(x.terms) {
				continue
			}
			// x - d*(x/d) is x%d.
			a = r.add(c.fieldAtom(field{x: f.x, d: 1, m: f.d}).scale(k), 1)
			changed = true
		}
		for i, t := range a.terms {
			if changed {
				break
			}
			if f, ok := c.in.fields[t.atom]; ok && f.d == 1 && f.m == 0 {
				if x, ok := c.in.affs[f.x]; ok {
					var rest affine
					rest.c = a.c
					for k, u := range a.terms {
						if k != i {
							rest.terms = append(rest.terms, u)
						}
					}
					a = rest.add(x.scale(t.coef), 1)
					changed = true
					break
				}
			}
		}
	}
	return a
}

func isInt(t ir.Type) bool { return t == ir.U32 || t == ir.I32 }

// index is op i as an affine form, false for an op that is not integer or a
// carried value that is no induction.
func (c *canon) index(i int) (affine, bool) {
	if a := c.aff[i]; a != nil {
		return *a, true
	}
	o := c.k.Ops[i]
	if !isInt(o.Type) || c.busy[i] {
		return affine{}, false
	}
	c.busy[i] = true
	arg := func(j int) (affine, bool) {
		if o.Args[j] == 0 {
			return affine{}, false
		}
		return c.index(int(o.Args[j]) - 1)
	}
	var r affine
	ok := true
	switch o.Kind {
	case ir.OpConst:
		r = affine{c: o.Imm}
	case ir.OpTID:
		r = c.atom(atomTID, 0, 0)
	case ir.OpCTAID:
		r = c.atom(atomCTAID, 0, 0)
	case ir.OpNTID:
		r = c.atom(atomNTID, 0, 0)
	case ir.OpAdd, ir.OpSub:
		x, ok1 := arg(0)
		y, ok2 := arg(1)
		if ok = ok1 && ok2; ok {
			sign := int64(1)
			if o.Kind == ir.OpSub {
				sign = -1
			}
			r = x.add(y, sign)
		}
	case ir.OpMul:
		x, ok1 := arg(0)
		y, ok2 := arg(1)
		switch {
		case ok1 && ok2 && y.isConst():
			r = x.scale(y.c)
		case ok1 && ok2 && x.isConst():
			r = y.scale(x.c)
		case ok1 && ok2 && len(x.terms) == 1 && len(y.terms) == 1 && x.c == 0 && y.c == 0:
			tx, ty := x.terms[0], y.terms[0]
			if tx.atom > ty.atom {
				tx, ty = ty, tx
			}
			r = c.atom(atomProd, tx.atom, ty.atom).scale(tx.coef * ty.coef)
		default:
			ok = false
		}
	case ir.OpShl:
		x, ok1 := arg(0)
		y, ok2 := arg(1)
		if ok = ok1 && ok2 && y.isConst() && y.c >= 0 && y.c < 32; ok {
			r = x.scale(1 << y.c)
		}
	case ir.OpDiv, ir.OpRem, ir.OpShr, ir.OpAnd:
		// An unsigned shift right by k is the quotient by 2^k, a mask of
		// 2^k-1 the remainder; any other shift or mask is no field.
		isDiv := o.Kind == ir.OpDiv || o.Kind == ir.OpShr
		y, ok2 := arg(1)
		switch o.Kind {
		case ir.OpShr:
			if ok2 = ok2 && y.isConst() && y.c >= 0 && y.c < 32; ok2 {
				y = affine{c: 1 << y.c}
			}
		case ir.OpAnd:
			if ok2 = ok2 && y.isConst() && y.c > 0 && (y.c+1)&y.c == 0; ok2 {
				y = affine{c: y.c + 1}
			}
		}
		if (o.Kind == ir.OpShr || o.Kind == ir.OpAnd) && (!ok2 || o.Type != ir.U32) {
			ok = false
			break
		}
		// By a constant past the dividend's bound, a quotient is zero and a
		// remainder the dividend.
		if hi, okh := c.bound(int(o.Args[0]) - 1); ok2 && y.isConst() && y.c > 0 && okh && hi < uint64(y.c) {
			if isDiv {
				r = affine{}
			} else {
				r, ok = arg(0)
			}
			break
		}
		// Otherwise a field of the dividend: a field of a field where the
		// dividend is one.
		x, ok1 := arg(0)
		if !ok1 || !ok2 || !y.isConst() || y.c <= 0 {
			ok = false
			break
		}
		f := field{x: c.affineName(x), d: 1}
		if len(x.terms) == 1 && x.c == 0 && x.terms[0].coef == 1 {
			if g, isF := c.in.fields[x.terms[0].atom]; isF {
				f = g
			}
		}
		if isDiv {
			f.d *= y.c
		} else {
			mm := f.d * y.c
			if f.m != 0 && f.m%mm != 0 {
				ok = false
				break
			}
			f.m = mm
		}
		r = c.fieldAtom(f)
	case ir.OpPhi:
		// An induction: phi(start, back) with back = phi + step, however
		// the step is reached, is start + step*iteration. The back edge is
		// read with the phi as an atom of its own; what that reading
		// cached is dropped after.
		ok = false
		start, ok1 := arg(0)
		if nx := int(o.Args[1]) - 1; ok1 && nx > i {
			unset := make([]bool, len(c.aff))
			for j, a := range c.aff {
				unset[j] = a == nil
			}
			p := c.atom(atomCarried, int32(i), 0)
			c.aff[i] = &p
			back, ok2 := c.index(nx)
			for j := range c.aff {
				if unset[j] {
					c.aff[j] = nil
				}
			}
			c.aff[i] = nil
			if ok2 && len(back.terms) == 1 && back.terms[0] == p.terms[0] {
				trip := int64(0)
				if lp := c.loop[i]; lp >= 0 {
					trip = c.k.Ops[lp].Imm
				}
				it := c.atom(atomIter, int32(c.depth[i]), int32(trip))
				r, ok = start.add(it.scale(back.c), 1), true
			}
		}
		if !ok {
			c.busy[i] = false
			return affine{}, false
		}
	default:
		ok = false
	}
	c.busy[i] = false
	if !ok {
		// An integer the affine form does not state (a minimum, a load):
		// an atom of its own form.
		r = c.atom(atomOpaque, c.instrName(i), 0)
		c.in.desc[r.terms[0].atom] = c.describeOp(i, 3)
	}
	r = c.merge(r)
	c.aff[i] = &r
	return r, true
}

// name is op i's canonical name.
func (c *canon) name(i int) int32 {
	if id := c.ids[i]; id != 0 {
		return id
	}
	var id int32
	if a, ok := c.index(i); ok {
		id = c.affineName(a)
	} else {
		id = c.instrName(i)
	}
	c.ids[i] = id
	return id
}

// instrName is op i's name by its kind and its operands' names, not as an
// affine value.
func (c *canon) instrName(i int) int32 {
	o := c.k.Ops[i]
	if c.busy[i] {
		return c.in.name(form{tag: fBack, kind: o.Kind})
	}
	c.busy[i] = true
	defer func() { c.busy[i] = false }()
	f := form{tag: fInstr, kind: o.Kind, typ: o.Type, d: c.depth[i]}
	if o.Tile != nil {
		f.tile = *o.Tile
	}
	arg := func(a ir.Value) int32 {
		if a == 0 {
			return 0
		}
		j := int(a) - 1
		if j >= i {
			return c.in.name(form{tag: fBack, kind: c.k.Ops[j].Kind})
		}
		return c.name(j)
	}
	switch o.Kind {
	case ir.OpParam:
		if int(o.Imm) < len(c.k.Params) {
			p := c.k.Params[o.Imm]
			f.imm = int64(p.Elem)<<8 | int64(p.Tile)
		}
	case ir.OpLoad, ir.OpStore, ir.OpLoadV, ir.OpStoreV, ir.OpPrefetch:
		// The buffer, and the address with the immediate folded in.
		f.a[0] = arg(o.Args[0])
		if a, ok := c.index(int(o.Args[1]) - 1); ok && o.Args[1] != 0 {
			f.a[1] = c.affineName(a.add(affine{c: o.Imm}, 1))
		} else {
			f.a[1], f.imm = arg(o.Args[1]), o.Imm
		}
		f.a[2] = arg(o.Args[2])
		f.a[3] = c.vargs(o.Vargs, arg)
	default:
		f.imm = o.Imm
		f.a = [4]int32{arg(o.Args[0]), arg(o.Args[1]), arg(o.Args[2]), c.vargs(o.Vargs, arg)}
	}
	return c.in.name(f)
}

// vargs names an operand list (an MMA's fragments, a vector store's values).
func (c *canon) vargs(vs []ir.Value, arg func(ir.Value) int32) int32 {
	id := int32(0)
	for _, v := range vs {
		id = c.in.name(form{tag: fVargs, a: [4]int32{id, arg(v)}})
	}
	return id
}

// bound is an upper bound on integer op i's value, false where none is
// known (a thread id, a load, a carried value).
func (c *canon) bound(i int) (uint64, bool) {
	o := c.k.Ops[i]
	arg := func(j int) (uint64, bool) {
		if o.Args[j] == 0 {
			return 0, false
		}
		return c.bound(int(o.Args[j]) - 1)
	}
	isConst := func(j int) bool { return o.Args[j] != 0 && c.k.Ops[int(o.Args[j])-1].Kind == ir.OpConst }
	switch o.Kind {
	case ir.OpConst:
		if o.Imm >= 0 {
			return uint64(o.Imm), true
		}
	case ir.OpMin:
		x, okx := arg(0)
		y, oky := arg(1)
		switch {
		case okx && oky:
			return min(x, y), true
		case okx:
			return x, true
		case oky:
			return y, true
		}
	case ir.OpAdd:
		x, okx := arg(0)
		y, oky := arg(1)
		if okx && oky {
			return x + y, true
		}
	case ir.OpMul:
		x, okx := arg(0)
		y, oky := arg(1)
		if okx && oky {
			return x * y, true
		}
	case ir.OpDiv:
		x, okx := arg(0)
		if y, oky := arg(1); okx && oky && y > 0 && isConst(1) {
			return x / y, true
		}
	case ir.OpRem:
		if y, oky := arg(1); oky && y > 0 && isConst(1) {
			if x, okx := arg(0); okx && x < y {
				return x, true
			}
			return y - 1, true
		}
	case ir.OpAnd:
		if y, oky := arg(1); oky && isConst(1) {
			return y, true
		}
	case ir.OpShr:
		x, okx := arg(0)
		if y, oky := arg(1); okx && oky && isConst(1) && y < 64 {
			return x >> y, true
		}
	}
	return 0, false
}

// Sorted is the missing instructions' count by main's kernel name prefix
// (the name up to its first underscore), most first.
func Sorted(missing []Miss, key func(*ir.Kernel) string) (names []string, counts map[string]int) {
	counts = map[string]int{}
	for _, m := range missing {
		counts[key(m.K)]++
	}
	for n := range counts {
		names = append(names, n)
	}
	slices.SortFunc(names, func(a, b string) int { return counts[b] - counts[a] })
	return names, counts
}

// Explain is instruction i of k as the comparison sees it, written out, its
// atoms named by in (share one interner to compare kernels): an index as
// its affine form, any other instruction as its kind of its operands', to
// depth levels. For reading a miss, never for comparing.
func Explain(in *Interner, k *ir.Kernel, i, depth int) string {
	return newCanon(k, in).describe(i, depth)
}

func (c *canon) describe(i, d int) string {
	if a, ok := c.index(i); ok {
		s := fmt.Sprint(a.c)
		for _, t := range a.terms {
			s += fmt.Sprintf(" + %d*%s", t.coef, c.in.desc[t.atom])
		}
		return "[" + s + "]"
	}
	return c.describeOp(i, d)
}

// describeOp is describe of op i by its kind and operands, not its affine
// form.
func (c *canon) describeOp(i, d int) string {
	k := c.k
	o := k.Ops[i]
	if d == 0 {
		return fmt.Sprintf("%v…", o.Kind)
	}
	s := fmt.Sprintf("%v.%v@%d", o.Kind, o.Type, c.depth[i])
	if o.Kind == ir.OpLoad || o.Kind == ir.OpStore {
		s += fmt.Sprintf("+%d", o.Imm)
	} else if o.Imm != 0 {
		s += fmt.Sprintf("#%d", o.Imm)
	}
	s += "("
	for j, a := range o.Args {
		if a == 0 {
			continue
		}
		if j > 0 {
			s += ", "
		}
		if int(a)-1 >= i {
			s += fmt.Sprintf("back:%v", k.Ops[int(a)-1].Kind)
			continue
		}
		s += c.describe(int(a)-1, d-1)
	}
	return s + ")"
}
