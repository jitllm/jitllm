package ir

import "fmt"

// Dependency counters: the persistent kernel's other synchronisation
// (docs/design/dep-block.md).
//
// A grid barrier makes every workgroup wait for every other. A dependency
// counter makes a workgroup wait only for the work it reads: a composed
// kernel's tasks (its parts, PersistDeps) each count their finished tiles or
// workgroups in a u32 word of the dep buffer, and a task waits on the words of
// the tasks it reads or overwrites before it touches their buffers.
//
//   - OpDepTask (Imm = task id) marks where a task's body begins. It lowers to
//     nothing; Validate reads it.
//   - OpDepEpoch (Args[0] = buf, Imm = word) yields the region's epoch: how
//     many launches of this kernel have finished. Counters never reset; after
//     n launches a task counter holds n times its signallers and a tile word
//     n, so a wait's target is a multiple of epoch+1.
//   - OpDepWait (Args = buf, word, target; Imm = producer task, DepTile set
//     when only some of its tiles are waited for) holds the workgroup until
//     the word reaches the target, compared modulo 2^32. Acquire, at device
//     scope, and a workgroup barrier after it.
//   - OpDepSignal (Args = buf, word; Imm = the task) adds one to the word
//     after a workgroup barrier, with release at device scope.
//   - OpDepExit (Args[0] = buf, Imm = the region's first word) arrives on the
//     exit count at word+1; the last arrival zeroes it and advances the epoch
//     at word. It is the kernel's last op.
//
// A wait and a signal hold a workgroup barrier, so every thread of the
// workgroup reaches them: they may sit in a loop whose count is the
// workgroup's own, never under a per-thread count.
//
// Like the grid barrier, the counters need every workgroup resident at once
// (a waiter spins for a workgroup that must be running), so a kernel holding
// them reports GridSync and is launched as one. And like it, they own their
// buffer: Validate refuses any other access to it, and refuses a kernel that
// reads a buffer a task earlier in it wrote (or writes one an earlier task
// touched) without a wait that orders the two -- RULE 13's in-place refusal,
// extended through the protocol rather than weakened.

// DepTile marks an OpDepWait that waits for some of the producer's tiles
// rather than all of them.
const DepTile = int64(1) << 32

// DepTask marks the start of task id's body.
func (b *Builder) DepTask(id int) { b.emit(Op{Kind: OpDepTask, Type: Void, Imm: int64(id)}) }

// DepEpoch reads the epoch at word of buf.
func (b *Builder) DepEpoch(buf Value, word int) Value {
	return b.emit(Op{Kind: OpDepEpoch, Type: U32, Args: [3]Value{buf}, Imm: int64(word)})
}

// DepWait waits until buf[word] reaches target; producer names the task that
// signals it, tile whether only some of its tiles are waited for.
func (b *Builder) DepWait(buf, word, target Value, producer int, tile bool) {
	imm := int64(producer)
	if tile {
		imm |= DepTile
	}
	b.emit(Op{Kind: OpDepWait, Type: Void, Args: [3]Value{buf, word, target}, Imm: imm})
}

// DepSignal adds one to buf[word] for task.
func (b *Builder) DepSignal(buf, word Value, task int) {
	b.emit(Op{Kind: OpDepSignal, Type: Void, Args: [3]Value{buf, word}, Imm: int64(task)})
}

// DepExit arrives on the region at base.
func (b *Builder) DepExit(buf Value, base int) {
	b.emit(Op{Kind: OpDepExit, Type: Void, Args: [3]Value{buf}, Imm: int64(base)})
}

func isDep(k Kind) bool {
	switch k {
	case OpDepTask, OpDepEpoch, OpDepWait, OpDepSignal, OpDepExit:
		return true
	}
	return false
}

// validateDeps checks the counters: one u32 buffer owned by them, an exit
// that ends the kernel, tasks in order, waits naming earlier tasks and
// signals their own, and every cross-task hazard ordered by a wait.
func (k *Kernel) validateDeps() error {
	buf, exits, last := -1, 0, -1
	for i, o := range k.Ops {
		if !isDep(o.Kind) {
			continue
		}
		last = i
		if o.Kind == OpDepTask {
			continue
		}
		p := k.paramOf(o.Args[0])
		if p < 0 {
			return fmt.Errorf("ir: op %d (%s): the dep buffer must be a parameter", i, o.Kind)
		}
		if k.Params[p].Elem != U32 {
			return fmt.Errorf("ir: op %d: dep buffer %q is %s, not u32", i, k.Params[p].Name, k.Params[p].Elem)
		}
		if buf >= 0 && buf != p {
			return fmt.Errorf("ir: op %d: a second dep buffer (param %d after %d)", i, p, buf)
		}
		buf = p
		if o.Kind == OpDepExit {
			exits++
		}
	}
	if last < 0 {
		return nil
	}
	if k.GridSyncParamBarrier() >= 0 {
		return fmt.Errorf("ir: a kernel with dependency counters has a grid barrier too")
	}
	if buf < 0 || exits != 1 || k.Ops[len(k.Ops)-1].Kind != OpDepExit {
		return fmt.Errorf("ir: dependency counters need exactly one dep.exit, as the kernel's last op")
	}
	for i, o := range k.Ops {
		if isDep(o.Kind) || o.Kind == OpParam {
			continue
		}
		for _, a := range o.Args {
			if a != 0 && k.paramOf(a) == buf {
				return fmt.Errorf("ir: op %d (%s) touches the dep buffer %q, which the counters own",
					i, o.Kind, k.Params[buf].Name)
			}
		}
		for _, a := range o.Vargs {
			if k.paramOf(a) == buf {
				return fmt.Errorf("ir: op %d (%s) touches the dep buffer", i, o.Kind)
			}
		}
	}

	// Hazards. covered[t] is the tasks whose every tile has finished once
	// task t's every tile has: those it waited on wholly, and theirs.
	type state struct {
		read, wrote [maxParams]bool
	}
	tasks := map[int]*state{}
	covered := map[int]map[int]bool{}
	var order []int
	cur := -1
	seen := map[int]bool{} // tasks the current one has waited on, and their cover
	depth := 0
	for i, o := range k.Ops {
		switch o.Kind {
		case OpLoop:
			depth++
		case OpEndLoop:
			depth--
		case OpDepTask:
			if depth != 0 {
				return fmt.Errorf("ir: op %d: dep.task inside a loop", i)
			}
			id := int(o.Imm)
			if id <= cur {
				return fmt.Errorf("ir: op %d: task %d after task %d", i, id, cur)
			}
			cur = id
			tasks[id] = &state{}
			covered[id] = map[int]bool{}
			order = append(order, id)
			seen = map[int]bool{}
		case OpDepWait:
			if cur < 0 {
				return fmt.Errorf("ir: op %d: dep.wait outside a task", i)
			}
			pr := int(o.Imm &^ DepTile)
			if pr >= cur || tasks[pr] == nil {
				return fmt.Errorf("ir: op %d: task %d waits on task %d, not an earlier one", i, cur, pr)
			}
			seen[pr] = true
			if o.Imm&DepTile == 0 {
				covered[cur][pr] = true
				for c := range covered[pr] {
					covered[cur][c] = true
					seen[c] = true
				}
			}
		case OpDepSignal:
			if int(o.Imm) != cur {
				return fmt.Errorf("ir: op %d: task %d signals for task %d", i, cur, o.Imm)
			}
		case OpDepEpoch:
			if depth != 0 {
				return fmt.Errorf("ir: op %d: dep.epoch inside a loop", i)
			}
		case OpDepExit:
			if depth != 0 {
				return fmt.Errorf("ir: op %d: dep.exit inside a loop", i)
			}
		case OpLoad, OpLoadV, OpStore, OpStoreV:
			p := k.paramOf(o.Args[0])
			if p < 0 {
				continue
			}
			if cur < 0 {
				return fmt.Errorf("ir: op %d: a global access before the first dep.task", i)
			}
			st := o.Kind == OpStore || o.Kind == OpStoreV
			for _, t := range order[:len(order)-1] {
				e := tasks[t]
				if (e.wrote[p] || (st && e.read[p])) && !seen[t] {
					return fmt.Errorf("ir: op %d: task %d %s param %d, which task %d %s, without waiting on it",
						i, cur, map[bool]string{true: "writes", false: "reads"}[st], p, t,
						map[bool]string{true: "wrote", false: "read"}[e.wrote[p]])
				}
			}
			if st {
				tasks[cur].wrote[p] = true
			} else {
				tasks[cur].read[p] = true
			}
		}
	}
	return nil
}

// GridSyncParamBarrier is the parameter of the kernel's grid barrier, or -1.
func (k *Kernel) GridSyncParamBarrier() int {
	for _, o := range k.Ops {
		if o.Kind == OpGridBarrier {
			return k.paramOf(o.Args[0])
		}
	}
	return -1
}

// span is what a kernel's accesses to one buffer touch, per workgroup: group
// g's elements lie in [g*a + lo, g*a + hi]. ok is false when the analysis
// could not bound them. A clamp min(x, c) keeps x's range for every group
// whose range starts at or below c; clampAt is the least such c - lo, so the
// span holds for groups g with g*a <= clampAt.
type span struct {
	ok      bool
	a       int64
	lo, hi  int64
	clampAt int64
}

const noClamp = int64(1) << 62

// form is an index value's shape: ctaid*a + [lo, hi].
type form struct {
	ok      bool
	a       int64
	lo, hi  int64
	clampAt int64
}

// spans bounds, per parameter, the elements each workgroup of k touches:
// ok only when every access to it is ctaid*a + r with r bounded.
func (k *Kernel) spans() map[int]span {
	w := int64(k.Group[0])
	f := make([]form, len(k.Ops)+1)
	cst := func(v Value) (int64, bool) {
		x := f[v]
		return x.lo, x.ok && x.a == 0 && x.lo == x.hi
	}
	for i, o := range k.Ops {
		v := Value(i + 1)
		x, y := f[o.Args[0]], f[o.Args[1]]
		r := form{clampAt: noClamp}
		both := x.ok && y.ok
		clamp := min(x.clampAt, y.clampAt)
		switch o.Kind {
		case OpConst:
			if o.Type != F32 && o.Type != Pred {
				r = form{ok: true, lo: int64(int32(uint32(o.Imm))), hi: int64(int32(uint32(o.Imm))), clampAt: noClamp}
				if o.Type == U32 {
					r.lo, r.hi = int64(uint32(o.Imm)), int64(uint32(o.Imm))
				}
			}
		case OpCTAID:
			r = form{ok: true, a: 1, clampAt: noClamp}
		case OpTID:
			r = form{ok: true, hi: w - 1, clampAt: noClamp}
		case OpNTID:
			r = form{ok: true, lo: w, hi: w, clampAt: noClamp}
		case OpLaneID:
			r = form{ok: true, hi: SubgroupLanes - 1, clampAt: noClamp}
		case OpAdd:
			// A clamp's bound is stated per group of one stride; two strides
			// added are not bounded by either.
			if both && (x.a == 0 || y.a == 0 || clamp == noClamp) {
				r = form{ok: true, a: x.a + y.a, lo: x.lo + y.lo, hi: x.hi + y.hi, clampAt: clamp}
			}
		case OpSub:
			if both && y.a == 0 {
				r = form{ok: true, a: x.a, lo: x.lo - y.hi, hi: x.hi - y.lo, clampAt: clamp}
			}
		case OpMul, OpShl:
			c, isC := cst(o.Args[1])
			other := x
			if !isC {
				if c2, ok2 := cst(o.Args[0]); ok2 && o.Kind == OpMul {
					c, isC, other = c2, true, y
				}
			}
			if o.Kind == OpShl {
				c = int64(1) << uint(c&31)
			}
			switch {
			case isC && other.ok && c >= 0:
				ca := other.clampAt
				if ca != noClamp {
					ca *= c
				}
				r = form{ok: true, a: other.a * c, lo: other.lo * c, hi: other.hi * c, clampAt: ca}
			case both && x.a == 0 && y.a == 0 && x.lo >= 0 && y.lo >= 0:
				r = form{ok: true, lo: x.lo * y.lo, hi: x.hi * y.hi, clampAt: noClamp}
			}
		case OpDiv, OpShr:
			c, isC := cst(o.Args[1])
			if o.Kind == OpShr {
				c = int64(1) << uint(c&31)
			}
			if isC && c > 0 && x.ok && x.a == 0 && x.lo >= 0 {
				r = form{ok: true, lo: x.lo / c, hi: x.hi / c, clampAt: noClamp}
			}
		case OpRem:
			c, isC := cst(o.Args[1])
			if isC && c > 0 && x.ok && x.a == 0 && x.lo >= 0 {
				r = form{ok: true, lo: 0, hi: min(c-1, x.hi), clampAt: noClamp}
			}
		case OpAnd:
			c, isC := cst(o.Args[1])
			if isC && c >= 0 && x.ok && x.a == 0 && x.lo >= 0 {
				r = form{ok: true, lo: 0, hi: min(c, x.hi), clampAt: noClamp}
			}
		case OpMin:
			c, isC := cst(o.Args[1])
			if !isC {
				if c2, ok2 := cst(o.Args[0]); ok2 {
					c, isC, x = c2, true, y
				}
			}
			switch {
			case isC && x.ok && x.a == 0:
				r = form{ok: true, lo: min(x.lo, c), hi: min(x.hi, c), clampAt: x.clampAt}
			case isC && x.ok && x.a > 0:
				// Group g's range is kept while c >= g*a + lo.
				r = x
				r.clampAt = min(x.clampAt, c-x.lo)
			}
		case OpMax:
			c, isC := cst(o.Args[1])
			if !isC {
				if c2, ok2 := cst(o.Args[0]); ok2 {
					c, isC, x = c2, true, y
				}
			}
			switch {
			case isC && x.ok && x.a == 0:
				r = form{ok: true, lo: max(x.lo, c), hi: max(x.hi, c), clampAt: x.clampAt}
			case isC && x.ok && x.a > 0 && c <= x.lo:
				r = x
			}
		case OpSelect:
			y, z := f[o.Args[1]], f[o.Args[2]]
			if y.ok && z.ok && y.a == z.a {
				r = form{ok: true, a: y.a, lo: min(y.lo, z.lo), hi: max(y.hi, z.hi), clampAt: min(y.clampAt, z.clampAt)}
			}
		}
		f[v] = r
	}
	out := map[int]span{}
	note := func(p int, x form, extra, n int64) {
		s, seen := out[p]
		if !x.ok {
			out[p] = span{}
			return
		}
		lo, hi := x.lo+extra, x.hi+extra+n-1
		if !seen {
			out[p] = span{ok: true, a: x.a, lo: lo, hi: hi, clampAt: x.clampAt}
			return
		}
		if !s.ok || s.a != x.a {
			out[p] = span{}
			return
		}
		out[p] = span{ok: true, a: s.a, lo: min(s.lo, lo), hi: max(s.hi, hi), clampAt: min(s.clampAt, x.clampAt)}
	}
	for _, o := range k.Ops {
		p := k.paramOf(o.Args[0])
		if p < 0 {
			continue
		}
		switch o.Kind {
		case OpLoad, OpStore:
			note(p, f[o.Args[1]], o.Imm, 1)
		case OpLoadV, OpStoreV:
			off, n := VecOf(o)
			note(p, f[o.Args[1]], off, int64(n))
		case OpTileLoad, OpTileStore:
			out[p] = span{}
		}
	}
	for p, s := range out {
		if s.ok && s.a <= 0 {
			out[p] = span{}
		}
	}
	return out
}

// spanFor is s if it holds for every one of groups workgroups.
func (s span) holds(groups int) bool {
	return s.ok && (s.clampAt == noClamp || int64(groups-1)*s.a <= s.clampAt)
}

// PersistDeps composes parts like Persist, ordering them with dependency
// counters in a region of the dep buffer that starts at word base rather than
// with grid barriers. Its last parameter is the dep buffer; Words is how many
// words from base the region takes. The tasks run in order in every
// workgroup and a tile waits only for the producer tiles it reads, or for
// the whole producer when the interval analysis (spans) cannot bound what
// either touches.
func PersistDeps(name string, parts []Part, w, grid, base int) (*Persisted, error) {
	if len(parts) == 0 || grid <= 0 || base < 0 {
		return nil, fmt.Errorf("ir: PersistDeps: %d parts on a grid of %d at %d", len(parts), grid, base)
	}
	n := len(parts)
	type bufSpan struct {
		read, wrote bool
		s           span
		seen        bool
	}
	acc := make([]map[int]*bufSpan, n)
	pack := make([]int, n)
	active := make([]int, n)
	lanes := 0
	for i, p := range parts {
		if err := PersistJoins(p.K, w); err != nil {
			return nil, err
		}
		if len(p.Bind) != len(p.K.Params) {
			return nil, fmt.Errorf("ir: PersistDeps: %s binds %d of %d parameters", p.K.Name, len(p.Bind), len(p.K.Params))
		}
		if p.Groups <= 0 {
			return nil, fmt.Errorf("ir: PersistDeps: %s launched %d groups", p.K.Name, p.Groups)
		}
		if p.K.Lanes != 0 {
			lanes = p.K.Lanes
		}
		pack[i] = w / p.K.Group[0]
		active[i] = min(grid, (p.Groups+pack[i]-1)/pack[i])
		acc[i] = map[int]*bufSpan{}
		sp := p.K.spans()
		rs, ws := p.K.access()
		mark := func(prm int, wr bool) {
			b := p.Bind[prm]
			e := acc[i][b]
			if e == nil {
				e = &bufSpan{}
				acc[i][b] = e
			}
			if wr {
				e.wrote = true
			} else {
				e.read = true
			}
			s := sp[prm]
			if !s.holds(p.Groups) || pack[i] != 1 {
				s = span{}
			}
			if !e.seen {
				e.s, e.seen = s, true
			} else if e.s != s {
				e.s = span{}
			}
		}
		for _, r := range rs {
			mark(r, false)
		}
		for _, x := range ws {
			mark(x, true)
		}
	}

	// Edges: whole[j] the tasks j waits on wholly, tile[j][i] the ranges of
	// i's tiles per consumer tile.
	type tileDep struct {
		i      int
		ac     int64 // consumer stride
		lo, hi int64 // consumer's range offsets
		ap     int64
		plo    int64
		phi    int64
	}
	whole := make([][]int, n)
	tiles := make([][]tileDep, n)
	covered := make([]map[int]bool, n)
	needWhole := make([]bool, n)
	needTile := make([]bool, n)
	for j := range parts {
		covered[j] = map[int]bool{}
		var cand []int
		isWhole := map[int]bool{}
		var tl []tileDep
		for i := j - 1; i >= 0; i-- {
			conflict, tileable := false, true
			var td tileDep
			first := true
			for b, cj := range acc[j] {
				ci := acc[i][b]
				if ci == nil || !(cj.read && ci.wrote || cj.wrote && (ci.read || ci.wrote)) {
					continue
				}
				conflict = true
				if !cj.s.ok || !ci.s.ok {
					tileable = false
					continue
				}
				d := tileDep{i: i, ac: cj.s.a, lo: cj.s.lo, hi: cj.s.hi, ap: ci.s.a, plo: ci.s.lo, phi: ci.s.hi}
				if first {
					td, first = d, false
				} else if td != d {
					tileable = false
				}
			}
			if !conflict {
				continue
			}
			cand = append(cand, i)
			if tileable {
				tl = append(tl, td)
			} else {
				isWhole[i] = true
			}
		}
		// Whole edges, latest first, dropping what an earlier-kept one covers.
		for _, i := range cand {
			if !isWhole[i] || covered[j][i] {
				continue
			}
			whole[j] = append(whole[j], i)
			needWhole[i] = true
			covered[j][i] = true
			for c := range covered[i] {
				covered[j][c] = true
			}
		}
		for _, d := range tl {
			if covered[j][d.i] {
				continue
			}
			tiles[j] = append(tiles[j], d)
			needTile[d.i] = true
		}
	}
	// The region: epoch, exit, then a counter per wholly-waited task and a
	// word per tile of each tile-waited task.
	word := 2
	cw := make([]int, n)
	tw := make([]int, n)
	for i, p := range parts {
		if needWhole[i] {
			cw[i] = base + word
			word++
		}
		if needTile[i] {
			tw[i] = base + word
			word += p.Groups
		}
	}

	b := New(name, [3]int{w, 1, 1})
	type pkey struct {
		buf  int
		elem Type
		tile TileElem
		is   bool
	}
	pidx := map[pkey]Value{}
	var bufs []int
	paramFor := func(buf int, p Param) Value {
		key := pkey{buf, p.Elem, p.Tile, p.IsTile}
		if v, ok := pidx[key]; ok {
			return v
		}
		v := b.Param(fmt.Sprintf("b%d_%d", buf, len(bufs)), p.Elem)
		if p.IsTile {
			b.k.Params[len(b.k.Params)-1].Tile = p.Tile
			b.k.Params[len(b.k.Params)-1].IsTile = true
		}
		pidx[key] = v
		bufs = append(bufs, buf)
		return v
	}
	for _, p := range parts {
		for j, prm := range p.K.Params {
			paramFor(p.Bind[j], prm)
		}
	}
	dep := b.Param("dep", U32)
	// Workgroup arrays: one per (type, size), shared by every task. A task's
	// accesses to it are its own workgroup's, and its tiles run one after
	// another in program order, each ending in its own barriers or none.
	type skey struct {
		t Type
		n int64
	}
	arrays := map[skey]Value{}
	shared := make([]map[Value]Value, n)
	for i, p := range parts {
		shared[i] = map[Value]Value{}
		for j, o := range p.K.Ops {
			if o.Kind != OpShared {
				continue
			}
			key := skey{o.Type, o.Imm}
			got, ok := arrays[key]
			if !ok {
				got = b.Shared(fmt.Sprintf("sh%d", len(arrays)), o.Type, int(o.Imm))
				arrays[key] = got
			}
			shared[i][Value(j+1)] = got
		}
	}

	c := func(v int64) Value { return b.Const(U32, v) }
	epoch := b.DepEpoch(dep, base)
	ctaid, tid := b.CTAID(), b.TID()
	e1 := b.Add(U32, epoch, c(1))
	zero := c(0)
	for j, p := range parts {
		b.DepTask(j)
		// act: one when this workgroup runs any tile of the task.
		act := b.Min(U32, c(1), b.Sub(U32, c(int64(active[j])), b.Min(U32, ctaid, c(int64(active[j])))))
		if len(whole[j]) > 0 {
			b.LoopN(act)
			for _, i := range whole[j] {
				b.DepWait(dep, c(int64(cw[i])), b.Mul(U32, e1, c(int64(active[i]))), i, false)
			}
			b.EndLoop()
		}
		pw := p.K.Group[0]
		pk := pack[j]
		slots := grid * pk
		vt, vg0 := tid, ctaid
		if pk > 1 {
			vt = b.Rem(U32, tid, c(int64(pw)))
			vg0 = b.Add(U32, b.Mul(U32, ctaid, c(int64(pk))), b.Div(U32, tid, c(int64(pw))))
		}
		g := int64(p.Groups)
		cnt := b.Div(U32, b.Sub(U32, c(g+int64(slots)-1), b.Min(U32, vg0, c(g))), c(int64(slots)))
		b.LoopN(cnt)
		it := b.Phi(U32, zero)
		vb := b.Add(U32, b.Mul(U32, it, c(int64(slots))), vg0)
		for _, d := range tiles[j] {
			// Producer tiles h whose range meets this tile's, shifted by K so
			// every quantity stays unsigned: h in [H0-K, H1-K].
			ap := d.ap
			kk := (abs64(d.lo)+abs64(d.hi)+abs64(d.plo)+abs64(d.phi))/ap + 1
			gac := b.Mul(U32, vb, c(d.ac))
			n0 := b.Add(U32, gac, c(d.lo-d.phi+kk*ap))
			m0 := b.Add(U32, gac, c(d.hi-d.plo+kk*ap))
			h0 := b.Max(U32, b.Div(U32, b.Add(U32, n0, c(ap-1)), c(ap)), c(kk))
			h1 := b.Min(U32, b.Div(U32, m0, c(ap)), c(kk+int64(parts[d.i].Groups)-1))
			h1p := b.Add(U32, h1, c(1))
			hn := b.Sub(U32, h1p, b.Min(U32, h0, h1p))
			b.LoopN(hn)
			h := b.Phi(U32, zero)
			at := b.Add(U32, b.Add(U32, h0, h), c(int64(tw[d.i])-kk))
			b.DepWait(dep, at, e1, d.i, true)
			b.SetPhi(h, b.Add(U32, h, c(1)))
			b.EndLoop()
		}
		if err := b.inline(p, func(buf int, prm Param) Value { return paramFor(buf, prm) }, shared[j], vb, vt, c(int64(pw))); err != nil {
			return nil, err
		}
		if p.K.usesGroupMemory() {
			// The next tile this workgroup runs, of this task or a later one,
			// reuses the arrays.
			b.Barrier()
		}
		if needTile[j] {
			b.DepSignal(dep, b.Add(U32, vb, c(int64(tw[j]))), j)
		}
		b.SetPhi(it, b.Add(U32, it, c(1)))
		b.EndLoop()
		if needWhole[j] {
			b.LoopN(act)
			b.DepSignal(dep, c(int64(cw[j])), j)
			b.EndLoop()
		}
	}
	b.DepExit(dep, base)
	k := b.Done()
	k.Lanes = lanes
	if err := k.Validate(); err != nil {
		return nil, fmt.Errorf("ir: PersistDeps %s: %w", name, err)
	}
	return &Persisted{K: k, Bufs: bufs, Phases: n, Grid: grid, Words: word}, nil
}

func abs64(x int64) int64 {
	if x < 0 {
		return -x
	}
	return x
}
