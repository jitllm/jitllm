package ir

import "fmt"

// Persist composes a sequence of launches into one persistent kernel: a grid
// of workgroups that runs every launch's body in order, a grid barrier
// (gridsync.go) wherever a launch reads or writes what an earlier one in the
// same phase touched. Launches with no such dependency share a phase, so a
// decode block's q, k and v matvecs, which read one activation and write three
// rows, are one phase and not three.
//
// It is composition rather than a hand-written megakernel: every body is the
// per-op kernel's own IR, copied with its group index replaced by a virtual
// one, so the persistent kernel computes what the launches compute, each
// output by the same arithmetic in the same order -- the per-op path stays the
// reference, and the two agree bit for bit.
//
// Each launch's groups are dealt over the grid: workgroup c runs the launch's
// groups c, c+grid, c+2*grid, ... in a loop whose count is its own, so a
// launch of fewer groups than the grid leaves the rest idle rather than
// repeating its last group. A launch narrower than the persistent width W
// (which must divide it) packs W/w of its groups into one workgroup; such a
// launch may not use workgroup memory or a workgroup barrier, which would then
// span groups that were never meant to meet.
//
// The caller keeps its buffers: Bind names, per parameter of a part, an index
// into the caller's own list, and the result's Bufs maps the persistent
// kernel's parameters back to those indices, its last parameter being the
// grid barrier's sync buffer (GridSyncWords words, zero when allocated).

// Part is one launch: its kernel, its workgroup count and, per parameter, the
// caller's index of the buffer it was bound to.
type Part struct {
	K      *Kernel
	Groups int
	Bind   []int
}

// Persisted is a composed kernel and how to launch it.
type Persisted struct {
	K *Kernel
	// Bufs is the caller's buffer index for each parameter but the last; the
	// last is the sync buffer.
	Bufs []int
	// Phases is how many grid-barrier-separated phases the parts took.
	Phases int
	// Grid is the workgroup count the kernel was composed for and must be
	// launched with.
	Grid int
	// Words is how many words of the dep buffer, from the base it was
	// composed at, a PersistDeps kernel's counters take; zero for Persist.
	Words int
}

// PersistJoins reports whether a launch of k can join a persistent kernel of
// width w, and why not.
func PersistJoins(k *Kernel, w int) error {
	kw := k.Group[0]
	switch {
	case k.Group[1] > 1 || k.Group[2] > 1:
		return fmt.Errorf("ir: %s: a %v workgroup", k.Name, k.Group)
	case k.GridSync():
		return fmt.Errorf("ir: %s: already has a grid barrier", k.Name)
	case kw <= 0 || w%kw != 0:
		return fmt.Errorf("ir: %s: width %d does not divide %d", k.Name, kw, w)
	case kw != w && k.usesGroupMemory():
		return fmt.Errorf("ir: %s: width %d below %d with workgroup memory or a barrier", k.Name, kw, w)
	case kw != w && k.Lanes != 0 && kw%SubgroupLanes != 0:
		return fmt.Errorf("ir: %s: width %d is not whole subgroups", k.Name, kw)
	}
	return nil
}

func (k *Kernel) usesGroupMemory() bool {
	for _, o := range k.Ops {
		if o.Kind == OpShared || o.Kind == OpBarrier {
			return true
		}
	}
	return false
}

// access lists the parameters k reads and writes.
func (k *Kernel) access() (reads, writes []int) {
	for _, o := range k.Ops {
		p := k.paramOf(o.Args[0])
		if p < 0 {
			continue
		}
		switch o.Kind {
		case OpLoad, OpLoadV, OpTileLoad:
			reads = append(reads, p)
		case OpStore, OpStoreV, OpTileStore:
			writes = append(writes, p)
		}
	}
	return reads, writes
}

// Persist composes parts into one kernel of grid workgroups of width w.
func Persist(name string, parts []Part, w, grid int) (*Persisted, error) {
	if len(parts) == 0 || grid <= 0 {
		return nil, fmt.Errorf("ir: Persist: %d parts on a grid of %d", len(parts), grid)
	}
	type acc struct{ r, w map[int]bool }
	newAcc := func() acc { return acc{map[int]bool{}, map[int]bool{}} }
	// Phases: a part joins the open phase unless it reads what the phase
	// writes or writes what the phase reads or writes.
	phase := make([]int, len(parts))
	cur, n := newAcc(), 0
	lanes := 0
	for i, p := range parts {
		if err := PersistJoins(p.K, w); err != nil {
			return nil, err
		}
		if len(p.Bind) != len(p.K.Params) {
			return nil, fmt.Errorf("ir: Persist: %s binds %d of %d parameters", p.K.Name, len(p.Bind), len(p.K.Params))
		}
		if p.Groups <= 0 {
			return nil, fmt.Errorf("ir: Persist: %s launched %d groups", p.K.Name, p.Groups)
		}
		if p.K.Lanes != 0 {
			lanes = p.K.Lanes
		}
		rs, ws := p.K.access()
		clash := false
		for _, r := range rs {
			clash = clash || cur.w[p.Bind[r]]
		}
		for _, x := range ws {
			clash = clash || cur.w[p.Bind[x]] || cur.r[p.Bind[x]]
		}
		if clash {
			cur, n = newAcc(), n+1
		}
		for _, r := range rs {
			cur.r[p.Bind[r]] = true
		}
		for _, x := range ws {
			cur.w[p.Bind[x]] = true
		}
		phase[i] = n
	}

	b := New(name, [3]int{w, 1, 1})
	// One parameter per (buffer, element type): two parts may read one buffer
	// as different types, and a load is typed by its parameter.
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
	sync := b.Param("gridsync", U32)

	// Workgroup arrays: declared before every loop, and reused by a later
	// phase (the grid barrier orders a workgroup's own accesses too).
	type skey struct {
		t Type
		n int64
	}
	var arrays []struct {
		k    skey
		v    Value
		last int // the last phase using it
	}
	shared := make([]map[Value]Value, len(parts)) // per part: its OpShared -> the array
	for i, p := range parts {
		shared[i] = map[Value]Value{}
		for j, o := range p.K.Ops {
			if o.Kind != OpShared {
				continue
			}
			key := skey{o.Type, o.Imm}
			got := Value(0)
			for a := range arrays {
				if arrays[a].k == key && arrays[a].last < phase[i] {
					arrays[a].last = phase[i]
					got = arrays[a].v
					break
				}
			}
			if got == 0 {
				got = b.Shared(fmt.Sprintf("sh%d", len(arrays)), o.Type, int(o.Imm))
				arrays = append(arrays, struct {
					k    skey
					v    Value
					last int
				}{key, got, phase[i]})
			}
			shared[i][Value(j+1)] = got
		}
	}

	ctaid, tid := b.CTAID(), b.TID()
	c := func(v int64) Value { return b.Const(U32, v) }
	zero := c(0)
	for i, p := range parts {
		if i > 0 && phase[i] != phase[i-1] {
			b.GridBarrier(sync)
		}
		pw := p.K.Group[0]
		pack := w / pw
		slots := grid * pack
		vt, vg0 := tid, ctaid
		if pack > 1 {
			vt = b.Rem(U32, tid, c(int64(pw)))
			vg0 = b.Add(U32, b.Mul(U32, ctaid, c(int64(pack))), b.Div(U32, tid, c(int64(pw))))
		}
		g := int64(p.Groups)
		// This slot's count of the launch's groups: ceil((g - vg0) / slots),
		// zero past the last.
		cnt := b.Div(U32, b.Sub(U32, c(g+int64(slots)-1), b.Min(U32, vg0, c(g))), c(int64(slots)))
		b.LoopN(cnt)
		it := b.Phi(U32, zero)
		vb := b.Add(U32, b.Mul(U32, it, c(int64(slots))), vg0)
		if err := b.inline(p, func(buf int, prm Param) Value { return paramFor(buf, prm) }, shared[i], vb, vt, c(int64(pw))); err != nil {
			return nil, err
		}
		if p.K.usesGroupMemory() {
			// The next group this workgroup runs reuses the arrays: no thread
			// may start writing them while another still reads this group's.
			b.Barrier()
		}
		b.SetPhi(it, b.Add(U32, it, c(1)))
		b.EndLoop()
	}
	k, err := b.Done()
	if err != nil {
		return nil, fmt.Errorf("ir: Persist %s: %w", name, err)
	}
	k.Lanes = lanes
	if err := k.Validate(); err != nil {
		return nil, fmt.Errorf("ir: Persist %s: %w", name, err)
	}
	_ = sync
	return &Persisted{K: k, Bufs: bufs, Phases: phase[len(phase)-1] + 1, Grid: grid}, nil
}

// inline copies part p's body into b: its parameters become param(buf, ...),
// its workgroup arrays the ones assigned, its group index vb, its thread index
// vt and its width ntid.
func (b *Builder) inline(p Part, param func(int, Param) Value, shared map[Value]Value, vb, vt, ntid Value) error {
	m := make([]Value, len(p.K.Ops)+1)
	var phis []Value // new phis whose back edge is patched at the end
	var olds []Value
	for j, o := range p.K.Ops {
		v := Value(j + 1)
		switch o.Kind {
		case OpParam:
			m[v] = param(p.Bind[o.Imm], p.K.Params[o.Imm])
			continue
		case OpShared:
			m[v] = shared[v]
			continue
		case OpCTAID:
			m[v] = vb
			continue
		case OpTID:
			m[v] = vt
			continue
		case OpNTID:
			m[v] = ntid
			continue
		case OpGridBarrier:
			return fmt.Errorf("ir: Persist: %s has a grid barrier", p.K.Name)
		}
		n := o
		for a := range n.Args {
			if n.Args[a] == 0 || (o.Kind == OpPhi && a == 1) {
				continue
			}
			n.Args[a] = m[n.Args[a]]
		}
		if o.Vargs != nil {
			n.Vargs = make([]Value, len(o.Vargs))
			for a, x := range o.Vargs {
				n.Vargs[a] = m[x]
			}
		}
		m[v] = b.emit(n)
		if o.Kind == OpPhi {
			phis = append(phis, m[v])
			olds = append(olds, o.Args[1])
		}
	}
	for i, ph := range phis {
		b.k.Ops[ph-1].Args[1] = m[olds[i]]
	}
	return nil
}
