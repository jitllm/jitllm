package jit

// Cursors replaces a loop's indexing of a buffer by an address the loop
// advances: where every access in the loop is its iteration times a stride
// plus the same part, the address moves the stride each trip and each
// access reads at a constant displacement from it. A loop picks up the
// address where the loop before it left it when its accesses start there;
// it advances a copy where a later loop needs the address as it was.
type Cursors struct{}

func (Cursors) Name() string { return "cursors" }

func (p Cursors) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Args == nil {
			return nil, false, nil
		}
		m := k.clone()
		changed := false
		for _, a := range m.Args {
			if a.Buf >= 0 && m.cursor(a.Buf, -1) {
				changed = true
			}
		}
		if !changed {
			return nil, false, nil
		}
		return []*Kernel{m}, true, nil
	})
}

// cursorAccess is an access of a buffer in a loop: its index's stride over
// the loop's iteration and the rest of its terms.
type cursorAccess struct {
	access
	stride int64
	rest   map[int]int64
	c      int64
}

// cursor advances buffer b's address through the loops inside loop parent.
func (k *Kernel) cursor(b, parent int) bool {
	base := k.Bufs[b].Base
	var loops []int
	for c, l := range k.Loops {
		if l.Parent == parent {
			loops = append(loops, c)
		}
	}
	type plan struct {
		c       int
		as      []cursorAccess
		stride  int64
		advance map[int]int64
		adv     int64
	}
	var plans []plan
	for _, c := range loops {
		as, ok := k.accessesIn(b, base, c)
		if !ok || len(as) == 0 {
			continue
		}
		s := as[0].stride
		for _, a := range as {
			if a.stride != s {
				s = 0
			}
		}
		if s == 0 {
			continue
		}
		l := k.Loops[c]
		step := int64(max(1, l.Lanes)) * s
		adv := map[int]int64{}
		var advc int64
		if l.Count > 0 {
			adv[l.Count-1] = step
		} else {
			advc = int64(l.Trip) * step
		}
		plans = append(plans, plan{c: c, as: as, stride: s, advance: adv, adv: advc})
	}
	if len(plans) == 0 {
		return false
	}
	at, atc := map[int]int64{}, int64(0)
	for i, pl := range plans {
		rest := pl.as[0].rest
		cont := k.sameTerms(rest, at)
		copyOf := !cont
		if cont {
			after := addTerms(at, pl.advance)
			for _, later := range plans[i+1:] {
				if !k.sameTerms(later.as[0].rest, after) {
					copyOf = true
				}
			}
		}
		if copyOf && len(rest) > 0 {
			continue
		}
		from, fromc := at, atc
		if copyOf {
			from, fromc = map[int]int64{}, 0
		}
		for _, a := range pl.as {
			if !k.sameTerms(a.rest, from) {
				return false
			}
			disp := k.u32(int(a.c - fromc))
			if a.node >= 0 {
				k.Nodes[a.node].A, k.Nodes[a.node].At = disp, pl.c+1
			} else {
				k.Stores[a.store].Ix = disp
			}
		}
		k.Loops[pl.c].Cursors = append(k.Loops[pl.c].Cursors, Cursor{Buf: b, Elems: int(pl.stride) * max(1, k.Loops[pl.c].Lanes), Copy: copyOf})
		if !copyOf {
			at, atc = addTerms(at, pl.advance), atc+pl.adv
		}
	}
	return true
}

// accessesIn is buffer b's accesses directly in loop c, each its index's
// stride over c's iteration and the rest; false where an access is not of
// that form.
func (k *Kernel) accessesIn(b, base, c int) ([]cursorAccess, bool) {
	it := k.iterOf(c)
	var out []cursorAccess
	add := func(a access, ix int) bool {
		t, tc := k.terms(ix)
		s := t[it]
		delete(t, it)
		for n := range t {
			if k.varies(n, c) {
				return false
			}
		}
		out = append(out, cursorAccess{access: a, stride: s, rest: t, c: tc})
		return true
	}
	for i, nd := range k.Nodes {
		if nd.Op == OpLoad && k.Bufs[nd.Buf].Base == base && k.loopOf(i) == c {
			if !add(access{node: i, store: -1}, nd.A) {
				return nil, false
			}
		}
	}
	for i, s := range k.Stores {
		if s.Barrier || s.Run || s.At != c || k.Bufs[s.Buf].Base != base {
			continue
		}
		if !add(access{node: -1, store: i}, s.Ix) {
			return nil, false
		}
	}
	return out, true
}

// loopOf is the innermost loop node i varies in, -1 for none.
func (k *Kernel) loopOf(i int) int {
	best := -1
	for c := range k.Loops {
		if k.varies(i, c) && (best < 0 || k.within(c, best)) {
			best = c
		}
	}
	return best
}

func addTerms(a, b map[int]int64) map[int]int64 {
	out := map[int]int64{}
	for n, v := range a {
		out[n] += v
	}
	for n, v := range b {
		out[n] += v
		if out[n] == 0 {
			delete(out, n)
		}
	}
	return out
}

// sameTerms reports whether two sums of terms are the same value: equal
// once each term is written as its own terms.
func (k *Kernel) sameTerms(a, b map[int]int64) bool {
	ea, eb := k.expand(a), k.expand(b)
	if len(ea) != len(eb) {
		return false
	}
	for n, v := range ea {
		if eb[n] != v {
			return false
		}
	}
	return true
}

func (k *Kernel) expand(a map[int]int64) map[int]int64 {
	out := map[int]int64{}
	for n, v := range a {
		t, _ := k.terms(n)
		if len(t) == 1 && t[n] == 1 {
			out[n] += v
			continue
		}
		for m, w := range t {
			out[m] += v * w
		}
	}
	for n, v := range out {
		if v == 0 {
			delete(out, n)
		}
	}
	return out
}
