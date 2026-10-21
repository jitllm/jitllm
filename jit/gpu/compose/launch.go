package compose

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// BindScratch is a launch parameter bound to the split scratch: a k split's
// planes.
const BindScratch BindKind = 63

// Launch is one kernel of a group's plan: its cache key and how to emit it,
// its grid, what each parameter binds to, and the scratch floats its split
// planes take (0 for none). Segmented says the key is a segmented kernel's.
type Launch struct {
	Key           []uint32
	Emit          func() (*ir.Kernel, error)
	Groups, Width int
	Binds         []Bind
	Planes        int
	Segmented     bool
}

// Launches is group g's plan on target t: a row group's one kernel, a
// contraction's kernels under its facts (a k split's planes and their sum),
// or a segmented group's one kernel over its members.
func (g *Group) Launches(t Target) ([]Launch, error) {
	if len(g.Segs) > 0 {
		return g.segmented(t)
	}
	f := g.Facts
	f.Target = t
	var ps []Planned
	var err error
	if g.Sched != nil {
		s := *g.Sched
		s.Target = t
		ps, err = g.Row.PlanAt(s)
	} else {
		ps, err = g.Row.Plan(f)
	}
	if err != nil {
		return nil, err
	}
	if !g.Contract && len(ps) != 1 {
		return nil, fmt.Errorf("compose: a row group planned as %d launches", len(ps))
	}
	var out []Launch
	for _, p := range ps {
		s, pg := p.S, p.G
		l := Launch{Key: pg.Encode(s), Emit: func() (*ir.Kernel, error) { return pg.Emit(s) }, Width: s.Group}
		if g.Contract {
			l.Groups = (p.Threads + s.Group - 1) / s.Group
			for _, b := range p.Bufs {
				if b < 0 {
					l.Binds = append(l.Binds, Bind{Kind: BindScratch})
					continue
				}
				l.Binds = append(l.Binds, g.Binds[b])
			}
			if s.Split > 1 {
				l.Planes = g.Row.Width * g.Row.Rows * s.Split
			}
		} else {
			l.Groups, l.Binds = g.Row.Rows, g.Binds
			if s.Flat {
				l.Groups = (p.Threads + s.Group - 1) / s.Group
			}
		}
		out = append(out, l)
	}
	return out, nil
}

// segmented is a segmented group's one kernel (EmitSegments): each member
// planned to one launch, the int8 operand they share its first parameters.
func (g *Group) segmented(t Target) ([]Launch, error) {
	var segs []Segment
	var binds []Bind
	var key []uint32
	var bufs [][]int // each segment's planned buffers, by its group's
	shared := map[Bind]int{}
	blocks, width := 0, 0
	for i := range g.Segs {
		m := &g.Segs[i]
		f := m.Facts
		f.Target = t
		ps, err := m.Row.Plan(f)
		if err != nil {
			return nil, err
		}
		if len(ps) != 1 {
			return nil, fmt.Errorf("compose: a segment planned as %d launches", len(ps))
		}
		p := ps[0]
		if sp := max(m.Facts.DecodeSplit, 1); sp > 1 && p.S.Split != sp {
			return nil, fmt.Errorf("compose: a segment split %d planned at %d", sp, p.S.Split)
		}
		share := make([]int, len(p.Bufs))
		for j, b := range p.Bufs {
			share[j] = -1
			if b < 0 {
				return nil, fmt.Errorf("compose: a segment with split planes")
			}
			bd := m.Binds[b]
			if bd.Kind != BindQ8 && bd.Kind != BindQ8X {
				continue
			}
			k, ok := shared[bd]
			if !ok && i == 0 {
				k, ok = len(shared), true
				shared[bd] = k
			}
			if ok {
				share[j] = k
			}
		}
		n := (p.Threads + p.S.Group - 1) / p.S.Group
		segs = append(segs, Segment{G: p.G, S: p.S, Blocks: n, Share: share})
		bufs = append(bufs, p.Bufs)
		key = append(append(key, uint32(n)), p.G.Encode(p.S)...)
		blocks, width = blocks+n, p.S.Group
	}
	for bd, k := range shared {
		for len(binds) <= k {
			binds = append(binds, Bind{})
		}
		binds[k] = bd
	}
	for i, s := range segs {
		for j, sh := range s.Share {
			if sh < 0 {
				binds = append(binds, g.Segs[i].Binds[bufs[i][j]])
			}
		}
	}
	return []Launch{{Key: key, Emit: func() (*ir.Kernel, error) { return EmitSegments(segs, len(shared)) },
		Groups: blocks, Width: width, Binds: binds, Segmented: true}}, nil
}
