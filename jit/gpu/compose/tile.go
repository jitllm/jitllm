package compose

import (
	"maps"
	"slices"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// offSplit is an index as its part the tile's envs share, core (-1 for
// none), plus a sum of tile offsets times their coefficients and a constant.
type offSplit struct {
	core  int
	terms map[int]int64
	c     int64
}

// liftOffsets rewrites every index that reads a tile offset as its shared
// core plus its offsets: (x + o)*k is x*k + o*k, (x + o) + y is (x + y) + o.
// The core is formed once for the tile and each env adds a constant, which
// the access takes as its offset.
func (g *Row) liftOffsets() {
	n0 := len(g.Nodes)
	memo := make([]*offSplit, n0)
	var split func(n int) *offSplit
	split = func(n int) *offSplit {
		if s := memo[n]; s != nil {
			return s
		}
		nd := g.Nodes[n]
		s := &offSplit{core: n}
		switch {
		case g.isTileOff(n):
			s = &offSplit{core: -1, terms: map[int]int64{n: 1}}
		case nd.Kind == NUConst:
			s = &offSplit{core: -1, c: int64(nd.At)}
		case nd.Kind == NUBinary && g.typeOf(n) == ir.U32 && (nd.Op == OpAdd || nd.Op == OpSub || nd.Op == OpMul):
			a, b := split(nd.A), split(nd.B)
			if len(a.terms) == 0 && len(b.terms) == 0 {
				break
			}
			sign := int64(1)
			if nd.Op == OpSub {
				sign = -1
			}
			switch {
			case nd.Op == OpMul && b.core < 0 && len(b.terms) == 0:
				s = a.scaled(g, b.c)
			case nd.Op == OpMul && a.core < 0 && len(a.terms) == 0:
				s = b.scaled(g, a.c)
			case nd.Op == OpMul, nd.Op == OpSub && a.core < 0 && b.core >= 0:
			default:
				s = &offSplit{core: a.core, terms: map[int]int64{}, c: a.c + sign*b.c}
				for k, v := range a.terms {
					s.terms[k] += v
				}
				for k, v := range b.terms {
					s.terms[k] += sign * v
				}
				if b.core >= 0 {
					if s.core < 0 {
						s.core = b.core
					} else {
						s.core = g.UBin(nd.Op, s.core, b.core)
					}
				}
			}
		}
		memo[n] = s
		return s
	}
	for n := 0; n < n0; n++ {
		nd := g.Nodes[n]
		if nd.Kind != NUBinary {
			continue
		}
		s := split(n)
		if len(s.terms) == 0 {
			continue
		}
		off := -1
		add := func(x int) {
			if off < 0 {
				off = x
			} else {
				off = g.UBin(OpAdd, off, x)
			}
		}
		for _, k := range slices.Sorted(maps.Keys(s.terms)) {
			switch v := s.terms[k]; v {
			case 0:
			case 1:
				add(k)
			default:
				add(g.UBin(OpMul, k, g.UConst(int(v))))
			}
		}
		if off < 0 {
			add(g.UConst(0))
		}
		if s.c != 0 {
			add(g.UConst(int(s.c)))
		}
		if s.core < 0 {
			g.Nodes[n] = g.Nodes[off]
		} else {
			g.Nodes[n] = Node{Kind: NUBinary, Op: OpAdd, A: s.core, B: off}
		}
	}
}

// scaled is s times k, its core multiplied once.
func (s *offSplit) scaled(g *Row, k int64) *offSplit {
	r := &offSplit{core: s.core, terms: map[int]int64{}, c: s.c * k}
	for t, v := range s.terms {
		r.terms[t] = v * k
	}
	if s.core >= 0 && k != 1 {
		r.core = g.UBin(OpMul, s.core, g.UConst(int(k)))
	}
	return r
}
