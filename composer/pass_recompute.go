package composer

import (
	"slices"

	"github.com/jitllm/jitllm/composer/ir"
)

// Recompute runs a map in each map that reads its output, at the coordinates
// each reads it at, where every reader of the output is such a map and the
// output is the part's own: the map's kernel dropped and its output never
// written. A reader's read of the output becomes the map's value there, its
// own reads of what nothing between them writes.
type Recompute struct {
	// MapsOnly runs it only in maps, never in a reduction reading it.
	MapsOnly bool
}

func (Recompute) Name() string { return "recompute" }

func (p Recompute) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	out := append([]*Kernel(nil), ks...)
	for i := 0; i < len(out); i++ {
		readers, ok := recomputed(out, i, p.MapsOnly)
		if !ok {
			continue
		}
		pk := out[i]
		o := pk.Bufs[pk.Stores[0].Buf].Operand
		for _, j := range readers {
			c := out[j].clone()
			for l := range c.Nodes {
				nd := c.Nodes[l]
				if nd.Op != OpLoad || c.Bufs[nd.Buf].Operand != o {
					continue
				}
				ax := nd.Ax
				if ax == nil {
					ax = c.coordsOf(nd.A, nd.Imm, pk.Space)
				}
				leaf := func(_ int, x Node) (int, bool) {
					if x.Op == OpCoord {
						return ax[x.A], true
					}
					return 0, false
				}
				c.redirectAll(l, copyInto(c, pk, pk.Stores[0].Val, leaf))
			}
			c.compact()
			c.Passes = append(append([]string(nil), c.Passes...), p.Name())
			out[j] = c
		}
		out = append(out[:i], out[i+1:]...)
		i--
	}
	return out, nil
}

// coordsOf is the coordinates in a space of extents sp of the flat index
// node ix plus off.
func (n *Kernel) coordsOf(ix int, off int64, sp []int) []int {
	U := ir.U32
	if off != 0 {
		ix = n.bin(OpAdd, U, ix, n.u32(int(off)))
	}
	ax := make([]int, len(sp))
	for d := len(sp) - 1; d > 0; d-- {
		ax[d] = n.bin(OpRem, U, ix, n.u32(sp[d]))
		ix = n.bin(OpDiv, U, ix, n.u32(sp[d]))
	}
	ax[0] = ix
	return ax
}

// recomputeReads is the most places a reader may read the map's output at:
// a pair of halves, not a gather of many, each a copy of the map.
const recomputeReads = 2

// recomputed is the kernels of ks after ks[i] that read its output, where
// ks[i] is a map of no loops storing one value of the part's own at its
// element and not itself a map run in another, every reader is a map of no
// loops reading it at coordinates, at most recomputeReads of them, and no
// kernel from ks[i] to its last reader writes what ks[i] reads.
func recomputed(ks []*Kernel, i int, mapsOnly bool) ([]int, bool) {
	pk := ks[i]
	if pk.Kind != PrimMap || pk.Work || len(pk.Loops) != 0 || len(pk.Stores) != 1 || pk.Few > 0 || pk.Many > 0 {
		return nil, false
	}
	st := pk.Stores[0]
	o := pk.Bufs[st.Buf].Operand
	if !pk.storesOwn(st) || o.Kind != OperandValue || o.Out || pk.reads(o) {
		return nil, false
	}
	again := slices.Contains(pk.Passes, "recompute")
	var readers []int
	for j := range ks {
		if j == i || !ks[j].reads(o) {
			continue
		}
		c := ks[j]
		if j < i || c.Kind != PrimMap && (mapsOnly || c.Kind != PrimReduce) || c.Work || c.Kind == PrimMap && len(c.Loops) != 0 || c.Part != pk.Part || c.Of != pk.Of || c.writes(o) {
			return nil, false
		}
		reads := 0
		for _, nd := range c.Nodes {
			if nd.Op == OpLoad && c.Bufs[nd.Buf].Operand == o {
				reads++
			}
			if sh := c.Bufs[nd.Buf].Shape; nd.Op == OpLoad && c.Bufs[nd.Buf].Operand == o && (nd.Ax != nil && (len(nd.Ax) != len(pk.Space) || nd.Imm != 0) ||
				sh != nil && !slices.Equal(sh, pk.Space)) {
				return nil, false
			}
		}
		if reads > recomputeReads || again && reads > 1 || c.Kind == PrimReduce && reads > 1 {
			return nil, false
		}
		readers = append(readers, j)
	}
	if len(readers) == 0 {
		return nil, false
	}
	for j := i + 1; j <= readers[len(readers)-1]; j++ {
		for _, b := range pk.Bufs {
			if b.Operand != o && ks[j].writesStorage(b.Operand) || b.Kind == BufIn && !writtenBy(ks, b.Operand) && writesBound(ks[j], b.Operand) {
				return nil, false
			}
		}
	}
	return readers, true
}

// writesBound says k writes an output a call may bind over the part's
// input in: its own stream's in a block of several, any other part's.
func writesBound(k *Kernel, in Operand) bool {
	if in.Kind != OperandValue || in.Out {
		return false
	}
	for _, b := range k.Bufs {
		if b.Operand.Out && b.Operand.Of == in.Of && (in.Stream == 0 || b.Operand.Stream == in.Stream) && k.writes(b.Operand) {
			return true
		}
	}
	return false
}
