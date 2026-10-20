package compose

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// Segment is one group of a horizontally fused launch: G under S over
// Blocks workgroups, Share naming for each of its buffers the launch's
// shared operand it is (-1, or past its end, for one of its own).
type Segment struct {
	G      *Row
	S      Schedule
	Blocks int
	Share  []int
}

// EmitSegments forms groups of one workgroup width into one kernel: the
// shared operands its first parameters, then each group's own buffers;
// group i over its own run of workgroups, guarded by a loop of one trip
// there and none elsewhere (a trip count uniform over a workgroup is the
// IR's conditional), its workgroups counted from the run's first. Workgroup
// arrays of one size are one array: the groups run in different
// workgroups.
func EmitSegments(segs []Segment, shared int) (*ir.Kernel, error) {
	if len(segs) < 2 {
		return nil, fmt.Errorf("compose: %d segments", len(segs))
	}
	G := segs[0].S.Group
	prep := make([]*Row, len(segs))
	for i, sg := range segs {
		if sg.S.Group != G || sg.Blocks < 1 || len(sg.Share) > len(sg.G.Bufs) {
			return nil, fmt.Errorf("compose: segment %d of %d threads, %d workgroups, against %d", i, sg.S.Group, sg.Blocks, G)
		}
		g, err := sg.G.prepared(sg.S)
		if err != nil {
			return nil, err
		}
		prep[i] = g
	}
	b := ir.New("rowseg", [3]int{G, 1, 1})
	ops := make([]ir.Value, shared)
	for j := range ops {
		ops[j] = ir.Value(0)
	}
	type arr struct {
		k BufKind
		n int
	}
	arrays := map[arr]ir.Value{}
	for i, g := range prep {
		for j, bf := range g.Bufs {
			if g.shared(j) && bf.Of == 0 {
				a := arr{bf.Kind, bf.N}
				if _, ok := arrays[a]; !ok {
					arrays[a] = b.Shared(fmt.Sprintf("s%d_%d", i, j), bufType(bf.Kind), bf.N)
				}
			}
		}
	}
	for i, g := range prep {
		for j, sl := range segs[i].Share {
			if sl < 0 || sl >= shared || j >= len(g.Bufs) || g.shared(j) {
				continue
			}
			if ops[sl] == 0 {
				ops[sl] = b.Param(fmt.Sprintf("a%d", sl), bufType(g.Bufs[j].Kind))
			}
		}
	}
	start := 0
	for i, g := range prep {
		lo, hi := b.Const(ir.U32, int64(start)), b.Const(ir.U32, int64(start+segs[i].Blocks))
		c := b.CTAID()
		zero, one := b.Const(ir.U32, 0), b.Const(ir.U32, 1)
		b.LoopN(b.Select(ir.U32, b.Lt(ir.U32, c, hi), b.Select(ir.U32, b.Lt(ir.U32, c, lo), zero, one), zero))
		cta := func() ir.Value { return b.Sub(ir.U32, b.CTAID(), lo) }
		sh := segs[i].Share
		params := g.bind(b, func(j int) (ir.Value, bool) {
			bf := g.Bufs[j]
			if g.shared(j) {
				if bf.Of > 0 {
					return 0, false
				}
				return arrays[arr{bf.Kind, bf.N}], true
			}
			if j < len(sh) && sh[j] >= 0 && sh[j] < shared {
				return ops[sh[j]], true
			}
			return 0, false
		})
		if _, err := g.emitIn(b, params, segs[i].S, cta, false); err != nil {
			return nil, fmt.Errorf("compose: segment %d: %w", i, err)
		}
		b.EndLoop()
		start += segs[i].Blocks
	}
	return simplify(b.Done()), nil
}
