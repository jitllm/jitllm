package graph

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
)

// StreamRead is the input u reads -- its program over the streams it names:
// stream k as it is now (xs), stream 0 as the block received it (in), the
// running residual (run) or the block's per-layer inputs (side) -- or run
// where it states none, and the values its program exports.
func StreamRead(b *Builder, f Facts, u *jlm.SubStreams, xs []ValueID, in, run, side ValueID,
	ops func([]ValueID, *jlm.TowerOp) (ValueID, error)) (ValueID, []ValueID, error) {
	if u.Read == nil {
		return run, nil, nil
	}
	ins := make([]ValueID, len(u.From))
	for j, k := range u.From {
		switch k {
		case jlm.StreamIn:
			ins[j] = in
		case jlm.StreamRun:
			ins[j] = run
		case jlm.StreamSide:
			if side == 0 {
				return 0, nil, fmt.Errorf("a read of the per-layer inputs in a block that has none")
			}
			ins[j] = side
		default:
			ins[j] = xs[k]
		}
	}
	if len(u.Read.Ops) == 0 {
		ex := make([]ValueID, len(u.Exports))
		for i, e := range u.Exports {
			ex[i] = ins[e]
		}
		return ins[u.Read.Out], ex, nil
	}
	vals, err := LowerValues(b, nil, u.Read, ins, 1, f, ops)
	if err != nil {
		return 0, nil, err
	}
	ex := make([]ValueID, len(u.Exports))
	for i, e := range u.Exports {
		ex[i] = vals[e]
	}
	return vals[u.Read.Out], ex, nil
}

// StreamWrites is every stream after a sublayer whose output is y: u's
// write of stream k over it as it is (xs[k]), y, the read's exports and what
// u's writes share (Post), the stream as it is where u writes none.
func StreamWrites(b *Builder, f Facts, u *jlm.SubStreams, xs []ValueID, y ValueID, ex []ValueID) ([]ValueID, error) {
	out := slices.Clone(xs)
	shared := ex
	if u.Post != nil {
		v, err := LowerStated(b, u.Post, append([]ValueID{y}, ex...), f)
		if err != nil {
			return nil, fmt.Errorf("what the writes share: %w", err)
		}
		shared = append(slices.Clone(ex), v)
	}
	for k, p := range u.Writes {
		if p == nil {
			continue
		}
		v, err := LowerStated(b, p, append([]ValueID{xs[k], y}, shared...), f)
		if err != nil {
			return nil, fmt.Errorf("stream %d's write: %w", k, err)
		}
		out[k] = v
	}
	return out, nil
}

// StreamJoin is the running residual after a sublayer: u's join over run
// and the sublayer's output y, or their sum where it states none.
func StreamJoin(b *Builder, f Facts, u *jlm.SubStreams, run, y ValueID, sum func(x, y ValueID) ValueID) (ValueID, error) {
	if u.Join == nil {
		return sum(run, y), nil
	}
	return LowerStated(b, u.Join, []ValueID{run, y}, f)
}

// LowerStated lowers a stated program over in; one of no op returns the
// input it names.
func LowerStated(b *Builder, p *jlm.Program, in []ValueID, f Facts) (ValueID, error) {
	if len(p.Ops) == 0 {
		return in[p.Out], nil
	}
	return LowerProgram(b, nil, p, in, 1, f)
}
