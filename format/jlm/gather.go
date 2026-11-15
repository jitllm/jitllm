package jlm

import "fmt"

// The operands a TOpGather's index reads (an ArgIn's Index).
const (
	GatherRow     = iota // the output row: past the op's bound, the last row written
	GatherOutRows        // the result's rows
	GatherInRows         // In[0]'s rows, 0 with no input
	GatherLive           // the step's live rows
	GatherPos            // the row's position
	GatherGridW          // the request's patch grid's width
	GatherGridH          // the request's patch grid's height
	GatherMark           // the request's run of table rows before: its first plus one, 0 for none
	GatherMarkN          // that run's length
	GatherTrail          // the request's run of table rows after: its first plus one, 0 for none
	GatherTrailN         // that run's length
	GatherSamples        // the request's clip's samples
	GatherCol            // the output element's column
	GatherOperands
)

// GatherNone is the source row no source holds: a gather writes zero there.
const GatherNone = 1<<32 - 1

// retired names a code an older container stated a row gather's geometry
// under, which this format states as a TOpGather's index; "" for any other.
func retired(c OpCode) string {
	switch c {
	case 13:
		return "an untile"
	case 39:
		return "a clip mask"
	}
	return ""
}

// validGather reports why op is not a gather this format defines: its index
// over unsigned words, its operands the gather's, its bound a prefix of it.
func (op *TowerOp) validGather() error {
	e := op.Expr
	if e == nil {
		return fmt.Errorf("states no index")
	}
	if err := e.ValidOver(GatherOperands); err != nil {
		return err
	}
	for k, t := range e.Terms {
		switch t.Fn {
		case FnAdd, FnSub, FnMul, FnDiv, FnMax, FnMin, FnLess, FnWhere:
		default:
			return fmt.Errorf("term %d of its index is %v, not a function of row indices", k, t.Fn)
		}
		if t.Reduce {
			return fmt.Errorf("term %d of its index reduces", k)
		}
		for _, a := range t.Args {
			if a.Kind == ArgStored || a.Kind == ArgWidth {
				return fmt.Errorf("term %d of its index reads a %d operand", k, a.Kind)
			}
		}
	}
	if op.Span < 0 || int(op.Span) > len(e.Terms) || op.Width < 0 {
		return fmt.Errorf("bounds its rows at term %d of %d, %d rows an input row", op.Span, len(e.Terms), op.Width)
	}
	return nil
}
