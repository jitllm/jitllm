package jit

import (
	"errors"
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
)

func decompose(b Block) ([]Prim, error) {
	var prims []Prim
	var errs []error
	for i, o := range b.Graph.Ops {
		ps, err := decomposeOp(b, o)
		if err != nil {
			errs = append(errs, fmt.Errorf("op %d (%s): %w", i, o.Kind, err))
			continue
		}
		prims = append(prims, ps...)
	}
	return prims, errors.Join(errs...)
}

func decomposeOp(b Block, o graph.Op) ([]Prim, error) {
	value := func(v graph.ValueID) Operand {
		return Operand{Kind: OperandValue, Value: v, Name: b.Graph.Values[v].Name}
	}
	switch o.Kind {
	case graph.OpMatVec:
		if len(o.Weight) != 1 || len(o.In) != 1 || len(o.Out) != 1 {
			return nil, fmt.Errorf("a matvec of %d weights", len(o.Weight))
		}
		w := o.Weight[0]
		f, ok := b.Formats[w.Role]
		if !ok {
			return nil, fmt.Errorf("no format for %s", w.Role)
		}
		return []Prim{{Kind: PrimContract, Rows: w.Rows, K: w.K,
			In:  []Operand{{Kind: OperandWeight, Role: w.Role, Format: f, Name: w.Role.String()}, value(o.In[0])},
			Out: value(o.Out[0])}}, nil
	case graph.OpAdd:
		if len(o.In) != 2 || len(o.Out) != 1 {
			return nil, fmt.Errorf("an add of %d operands", len(o.In))
		}
		w := b.Graph.Values[o.Out[0]].Width
		return []Prim{{Kind: PrimMap, Fn: OpAdd, Rows: w, In: []Operand{value(o.In[0]), value(o.In[1])}, Out: value(o.Out[0])}}, nil
	}
	return nil, errNoDecomposition
}

var errNoDecomposition = errors.New("no decomposition into primitives yet")
