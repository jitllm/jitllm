package compose

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// Epi is a row group a former applies to each of its outputs before the
// store (kernels.Epilogue): Out is the node the former stores, NInput nodes
// read the former's own value, a BufRow loads at the output's element, a
// BufVec at its row, and the group's stores write further outputs at the
// element. A group with a reduction cannot be an epilogue: one output does
// not see its row.
type Epi struct {
	G   *Row
	Out int

	params []ir.Value
}

// Check refuses a group that is not an epilogue.
func (e *Epi) Check() error {
	for i, n := range e.G.Nodes {
		if n.Kind == NUnary && n.Op == OpAct {
			if err := kernels.ActElemOK(n.Act); err != nil {
				return fmt.Errorf("compose: node %d of an epilogue: %w", i, err)
			}
		}
		if n.Kind == NReduce || n.Kind == NColumn || n.Kind == NRowIdx || n.Kind == NLoad && n.Indexed {
			return fmt.Errorf("compose: node %d of an epilogue reads beyond its element", i)
		}
	}
	if e.Out < 0 || e.Out >= len(e.G.Nodes) || e.G.Q8 != nil {
		return fmt.Errorf("compose: an epilogue storing node %d", e.Out)
	}
	return nil
}

// Encode is the epilogue's structure, for a cache key beside the former's.
func (e *Epi) Encode() []uint32 {
	return append(e.G.Encode(Schedule{}), 2, uint32(e.Out))
}

// Params declares the group's buffers.
func (e *Epi) Params(b *ir.Builder) {
	e.params = e.params[:0]
	for i, bf := range e.G.Bufs {
		t := ir.F32
		if bf.Kind == BufQ8 || bf.Kind == BufU32 {
			t = ir.U32
		}
		e.params = append(e.params, b.Param(fmt.Sprintf("e%d", i), t))
	}
}

// Apply emits the group at one output.
func (e *Epi) Apply(b *ir.Builder, v, at, row ir.Value) ir.Value {
	memo := map[int]ir.Value{}
	var val func(n int) ir.Value
	val = func(n int) ir.Value {
		if x, ok := memo[n]; ok {
			return x
		}
		nd := e.G.Nodes[n]
		var x ir.Value
		switch nd.Kind {
		case NInput:
			x = v
		case NConst:
			x = b.ConstF32(nd.F)
		case NUConst:
			x = b.Const(ir.U32, int64(nd.At))
		case NLoad:
			switch e.G.Bufs[nd.Buf].Kind {
			case BufRow:
				x = b.Load(ir.F32, e.params[nd.Buf], at, 0)
			case BufVec:
				x = b.Load(ir.F32, e.params[nd.Buf], row, 0)
			default:
				x = b.Load(ir.F32, e.params[nd.Buf], b.Const(ir.U32, int64(nd.At)), 0)
			}
		case NUnary:
			a := val(nd.A)
			switch nd.Op {
			case OpAct:
				x = kernels.ActFragment(b, a, nd.Act)
			case OpRsqrt:
				x = b.Div(ir.F32, b.ConstF32(1), b.Sqrt(a))
			case OpAbs:
				x = b.Max(ir.F32, a, b.Sub(ir.F32, b.ConstF32(0), a))
			}
		case NBinary:
			a := val(nd.A)
			switch nd.Op {
			case OpActMul:
				x = kernels.ActMulFragment(b, nd.Act, a, func() ir.Value { return val(nd.B) })
			case OpFma:
				x = b.Fma(a, val(nd.B), val(nd.C))
			default:
				c := val(nd.B)
				switch nd.Op {
				case OpAdd:
					x = b.Add(ir.F32, a, c)
				case OpSub:
					x = b.Sub(ir.F32, a, c)
				case OpMul:
					x = b.Mul(ir.F32, a, c)
				case OpDiv:
					x = b.Div(ir.F32, a, c)
				case OpMax:
					x = b.Max(ir.F32, a, c)
				case OpMin:
					x = b.Min(ir.F32, a, c)
				}
			}
		case NSelect:
			x = b.Select(ir.F32, val(nd.C), val(nd.A), val(nd.B))
		}
		memo[n] = x
		return x
	}
	for _, st := range e.G.Stores {
		b.Store(e.params[st.Buf], at, val(st.Val), 0)
	}
	return val(e.Out)
}
