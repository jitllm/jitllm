package compose

import (
	"fmt"

	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// Epi is a row group a former applies to each of its outputs before the
// store: Out is the node the former stores, NInput nodes
// read the former's own value, a BufRow loads at the output's element, a
// BufVec at its row, and the group's stores write further outputs at the
// element. A group with a reduction cannot be an epilogue: one output does
// not see its row.
type Epi struct {
	G   *Row
	Out int

	params []ir.Value
	// x and out are G with its activations expanded into primitive nodes,
	// and Out's place in it (Check sets them).
	x   *Row
	out int
}

// Check refuses a group that is not an epilogue.
func (e *Epi) Check() error {
	for i, n := range e.G.Nodes {
		if n.Kind == NReduce || n.Kind == NColumn || n.Kind == NRowIdx || n.Kind == NLoad && n.Indexed {
			return fmt.Errorf("compose: node %d of an epilogue reads beyond its element", i)
		}
	}
	if e.Out < 0 || e.Out >= len(e.G.Nodes) || e.G.Q8 != nil {
		return fmt.Errorf("compose: an epilogue storing node %d", e.Out)
	}
	x, at, err := e.G.expanded()
	if err != nil {
		return err
	}
	e.x, e.out = x, at[e.Out]
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

// Apply emits the group at one output. Check runs first.
func (e *Epi) Apply(b *ir.Builder, v, at, row ir.Value) ir.Value {
	g := e.x
	memo := map[int]ir.Value{}
	var val func(n int) ir.Value
	val = func(n int) ir.Value {
		if x, ok := memo[n]; ok {
			return x
		}
		nd := g.Nodes[n]
		var x ir.Value
		switch nd.Kind {
		case NInput:
			x = v
		case NConst:
			x = b.ConstF32(nd.F)
		case NUConst:
			x = b.Const(ir.U32, int64(nd.At))
		case NLoad:
			switch g.Bufs[nd.Buf].Kind {
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
			case OpExp:
				x = b.Exp(a)
			case OpSqrt:
				x = b.Sqrt(a)
			case OpRsqrt:
				x = b.Div(ir.F32, b.ConstF32(1), b.Sqrt(a))
			case OpAbs:
				x = b.Max(ir.F32, a, b.Sub(ir.F32, b.ConstF32(0), a))
			}
		case NBinary:
			a := val(nd.A)
			switch nd.Op {
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
		case NLess:
			x = b.Lt(ir.F32, val(nd.A), val(nd.B))
		case NSelect:
			x = b.Select(ir.F32, val(nd.C), val(nd.A), val(nd.B))
		}
		memo[n] = x
		return x
	}
	for _, st := range g.Stores {
		b.Store(e.params[st.Buf], at, val(st.Val), 0)
	}
	return val(e.out)
}
