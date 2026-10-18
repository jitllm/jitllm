package model

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
)

// prepStep is one op between a block's q/k/v projections and its attention,
// lowered on the host: a clamp, a norm (per head or over the whole
// projection), a rotary or the cache append. Each step names the generated
// kernel it runs and the operand it reads, both read off the graph op; the
// steps run in the graph's order, so a block's attention prologue is composed
// from its ops rather than one routine that re-derives them from the Config.
type prepStep struct {
	kind graph.OpKind
	// arg is the projection the op reads: 0 q, 1 k, 2 v.
	arg int
	// unit is a norm's width: HeadDim for a per-head norm, the projection's
	// width for a whole one.
	unit int
	neox bool
	// xd turns each element by its own coordinate (RoPESplit, XD-RoPE).
	xd    bool
	bound float32
	// ones is a norm with no weight (Llama 4's after the rotary): the
	// generated RMSNorm against a ones vector.
	ones bool
}

// prepArg is which projection a step's input traces back to.
func prepArg(of map[graph.ValueID]int, v graph.ValueID) (int, bool) {
	a, ok := of[v]
	return a, ok
}

// lowerPrep lowers the ops from the q/k/v projections to the attention, the
// ops lowerDense has matched in [from, to), into steps. qkv are the three
// projections' values.
func lowerPrep(g *graph.Graph, from, to int, qkv [3]graph.ValueID) ([]prepStep, error) {
	of := map[graph.ValueID]int{qkv[0]: 0}
	if qkv[1] >= 0 {
		of[qkv[1]] = 1
	}
	// v as k's projection (VFromK): the row is copied before the prologue,
	// so the ops on v's chain read the copy -- v's weightless norm, or the
	// append's v, where either reads k's raw projection.
	vFromK := qkv[2] >= 0 && qkv[2] == qkv[1]
	if qkv[2] >= 0 && !vFromK {
		of[qkv[2]] = 2
	}
	var out []prepStep
	for i := from; i < to; i++ {
		o := g.Ops[i]
		a, ok := prepArg(of, o.In[0])
		if vFromK && o.In[0] == qkv[1] && o.Kind == graph.OpHeadNorm && len(o.Weight) == 0 {
			a = 2
		}
		if !ok {
			return nil, fmt.Errorf("model: block %d op %d (%s) reads no projection", g.Block, i, o.Kind)
		}
		st := prepStep{kind: o.Kind, arg: a}
		switch o.Kind {
		case graph.OpClamp:
			st.bound = float32(o.Attr.Bound)
		case graph.OpHeadNorm:
			st.unit, st.ones = o.Attr.HeadDim, len(o.Weight) == 0
		case graph.OpNorm:
			st.unit = g.Values[o.In[0]].Width
		case graph.OpRope:
			st.neox, st.xd = o.Attr.Layout == graph.RopeNeox, o.Attr.Layout == graph.RopeXD
		case graph.OpKVAppend:
			b, ok := prepArg(of, o.In[1])
			if vFromK && o.In[1] == qkv[1] {
				b = 2
			}
			if a != 1 || !ok || b != 2 {
				return nil, fmt.Errorf("model: block %d op %d appends no k and v", g.Block, i)
			}
		default:
			return nil, fmt.Errorf("model: block %d op %d (%s) has no host prologue lowering", g.Block, i, o.Kind)
		}
		if len(o.Out) > 0 {
			of[o.Out[0]] = a
		}
		out = append(out, st)
	}
	return out, nil
}

// kNormFirst is steps with k's head norm moved before k's rotary: the
// violation xdKNormFirst runs.
func kNormFirst(steps []prepStep) []prepStep {
	r, n := -1, -1
	for i, st := range steps {
		switch {
		case st.kind == graph.OpRope && st.arg == 1 && r < 0:
			r = i
		case st.kind == graph.OpHeadNorm && st.arg == 1 && r >= 0 && n < 0:
			n = i
		}
	}
	if r < 0 || n < 0 {
		return steps
	}
	out := append([]prepStep(nil), steps[:r]...)
	out = append(out, steps[n])
	out = append(out, steps[r:n]...)
	return append(out, steps[n+1:]...)
}

// runPrep runs a block's lowered prologue over one row: q, k and v are the
// row's projections, cs its rotary table, kvSlot and kvPos where its k and v
// are cached. Each step is one generated kernel (per head for a per-head
// norm), in the graph's order.
func (s *State) runPrep(l *layer, steps []prepStep, q, k, v, cs []float32, li, kvSlot, kvPos int) {
	c := s.c
	hd := c.HeadDimAt(li)
	// A selecting block's row carries the indexer's key past the attention's
	// heads (msaRow): the norms and the rotary turn the heads, the append
	// writes the whole row.
	kvd := c.KVDimAt(li)
	rows := [3][]float32{q, k[:min(len(k), kvd)], v[:min(len(v), kvd)]}
	if s.m.faults.xdKNormFirst {
		steps = kNormFirst(steps)
	}
	for _, st := range steps {
		x := rows[st.arg]
		switch st.kind {
		case graph.OpClamp:
			nn.Clamp32JIT(x, st.bound)
		case graph.OpHeadNorm, graph.OpNorm:
			w := l.qNorm
			if st.ones {
				w = s.m.onesHD[:st.unit]
			} else if st.arg == 1 {
				w = l.kNorm
			}
			for o := 0; o+st.unit <= len(x); o += st.unit {
				s.rmsnorm(x[o:o+st.unit], x[o:o+st.unit], w, c.RMSEps)
			}
		case graph.OpRope:
			t0 := s.tick()
			if st.xd {
				nn.RoPESplit32JIT(x, hd, cs)
			} else {
				nn.RoPE32JIT(x, hd, cs, st.neox)
			}
			s.tock(opRoPE, t0)
		case graph.OpKVAppend:
			s.kvWritable(li, kvPos)
			s.kv.layers[li].write(s.kvlAt(li), kvSlot, kvPos, k, v)
		}
	}
}
