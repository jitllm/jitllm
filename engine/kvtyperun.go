package engine

import (
	"slices"

	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// KVType is the format this session's cache holds.
func (s *Instance) KVType() model.KVType {
	switch s.kvFmt {
	case nn.KVF16:
		return model.KVF16
	case nn.KVQ8:
		return model.KVQ8_0
	case nn.KVQ4_0:
		return model.KVQ4_0
	case nn.KVQ4_1:
		return model.KVQ4_1
	case nn.KVQ5_0:
		return model.KVQ5_0
	case nn.KVQ5_1:
		return model.KVQ5_1
	case nn.KVIQ4NL:
		return model.KVIQ4_NL
	case nn.KVBF16:
		return model.KVBF16
	}
	return model.KVF32
}

// KVBytesPerToken is what one position of history costs in this instance's
// cache, summed over every layer that keeps one: the format's row bytes times
// the layer's kv heads, K and V (an MLA row once). A recurrent layer costs
// nothing a position.
func (s *Instance) KVBytesPerToken() int {
	n := 0
	for li := s.lo; li < s.hi && li < len(s.kv.layers); li++ {
		if s.kv.layers[li].p == 0 || s.c.KVShared(li) {
			continue
		}
		l := s.kvlAt(li)
		b := l.slots(l.kvDim()) * 4
		if s.kv.layers[li].latent {
			n += b
		} else {
			n += 2 * b
		}
	}
	return n
}

// kvConvOps is the ops that narrow rows of floats into history format f's
// words and widen them back, and their attributes: binary16 or bfloat16
// halves, q8 rows or llama.cpp's blocks.
func kvConvOps(f nn.KVFmt) (narrow, widen graph.OpKind, at graph.Attrs) {
	at = graph.Attrs{Block: f.Block(), BF16: f == nn.KVBF16}
	switch {
	case at.Block != 0:
		return graph.OpBlockRows, graph.OpWidenBlockRows, at
	case f.Quantized():
		return graph.OpQ8Rows, graph.OpWidenQ8Rows, at
	}
	return graph.OpHalves, graph.OpWidenHalves, at
}

// kvRowWords is the words kvConvOps' narrowing writes for a row of w floats.
func kvRowWords(f nn.KVFmt, w int) int {
	switch {
	case f.Block() != 0:
		return graph.BlockRowWords(f.Block(), w)
	case f.Quantized():
		return graph.Q8RowWords(w)
	}
	return w / 2
}

// typedAppends is g with each append to the block's own history taking its
// key and value through f's narrowing and widening first, g itself where it
// appends to none. A row is narrowed whole, so its blocks are its heads'
// where a head is whole blocks.
func typedAppends(g *graph.Graph, f nn.KVFmt) *graph.Graph {
	narrow, widen, at := kvConvOps(f)
	b := &graph.Builder{G: &graph.Graph{Version: g.Version, Block: g.Block, Values: slices.Clone(g.Values), In: g.In, Out: g.Out, Routed: g.Routed}}
	typed := func(v graph.ValueID) graph.ValueID {
		x := b.G.Values[v]
		h := b.Emit(narrow, []graph.ValueID{v}, []graph.Result{{Name: x.Name + ".typed", Width: kvRowWords(f, x.Width), Elem: graph.ElemU32}}, nil, at, nil, nil)[0]
		return b.Emit(widen, []graph.ValueID{h}, []graph.Result{{Name: x.Name + ".widened", Width: x.Width}}, nil, at, nil, nil)[0]
	}
	n := 0
	for _, o := range g.Ops {
		if o.Kind == graph.OpKVAppend && len(o.Writes) == 1 && o.Writes[0].Hist == 0 {
			lo := len(b.G.Ops)
			in := slices.Clone(o.In)
			for i, v := range o.In {
				if i > 0 && v == o.In[0] {
					in[i] = in[0]
					continue
				}
				in[i] = typed(v)
			}
			o.In = in
			if added := len(b.G.Ops) - lo; b.G.Routed.Hi > lo {
				b.G.Routed.Hi += added
				if b.G.Routed.Lo >= lo {
					b.G.Routed.Lo += added
				}
			}
			n++
		}
		b.G.Ops = append(b.G.Ops, o)
	}
	if n == 0 {
		return g
	}
	return b.G
}

// deviceGraph is block li's graph as a card is offered it (planFor). A
// card's history pages hold floats; under a quantized or bfloat16 history
// each append to the block's own history takes its key and value through
// the type's narrowing and widening first (typedAppends), so the pages hold
// the type's values from the moment a row is written and a step's own
// attention reads its newest row as the type holds it: the CPU device's
// answer. The type's bytes on a card are owed; a quantized history whose
// head rows are not whole blocks of 32 is still written as floats there.
func (s *Instance) deviceGraph(li int) *graph.Graph {
	f := s.kvFmt
	if !(f.Quantized() || f == nn.KVBF16) || f.Quantized() && s.kvlAt(li).headDim%32 != 0 {
		return s.m.Composing().BlockGraph(li)
	}
	return s.m.DerivedBlockGraph(li, int(f), func(g *graph.Graph) *graph.Graph { return typedAppends(g, f) })
}
