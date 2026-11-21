package engine

import (
	"github.com/jitllm/jitllm/composer"
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
		if s.kv.layers[li].p == 0 || s.readsOther(li) {
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
// halves, q8 rows or GGUF quant blocks.
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

// historySelected reports that some block of [lo, hi) reads its history
// through a selection (graph.OpSelect), which ranks the history's keys
// discretely.
func historySelected(m composer.Model, lo, hi int) bool {
	for li := lo; li < hi; li++ {
		g := m.BlockGraph(li)
		if g == nil {
			continue
		}
		for _, o := range g.Ops {
			if o.Kind == graph.OpSelect {
				return true
			}
		}
	}
	return false
}

// historyRow is a position's row in the history graph g reads: the key row
// g appends to its own (graph.OpKVAppend's In[0]), or the source block's
// where g reads another's history (StateRef.From); 0 where it reads none.
func historyRow(m composer.Model, g *graph.Graph) int {
	if g == nil {
		return 0
	}
	for _, o := range g.Ops {
		if o.Kind == graph.OpKVAppend && len(o.Writes) == 1 && o.Writes[0].Hist == 0 && o.Writes[0].From == 0 {
			return g.Values[o.In[0]].Width
		}
	}
	for _, o := range g.Ops {
		for _, r := range o.Reads {
			if r.State == graph.StateKV && r.Hist == 0 && r.From > 0 {
				return historyRow(m, m.BlockGraph(r.From-1))
			}
		}
	}
	return 0
}
