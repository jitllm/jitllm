package engine

import (
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
