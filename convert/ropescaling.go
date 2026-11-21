package convert

import (
	"encoding/binary"
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// The published rope scaling methods the rope_scaling statement implements
// (mjRopeScaling), each once, for every model that states it.

// ropeFreqsTensor is per-pair rotary factors as the container carries them,
// rope_freqs.weight beside the blocks.
func ropeFreqsTensor(f []float32) jlm.Tensor {
	b := make([]byte, 4*len(f))
	for i, x := range f {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return jlm.Tensor{Role: jlm.RoleRopeFreqs, Block: jlm.DenseBlock, Index: -1,
		Type: jlm.TypeF32, NDim: 1, Dims: [4]uint64{uint64(len(f))}, Data: b, Name: "rope_freqs.weight"}
}
