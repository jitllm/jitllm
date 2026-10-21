package cpu

import (
	"encoding/binary"
	"math"

	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// PackedRowScratch is how many bytes of Args.Scratch EmitPackedRow uses: the
// gathered words, then the split nibbles.
const PackedRowScratch = 96

// PackedRowConsts is the Scr block EmitPackedRow reads for q:
//
//	[0,16)     the code table, or zero
//	16         0x0F0F0F0F     20  0xFF
//	24         scOff (f32)    28  biasK (f32)
//	32         (1<<hi)-1
//	[64, 64+4*sub)         per element l, the secondary plane's shift:
//	                       8*(l%4) + hi*((l/4)%(8/hi))
//	[64+4*sub, 64+8*sub)   the same, negated, for NEON's USHL
func PackedRowConsts(q kernels.Quant) []byte {
	sub, _, biasK, _ := kernels.Layout(q)
	hi := kernels.HiPlane(q)
	_, scOff := kernels.ScaleLayout(q)
	b := make([]byte, 64+8*sub)
	if c := kernels.Codes(q); c != nil {
		copy(b[0:16], c[:])
	}
	le := binary.LittleEndian
	le.PutUint32(b[16:], 0x0F0F0F0F)
	le.PutUint32(b[20:], 0xFF)
	le.PutUint32(b[24:], math.Float32bits(float32(scOff)))
	le.PutUint32(b[28:], math.Float32bits(biasK))
	if hi > 0 {
		le.PutUint32(b[32:], 1<<uint(hi)-1)
		lanes := 8 / hi
		for l := 0; l < sub; l++ {
			sh := int32(8*(l%4) + hi*((l/4)%lanes))
			le.PutUint32(b[64+4*l:], uint32(sh))
			le.PutUint32(b[64+4*sub+4*l:], uint32(-sh))
		}
	}
	return b
}
