package tier

import (
	"github.com/jitllm/jitllm/format/jlm"
)

// A convolution-path block (one ConvLayers runs: the call's rows on the card
// from block to block, a program built at admission) lowered from its graph
// rather than from a list written for its kind. The walk is the device
// program's: each graph op becomes typed program ops, every value takes the
// first free buffer of a pool coloured by liveness (RULE 13 the colouring's
// constraint: an op's output never takes a buffer it reads), and each op
// records its launches through the path's own builders (kern, launch, mm), so
// a block of any composition of these primitives builds -- the FastConformer
// is one.

// numPool is the liveness pool's size.
const numPool = 12

// poolBufs is the scratch each pool slot names.
var poolBufs = [numPool]cbuf{cT1, cT2, cE1, cE2, cQ, cK, cV, cAO, cX1, cX2, cX3, cCol}

// vecAt is a block vector by role: the scratch name its upload is reached by.
type vecAt struct {
	role jlm.Role
	b    cbuf
}
