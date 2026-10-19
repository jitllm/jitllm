package tier

import "github.com/jitllm/jitllm/jit/gpu/kernels"

// kernID names a kernel builder in the shape-keyed cache (convKey): an
// integer, so a lookup hashes no string and builds none; its name is for
// messages only.
type kernID uint16

const (
	kidNone kernID = iota
	kidAct
	kidAdd
	kidAddhalf
	kidAffinerows
	kidAttnaccrows
	kidBiasAct
	kidDwconv
	kidDwconv1drows
	kidIm2col
	kidLnapply
	kidLnpart
	kidLnvar
	kidMqaacc
	kidMqascores
	kidPadrows
	kidQuantize
	kidRelposscores
	kidRmsnormrows
	kidRowsoftmax
	kidSigmoidmul
)

var kernNames = [...]string{"none", "act", "add", "addhalf", "affinerows", "attnaccrows", "biasact", "dwconv", "dwconv1drows", "im2col", "lnapply", "lnpart", "lnvar", "mqaacc", "mqascores", "padrows", "quantize", "relposscores", "rmsnormrows", "rowsoftmax", "sigmoidmul"}

func (k kernID) String() string {
	if int(k) < len(kernNames) {
		return kernNames[k]
	}
	return "kernel"
}

// convKey is a compiled kernel's shape: its builder, an activation where the
// builder takes one, its sizes and an epsilon or bound.
type convKey struct {
	id  kernID
	act kernels.ActKind
	a   [8]int
	eps float32
}
