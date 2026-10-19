package tier

import "github.com/jitllm/jitllm/jit/gpu/kernels"

// kernID names a kernel builder in the shape-keyed cache (convKey): an
// integer, so a lookup hashes no string and builds none; its name is for
// messages only.
type kernID uint16

const (
	kidNone kernID = iota
	kidAct
	kidActMul
	kidAdd
	kidAddScaled
	kidAddhalf
	kidAffinerows
	kidAttnaccrows
	kidBiasAct
	kidClamp
	kidClampat
	kidDwconv
	kidDwconv1drows
	kidIm2col
	kidLaureljoin
	kidHeadnorm
	kidCopyat
	kidRope
	kidRopesplit
	kidLnapply
	kidLnpart
	kidLnvar
	kidMqaacc
	kidMqascores
	kidNormapply
	kidNormpart
	kidPadrows
	kidQuantize
	kidRelposscores
	kidRestride
	kidRmsnormrows
	kidRowsoftmax
	kidScale
	kidSharedexpertadd
	kidSigmoidmul
	kidSplitheadgate
)

var kernNames = [...]string{"none", "act", "actmul", "add", "add_scaled", "addhalf", "affinerows", "attnaccrows", "biasact", "clamp", "clampat", "dwconv", "dwconv1drows", "im2col", "laureljoin", "headnorm", "copyat", "rope", "ropesplit", "lnapply", "lnpart", "lnvar", "mqaacc", "mqascores", "normapply", "normpart", "padrows", "quantize", "relposscores", "restride", "rmsnormrows", "rowsoftmax", "scale", "sharedexpertadd", "sigmoidmul", "splitheadgate"}

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
