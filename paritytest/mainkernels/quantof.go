package mainkernels

import "github.com/jitllm/jitllm/format/quant"

// QuantOf maps a GGUF weight type onto the device packer's.
//
// A type that is not here is not an error: F32 and F16 norms, biases and
// anything else the packer does not handle are carried verbatim by a
// container, so false means "read these bytes as they are", not "refuse".
func QuantOf(t quant.Type) (Quant, bool) {
	switch t {
	case quant.Q4_0:
		return Q4_0, true
	case quant.Q5_0:
		return Q5_0, true
	case quant.Q5_1:
		return Q5_1, true
	case quant.Q8_0:
		return Q8_0, true
	case quant.Q4_K:
		return Q4_K, true
	case quant.Q3_K:
		return Q3_K, true
	case quant.Q5_K:
		return Q5_K, true
	case quant.Q6_K:
		return Q6_K, true
	case quant.MXFP4:
		return MXFP4, true
	case quant.TQ2_0:
		return TQ2, true
	case quant.TQ2_F32:
		return TQ2F, true
	}
	return 0, false
}
