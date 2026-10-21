package cpu

import "errors"

// errQuantActOwed is what a tier without the activation-quantize kernel
// returns. It is deliberately not ErrNoSSEKernel, which marks the tier
// incomplete and makes model.Open refuse the host (cpu.SSEPending).
var errQuantActOwed = errors.New("jit: this tier has no activation-quantize kernel yet")

// EmitQuantActSSE and EmitQuantActNarrowSSE are the SSE tier's entries for the
// two window shapes. sse_quantact.go has the bodies; both are bit-identical to
// cpu.QuantizeQ8Window and QuantizeHalfSums over every packed format.
func EmitQuantActSSE(half bool) ([]byte, error) {
	return must("quantact_sse", emitQuantActSSEBody(half))
}

func EmitQuantActNarrowSSE(half bool) ([]byte, error) {
	return must("quantact_narrow_sse", emitQuantActNarrowSSEBody(half))
}

// EmitArgmaxSSE is the SSE tier's greedy argmax; sse_argmax.go has the body.
// The tie fold needs PMINSD, which is SSE4.1 and inside that tier's budget.
func EmitArgmaxSSE() ([]byte, error) { return must("argmax_sse", emitArgmaxSSEBody()) }
