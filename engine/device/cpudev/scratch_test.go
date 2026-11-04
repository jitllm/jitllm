package cpudev

import (
	"testing"

	"github.com/jitllm/jitllm/composer"
)

// TestCallsOwnScratchPastAnAxisOfOneCall holds callsOwnScratch to a code
// whose calls' leading axis is a single call: the scratch moves along every
// axis of more than one call, so each call writes its own slice, and a code
// whose address stands still along such an axis shares it.
func TestCallsOwnScratchPastAnAxisOfOneCall(t *testing.T) {
	o := composer.Operand{Kind: composer.OperandScratch, Name: "s"}
	code := func(per []int) composer.Code {
		return composer.Code{Calls: []int{1, 32}, Args: []composer.CallArg{{Operand: o, Per: per}}}
	}
	if !callsOwnScratch(code([]int{0, 16}), o, 32*16*4) {
		t.Error("calls of [1 32] moving 16 words a call along the second axis do not own their scratch")
	}
	if callsOwnScratch(code([]int{0, 0}), o, 16*4) {
		t.Error("calls of [1 32] whose address stands still own their scratch")
	}
	if callsOwnScratch(code([]int{0, 16}), o, 33*16*4) {
		t.Error("calls reaching past their slices own their scratch")
	}
}
