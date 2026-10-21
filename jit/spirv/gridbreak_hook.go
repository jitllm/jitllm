//go:build jitllmtest

package spirv

// BreakGridBarrierForTest makes every grid barrier lowered after it count its
// arrival and leave without waiting, and returns the setting it replaced: the
// violation kernels.GridProbe must fail under (RULE 10). Not safe against
// concurrent lowering.
func BreakGridBarrierForTest(on bool) bool {
	old := gridBroken
	gridBroken = on
	return old
}
