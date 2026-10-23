//go:build jitllmtest

package ptx

// BreakGridBarrierForTest makes every grid barrier lowered after it count its
// arrival and leave without waiting, and returns the setting it replaced: the
// violation kernels.GridProbe must fail under (RULE 10). Not safe against
// concurrent lowering.
func BreakGridBarrierForTest(on bool) bool {
	old := gridBroken
	gridBroken = on
	return old
}

// BreakDepWaitForTest makes every dependency-counter wait lowered after it
// return without waiting, and returns the setting it replaced: the violation
// the dep block's gate must fail under (RULE 10). Not safe against concurrent
// lowering.
func BreakDepWaitForTest(on bool) bool {
	old := depBroken
	depBroken = on
	return old
}
