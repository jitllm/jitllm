//go:build jitllmtest

package maintier

// PersistByDefaultForTest makes every tier opened after it start with
// Config.Persist set, and returns the default it replaced: the regression
// gates (relocation, paging, batching, sessions, seam moves, the allocation
// gate) run through the persistent path without each naming it. Not safe
// against a concurrent open.
func PersistByDefaultForTest(on bool) bool {
	old := persistDefault
	persistDefault = on
	return old
}

// PersistDepsByDefaultForTest is PersistByDefaultForTest for
// Config.PersistDeps: every tier opened after it orders its persistent
// kernels with dependency counters (Persist must be on as well).
func PersistDepsByDefaultForTest(on bool) bool {
	old := persistDepsDefault
	persistDepsDefault = on
	return old
}

// PersistDepLaunchesForTest is how many dependency-counter persistent kernels
// this process has launched.
func PersistDepLaunchesForTest() int64 { return persistDepTotal.Load() }

// PersistLaunchesForTest is how many persistent kernels this process has
// launched, for a gate that must show the path ran.
func PersistLaunchesForTest() int64 { return persistTotal.Load() }

// SwapGateUpForTest makes every fused gate/up FFN (Config.GateUpFuse) hand
// down the two rows the wrong way round, and returns the setting it replaced.
func SwapGateUpForTest(on bool) bool {
	old := gateUpSwapped
	gateUpSwapped = on
	return old
}

// SwapGateUp2ForTest makes every Gate2 FFN (Config.GateUp2) read the two
// weights the wrong way round, and returns the setting it replaced.
func SwapGateUp2ForTest(on bool) bool {
	old := gateUp2Swapped
	gateUp2Swapped = on
	return old
}

// BreakAddNormForTest makes every folded block-end add (Config.NoAddNormFold
// false) add the wrong row, and returns the setting it replaced.
func BreakAddNormForTest(on bool) bool {
	old := addNormBroken
	addNormBroken = on
	return old
}

// SwapQKVGemmForTest makes every q/k/v grid (qkvGemm) read k's and v's
// weights the wrong way round, and returns the setting it replaced.
func SwapQKVGemmForTest(on bool) bool {
	old := qkvGemmSwapped
	qkvGemmSwapped = on
	return old
}
