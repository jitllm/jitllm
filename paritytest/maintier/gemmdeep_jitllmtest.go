//go:build jitllmtest

package maintier

// ForceGemmDeepForTest sets the deep m16n8 tiles' default for every tier
// opened after it (Config.GemmDeep where the caller left it 0): 1 on, -1 off.
// It exists so the model gates, which open their tiers with their own
// options, can run with the deep tiles selected; a release build has no way
// to call it.
func ForceGemmDeepForTest(v int) { gemmDeepDefault = v }
