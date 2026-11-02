//go:build (amd64 || arm64) && jitllmtest

package cpu

// ForceNoDotProdForTest makes the arm64 probe answer "no FEAT_DotProd" from now
// on, so every A64 kernel emitted afterwards widens its SDOTs (sdotemu.go), and
// returns the previous setting. Like ForceTierForTest it is called before
// NewJIT/Open: a kernel already emitted keeps the form it was emitted in. On
// amd64 it changes only what the A64 emitters produce, which nothing executes.
func ForceNoDotProdForTest(on bool) bool {
	old := forceNoDotProd
	forceNoDotProd = on
	return old
}
