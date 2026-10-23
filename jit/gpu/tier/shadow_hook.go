//go:build jitllmtest

package tier

// ShadowForTest makes every device opened after it shadow its blocks into
// log (shadow.go), and returns the function that stops it. Not safe against
// a concurrent open.
func ShadowForTest(log *ShadowLog) (restore func()) {
	old := shadowDefault
	shadowDefault = log
	return func() { shadowDefault = old }
}
