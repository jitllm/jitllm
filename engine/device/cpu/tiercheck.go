//go:build !jitllmtest

package cpu

// checkTier is the shipping no-op. The test build's twin asserts that the
// kernel about to run was generated for the tier this host reports.
func checkTier(*Code) {}
