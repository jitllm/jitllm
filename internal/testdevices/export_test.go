package testdevices

import "testing"

// SetForTest puts spec in force as the selection until the test ends, as if
// JITLLM_TEST_DEVICES had been set to it. An empty spec is no selection.
func SetForTest(t *testing.T, spec string) {
	t.Helper()
	t.Setenv(Var, spec)
}

// RestrictionForTest is the backend restriction with the ledger in the
// caller's hands: onViolation is told each device a test named outside the
// selection, onNarrowed each one a search skipped.
func RestrictionForTest(onViolation, onNarrowed func(name string)) func(string, int, bool) (int, error) {
	return restriction(onViolation, onNarrowed)
}
