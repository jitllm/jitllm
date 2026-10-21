//go:build (amd64 || arm64) && !linux && !darwin

package cpu

import "testing"

// imgGuard without a guard page: a copy, so the numeric halves still run.
func imgGuard(t *testing.T, src []byte) []byte {
	t.Helper()
	return append([]byte(nil), src...)
}
