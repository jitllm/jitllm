//go:build arm64

package cpu

import "testing"

// xieluVEXGate has nothing to check on arm64: no SSE tier runs here.
func xieluVEXGate(*testing.T, string, []byte) {}
