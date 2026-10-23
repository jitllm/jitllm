package cuda

import (
	"strings"
	"testing"
)

// TestDriverNamesCoverEveryPlatform asserts the candidate list carries a name
// the CUDA driver actually ships under on each OS jitllm builds for.
//
// A missing Windows name made Windows report no CUDA device, silently. Running
// on Linux cannot catch that, so the gate is over the list rather than over a
// successful load. macOS is absent: NVIDIA has shipped no macOS driver since
// CUDA 10.2.
func TestDriverNamesCoverEveryPlatform(t *testing.T) {
	for _, c := range []struct{ os, suffix string }{
		{"linux", ".so"}, // matches .so and .so.1
		{"windows", ".dll"},
	} {
		found := ""
		for _, n := range driverNames {
			if strings.Contains(n, c.suffix) {
				found = n
				break
			}
		}
		if found == "" {
			t.Errorf("no %s candidate (%q) in %v -- the driver cannot be opened on "+
				"that platform and the failure is SILENT: Open returns an error, the "+
				"tier reports no device, and the app shows a box with a working GPU "+
				"as having none", c.os, c.suffix, driverNames)
			continue
		}
		t.Logf("%-7s -> %s", c.os, found)
	}
	// The WSL path is its own case: a Linux binary under WSL finds the driver
	// only at that absolute path.
	wsl := false
	for _, n := range driverNames {
		if strings.HasPrefix(n, "/usr/lib/wsl/") {
			wsl = true
		}
	}
	if !wsl {
		t.Errorf("no /usr/lib/wsl/ candidate in %v -- a Linux build under WSL "+
			"resolves libcuda nowhere else", driverNames)
	}
}
