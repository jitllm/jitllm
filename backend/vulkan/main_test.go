package vulkan

import (
	"fmt"
	"os"
	"testing"
)

// TestMain holds one instance for the binary's life, as every process that
// opens and closes devices again and again does (Driver): the ICDs stay
// loaded while the tests open and close their own. It enumerates, as every
// test in this package does, so it is made here rather than through backend's
// selection, which this package's tests cannot import.
func TestMain(m *testing.M) {
	keep, _ := NewDriver()
	code := m.Run()
	if first, now := keep.Vanished(); first > 0 {
		fmt.Fprintf(os.Stderr, "FAIL: the Vulkan loader enumerated %d device(s) when this binary started and "+
			"%d at its end -- every Vulkan test after that skipped or ran elsewhere\n", first, now)
		code = 1
	}
	keep.Close()
	os.Exit(code)
}
