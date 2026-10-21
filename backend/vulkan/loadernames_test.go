package vulkan

import (
	"strings"
	"testing"
)

// TestLoaderNamesCoverEveryPlatform is the CUDA gate's twin: the loader list
// must carry a name for Linux, Windows (vulkan-1.dll) and macOS. On macOS the
// loader (libvulkan.1.dylib) must be present and preferred over MoltenVK, which
// is the ICD and skips layer discovery.
func TestLoaderNamesCoverEveryPlatform(t *testing.T) {
	for _, c := range []struct{ os, suffix string }{
		{"linux", ".so"},
		{"windows", ".dll"},
		{"macOS", ".dylib"},
	} {
		found := ""
		for _, n := range loaderNames {
			if strings.Contains(n, c.suffix) {
				found = n
				break
			}
		}
		if found == "" {
			t.Errorf("no %s candidate (%q) in %v -- Vulkan cannot be opened on that "+
				"platform, and it fails as an ABSENCE rather than an error: the tier "+
				"reports no device and a working GPU looks like no GPU",
				c.os, c.suffix, loaderNames)
			continue
		}
		t.Logf("%-7s -> %s", c.os, found)
	}
	// The loader must be preferred over the ICD, which is an order property and
	// not a membership one -- ffi.Open takes the first name that loads.
	li, mi := -1, -1
	for i, n := range loaderNames {
		switch {
		case strings.Contains(n, "libvulkan.1.dylib"):
			li = i
		case strings.Contains(n, "libMoltenVK"):
			mi = i
		}
	}
	if li >= 0 && mi >= 0 && li > mi {
		t.Errorf("libMoltenVK is tried before the macOS loader in %v -- opening the "+
			"ICD directly skips layer discovery and any second ICD", loaderNames)
	}
}
