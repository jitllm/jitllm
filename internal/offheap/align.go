package offheap

import (
	"os"
)

// HostAlign is the alignment every packed array starts on, and the multiple its
// length is padded to.
//
// A copy works at any alignment; an import does not: VK_EXT_external_memory_host
// wants minImportedHostPointerAlignment (0x1000 here) and Metal's
// newBufferWithBytesNoCopy: wants whole pages. It is the larger of the two
// because the second and third arrays start at offsets inside the region, and
// Apple Silicon's page is 16 KiB: a constant 4096 made Metal refuse two arrays
// of three and silently fall back to copying.
func HostAlign() uint64 {
	if a := uint64(os.Getpagesize()); a > hostAlignFloor {
		return a
	}
	return hostAlignFloor
}

// hostAlignFloor is minImportedHostPointerAlignment as the Vulkan devices here
// report it; a host with smaller pages still lays its arrays out on it.
const hostAlignFloor = 4096
