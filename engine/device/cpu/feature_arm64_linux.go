//go:build arm64 && linux

package cpu

import (
	"encoding/binary"
	"os"
)

// hwcapASIMDDP is HWCAP_ASIMDDP, the AT_HWCAP bit for FEAT_DotProd on Linux.
const hwcapASIMDDP = 1 << 20

// probeDotProd reads AT_HWCAP out of /proc/self/auxv rather than depending on
// golang.org/x/sys/cpu. The file is a flat array of (type, value) pairs, eight
// bytes each on a 64-bit host, terminated by type 0.
func probeDotProd() bool {
	b, err := os.ReadFile("/proc/self/auxv")
	if err != nil {
		// Unreadable means absent, not present: guessing present turns a
		// restricted /proc into a SIGILL instead of a slow answer.
		return false
	}
	for i := 0; i+16 <= len(b); i += 16 {
		t := binary.LittleEndian.Uint64(b[i:])
		v := binary.LittleEndian.Uint64(b[i+8:])
		if t == 0 {
			break // AT_NULL
		}
		if t == 16 { // AT_HWCAP
			return v&hwcapASIMDDP != 0
		}
	}
	return false
}
