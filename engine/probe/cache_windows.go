//go:build windows

package probe

import (
	"encoding/binary"
	"fmt"
	"math/bits"
	"slices"
	"syscall"
	"unsafe"
)

// cacheSizes is the data caches the first logical processor reads through,
// nearest first, as GetLogicalProcessorInformationEx's cache records state
// them: a level is shared where its mask names more than two processors, a
// core's own and its sibling's.
func cacheSizes() ([]cache, error) {
	proc := syscall.NewLazyDLL("kernel32.dll").NewProc("GetLogicalProcessorInformationEx")
	if err := proc.Find(); err != nil {
		return nil, fmt.Errorf("probe: %w", err)
	}
	const relationCache = 2
	var n uint32
	proc.Call(relationCache, 0, uintptr(unsafe.Pointer(&n)))
	if n == 0 {
		return nil, fmt.Errorf("probe: GetLogicalProcessorInformationEx states no cache records")
	}
	buf := make([]byte, n)
	if r, _, err := proc.Call(relationCache, uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(&n))); r == 0 {
		return nil, fmt.Errorf("probe: GetLogicalProcessorInformationEx: %w", err)
	}
	type level struct {
		level int
		cache
	}
	var found []level
	le := binary.LittleEndian
	for off := 0; off+56 <= int(n); {
		size := int(le.Uint32(buf[off+4:]))
		if size < 56 || off+size > int(n) {
			return nil, fmt.Errorf("probe: a cache record of %d bytes at %d of %d", size, off, n)
		}
		r := buf[off : off+size]
		off += size
		if le.Uint32(r) != relationCache {
			continue
		}
		const instruction, trace = 1, 3
		if t := le.Uint32(r[16:]); t == instruction || t == trace {
			continue
		}
		mask, group := le.Uint64(r[40:]), le.Uint16(r[48:])
		if group != 0 || mask&1 == 0 {
			continue
		}
		found = append(found, level{int(r[8]), cache{bytes: int64(le.Uint32(r[12:])), shared: bits.OnesCount64(mask) > 2}})
	}
	if len(found) == 0 {
		return nil, fmt.Errorf("probe: no data cache record names the first processor")
	}
	slices.SortFunc(found, func(a, b level) int { return a.level - b.level })
	out := make([]cache, len(found))
	for i, f := range found {
		out[i] = f.cache
	}
	return out, nil
}
