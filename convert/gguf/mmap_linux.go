package gguf

import (
	"fmt"
	"os"
	"syscall"
)

// mmapFile maps the whole file read-only and private. Read-only file mappings
// are exempt from RLIMIT_DATA (Linux >= 4.7), which is what lets jitllm cap its
// anonymous memory hard while leaving weights unlimited.
func mmapFile(path string, noWarm bool) ([]byte, func() error, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	st, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	n := st.Size()
	if n <= 0 {
		return nil, nil, fmt.Errorf("gguf: %s: empty file", path)
	}
	if n != int64(int(n)) {
		return nil, nil, fmt.Errorf("gguf: %s: %d bytes does not fit in int", path, n)
	}

	b, err := syscall.Mmap(int(f.Fd()), 0, int(n), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		return nil, nil, fmt.Errorf("gguf: mmap %s (%d bytes): %w", path, n, err)
	}
	// No readahead advice (fadvise SEQUENTIAL, MAP_POPULATE, WILLNEED): measured
	// cold, it bought nothing on a dense model (fault-around is already at the
	// device limit) and was worse on a mixture, where whole-file advice reads
	// experts the router never selects. See docs/engineering-history/placement.md
	// ("No mmap of files") for the madvise measurement, and model-correctness.md
	// 14h for what the missing advice means to a cold comparison row.
	//
	// Fault the mapping in from several goroutines behind the caller, bounded
	// by the budget; see warm and warmSpan. The stop must run before the
	// unmap, or those goroutines read freed pages.
	stop := func() {}
	if !noWarm {
		stop = warm(b)
	}
	return b, func() error { stop(); return syscall.Munmap(b) }, nil
}
