package cpudev

import (
	"runtime"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/device/cpu"
)

// TestCPUArgWordsAreNotScanned holds in every word of the composed CPU
// path's call blocks an address the collector refuses as a pointer: into a
// span it has freed, as an address past a slice's last element can be. A
// block allocated as cpu.Args, whose fields are pointers, ends the process
// at the next collection ("found bad pointer in Go heap", synth-q4e's
// TestSpecGreedyOnDevice); the words are not scanned.
func TestCPUArgWordsAreNotScanned(t *testing.T) {
	big := make([]byte, 8<<20)
	gone := uint64(uintptr(unsafe.Pointer(&big[0]))) + 4096
	big = nil
	runtime.GC()
	runtime.GC()
	args := argBlocks(16)
	for i := range args {
		base := unsafe.Pointer(&args[i])
		for off := uintptr(0); off < unsafe.Sizeof(cpu.Args{}); off += 8 {
			*(*uint64)(unsafe.Add(base, off)) = gone + uint64(off)
		}
	}
	for range 4 {
		runtime.GC()
	}
	runtime.KeepAlive(args)
}
