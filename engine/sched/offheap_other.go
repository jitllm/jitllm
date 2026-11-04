//go:build !linux && !darwin && !windows

package sched

// offHeapPlatform: no anonymous mmap here, so the weights are on the heap.
const offHeapPlatform = false
