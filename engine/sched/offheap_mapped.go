//go:build linux || darwin || windows

package sched

// offHeapPlatform: format/jlm maps its weight memory here (offheap.MapOffHeap).
const offHeapPlatform = true
