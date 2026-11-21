//go:build !linux

package backend

// threadID is -1 where the thread cannot be named cheaply, and cudaDev.do then
// skips its owner-thread check; CUDA hosts here are Linux.
func threadID() int64 { return -1 }
