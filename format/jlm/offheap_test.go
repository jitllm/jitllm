package jlm

import (
	"runtime/debug"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// BenchmarkFrame prices a page frame three ways at two page sizes: a fresh Go
// allocation (the heap returned to the OS first, so it is fresh memory, as a
// frame's first allocation is), a fresh anonymous mapping, and a frame reused
// from the free list. Every arm is placed as take() places a frame -- huge-page
// advice on its 2 MiB interior -- and each iteration writes one byte per 4 KiB,
// which is what a page-in's read does to a frame first, so first-touch faults
// are in the price.
//
//	go test ./format/jlm -run XXX -bench Frame -benchtime 40x
func BenchmarkFrame(b *testing.B) {
	for _, size := range []int{4 << 20, 64 << 20} {
		touch := func(p []byte) {
			placeMemory(p, false, nil)
			for i := 0; i < len(p); i += 4096 {
				p[i] = 1
			}
		}
		b.Run("make/"+mib(size), func(b *testing.B) {
			b.SetBytes(int64(size))
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				debug.FreeOSMemory()
				b.StartTimer()
				touch(alignedBuf(size))
			}
		})
		b.Run("mmap/"+mib(size), func(b *testing.B) {
			b.SetBytes(int64(size))
			for i := 0; i < b.N; i++ {
				p, _, _ := kernels.MapOffHeap(size)
				touch(p)
				b.StopTimer()
				kernels.UnmapOffHeap(p)
				b.StartTimer()
			}
		})
		b.Run("reuse/"+mib(size), func(b *testing.B) {
			b.SetBytes(int64(size))
			f := newFake(1, uint64(size), 0, Align)
			p := f.take(uint64(size))
			for i := 0; i < b.N; i++ {
				f.free[uint64(size)] = append(f.free[uint64(size)], p)
				p = f.take(uint64(size))
				touch(p)
			}
		})
	}
}

func mib(n int) string {
	switch n >> 20 {
	case 4:
		return "4MiB"
	case 64:
		return "64MiB"
	}
	return "other"
}
