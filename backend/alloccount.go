package backend

import "sync/atomic"

// AllocCounter is a device that counts what its Alloc'd buffers hold: added at
// Alloc, given back at the buffer's Free. An imported host buffer
// (HostImport) is not counted -- the device did not allocate it.
//
// Allocated is the bytes asked for, and Footprint what the card spends on
// them: a driver hands memory out in pages, so a buffer costs its size rounded
// up to the driver's granularity. The tier charges what it allocates for
// itself (the block scratches, the staging) by Allocated's difference across
// the build, and the rounding of every buffer by Footprint less Allocated,
// which on a model whose matrices are not page multiples is no small thing:
// a 2 GB model on CUDA spends 250 MB more than its buffers ask for.
//
// FootprintOf is what one n-byte buffer would cost the card, so a caller can
// price a buffer before asking for it.
type AllocCounter interface {
	Allocated() uint64
	Footprint() uint64
	FootprintOf(n int) uint64
}

// allocCount is the counter every backend embeds.
type allocCount struct{ n, foot atomic.Int64 }

// Allocated implements AllocCounter.
func (a *allocCount) Allocated() uint64 { return nonNeg(a.n.Load()) }

// Footprint implements AllocCounter.
func (a *allocCount) Footprint() uint64 { return nonNeg(a.foot.Load()) }

func nonNeg(v int64) uint64 {
	if v > 0 {
		return uint64(v)
	}
	return 0
}

// counted is one buffer's share of its device's count, given back once.
type counted struct {
	cnt     *allocCount
	n, foot atomic.Int64
}

// take counts an n-byte buffer that costs the card foot bytes.
func (a *allocCount) take(n, foot int) *counted {
	a.n.Add(int64(n))
	a.foot.Add(int64(foot))
	c := &counted{cnt: a}
	c.n.Store(int64(n))
	c.foot.Store(int64(foot))
	return c
}

// give returns the buffer's bytes to the count; a second Free gives nothing.
func (c *counted) give() {
	if c == nil || c.cnt == nil {
		return
	}
	if n := c.n.Swap(0); n > 0 {
		c.cnt.n.Add(-n)
	}
	if f := c.foot.Swap(0); f > 0 {
		c.cnt.foot.Add(-f)
	}
}

// roundUp rounds n up to a multiple of g.
func roundUp(n, g int) int { return (n + g - 1) / g * g }

// The drivers' granularity, measured by allocating 32 buffers of each size and
// reading the card's free memory either side:
//
//	           4 KiB   100 KiB   1 MiB   1 MiB+4 KiB   2 MiB+1   10.6 MiB
//	CUDA      64 KiB   128 KiB   1 MiB   2 MiB         4 MiB     12 MiB
//	Vulkan    64 KiB   128 KiB   1 MiB   1.06 MiB      2.06 MiB  10.63 MiB
//
// cuMemAlloc takes a large allocation in 2 MiB pages and a small one in 64
// KiB pieces; the Vulkan driver hands out 64 KiB pieces at every size.
const (
	smallPage = 64 << 10
	largePage = 2 << 20
)

// cudaFootprint is what an n-byte cuMemAlloc costs the card.
func cudaFootprint(n int) int {
	if n <= 1<<20 {
		return roundUp(n, smallPage)
	}
	return roundUp(n, largePage)
}

// vulkanFootprint is what an n-byte device allocation costs the card.
func vulkanFootprint(n int) int { return roundUp(n, smallPage) }

// FootprintOf implements AllocCounter.
func (c *cudaDev) FootprintOf(n int) uint64 { return uint64(cudaFootprint(n)) }

// FootprintOf implements AllocCounter.
func (v *vkDev) FootprintOf(n int) uint64 { return uint64(vulkanFootprint(n)) }

// FootprintOf implements AllocCounter: counted as asked (see mtlDev.Alloc).
func (m *mtlDev) FootprintOf(n int) uint64 { return uint64(n) }
