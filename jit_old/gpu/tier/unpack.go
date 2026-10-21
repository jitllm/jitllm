package tier

import (
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// A page-in that uploads the GGUF bytes raw and unpacks them on the device.
//
// When a model is paged through a card too small for it, the host repack was
// most of the token, and the arena cannot help because the packed model does
// not fit host RAM either. So the raw bytes are uploaded as they lie and the
// card does the bit extraction (mainkernels.Unpack carries the per-tensor A/B). No
// host allocation happens on this path: Buf.Write takes the weight slice
// straight to the driver. The arena still wins on a device whose heap is host
// memory, and is the only answer for formats whose GGUF block is not a
// multiple of 4 bytes (see arena.go).
//
// The kernel reads pSrc and writes pQS/pD/pSC, four distinct buffers, so the
// IR's clamped surplus threads only recompute and re-store the same words
// (ir.Validate enforces it). The destinations are not poisoned: the thread
// mapping writes every word of all three arrays exactly once, which
// TestUnpackMatchesHostPacker checks with poisoned buffers.

// unpackWidth is the workgroup width the unpack launches at. It matches the
// [128,1,1] the kernel was built with; the two are one number in two places and
// UnpackThreads is what stops the launch re-deriving the mapping.
const unpackWidth = 128

// unpackKey is one compiled unpack. The shape is in the key because nrows is
// baked into every destination stride: a kernel built for one row count would
// write another's tensor to the wrong words, at the same speed.
type unpackKey struct {
	t        kernels.Quant
	nrows, k int
}

// rawStage makes the staging buffer at least n bytes and charges it.
//
// It is one buffer per device, sized by the widest covered tensor rather than a
// block, because resident() uploads one tensor at a time and reuses it. It is
// charged like any other non-pageable allocation (in used, not pageBytes), so
// slots() accounts for it without knowing it exists.
//
// The old buffer is freed before the new one is asked for, so growing never
// holds both at the peak. Callers hold g.mu.
func (g *devTier) rawStage(n uint64) bool {
	if n == 0 {
		return false
	}
	if g.rawBuf != nil && g.rawBytes >= n {
		return true
	}
	g.freeRaw()
	if !g.room(n) {
		g.LastErr = "unpack staging does not fit the budget; the host packer runs instead"
		return false
	}
	b, err := g.dev.Alloc(int(n))
	if err != nil {
		g.LastErr = "unpack staging: " + err.Error()
		return false
	}
	g.rawBuf, g.rawBytes = b, n
	g.charge(n)
	if n > g.StageVRAM {
		g.StageVRAM = n
	}
	return true
}

// freeRaw releases the staging buffer and refunds it, so a budget shrink can
// reclaim it. Callers hold g.mu.
func (g *devTier) freeRaw() uint64 {
	if g.rawBuf == nil {
		return 0
	}
	n := g.rawBytes
	g.rawBuf.Free()
	g.rawBuf, g.rawBytes = nil, 0
	g.refund(n)
	if g.StageVRAM == n {
		g.StageVRAM = 0
	}
	return n
}
