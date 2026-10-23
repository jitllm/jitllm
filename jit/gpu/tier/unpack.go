package tier

import (
	"time"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// A page-in that uploads the GGUF bytes raw and unpacks them on the device.
//
// When a model is paged through a card too small for it, the host repack was
// most of the token, and the arena cannot help because the packed model does
// not fit host RAM either. So the raw bytes are uploaded as they lie and the
// card does the bit extraction (kernels.Unpack carries the per-tensor A/B). No
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

// canUnpack reports that this device would unpack a format rather than ask the
// host to pack it: kernels.UnpackSupported enumerates, and NoUnpack is the A/B
// arm.
func (g *devTier) canUnpack(q kernels.Quant) bool {
	return !g.NoUnpack && kernels.UnpackSupported(q)
}

// unpackKernel compiles and caches the unpack for one shape. A compile failure
// is cached as nil and recorded in UnpackWhy, so a silent fallback is visible.
// Callers hold g.mu.
func (g *devTier) unpackKernel(q kernels.Quant, nrows, k int) backend.Kernel {
	kk := unpackKey{q, nrows, k}
	if c, ok := g.unks[kk]; ok {
		return c
	}
	if g.unks == nil {
		g.unks = map[unpackKey]backend.Kernel{}
	}
	ker, err := kernels.Unpack(q, nrows, k)
	if err != nil {
		g.unks[kk], g.UnpackWhy = nil, err.Error()
		return nil
	}
	c, err := g.dev.Compile(ker)
	if err != nil {
		g.unks[kk], g.UnpackWhy = nil, err.Error()
		return nil
	}
	g.unks[kk] = c
	return c
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

// ensureRaw sizes the staging for a block, after that block has been admitted
// and reserving what it has still to spend. It is the only thing that allocates
// the staging buffer: sized earlier, or from inside a block's upload, the
// buffer takes the bytes a block needed and a model that fits loses a block
// (TestAModelThatFitsAdmitsTheSameBlocksEitherWay).
//
// A device that cannot afford the staging keeps whatever it has and the covered
// tensors fall back to the host packer, counted. Callers hold g.mu.
func (g *devTier) ensureRaw(ws []nn.Weight, reserve uint64) {
	var want uint64
	for _, x := range ws {
		q, ok := quantOf(x.T)
		if !ok || len(x.Data) == 0 || !g.canUnpack(q) {
			continue
		}
		if n := uint64(len(x.Data)); n > want {
			want = n
		}
	}
	if want == 0 {
		return
	}
	// The cap is the widest covered tensor of any block this device has been
	// offered, and it only ever grows: a model whose blocks differ in width has
	// to serve the widest of them, for slots()'s reason.
	if want > g.rawCap {
		g.rawCap = want
	}
	if g.rawBuf != nil && g.rawBytes >= g.rawCap {
		return // already held, and already inside the budget the block was priced against
	}
	if !g.room(g.rawCap + reserve) {
		return
	}
	g.rawStage(g.rawCap)
}

// unpackInto uploads a tensor's raw GGUF bytes and unpacks them on the device,
// filling r's three buffers. It reports whether it did; false means the caller
// runs the host packer, which is a fallback and not a failure.
//
// The bytes are not charged here. resident() checked room() for the packed size
// before calling and charges it after, so that the two paths spend the budget
// through one expression. Callers hold g.mu.
func (g *devTier) unpackInto(r *resident, q kernels.Quant, w []byte, nrows, k int) bool {
	if len(w) == 0 {
		return false
	}
	if !g.canUnpack(q) {
		// A format with no kernel (reason from kernels' own table) and the
		// NoUnpack measurement arm are reported apart.
		g.NoUnpackKernel++
		if g.UnpackWhy == "" {
			if g.NoUnpack {
				g.UnpackWhy = "the device unpack is off (Config.NoUnpack)"
			} else {
				g.UnpackWhy = q.String() + ": " + kernels.UnpackWhyNot(q)
			}
		}
		return false
	}
	// This never allocates the staging (see ensureRaw). The staging is capped
	// at the widest tensor of a block, so a wider tensor (the output
	// projection, uploaded once) is host-packed rather than growing a buffer
	// that exists to serve page-ins.
	if g.rawBuf == nil || uint64(len(w)) > g.rawBytes {
		g.NoUnpackRoom++
		return false
	}
	kern := g.unpackKernel(q, nrows, k)
	if kern == nil {
		g.NoUnpackKernel++
		return false
	}
	nq, nd, nsc, err := kernels.PackedWords(q, nrows, k)
	if err != nil {
		g.LastErr = "PackedWords " + q.String() + ": " + err.Error()
		return false
	}
	// The destinations, allocated and not written: the kernel assigns every word
	// of all three. An absent array still gets a word, because the matvec kernel
	// takes the parameter either way -- the same rule the upload path follows.
	bufs := [3]backend.Buf{}
	ok := true
	for i, n := range [3]int{nq, nd, nsc} {
		if n == 0 {
			n = 1
		}
		b, err := g.dev.Alloc(n * 4)
		if err != nil {
			g.LastErr = "unpack destination: " + err.Error()
			ok = false
			break
		}
		bufs[i] = b
	}
	if ok {
		tu := time.Now()
		err = g.rawBuf.Write(w)
		g.TUpload += time.Since(tu)
		if err != nil {
			g.LastErr = "raw upload: " + err.Error()
			ok = false
		}
	}
	if ok {
		threads := kernels.UnpackThreads(q, nrows, k)
		groups := (threads + unpackWidth - 1) / unpackWidth
		t0 := time.Now()
		err = kern.Launch(groups, unpackWidth, g.rawBuf, bufs[0], bufs[1], bufs[2])
		g.TUnpack += time.Since(t0)
		if err != nil {
			g.LastErr = "unpack launch: " + err.Error()
			ok = false
		}
	}
	if !ok {
		// A half-built tensor would run and answer wrongly: free it all and
		// let the host packer take it.
		for _, b := range bufs {
			if b != nil {
				b.Free()
			}
		}
		g.NoUnpackRoom++
		return false
	}
	r.qs, r.d, r.sc, r.ok = bufs[0], bufs[1], bufs[2], true
	g.Unpacks++
	g.UnpackBytes += uint64(len(w))
	return true
}
