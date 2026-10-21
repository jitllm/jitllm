package vulkan

import (
	"fmt"
	"math"
	"runtime"
	"slices"
	"unsafe"
)

// Buf is a device buffer. host is non-nil only when the memory the driver gave
// us is mappable; otherwise Write and Read stage through Ctx.stage.
type Buf struct {
	c    *Ctx
	buf  Buffer
	mem  Memory
	n    int
	host []byte
	// imported says host is the caller's memory, wrapped by Import rather than
	// allocated here. Free is the same either way (vkFreeMemory releases only
	// the wrapper); it lets a caller tell a descriptor update from a transfer.
	imported bool
}

// Imported reports that this buffer ALIASES host memory the caller owns, rather
// than holding a copy of it.
func (b *Buf) Imported() bool { return b != nil && b.imported }

// Alloc creates an n-byte device-local storage buffer.
func (c *Ctx) Alloc(n int) (*Buf, error) {
	return c.alloc(n, usageStorageBuffer|usageTransferSrc|usageTransferDst, memDeviceLocal)
}

// alloc takes the memory properties to PREFER. A device-local request falls
// back to host-visible when the device has no device-local type that fits,
// which is what happens on llvmpipe.
func (c *Ctx) alloc(n int, usage, prefer uint32) (*Buf, error) {
	b := &Buf{c: c, n: n}
	bci := bufferCI{sType: stBufferCI, size: uint64(n), usage: usage}
	if err := check(vkCreateBuffer(c.dev, up(&bci), nil, up(&b.buf)), "vkCreateBuffer"); err != nil {
		return nil, err
	}
	var req memReq
	vkGetBufferMemoryRequirements(c.dev, b.buf, up(&req))

	idx, mappable, err := c.memType(req.typeBits, prefer)
	if err != nil {
		vkDestroyBuffer(c.dev, b.buf, nil)
		return nil, err
	}
	ma := memAlloc{sType: stMemAllocInfo, size: req.size, typeIdx: idx}
	if err := check(vkAllocateMemory(c.dev, up(&ma), nil, up(&b.mem)), "vkAllocateMemory"); err != nil {
		vkDestroyBuffer(c.dev, b.buf, nil)
		return nil, err
	}
	if err := check(vkBindBufferMemory(c.dev, b.buf, b.mem, 0), "vkBindBufferMemory"); err != nil {
		b.Free()
		return nil, err
	}
	if mappable {
		var p unsafe.Pointer
		if err := check(vkMapMemory(c.dev, b.mem, 0, uint64(n), 0, up(&p)), "vkMapMemory"); err != nil {
			b.Free()
			return nil, err
		}
		b.host = unsafe.Slice((*byte)(p), n)
	}
	return b, nil
}

// memType picks a memory type from the mask, preferring `prefer` and reporting
// whether the winner is host-mappable.
//
// The ORDER is the whole point. A type that is both device-local and
// host-visible is the best of both, and is what integrated GPUs, llvmpipe and
// resizable-BAR cards offer. Device-local alone is next and costs a staging
// copy. Host-visible alone is the last resort: on a discrete card it is many
// times slower.
func (c *Ctx) memType(bits, prefer uint32) (idx uint32, mappable bool, err error) {
	const hostRW = memHostVisible | memHostCoherent
	for _, want := range []struct {
		flags    uint32
		mappable bool
	}{
		{prefer | hostRW, true},
		{prefer, false},
		{hostRW, true},
	} {
		for i := uint32(0); i < c.mem.typeCount; i++ {
			if bits&(1<<i) != 0 && c.mem.types[i].flags&want.flags == want.flags {
				return i, want.mappable, nil
			}
		}
	}
	return 0, false, fmt.Errorf("vulkan: no usable memory type in mask %#x", bits)
}

// Write copies host bytes in, staging when the device memory is not mappable.
func (b *Buf) Write(p []byte) error { return b.WriteAt(0, p) }

// WriteAt copies host bytes in at a byte offset, so a caller can assemble one
// device buffer out of several host runs without concatenating them first. On
// the mapped path that is a slice of the mapping; on the staged path it is the
// copy region's dstOffset.
func (b *Buf) WriteAt(off int, p []byte) error {
	if off < 0 || off+len(p) > b.n {
		return fmt.Errorf("vulkan: writing %d bytes at offset %d of a %d-byte buffer", len(p), off, b.n)
	}
	return b.c.rec.writeAt(b, off, p)
}

// writeAt is Buf.WriteAt with a staged transfer on r.
func (r *rec) writeAt(b *Buf, off int, p []byte) error {
	if off < 0 || off+len(p) > b.n {
		return fmt.Errorf("vulkan: writing %d bytes at offset %d of a %d-byte buffer", len(p), off, b.n)
	}
	if b.host != nil {
		copy(b.host[off:], p)
		return nil
	}
	if err := r.outsideQueues(); err != nil {
		return err
	}
	st, err := r.staging(len(p))
	if err != nil {
		return err
	}
	copy(st.host, p)
	return r.copyBuf(st.buf, b.buf, 0, uint64(off), len(p))
}

// Len is the buffer's size in bytes.
func (b *Buf) Len() int { return b.n }

// Copy copies n bytes from src at srcOff to dst at dstOff with
// vkCmdCopyBuffer, on the one-shot command buffer like every transfer here,
// and waits for the queue. Both buffers must be this context's; ranges are
// the caller's to have checked (vkCmdCopyBuffer leaves an overlap within one
// buffer undefined).
func (c *Ctx) Copy(dst *Buf, dstOff int, src *Buf, srcOff, n int) error {
	if dst.c != c || src.c != c {
		return fmt.Errorf("vulkan: a copy between buffers of another device")
	}
	// Ordered after every lane's work, not only this context's own.
	if err := c.waitAll(); err != nil {
		return err
	}
	return c.copyBuf(src.buf, dst.buf, uint64(srcOff), uint64(dstOff), n)
}

// Read copies the buffer back, staging if it must.
func (b *Buf) Read(p []byte) error {
	return b.c.rec.read(b, p)
}

// read is Buf.Read with a staged transfer on r.
func (r *rec) read(b *Buf, p []byte) error {
	if len(p) > b.n {
		return fmt.Errorf("vulkan: reading %d bytes from a %d-byte buffer", len(p), b.n)
	}
	if b.host != nil {
		copy(p, b.host)
		return nil
	}
	if err := r.outsideQueues(); err != nil {
		return err
	}
	st, err := r.staging(len(p))
	if err != nil {
		return err
	}
	if err := r.copyBuf(b.buf, st.buf, 0, 0, len(p)); err != nil {
		return err
	}
	copy(p, st.host)
	return nil
}

// staging returns a host-visible scratch buffer of at least n bytes, growing it
// as needed. One per rec: its transfers wait before the next one starts.
func (r *rec) staging(n int) (*Buf, error) {
	if r.stage != nil && r.stage.n >= n {
		return r.stage, nil
	}
	if r.stage != nil {
		r.stage.Free()
		r.stage = nil
	}
	b, err := r.c.alloc(n, usageTransferSrc|usageTransferDst, memHostVisible|memHostCoherent)
	if err != nil {
		return nil, err
	}
	if b.host == nil {
		b.Free()
		return nil, fmt.Errorf("vulkan: staging buffer came back unmappable")
	}
	r.stage = b
	return b, nil
}

// copyBuf stages one buffer copy. It runs on c.one, NOT on c.cmd: a Session
// may be recording a batch into c.cmd right now, and a one-shot that resets it
// would discard everything encoded so far and silently drop everything encoded
// after. See Ctx.one.
func (r *rec) copyBuf(src, dst Buffer, srcOff, dstOff uint64, n int) error {
	if err := check(vkResetCommandBuffer(r.one, 0), "vkResetCommandBuffer"); err != nil {
		return err
	}
	x := &r.scr
	x.oneBI = cmdBeginI{sType: stCmdBufBegin, flags: cmdBufOneTime}
	if err := check(vkBeginCommandBuffer(r.one, up(&x.oneBI)), "vkBeginCommandBuffer"); err != nil {
		return err
	}
	x.region = bufCopy{srcOff: srcOff, dstOff: dstOff, size: uint64(n)}
	vkCmdCopyBuffer(r.one, src, dst, 1, up(&x.region))
	if err := check(vkEndCommandBuffer(r.one), "vkEndCommandBuffer"); err != nil {
		return err
	}
	return r.submitOne()
}

// submitOne submits r.one and waits. Separate from submit for the reason one
// is separate: the two must not name the same buffer.
func (r *rec) submitOne() error {
	x := &r.scr
	x.oneCmd = r.one
	x.oneSI = submitInfo{sType: stSubmitInfo, nCmd: 1, pCmd: uintptr(up(&x.oneCmd))}
	return r.run(&x.oneSI)
}

func (r *rec) submit() error {
	x := &r.scr
	x.cmd = r.cmd
	x.si = submitInfo{sType: stSubmitInfo, nCmd: 1, pCmd: uintptr(up(&x.cmd))}
	return r.run(&x.si)
}

func (b *Buf) Free() {
	if b.mem != 0 {
		vkFreeMemory(b.c.dev, b.mem, nil)
		b.mem = 0
	}
	if b.buf != 0 {
		vkDestroyBuffer(b.c.dev, b.buf, nil)
		b.buf = 0
	}
	b.host = nil
}

// Kernel is a compiled compute pipeline plus its descriptor plumbing.
type Kernel struct {
	c      *Ctx
	mod    ShaderMod
	layout DescLayout
	plyt   PipeLayout
	pipe   Pipeline
	pool   DescPool
	set    DescSet
	nbuf   int
	name   []byte
}

// kernName is the entry point this kernel was compiled for, without its
// trailing NUL, so an arity error says which kernel.
func kernName(k *Kernel) string {
	if n := len(k.name); n > 0 && k.name[n-1] == 0 {
		return string(k.name[:n-1])
	}
	return string(k.name)
}

// Compile builds a pipeline from a SPIR-V module with nbuf storage buffers
// bound at set 0, bindings 0..nbuf-1 -- the layout spirv.Emit decorates.
//
// lanes is the subgroup width the kernel's ARITHMETIC needs, or 0 when it needs
// none. It is not a hint:
//
//   - a width this device cannot promise is an ERROR here, before a pipeline
//     exists, so the caller chooses another kernel instead of launching a
//     wrong one. See Subgroups.Guarantees for what counts as a promise.
//   - a width it can promise is PINNED into the pipeline with
//     VkPipelineShaderStageRequiredSubgroupSizeCreateInfo, and
//     REQUIRE_FULL_SUBGROUPS is set so a workgroup can never hold a partial
//     subgroup. The driver must then honour it or refuse to create the
//     pipeline, and a refusal is a thing a caller can act on.
//
// REQUIRE_FULL_SUBGROUPS is redundant (ir.Validate requires a shuffling
// kernel's workgroup to be 32n x 1 x 1) and is set anyway, because that
// invariant lives in another package.
func (c *Ctx) Compile(spv []byte, entry string, nbuf, lanes int) (*Kernel, error) {
	if len(spv)%4 != 0 || len(spv) < 20 {
		return nil, fmt.Errorf("vulkan: SPIR-V must be a whole number of words, got %d bytes", len(spv))
	}
	if ok, why := c.sg.Guarantees(lanes); !ok {
		return nil, fmt.Errorf("vulkan: %s cannot guarantee a %d-lane subgroup for %q: %s",
			c.name, lanes, entry, why)
	}
	k := &Kernel{c: c, nbuf: nbuf}
	sm := shaderCI{sType: stShaderModuleCI, size: uint64(len(spv)), pCode: uintptr(up(&spv[0]))}
	if err := check(vkCreateShaderModule(c.dev, up(&sm), nil, up(&k.mod)), "vkCreateShaderModule"); err != nil {
		return nil, err
	}

	binds := make([]descBinding, nbuf)
	for i := range binds {
		binds[i] = descBinding{binding: uint32(i), kind: descStorageBuffer, count: 1, stages: stageCompute}
	}
	dl := descLayoutCI{sType: stDescLayoutCI, nBind: uint32(nbuf), pBind: uintptr(up(&binds[0]))}
	if err := check(vkCreateDescriptorSetLayout(c.dev, up(&dl), nil, up(&k.layout)), "vkCreateDescriptorSetLayout"); err != nil {
		k.Close()
		return nil, err
	}
	lay := k.layout
	// One push constant, the CTAID base spirv.Emit declares; see dispatch.
	pr := pushRange{stages: stageCompute, size: groupBaseBytes}
	pl := pipeLayoutCI{sType: stPipeLayoutCI, nSets: 1, pSets: uintptr(up(&lay)),
		nPush: 1, pPush: uintptr(up(&pr))}
	err := check(vkCreatePipelineLayout(c.dev, up(&pl), nil, up(&k.plyt)), "vkCreatePipelineLayout")
	// lay and pr are reached through uintptr fields, which keep nothing alive.
	runtime.KeepAlive(lay)
	runtime.KeepAlive(pr)
	if err != nil {
		k.Close()
		return nil, err
	}

	k.name = append([]byte(entry), 0)
	cp := computeCI{sType: stComputePipeCI, layout: uint64(k.plyt), baseIdx: -1,
		stage: stageCI{sType: stPipeShaderCI, stage: stageCompute,
			module: uint64(k.mod), pName: uintptr(up(&k.name[0]))}}
	req := reqSizeCI{sType: stReqSizeCI, size: uint32(lanes)}
	if lanes > 0 && c.sg.CanPin() {
		cp.stage.pNext = uintptr(up(&req))
		cp.stage.flags |= pipeStageRequireFullSubgroups
	}
	err = check(vkCreateComputePipelines(c.dev, 0, 1, up(&cp), nil, up(&k.pipe)), "vkCreateComputePipelines")
	// req is reachable only through a uintptr pNext, which is not a reference.
	runtime.KeepAlive(req)
	if err != nil {
		k.Close()
		return nil, err
	}

	ps := poolSize{kind: descStorageBuffer, count: uint32(nbuf)}
	dp := descPoolCI{sType: stDescPoolCI, maxSets: 1, nSizes: 1, pSizes: uintptr(up(&ps))}
	if err := check(vkCreateDescriptorPool(c.dev, up(&dp), nil, up(&k.pool)), "vkCreateDescriptorPool"); err != nil {
		k.Close()
		return nil, err
	}
	da := descSetAI{sType: stDescSetAlloc, pool: uint64(k.pool), n: 1, pSets: uintptr(up(&lay))}
	if err := check(vkAllocateDescriptorSets(c.dev, up(&da), up(&k.set)), "vkAllocateDescriptorSets"); err != nil {
		k.Close()
		return nil, err
	}
	c.kerns.Lock()
	if c.kerns.live == nil {
		c.kerns.live = map[*Kernel]struct{}{}
	}
	c.kerns.live[k] = struct{}{}
	c.kerns.Unlock()
	return k, nil
}

// Bind points the descriptor set at these buffers, in binding order.
func (k *Kernel) Bind(bufs ...*Buf) error {
	if len(bufs) != k.nbuf {
		return fmt.Errorf("vulkan: kernel %q wants %d buffers, got %d", kernName(k), k.nbuf, len(bufs))
	}
	infos := make([]descBufInfo, len(bufs))
	writes := make([]writeDesc, len(bufs))
	for i, b := range bufs {
		infos[i] = descBufInfo{buffer: uint64(b.buf), rng: ^uint64(0) /* VK_WHOLE_SIZE */}
		writes[i] = writeDesc{sType: stWriteDescSet, dstSet: uint64(k.set), dstBind: uint32(i),
			count: 1, kind: descStorageBuffer, pBuffer: uintptr(up(&infos[i]))}
	}
	vkUpdateDescriptorSets(k.c.dev, uint32(len(writes)), up(&writes[0]), 0, nil)
	// infos is reached only through writeDesc.pBuffer, a uintptr, which keeps
	// nothing alive; the collector could take it before the driver reads it.
	runtime.KeepAlive(infos)
	runtime.KeepAlive(writes)
	return nil
}

// groupBaseBytes is the push-constant block every kernel's pipeline layout
// carries: one uint32, the CTAID of the dispatch's first workgroup.
const groupBaseBytes = 4

// launchable refuses a launch whose workgroup indices the IR cannot hold. The
// IR's CTAID is a u32 on every target, so past 2^32 workgroups it would wrap
// and two workgroups would compute the same item while another went unwritten.
// It is checked before anything is recorded, so a refusal leaves no partial
// launch behind.
func (c *Ctx) launchable(k *Kernel, groups int) error {
	if groups < 0 || uint64(groups) > math.MaxUint32 {
		return fmt.Errorf("vulkan: kernel %q launched with %d workgroups; the IR's workgroup index "+
			"is 32-bit, so a launch holds at most %d", kernName(k), groups, uint64(math.MaxUint32))
	}
	if c.maxGroups == 0 {
		return fmt.Errorf("vulkan: %s reports maxComputeWorkGroupCount[0] = 0", c.name)
	}
	return nil
}

// dispatch records a launch of groups workgroups of k into cmd, which already
// has k's pipeline and descriptor set bound; launchable has passed it.
//
// A device bounds the workgroups of ONE vkCmdDispatch (maxComputeWorkGroupCount,
// 65535 along x on an Intel integrated GPU), and a launch past it is invalid
// usage (VUID-vkCmdDispatch-groupCountX-00386). That driver happens to run
// one correctly; nothing obliges the next driver to, and no error comes
// back either way. So a launch is cut into dispatches of at most MaxGroups workgroups,
// and each pushes the index of its first workgroup, which spirv.Emit adds to
// WorkgroupId.x to form the IR's CTAID. The pieces cover [0, groups) exactly,
// so every kernel sees the CTAIDs an unsplit launch gives it: no surplus
// workgroup exists, and nothing in a kernel -- its clamp, its shared memory,
// its barriers, an in-group reduction keyed on the workgroup -- can tell.
//
// No barrier separates the pieces. They are workgroups of one launch, which a
// single dispatch would not have ordered either, and the IR has no way for one
// workgroup to wait on another. The caller's barrier after the launch orders
// all of it against what follows. b is the context scratch word the group
// base is pushed from, one per command buffer.
func (c *Ctx) dispatch(cmd CmdBuffer, b *uint32, k *Kernel, groups int) {
	step := uint64(c.maxGroups)
	for base := uint64(0); base < uint64(groups); base += step {
		*b = uint32(base)
		vkCmdPushConstants(cmd, k.plyt, stageCompute, 0, groupBaseBytes, up(b))
		vkCmdDispatch(cmd, uint32(min(step, uint64(groups)-base)), 1, 1)
		c.dispatches.Add(1)
	}
}

// Dispatches is how many vkCmdDispatch this context has recorded. A launch
// within MaxGroups is one; a split launch is one per piece. It is the check
// that a split was taken at all: a launch that fits and one that was cut into
// pieces leave the same buffer behind.
func (c *Ctx) Dispatches() uint64 { return c.dispatches.Load() }

// Launch dispatches groups workgroups and blocks until the queue drains.
func (k *Kernel) Launch(groups int) error {
	c := k.c
	if err := c.launchable(k, groups); err != nil {
		return err
	}
	// On c.one, not c.cmd: this is a one-shot that resets, records, ends and
	// waits, and a Session may be recording a batch into c.cmd. See Ctx.one.
	if err := check(vkResetCommandBuffer(c.one, 0), "vkResetCommandBuffer"); err != nil {
		return err
	}
	bi := cmdBeginI{sType: stCmdBufBegin, flags: cmdBufOneTime}
	if err := check(vkBeginCommandBuffer(c.one, up(&bi)), "vkBeginCommandBuffer"); err != nil {
		return err
	}
	vkCmdBindPipeline(c.one, pipeBindCompute, k.pipe)
	set := k.set
	vkCmdBindDescriptorSets(c.one, pipeBindCompute, k.plyt, 0, 1, up(&set), 0, nil)
	c.dispatch(c.one, &c.scr.oneBase, k, groups)
	if err := check(vkEndCommandBuffer(c.one), "vkEndCommandBuffer"); err != nil {
		return err
	}
	return c.submitOne()
}

func (k *Kernel) Close() {
	k.c.kerns.Lock()
	delete(k.c.kerns.live, k)
	k.c.kerns.Unlock()
	d := k.c.dev
	if k.pool != 0 {
		vkDestroyDescriptorPool(d, k.pool, nil)
		k.pool = 0
	}
	if k.pipe != 0 {
		vkDestroyPipeline(d, k.pipe, nil)
		k.pipe = 0
	}
	if k.plyt != 0 {
		vkDestroyPipelineLayout(d, k.plyt, nil)
		k.plyt = 0
	}
	if k.layout != 0 {
		vkDestroyDescriptorSetLayout(d, k.layout, nil)
		k.layout = 0
	}
	if k.mod != 0 {
		vkDestroyShaderModule(d, k.mod, nil)
		k.mod = 0
	}
}

// Batch records many dispatches into one command buffer, so a decode token
// pays one queue drain rather than one per dispatch (Kernel.Launch drains
// every time).
//
// Two things Metal's version does not need:
//
//   - A barrier between dispatches. Vulkan does not order compute dispatches
//     in a command buffer the way a CUDA stream orders launches. Each encode
//     ends with a conservative whole-pipeline barrier; a precise one would
//     need to know which buffers alias, and being wrong is a race.
//
//   - A descriptor set per dispatch. Kernel.Bind updates the one set a kernel
//     owns, so a kernel encoded twice in a batch with different buffers would
//     run both with the last binding. Sets come from a per-Ctx pool that the
//     commit resets.
type Batch struct {
	r    *rec
	n    int
	open bool
	err  error
}

// NewBatch begins recording. Commit must be called, and Ctx holds one command
// buffer, so batches do not nest -- which is why the Batch is the context's
// own, reset here, rather than a new one per session.
func (c *Ctx) NewBatch() *Batch { return c.rec.newBatch() }

func (r *rec) newBatch() *Batch {
	r.batch = Batch{r: r}
	r.batch.begin()
	return &r.batch
}

func (b *Batch) begin() {
	if b.err != nil {
		return
	}
	r := b.r
	if b.err = check(vkResetCommandBuffer(r.cmd, 0), "vkResetCommandBuffer"); b.err != nil {
		return
	}
	r.scr.bi = cmdBeginI{sType: stCmdBufBegin, flags: cmdBufOneTime}
	b.err = check(vkBeginCommandBuffer(r.cmd, up(&r.scr.bi)), "vkBeginCommandBuffer")
	b.open = b.err == nil
	b.n = 0
}

// Encode records one dispatch of k over bufs.
func (b *Batch) Encode(k *Kernel, groups int, bufs ...*Buf) error {
	if b.err != nil {
		return b.err
	}
	if len(bufs) != k.nbuf {
		return fmt.Errorf("vulkan: kernel %q wants %d buffers, got %d", kernName(k), k.nbuf, len(bufs))
	}
	if err := b.r.c.launchable(k, groups); err != nil {
		return err
	}
	if b.n >= batchSets {
		// Out of descriptor sets: submit what is recorded and start again. The
		// result is correct either way; only the number of drains changes.
		if err := b.Commit(); err != nil {
			return err
		}
		b.begin()
		if b.err != nil {
			return b.err
		}
	}
	r := b.r
	c := r.c
	set, err := r.allocSet(k.layout)
	if err != nil {
		b.err = err
		return err
	}
	x := &r.scr
	x.infos = slices.Grow(x.infos[:0], len(bufs))[:len(bufs)]
	x.writes = slices.Grow(x.writes[:0], len(bufs))[:len(bufs)]
	infos, writes := x.infos, x.writes
	for i, buf := range bufs {
		infos[i] = descBufInfo{buffer: uint64(buf.buf), rng: ^uint64(0)}
		writes[i] = writeDesc{sType: stWriteDescSet, dstSet: uint64(set), dstBind: uint32(i),
			count: 1, kind: descStorageBuffer, pBuffer: uintptr(up(&infos[i]))}
	}
	vkUpdateDescriptorSets(c.dev, uint32(len(writes)), up(&writes[0]), 0, nil)
	// See Kernel.Bind: pBuffer is a uintptr, so `infos` has no live reference
	// once the loop ends and the collector may take it mid-call.
	runtime.KeepAlive(infos)
	runtime.KeepAlive(writes)

	vkCmdBindPipeline(r.cmd, pipeBindCompute, k.pipe)
	x.bindSet = set
	vkCmdBindDescriptorSets(r.cmd, pipeBindCompute, k.plyt, 0, 1, up(&x.bindSet), 0, nil)
	c.dispatch(r.cmd, &r.scr.base, k, groups)
	// Both masks cover read and write: write -> read alone leaves write ->
	// write and read -> write unordered. Most kernels form a chain, but two
	// shapes are not:
	//
	//	read -> write   emitLinear convolves OUT of bs.dMixed (convRows,
	//	                convShift) and then writes the activation back INTO it
	//	                (ssmSiLU). The write may overtake the reads.
	//	write -> write  partial rotary copies the whole head into the
	//	                destination and then rotates the first NRot dimensions
	//	                of it in place -- copyQ then ropeQ into bs.qr, copyK
	//	                then ropeK into the KV cache -- because RoPERows writes
	//	                only what it rotates. The copy may land after the
	//	                rotation and put the unrotated values back.
	x.mb = memBarrier{sType: stMemBarrier,
		src: accessShaderRead | accessShaderWrite,
		dst: accessShaderRead | accessShaderWrite}
	vkCmdPipelineBarrier(r.cmd, stageComputeShader, stageComputeShader, 0, 1, up(&x.mb), 0, nil, 0, nil)
	b.n++
	return nil
}

// Commit ends the command buffer, submits it and waits.
func (b *Batch) Commit() error {
	if !b.open {
		return b.err
	}
	b.open = false
	r := b.r
	if err := check(vkEndCommandBuffer(r.cmd), "vkEndCommandBuffer"); err != nil {
		b.err = err
		return err
	}
	if b.n == 0 {
		return nil // nothing recorded; submitting an empty buffer is a wasted drain
	}
	if err := r.submit(); err != nil {
		b.err = err
		return err
	}
	// The sets are free the moment the submission is done, which submit waits
	// for.
	if r.bpool != 0 {
		check(vkResetDescriptorPool(r.c.dev, r.bpool, 0), "vkResetDescriptorPool")
	}
	return nil
}

// allocSet hands out one descriptor set from the batch pool, creating it on
// first use.
func (r *rec) allocSet(layout DescLayout) (DescSet, error) {
	c := r.c
	if r.bpool == 0 {
		ps := poolSize{kind: descStorageBuffer, count: batchSets * batchBufs}
		dp := descPoolCI{sType: stDescPoolCI, maxSets: batchSets, nSizes: 1, pSizes: uintptr(up(&ps))}
		if err := check(vkCreateDescriptorPool(c.dev, up(&dp), nil, up(&r.bpool)), "vkCreateDescriptorPool"); err != nil {
			return 0, err
		}
	}
	x := &r.scr
	x.lay = layout
	x.da = descSetAI{sType: stDescSetAlloc, pool: uint64(r.bpool), n: 1, pSets: uintptr(up(&x.lay))}
	if err := check(vkAllocateDescriptorSets(c.dev, up(&x.da), up(&x.set)), "vkAllocateDescriptorSets"); err != nil {
		return 0, err
	}
	return x.set, nil
}

// ImportAlign is the alignment a host pointer AND its length must have for
// Import to take them, or 0 when this device cannot import host memory at all.
//
// The capability and the alignment are one number, so a caller cannot ask
// one and forget the other.
func (c *Ctx) ImportAlign() int { return int(c.hostAlign) }

// Import wraps n bytes of HOST memory at p as a device buffer that ALIASES
// them: no transfer, no second copy, and no device allocation beyond the
// wrapper.
//
// The caller owns the memory and must outlive the buffer: vkFreeMemory
// releases the import, not the pages. kernels.Arena is the intended owner (its
// regions are page aligned and padded for this call), so a buffer imported from
// an arena entry must be freed before that entry is released.
//
// This does not check UnifiedMemory(); tier does, since on a discrete card an
// imported weight is re-read across PCIe on every matvec.
func (c *Ctx) Import(p unsafe.Pointer, n int) (*Buf, error) {
	if c.hostAlign == 0 || c.getHostPtrProps == nil {
		return nil, fmt.Errorf("vulkan: %s is not available on %s, so host memory cannot be imported",
			extExternalMemoryHost, c.name)
	}
	if p == nil || n <= 0 {
		return nil, fmt.Errorf("vulkan: cannot import %d bytes at %p", n, p)
	}
	// Both the pointer and the length must be aligned, as the spec requires;
	// a rounded length would import pages the caller does not own.
	if uintptr(p)%uintptr(c.hostAlign) != 0 || uint64(n)%c.hostAlign != 0 {
		return nil, fmt.Errorf("vulkan: %s wants %d-byte alignment; got pointer %p and length %d",
			extExternalMemoryHost, c.hostAlign, p, n)
	}

	b := &Buf{c: c, n: n}
	emb := extMemBufCI{sType: stExtMemBufferCI, handles: handleTypeHostAlloc}
	bci := bufferCI{sType: stBufferCI, pNext: uintptr(up(&emb)), size: uint64(n),
		usage: usageStorageBuffer | usageTransferSrc | usageTransferDst}
	if err := check(vkCreateBuffer(c.dev, up(&bci), nil, up(&b.buf)), "vkCreateBuffer"); err != nil {
		return nil, err
	}
	runtime.KeepAlive(emb)
	var req memReq
	vkGetBufferMemoryRequirements(c.dev, b.buf, up(&req))

	// The memory type has to satisfy the BUFFER and the POINTER at once: the
	// driver decides which types can wrap this particular allocation, and
	// asking is not optional (see memHostPtrProps).
	props := memHostPtrProps{sType: stMemHostPtrProps}
	if err := check(c.getHostPtrProps(c.dev, handleTypeHostAlloc, p, up(&props)),
		"vkGetMemoryHostPointerPropertiesEXT"); err != nil {
		vkDestroyBuffer(c.dev, b.buf, nil)
		return nil, err
	}
	idx, _, err := c.memType(req.typeBits&props.typeBits, memDeviceLocal)
	if err != nil {
		vkDestroyBuffer(c.dev, b.buf, nil)
		return nil, fmt.Errorf("%w (buffer types %#x, host-pointer types %#x)",
			err, req.typeBits, props.typeBits)
	}
	imp := importHostPtrInfo{sType: stImportMemHostPtr, handleType: handleTypeHostAlloc, p: uintptr(p)}
	ma := memAlloc{sType: stMemAllocInfo, pNext: uintptr(up(&imp)), size: uint64(n), typeIdx: idx}
	if err := check(vkAllocateMemory(c.dev, up(&ma), nil, up(&b.mem)), "vkAllocateMemory (imported host pointer)"); err != nil {
		vkDestroyBuffer(c.dev, b.buf, nil)
		return nil, err
	}
	runtime.KeepAlive(imp)
	if err := check(vkBindBufferMemory(c.dev, b.buf, b.mem, 0), "vkBindBufferMemory"); err != nil {
		b.Free()
		return nil, err
	}
	// No vkMapMemory: the host pointer is already the mapping, so Write and
	// Read are memmoves over the caller's own bytes.
	b.host = unsafe.Slice((*byte)(p), n)
	b.imported = true
	return b, nil
}

// ctxScratch is where a context keeps the structs a recording hands the driver
// by address. Every vk* binding is a func value, so escape analysis cannot see
// that the driver keeps nothing, and a local passed by address was a heap
// object per dispatch. A context records into one command buffer at a time
// and runs one one-shot at a time on another (Ctx.one), so one set of each
// serves; the one-shot's are separate because a copy may be issued while a
// batch is recording.
type ctxScratch struct {
	bi, oneBI     cmdBeginI
	si, oneSI     submitInfo
	cmd, oneCmd   CmdBuffer
	region        bufCopy
	lay           DescLayout
	da            descSetAI
	set, bindSet  DescSet
	mb            memBarrier
	base, oneBase uint32 // a dispatch's pushed group base (Ctx.dispatch)
	infos         []descBufInfo
	writes        []writeDesc
}
