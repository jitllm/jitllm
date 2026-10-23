package vulkan

import (
	"fmt"
	"math"
	"sync"

	"github.com/jitllm/jitllm/backend/ffi"
)

// A rec is one line of submissions: its own command pool (a pool is not safe
// to use from two threads), command buffers, batch, descriptor pool, staging
// buffer, driver-facing structs and fence, submitting to one hardware queue.
// The Ctx holds one for the calls made on it directly; a Queue is another, so
// two sessions record and submit at once instead of one after another. A
// rec waits for its own fence, never the queue: vkQueueWaitIdle would wait
// for every other queue's work submitted to that queue too.
type rec struct {
	c       *Ctx
	hw      *hwQueue
	cmdPool CmdPool
	cmd     CmdBuffer
	// one is a second command buffer, for the one-shot operations that submit
	// and wait by themselves: copyBuf and Kernel.Launch. They reset and end
	// their command buffer, so on cmd a copy issued while a batch is recording
	// (a Session.Write of a device-local buffer) would silently drop every
	// dispatch encoded after it (VUID-vkCmdDispatch-commandBuffer-recording).
	one CmdBuffer
	// batch is the one Batch newBatch hands out, and scr the driver-facing
	// structs recording reuses; see ctxScratch.
	batch Batch
	scr   ctxScratch
	// bpool serves descriptor sets to the batch, one per encoded dispatch, and
	// is reset whole when the batch commits.
	bpool DescPool
	stage *Buf // lazily grown host-visible scratch for device-local transfers
	fence Fence
}

// hwQueue is one VkQueue. Submitting to a queue needs the host to serialise
// the calls (VUID-vkQueueSubmit-queue-00893), so queues sharing one take mu.
type hwQueue struct {
	mu sync.Mutex
	q  VkQueue
}

// maxHWQueues bounds how many queues of the compute family a context asks
// for. Queues past it share them, which orders their submissions but still
// lets each record and wait apart.
const maxHWQueues = 4

type fenceCI struct {
	sType uint32
	_     uint32
	pNext uintptr
	flags uint32
	_     uint32
}

const stFenceCI = 8

var (
	vkCreateFence   func(Device, up, up, up) Result
	vkDestroyFence  func(Device, Fence, up) ffi.None
	vkWaitForFences func(Device, uint32, up, uint32, uint64) Result
	vkResetFences   func(Device, uint32, up) Result
)

func loadFences(lib *ffi.Lib) {
	vkCreateFence = ffi.Fn4[Result, Device, up, up, up](lib, "vkCreateFence")
	vkDestroyFence = ffi.Fn3[ffi.None, Device, Fence, up](lib, "vkDestroyFence")
	vkWaitForFences = ffi.Fn5[Result, Device, uint32, up, uint32, uint64](lib, "vkWaitForFences")
	vkResetFences = ffi.Fn3[Result, Device, uint32, up](lib, "vkResetFences")
}

// initRec builds r's pool, command buffers and fence on hw.
func (c *Ctx) initRec(r *rec, hw *hwQueue) error {
	r.c, r.hw = c, hw
	cpc := cmdPoolCI{sType: stCmdPoolCI, flags: 0x2 /* RESET_COMMAND_BUFFER */, family: c.family}
	if err := check(vkCreateCommandPool(c.dev, up(&cpc), nil, up(&r.cmdPool)), "vkCreateCommandPool"); err != nil {
		return err
	}
	cba := cmdBufAI{sType: stCmdBufAlloc, pool: uint64(r.cmdPool), level: 0, count: 1}
	if err := check(vkAllocateCommandBuffers(c.dev, up(&cba), up(&r.cmd)), "vkAllocateCommandBuffers"); err != nil {
		return err
	}
	if err := check(vkAllocateCommandBuffers(c.dev, up(&cba), up(&r.one)), "vkAllocateCommandBuffers"); err != nil {
		return err
	}
	fci := fenceCI{sType: stFenceCI}
	return check(vkCreateFence(c.dev, up(&fci), nil, up(&r.fence)), "vkCreateFence")
}

// freeRec releases what initRec and the rec's use built. The device must be
// idle for it.
func (r *rec) freeRec() {
	c := r.c
	if c == nil || c.dev == 0 {
		return
	}
	if r.stage != nil {
		r.stage.Free()
		r.stage = nil
	}
	if r.bpool != 0 {
		vkDestroyDescriptorPool(c.dev, r.bpool, nil)
		r.bpool = 0
	}
	if r.fence != 0 {
		vkDestroyFence(c.dev, r.fence, nil)
		r.fence = 0
	}
	if r.cmdPool != 0 {
		vkDestroyCommandPool(c.dev, r.cmdPool, nil)
		r.cmdPool = 0
	}
}

// run submits si to r's queue and waits for r's fence alone.
func (r *rec) run(si *submitInfo) error {
	r.hw.mu.Lock()
	err := check(vkQueueSubmit(r.hw.q, 1, up(si), uint64(r.fence)), "vkQueueSubmit")
	r.hw.mu.Unlock()
	if err != nil {
		return err
	}
	// The fence's own field, not a copy: a local whose address goes to the
	// driver escapes, and a warm decode allocated one per submission.
	if err := check(vkWaitForFences(r.c.dev, 1, up(&r.fence), 1, math.MaxUint64), "vkWaitForFences"); err != nil {
		return err
	}
	return check(vkResetFences(r.c.dev, 1, up(&r.fence)), "vkResetFences")
}

// waitAll waits until every queue of the context is idle: what a call made
// outside any queue must be ordered after. Every queue's lock is held, since
// waiting on a queue is host access to it as submitting is.
func (c *Ctx) waitAll() error {
	for _, h := range c.hw {
		h.mu.Lock()
		err := check(vkQueueWaitIdle(h.q), "vkQueueWaitIdle")
		h.mu.Unlock()
		if err != nil {
			return err
		}
	}
	return nil
}

// Queue is a line of submissions of its own on this context: a session that
// records on a Queue runs beside sessions on other queues. One goroutine uses a
// Queue at a time.
type Queue struct{ r rec }

// NewQueue makes a queue, on the context's hardware queues in turn.
func (c *Ctx) NewQueue() (*Queue, error) {
	if len(c.hw) == 0 {
		return nil, fmt.Errorf("vulkan: no queue")
	}
	c.queues.Lock()
	hw := c.hw[c.queues.n%len(c.hw)]
	c.queues.n++
	c.queues.Unlock()
	q := &Queue{}
	if err := c.initRec(&q.r, hw); err != nil {
		q.r.freeRec()
		return nil, err
	}
	c.queues.Lock()
	if c.queues.live == nil {
		c.queues.live = map[*Queue]bool{}
	}
	c.queues.live[q] = true
	c.queues.Unlock()
	return q, nil
}

// Close releases the queue once its last submission has finished, which every
// one of its calls waits for before returning.
func (q *Queue) Close() {
	c := q.r.c
	c.queues.Lock()
	open := c.queues.live[q]
	delete(c.queues.live, q)
	c.queues.Unlock()
	if open {
		q.r.freeRec()
	}
}

// NewBatch begins recording on the queue; see Ctx.NewBatch.
func (q *Queue) NewBatch() *Batch { return q.r.newBatch() }

// WriteAt is Buf.WriteAt with any transfer on the queue.
func (q *Queue) WriteAt(b *Buf, off int, p []byte) error { return q.r.writeAt(b, off, p) }

// Read is Buf.Read with any transfer on the queue.
func (q *Queue) Read(b *Buf, p []byte) error { return q.r.read(b, p) }

// outsideQueues orders a transfer on the context's own rec after every queue's
// work, as a call outside any session must be; a queue's own transfer is
// ordered by its fence alone.
func (r *rec) outsideQueues() error {
	c := r.c
	if r != &c.rec {
		return nil
	}
	c.queues.Lock()
	n := len(c.queues.live)
	c.queues.Unlock()
	if n == 0 {
		return nil
	}
	return c.waitAll()
}
