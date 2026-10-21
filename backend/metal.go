package backend

import (
	"fmt"
	"sync"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/jit/ir"
	"github.com/jitllm/jitllm/backend/metal"
	"github.com/jitllm/jitllm/jit/msl"
)

type mtlDev struct {
	allocCount
	once
	c *metal.Ctx
	// free is Session's free list: a session handed to the caller's f
	// escapes, so a fresh one per call was a heap object per token. Nested
	// sessions take one each.
	mu   sync.Mutex
	free []*mtlSession
	// ops is CopyList's scratch, under mu: kept, so a list allocates
	// nothing.
	ops []metal.CopyOp
}

// CopyList is Copy for every range of cs in one blit command buffer
// (CopyLister).
func (m *mtlDev) CopyList(cs []CopyRange) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	ops := m.ops[:0]
	for _, r := range cs {
		db, ok := r.Dst.(*mtlBuf)
		sb, ok2 := r.Src.(*mtlBuf)
		if !ok || !ok2 || db == nil || sb == nil {
			return fmt.Errorf("backend: msl: a copy between buffers this backend did not allocate")
		}
		if err := copyRange(len(db.b.Bytes()), r.DstOff, len(sb.b.Bytes()), r.SrcOff, r.N, db.b == sb.b); err != nil {
			return err
		}
		ops = append(ops, metal.CopyOp{Dst: db.b, DstOff: r.DstOff, Src: sb.b, SrcOff: r.SrcOff, N: r.N})
	}
	err := m.c.CopyList(ops)
	clear(ops)
	m.ops = ops[:0]
	return err
}

var _ BufferLimited = (*mtlDev)(nil)

// MaxBuffer is the device's maxBufferLength (BufferLimited).
func (d *mtlDev) MaxBuffer() uint64 { return d.c.MaxBuffer() }

// OpenMetalWith opens the system default Metal device with this device's open
// choices. It is exported, like OpenCUDA and OpenVulkan, so a caller that names
// its devices need not open every backend through backend.Open. It takes no
// selector: every Apple Silicon machine has exactly one GPU.
func OpenMetalWith(o Opts) (Device, error) { return openMetal(o) }

// openMetal needs no build tag: metal.Open returns an error on every
// non-Apple host, which is the same shape as a Linux host with no CUDA.
func openMetal(o Opts) (Device, error) {
	if _, err := admit("msl", 0, o.Enumerating); err != nil {
		return nil, err
	}
	c, err := metal.OpenWith(o.Metal)
	if err != nil {
		return nil, err
	}
	c.Unretained(mtlUnretained)
	if mtlResidency {
		if err := c.Residency(); err != nil {
			c.Close()
			return nil, err
		}
	}
	ownedDevices.Add(1)
	return &mtlDev{c: c}, nil
}

func (m *mtlDev) Name() string { return m.c.Name() }
func (m *mtlDev) API() string  { return "msl" }

// Slots is unknown on Metal: MTLDevice exposes maxThreadsPerThreadgroup but not
// a core count, and deriving one from the GPU name is the kind of table this
// project exists to avoid. 0 means "use the caller's default".
func (m *mtlDev) Slots() int { return 0 }

// Multiprocessors is the GPU's cores as the IORegistry states them, 0 where
// it does not.
func (m *mtlDev) Multiprocessors() int { return metal.GPUCores() }
func (m *mtlDev) Close() {
	m.c.Close()
	m.release(&ownedDevices)
}

// Mem satisfies the same optional interface cudaDev and vkDev do:
// recommendedMaxWorkingSetSize as total, minus currentAllocatedSize as free.
// Metal needs none of CUDA's owner-goroutine dance -- these are property reads
// on an MTLDevice, with no current context to be on the wrong thread for.
func (m *mtlDev) Mem() (free, total uint64, err error) { return m.c.Mem() }

// UnifiedMemory says the bytes Mem reports are the host's bytes -- true on
// every Apple Silicon machine. A caller that budgets host and device memory
// separately must subtract rather than add when this is set; see
// metal.Ctx.UnifiedMemory.
func (m *mtlDev) UnifiedMemory() bool { return m.c.UnifiedMemory() }

// The optional interfaces are asserted at compile time here, because
// cmd/jitllm reaches them by type assertion and a misspelled method would
// otherwise fail silently at run time.
var (
	_ Device  = (*mtlDev)(nil)
	_ Unified = (*mtlDev)(nil)
)

func (m *mtlDev) Alloc(n int) (Buf, error) {
	b, err := m.c.Alloc(n)
	if err != nil {
		return nil, err
	}
	// Metal's page rounding is not measured (its memory is the host's, so a
	// second context cannot fill it the way alloccount.go's table was taken):
	// counted as asked.
	return &mtlBuf{b: b, c: m.c, acc: m.take(n, n)}, nil
}

// GuaranteedLanes: a simdgroup is 32 lanes wide on every Apple GPU, which is the
// same fact msl.go's OpShuffleXor lowering already rests on -- simd_shuffle_xor
// is defined over exactly that group. Any other width is refused rather than
// approximated, for the reason the CUDA half gives: the width is what the
// lowerer assumed, not a property to be discovered.
// MTLComputePipelineState.threadExecutionWidth is known only after compiling,
// so it cannot gate the selection.
func (m *mtlDev) GuaranteedLanes(w int) (bool, string) {
	if w == ir.SubgroupLanes {
		return true, ""
	}
	return false, fmt.Sprintf("an Apple simdgroup is %d lanes and msl emits simd_shuffle_xor "+
		"over exactly those; %d cannot be asked for", ir.SubgroupLanes, w)
}

func (m *mtlDev) Compile(k *ir.Kernel) (Kernel, error) {
	compiles.Add(1)
	if CompileLog != nil {
		CompileLog(k.Name)
	}
	src, err := msl.Emit(k)
	if err != nil {
		return nil, err
	}
	kern, err := m.c.Compile(src, k.Name)
	if err != nil {
		return nil, err
	}
	// A dispatch past either limit does not fail on Metal: the kernel does not
	// run and its output stays whatever the buffer held. Apple's paravirtual
	// GPU (a macOS VM) returned unwritten rows and index-0 argmaxes that way.
	maxT, simd := kern.Limits()
	if wg := k.Group[0] * max(k.Group[1], 1) * max(k.Group[2], 1); wg > maxT {
		kern.Close()
		return nil, fmt.Errorf("metal: %s wants a %d-thread threadgroup and its pipeline allows %d on %s",
			k.Name, wg, maxT, m.c.Name())
	}
	if k.Lanes != 0 && simd != k.Lanes {
		kern.Close()
		return nil, fmt.Errorf("metal: %s is written for %d-lane simdgroups and %s runs it at %d",
			k.Name, k.Lanes, m.c.Name(), simd)
	}
	kern.SetWrites(k.WrittenParams())
	mk := &mtlKern{k: kern, group: k.Group, name: k.Name, src: k, grid: k.GridSync()}
	if k.GridSync() {
		mk.resident, mk.residentErr = mtlGridLimit(k)
	}
	return mk, nil
}

// GridLimit is one threadgroup per GPU core, as the IORegistry states the
// count (metal.GPUCores); a device it does not state cannot run a grid
// barrier.
func (d *mtlDev) GridLimit(k Kernel) (int, error) { return mtlGridLimit(k.(*mtlKern).src) }

func mtlGridLimit(k *ir.Kernel) (int, error) {
	if !k.GridSync() {
		return 0, fmt.Errorf("backend: msl: %s has no grid barrier", k.Name)
	}
	n := metal.GPUCores()
	if n == 0 {
		return 0, fmt.Errorf("backend: msl: the IORegistry states no gpu-core-count, and Metal no co-residency")
	}
	return n, nil
}

func (k *mtlKern) IR() *ir.Kernel { return k.src }

// ImportAlign and Import are newBufferWithBytesNoCopy:, which on Apple Silicon
// is the whole of what a page-in has to do: the GPU and the CPU share one pool,
// so a buffer that wraps the host pointer is the upload.
//
// The alignment is the host's page, 16 KiB on Apple Silicon, not 4096. See
// metal.Ctx.ImportAlign and kernels.HostAlign, which is the same max() from the
// other side.
func (m *mtlDev) ImportAlign() int { return m.c.ImportAlign() }

func (m *mtlDev) Import(p unsafe.Pointer, n int) (Buf, error) {
	b, err := m.c.Import(p, n)
	if err != nil {
		return nil, err
	}
	return &mtlBuf{b: b, c: m.c}, nil
}

var _ HostImport = (*mtlDev)(nil)

type mtlBuf struct {
	b   *metal.Buf
	c   *metal.Ctx
	acc *counted
}

// Write and Read are the synchronisation points. They are memmoves over shared
// memory, and since Commit does not block, the bytes are correct only after the
// work that produces them has run. Ctx.Wait is idempotent, so a read that
// follows a read costs a mutex.
func (b *mtlBuf) Write(p []byte) error {
	if err := b.c.Wait(); err != nil {
		return err
	}
	copy(b.b.Bytes(), p)
	return nil
}

func (b *mtlBuf) WriteAt(off int, p []byte) error {
	m := b.b.Bytes()
	if off < 0 || off+len(p) > len(m) {
		return fmt.Errorf("metal: writing %d bytes at offset %d of a %d-byte buffer",
			len(p), off, len(m))
	}
	if err := b.c.Wait(); err != nil {
		return err
	}
	copy(m[off:], p)
	return nil
}

func (b *mtlBuf) Read(p []byte) error {
	if err := b.c.Wait(); err != nil {
		return err
	}
	countRead(len(p))
	copy(p, b.b.Bytes())
	return nil
}

func (b *mtlBuf) Free() {
	b.b.Free()
	b.acc.give()
}

// Copy is a blit, not a memmove over the shared mapping: a memmove would need
// the queue drained first and would run on a CPU core, where the blit is
// ordered behind the committed work by the queue itself.
func (m *mtlDev) Copy(dst Buf, dstOff int, src Buf, srcOff, n int) error {
	db, ok := dst.(*mtlBuf)
	sb, ok2 := src.(*mtlBuf)
	if !ok || !ok2 || db == nil || sb == nil {
		return fmt.Errorf("backend: msl: a copy between buffers this backend did not allocate")
	}
	if err := copyRange(len(db.b.Bytes()), dstOff, len(sb.b.Bytes()), srcOff, n, db.b == sb.b); err != nil || n == 0 {
		return err
	}
	return m.c.Copy(db.b, dstOff, sb.b, srcOff, n)
}

type mtlKern struct {
	k *metal.Kernel
	// group is the workgroup the kernel was lowered for; see checkWidth.
	group [3]int
	name  string
	// src is the kernel's IR (IRSource); resident and residentErr its
	// co-residency limit when it has a grid barrier.
	src         *ir.Kernel
	resident    int
	grid        bool // src has a grid barrier: decided once at compile, read by every launch
	residentErr error
}

// checkWidth is the package's checkWidth for this kernel. On Metal a wrong
// width is a wrong answer: msl lowers ir.OpNTID to the declared group as a
// constant, so a kernel dispatched narrower than it was lowered for leaves part
// of its output unwritten.
func (k *mtlKern) checkWidth(width int) error { return checkWidth("metal", k.name, k.group, width) }

func (k *mtlKern) Launch(groups, width int, bufs ...Buf) error {
	if err := k.checkWidth(width); err != nil {
		return err
	}
	if err := checkGrid("msl", k.name, k.grid, groups, k.resident, k.residentErr); err != nil {
		return err
	}
	mb := make([]*metal.Buf, len(bufs))
	for i, b := range bufs {
		mb[i] = b.(*mtlBuf).b
	}
	return k.k.Launch(groups, width, mb...)
}

func (k *mtlKern) Close() { k.k.Close() }

// Session encodes every launch in the batch into one command buffer. Metal's
// cost is building, committing and waiting on a command buffer, not the
// dispatch, so one per launch would dominate a decode token.
//
// Uploads and readbacks need no encoding at all: on Apple Silicon the buffers
// are unified memory and Write/Read are memmoves, so they are ordered simply by
// committing before the read.
func (d *mtlDev) Session(f func(Session)) {
	t0 := time.Now()
	d.mu.Lock()
	var s *mtlSession
	if n := len(d.free); n > 0 {
		s, d.free = d.free[n-1], d.free[:n-1]
	}
	d.mu.Unlock()
	if s == nil {
		s = &mtlSession{c: d.c}
	}
	s.err = nil
	f(s)
	s.finish()
	// The context's own queue does not order against a session's queue: what
	// this Session committed (a table write, a copy) must have landed before
	// a queued session's next command buffer reads it, as on CUDA's legacy
	// stream. Without queues, in-order commit is enough.
	if d.c.HasQueues() {
		if err := d.c.WaitOwn(); s.err == nil {
			s.err = err
		}
	}
	d.mu.Lock()
	d.free = append(d.free, s)
	d.mu.Unlock()
	metal.AddSessionNs(uint64(time.Since(t0)))
}

type mtlSession struct {
	c *metal.Ctx
	// q is the queue a queued session commits to, nil for the context's own.
	q     *metal.Queue
	batch *metal.Batch
	err   error
	// mb is Launch's buffer list, reused.
	mb []*metal.Buf
}

func (s *mtlSession) Write(b Buf, p []byte) error { return s.WriteAt(b, 0, p) }

// WriteAt on a queued session waits for that queue's work alone: the memory
// is shared with the device, and only this session's earlier submissions can
// be reading its buffers. Waiting for every queue, as a write outside any
// session must, would make one session's writes a barrier for all of them.
func (s *mtlSession) WriteAt(b Buf, off int, p []byte) error {
	if s.q == nil {
		return b.WriteAt(off, p)
	}
	m := b.(*mtlBuf).b.Bytes()
	if off < 0 || off+len(p) > len(m) {
		return fmt.Errorf("metal: writing %d bytes at offset %d of a %d-byte buffer",
			len(p), off, len(m))
	}
	if err := s.q.Wait(); err != nil {
		return err
	}
	copy(m[off:], p)
	return nil
}

func (s *mtlSession) Read(b Buf, p []byte) error {
	// A read needs everything encoded so far to have run.
	if err := s.Sync(); err != nil {
		return err
	}
	if s.q != nil {
		// Sync waited for this queue's work, which is what wrote it.
		countRead(len(p))
		copy(p, b.(*mtlBuf).b.Bytes())
		return nil
	}
	return b.Read(p)
}

func (s *mtlSession) Launch(k Kernel, groups, width int, bufs ...Buf) error {
	mk := k.(*mtlKern)
	if err := mk.checkWidth(width); err != nil {
		return err
	}
	if err := checkGrid("msl", mk.name, mk.grid, groups, mk.resident, mk.residentErr); err != nil {
		return err
	}
	if s.batch == nil {
		if s.q != nil {
			s.batch = s.q.NewBatch()
		} else {
			s.batch = s.c.NewBatch()
		}
	}
	s.mb = s.mb[:0]
	for _, b := range bufs {
		s.mb = append(s.mb, b.(*mtlBuf).b)
	}
	err := s.batch.Encode(k.(*mtlKern).k, groups, width, s.mb...)
	clear(s.mb)
	return err
}

// Sync means everything encoded so far has run. Batch.Commit does not wait, so
// Sync must: tier.tuneSplit and tier.choose time a loop of launches and then
// call Sync, and without the wait they would time only the encode.
func (s *mtlSession) Sync() error {
	wait := s.c.Wait
	if s.q != nil {
		// A queued session's work is its queue's alone.
		wait = s.q.Wait
	}
	if s.batch == nil {
		return wait()
	}
	err := s.batch.Commit()
	s.batch = nil
	if e := wait(); err == nil {
		err = e
	}
	return err
}

// finish ends a session WITHOUT waiting, and that is the asynchronous
// submission this tier is built on: command buffers on one queue run in order,
// so the next session's work cannot overtake this one's, and the only consumer
// that needs the results on the host is Read -- which waits. Calling Sync here
// would put a drain at the end of every layer and give back the gain.
func (s *mtlSession) finish() {
	if s.batch != nil {
		s.err = s.batch.Commit()
		s.batch = nil
	}
}

// mtlQueue is a command queue of the device's own, with the session SessionOn
// hands out on it.
type mtlQueue struct {
	once
	mu sync.Mutex
	q  *metal.Queue
	s  mtlSession
}

func (q *mtlQueue) Close() {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.q.Close()
	q.release(&ownedQueues)
}

// NewQueue is a command queue of its own (backend.Queued).
func (d *mtlDev) NewQueue() (Queue, error) {
	q, err := d.c.NewQueue()
	if err != nil {
		return nil, err
	}
	ownedQueues.Add(1)
	return &mtlQueue{q: q}, nil
}

// SessionOn commits f's work to q's command queue. Like Session it does not
// wait at the end: a read waits, for q's work alone.
func (d *mtlDev) SessionOn(q Queue, f func(Session)) {
	mq := q.(*mtlQueue)
	mq.mu.Lock()
	defer mq.mu.Unlock()
	mq.s.c, mq.s.q, mq.s.err = d.c, mq.q, nil
	f(&mq.s)
	mq.s.finish()
}
