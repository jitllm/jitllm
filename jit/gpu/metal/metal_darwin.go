//go:build darwin

// Package metal runs MSL compute kernels through Metal.framework.
//
// No cgo: the framework is dlopened for MTLCreateSystemDefaultDevice and
// everything else goes through the Objective-C runtime. On Apple Silicon every
// buffer is unified memory, so there is no staging path.
package metal

import (
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/jit/gpu/ffi"
)

// The Objective-C runtime, hand-bound: goffi has no objc package, and each
// message below has a fixed signature the Go compiler can check.
type (
	// ID is an Objective-C object pointer, SEL a selector.
	ID  uintptr
	SEL uintptr

	// objcBool is BOOL: one byte, not a word. The runtime may return any
	// non-zero byte for YES; ok() is the one place that is decided.
	objcBool uint8
)

func (b objcBool) ok() bool { return b != 0 }

type mtlSize struct{ w, h, d uint64 }

// nsRange is NSRange: two NSUIntegers by value. The struct descriptor goes to
// goffi, which applies the ABI.
type nsRange struct{ loc, len uint64 }

var (
	objcGetClass    func(*byte) ID
	selRegisterName func(*byte) SEL

	// One binding per message shape. objc_msgSend has no signature of its own
	// (the callee's applies), so a wrong shape is a wrong-register bug rather
	// than a type error.
	send0    func(ID, SEL) ID                 // -foo
	send1id  func(ID, SEL, ID) ID             // -foo:anObject
	send1ptr func(ID, SEL, unsafe.Pointer) ID // -foo:aCString

	// Scalar-returning shapes. The width of a return value belongs to the
	// callee: reading an NSUInteger through the BOOL binding truncates it to a
	// plausible 0, and reading a BOOL through the 64-bit binding relies on
	// upper bits AAPCS64 does not define.
	sendU64  func(ID, SEL) uint64   // -fooSize, NSUInteger
	sendBool func(ID, SEL) objcBool // -hasFoo, BOOL
	// -supportsCounterSampling: takes an NSUInteger and answers BOOL.
	sendBoolU64 func(ID, SEL, uint64) objcBool
	// -objectAtIndex: takes an NSUInteger and answers an object.
	sendIdxID func(ID, SEL, uint64) ID
	// -respondsToSelector: takes a SEL and answers BOOL.
	sendResponds func(ID, SEL, SEL) objcBool
	sendF64      func(ID, SEL) float64 // -GPUStartTime, CFTimeInterval
)

var (
	createDevice func() ID
	// dispatchThreadgroups takes two MTLSize by value (24 bytes, where arm64
	// passes a pointer to a copy); goffi applies that rule.
	sendDispatch func(ID, SEL, mtlSize, mtlSize) ffi.None
	sendBuffer   func(ID, SEL, ID, uint64, uint64) ffi.None
	// setBuffers:offsets:withRange: binds the whole argument list in one
	// message instead of one per buffer.
	sendBuffers func(ID, SEL, *ID, *uint64, nsRange) ffi.None
	sendPtr     func(ID, SEL) unsafe.Pointer
	sendAlloc   func(ID, SEL, uint64, uint64) ID
	sendErr2    func(ID, SEL, ID, ID, *ID) ID
	sendErr1    func(ID, SEL, ID, *ID) ID
	// newBufferWithBytesNoCopy:length:options:deallocator: makes a buffer that
	// aliases host memory, so a page-in on unified memory moves nothing. The
	// deallocator block is NULL: the caller keeps ownership. See Ctx.Import.
	sendNoCopy func(ID, SEL, unsafe.Pointer, uint64, uint64, uintptr) ID
	// copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size: is a
	// blit encoder's buffer-to-buffer copy. See Ctx.Copy.
	sendCopy func(ID, SEL, ID, uint64, ID, uint64, uint64) ffi.None
)

// Hot selectors are resolved once: sel() allocates and calls sel_registerName
// on every use, and the encode path sends about nine messages per dispatch.
var (
	selSetPSO                        SEL
	selSetBuffers                    SEL
	selSetBuffer                     SEL
	selDispatchTG                    SEL
	selEndEncoding                   SEL
	selCommit                        SEL
	selWaitDone                      SEL
	selStatus                        SEL
	selError                         SEL
	selCmdBuf                        SEL
	selCmdBufUnret                   SEL
	selComputeEnc                    SEL
	selComputeEncType, selMemBarrier SEL
	selRetain                        SEL
	selRelease                       SEL
	selGPUStart                      SEL
	selGPUEnd                        SEL
	selResCommit                     SEL
	selResRequest                    SEL
	selResponds                      SEL
)

var loaded bool

func load() error {
	if loaded {
		return nil
	}
	lib, err := ffi.Open("/System/Library/Frameworks/Metal.framework/Metal")
	if err != nil {
		return fmt.Errorf("metal: %w", err)
	}
	createDevice = ffi.Fn0[ID](lib, "MTLCreateSystemDefaultDevice")

	objcLib, err := ffi.Open("/usr/lib/libobjc.A.dylib", "libobjc.A.dylib")
	if err != nil {
		return fmt.Errorf("metal: %w", err)
	}
	objcGetClass = ffi.Fn1[ID, *byte](objcLib, "objc_getClass")
	selRegisterName = ffi.Fn1[SEL, *byte](objcLib, "sel_registerName")
	send0 = ffi.Fn2[ID, ID, SEL](objcLib, "objc_msgSend")
	send1id = ffi.Fn3[ID, ID, SEL, ID](objcLib, "objc_msgSend")
	send1ptr = ffi.Fn3[ID, ID, SEL, unsafe.Pointer](objcLib, "objc_msgSend")
	sendU64 = ffi.Fn2[uint64, ID, SEL](objcLib, "objc_msgSend")
	// GPUStartTime is a CFTimeInterval (a C double), returned in d0 on arm64
	// rather than x0, so it needs its own binding.
	sendF64 = ffi.Fn2[float64, ID, SEL](objcLib, "objc_msgSend")
	sendBool = ffi.Fn2[objcBool, ID, SEL](objcLib, "objc_msgSend")
	sendBoolU64 = ffi.Fn3[objcBool, ID, SEL, uint64](objcLib, "objc_msgSend")
	sendIdxID = ffi.Fn3[ID, ID, SEL, uint64](objcLib, "objc_msgSend")
	sendResponds = ffi.Fn3[objcBool, ID, SEL, SEL](objcLib, "objc_msgSend")
	sendDispatch = ffi.Fn4[ffi.None, ID, SEL, mtlSize, mtlSize](objcLib, "objc_msgSend")
	sendBuffer = ffi.Fn5[ffi.None, ID, SEL, ID, uint64, uint64](objcLib, "objc_msgSend")
	sendBuffers = ffi.Fn5[ffi.None, ID, SEL, *ID, *uint64, nsRange](objcLib, "objc_msgSend")
	sendPtr = ffi.Fn2[unsafe.Pointer, ID, SEL](objcLib, "objc_msgSend")
	sendAlloc = ffi.Fn4[ID, ID, SEL, uint64, uint64](objcLib, "objc_msgSend")
	sendErr2 = ffi.Fn5[ID, ID, SEL, ID, ID, *ID](objcLib, "objc_msgSend")
	sendErr1 = ffi.Fn4[ID, ID, SEL, ID, *ID](objcLib, "objc_msgSend")
	sendNoCopy = ffi.Fn6[ID, ID, SEL, unsafe.Pointer, uint64, uint64, uintptr](objcLib, "objc_msgSend")
	sendCopy = ffi.Fn7[ffi.None, ID, SEL, ID, uint64, ID, uint64, uint64](objcLib, "objc_msgSend")
	selSetPSO = sel("setComputePipelineState:")
	selSetBuffers = sel("setBuffers:offsets:withRange:")
	selSetBuffer = sel("setBuffer:offset:atIndex:")
	selDispatchTG = sel("dispatchThreadgroups:threadsPerThreadgroup:")
	selEndEncoding = sel("endEncoding")
	selCommit = sel("commit")
	selGPUStart, selGPUEnd = sel("GPUStartTime"), sel("GPUEndTime")
	selResCommit, selResRequest = sel("commit"), sel("requestResidency")
	selResponds = sel("respondsToSelector:")
	selWaitDone = sel("waitUntilCompleted")
	selStatus = sel("status")
	selError = sel("error")
	selCmdBuf = sel("commandBuffer")
	selCmdBufUnret = sel("commandBufferWithUnretainedReferences")
	selComputeEnc = sel("computeCommandEncoder")
	selComputeEncType = sel("computeCommandEncoderWithDispatchType:")
	selMemBarrier = sel("memoryBarrierWithScope:")
	selRetain = sel("retain")
	selRelease = sel("release")
	loaded = true
	return nil
}

func sel(s string) SEL {
	b := append([]byte(s), 0)
	return selRegisterName(&b[0])
}

func class(s string) ID {
	b := append([]byte(s), 0)
	return objcGetClass(&b[0])
}

// nsstr wraps a Go string as an NSString. The byte slice is returned so the
// caller can keep it alive across the call that consumes it.
func nsstr(s string) (ID, []byte) {
	b := append([]byte(s), 0)
	id := send1ptr(class("NSString"), sel("stringWithUTF8String:"), unsafe.Pointer(&b[0]))
	return id, b
}

func gostr(id ID) string {
	if id == 0 {
		return ""
	}
	p := sendPtr(id, sel("UTF8String"))
	if p == nil {
		return ""
	}
	var out []byte
	for i := 0; ; i++ {
		c := *(*byte)(unsafe.Add(p, i))
		if c == 0 {
			return string(out)
		}
		out = append(out, c)
	}
}

func nserr(e ID, what string) error {
	if e == 0 {
		return fmt.Errorf("metal: %s failed with no NSError", what)
	}
	return fmt.Errorf("metal: %s failed: %s", what,
		gostr(send0(e, sel("localizedDescription"))))
}

// Ctx is an open device and its command queue.
type Ctx struct {
	dev  ID
	name string
	// unretained selects -commandBufferWithUnretainedReferences; see
	// Unretained.
	unretained bool
	// resSet is an MTLResidencySet attached to every queue, or 0. See Residency.
	resSet   ID
	resDirty bool
	// sub is the context's own command queue, for calls made on it directly;
	// queues (NewQueue) are others, each with its own pending list, so a
	// session waits for its own work alone. qs is every queue not closed.
	sub
	qmu sync.RWMutex
	qs  map[*Queue]bool

	// unified is -hasUnifiedMemory, read once: it is a property of the
	// hardware and cannot change under an open device.
	unified bool
	// opts is the MTLCompileOptions every library compiles under; see
	// strictMath.
	opts ID
	// spin is how long await polls before it blocks; Opts.Spin.
	spin time.Duration
	// concurrent is Opts.Concurrent.
	concurrent bool
}

func Open() (*Ctx, error) { return OpenWith(Opts{}) }

// OpenWith opens the system default device with this context's choices.
func OpenWith(o Opts) (*Ctx, error) {
	if err := load(); err != nil {
		return nil, err
	}
	d := createDevice()
	if d == 0 {
		return nil, fmt.Errorf("metal: MTLCreateSystemDefaultDevice returned nil")
	}
	q := send0(d, sel("newCommandQueue"))
	if q == 0 {
		return nil, fmt.Errorf("metal: newCommandQueue returned nil")
	}
	c := &Ctx{
		dev:        d,
		name:       gostr(send0(d, sel("name"))),
		unified:    sendBool(d, sel("hasUnifiedMemory")).ok(),
		opts:       strictMath(o.FastMath),
		spin:       o.spin(),
		concurrent: o.Concurrent,
	}
	c.sub = sub{c: c, queue: q}
	return c, nil
}

// strictMath is the compile options every library is built with: fast math
// off, where Metal's own default is on, unless the context asked for it. PTX and SPIR-V lower the IR with IEEE
// rounding; fast math may contract, reassociate and assume finite values, so
// the same IR would give a different number per kernel, and masked softmaxes
// depend on -inf handling. backend.TestRMSNormQuantIsNormThenQuantize guards
// it. An explicit fma() is still one instruction.
func strictMath(fast bool) ID {
	if fast {
		return 0 // nil options: Metal's defaults, fast math on
	}
	o := send0(send0(class("MTLCompileOptions"), sel("alloc")), sel("init"))
	if o == 0 {
		return 0
	}
	send1id(o, sel("setFastMathEnabled:"), 0)
	return o
}

func (c *Ctx) Name() string { return c.name }

// Mem reports device memory in bytes, satisfying the same optional interface
// cuda.Device and vulkan.Ctx do.
//
// Neither number describes VRAM, because there is none. total is
// -recommendedMaxWorkingSetSize, the working set the driver asks a process to
// stay under inside the machine's one pool. free is that minus
// -currentAllocatedSize for this device object only; other processes' Metal
// allocations are invisible, so free is an upper bound.
func (c *Ctx) Mem() (free, total uint64, err error) {
	if c.dev == 0 {
		return 0, 0, fmt.Errorf("metal: device is closed")
	}
	total = sendU64(c.dev, sel("recommendedMaxWorkingSetSize"))
	if total == 0 {
		return 0, 0, fmt.Errorf("metal: %s reports recommendedMaxWorkingSetSize 0", c.name)
	}
	if used := sendU64(c.dev, sel("currentAllocatedSize")); used < total {
		free = total - used
	}
	return free, total, nil
}

// UnifiedMemory reports that the bytes Mem describes are host RAM. It is asked
// rather than assumed because an Intel Mac with a discrete card answers false.
// A caller must treat the device budget as a slice of the host budget, not add
// the two.
func (c *Ctx) UnifiedMemory() bool { return c.unified }

// MaxBuffer is -maxBufferLength: the largest buffer this device will create.
func (c *Ctx) MaxBuffer() uint64 {
	if c.dev == 0 {
		return 0
	}
	return sendU64(c.dev, sel("maxBufferLength"))
}

func (c *Ctx) Close() {
	c.Wait()
	c.qmu.Lock()
	for q := range c.qs {
		q.release()
	}
	c.qs = nil
	c.qmu.Unlock()
	if c.queue != 0 {
		send0(c.queue, sel("release"))
		c.queue = 0
	}
	if c.dev != 0 {
		send0(c.dev, sel("release"))
		c.dev = 0
	}
}

// Buf is a shared-storage buffer. On Apple Silicon that is unified memory, so
// Bytes() is the same memory the GPU reads -- no upload, no flush.
type Buf struct {
	c    *Ctx
	id   ID
	host []byte
	// imported says host is the caller's memory, wrapped by Import rather
	// than allocated by Metal. -release never owned those pages.
	imported bool
}

// Imported reports that this buffer ALIASES host memory the caller owns.
func (b *Buf) Imported() bool { return b != nil && b.imported }

// ImportAlign is the alignment a host pointer and its length must have for
// Import to take them: the host page size, which is 16384 on Apple Silicon.
// newBufferWithBytesNoCopy: returns nil without an error object otherwise.
func (c *Ctx) ImportAlign() int { return os.Getpagesize() }

// Import wraps n bytes of HOST memory at p as a buffer that ALIASES them.
//
// The caller owns the memory and must outlive the buffer: the deallocator is
// NULL, and kernels.Arena's Release unmaps, so an imported buffer must be
// freed before its arena entry is released.
func (c *Ctx) Import(p unsafe.Pointer, n int) (*Buf, error) {
	const storageModeShared = 0
	a := uintptr(c.ImportAlign())
	if p == nil || n <= 0 || uintptr(p)%a != 0 || uintptr(n)%a != 0 {
		return nil, fmt.Errorf("metal: newBufferWithBytesNoCopy wants %d-byte alignment; "+
			"got pointer %p and length %d", a, p, n)
	}
	id := sendNoCopy(c.dev, sel("newBufferWithBytesNoCopy:length:options:deallocator:"),
		p, uint64(n), storageModeShared, 0)
	if id == 0 {
		return nil, fmt.Errorf("metal: newBufferWithBytesNoCopy(%p, %d) returned nil", p, n)
	}
	// The contents pointer must equal p: that equality is the no-copy claim.
	if q := sendPtr(id, sel("contents")); q != p {
		send0(id, sel("release"))
		return nil, fmt.Errorf("metal: newBufferWithBytesNoCopy(%p) has contents %p: that is a COPY", p, q)
	}
	// Imported buffers hold the weights, so they must join the residency set.
	c.addResident(id)
	return &Buf{c: c, id: id, host: unsafe.Slice((*byte)(p), n), imported: true}, nil
}

func (c *Ctx) Alloc(n int) (*Buf, error) {
	const storageModeShared = 0
	id := sendAlloc(c.dev, sel("newBufferWithLength:options:"), uint64(n), storageModeShared)
	if id == 0 {
		return nil, fmt.Errorf("metal: newBufferWithLength(%d) returned nil", n)
	}
	p := sendPtr(id, sel("contents"))
	if p == nil {
		send0(id, sel("release"))
		return nil, fmt.Errorf("metal: buffer has no contents pointer")
	}
	c.addResident(id)
	return &Buf{c: c, id: id, host: unsafe.Slice((*byte)(p), n)}, nil
}

func (b *Buf) Bytes() []byte { return b.host }

func (b *Buf) Free() {
	// The GPU may still reference the buffer through asynchronously committed
	// work, and an imported buffer's pages were never retained by Metal.
	if b.c != nil {
		b.c.Wait()
	}
	if b.id != 0 {
		send0(b.id, sel("release"))
		b.id = 0
	}
}

// Kernel is a compiled compute pipeline state.
type Kernel struct {
	c    *Ctx
	lib  ID
	fn   ID
	pso  ID
	name string
	// writes is a bit per buffer argument the kernel stores into; see
	// SetWrites. All set until told otherwise.
	writes uint64
}

// SetWrites records which buffer arguments the kernel writes (bit i for
// argument i), which a concurrent encoder orders dispatches by. A kernel never
// told is taken to write every argument.
func (k *Kernel) SetWrites(m uint64) { k.writes = m }

// Compile builds a pipeline state from MSL source. Metal compiles at runtime,
// the way the CUDA driver JITs PTX.
func (c *Ctx) Compile(src, entry string) (*Kernel, error) {
	nsSrc, keepSrc := nsstr(src)
	var e ID
	lib := sendErr2(c.dev, sel("newLibraryWithSource:options:error:"), nsSrc, c.opts, &e)
	if lib == 0 {
		return nil, nserr(e, "newLibraryWithSource")
	}
	nsName, keepName := nsstr(entry)
	fn := send1id(lib, sel("newFunctionWithName:"), nsName)
	if fn == 0 {
		send0(lib, sel("release"))
		return nil, fmt.Errorf("metal: no function %q in the compiled library", entry)
	}
	pso := sendErr1(c.dev, sel("newComputePipelineStateWithFunction:error:"), fn, &e)
	if pso == 0 {
		send0(fn, sel("release"))
		send0(lib, sel("release"))
		return nil, nserr(e, "newComputePipelineStateWithFunction")
	}
	_, _ = keepSrc, keepName
	return &Kernel{c: c, lib: lib, fn: fn, pso: pso, name: entry}, nil
}

// Batch is one command buffer holding several dispatches. Building and
// committing a command buffer is what costs on Metal, so a token's launches
// are encoded into one and committed once.
type Batch struct {
	q   *sub
	cb  ID
	enc ID
	// Scratch for setBuffers:offsets:withRange:, sized past the widest kernel
	// (the fused q/k/v matvec binds seventeen) so no dispatch allocates.
	ids  [32]ID
	offs [32]uint64
	// rd and wr are the buffers the dispatches since the last barrier read
	// and wrote, in a concurrent encoder (Encode).
	rd, wr   [64]ID
	nrd, nwr int
	conc     bool
}

// Unretained selects -commandBufferWithUnretainedReferences for every batch
// this Ctx opens afterwards. Default off. It removes the command buffer's
// retain/release of every resource, which is safe only while the buffers
// outlive the submission: true of the packed arena on a fully resident model,
// not across a page-in. It is an instrument; it made no measurable
// difference to device time.
func (c *Ctx) Unretained(on bool) { c.unretained = on }

// Residency creates an MTLResidencySet, attaches it to this Ctx's command
// queue, and adds every buffer allocated afterwards to it. Default off.
//
// The set goes on the queue, not the command buffer, so the association is
// paid once rather than per token. It is an instrument (measured neutral).
//
// macOS 15+. On an older host this returns an error rather than sending to
// nil, so a set that failed to attach cannot read as a null result.
func (c *Ctx) Residency() error {
	if c.resSet != 0 {
		return nil
	}
	d := class("MTLResidencySetDescriptor")
	if d == 0 {
		return fmt.Errorf("metal: MTLResidencySetDescriptor is not available (macOS 15+)")
	}
	desc := send0(send0(d, sel("alloc")), sel("init"))
	if desc == 0 {
		return fmt.Errorf("metal: MTLResidencySetDescriptor alloc/init returned nil")
	}
	defer send0(desc, selRelease)
	var nserrID ID
	set := sendErr1(c.dev, sel("newResidencySetWithDescriptor:error:"), desc, &nserrID)
	if set == 0 {
		if nserrID != 0 {
			return nserr(nserrID, "newResidencySetWithDescriptor")
		}
		return fmt.Errorf("metal: newResidencySetWithDescriptor returned nil")
	}
	// A queue that does not answer addResidencySet: would leave the set
	// attached to nothing while every buffer is added to it.
	if sendResponds(c.queue, selResponds, sel("addResidencySet:")) == 0 {
		send0(set, selRelease)
		return fmt.Errorf("metal: this command queue does not respond to " +
			"addResidencySet: -- the set would attach to nothing")
	}
	send1id(c.queue, sel("addResidencySet:"), set)
	c.qmu.RLock()
	for q := range c.qs {
		send1id(q.s.queue, sel("addResidencySet:"), set)
	}
	c.qmu.RUnlock()
	c.resSet = set
	c.resDirty = true
	return nil
}

// addResident puts one buffer in the set, if there is one. The set is
// committed lazily in NewBatch; committing per allocation would be quadratic.
func (c *Ctx) addResident(id ID) {
	if c.resSet == 0 || id == 0 {
		return
	}
	send1id(c.resSet, sel("addAllocation:"), id)
	c.resDirty = true
}

// NewBatch opens a command buffer and its compute encoder on the context's
// own queue.
func (c *Ctx) NewBatch() *Batch { return c.sub.newBatch() }

// newBatch opens a command buffer and its compute encoder on q's queue.
func (q *sub) newBatch() *Batch {
	c := q.c
	t0 := time.Now()
	defer func() { statNewNs.Add(uint64(time.Since(t0))) }()
	if c.resSet != 0 && c.resDirty {
		// commit publishes the additions; requestResidency asks the OS to
		// make them resident and keep them so.
		send0(c.resSet, selResCommit)
		send0(c.resSet, selResRequest)
		c.resDirty = false
	}
	sel := selCmdBuf
	if c.unretained {
		sel = selCmdBufUnret
	}
	cb := send0(q.queue, sel)
	q.mu.Lock()
	var b *Batch
	if n := len(q.freeBatch); n > 0 {
		b, q.freeBatch = q.freeBatch[n-1], q.freeBatch[:n-1]
	}
	q.mu.Unlock()
	if b == nil {
		b = &Batch{q: q}
	}
	b.cb, b.enc = cb, 0
	b.nrd, b.nwr, b.conc = 0, 0, c.concurrent
	if cb != 0 {
		if b.conc {
			b.enc = sendIdxID(cb, selComputeEncType, mtlDispatchConcurrent)
		} else {
			b.enc = send0(cb, selComputeEnc)
		}
	}
	return b
}

// Encode adds one dispatch to the batch.
func (b *Batch) Encode(k *Kernel, groups, width int, bufs ...*Buf) error {
	if b.cb == 0 || b.enc == 0 {
		return fmt.Errorf("metal: could not open a command buffer")
	}
	if err := launchable(groups); err != nil {
		return err
	}
	t0 := time.Now()
	statLaunch.Add(1)
	if b.conc {
		b.order(k, bufs)
	}
	send1id(b.enc, selSetPSO, k.pso)
	// One message for the whole argument list, from fixed arrays so no
	// dispatch allocates. A kernel with more buffers than len(b.ids) uses the
	// per-buffer setter rather than binding a prefix.
	if len(bufs) <= len(b.ids) {
		for i, buf := range bufs {
			b.ids[i], b.offs[i] = buf.id, 0
		}
		sendBuffers(b.enc, selSetBuffers, &b.ids[0], &b.offs[0],
			nsRange{0, uint64(len(bufs))})
	} else {
		for i, buf := range bufs {
			sendBuffer(b.enc, selSetBuffer, buf.id, 0, uint64(i))
		}
	}
	sendDispatch(b.enc, selDispatchTG,
		mtlSize{uint64(groups), 1, 1}, mtlSize{uint64(width), 1, 1})
	statEncodeNs.Add(uint64(time.Since(t0)))
	if kernTiming.Load() {
		b.timedSplit(k.name)
	}
	return nil
}

// kernTiming puts every dispatch in a command buffer of its own, whose
// GPUStartTime/GPUEndTime then time that kernel alone; see SetKernelTiming.
var kernTiming atomic.Bool

// SetKernelTiming arms per-kernel device timing on Metal: each dispatch is
// committed in its own command buffer, and the device's own start and end
// stamps of it are read when the work is waited for (KernelTimes). A
// measurement instrument, default off: a command buffer a dispatch costs its
// own launch and loses the concurrency between dispatches, so the split it
// gives is each kernel's time alone, not the token's.
func SetKernelTiming(on bool) { kernTiming.Store(on) }

// timedCBs names the command buffers committed under SetKernelTiming until a
// wait reads them.
var timedCBs struct {
	sync.Mutex
	name map[ID]string
	us   map[string]float64
	n    map[string]int
}

// timedSplit commits the batch's command buffer with the one dispatch just
// encoded, remembers it under the kernel's name, and opens the next.
func (b *Batch) timedSplit(name string) {
	cb := b.cb
	send0(b.enc, selEndEncoding)
	send0(cb, selCommit)
	b.q.hold(cb)
	timedCBs.Lock()
	if timedCBs.name == nil {
		timedCBs.name = map[ID]string{}
	}
	timedCBs.name[cb] = name
	timedCBs.Unlock()
	nb := b.q.newBatch()
	b.cb, b.enc, b.nrd, b.nwr = nb.cb, nb.enc, 0, 0
	nb.cb, nb.enc = 0, 0
	b.q.mu.Lock()
	b.q.freeBatch = append(b.q.freeBatch, nb)
	b.q.mu.Unlock()
}

// timedDone adds a completed command buffer's device time to its kernel's,
// when it was a timed one.
func timedDone(cb ID, t0, t1 float64) {
	timedCBs.Lock()
	defer timedCBs.Unlock()
	name, ok := timedCBs.name[cb]
	if !ok {
		return
	}
	delete(timedCBs.name, cb)
	if timedCBs.us == nil {
		timedCBs.us, timedCBs.n = map[string]float64{}, map[string]int{}
	}
	timedCBs.us[name] += (t1 - t0) * 1e6
	timedCBs.n[name]++
}

// KernelTimes returns, and clears, the per-kernel device microseconds and
// launch counts SetKernelTiming recorded, for the work waited for so far.
func KernelTimes() (us map[string]float64, n map[string]int) {
	timedCBs.Lock()
	defer timedCBs.Unlock()
	us, n = timedCBs.us, timedCBs.n
	timedCBs.us, timedCBs.n = nil, nil
	return us, n
}

// MTLDispatchTypeConcurrent and MTLBarrierScopeBuffers.
const (
	mtlDispatchConcurrent = 1
	mtlBarrierBuffers     = 1
)

// order puts a buffer barrier before a dispatch that depends on one encoded
// since the last barrier: it reads or writes a buffer one of them wrote, or
// writes one they read. Everything else runs beside what came before it,
// which is what fills the device with a block's small, independent kernels
// (k and v, gate and up) that a serial encoder ran one after another.
func (b *Batch) order(k *Kernel, bufs []*Buf) {
	has := func(set []ID, id ID) bool {
		for _, x := range set {
			if x == id {
				return true
			}
		}
		return false
	}
	hazard := b.nrd+len(bufs) > len(b.rd) || b.nwr+len(bufs) > len(b.wr)
	for i, buf := range bufs {
		if hazard {
			break
		}
		w := i >= 64 || k.writes&(1<<uint(i)) != 0
		hazard = has(b.wr[:b.nwr], buf.id) || w && has(b.rd[:b.nrd], buf.id)
	}
	if hazard {
		sendIdxID(b.enc, selMemBarrier, mtlBarrierBuffers)
		statBarrier.Add(1)
		b.nrd, b.nwr = 0, 0
	}
	for i, buf := range bufs {
		if i >= 64 || k.writes&(1<<uint(i)) != 0 {
			b.wr[b.nwr] = buf.id
			b.nwr++
		} else {
			b.rd[b.nrd] = buf.id
			b.nrd++
		}
	}
}

// statBarrier counts the buffer barriers concurrent encoders put in.
var statBarrier atomic.Uint64

// Barriers is how many buffer barriers concurrent encoders have put between
// dispatches; against SubmitTimes' dispatch count it says how much of a token
// runs side by side.
func Barriers() uint64 { return statBarrier.Load() }

// Commit ends encoding and submits. It does not wait: blocking here made every
// batch a full GPU round trip with the CPU idle. The wait is in Ctx.Wait, which
// every host read goes through; correctness rests on Metal's in-order queue.
func (b *Batch) Commit() error {
	if b.cb == 0 {
		return nil
	}
	t0 := time.Now()
	if w := lastWaitDone.Load(); w != 0 {
		// The host's share of the device's idle gap since the previous
		// Wait returned. Gaps over 5 ms are between runs, not tokens.
		if g := t0.UnixNano() - w; g < 5e6 {
			statHostGapNs.Add(uint64(g))
			statHostGapN.Add(1)
		}
		lastWaitDone.Store(0)
	}
	lastCommit.Store(t0.UnixNano())
	send0(b.enc, selEndEncoding)
	send0(b.cb, selCommit)
	b.q.hold(b.cb)
	statCommitNs.Add(uint64(time.Since(t0)))
	b.cb, b.enc = 0, 0
	// A committed batch goes back to NewBatch's free list: the caller is done
	// with it (Commit is a batch's last call), and a Batch per session was a
	// heap object per decode token.
	b.q.mu.Lock()
	b.q.freeBatch = append(b.q.freeBatch, b)
	b.q.mu.Unlock()
	return nil
}

// hold takes ownership of a committed command buffer until the next wait.
func (q *sub) hold(cb ID) {
	send0(cb, selRetain)
	q.mu.Lock()
	q.pending = append(q.pending, cb)
	statCommit.Add(1)
	q.mu.Unlock()
}

// Wait blocks until every committed command buffer has completed, every
// queue's included, and reports the first error any of them recorded.
//
// It is cheap when nothing is outstanding: it is on every host read's path.
func (c *Ctx) Wait() error {
	err := c.sub.wait()
	// Read-held across the waits, not copied out: Wait is on every write's
	// and read's path outside a session, and a snapshot was an allocation per
	// call. Close and NewQueue wait for it.
	c.qmu.RLock()
	for q := range c.qs {
		q.s.settled()
	}
	c.qmu.RUnlock()
	return err
}

// settled returns once everything committed to q's queue so far has
// completed, without taking the pending list: the session using the queue
// waits on it for its own errors and timings, and a wait that took the list
// from under it would return before its work was done.
func (q *sub) settled() {
	q.mu.Lock()
	var last ID
	if n := len(q.pending); n > 0 {
		last = q.pending[n-1]
		send0(last, selRetain)
	}
	q.mu.Unlock()
	if last != 0 {
		q.c.await(last)
		send0(last, selRelease)
	}
}

// wait blocks until every buffer committed to q's queue has completed.
func (q *sub) wait() error {
	c := q.c
	q.mu.Lock()
	cbs := q.pending
	// The other list takes the commits that arrive meanwhile, and this one
	// becomes the spare once it is read: two lists that swap, where a fresh
	// one per Wait was a heap object per decode token.
	q.pending, q.spare = q.spare[:0], nil
	statWait.Add(1)
	if len(cbs) > 0 {
		statBlock.Add(1)
	} else {
		q.spare = cbs[:0]
	}
	q.mu.Unlock()
	if len(cbs) == 0 {
		return nil
	}
	// In order on one queue: the last to complete is the last committed.
	tw := time.Now()
	c.await(cbs[len(cbs)-1])
	done := time.Now()
	statWaitNs.Add(uint64(done.Sub(tw)))
	lastWaitDone.Store(done.UnixNano())
	commitAt := lastCommit.Load()
	var err error
	for _, cb := range cbs {
		if e := send0(cb, selError); e != 0 && err == nil {
			err = nserr(e, "command buffer")
		}
		// GPUStartTime/GPUEndTime bracket execution on the device, so
		// token wall time minus this is measured rather than fitted. Valid
		// only after completion.
		if t0, t1 := sendF64(cb, selGPUStart), sendF64(cb, selGPUEnd); t1 > t0 {
			if kernTiming.Load() {
				timedDone(cb, t0, t1)
			}
			d := uint64((t1 - t0) * 1e9)
			statGPUNs.Add(d)
			statGPUN.Add(1)
			// Device idle between the previous command buffer's end and
			// this one's start, on the device clock.
			if prev := q.prevEnd; prev > 0 && t0 > prev && t0-prev < 1 {
				statGPUGapNs.Add(uint64((t0 - prev) * 1e9))
				statGPUGapN.Add(1)
			}
			q.prevEnd = t1
			// Commit-to-Wait-return host time less device busy time: the
			// launch and wake-up latency of a round trip.
			if commitAt != 0 && len(cbs) == 1 {
				if lat := done.UnixNano() - commitAt - int64(d); lat > 0 && lat < 5e6 {
					statLatNs.Add(uint64(lat))
					statLatN.Add(1)
				}
			}
			// The last buffer separately: under PerLayerSubmit it holds
			// only the head sequence, so this times it in situ.
			statGPULast.Store(d)
		}
		send0(cb, selRelease)
	}
	q.mu.Lock()
	if q.spare == nil {
		q.spare = cbs[:0]
	}
	q.mu.Unlock()
	return err
}

// await returns once cb has completed.
//
// It polls status before blocking in waitUntilCompleted, whose semaphore wake
// adds latency to every decode token, on the critical path. Polling sees
// Completed (4) or Error (5) within a few
// hundred nanoseconds; the budget bounds the spin for long buffers.
func (c *Ctx) await(cb ID) {
	if budget := c.spin; budget > 0 {
		t0 := time.Now()
		for i := 0; ; i++ {
			if st := send0(cb, selStatus); st >= 4 {
				return
			}
			// Read the clock once per 64 polls.
			if i&63 == 63 && time.Since(t0) > budget {
				break
			}
		}
	}
	send0(cb, selWaitDone)
}

// Launch dispatches groups threadgroups of width threads each and blocks.
func (k *Kernel) Launch(groups, width int, bufs ...*Buf) error {
	if err := launchable(groups); err != nil {
		return err
	}
	t0 := time.Now()
	statSolo.Add(1)
	defer func() { statSoloNs.Add(uint64(time.Since(t0))) }()
	cb := send0(k.c.queue, sel("commandBuffer"))
	if cb == 0 {
		return fmt.Errorf("metal: commandBuffer returned nil")
	}
	enc := send0(cb, sel("computeCommandEncoder"))
	if enc == 0 {
		return fmt.Errorf("metal: computeCommandEncoder returned nil")
	}
	send1id(enc, sel("setComputePipelineState:"), k.pso)
	for i, b := range bufs {
		sendBuffer(enc, sel("setBuffer:offset:atIndex:"), b.id, 0, uint64(i))
	}
	sendDispatch(enc, sel("dispatchThreadgroups:threadsPerThreadgroup:"),
		mtlSize{uint64(groups), 1, 1}, mtlSize{uint64(width), 1, 1})
	send0(enc, sel("endEncoding"))
	send0(cb, sel("commit"))
	send0(cb, sel("waitUntilCompleted"))
	if e := send0(cb, sel("error")); e != 0 {
		return nserr(e, "command buffer")
	}
	return nil
}

// Copy copies n bytes from src at srcOff to dst at dstOff with a blit
// encoder, in a command buffer of its own on the queue every session commits
// to, so it runs after everything committed before it, and blocks until it is
// done. Offsets and size are the caller's to have checked: macOS wants whole
// 4-byte words, and a blit within one buffer must not overlap itself.
func (c *Ctx) Copy(dst *Buf, dstOff int, src *Buf, srcOff, n int) error {
	if dst.c != c || src.c != c {
		return fmt.Errorf("metal: a copy between buffers of another device")
	}
	// The context's queue orders the blit after its own work; a session
	// queue's runs apart, so it is waited for first.
	if err := c.Wait(); err != nil {
		return err
	}
	cb := send0(c.queue, selCmdBuf)
	if cb == 0 {
		return fmt.Errorf("metal: commandBuffer returned nil")
	}
	enc := send0(cb, sel("blitCommandEncoder"))
	if enc == 0 {
		return fmt.Errorf("metal: blitCommandEncoder returned nil")
	}
	sendCopy(enc, sel("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"),
		src.id, uint64(srcOff), dst.id, uint64(dstOff), uint64(n))
	send0(enc, selEndEncoding)
	send0(cb, selCommit)
	send0(cb, selWaitDone)
	if e := send0(cb, selError); e != 0 {
		return nserr(e, "blit command buffer")
	}
	return nil
}

// Limits is what this compiled pipeline can be dispatched with: its largest
// threadgroup, which falls below the device's with register pressure, and its
// SIMD width. Metal does not refuse a dispatch past either; the kernel
// silently does not run.
func (k *Kernel) Limits() (maxThreads, simdWidth int) {
	return int(sendU64(k.pso, sel("maxTotalThreadsPerThreadgroup"))),
		int(sendU64(k.pso, sel("threadExecutionWidth")))
}

func (k *Kernel) Close() {
	for _, id := range []*ID{&k.pso, &k.fn, &k.lib} {
		if *id != 0 {
			send0(*id, sel("release"))
			*id = 0
		}
	}
}

// statCommit, statWait and statBlock count commits, waits and waits that
// blocked. Package-level because a process has one Metal device.
var statCommit, statWait, statBlock atomic.Uint64

// statEncodeNs and statWaitNs split the CPU's time between building the
// command buffer and blocking on the GPU.
var statEncodeNs, statWaitNs, statLaunch atomic.Uint64

// statSolo counts dispatches that went through Kernel.Launch -- one command
// buffer, committed AND WAITED, for a single dispatch. Inside a Session they go
// through Batch.Encode instead and share one buffer.
var statSolo, statSoloNs atomic.Uint64

// statSessionNs is wall time inside backend Session. It bounds submission from
// above.
var statSessionNs, statSessionN atomic.Uint64

// statNewNs is time creating the command buffer and its encoder; statCommitNs
// is endEncoding plus commit.
var statNewNs, statCommitNs atomic.Uint64

// statGPUNs is the device's own execution time, summed from every completed
// command buffer's GPUStartTime/GPUEndTime.
var statGPUNs, statGPUN, statGPULast atomic.Uint64

// statGPUGapNs sums the device's idle time between consecutive command
// buffers (GPUStartTime of one less GPUEndTime of the one before), on the
// device's own clock; each queue keeps its previous end (sub.prevEnd).
var statGPUGapNs, statGPUGapN atomic.Uint64

// statHostGapNs is host time from a blocking Wait's return to the next
// Commit: the CPU's share of the device's idle gap.
var statHostGapNs, statHostGapN atomic.Uint64
var lastWaitDone atomic.Int64

// statLatNs is commit-to-Wait-return host time less GPU busy time, for a Wait
// covering one command buffer: the launch and wake-up latency of a round trip.
var statLatNs, statLatN atomic.Uint64
var lastCommit atomic.Int64

// RoundTrip reports the launch plus wake-up latency summed over n round trips.
func RoundTrip() (ns, n uint64) { return statLatNs.Load(), statLatN.Load() }

// HostGap reports the host's nanoseconds from a Wait's return to the next
// commit, and over how many.
func HostGap() (ns, n uint64) { return statHostGapNs.Load(), statHostGapN.Load() }

// GPUIdle reports the device's idle nanoseconds between consecutive command
// buffers, and how many gaps that sums.
func GPUIdle() (ns, gaps uint64) { return statGPUGapNs.Load(), statGPUGapN.Load() }

// GPUBusy reports nanoseconds the device spent executing, and over how many
// command buffers. Against a token's wall time it splits the token without a
// fitted parameter.
func GPUBusy() (ns, buffers uint64) { return statGPUNs.Load(), statGPUN.Load() }

// GPUBusyLast is the LAST completed command buffer's own execution time. Under
// one-submission-per-block it is the head sequence's, measured where it runs.
func GPUBusyLast() uint64 { return statGPULast.Load() }

// SubmitStats reports commits, Wait calls, and Waits that actually blocked.
func SubmitStats() (commits, waits, blocked uint64) {
	return statCommit.Load(), statWait.Load(), statBlock.Load()
}

// SubmitTimes reports nanoseconds spent encoding and nanoseconds spent blocked,
// plus the number of dispatches encoded.
func SubmitTimes() (encodeNs, waitNs, launches uint64) {
	return statEncodeNs.Load(), statWaitNs.Load(), statLaunch.Load()
}

// SoloStats reports dispatches that took a command buffer of their own, and the
// nanoseconds they cost. Each is a full GPU round trip.
func SoloStats() (n, ns uint64) { return statSolo.Load(), statSoloNs.Load() }

// AddSessionNs records one backend Session's wall time.
func AddSessionNs(ns uint64) { statSessionNs.Add(ns); statSessionN.Add(1) }

// SessionStats reports sessions and their total wall nanoseconds.
func SessionStats() (n, ns uint64) { return statSessionN.Load(), statSessionNs.Load() }

// PhaseStats splits a Session: creating the command buffer, and committing it.
func PhaseStats() (newNs, commitNs uint64) { return statNewNs.Load(), statCommitNs.Load() }

// MTLCounterSamplingPoint. Only the two that could time a compute dispatch are
// named; the draw and tile-dispatch points are graphics.
const (
	sampleAtStageBoundary    = 0
	sampleAtDispatchBoundary = 2
)

// CounterSampling reports where this device can sample GPU counters, which
// decides whether a per-dispatch timing instrument is possible. At a dispatch
// boundary the shipping encoder can be sampled between launches; at a stage
// boundary only, per-dispatch timing would need one serial encoder per
// dispatch and would measure itself.
func (c *Ctx) CounterSampling() (stage, dispatch bool) {
	selSupports := sel("supportsCounterSampling:")
	if !sendResponds(c.dev, selResponds, selSupports).ok() {
		return false, false
	}
	return sendBoolU64(c.dev, selSupports, sampleAtStageBoundary).ok(),
		sendBoolU64(c.dev, selSupports, sampleAtDispatchBoundary).ok()
}

// CounterSets lists the device's counter set names, so "there is no timestamp
// counter set" is distinguishable from "sampling is unsupported".
func (c *Ctx) CounterSets() []string {
	sets := send0(c.dev, sel("counterSets"))
	if sets == 0 {
		return nil
	}
	n := sendU64(sets, sel("count"))
	out := make([]string, 0, n)
	for i := uint64(0); i < n; i++ {
		cs := sendIdxID(sets, sel("objectAtIndex:"), i)
		if cs == 0 {
			continue
		}
		out = append(out, nsString(send0(cs, sel("name"))))
	}
	return out
}

// nsString copies an NSString's UTF-8 bytes out. Returns "" for nil, which is
// what -name gives for a counter set the driver does not label.
func nsString(s ID) string {
	if s == 0 {
		return ""
	}
	p := sendPtr(s, sel("UTF8String"))
	if p == nil {
		return ""
	}
	b := (*[1 << 12]byte)(p)
	for i := 0; i < len(b); i++ {
		if b[i] == 0 {
			return string(b[:i])
		}
	}
	return ""
}

// sub is one command queue and what its submissions need: the committed
// buffers not yet waited for (in order on the queue, so waiting on the last
// implies all completed, but each is checked for its own error, and each is
// retained while held: -commandBuffer returns an autoreleased object and no
// pool is drained), the list pending swaps with at each wait, the committed
// batches newBatch hands out again, and the device-clock end of the last
// buffer waited for.
type sub struct {
	c         *Ctx
	queue     ID
	mu        sync.Mutex
	pending   []ID
	spare     []ID
	freeBatch []*Batch
	prevEnd   float64
}

// Queue is a command queue of its own on the device: a session committing to
// it runs beside sessions on other queues, and waits for its own buffers
// alone. One goroutine uses a Queue at a time.
type Queue struct{ s sub }

// NewQueue opens a command queue on this device.
func (c *Ctx) NewQueue() (*Queue, error) {
	id := send0(c.dev, sel("newCommandQueue"))
	if id == 0 {
		return nil, fmt.Errorf("metal: newCommandQueue returned nil")
	}
	q := &Queue{s: sub{c: c, queue: id}}
	c.qmu.Lock()
	if c.resSet != 0 {
		send1id(id, sel("addResidencySet:"), c.resSet)
	}
	if c.qs == nil {
		c.qs = map[*Queue]bool{}
	}
	c.qs[q] = true
	c.qmu.Unlock()
	return q, nil
}

// NewBatch opens a command buffer and its compute encoder on the queue.
func (q *Queue) NewBatch() *Batch { return q.s.newBatch() }

// Wait blocks until everything committed to the queue has completed.
func (q *Queue) Wait() error { return q.s.wait() }

// Close waits for the queue's work and releases it.
func (q *Queue) Close() {
	c := q.s.c
	c.qmu.Lock()
	open := c.qs[q]
	delete(c.qs, q)
	c.qmu.Unlock()
	if open {
		q.s.wait()
		q.release()
	}
}

func (q *Queue) release() {
	if q.s.queue != 0 {
		send0(q.s.queue, sel("release"))
		q.s.queue = 0
	}
}

// HasQueues reports whether a session queue is open on this context.
func (c *Ctx) HasQueues() bool {
	c.qmu.RLock()
	defer c.qmu.RUnlock()
	return len(c.qs) > 0
}

// WaitOwn blocks until the context's own queue has run everything committed
// to it, without waiting for the session queues.
func (c *Ctx) WaitOwn() error { return c.sub.wait() }
