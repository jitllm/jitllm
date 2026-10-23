// Package backend puts one interface over the three GPU runtimes so a
// benchmark or a tuner can ask the same question of each.
//
// What it deliberately does not abstract: memory. CUDA buffers are device
// memory reached by explicit copy, Vulkan and Metal buffers are host-mapped.
// Write/Read is the honest common denominator -- it is a real copy on CUDA and
// a memmove into a mapping elsewhere -- rather than pretending every device has
// a pointer the host can hold.
package backend

import (
	"fmt"
	"os"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/backend/metal"
	"github.com/jitllm/jitllm/backend/vulkan"
	"github.com/jitllm/jitllm/jit/ir"
)

// verbose traces every backend's open attempt, and the tier's device probe and
// ordering, to stderr. The package reads no environment; tier sets it.
var verbose bool

// SetVerbose turns the open/probe trace on. tier.WithVerbose calls it.
func SetVerbose(on bool) { verbose = on }

// Verbose reports whether the trace is on.
func Verbose() bool { return verbose }

// mtlUnretained selects -commandBufferWithUnretainedReferences on Metal, for
// every device opened afterwards. Default off. It is a package knob because it
// configures a device before there is an instance, and a measurement
// instrument; see metal.Ctx's Unretained for what it trades.
var mtlUnretained bool

// SetMetalUnretained arms the unretained command buffer on Metal devices opened
// after this call.
func SetMetalUnretained(on bool) { mtlUnretained = on }

// cudaTiming arms device-clock timing of every replayed graph on CUDA; see
// CUDAGraphTimes. A measurement instrument, default off: it adds two event
// records per replay.
var cudaTiming atomic.Bool

// SetCUDATiming arms or disarms graph timing on CUDA devices.
func SetCUDATiming(on bool) { cudaTiming.Store(on) }

// mtlResidency arms an MTLResidencySet on Metal devices opened after this call.
var mtlResidency bool

// SetMetalResidency arms the residency set. Opening fails on a host or queue
// that cannot carry one, rather than silently measuring without it.
func SetMetalResidency(on bool) { mtlResidency = on }

type Buf interface {
	Write([]byte) error
	// WriteAt copies p in at a byte offset, and refuses an offset+length that
	// would run past the allocation. It assembles a buffer from runs that are
	// not adjacent on the host (a routed mixture's k expert sheets) without a
	// host-side concatenation.
	WriteAt(off int, p []byte) error
	Read([]byte) error
	Free()
}

type Kernel interface {
	// Launch runs groups workgroups of width threads and blocks until done.
	// width must be the workgroup the kernel declares (ir.Kernel.Group, one
	// dimension); every backend refuses any other, see checkWidth.
	Launch(groups, width int, bufs ...Buf) error
	Close()
}

// checkWidth refuses a launch whose workgroup is not the one the kernel
// declares, on every backend, by the kernel's name.
//
// The three runtimes disagree about which of the two widths runs. CUDA takes
// the launch's (cuLaunchKernel's blockDim; ptx reads %ntid.x at run time);
// Vulkan runs the LocalSize the SPIR-V declares and ignores the launch's; Metal
// dispatches the launch's while msl lowers ir.OpNTID to the declared group as a
// constant. So a mismatch is a different answer per backend: kernels.HCMix
// declared 64 and was launched 128 wide, which CUDA ran right and Vulkan ran on
// the first 64 rows of every 128, leaving the rest stale with no error. The
// launch interface is one-dimensional, so a kernel declaring a second or third
// dimension is refused too: no launch could match it.
func checkWidth(api, name string, group [3]int, width int) error {
	if width == group[0] && group[1] <= 1 && group[2] <= 1 {
		return nil
	}
	return fmt.Errorf("backend: %s: %s declares a %dx%dx%d workgroup and was launched %d wide",
		api, name, group[0], group[1], group[2], width)
}

// copyRange is every backend's refusal for Device.Copy, so the three refuse
// the same copies: dl and sl are the two buffers' sizes in bytes, and same
// says both ranges are in one buffer.
func copyRange(dl, dstOff, sl, srcOff, n int, same bool) error {
	switch {
	case n < 0 || dstOff < 0 || srcOff < 0:
		return fmt.Errorf("backend: copy of %d bytes from offset %d to %d: negative", n, srcOff, dstOff)
	case n%4 != 0 || dstOff%4 != 0 || srcOff%4 != 0:
		return fmt.Errorf("backend: copy of %d bytes from offset %d to %d: not whole 4-byte words",
			n, srcOff, dstOff)
	case srcOff+n > sl:
		return fmt.Errorf("backend: copy of %d bytes from offset %d of a %d-byte buffer", n, srcOff, sl)
	case dstOff+n > dl:
		return fmt.Errorf("backend: copy of %d bytes to offset %d of a %d-byte buffer", n, dstOff, dl)
	case same && n > 0 && srcOff < dstOff+n && dstOff < srcOff+n:
		return fmt.Errorf("backend: copy of %d bytes from offset %d to %d of one buffer overlaps itself",
			n, srcOff, dstOff)
	}
	return nil
}

// hostReads and hostReadBytes count every readback from a device buffer to
// the host, on every backend; see HostReads.
var hostReads, hostReadBytes atomic.Uint64

// HostReads is how many reads from a device buffer to the host this process
// has made, and the bytes they carried. A path that must stay on the device --
// a recurrent pool's resize carries every session's state across with Copy --
// is held to it: the count does not move.
func HostReads() (calls, bytes uint64) { return hostReads.Load(), hostReadBytes.Load() }

func countRead(n int) {
	hostReads.Add(1)
	hostReadBytes.Add(uint64(n))
}

// BufferLimited is a Device that cannot address a buffer past some size in one
// binding (Vulkan's maxStorageBufferRange, Metal's maxBufferLength). A device
// without it has no such limit below what it can allocate.
type BufferLimited interface {
	MaxBuffer() uint64
}

// Unified is a Device that can say whether its memory is the host's (an
// integrated GPU, Apple Silicon): a budget for it is a slice of the host's,
// subtracted rather than added.
type Unified interface {
	UnifiedMemory() bool
}

// OrdinalDevice is a Device that knows its index among its backend's devices.
type OrdinalDevice interface{ Ordinal() int }

type Device interface {
	// Name is the hardware; API is which of PTX/SPIR-V/MSL got it there.
	Name() string
	API() string
	// Mem is how much memory the device has: total, and what is free of it,
	// as its driver reports it.
	Mem() (free, total uint64, err error)

	// Slots is how many threads this device wants resident to hide memory
	// latency, or 0 when the runtime will not say. It is a capacity, not a
	// limit: the attention accumulation splits its position loop across
	// Slots/qdim threads (tier.accSplitFor), and the right number differs by an
	// order of magnitude across cards. CUDA answers; the others return 0 and
	// the caller uses a default.
	Slots() int
	Compile(k *ir.Kernel) (Kernel, error)
	Alloc(n int) (Buf, error)
	// Copy moves n bytes from src at srcOff to dst at dstOff without the host
	// in between: cuMemcpyDtoD, vkCmdCopyBuffer, a Metal blit. The offsets and
	// n are whole 4-byte words (Metal on macOS takes nothing finer, and the
	// three must mean one thing), both ranges lie inside their buffers, and
	// two ranges of ONE buffer must not overlap -- every driver leaves that
	// undefined. Anything else is refused before the device sees it. It runs
	// outside a Session, as Buf's methods do, and returns once the bytes have
	// landed, ordered after every launch and session before it.
	Copy(dst Buf, dstOff int, src Buf, srcOff, n int) error
	Close()

	// Session runs f with the device held, so a sequence of calls pays the
	// ownership cost once. On CUDA the hop to the device-owning goroutine costs
	// far more than the FFI call itself. On Vulkan and Metal there is no
	// per-thread context and Session simply calls f.
	Session(f func(s Session))
}

// Queued is a Device whose sessions can each run on a queue of their own, so
// several sessions are on the device at once instead of one after another:
// a CUDA non-blocking stream, a Vulkan command buffer and fence of its own, a
// Metal command queue. A Session on a queue is ordered against its own
// queue's earlier work only; Copy and the Buf methods, outside any Session,
// still wait for everything before them, every queue included.
//
// One queue runs one Session at a time: SessionOn for a queue already in a
// Session waits. A Device without it serialises Session as before.
type Queued interface {
	NewQueue() (Queue, error)
	SessionOn(q Queue, f func(s Session))
}

// Queue is one of a Queued device's queues. Close releases it; a Session on
// it must not be running.
type Queue interface{ Close() }

// HostPinner is an optional Device that can hand out page-locked host memory,
// which its transfers read at the link's rate. Pin and Unpin may be called
// inside a Session. CUDA implements it; on Vulkan and Metal the mapped
// buffers are already the transfer path and nothing is gained.
// Uploader is an optional Session that copies host memory to a buffer on a
// copy queue of the device's own, beside the session's launches: an upload
// the next launches do not read needs no place in their order. p must stay
// untouched until UploadWait returns (page-locked memory makes the copy
// asynchronous; pageable memory is staged by the driver before Upload
// returns).
type Uploader interface {
	Upload(b Buf, off int, p []byte) error
	// UploadWait returns once every Upload issued before it has landed.
	UploadWait() error
}

type HostPinner interface {
	PinHost(n int) ([]byte, error)
	UnpinHost(p []byte)
}

// PeerCopier is an optional Device that copies from another device of its own
// backend without the bytes passing through memory the caller owns: a block
// moving between two cards takes its history across in one call instead of a
// read home and a write up. CUDA implements it (cuMemcpyPeer).
type PeerCopier interface {
	// CopyPeer copies n bytes of src, a buffer of device from, at srcOff into
	// dst, a buffer of this device, at dstOff, landed when it returns. direct
	// reports that the bytes went card to card (NVLink or PCIe peer access)
	// rather than staged through the host by the driver.
	CopyPeer(dst Buf, dstOff int, from Device, src Buf, srcOff, n int) (direct bool, err error)
	// PeerAccess enables what CopyPeer from device from would enable on its
	// first call, and reports whether the copies go card to card: a caller
	// that knows a copy is coming asks ahead, outside the copy's own time.
	PeerAccess(from Device) bool
}

// CopyRange is one device-to-device copy of a CopyLister's list: N bytes of
// Src at SrcOff into Dst at DstOff, both buffers of the one device.
type CopyRange struct {
	Dst       Buf
	DstOff    int
	Src       Buf
	SrcOff, N int
}

// CopyLister is an optional Device that runs a list of copies as one call:
// one ownership hop and one wait for the lot, where Copy pays both per copy.
// A restore point's snapshot of every recurrent block is such a list. CUDA
// implements it; CopyAll runs a list on any Device.
type CopyLister interface {
	CopyList(cs []CopyRange) error
}

// CopyAll runs cs on d: as one CopyList where d implements it, one Copy at a
// time otherwise, stopping at the first error.
func CopyAll(d Device, cs []CopyRange) error {
	if l, ok := d.(CopyLister); ok {
		return l.CopyList(cs)
	}
	for _, c := range cs {
		if err := d.Copy(c.Dst, c.DstOff, c.Src, c.SrcOff, c.N); err != nil {
			return err
		}
	}
	return nil
}

// Session is a batch of device calls that runs under one ownership hop. The
// methods mirror Buf and Kernel but skip the hop, so they are only valid inside
// the closure Device.Session was given.
// TrialPrefix leads the name of a kernel-timing entry a tuner's trial
// launched (MarkTrial), so a profile reads the work apart from the trials
// that ran beside it.
const TrialPrefix = "trial:"

// TrialMarker is a Session whose launches can be marked as a tuner's trial
// for the kernel-timing instrument.
type TrialMarker interface {
	MarkTrial()
}

// MarkTrial marks every launch s makes from here as a tuner's trial in the
// kernel-timing instrument (TrialPrefix); a backend with no instrument
// ignores it.
func MarkTrial(s Session) {
	if t, ok := s.(TrialMarker); ok {
		t.MarkTrial()
	}
}

type Session interface {
	Write(b Buf, p []byte) error
	// WriteAt is Buf.WriteAt without the per-call ownership hop; see Session.
	WriteAt(b Buf, off int, p []byte) error
	Read(b Buf, p []byte) error
	// Launch does not wait for the kernel. Consecutive launches on one stream
	// are already ordered and the readback that follows synchronises, so a
	// wait per launch would only add round trips. Sync is for callers that need
	// the wait without a readback (benchmarks).
	Launch(k Kernel, groups, width int, bufs ...Buf) error
	Sync() error
}

// HostImport is the optional interface a backend implements when the device can
// use host memory as a buffer without copying it.
//
// On a device whose heap is system memory (an integrated GPU, Apple Silicon),
// uploading a weight copies host memory to host memory and holds it twice; an
// import makes a page-in a descriptor update instead.
//
// Implementing it is not a reason to use it: a discrete card may advertise
// VK_EXT_external_memory_host, where an imported weight is re-read across PCIe
// by every matvec. UnifiedMemory() is the guard, and applying it is the
// caller's job.
type HostImport interface {
	// ImportAlign is the alignment the pointer AND the length must both have,
	// or 0 when this device cannot import at all. Both, because the drivers
	// require both: Vulkan's minImportedHostPointerAlignment applies to the
	// allocation size as well as the address, and Metal wants whole pages.
	ImportAlign() int
	// Import wraps n bytes at p as a buffer that ALIASES them. The caller keeps
	// ownership of the memory and must outlive the buffer: freeing the buffer
	// releases the wrapper, never the pages.
	Import(p unsafe.Pointer, n int) (Buf, error)
}

// ImportAlign asks a device what a host pointer must satisfy to be imported. 0
// means it cannot be, which is the answer for every discrete card and for any
// backend that does not implement HostImport at all.
func ImportAlign(d Device) int {
	h, ok := d.(HostImport)
	if !ok {
		return 0
	}
	return h.ImportAlign()
}

// Lanes is the optional interface a backend implements to say what subgroup
// width it can guarantee for a compute pipeline -- not what it usually does,
// and not what it reports as a default.
//
// It is a precondition, not a report. Kernels whose answer depends on 32 lanes
// in one subgroup (a butterfly ShuffleXor reduction, the warp matrix
// instruction) are undefined behaviour at a narrower width, and a probe passing
// does not prove the width inside a real submission. So the width is asked for
// before a kernel is selected, and a device that will not promise it gets the
// scalar twin.
//
// A backend that does not implement this promises nothing -- see
// GuaranteedLanes, which is what every caller should use.
type Lanes interface {
	// GuaranteedLanes reports whether a kernel needing w lanes per subgroup can
	// be created on this device, and says why not when it cannot. The reason is
	// reported beside the kernel that was chosen instead, because "the check
	// failed" and "the device cannot be asked" are different facts.
	GuaranteedLanes(w int) (bool, string)
}

// GuaranteedLanes asks a device for a subgroup-width promise. A device that
// cannot answer refuses, which is the safe direction: the question only ever
// gates a kernel that has a scalar twin.
func GuaranteedLanes(d Device, w int) (bool, string) {
	if w <= 0 {
		return true, ""
	}
	if l, ok := d.(Lanes); ok {
		return l.GuaranteedLanes(w)
	}
	return false, fmt.Sprintf("%s does not report a subgroup-width guarantee, so %d lanes "+
		"cannot be assumed", d.API(), w)
}

// Recording is a launch sequence captured once, replayable with one driver
// call. Free must be called to release it.
type Recording interface{ Free() }

// Recorder is a Session that can record the launches issued between
// BeginRecord and EndRecord instead of running them, and replay that sequence
// later. Record is the usual way to drive it.
//
// It is an optional interface because only CUDA has it (stream capture);
// Vulkan and Metal build a command buffer per submission anyway. A caller
// type-asserts and keeps its direct path for the rest.
//
// A recording does NOT run its launches -- capture builds a graph and executes
// nothing -- so a caller that captures must Replay before reading a result.
// Writes and reads must stay outside it: a captured memcpy would bake in the
// HOST address it was given, and jitllm's host staging buffers belong to the Go
// heap. EndRecord must follow a successful BeginRecord even when a launch in
// between failed.
type Recorder interface {
	BeginRecord() error
	EndRecord() (Recording, error)
	Replay(r Recording) error
}

// Record captures the launches f issues on rec's session.
//
// It is a function rather than a Recorder method so that f does not escape: a
// func handed to an interface method is a heap object, and the decode
// submission records with a closure over the whole submission's state, which
// made every token's closures heap objects even on the tokens that only
// replay.
func Record(rec Recorder, f func()) (Recording, error) {
	if err := rec.BeginRecord(); err != nil {
		return nil, err
	}
	f()
	return rec.EndRecord()
}

// Open returns every backend that initialises on this host, in the order
// CUDA, HIP, Vulkan, Metal. Absent hardware is not an error: a machine with no GPU
// returns an empty slice, which is the same shape as a machine with three.
func Open() []Device { return OpenWith(openDefaults) }

// openDefaults is what Open opens with. It is the zero Opts in every build that
// ships; only this package's own tests set it (export_test.go), so the whole
// suite -- which opens its devices through Open -- can be pointed at one
// Vulkan device, or run with a lowered workgroup limit so every kernel family
// goes through a split launch.
var openDefaults Opts

// Opts are the choices a device is opened with. They belong to the device, not
// the process, so two tiers in one process can open theirs differently. Zero is
// the shipping default.
type Opts struct {
	// Vulkan is the Vulkan device pin and subgroup override.
	Vulkan vulkan.Config
	// CUDAPosted posts every CUDA device call to the device's owner goroutine
	// instead of running it inline on the caller's thread.
	CUDAPosted bool
	// CUDAMaxRegs caps every CUDA kernel's registers per thread (.maxnreg),
	// 0 leaving ptxas its own choice. It exists to force a spill: the
	// register gate is run against it to show it fires.
	CUDAMaxRegs int
	// Metal is the Metal context's compile and wait choices.
	Metal metal.Opts
	// HIP is where ROCm is looked for. Zero searches the default places.
	HIP HIPConfig
	// Enumerating marks an open that is part of finding devices (a default
	// pick, a loop over a backend's ordinals) rather than a caller naming one.
	// It changes nothing about the open; an installed Restriction treats a
	// refused enumerating open as the restriction narrowing the search, and a
	// refused named open as a violation.
	Enumerating bool
}

// OpenWith is Open with the devices' open choices.
func OpenWith(o Opts) []Device {
	var out []Device
	for _, f := range []func(Opts) (Device, error){openCUDA, openHIP, openVulkan, openMetal} {
		t0 := time.Now()
		d, err := f(enumerating(o))
		if verbose {
			name, api := "-", "-"
			if d != nil {
				name, api = d.Name(), d.API()
			}
			fmt.Fprintf(os.Stderr, "jitllm: open %-6s %-40s %6.0f ms  %v\n", api, name,
				float64(time.Since(t0).Microseconds())/1000, err)
		}
		if err == nil && d != nil {
			out = append(out, d)
		}
	}
	return out
}
