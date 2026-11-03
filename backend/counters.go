package backend

import (
	"sync"
	"sync/atomic"

	"github.com/jitllm/jitllm/backend/metal"
)

// Counters is what a set of devices counts together: the kernels they
// compile, their readbacks to the host, the allocations refused for want of
// memory, the CUDA calls posted to an owner goroutine, the driver objects held
// open, the kernels a close found still open, and the device-clock timing a
// measurement arms. Devices opened with one Counters (Opts.Counters) count into
// it together; a device opened without counts into its own. CountersOf finds a
// device's.
type Counters struct {
	compiles atomic.Int64
	// CompileLog, when set, is told the name of every kernel Metal and CUDA
	// compile.
	CompileLog func(name string)

	hostReads, hostReadBytes atomic.Uint64

	allocRefused atomic.Int64
	postedCalls  atomic.Int64
	ownerCalls   atomic.Int64

	ownedDevices, ownedQueues atomic.Int64

	orphans struct {
		sync.Mutex
		names []string
	}

	// cudaTiming arms device-clock timing of every replayed graph on CUDA;
	// see CUDAGraphTimes. cudaKernTiming arms an event pair around every
	// launch outside a capture; see SetCUDAKernelTiming.
	cudaTiming, cudaKernTiming atomic.Bool
	// directMarks holds the launches timed outside a capture until
	// CUDAKernelTimes reads them; kernTimes sums device microseconds and
	// counts per kernel name; graphTimes holds the last replays' device
	// times, in microseconds.
	directMarks kernPairs
	kernTimes   kernTotals
	graphTimes  replayTimes

	// Metal is what the Metal contexts opened with these counters count,
	// unless their Opts.Metal names other Stats.
	Metal metal.Stats
}

// counter is a device that knows the Counters it counts into.
type counter interface{ counters() *Counters }

// CountersOf is the Counters device d counts into, or nil for a device this
// package did not open.
func CountersOf(d Device) *Counters {
	if c, ok := d.(counter); ok {
		return c.counters()
	}
	return nil
}

// counters is o's Counters, or new ones when it names none.
func (o Opts) counters() *Counters {
	if o.Counters != nil {
		return o.Counters
	}
	return &Counters{}
}

// Compiles is how many kernels these devices have been asked to compile: what
// a gate reads across a prompt to show that a placed model's first prompt
// builds nothing. A compile belongs before the device is ready, not inside a
// time to first token.
func (c *Counters) Compiles() int64 { return c.compiles.Load() }

// compiled counts one compile and tells CompileLog its name.
func (c *Counters) compiled(name string) {
	c.compiles.Add(1)
	if c.CompileLog != nil {
		c.CompileLog(name)
	}
}

// HostReads is how many reads from a device buffer to the host these devices
// have made, and the bytes they carried. A path that must stay on the device
// -- a recurrent pool's resize carries every session's state across with Copy
// -- is held to it: the count does not move.
func (c *Counters) HostReads() (calls, bytes uint64) {
	return c.hostReads.Load(), c.hostReadBytes.Load()
}

func (c *Counters) countRead(n int) {
	c.hostReads.Add(1)
	c.hostReadBytes.Add(uint64(n))
}

// AllocRefused is how many device allocations were refused because the card
// was full -- by this process or, on a shared card, by another one. Placement
// declines around it; a gate that needs a model on the card reads it to tell
// "the card is too small here" from a failure.
func (c *Counters) AllocRefused() int64 { return c.allocRefused.Load() }

// PostedCalls is how many CUDA calls were posted to an owner goroutine rather
// than run inline; a gate holds it still across a Session.
func (c *Counters) PostedCalls() int64 { return c.postedCalls.Load() }

// OwnerCalls is how many device calls were made from inside a Session. The
// backend completes them, so this is a hygiene count, not an error count.
func (c *Counters) OwnerCalls() int64 { return c.ownerCalls.Load() }

// OwnedNow is the count of live driver objects these devices hold, every
// backend together.
func (c *Counters) OwnedNow() Owned {
	return Owned{Devices: c.ownedDevices.Load(), Queues: c.ownedQueues.Load()}
}

// Orphans is the entry point of every Vulkan kernel a device's Close found its
// owner had not closed, in the order found.
func (c *Counters) Orphans() []string {
	c.orphans.Lock()
	defer c.orphans.Unlock()
	return append([]string(nil), c.orphans.names...)
}

func (c *Counters) orphaned(names []string) {
	if len(names) == 0 {
		return
	}
	c.orphans.Lock()
	c.orphans.names = append(c.orphans.names, names...)
	c.orphans.Unlock()
}

// SetCUDATiming arms or disarms graph timing on these CUDA devices; see
// CUDAGraphTimes. A measurement instrument, default off: it adds two event
// records per replay.
func (c *Counters) SetCUDATiming(on bool) { c.cudaTiming.Store(on) }

// SetCUDAKernelTiming arms per-kernel device-clock timing on CUDA: a launch
// outside a capture runs between two events, read when CUDAKernelTimes is
// called after the work is done; a capture records an event node after every
// launch, and each replay's kernels are read back one replay later -- the SAME
// measurement scripts/kshim.c takes of llama.cpp's graphs, so the two engines'
// kernels can be set side by side. It applies to graphs recorded after it is
// armed. A measurement instrument, default off: the event nodes cost device
// time of their own.
func (c *Counters) SetCUDAKernelTiming(on bool) { c.cudaKernTiming.Store(on) }

// CUDAKernelTimes returns, and clears, the per-kernel device microseconds and
// launch counts SetCUDAKernelTiming collected.
func (c *Counters) CUDAKernelTimes() (us map[string]float64, n map[string]int) {
	c.directMarks.settle(&c.kernTimes)
	c.kernTimes.Lock()
	defer c.kernTimes.Unlock()
	us, n = c.kernTimes.us, c.kernTimes.n
	c.kernTimes.us, c.kernTimes.n = nil, nil
	return us, n
}

// CUDAGraphTimes returns, and clears, the device-clock time of every timed
// replay and the idle gap before it (-1 where there was no previous one), in
// microseconds. See SetCUDATiming.
func (c *Counters) CUDAGraphTimes() (busy, gap []float32) {
	c.graphTimes.Lock()
	defer c.graphTimes.Unlock()
	busy, gap = c.graphTimes.busy, c.graphTimes.gap
	c.graphTimes.busy, c.graphTimes.gap = nil, nil
	return busy, gap
}

// KernelClock is a device that times each kernel on its own clock while armed:
// CUDA's event pairs, Metal's command buffers' own stamps.
type KernelClock interface {
	// SetKernelTiming arms or disarms the clock.
	SetKernelTiming(on bool)
	// KernelTimes returns, and clears, each kernel's summed device
	// microseconds and launches since the last read, by name.
	KernelTimes() (us map[string]float64, n map[string]int)
}

func (c *cudaDev) SetKernelTiming(on bool) { c.ctr.SetCUDAKernelTiming(on) }

func (c *cudaDev) KernelTimes() (us map[string]float64, n map[string]int) {
	return c.ctr.CUDAKernelTimes()
}

func (m *mtlDev) SetKernelTiming(on bool) { m.stats.SetKernelTiming(on) }

func (m *mtlDev) KernelTimes() (us map[string]float64, n map[string]int) {
	return m.stats.KernelTimes()
}

func (c *cudaDev) counters() *Counters { return c.ctr }
func (v *vkDev) counters() *Counters   { return v.ctr }
func (m *mtlDev) counters() *Counters  { return m.ctr }
func (h *hipDev) counters() *Counters  { return h.ctr }
