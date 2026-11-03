package metal

import (
	"sync"
	"sync/atomic"
)

// Stats is what Metal contexts count: submissions, waits, the device's own
// busy and idle time, and per-kernel device time when it is armed. Contexts
// opened with the same Stats (Opts.Stats) count together; one opened without
// counts into its own.
type Stats struct {
	// statCommit, statWait and statBlock count commits, waits and waits that
	// blocked.
	statCommit, statWait, statBlock atomic.Uint64

	// statEncodeNs and statWaitNs split the CPU's time between building the
	// command buffer and blocking on the GPU.
	statEncodeNs, statWaitNs, statLaunch atomic.Uint64

	// statSolo counts dispatches that went through Kernel.Launch -- one command
	// buffer, committed AND WAITED, for a single dispatch. Inside a Session they
	// go through Batch.Encode instead and share one buffer.
	statSolo, statSoloNs atomic.Uint64

	// statSessionNs is wall time inside backend Session. It bounds submission
	// from above.
	statSessionNs, statSessionN atomic.Uint64

	// statNewNs is time creating the command buffer and its encoder;
	// statCommitNs is endEncoding plus commit.
	statNewNs, statCommitNs atomic.Uint64

	// statGPUNs is the device's own execution time, summed from every
	// completed command buffer's GPUStartTime/GPUEndTime.
	statGPUNs, statGPUN, statGPULast atomic.Uint64

	// statGPUGapNs sums the device's idle time between consecutive command
	// buffers (GPUStartTime of one less GPUEndTime of the one before), on the
	// device's own clock; each queue keeps its previous end (sub.prevEnd).
	statGPUGapNs, statGPUGapN atomic.Uint64

	// statHostGapNs is host time from a blocking Wait's return to the next
	// Commit: the CPU's share of the device's idle gap.
	statHostGapNs, statHostGapN atomic.Uint64
	lastWaitDone                atomic.Int64

	// statLatNs is commit-to-Wait-return host time less GPU busy time, for a
	// Wait covering one command buffer: the launch and wake-up latency of a
	// round trip.
	statLatNs, statLatN atomic.Uint64
	lastCommit          atomic.Int64

	// statBarrier counts the buffer barriers concurrent encoders put in.
	statBarrier atomic.Uint64

	// kernTiming puts every dispatch in a command buffer of its own, whose
	// GPUStartTime/GPUEndTime then time that kernel alone; see
	// SetKernelTiming.
	kernTiming atomic.Bool
	// timedCBs names the command buffers committed under SetKernelTiming
	// until a wait reads them.
	timedCBs struct {
		sync.Mutex
		name map[uintptr]string
		us   map[string]float64
		n    map[string]int
	}
}

// RoundTrip reports the launch plus wake-up latency summed over n round trips.
func (s *Stats) RoundTrip() (ns, n uint64) { return s.statLatNs.Load(), s.statLatN.Load() }

// HostGap reports the host's nanoseconds from a Wait's return to the next
// commit, and over how many.
func (s *Stats) HostGap() (ns, n uint64) { return s.statHostGapNs.Load(), s.statHostGapN.Load() }

// GPUIdle reports the device's idle nanoseconds between consecutive command
// buffers, and how many gaps that sums.
func (s *Stats) GPUIdle() (ns, gaps uint64) { return s.statGPUGapNs.Load(), s.statGPUGapN.Load() }

// GPUBusy reports nanoseconds the device spent executing, and over how many
// command buffers. Against a token's wall time it splits the token without a
// fitted parameter.
func (s *Stats) GPUBusy() (ns, buffers uint64) { return s.statGPUNs.Load(), s.statGPUN.Load() }

// GPUBusyLast is the LAST completed command buffer's own execution time. Under
// one-submission-per-block it is the head sequence's, measured where it runs.
func (s *Stats) GPUBusyLast() uint64 { return s.statGPULast.Load() }

// SubmitStats reports commits, Wait calls, and Waits that actually blocked.
func (s *Stats) SubmitStats() (commits, waits, blocked uint64) {
	return s.statCommit.Load(), s.statWait.Load(), s.statBlock.Load()
}

// SubmitTimes reports nanoseconds spent encoding and nanoseconds spent blocked,
// plus the number of dispatches encoded.
func (s *Stats) SubmitTimes() (encodeNs, waitNs, launches uint64) {
	return s.statEncodeNs.Load(), s.statWaitNs.Load(), s.statLaunch.Load()
}

// SoloStats reports dispatches that took a command buffer of their own, and the
// nanoseconds they cost. Each is a full GPU round trip.
func (s *Stats) SoloStats() (n, ns uint64) { return s.statSolo.Load(), s.statSoloNs.Load() }

// AddSessionNs records one backend Session's wall time.
func (s *Stats) AddSessionNs(ns uint64) { s.statSessionNs.Add(ns); s.statSessionN.Add(1) }

// SessionStats reports sessions and their total wall nanoseconds.
func (s *Stats) SessionStats() (n, ns uint64) { return s.statSessionN.Load(), s.statSessionNs.Load() }

// PhaseStats splits a Session: creating the command buffer, and committing it.
func (s *Stats) PhaseStats() (newNs, commitNs uint64) {
	return s.statNewNs.Load(), s.statCommitNs.Load()
}

// Barriers is how many buffer barriers concurrent encoders have put between
// dispatches; against SubmitTimes' dispatch count it says how much of a token
// runs side by side.
func (s *Stats) Barriers() uint64 { return s.statBarrier.Load() }

// SetKernelTiming arms per-kernel device timing on Metal: each dispatch is
// committed in its own command buffer, and the device's own start and end
// stamps of it are read when the work is waited for (KernelTimes). A
// measurement instrument, default off: a command buffer a dispatch costs its
// own launch and loses the concurrency between dispatches, so the split it
// gives is each kernel's time alone, not the token's.
func (s *Stats) SetKernelTiming(on bool) { s.kernTiming.Store(on) }

// KernelTimes returns, and clears, the per-kernel device microseconds and
// launch counts SetKernelTiming recorded, for the work waited for so far.
func (s *Stats) KernelTimes() (us map[string]float64, n map[string]int) {
	s.timedCBs.Lock()
	defer s.timedCBs.Unlock()
	us, n = s.timedCBs.us, s.timedCBs.n
	s.timedCBs.us, s.timedCBs.n = nil, nil
	return us, n
}

// timedDone adds a completed command buffer's device time to its kernel's,
// when it was a timed one.
func (s *Stats) timedDone(cb uintptr, t0, t1 float64) {
	s.timedCBs.Lock()
	defer s.timedCBs.Unlock()
	name, ok := s.timedCBs.name[cb]
	if !ok {
		return
	}
	delete(s.timedCBs.name, cb)
	if s.timedCBs.us == nil {
		s.timedCBs.us, s.timedCBs.n = map[string]float64{}, map[string]int{}
	}
	s.timedCBs.us[name] += (t1 - t0) * 1e6
	s.timedCBs.n[name]++
}
