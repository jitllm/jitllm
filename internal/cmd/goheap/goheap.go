// Package goheap is the process-wide collector setup both binaries share:
// cmd/jitllm and server/cmd/jitllmd. It reads the environment, which is why it
// sits under a cmd/ directory, not in a library (AGENTS.md Scope: the
// libraries read none), and it is one copy so the two binaries cannot drift
// apart.
package goheap

import (
	"math"
	"os"
	"runtime/debug"
	"strconv"

	"github.com/jitllm/jitllm/sched"
)

// Cap tells Go's collector about the cgroup, which it cannot see, and returns
// the memory limit it set, or 0 when it set none. Without it GOGC lets the heap
// target twice the live page frames, far past memory.high, and the run spends
// its time in direct reclaim (AGENTS.md RULE 2f).
//
// The headroom covers what the limit charges and the heap does not: goroutine
// stacks, runtime allocations and device-backend mappings. It is set by a
// binary, not by a library, because debug.SetMemoryLimit is process-wide.
func Cap() uint64 {
	lim := sched.MemLimit()
	// JITLLM_GOMEMLIMIT=0 sets no limit, leaving the collector on GOGC alone:
	// the control arm for an off-heap measurement, with the frames on the heap.
	if lim == 0 || os.Getenv("JITLLM_GOMEMLIMIT") == "0" {
		gogc()
		return 0
	}
	// The same constant sched.MemBudget reserves against, so the budget and
	// the goal cannot drift apart.
	if lim <= sched.GCHeadroom {
		return 0
	}
	set := lim - sched.GCHeadroom
	debug.SetMemoryLimit(int64(set))
	// JITLLM_GOGC is the ordinary GOGC, applied after the limit so it
	// composes with it.
	gogc()
	return set
}

// Reserve is JITLLM_GC_RESERVE, the bytes a weight budget on the Go heap
// leaves below the collector's goal (sched.GCPlan.Reserve): lower it to hold
// more weights, raise it if GODEBUG=gctrace=1 shows cycles that free nothing.
// 0, unset, is the derived default.
func Reserve() uint64 {
	n, err := strconv.ParseUint(os.Getenv("JITLLM_GC_RESERVE"), 10, 64)
	if err != nil {
		return 0
	}
	return n
}

// OffHeap re-derives the collector's memory limit for the weight memory off
// the heap: the runtime cannot see those bytes and the cgroup can, so they come
// off the limit Cap set. budgets is the page budget the loaded models' frames
// will grow to -- one model's in cmd/jitllm, every loaded model's summed in
// jitllmd, which calls it on each load and unload -- and mapped what they
// hold off the heap already (dense regions, frames faulted,
// model.Model.MappedBytes), added on top, conservatively.
//
// The limit is a backstop, set only when it leaves the real heap at least
// GCHeadroom: where the frames fill the cgroup, a limit just above the heap is
// the spiral moved rather than removed, so the collector is left on GOGC,
// whose goal is then a multiple of the heap it actually collects. With nothing
// off the heap -- the last model unloaded -- the limit is Cap's again; with
// weights on the heap (JITLLM_OFFHEAP=0) it is left as Cap set it, and
// JITLLM_GOMEMLIMIT=0 sets none here either.
func OffHeap(budgets, mapped uint64) {
	if os.Getenv("JITLLM_GOMEMLIMIT") == "0" {
		return
	}
	if mapped == 0 {
		if budgets == 0 {
			Cap()
		}
		return
	}
	lim := sched.MemLimit()
	off := budgets + mapped
	if lim == 0 || lim < 2*sched.GCHeadroom+off {
		debug.SetMemoryLimit(math.MaxInt64)
		return
	}
	debug.SetMemoryLimit(int64(lim - sched.GCHeadroom - off))
}

// gogc applies JITLLM_GOGC, the ordinary GOGC.
func gogc() {
	if v := os.Getenv("JITLLM_GOGC"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			sched.SetGCPercent(n)
		}
	}
}
