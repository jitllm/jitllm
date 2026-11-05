package sched

import "runtime/debug"

// GCHeadroom is what the Go collector is not allowed to plan into, out of the
// cgroup's hard limit. cmd/jitllm hands the runtime `MemLimit() - GCHeadroom`
// and MemBudget reserves against the same figure, so the two cannot drift.
const GCHeadroom = 1 << 30

// GCPlan is where a weight budget's memory lives and what it leaves the Go
// collector. The zero plan is the one that ships: the weights off the heap
// where the platform maps anonymous memory, the derived reserve.
//
// OnHeap says the weights are on the Go heap (jlm.OnHeap, the control arm
// for a measurement), as they are wherever the platform has no anonymous
// mmap; only then does a budget keep a reserve. Reserve is how many bytes of
// the collector's usable heap such a budget leaves free, 0 the derived
// default.
//
// The collector's goal under a memory limit is `limit - non-heap`, and page
// frames are permanently live, so a weight budget close to the limit leaves a
// live set sitting on the goal: the collector runs back to back and frees
// nothing. The cliff is sharp (a few hundred MiB separate 6 cycles from 648),
// so this is a reserve, not a tuning curve. The measurements are in
// docs/engineering-history/scheduling-and-measurement.md (AGENTS.md RULE 2f).
type GCPlan struct {
	OnHeap  bool
	Reserve uint64
}

// onHeap reports whether the plan's weights are on the Go heap.
func (g GCPlan) onHeap() bool { return g.OnHeap || !offHeapPlatform }

// SetGCPercent sets the collector's target growth, the GOGC knob, and returns
// what it was. It is here rather than left to the environment for the reason
// every other knob in this engine is: a library reads no environment, and the
// one caller that knows the workload is the one that should decide.
//
// It is not the fix for the spiral GCPlan.Reserve prevents: under a memory limit
// the goal is `min(live*(1+p/100), limit-nonheap)`, so once live is at the
// limit no value of p moves it.
func SetGCPercent(p int) int { return debug.SetGCPercent(p) }

// GCUsable is the heap the collector is actually given: the cgroup limit less
// GCHeadroom, or 0 when there is no cgroup and the collector is unconstrained.
func GCUsable() uint64 {
	lim := MemLimit()
	if lim <= GCHeadroom {
		return 0
	}
	return lim - GCHeadroom
}

// GCBudgetCap is the zero plan's Cap.
func GCBudgetCap() uint64 { return GCPlan{}.Cap() }

// MemBudget is how many bytes of weights may be resident, when the caller does
// not say: the zero plan's Budget.
func MemBudget() uint64 { return GCPlan{}.Budget() }

// Cap is the largest weight budget that leaves the collector room, or 0 when
// there is no limit to reserve against or the weights are off the heap, where
// the collector's goal never sees them. Every path that sizes the weight
// budget has to pass through it -- Budget does, and so does cmd/jitllm's
// O_DIRECT raise.
//
// The derived reserve is a quarter of the usable heap, capped: a quarter
// lands a 12 GiB cgroup in the quiet zone, and the cap prevents a small margin
// from being wrongly applied to a large server, where the non-heap overhead is the same.
func (g GCPlan) Cap() uint64 {
	if !g.onHeap() {
		return 0
	}
	u := GCUsable()
	if u == 0 {
		return 0
	}
	r := g.Reserve
	if r == 0 {
		r = u / 4
		if r > 8<<30 {
			r = 8 << 30
		}
	}
	if r >= u {
		return 0
	}
	return u - r
}
