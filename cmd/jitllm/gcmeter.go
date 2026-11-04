package main

import (
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"runtime/metrics"
	"runtime/pprof"
)

// gcSummary is JITLLM_GCSTATS=1's whole-process line: what the collector cost
// against the process's CPU, its pauses, and where memory peaked.
func gcSummary() {
	g := readGC()
	s := []metrics.Sample{
		{Name: "/cpu/classes/total:cpu-seconds"},
		{Name: "/gc/heap/live:bytes"},
	}
	metrics.Read(s)
	// Process CPU from the kernel: the runtime/metrics CPU classes are only
	// brought up to date at the end of a cycle.
	total, maxRSS := processUsage()
	share := 0.0
	if total > 0 {
		share = 100 * g.gcCPU / total
	}
	fmt.Fprintf(os.Stderr, "gcstats: %d cycle(s), gc cpu %.2f s of %.2f s process cpu (%.1f%%, assist %.2f s), %d pause(s) %.2f ms, "+
		"heap live %.2f GiB goal %.2f GiB limit %.2f GiB, maxrss %.2f GiB\n",
		g.cycles, g.gcCPU, total, share, g.assistCPU, g.pauses, g.pause*1e3,
		float64(s[1].Value.Uint64())/(1<<30), float64(g.heapGoal)/(1<<30),
		float64(debug.SetMemoryLimit(-1))/(1<<30),
		float64(maxRSS)/(1<<30))
}

// gcSample is the collector's counters at one instant. Every field is
// cumulative, so a phase is the difference of two samples.
type gcSample struct {
	allocBytes, allocObjects uint64
	cycles                   uint64
	gcCPU, assistCPU         float64
	pause                    float64
	pauses                   uint64
	heapGoal                 uint64
}

// gcMeterNames are the runtime/metrics a gcSample reads, in field order.
func gcMeterNames() []string {
	return []string{
		"/gc/heap/allocs:bytes",
		"/gc/heap/allocs:objects",
		"/gc/cycles/total:gc-cycles",
		"/cpu/classes/gc/total:cpu-seconds",
		"/cpu/classes/gc/mark/assist:cpu-seconds",
		"/sched/pauses/total/gc:seconds",
		"/gc/heap/goal:bytes",
	}
}

// readGC samples the collector. The pause histogram is reduced to a count and
// a sum of bucket midpoints, which is as close as the histogram allows.
func readGC() gcSample {
	names := gcMeterNames()
	s := make([]metrics.Sample, len(names))
	for i, n := range names {
		s[i].Name = n
	}
	metrics.Read(s)
	var g gcSample
	g.allocBytes = s[0].Value.Uint64()
	g.allocObjects = s[1].Value.Uint64()
	g.cycles = s[2].Value.Uint64()
	g.gcCPU = s[3].Value.Float64()
	g.assistCPU = s[4].Value.Float64()
	h := s[5].Value.Float64Histogram()
	for i, c := range h.Counts {
		if c == 0 {
			continue
		}
		lo, hi := h.Buckets[i], h.Buckets[i+1]
		if lo < 0 {
			lo = 0
		}
		if hi > 1 {
			hi = lo
		}
		g.pauses += c
		g.pause += float64(c) * (lo + hi) / 2
	}
	g.heapGoal = s[6].Value.Uint64()
	return g
}

// gcDelta prints what the collector did between two samples, per unit of work.
func gcDelta(phase string, a, b gcSample, units int) {
	u := float64(units)
	fmt.Printf("gc %s: %.1f allocs/tok %.0f B/tok, %d cycle(s), %d pause(s) %.3f ms, gc cpu %.3f s (assist %.3f s), goal %.2f GiB\n",
		phase, float64(b.allocObjects-a.allocObjects)/u, float64(b.allocBytes-a.allocBytes)/u,
		b.cycles-a.cycles, b.pauses-a.pauses, (b.pause-a.pause)*1e3,
		b.gcCPU-a.gcCPU, b.assistCPU-a.assistCPU, float64(b.heapGoal)/(1<<30))
}

// writeAllocProfile publishes every allocation so far (the profile only
// reflects completed cycles) and writes it to path.
func writeAllocProfile(path string) error {
	runtime.GC()
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return pprof.Lookup("allocs").WriteTo(f, 0)
}
