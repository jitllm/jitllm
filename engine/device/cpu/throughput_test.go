//go:build amd64 && linux

package cpu

import (
	"testing"
	"time"

	"github.com/jitllm/jitllm/dev/bench"
)

// TestGeneratedThroughput measures what the emitter actually produces, streaming
// a DRAM-resident weight buffer through VPDPBUSD.
//
// If generated code cannot get close to the memory wall on a pure streaming
// loop, no kernel built on it will either.
//
// Not a gate: perf floors do not live in _test.go. It logs.
func TestGeneratedThroughput(t *testing.T) {
	if testing.Short() {
		t.Skip("allocates 512 MB")
	}
	// 512 MB: far past the last-level cache, so this measures DRAM and not cache.
	const bytes = 512 << 20
	const blocksPerCall = 8192 // 8192*32 = 256 KB per call, ~L2 sized, budget-safe
	w := make([]byte, bytes)
	act := make([]int8, 32)
	scr := make([]byte, 32)
	for i := range w {
		w[i] = byte(i)
	}

	// Four independent accumulators: one dependent chain would measure the
	// 5-cycle VPDPBUSD latency instead of memory.
	var a Buf
	a.MOVLoad(RSI, At(RDI, 16)) // A
	a.MOVLoad(RDX, At(RDI, 8))  // W
	a.MOVLoad(RCX, At(RDI, 56)) // Scr
	a.MOVLoad(RAX, At(RDI, 40)) // K = iterations
	a.VPXOR(Y0, Y0, Y0)
	a.VPXOR(Y2, Y2, Y2)
	a.VPXOR(Y3, Y3, Y3)
	a.VPXOR(Y4, Y4, Y4)
	a.VMOVDQULoad(Y1, At(RSI, 0))
	top := a.Label()
	a.Bind(top)
	a.VPDPBUSDMem(Y0, Y1, At(RDX, 0))
	a.VPDPBUSDMem(Y2, Y1, At(RDX, 32))
	a.VPDPBUSDMem(Y3, Y1, At(RDX, 64))
	a.VPDPBUSDMem(Y4, Y1, At(RDX, 96))
	a.ADDimm(RDX, 128)
	a.DEC(RAX)
	a.JNZ(top)
	a.VPADDD(Y0, Y0, Y2)
	a.VPADDD(Y3, Y3, Y4)
	a.VPADDD(Y0, Y0, Y3)
	a.VMOVDQUStore(At(RCX, 0), Y0)
	a.VZEROUPPER()
	a.RET()

	if HostTier() == TierSSE {
		avx2Primitive(t, a.Bytes(), "the packed family's SSE throughput in dev/bench (this is a VPDPBUSD streaming probe)")
		return
	}
	code := mustMap(t, a.Bytes())
	defer code.Close()
	t.Logf("kernel is %d bytes", code.Size)

	calls := bytes / (blocksPerCall * 32)
	run := func() time.Duration {
		start := time.Now()
		for c := 0; c < calls; c++ {
			args := Args{
				A: &act[0], W: &w[c*blocksPerCall*32], Scr: &scr[0],
				K: int64(blocksPerCall * 32 / 128),
			}
			code.Call(&args)
		}
		return time.Since(start)
	}
	run() // warm the page cache and the branch predictors
	best := time.Hour
	for i := 0; i < 5; i++ {
		if d := run(); d < best {
			best = d
		}
	}
	t.Log(bench.Report("generated VPDPBUSD stream, 1 core", uint64(bytes), best, 1))

	// The per-call cost is what invariant I3 budgets.
	t.Logf("per call: %v over %d calls (%d KB of weights each)",
		(best / time.Duration(calls)).Round(time.Microsecond), calls, blocksPerCall*32/1024)
}
