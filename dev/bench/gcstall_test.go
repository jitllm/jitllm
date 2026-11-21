//go:build amd64 && linux

package bench

import (
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/jitllm/jitllm/composer/x86"
	"github.com/jitllm/jitllm/engine/device/cpu"
)

// TestGCNotBlocked is engine/device/cpu's invariant I3, measured -- a timing
// assertion, so it lives here beside the other perf floors rather than in the
// correctness gates (AGENTS.md RULE 4). A goroutine inside generated code
// cannot be async-preempted (runtime.findfunc fails for a PC outside Go's
// text), so a long kernel stalls every runtime.GC(). The defence is a budget:
// kernels are kept short and callers chunk their work.
func TestGCNotBlocked(t *testing.T) {
	if err := Guard(false); err != nil {
		t.Skip(err)
	}
	// Executes VPDPBUSD (see cpu.TestExecuteVPDPBUSD).
	if !cpu.Probe().AVXVNNI {
		t.Skip("this host has no AVX-VNNI, so VPDPBUSD cannot be executed here")
	}
	var a x86.Buf
	a.MOVLoad(x86.RSI, x86.At(x86.RDI, 16))
	a.MOVLoad(x86.RDX, x86.At(x86.RDI, 8))
	a.MOVLoad(x86.RCX, x86.At(x86.RDI, 56))
	a.MOVLoad(x86.RAX, x86.At(x86.RDI, 40))
	a.VPXOR(x86.Y0, x86.Y0, x86.Y0)
	top := a.Label()
	a.Bind(top)
	a.VMOVDQULoad(x86.Y1, x86.At(x86.RSI, 0))
	a.VPDPBUSDMem(x86.Y0, x86.Y1, x86.At(x86.RDX, 0))
	a.DEC(x86.RAX)
	a.JNZ(top)
	a.VMOVDQUStore(x86.At(x86.RCX, 0), x86.Y0)
	a.VZEROUPPER()
	a.RET()

	b, err := a.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	code, err := cpu.New().Map(b)
	if err != nil {
		t.Fatal(err)
	}
	defer code.Close()

	act := make([]int8, 32)
	w := make([]byte, 32)
	scr := make([]byte, 32)

	// Control first: runtime.GC() cost with no kernel running, so the test
	// separates a blocked GC from a slow machine.
	var base time.Duration
	for i := 0; i < 12; i++ {
		start := time.Now()
		runtime.GC()
		if d := time.Since(start); d > base {
			base = d
		}
	}

	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		// ~30us of kernel per call, called repeatedly — the shape a real decode
		// loop has, and the shape the budget exists to enforce.
		args := cpu.Args{A: &act[0], W: &w[0], K: 20000, Scr: &scr[0]}
		for {
			select {
			case <-stop:
				return
			default:
			}
			code.Call(&args)
		}
	}()

	var worst time.Duration
	for i := 0; i < 12; i++ {
		start := time.Now()
		runtime.GC()
		if d := time.Since(start); d > worst {
			worst = d
		}
	}
	close(stop)
	wg.Wait()

	t.Logf("worst runtime.GC(): %v idle, %v alongside a running kernel (+%v)",
		base, worst, worst-base)
	// The kernel may add latency; it may not dominate. A budget-sized kernel
	// should cost far less than the GC already costs.
	if worst > base+10*time.Millisecond {
		t.Errorf("generated code added %v to the worst GC stall (idle %v, running %v); "+
			"kernels are too long — a goroutine in JIT'd code cannot be async-preempted",
			worst-base, base, worst)
	}
}
