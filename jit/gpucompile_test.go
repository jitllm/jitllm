package jit_test

import (
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testlock"
	"github.com/jitllm/jitllm/jit"
)

// heldKB is the growth of the resident set over the side-by-side compiles
// that reads as kernels held.
const heldKB = 512 << 10

// residentKB is this process's resident set.
func residentKB() int {
	b, err := os.ReadFile("/proc/self/statm")
	if err != nil {
		return 0
	}
	n, _ := strconv.Atoi(strings.Fields(string(b))[1])
	return n * 4
}

// TestComposedGPUCompilesSideBySide compiles the sampler's candidates (52
// kernels) from eight goroutines at once, round after round for ten
// seconds, beside a collector running every two milliseconds at a GC percent
// of 5. Compile hands the driver structs through uintptr fields, which keep
// nothing alive and name no stack that moves: once the descriptor layout's
// bindings were collected mid-call and once a stack-resident push range was
// read after its stack grew. Before the fix the same loop died in the
// driver (SIGSEGV, 1080 rounds of four goroutines) or held 8 GB and never
// returned (sixteen goroutines); after it, 193 rounds of sixteen and 1043
// of four made no error and held 322 MB. What the rounds hold is the growth
// of the resident set over them: what earlier tests left resident is not
// the compiles'. On an RTX 3050 Ti, 125 rounds grew it 26 MB, and the same
// loop keeping every kernel it compiled grew it 1012 MB in 55.
func TestComposedGPUCompilesSideBySide(t *testing.T) {
	testlock.GPU(t)
	ran := 0
	for _, d := range backend.Open() {
		defer d.Close()
		if d.API() != "spirv" {
			continue
		}
		ran++
		s := graph.SampleTopK{NVocab: 37000, K: 256}
		_, codes, err := jit.Default().Form(jit.Request{Graph: s.Build(), Part: jit.PartSample, Formats: map[jlm.Role]jlm.Type{},
			Step: jit.Step{Rows: 1, Page: 64, Keys: 1}, Device: gpuDevice(d)})
		if err != nil {
			t.Fatal(err)
		}
		old := debug.SetGCPercent(5)
		stop := make(chan struct{})
		go func() {
			for {
				select {
				case <-stop:
					return
				default:
					runtime.GC()
					time.Sleep(2 * time.Millisecond)
				}
			}
		}()
		deadline := time.Now().Add(10 * time.Second)
		rounds, failed := 0, 0
		var mu sync.Mutex
		base := residentKB()
		for time.Now().Before(deadline) && residentKB()-base < heldKB {
			var wg sync.WaitGroup
			for range 8 {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for _, c := range codes {
						k, err := d.Compile(c.GPU)
						if err != nil {
							mu.Lock()
							failed++
							mu.Unlock()
							return
						}
						k.Close()
					}
				}()
			}
			wg.Wait()
			rounds++
		}
		close(stop)
		debug.SetGCPercent(old)
		if failed > 0 {
			t.Errorf("%s: %d compiles failed in %d rounds", d.Name(), failed, rounds)
		}
		if kb := residentKB() - base; kb >= heldKB {
			t.Errorf("%s: %d MB more resident after %d rounds: the compiles hold memory", d.Name(), kb>>10, rounds)
		}
		t.Logf("%s: %d rounds of eight goroutines compiled %d kernels each, %d failed, %d MB more resident (%d MB before)",
			d.Name(), rounds, len(codes), failed, (residentKB()-base)>>10, base>>10)
	}
	if ran == 0 {
		t.Fatal("no SPIR-V device on this host: the gate runs on one")
	}
}
