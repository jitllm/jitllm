package server

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// serveModel is the 1B model jitllmd served when four of them on one NUMA
// node grew until the kernel killed one: Llama-3.2-1B-Instruct at Q4_K_M,
// ~0.8 GB, with a 131072-position context.
const serveModel = "Llama-3.2-1B-Instruct-Q4_K_M.jlm"

// ledger is what a loaded model's server holds, counted by the engine rather
// than by RSS (AGENTS.md RULE 2f: a gate that measures RSS measures the
// allocator).
type ledger struct {
	heap, offHeap, store, storeLimit uint64
}

func (l ledger) String() string {
	mib := func(b uint64) float64 { return float64(b) / (1 << 20) }
	return fmt.Sprintf("live heap %.1f MiB, off-heap weights %.1f MiB, memory cache %.1f MiB (bound %.1f MiB)",
		mib(l.heap), mib(l.offHeap), mib(l.store), mib(l.storeLimit))
}

func (l ledger) total() uint64 { return l.heap + l.offHeap + l.store }

func readLedger(e *Engine, lm *LoadedModel) ledger {
	runtime.GC()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	st := e.MemCacheStats(lm)
	return ledger{heap: ms.HeapAlloc, offHeap: uint64(jlm.OffHeapBytes()), store: st.Store.Bytes,
		storeLimit: st.Store.Limit}
}

// TestAServerUnderLoadStaysUnderItsBound serves the 1B model as jitllmd did on
// the two-socket Xeon -- the default context, the memory cache, the instance pool
// and a warm load, a host share of 190 GiB (eight tenths of that box's free
// memory) -- to sixteen distinct 170-token prompts, four at a time, while the
// host reports 512 MiB available: the NUMA node filling under the other servers.
//
// The memory cache may hold an eighth of what it holds plus what is
// available, so at most 512 MiB/7; the instance's score rows are sized by the keys
// a prompt reaches. The ledger's total stays under the weights plus that
// bound plus 320 MiB of heap.
//
// The violations: an eighth of the share alone lets the cache take every
// prompt (about 6 MB each, the fills of a prompt's last page being held over one
// copy; past the bound by the thirteenth); the score rows sized at
// the context hold 89 MB per Instance.
func TestAServerUnderLoadStaysUnderItsBound(t *testing.T) {
	path := modelPath(t, serveModel)
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	const avail = 512 << 20
	e := New(Config{ModelDir: filepath.Dir(path), Probe: oneCardProbe, Version: "test",
		HostBudget: 190 << 30, MaxBatchRows: 64, WarmLoads: true})
	e.hostAvail = func() uint64 { return avail }
	t.Cleanup(e.Close)
	lm, err := e.LoadModel(LoadOptions{Path: path, ModelID: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if got := e.defaultMaxSeq(lm); got != lm.m.Cfg.NCtx || got < 1<<16 {
		t.Fatalf("a session's default capacity is %d; the gate is about the model's whole context (%d)", got, lm.m.Cfg.NCtx)
	}
	t.Logf("loaded: %v", readLedger(e, lm))

	const workers, each = 4, 4
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < each; i++ {
				p := make([]int32, 170)
				for j := range p {
					p[j] = int32(1000 + (w*7919+i*104729+j*31)%20000)
				}
				generate(t, e, GenerateOptions{ModelID: "m", Prompt: idsPrompt(p), MaxTokens: 8, IgnoreEOS: true})
			}
		}(w)
	}
	wg.Wait()
	l := readLedger(e, lm)
	t.Logf("after %d requests: %v", workers*each, l)

	st := e.MemCacheStats(lm)
	if st.Computed < workers*each*170 || st.Store.Pages == 0 {
		t.Fatalf("the requests did not go through the memory cache (%+v): the gate ran nothing", st)
	}
	if st.Store.Evicted == 0 {
		t.Fatalf("the memory cache evicted nothing at %d MiB: the load never reached its bound", l.store>>20)
	}
	if bound := uint64(avail) / 7; l.store > bound {
		t.Errorf("the memory cache holds %d MiB with %d MiB available: past an eighth of what it holds plus that (%d MiB)",
			l.store>>20, avail>>20, bound>>20)
	}
	const heapBound = 320 << 20
	if l.heap > heapBound {
		t.Errorf("the live heap is %d MiB, past %d MiB", l.heap>>20, heapBound>>20)
	}
	if bound := uint64(fi.Size()) + avail/7 + heapBound; l.total() > bound {
		t.Errorf("the server holds %d MiB, past the weights plus the cache's bound plus the heap's (%d MiB)",
			l.total()>>20, bound>>20)
	}
}
