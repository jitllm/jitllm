package composer_test

import (
	"runtime"
	"testing"
	"time"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestCoarsenLeavesALongChainUnformed forms the sampler's candidates over
// vocabularies whose reductions Coarsen once unrolled into millions of nodes
// (65536: 7,976,810 nodes, 3 s; 131072: no end in five minutes and more than
// 3 GB) on a device that states no exact division: the chain past 1<<20
// nodes is left as its members. The unrolled chain never reaches a final
// kernel, so the gate counts what forming allocates (1.7 and 2.0 GB with the
// bound, 8.8 GB at 65536 and no end at 131072 without it) and every kernel's
// instructions, not the time, which follows the host (0.7 s on the laptop,
// 2.2 s on the V100's).
func TestCoarsenLeavesALongChainUnformed(t *testing.T) {
	dev := composer.Device{Kind: composer.DeviceGPU, Slots: 30720, Lanes: 32, Units: 20, Regs: 42}
	for _, v := range []int{65536, 131072} {
		s := graph.SampleTopK{NVocab: v, K: 256}
		g := s.Build()
		type formed struct {
			codes []composer.Code
			err   error
		}
		done := make(chan formed, 1)
		var before, after runtime.MemStats
		go func() {
			runtime.ReadMemStats(&before)
			_, codes, err := composer.Default().Form(composer.Request{Graph: g, Part: composer.PartSample, Formats: map[jlm.Role]jlm.Type{},
				Step: composer.Step{Rows: 1, Page: 64, Keys: 1}, Device: dev})
			runtime.ReadMemStats(&after)
			done <- formed{codes, err}
		}()
		select {
		case f := <-done:
			if f.err != nil {
				t.Fatalf("vocabulary %d: %v", v, f.err)
			}
			most := 0
			for _, c := range f.codes {
				if c.GPU == nil {
					t.Fatalf("vocabulary %d: %s has no GPU code", v, c.Name)
				}
				most = max(most, len(c.GPU.Ops))
			}
			mb := (after.TotalAlloc - before.TotalAlloc) >> 20
			if most >= 1<<20 || mb >= 4<<10 {
				t.Errorf("vocabulary %d: a kernel of %d instructions, %d MB allocated forming them", v, most, mb)
			}
			t.Logf("vocabulary %d: %d kernels, the longest %d instructions, %d MB allocated forming them", v, len(f.codes), most, mb)
		case <-time.After(30 * time.Second):
			t.Fatalf("vocabulary %d: still forming after 30 s", v)
		}
	}
}
