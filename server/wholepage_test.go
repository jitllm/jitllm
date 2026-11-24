package server

import "testing"

// TestARepeatedPromptOfWholePagesRestoresAllButItsLast: a request of one token
// whose prompt is whole pages, sent again and again, restores every position but the
// last from the memory cache, on the host and through the step loop, and
// answers the same token: a request that restored the whole page leaves it in the cache for the next.
func TestARepeatedPromptOfWholePagesRestoresAllButItsLast(t *testing.T) {
	for _, device := range []bool{false, true} {
		name := "host"
		if device {
			name = "device"
		}
		t.Run(name, func(t *testing.T) {
			cfg := Config{MemCacheBytes: 1 << 30, DefaultMaxSeq: 2048}
			var e *Engine
			var lm *LoadedModel
			id := "m"
			if device {
				e, lm, _ = batchEngine(t, serveModel, cfg, 0, nil)
				id = "dev"
			} else {
				e, lm = ttftEngine(t, serveModel, cfg, LoadOptions{})
			}
			e.hostAvail = func() uint64 { return 64 << 10 }
			if lm.ttft.store == nil {
				t.Fatal("the memory cache is off")
			}
			for _, n := range []int{512, 1024} {
				p := make([]int32, n)
				for j := range p {
					p[j] = int32((300 + j*7 + n) % lm.m.Cfg.NVocab)
				}
				o := GenerateOptions{ModelID: id, Prompt: idsPrompt(p), MaxTokens: 1}
				first := generate(t, e, o)
				if first.started != 0 {
					t.Fatalf("%d ids: the first request restored %d positions", n, first.started)
				}
				for r := 1; r <= 3; r++ {
					again := generate(t, e, o)
					if again.started != n-1 {
						t.Fatalf("%d ids: repeat %d restored %d positions, want %d", n, r, again.started, n-1)
					}
					if len(first.ids) != 1 || len(again.ids) != 1 || first.ids[0] != again.ids[0] {
						t.Fatalf("%d ids: repeat %d answered %v, the first %v", n, r, again.ids, first.ids)
					}
				}
			}
		})
	}
}
