package server

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jitllm/jitllm/internal/testdevices"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// TestBatchKeepsBatchingRoundAfterRound: a model loaded as jitllmd loads it
// (-sessions and -max-batch at the load's concurrency, the prompt cache off)
// serves conc clients, each sending its next request as its last ends, round
// after round, and every round batches as the first did: its mean rows a step
// is at least half the first round's. A history the finished requests left
// on the card, or a refusal that never re-arms, admits one row at a time from
// the second round on (Qwen3-30B-A3B over two V100s at 64 requests: 86 tok/s
// in the first round, then 35 with every request alone).
//
// JITLLM_SERVER_CURVE is conc,reqs,prompt,gen,rounds (default
// 64,2,512,128,3, the prompt cut to the model's context);
// JITLLM_TEST_DEVICES names the devices, every one of them used.
func TestBatchKeepsBatchingRoundAfterRound(t *testing.T) {
	conc, reqs, plen, gen, rounds := 64, 2, 512, 128, 3
	if v := os.Getenv("JITLLM_SERVER_CURVE"); v != "" {
		if _, err := fmt.Sscanf(v, "%d,%d,%d,%d,%d", &conc, &reqs, &plen, &gen, &rounds); err != nil {
			t.Fatalf("JITLLM_SERVER_CURVE=%q: want conc,reqs,prompt,gen,rounds: %v", v, err)
		}
	}
	devs := testdevices.Take(t, 2, "cuda:0,vulkan:1")
	eachBatchModel(t, func(t *testing.T, name string) {
		e := New(Config{Version: "test", NoMemCache: true, DefaultMaxSeq: 4608, MaxBatchRows: conc, NoOrchestrate: true})
		t.Cleanup(e.Close)
		lm, err := e.LoadModel(LoadOptions{Path: modelPath(t, name), ModelID: "dev", DeviceIDs: devs, Sessions: conc})
		if err != nil {
			t.Fatalf("loading %s onto %s: %v", name, strings.Join(devs, ","), err)
		}
		if lm.loop == nil {
			t.Fatal("a model loaded onto devices has no step loop")
		}
		c := serveEngine(t, e)
		cfg := lm.m.Cfg
		plen := min(plen, int(cfg.NCtx)-gen-1)
		vocab := uint32(min(cfg.NVocab-1, 20000))
		prompt := func(i int) []int32 {
			p := make([]int32, plen)
			x := uint32(i)*2654435761 + 1
			for j := range p {
				x ^= x << 13
				x ^= x >> 17
				x ^= x << 5
				p[j] = 1 + int32(x%vocab)
			}
			return p
		}
		lp := lm.loop
		first := 0.0
		for r := range rounds {
			st := &lp.stats
			steps0, rows0, waits0, adm0 := st.steps.Load(), st.rowsStepped.Load(), st.kvWaits.Load(), st.admissions.Load()
			t0 := time.Now()
			errs := make([]error, conc)
			var wg sync.WaitGroup
			for k := range conc {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for w := range reqs {
						o := complete(c, &v1.GenerateRequest{ModelId: "dev", Prompt: ids(prompt(r*conc*reqs + k*reqs + w)...),
							MaxTokens: int32(gen), IgnoreEos: true})
						if o.err == nil && len(o.ids) != gen {
							o.err = fmt.Errorf("%d tokens, want %d", len(o.ids), gen)
						}
						if o.err != nil {
							errs[k] = o.err
							return
						}
					}
				}()
			}
			wg.Wait()
			for k, err := range errs {
				if err != nil {
					t.Fatalf("round %d, client %d: %v", r, k, err)
				}
			}
			d := time.Since(t0)
			steps, rows := st.steps.Load()-steps0, st.rowsStepped.Load()-rows0
			mean := float64(rows) / float64(max(steps, 1))
			room, page := lm.gpuTier().KVRoom()
			s := lm.gpuTier().Stats()
			t.Logf("round %d: %s, %.0f tok/s; %d steps, %.1f rows a step, %d admissions, %d waits for history room; room %d positions (page %d), KV held %d MiB of a pool of %d MiB, budget used %d MiB",
				r, d.Round(time.Millisecond), float64(conc*reqs*gen)/d.Seconds(), steps, mean, st.admissions.Load()-adm0,
				st.kvWaits.Load()-waits0, room, page, s.KVBytes>>20, s.KVPoolBytes>>20, s.BudgetUsed>>20)
			if r == 0 {
				first = mean
				continue
			}
			if mean < first/2 {
				t.Errorf("round %d steps %.1f rows a step where the first stepped %.1f: the loop stopped batching", r, mean, first)
			}
		}
	})
}
