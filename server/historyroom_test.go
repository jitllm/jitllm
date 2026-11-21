package server

import (
	"fmt"
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// TestLongPromptHasHistoryRoom: a model loaded onto one device as jitllmd
// loads it (four sessions and rows, a context of 4608) answers a prompt of
// 2051 tokens and then a second one beside it: the device finds a page of
// history for every position of both, by its pool, the room it reserved or
// a block paged out, rather than refusing for want of one.
//
// JITLLM_SERVER_ROOM_LEN overrides the prompt's length.
func TestLongPromptHasHistoryRoom(t *testing.T) {
	devs := testdevices.Take(t, 1, "cuda:0")
	n := 2051
	if v := os.Getenv("JITLLM_SERVER_ROOM_LEN"); v != "" {
		if _, err := fmt.Sscanf(v, "%d", &n); err != nil {
			t.Fatalf("JITLLM_SERVER_ROOM_LEN=%q: %v", v, err)
		}
	}
	eachBatchModel(t, func(t *testing.T, name string) {
		e := New(admitConfig(Config{Version: "test", NoMemCache: true, DefaultMaxSeq: 4608, JointSteps: JointAlways,
			MaxBatchRows: 4, NoOrchestrate: true}))
		t.Cleanup(e.Close)
		lm, err := e.LoadModel(LoadOptions{Path: modelPath(t, name), ModelID: "dev", DeviceIDs: devs, Sessions: 4})
		if err != nil {
			t.Fatal(err)
		}
		c := serveEngine(t, e)
		cfg := lm.m.Cfg
		n := min(n, int(cfg.NCtx)-9)
		vocab := uint32(min(cfg.NVocab-1, 20000))
		prompt := func(seed uint32) []int32 {
			p := make([]int32, n)
			x := seed*2654435761 + 1
			for j := range p {
				x ^= x << 13
				x ^= x >> 17
				x ^= x << 5
				p[j] = 1 + int32(x%vocab)
			}
			return p
		}
		report := func(what string) {
			s := lm.gpuTier().Stats()
			t.Logf("%s: budget used %d MiB, scratch %d MiB, composed instances %d MiB, KV held %d MiB of a pool of %d MiB, reserved prompt %d rows and step %d, KV grows %d, evictions %d",
				what, s.BudgetUsed>>20, s.ScratchBytes>>20, s.ComposedBytes>>20, s.KVBytes>>20, s.KVPoolBytes>>20, s.ReservedPrompt, s.ReservedStep,
				s.KVGrows, s.KVEvictions)
		}
		report("loaded")
		one := func(seed uint32) completion {
			return complete(c, &v1.GenerateRequest{ModelId: "dev", Prompt: ids(prompt(seed)...), MaxTokens: 8, IgnoreEos: true})
		}
		if o := one(1); o.err != nil {
			report("refused")
			t.Fatalf("a prompt of %d tokens: %v", n, o.err)
		}
		report("one prompt")
		out := make([]completion, 2)
		done := make(chan int)
		for i := range out {
			go func() {
				out[i] = one(uint32(2 + i))
				done <- i
			}()
		}
		<-done
		<-done
		for i, o := range out {
			if o.err != nil {
				report("refused")
				t.Fatalf("two prompts of %d tokens at once, request %d: %v", n, i, o.err)
			}
		}
		report("two prompts")
	})
}
