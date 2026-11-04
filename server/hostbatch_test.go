package server

import (
	"fmt"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"

	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// The step loop's gates on the host (batch.go, model.StepRuns' host arm):
// concurrent generates on a model with no device decode as rows of shared
// host passes over the weights, through the real handler, and come out as
// each would alone.
//
// They run on a dense 15M, Llama-3.2-1B and Qwen3.5-0.8B (a hybrid: a
// recurrent state per row) by default; JITLLM_SERVER_HOST_BATCH_MODELS names more,
// comma-separated, a bare name looked up in JITLLM_MODELS.

func hostBatchModels() []string {
	out := []string{deviceModel, "Llama-3.2-1B-Instruct-Q4_K_M.jlm", "qwen35/Qwen3.5-0.8B-Q4_K_M.jlm"}
	for _, n := range strings.Split(os.Getenv("JITLLM_SERVER_HOST_BATCH_MODELS"), ",") {
		if n = strings.TrimSpace(n); n != "" && !slices.Contains(out, n) {
			out = append(out, n)
		}
	}
	return out
}

// hostPrompts() are the requests the host gates send, one per row.
func hostPrompts() []string {
	return []string{
		"Once upon a time",
		"The little dog ran to the park and",
		"One day, a girl named Lily found a",
		"Tom and his best friend Sam wanted to",
		"The sun was hot, so the children went to the",
		"In a small house by the sea there lived an old",
		"Ben had a red ball. He threw it",
		"Mia wanted a cake for her birthday, but",
	}
}

// hostBatchEngine loads name on the host alone, and demands the step loop.
func hostBatchEngine(t *testing.T, name string, cfg Config) (*Engine, *LoadedModel, clients) {
	t.Helper()
	path := modelPath(t, name)
	cfg.Probe, cfg.Version = oneCardProbe, "test"
	if cfg.DefaultMaxSeq == 0 {
		cfg.DefaultMaxSeq = 512
	}
	e := New(admitConfig(cfg))
	t.Cleanup(e.Close)
	lm, err := e.LoadModel(LoadOptions{Path: path, ModelID: "dev"})
	if err != nil {
		// A stale container is a task, not a skip.
		t.Fatalf("loading %s on the host: %v", name, err)
	}
	if lm.gpu != nil || lm.loop == nil {
		t.Fatalf("a host model: gpu %v, loop %v", lm.gpu != nil, lm.loop != nil)
	}
	return e, lm, serveEngine(t, e)
}

// concurrently sends reqs at once to a held loop, so they are admitted
// together and share every step from the first.
func concurrently(t *testing.T, e *Engine, lm *LoadedModel, c clients, reqs []*v1.GenerateRequest) []completion {
	t.Helper()
	got := sendTogether(t, e, lm, c, reqs)
	for i, g := range got {
		if g.err != nil {
			t.Fatalf("batched %d: %v", i, g.err)
		}
		if !g.started.GetBatched() {
			t.Fatalf("batched %d did not run as a row", i)
		}
	}
	return got
}

// sendTogether is concurrently without its verdicts: a violation's requests
// may fail.
func sendTogether(t *testing.T, e *Engine, lm *LoadedModel, c clients, reqs []*v1.GenerateRequest) []completion {
	t.Helper()
	release := holdGates(t, e, lm)
	got := make([]completion, len(reqs))
	var wg sync.WaitGroup
	for i, r := range reqs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got[i] = complete(c, r)
		}()
	}
	waitFor(t, "every request to wait for a row", func() bool { return waitingFor(lm.loop) == len(reqs) })
	release()
	wg.Wait()
	return got
}

// parted is the first token at which got and want part by more than a tie on
// the host, or -1 when they agree up to their common length and have it.
func parted(t *testing.T, lm *LoadedModel, prompt string, got, want []int32) int {
	t.Helper()
	p := lm.m.Vocab.Encode(prompt, true)
	for j := range min(len(got), len(want)) {
		if got[j] == want[j] {
			continue
		}
		if hostGap(t, lm.m, p, want[:j], want[j], got[j]) > tieMargin {
			return j
		}
		return -1
	}
	if len(got) != len(want) {
		return min(len(got), len(want))
	}
	return -1
}

// TestHostBatchConcurrentGeneratesEqualAlone: N greedy generates on a host
// model, admitted together, each produce what it produces alone; their rows
// really shared host passes (BatchStats' joint rows against the model's own
// HostStepRows), N sessions a step; and the text of every token was made
// beside its step (the overlap helper's count equals the tokens sent). Then
// the violations: a step that feeds each row the next row's token, and a loop
// that never retires a row at max_tokens, must each part some row from its
// run alone.
func TestHostBatchConcurrentGeneratesEqualAlone(t *testing.T) {
	for _, name := range hostBatchModels() {
		t.Run(name, func(t *testing.T) {
			e, lm, c := hostBatchEngine(t, name, Config{JointSteps: JointAlways})
			const gen = 16
			reqOf := func(i int) *v1.GenerateRequest {
				return &v1.GenerateRequest{ModelId: "dev", Prompt: text(hostPrompts()[i]), MaxTokens: gen}
			}
			alone := make([]completion, len(hostPrompts()))
			for i := range hostPrompts() {
				if alone[i] = complete(c, reqOf(i)); alone[i].err != nil {
					t.Fatalf("alone %d: %v", i, alone[i].err)
				}
			}
			for _, n := range []int{2, 4, 8} {
				t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
					reqs := make([]*v1.GenerateRequest, n)
					for i := range reqs {
						reqs[i] = reqOf(i)
					}
					before, h0 := batchStats(t, c), lm.m.HostStepRows()
					got := concurrently(t, e, lm, c, reqs)
					sent, exact := 0, 0
					for i := range n {
						if j := parted(t, lm, hostPrompts()[i], got[i].ids, alone[i].ids); j >= 0 {
							t.Fatalf("row %d parts from its run alone at token %d\nbatched %v\nalone   %v",
								i, j, got[i].ids, alone[i].ids)
						}
						if slices.Equal(got[i].ids, alone[i].ids) {
							exact++
						}
						sent += len(got[i].ids)
					}
					after := batchStats(t, c)
					rows := lm.m.HostStepRows() - h0
					if rows == 0 || after.GetJointRows()-before.GetJointRows() != rows {
						t.Fatalf("no rows shared a host pass: the model ran %d host step row(s), stats %+v", rows, after)
					}
					if after.GetMaxSessionsPerStep() < int32(n) {
						t.Fatalf("at most %d sessions a step, want %d", after.GetMaxSessionsPerStep(), n)
					}
					if posted := lm.loop.stats.postedTokens.Load(); posted < int64(sent) {
						t.Fatalf("%d token events made beside their step, %d tokens sent", posted, sent)
					}
					t.Logf("%d of %d rows token-identical to alone; %d host step rows, %d joint steps",
						exact, n, rows, after.GetJointSteps()-before.GetJointSteps())
				})
			}

			t.Run("violation-feed-next", func(t *testing.T) {
				lm.loop.feedNext = true
				defer func() { lm.loop.feedNext = false }()
				got := concurrently(t, e, lm, c, []*v1.GenerateRequest{reqOf(0), reqOf(1), reqOf(2)})
				for i := range got {
					if parted(t, lm, hostPrompts()[i], got[i].ids, alone[i].ids) >= 0 {
						t.Logf("row %d parted, as it must", i)
						return
					}
				}
				t.Fatal("rows fed each other's tokens still matched their runs alone: this gate cannot see a row's input")
			})
			// A row never retired runs to the end of its context: on the 15M's
			// 128 positions that is seconds, on the others minutes, for the
			// same verdict, so it runs there alone.
			if name != deviceModel {
				return
			}
			t.Run("violation-drop-retirement", func(t *testing.T) {
				lm.loop.dropMaxTokens = true
				defer func() { lm.loop.dropMaxTokens = false }()
				// A row never retired runs on until its context is full,
				// which ends it with an error: either is the gate firing.
				got := sendTogether(t, e, lm, c, []*v1.GenerateRequest{reqOf(0), reqOf(1)})
				for i := range got {
					if got[i].err != nil {
						t.Logf("row %d failed past max_tokens %d (%v), as it must", i, gen, got[i].err)
						return
					}
					if parted(t, lm, hostPrompts()[i], got[i].ids, alone[i].ids) >= 0 {
						t.Logf("row %d ran %d tokens past max_tokens %d, as it must", i, len(got[i].ids), gen)
						return
					}
				}
				t.Fatal("a loop that never retires at max_tokens still matched: this gate cannot see a retirement")
			})
		})
	}
}

// TestHostBatchSampledRowsEqualAlone: seeded sampled generates on a host
// model, admitted together, draw exactly the tokens they draw admitted in
// the reverse order: a row's draws are its own seed's and its own tokens'
// wherever the joint step puts it. Against rows sharing a random stream, or
// fed each other's tokens, the draws part with the order. A row alone is
// not the reference: it runs the one-row forms and a joint step the rows'
// forms, which fold apart (RULE 8; model.TestBatchMatchesForward bounds them).
func TestHostBatchSampledRowsEqualAlone(t *testing.T) {
	for _, name := range hostBatchModels() {
		t.Run(name, func(t *testing.T) {
			if i, got, want := sampledBatchParts(t, name); i >= 0 {
				t.Fatalf("sampled row %d draws apart with the rows' order\nadmitted first %v\nadmitted last  %v", i, got, want)
			}
		})
	}
}

// sampledBatchParts runs four seeded sampled generates on name admitted
// together, then admitted together in the reverse order, and returns the
// first row whose draws differ between the two, with both sequences, or -1
// when every row drew the same tokens. The memory cache is off: each run's
// prompts are its own.
func sampledBatchParts(t *testing.T, name string) (int, []int32, []int32) {
	t.Helper()
	e, lm, c := hostBatchEngine(t, name, Config{JointSteps: JointAlways, NoMemCache: true})
	temp, topP := float32(0.9), float32(0.95)
	reqOf := func(i int) *v1.GenerateRequest {
		seed := uint64(7 + i)
		return &v1.GenerateRequest{ModelId: "dev", Prompt: text(hostPrompts()[i]), MaxTokens: 16,
			Sampling: &v1.SamplingParams{Temperature: &temp, TopP: &topP, Seed: &seed}}
	}
	const n = 4
	reqs := make([]*v1.GenerateRequest, n)
	for i := range n {
		reqs[i] = reqOf(i)
	}
	got := concurrently(t, e, lm, c, reqs)
	rev := slices.Clone(reqs)
	slices.Reverse(rev)
	back := concurrently(t, e, lm, c, rev)
	for i := range n {
		if !slices.Equal(got[i].ids, back[n-1-i].ids) {
			return i, got[i].ids, back[n-1-i].ids
		}
	}
	return -1, nil, nil
}

// TestHostBatchLongPromptTakesItsPipelinedChunk: a long prompt admitted while
// two rows decode, on an instance whose prefill pipelines (here told so: chunkOf
// reports a pipelined chunk wider than the step's budget), runs as pipelined
// chunks beside the joint step rather than cut at the budget -- counted, one
// chunk for a prompt the budget cuts into several -- and every row still
// comes out as it does alone.
func TestHostBatchLongPromptTakesItsPipelinedChunk(t *testing.T) {
	e, lm, c := hostBatchEngine(t, "Llama-3.2-1B-Instruct-Q4_K_M.jlm",
		Config{JointSteps: JointAlways, PromptChunk: 64, DefaultMaxSeq: 1024,
			// Each run sends the same prompts: the memory cache would restore them.
			NoMemCache: true})
	long := strings.Repeat("The little dog ran to the park and played with a red ball. ", 12)
	nlong := len(lm.m.Vocab.Encode(long, true))
	reqs := []*v1.GenerateRequest{
		{ModelId: "dev", Prompt: text(hostPrompts()[0]), MaxTokens: 24},
		{ModelId: "dev", Prompt: text(hostPrompts()[1]), MaxTokens: 24},
		{ModelId: "dev", Prompt: text(long), MaxTokens: 8},
	}
	alone := make([]completion, len(reqs))
	for i, r := range reqs {
		if alone[i] = complete(c, r); alone[i].err != nil {
			t.Fatal(alone[i].err)
		}
	}
	run := func(pipe int) ([]completion, int64, int64) {
		lm.loop.chunkOf = func(*row) int { return pipe }
		p0, c0 := lm.loop.stats.pipelinedChunks.Load(), lm.loop.stats.promptChunks.Load()
		got := concurrently(t, e, lm, c, reqs)
		return got, lm.loop.stats.pipelinedChunks.Load() - p0, lm.loop.stats.promptChunks.Load() - c0
	}
	cut, cutPiped, cutChunks := run(64)
	piped, pipedN, pipedChunks := run(4096)
	t.Logf("a %d-token prompt beside two decoding rows: budget-cut %d chunks (%d pipelined), pipelined %d chunks (%d pipelined)",
		nlong, cutChunks, cutPiped, pipedChunks, pipedN)
	if cutPiped != 0 || pipedN == 0 || pipedChunks >= cutChunks {
		t.Fatalf("the long prompt did not take its pipelined chunk: %d pipelined of %d chunks, against %d cut",
			pipedN, pipedChunks, cutChunks)
	}
	prompts := []string{hostPrompts()[0], hostPrompts()[1], long}
	for i := range reqs {
		for _, g := range [][]completion{cut, piped} {
			if j := parted(t, lm, prompts[i], g[i].ids, alone[i].ids); j >= 0 {
				t.Fatalf("row %d parts from alone at token %d\nbatched %v\nalone   %v", i, j, g[i].ids, alone[i].ids)
			}
		}
	}
}
