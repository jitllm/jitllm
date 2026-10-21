package server

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// The step loop's gates (batch.go): concurrent generates on one device model
// decode as rows of shared steps, through the real handler, and come out as
// each would alone.
//
// They run on deviceModel by default -- every block on a card, milliseconds a
// token. JITLLM_SERVER_BATCH_MODELS names more, comma-separated (a bare name
// is looked up in JITLLM_MODELS), so a dense 1B and a hybrid run the same
// gates:
//
//	JITLLM_SERVER_BATCH_MODELS=Llama-3.2-1B-Instruct-Q4_K_M.jlm,qwen35/Qwen3.5-0.8B-Q4_K_M.jlm \
//	  ../scripts/cap 8G -- go test . -run Batch -count=1 -v

// batchModels is every container the batch gates run on.
func batchModels() []string {
	out := []string{deviceModel}
	for _, n := range strings.Split(os.Getenv("JITLLM_SERVER_BATCH_MODELS"), ",") {
		if n = strings.TrimSpace(n); n != "" && n != deviceModel {
			out = append(out, n)
		}
	}
	return out
}

// eachBatchModel runs fn as a subtest per batch model.
func eachBatchModel(t *testing.T, fn func(t *testing.T, name string)) {
	for _, name := range batchModels() {
		t.Run(name, func(t *testing.T) { fn(t, name) })
	}
}

// tieMargin is how far apart two tokens' logits may be, on the host, for the
// first place a batched row and the same request alone part to be a tie
// rather than a wrong row. It is engine/model's own bound: a joint step
// differs from decode alone by up to 0.53-0.70 of a logit on the device
// (schedulerdev_test.go has the measurement), and a row reading another row's
// history diverges by whole logits.
const tieMargin = 0.5

// batchEngine loads name onto the first GPU with a history reserved for
// sessions sessions, or skips naming why. The loop must exist: a device model
// that did not get one would run every gate below on the one-at-a-time path
// and compare it with itself.
func batchEngine(t *testing.T, name string, cfg Config, sessions int, tc func(*tier.Config)) (*Engine, *LoadedModel, clients) {
	t.Helper()
	path := modelPath(t, name)
	cfg.Probe, cfg.Version = oneCardProbe, "test"
	if cfg.DefaultMaxSeq == 0 {
		// A session asks for the model's whole context otherwise, and on a
		// small card the history of a few such sessions sends blocks home,
		// which takes them out of the joint step.
		cfg.DefaultMaxSeq = 512
	}
	e := New(cfg)
	t.Cleanup(e.Close)
	lm, err := e.LoadModel(LoadOptions{Path: path, ModelID: "dev", DeviceIDs: []string{"gpu:0"},
		Sessions: sessions, tierConfig: tc})
	if err != nil {
		if name != deviceModel {
			// Named on purpose: a stale container is a task, not a skip.
			t.Fatalf("loading %s onto -devices gpu:0: %v", name, err)
		}
		t.Skipf("NO DEVICE: loading %s onto -devices gpu:0 failed (%v) -- this gate proved nothing", name, err)
	}
	if lm.loop == nil {
		t.Fatal("a model loaded onto a device has no step loop")
	}
	requireWholeOnDevice(t, e, lm)
	return e, lm, serveEngine(t, e)
}

// requireWholeOnDevice fails, naming the cause, when a fresh session of lm
// cannot take a row of the step loop -- which every gate that asserts a
// generate ran as a row presumes.
//
// The tier sizes what it places from the card's free memory, and the CUDA
// figure is machine-wide (backend.cudaDev.Mem): another process holding the
// card -- another test binary, another session's jitllmd -- leaves the model
// split between the card and the host, and a split session is stepped alone
// (State.StepRefusal), correctly. Those gates then failed far from the cause,
// as "did not run as a row", and looked like an order dependence between
// tests in this package: they failed after the TestBatch* gates and passed
// alone only because the card happened to be freer on the second run. Nothing
// in this process holds the card after an Engine closes (measured: the
// process's own figure in nvidia-smi is gone after Close). So the
// precondition is checked here and its failure says what it is; it is not a
// skip, because a gate that did not run is not a pass (RULE 10).
func requireWholeOnDevice(t *testing.T, e *Engine, lm *LoadedModel) {
	t.Helper()
	s, err := e.CreateSession(SessionOptions{ModelID: lm.id})
	if err != nil {
		t.Fatal(err)
	}
	why, placed := s.st.StepRefusal(), s.st.GPULayers()
	if err := e.CloseSession(s.id); err != nil {
		t.Fatal(err)
	}
	if why != nil {
		t.Fatalf("PRECONDITION: %s places %d of %d blocks on the card (%v), so no generate on it can run as a "+
			"row of the step loop and this gate cannot run. The tier sizes placement from the card's "+
			"machine-wide free memory: another process is holding the card (nvidia-smi names it).",
			lm.id, placed, lm.m.Cfg.NLayer, why)
	}
}

// prompts are the requests the gates send, one per row.
var prompts = []string{
	"Once upon a time",
	"The little dog ran to the park and",
	"One day, a girl named Lily found a",
	"Tom and his best friend Sam wanted to",
}

// holdGates holds the model's step loop before it serves, so requests pile
// up waiting for a row and the loop admits them together once released: the
// rows then share every step from the first, whatever the goroutines'
// scheduling.
func holdGates(t *testing.T, e *Engine, lm *LoadedModel) func() {
	t.Helper()
	lm.loop.hold.Lock()
	var once sync.Once
	release := func() { once.Do(lm.loop.hold.Unlock) }
	t.Cleanup(release)
	return release
}

func waitingFor(lp *stepLoop) int {
	_, w := lp.snapshot()
	return len(w)
}

// completion is one request's answer as the Connect client saw it.
type completion struct {
	started  *v1.GenerateStarted
	finished *v1.GenerateFinished
	ids      []int32
	text     string
	err      error
}

func complete(c clients, m *v1.GenerateRequest) completion {
	r, err := c.inference.Complete(context.Background(), req(m))
	if err != nil {
		return completion{err: err}
	}
	return completion{started: r.Msg.GetStarted(), finished: r.Msg.GetFinished(),
		ids: r.Msg.GetTokenIds(), text: r.Msg.GetText()}
}

// batchStats is the model's BatchStats as GetStats reports it.
func batchStats(t *testing.T, c clients) *v1.BatchStats {
	t.Helper()
	st, err := c.telemetry.GetStats(context.Background(), req(&v1.GetStatsRequest{ModelId: "dev"}))
	if err != nil {
		t.Fatal(err)
	}
	if len(st.Msg.GetModels()) != 1 || st.Msg.GetModels()[0].GetBatch() == nil {
		t.Fatalf("GetStats carries no batch stats for a device model: %+v", st.Msg.GetModels())
	}
	return st.Msg.GetModels()[0].GetBatch()
}

// hostGap is how far apart tokens a and b are in the host's logits after
// prompt and prefix: the host is the reference that is in neither arm.
func hostGap(t *testing.T, m *model.Model, prompt, prefix []int32, a, b int32) float64 {
	t.Helper()
	st := m.NewState(len(prompt) + len(prefix) + 1)
	defer st.Close()
	lg, err := st.Prefill(append(append([]int32{}, prompt...), prefix...))
	if err != nil {
		t.Fatal(err)
	}
	return math.Abs(float64(lg[a] - lg[b]))
}

// sameAsAlone demands got equal want up to the first place they part, and
// that place be a tie on the host. Past a tie the two continuations are
// different texts, so nothing after it is compared.
func sameAsAlone(t *testing.T, m *model.Model, what, prompt string, got, want []int32) {
	t.Helper()
	ids := m.Vocab.Encode(prompt, true)
	for j := range min(len(got), len(want)) {
		if got[j] == want[j] {
			continue
		}
		if gap := hostGap(t, m, ids, want[:j], want[j], got[j]); gap > tieMargin {
			t.Fatalf("%s: token %d is %d batched and %d alone, %.4f apart on the host\nbatched %v\nalone   %v",
				what, j, got[j], want[j], gap, got, want)
		} else {
			t.Logf("%s: token %d a tie (%.4f apart on the host), not compared past it", what, j, gap)
		}
		return
	}
	if len(got) != len(want) {
		t.Fatalf("%s: %d tokens batched and %d alone with no token apart\nbatched %v\nalone   %v",
			what, len(got), len(want), got, want)
	}
}

// TestBatchConcurrentGreedyGeneratesShareSteps: N greedy generates through the
// shipping handler, held until all N wait so the loop admits them together,
// produce what each produces alone -- and the tier's own SessionRows says
// their rows really shared device steps. While they wait the device's queue
// lists them and each reports where it stood.
//
// Against a step that feeds row i the token of row i+1, every row parts from
// its alone run by whole logits at its second token. Against a loop that
// steps each row with State.Forward, the outputs match and JointSteps and
// SessionRows stay at zero.
func TestBatchConcurrentGreedyGeneratesShareSteps(t *testing.T) {
	eachBatchModel(t, func(t *testing.T, name string) {
		for _, n := range []int{2, 4} {
			t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) { concurrentGreedy(t, name, n) })
		}
	})
}

func concurrentGreedy(t *testing.T, name string, n int) {
	e, lm, c := batchEngine(t, name, Config{}, n+1, nil)
	const gen = 24
	reqOf := func(i int) *v1.GenerateRequest {
		return &v1.GenerateRequest{ModelId: "dev", Prompt: text(prompts[i]), MaxTokens: gen}
	}

	alone := make([]completion, n)
	for i := range n {
		if alone[i] = complete(c, reqOf(i)); alone[i].err != nil {
			t.Fatalf("alone %d: %v", i, alone[i].err)
		}
		if !alone[i].started.GetBatched() {
			t.Fatalf("alone %d did not run as a row of the step loop", i)
		}
	}
	before := batchStats(t, c)
	rows0 := lm.gpu.Stats().SessionRows

	release := holdGates(t, e, lm)
	got := make([]completion, n)
	var wg sync.WaitGroup
	for i := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got[i] = complete(c, reqOf(i))
		}()
	}
	waitFor(t, "every request to wait for a row", func() bool { return waitingFor(lm.loop) == n })
	q, err := c.session.GetDeviceQueue(context.Background(), req(&v1.GetDeviceQueueRequest{DeviceId: "gpu:0"}))
	if err != nil {
		t.Fatal(err)
	}
	if q.Msg.GetWaiting() != int32(n) || len(q.Msg.GetSessionIds()) != n {
		t.Fatalf("the device queue reads %d waiting, %v, with %d requests waiting for a row of a held loop",
			q.Msg.GetWaiting(), q.Msg.GetSessionIds(), n)
	}
	release()
	wg.Wait()

	var depths []int32
	for i := range n {
		if got[i].err != nil {
			t.Fatalf("batched %d: %v", i, got[i].err)
		}
		if !got[i].started.GetBatched() {
			t.Fatalf("batched %d did not run as a row", i)
		}
		depths = append(depths, got[i].started.GetQueueDepthOnEntry())
		sameAsAlone(t, lm.m, fmt.Sprintf("row %d", i), prompts[i], got[i].ids, alone[i].ids)
		t.Logf("row %d: %q", i, got[i].text)
	}
	slices.Sort(depths)
	for i, d := range depths {
		if d != int32(i) {
			t.Fatalf("queue depths on entry %v, want 0..%d: each request counts the ones ahead of it", depths, n-1)
		}
	}

	after := batchStats(t, c)
	joint := after.GetJointSteps() - before.GetJointSteps()
	rows := lm.gpu.Stats().SessionRows - rows0
	if joint == 0 || rows == 0 || after.GetJointRows()-before.GetJointRows() != int64(rows) {
		t.Fatalf("no rows shared a step: %d joint step(s), the tier ran %d session row(s), stats %+v",
			joint, rows, after)
	}
	if after.GetMaxSessionsPerStep() != int32(n) {
		t.Fatalf("at most %d sessions a step, want %d: the requests were admitted together",
			after.GetMaxSessionsPerStep(), n)
	}
	t.Logf("%d joint step(s) over %d row(s); %+v", joint, rows, after)
}

// TestBatchARequestJoinsWhileOthersDecode: a request whose prompt arrives
// while two rows decode is fed a chunk at a time inside their steps
// (model.StepRuns), so they keep producing a token every step across its
// admission -- counted by the loop as steps a chunk rode in beside decoding
// rows -- and every row still comes out as it does alone. Against a feed that
// takes the whole prompt in one step, PromptSteps counts one step where the
// prompt has many chunks.
func TestBatchARequestJoinsWhileOthersDecode(t *testing.T) {
	eachBatchModel(t, func(t *testing.T, name string) {
		const chunk = 4
		// No prompt store: the "alone" pass would store the long prompt and
		// the joined one restore it, leaving no chunks to count.
		e, lm, c := batchEngine(t, name, Config{PromptChunk: chunk, NoMemCache: true}, 4, nil)
		const gen = 32
		long := strings.Repeat("The sun was warm and the birds sang in the tall green trees. ", 3)
		reqs := []*v1.GenerateRequest{
			{ModelId: "dev", Prompt: text(prompts[0]), MaxTokens: gen},
			{ModelId: "dev", Prompt: text(prompts[1]), MaxTokens: gen},
			{ModelId: "dev", Prompt: text(long), MaxTokens: 4},
		}
		alone := make([]completion, len(reqs))
		for i, m := range reqs {
			if alone[i] = complete(c, m); alone[i].err != nil {
				t.Fatal(alone[i].err)
			}
		}
		chunks := (len(lm.m.Vocab.Encode(long, true)) + chunk - 1) / chunk
		before := batchStats(t, c)

		got := make([]completion, len(reqs))
		var wg sync.WaitGroup
		release := holdGates(t, e, lm)
		for i := range 2 {
			wg.Add(1)
			go func() { defer wg.Done(); got[i] = complete(c, reqs[i]) }()
		}
		waitFor(t, "the first two to wait", func() bool { return waitingFor(lm.loop) == 2 })
		release()
		waitFor(t, "the first two to decode", func() bool {
			return batchStats(t, c).GetSteps()-before.GetSteps() >= 2
		})
		wg.Add(1)
		go func() { defer wg.Done(); got[2] = complete(c, reqs[2]) }()
		wg.Wait()

		for i, g := range got {
			if g.err != nil {
				t.Fatal(g.err)
			}
			if g.finished.GetCompletionTokens() == 0 {
				t.Fatalf("request %d produced nothing", i)
			}
			sameAsAlone(t, lm.m, fmt.Sprintf("request %d", i), []string{prompts[0], prompts[1], long}[i],
				g.ids, alone[i].ids)
		}
		after := batchStats(t, c)
		mixed := after.GetPromptSteps() - before.GetPromptSteps()
		if mixed < int64(chunks-1) {
			t.Fatalf("%d step(s) ran beside the %d-chunk prompt, want at least %d: admission held the "+
				"decoding rows up (stats %+v)", mixed, chunks, chunks-1, after)
		}
		t.Logf("a %d-chunk prompt joined with %d decode step(s) beside it", chunks, mixed)
	})
}

// streamed is one in-process generate: its events, and how it ended.
type streamed struct {
	ids      []int32
	text     strings.Builder
	finished *Finished
	err      error
}

// TestBatchCancelOneRowLeavesTheOthers: three rows decode together and one is
// cancelled after its first token. That row finishes CANCELLED; the other two
// run to max_tokens exactly as they do alone. Run twice, the second round
// ends with the device's ledger where the first left it. Against a cancel
// that retires the whole step, the other two stop short and are cancelled
// with it; against a closed session whose State is not closed, the ledger
// keeps its history.
func TestBatchCancelOneRowLeavesTheOthers(t *testing.T) {
	eachBatchModel(t, func(t *testing.T, name string) {
		e, lm, c := batchEngine(t, name, Config{}, 4, nil)
		const gen = 20
		ctx := context.Background()

		alone := make([]completion, 3)
		for i := range alone {
			alone[i] = complete(c, &v1.GenerateRequest{ModelId: "dev", Prompt: text(prompts[i]), MaxTokens: gen})
			if alone[i].err != nil {
				t.Fatal(alone[i].err)
			}
		}

		round := func() {
			for i := range 3 {
				if _, err := e.CreateSession(SessionOptions{ModelID: "dev", SessionID: fmt.Sprint("c", i), MaxSeq: 128}); err != nil {
					t.Fatal(err)
				}
			}
			release := holdGates(t, e, lm)
			out := make([]streamed, 3)
			var wg sync.WaitGroup
			for i := range 3 {
				wg.Add(1)
				go func() {
					defer wg.Done()
					o := &out[i]
					o.err = e.Generate(ctx, GenerateOptions{SessionID: fmt.Sprint("c", i),
						Prompt: Prompt{Kind: PromptText, Text: prompts[i]}, MaxTokens: gen},
						func(ev Event) error {
							switch ev.Kind {
							case EventToken:
								if ev.Token.ID >= 0 {
									o.ids = append(o.ids, ev.Token.ID)
								}
								if i == 1 && ev.Token.Index == 0 {
									if was, err := e.CancelSession("c1"); err != nil || !was {
										return fmt.Errorf("cancel: %v, was generating %v", err, was)
									}
								}
							case EventFinished:
								o.finished = ev.Finished
							}
							return nil
						})
				}()
			}
			waitFor(t, "three rows to wait", func() bool { return waitingFor(lm.loop) == 3 })
			release()
			wg.Wait()

			for i, o := range out {
				if o.err != nil || o.finished == nil {
					t.Fatalf("row %d: %v, finished %+v", i, o.err, o.finished)
				}
			}
			if f := out[1].finished; f.Reason != FinishCancelled || f.CompletionTokens >= gen {
				t.Fatalf("the cancelled row finished %v after %d token(s)", f.Reason, f.CompletionTokens)
			}
			for _, i := range []int{0, 2} {
				if out[i].finished.Reason == FinishCancelled {
					t.Fatalf("row %d was cancelled with row 1", i)
				}
				sameAsAlone(t, lm.m, fmt.Sprint("row ", i), prompts[i], out[i].ids, alone[i].ids)
			}
			for i := range 3 {
				if err := e.CloseSession(fmt.Sprint("c", i)); err != nil {
					t.Fatal(err)
				}
			}
		}
		// The first round leaves what stays on the card while the model is
		// loaded -- its blocks, and the joint step's scratch at the widths it
		// ran -- so the second must end where the first did: anything more is
		// a session's history that never went back.
		round()
		used := deviceUsed(lm)
		round()
		if u := deviceUsed(lm); u != used {
			t.Fatalf("the device holds %d bytes with every session closed, %d after the same round before", u, used)
		}
		t.Logf("the ledger reads %d bytes after each round", used)
	})
}

// deviceUsed is what the tier's ledger has charged on every device.
func deviceUsed(lm *LoadedModel) uint64 {
	var n uint64
	for _, b := range lm.gpu.Budgets() {
		n += b.Used
	}
	return n
}

// TestBatchForceUnloadEndsEveryRow: an unload with force while three rows
// decode ends every one of them CANCELLED -- a Finished event, not a fault
// from a closed model -- closes their sessions, stops the loop, and leaves no
// goroutine behind. Against a loop that is not stopped, the goroutine count
// never comes back; against a close that does not cancel the rows first, the
// unload deadlocks against the generates holding their sessions.
func TestBatchForceUnloadEndsEveryRow(t *testing.T) {
	eachBatchModel(t, func(t *testing.T, name string) {
		e, lm, _ := batchEngine(t, name, Config{}, 4, nil)
		lp := lm.loop
		// Taken with the loop running: it, and whatever the tier started,
		// must be gone after the unload.
		g0 := runtime.NumGoroutine()
		ctx := context.Background()
		release := holdGates(t, e, lm)
		out := make([]streamed, 3)
		var wg, first sync.WaitGroup
		first.Add(3)
		for i := range 3 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				o := &out[i]
				seen := false
				o.err = e.Generate(ctx, GenerateOptions{ModelID: "dev",
					Prompt: Prompt{Kind: PromptText, Text: prompts[i]}, MaxTokens: 4000},
					func(ev Event) error {
						switch ev.Kind {
						case EventToken:
							if !seen {
								seen = true
								first.Done()
							}
						case EventFinished:
							o.finished = ev.Finished
						}
						return nil
					})
			}()
		}
		waitFor(t, "three rows to wait", func() bool { return waitingFor(lp) == 3 })
		release()
		first.Wait()
		closed, err := e.UnloadModel("dev", true)
		if err != nil {
			t.Fatal(err)
		}
		wg.Wait()
		if closed != 3 {
			t.Fatalf("the unload closed %d session(s), want the 3 decoding", closed)
		}
		for i, o := range out {
			if o.err != nil || o.finished == nil || o.finished.Reason != FinishCancelled {
				t.Fatalf("row %d ended %v, finished %+v: want CANCELLED", i, o.err, o.finished)
			}
		}
		select {
		case <-lp.done:
		default:
			t.Fatal("the step loop is still running after its model was unloaded")
		}
		err = e.Generate(ctx, GenerateOptions{ModelID: "dev", Prompt: Prompt{Kind: PromptText, Text: story}},
			func(Event) error { return nil })
		if !errors.Is(err, ErrNotFound) {
			t.Fatalf("a generate on the unloaded model: %v, want not found", err)
		}
		waitFor(t, "every goroutine the model started to end", func() bool { return runtime.NumGoroutine() < g0 })
	})
}

// TestBatchSampledRowsAreTheirOwn: a seeded, sampled request decodes beside
// another with its own sampler, its own random stream and its own history,
// and a stop string and max_tokens end it as they end a request alone.
//
// Token-for-token agreement with the request run ALONE is not the property: a
// joint step's logits sit within a band of decode's (tieMargin), and a draw
// near the boundary between two candidates falls either side of it. What is
// exact is the pair itself: the same two sessions stepped together through
// model.Step, outside the server, each fed the tokens the server emitted,
// give the subject the logits the server's row saw, and the subject's sampler
// -- fresh, seeded as the request was -- must draw from them exactly the
// tokens the server sent. Against a random stream shared between the rows,
// the server's draws are interleaved with the companion's and the replay parts
// from it at the second token; against a row fed another row's token, at the
// first one after.
func TestBatchSampledRowsAreTheirOwn(t *testing.T) {
	eachBatchModel(t, func(t *testing.T, name string) {
		// Always joint: the replay below steps the pair together, and a step
		// the loop chose to run one session after another would be a
		// different reduction order (TestBatchChoosesByMeasurement covers
		// that arm against each request alone).
		e, lm, c := batchEngine(t, name, Config{JointSteps: JointAlways}, 4, nil)
		const gen = 24
		temp, topP := float32(0.9), float32(0.95)
		sampled := func(prompt string, seed uint64, maxTok int32, stop ...string) *v1.GenerateRequest {
			return &v1.GenerateRequest{ModelId: "dev", Prompt: text(prompt), MaxTokens: maxTok, Stop: stop,
				Sampling: &v1.SamplingParams{Temperature: &temp, TopP: &topP, Seed: &seed}}
		}
		pair := func(subject *v1.GenerateRequest) (got, other completion) {
			// The subject queues first, so it is the first row of every step,
			// as it is in the replay.
			release := holdGates(t, e, lm)
			var wg sync.WaitGroup
			wg.Add(2)
			go func() { defer wg.Done(); got = complete(c, subject) }()
			waitFor(t, "the subject to wait", func() bool { return waitingFor(lm.loop) == 1 })
			go func() { defer wg.Done(); other = complete(c, sampled(prompts[1], 11, 4*gen)) }()
			waitFor(t, "both to wait", func() bool { return waitingFor(lm.loop) == 2 })
			release()
			wg.Wait()
			if got.err != nil || other.err != nil {
				t.Fatal(got.err, other.err)
			}
			if r := other.finished.GetReason(); r != v1.FinishReason_FINISH_REASON_MAX_TOKENS &&
				r != v1.FinishReason_FINISH_REASON_EOS {
				t.Fatalf("the companion finished %v; the replay feeds it every token it sent", r)
			}
			return got, other
		}

		// The first pair after a load is a cold engine: on a hybrid its rows
		// come out within the band of every later pair's but not equal to
		// them (the first of three identical sampled pairs parted from the
		// other two at token 10, which agreed throughout). The
		// replay runs warm, so the pair it checks does too.
		pair(sampled(prompts[0], 7, gen))
		before := batchStats(t, c)
		got, other := pair(sampled(prompts[0], 7, gen))
		if batchStats(t, c).GetJointSteps() == before.GetJointSteps() {
			t.Fatal("the pair never shared a step: the replay would compare the one-at-a-time path")
		}
		if got.finished.GetReason() == v1.FinishReason_FINISH_REASON_MAX_TOKENS && len(got.ids) != gen {
			t.Fatalf("max_tokens %d gave %d tokens", gen, len(got.ids))
		}
		t.Logf("the pair: %+v; subject %d tokens, companion %d", batchStats(t, c), len(got.ids), len(other.ids))
		sp := model.Sampler{Temp: float64(temp), TopP: float64(topP), Seed: 7}
		replayPair(t, lm, e.cfg.DefaultMaxSeq, sp, lm.m.Vocab.Encode(prompts[0], true), lm.m.Vocab.Encode(prompts[1], true),
			got.ids, other.ids)

		// A stop string out of the middle of that text ends the row before it
		// and never reaches the client. The same pair runs the same steps, so
		// the text up to the stop is the text above.
		if len(got.text) < 12 {
			t.Skipf("the sampled text %q is too short to cut a stop string from", got.text)
		}
		stop := got.text[len(got.text)/2 : len(got.text)/2+4]
		st, _ := pair(sampled(prompts[0], 7, gen, stop))
		if st.finished.GetReason() != v1.FinishReason_FINISH_REASON_STOP || st.finished.GetStopMatched() != stop ||
			strings.Contains(st.text, stop) || !strings.HasPrefix(got.text, st.text) {
			t.Fatalf("stop %q: finished %v with %q, out of %q", stop, st.finished.GetReason(), st.text, got.text)
		}
	})
}

// replayPair steps a subject and a companion session through
// model.StepRuns on lm's device as the server's loop stepped them -- both
// prompts as one step, then one token each a step, the subject first, and the
// subject alone once the companion is done -- feeding each the tokens the
// server emitted for it, and demands the subject's sampler sp draw exactly
// subj from the subject's logits. The companion's tokens are its history
// here, not something under test; it finished on max_tokens or EOS, so every
// token it sent was stepped and nothing else was.
func replayPair(t *testing.T, lm *LoadedModel, maxSeq int, sp model.Sampler, subjPrompt, compPrompt, subj, comp []int32) {
	t.Helper()
	// The server's sessions' capacity: a plan's capacity can choose a
	// different attention kernel, which is a different reduction order.
	open := func() *model.State {
		st := lm.m.NewState(maxSeq)
		t.Cleanup(func() { st.Close() })
		if err := st.SetDeviceLayers(lm.dev, -1); err != nil {
			t.Fatal(err)
		}
		if !st.Steppable() {
			t.Fatalf("a replay session is not wholly on the device (%d blocks)", st.GPULayers())
		}
		return st
	}
	a, b := open(), open()
	// The loop resets a session before a generate that does not continue it.
	a.Reset()
	b.Reset()
	out, err := model.StepRuns([]model.Run{{State: a, Tokens: subjPrompt, Logits: true},
		{State: b, Tokens: compPrompt, Logits: true}})
	if err != nil {
		t.Fatal(err)
	}
	lg := out[0]
	for j, want := range subj {
		if got := sp.Sample(lg); got != want {
			t.Fatalf("token %d: the server sent %d, and the same pair stepped together draws %d\nserver %v",
				j, want, got, subj)
		}
		sp.Observe(want)
		if j == len(subj)-1 {
			break
		}
		runs := []model.Run{{State: a, Tokens: subj[j : j+1], Logits: true}}
		if j < len(comp) {
			runs = append(runs, model.Run{State: b, Tokens: comp[j : j+1], Logits: true})
		}
		if out, err = model.StepRuns(runs); err != nil {
			t.Fatal(err)
		}
		lg = out[0]
	}
}

// TestBatchARefusedJointStepRunsTheRowsAlone: a device that refuses steps
// across sessions (tier.Config.NoBatch, refused before anything runs) costs
// the batch its shared step and nothing else: every row runs alone, comes out
// identical to its run alone, and the refusal is counted once for the rows it
// was refused for rather than once a step. Against a loop that treats the
// refusal as the requests' error, every request fails.
func TestBatchARefusedJointStepRunsTheRowsAlone(t *testing.T) {
	eachBatchModel(t, func(t *testing.T, name string) {
		e, lm, c := batchEngine(t, name, Config{}, 3, func(c *tier.Config) { c.NoBatch = true })
		const gen = 12
		alone := make([]completion, 2)
		for i := range alone {
			if alone[i] = complete(c, &v1.GenerateRequest{ModelId: "dev", Prompt: text(prompts[i]), MaxTokens: gen}); alone[i].err != nil {
				t.Fatal(alone[i].err)
			}
		}
		release := holdGates(t, e, lm)
		got := make([]completion, 2)
		var wg sync.WaitGroup
		for i := range got {
			wg.Add(1)
			go func() {
				defer wg.Done()
				got[i] = complete(c, &v1.GenerateRequest{ModelId: "dev", Prompt: text(prompts[i]), MaxTokens: gen})
			}()
		}
		waitFor(t, "both to wait", func() bool { return waitingFor(lm.loop) == 2 })
		release()
		wg.Wait()
		for i, g := range got {
			if g.err != nil {
				t.Fatalf("row %d failed on a refused joint step: %v", i, g.err)
			}
			if !slices.Equal(g.ids, alone[i].ids) {
				t.Fatalf("row %d run alone after a refusal differs from its run alone:\n%v\n%v", i, g.ids, alone[i].ids)
			}
		}
		b := batchStats(t, c)
		if b.GetJointRefusals() == 0 || !strings.Contains(b.GetLastRefusal(), "NoBatch") || b.GetJointSteps() != 0 {
			t.Fatalf("stats %+v: want a refusal naming NoBatch and no joint step", b)
		}
		// Admissions and retirements change the rows: two admitted together and
		// two retired is at most three shapes worth retrying for.
		if b.GetJointRefusals() > 3 {
			t.Fatalf("%d refusals over %d steps: the loop retried a refused step every step", b.GetJointRefusals(), b.GetSteps())
		}
	})
}

// TestBatchASessionNotWhollyOnTheDeviceRunsAlone: a session holding some of
// the model's blocks on the host cannot be a row; it runs on its gates as
// every session did before the loop, beside a session that is a row, and both
// answer. With batching off (MaxBatchRows 1) there is no loop at all.
func TestBatchASessionNotWhollyOnTheDeviceRunsAlone(t *testing.T) {
	eachBatchModel(t, notWhollyOnTheDevice)
}

func notWhollyOnTheDevice(t *testing.T, name string) {
	e, lm, c := batchEngine(t, name, Config{}, 3, nil)
	ctx := context.Background()
	if _, err := e.CreateSession(SessionOptions{ModelID: "dev", SessionID: "split", MaxSeq: 64, MaxDeviceBlocks: 2}); err != nil {
		t.Fatal(err)
	}
	if _, err := e.CreateSession(SessionOptions{ModelID: "dev", SessionID: "whole", MaxSeq: 64}); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"split", "whole"} {
		q, err := c.session.GetSession(ctx, req(&v1.GetSessionRequest{SessionId: id}))
		if err != nil {
			t.Fatal(err)
		}
		// Both run beside other sessions: the whole one as a row of the step,
		// the split one interleaved with it a step at a time.
		dev, want := q.Msg.GetSession().GetDeviceBlocks(), v1.ExecutionMode_EXECUTION_MODE_PARALLEL
		if id == "whole" {
			if dev != int32(lm.m.Cfg.NLayer) {
				t.Skipf("NO DEVICE ROOM: the card took %d of %d blocks", dev, lm.m.Cfg.NLayer)
			}
		} else if dev >= int32(lm.m.Cfg.NLayer) {
			t.Fatalf("the split session holds all %d blocks", dev)
		}
		if got := q.Msg.GetSession().GetExecution(); got != want {
			t.Fatalf("session %s (%d device blocks) reports %v, want %v", id, dev, got, want)
		}
	}
	var wg sync.WaitGroup
	got := map[string]completion{}
	var mu sync.Mutex
	for _, id := range []string{"split", "whole"} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r := complete(c, &v1.GenerateRequest{SessionId: id, Prompt: text(story), MaxTokens: 8})
			mu.Lock()
			got[id] = r
			mu.Unlock()
		}()
	}
	wg.Wait()
	for id, r := range got {
		if r.err != nil || r.finished.GetCompletionTokens() == 0 {
			t.Fatalf("%s: %v, finished %+v", id, r.err, r.finished)
		}
		if r.started.GetBatched() != (id == "whole") {
			t.Fatalf("%s ran batched=%v", id, r.started.GetBatched())
		}
	}
	off, _, offc := batchEngineNoLoop(t, name)
	r := complete(offc, &v1.GenerateRequest{ModelId: "dev", Prompt: text(story), MaxTokens: 4})
	if r.err != nil || r.started.GetBatched() {
		t.Fatalf("with batching off: %v, batched %v", r.err, r.started.GetBatched())
	}
	st, err := offc.telemetry.GetStats(ctx, req(&v1.GetStatsRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	if st.Msg.GetModels()[0].GetBatch() != nil {
		t.Fatal("a model with batching off reports batch stats")
	}
	off.Close()
}

// batchEngineNoLoop is name on the GPU with MaxBatchRows 1.
func batchEngineNoLoop(t *testing.T, name string) (*Engine, *LoadedModel, clients) {
	t.Helper()
	path := modelPath(t, name)
	e := New(Config{Probe: oneCardProbe, Version: "test", MaxBatchRows: 1})
	t.Cleanup(e.Close)
	lm, err := e.LoadModel(LoadOptions{Path: path, ModelID: "dev", DeviceIDs: []string{"gpu:0"}})
	if err != nil {
		t.Skipf("NO DEVICE: %v -- this gate proved nothing", err)
	}
	if lm.loop != nil {
		t.Fatal("MaxBatchRows 1 still built a step loop")
	}
	return e, lm, serveEngine(t, e)
}

// TestBatchAQueueTimeoutLeavesTheQueue: a request that stops waiting for a row
// gets ResourceExhausted, leaves the device's queue, and the next request on
// the same session runs. Against a withdrawal that never finds the request,
// it waits for a row the holder never gives up; the call's own deadline turns
// that hang into a wrong code.
func TestBatchAQueueTimeoutLeavesTheQueue(t *testing.T) {
	eachBatchModel(t, queueTimeout)
}

func queueTimeout(t *testing.T, name string) {
	e, lm, c := batchEngine(t, name, Config{}, 2, nil)
	ctx := context.Background()
	if _, err := e.CreateSession(SessionOptions{ModelID: "dev", SessionID: "s", MaxSeq: 64}); err != nil {
		t.Fatal(err)
	}
	release := holdGates(t, e, lm)
	dctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	_, err := c.inference.Complete(dctx, req(&v1.GenerateRequest{
		SessionId: "s", Prompt: text(story), MaxTokens: 2, QueueTimeoutMillis: 20,
	}))
	wantCode(t, "a batched generate that gave up waiting for a row", err, connect.CodeResourceExhausted)
	if w := waitingFor(lm.loop); w != 0 {
		t.Fatalf("%d request(s) still waiting after the only one timed out", w)
	}
	if s, _ := e.Session("s"); s.queuePos.Load() != 0 {
		t.Fatalf("the session reports queue position %d after leaving the queue", s.queuePos.Load())
	}
	release()
	r := complete(c, &v1.GenerateRequest{SessionId: "s", Prompt: text(story), MaxTokens: 2})
	if r.err != nil || r.finished.GetCompletionTokens() == 0 || !r.started.GetBatched() {
		t.Fatalf("the session after a queue timeout: %v, %+v", r.err, r.finished)
	}
}

// TestBatchChoosesByMeasurement: with the choice left to measurement, a pair
// of requests long enough for a probe runs decode steps BOTH ways -- joint,
// and one session after another -- settles a choice for two rows with a time
// for each arm, and both rows still come out as they do alone. With joint
// steps forbidden, nothing but the admission step runs joint and the rows
// still come out as they do alone. The choice between the arms is gated on
// numbers it is handed in TestJointChoiceFollowsTheMeasurement; this is the
// selection check that the loop really runs and times both, which a loop that
// never consulted the choice (every step joint) fails on SeparateSteps.
func TestBatchChoosesByMeasurement(t *testing.T) {
	eachBatchModel(t, func(t *testing.T, name string) {
		for _, mode := range []JointSteps{JointMeasured, JointNever} {
			t.Run(map[JointSteps]string{JointMeasured: "measured", JointNever: "never"}[mode], func(t *testing.T) {
				chooses(t, name, mode)
			})
		}
	})
}

func chooses(t *testing.T, name string, mode JointSteps) {
	e, lm, c := batchEngine(t, name, Config{JointSteps: mode}, 3, nil)
	// Long enough for one probe of two rows and some steps after it.
	gen := int32(jointRounds*len(abba)*jointRun + 16)
	reqOf := func(i int) *v1.GenerateRequest {
		return &v1.GenerateRequest{ModelId: "dev", Prompt: text(prompts[i]), MaxTokens: gen}
	}
	alone := []completion{complete(c, reqOf(0)), complete(c, reqOf(1))}
	release := holdGates(t, e, lm)
	got := make([]completion, 2)
	var wg sync.WaitGroup
	for i := range got {
		wg.Add(1)
		go func() { defer wg.Done(); got[i] = complete(c, reqOf(i)) }()
	}
	waitFor(t, "both to wait", func() bool { return waitingFor(lm.loop) == 2 })
	release()
	wg.Wait()
	for i := range got {
		if got[i].err != nil || alone[i].err != nil {
			t.Fatal(got[i].err, alone[i].err)
		}
		if len(got[i].ids) < int(gen) {
			t.Fatalf("row %d ended after %d tokens, before a probe could finish: pick another prompt",
				i, len(got[i].ids))
		}
		sameAsAlone(t, lm.m, fmt.Sprint("row ", i), prompts[i], got[i].ids, alone[i].ids)
	}
	b := batchStats(t, c)
	t.Logf("%+v", b)
	if b.GetSeparateSteps() == 0 {
		t.Fatalf("no step ran one session after another: the choice was never consulted (%+v)", b)
	}
	if mode == JointNever {
		// The prompts go in one joint step: admission is always joint.
		if b.GetJointSteps() > 1 {
			t.Fatalf("%d joint steps with joint steps forbidden", b.GetJointSteps())
		}
		return
	}
	var two *v1.JointChoice
	for _, ch := range b.GetChoices() {
		if ch.GetRows() == 2 {
			two = ch
		}
	}
	if b.GetJointSteps() < 2 || two == nil || !two.GetSettled() ||
		two.GetJointStepMillis() <= 0 || two.GetSeparateStepMillis() <= 0 {
		t.Fatalf("the probe for two rows did not time both arms and settle: %+v", b)
	}
}
