package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/engine/model"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// hybridModel is a recurrent hybrid (Mamba-2 layers beside attention): its
// restore needs the recurrent summary at the restored position as well as the
// pages, which a text-only model never exercises.
const hybridModel = "synth-granitehybrid.jlm"

// ttftEngine is an Engine with name loaded under id "m", host only.
func ttftEngine(t *testing.T, name string, cfg Config, o LoadOptions) (*Engine, *LoadedModel) {
	t.Helper()
	path := modelPath(t, name)
	cfg.ModelDir, cfg.Probe, cfg.Version = filepath.Dir(path), oneCardProbe, "test"
	if cfg.DefaultMaxSeq == 0 {
		cfg.DefaultMaxSeq = 512
	}
	e := New(cfg)
	t.Cleanup(e.Close)
	o.Path, o.ModelID = path, "m"
	lm, err := e.LoadModel(o)
	if err != nil {
		t.Fatalf("LoadModel(%s): %v", path, err)
	}
	return e, lm
}

// answer is what one generate produced and how much of its prompt came from
// the store.
type answer struct {
	ids                  []int32
	started, finished    int // Restored as Started and Finished report it
	batched, gotStarted  bool
	promptTokens, events int
}

func generate(t *testing.T, e *Engine, o GenerateOptions) answer {
	t.Helper()
	var a answer
	err := e.Generate(context.Background(), o, func(ev Event) error {
		a.events++
		switch ev.Kind {
		case EventStarted:
			a.gotStarted, a.started, a.batched = true, ev.Started.Restored, ev.Started.Batched
			a.promptTokens = ev.Started.PromptTokens
		case EventToken:
			if ev.Token.ID >= 0 {
				a.ids = append(a.ids, ev.Token.ID)
			}
		case EventFinished:
			a.finished = ev.Finished.Restored
		}
		return nil
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	return a
}

// longPrompt is a prompt long enough to fill more than one KV page (256
// positions) where the model's context allows, so a restore covers whole
// pages and a tail; on a short context (stories15M's 128, one page of 128) it
// is half the context, and the restore is the tail page's.
func longPrompt(m *model.Model) []int32 {
	ids := m.Vocab.Encode(strings.Repeat("Once upon a time there was a little dog who loved to run. ", 30), true)
	n := 300
	if c := m.Cfg.NCtx; c > 0 && c < 2*n {
		n = c / 2
	}
	return ids[:min(len(ids), n)]
}

func idsPrompt(ids []int32) Prompt { return Prompt{Kind: PromptIDs, IDs: ids} }

// TestTheMemCacheRestoresAndAnswersAsAColdPrefill runs a prompt, the same
// prompt again, and the prompt with a turn appended, on an engine with the
// memory cache and on one without: the second and third restore positions
// (counted by the engine and reported in Started and Finished), and every
// answer -- greedy and seeded sampling -- is the cold engine's. On a host
// model the restore is PrefillCached; on a device model the step loop's
// restore-then-join (RestorePrefix, SealPrompt); on a hybrid the recurrent
// summary comes back with the pages.
func TestTheMemCacheRestoresAndAnswersAsAColdPrefill(t *testing.T) {
	type arm struct {
		name   string
		model  string
		device bool
	}
	for _, a := range []arm{{"host", smallModel, false}, {"hybrid", hybridModel, false}, {"device", deviceModel, true}} {
		t.Run(a.name, func(t *testing.T) {
			open := func(cfg Config) (*Engine, *LoadedModel) {
				if a.device {
					e, lm, _ := batchEngine(t, a.model, cfg, 0, nil)
					return e, lm
				}
				return ttftEngine(t, a.model, cfg, LoadOptions{})
			}
			cold, clm := open(Config{NoMemCache: true})
			// The cache's bound is named, not left to follow the host: by
			// default it is an eighth of what the host has free at each
			// generate (storeLimit), and on a box short of memory -- the
			// model-free suite running every package at once under its cap --
			// that fell below one prompt's pages, evicted them, and request 1
			// restored nothing. The starved host below holds the test to it.
			warm, wlm := open(Config{MemCacheBytes: 64 << 20})
			warm.hostAvail = func() uint64 { return 64 << 10 }
			if wlm.ttft.store == nil || clm.ttft.store != nil {
				t.Fatal("NoMemCache did not select the arms: the store is on in both or neither")
			}
			p := longPrompt(wlm.m)
			turn := append(slices.Clone(p), wlm.m.Vocab.Encode(" The dog said hello to the cat.", false)...)
			seeded := &model.Sampler{Temp: 0.8, TopK: 40, Seed: 7}

			for i, o := range []GenerateOptions{
				{Prompt: idsPrompt(p), MaxTokens: 12},
				{Prompt: idsPrompt(p), MaxTokens: 12},
				{Prompt: idsPrompt(turn), MaxTokens: 12},
				{Prompt: idsPrompt(turn), MaxTokens: 12, Sampling: seeded},
			} {
				oc, ow := o, o
				oc.ModelID, ow.ModelID = "m", "m"
				if a.device {
					oc.ModelID, ow.ModelID = "dev", "dev"
				}
				if o.Sampling != nil {
					sc, sw := *o.Sampling, *o.Sampling
					oc.Sampling, ow.Sampling = &sc, &sw
				}
				var tapC, tapW sampleTrace
				if a.device && o.Sampling != nil {
					cold.sampleTap = tapC.add
				}
				want := generate(t, cold, oc)
				cold.sampleTap = nil
				if a.device && o.Sampling != nil {
					warm.sampleTap = tapW.add
				}
				got := generate(t, warm, ow)
				warm.sampleTap = nil
				if want.started != 0 {
					t.Fatalf("request %d: the engine with no store restored %d positions", i, want.started)
				}
				if a.device && !got.batched {
					t.Fatalf("request %d: on the device model the request did not run as a row of the step loop", i)
				}
				if got.started != got.finished {
					t.Fatalf("request %d: Started says %d restored, Finished %d", i, got.started, got.finished)
				}
				if i == 0 && got.started != 0 {
					t.Fatalf("the first request restored %d positions from an empty store", got.started)
				}
				if i > 0 && got.started == 0 {
					t.Fatalf("request %d restored nothing: its prefix ran before, so the store is not on the path", i)
				}
				// A whole-prompt hit is legitimate alone (the store holds the
				// logits the prompt ended on); a row of the step loop always
				// runs at least its last token.
				if got.started > got.promptTokens || got.batched && got.started == got.promptTokens {
					t.Fatalf("request %d restored %d of %d positions", i, got.started, got.promptTokens)
				}
				if a.device && o.Sampling != nil {
					// Sampled tokens are not the bar on a device (RULE 11c).
					// The restored history is bit-identical to the cold one
					// (the first 63 positions' K and V hash equal going out
					// and coming back), but the warm arm runs the prompt's
					// tail as a few rows and the cold arm as part of one wide
					// prefill, and the two reduce in different orders: logits
					// part by ~0.2 on a V100. A seeded draw lands in the sliver
					// between two near-tied candidates and picks a different
					// one, deterministically. What must hold is that the
					// distributions the draws were made from agree and that
					// the sampler drew once per token in both arms.
					tapC.compare(t, &tapW, want.ids, got.ids)
				} else if !slices.Equal(want.ids, got.ids) {
					t.Fatalf("request %d: with the store %v, without %v", i, got.ids, want.ids)
				}
				t.Logf("request %d: %d of %d prompt positions restored; %d tokens identical",
					i, got.started, got.promptTokens, len(got.ids))
			}
			st := warm.MemCacheStats(wlm)
			if st.Restored == 0 || st.Computed == 0 || st.Store.Sets == 0 || st.Store.Hits == 0 {
				t.Fatalf("the counters saw no restore: %+v", st)
			}
			if st.Store.Bytes == 0 || st.Store.Limit == 0 || st.Store.Bytes > st.Store.Limit {
				t.Fatalf("the store holds %d bytes against a %d limit", st.Store.Bytes, st.Store.Limit)
			}
		})
	}
}

// TestPooledStatesAnswerAsFreshOnes alternates two prompts whose answers
// differ on an engine that pools its States and on one that does not: the
// answers agree request for request (no history leaks from one request into
// the next), the pool builds one State and hands it out again, and the other
// engine builds and closes one per request.
func TestPooledStatesAnswerAsFreshOnes(t *testing.T) {
	pooled, plm := ttftEngine(t, smallModel, Config{NoMemCache: true}, LoadOptions{})
	fresh, flm := ttftEngine(t, smallModel, Config{NoMemCache: true, SessionPool: -1}, LoadOptions{})
	x := pooled.encodeText(t, plm, "Once upon a time")
	y := pooled.encodeText(t, plm, "The little dog ran to the park and")
	var answers [2][]int32
	for i := 0; i < 6; i++ {
		p := [][]int32{x, y}[i%2]
		want := generate(t, fresh, GenerateOptions{ModelID: "m", Prompt: idsPrompt(p), MaxTokens: 16})
		got := generate(t, pooled, GenerateOptions{ModelID: "m", Prompt: idsPrompt(p), MaxTokens: 16})
		if !slices.Equal(want.ids, got.ids) {
			t.Fatalf("request %d: a pooled State answered %v, a fresh one %v", i, got.ids, want.ids)
		}
		answers[i%2] = got.ids
	}
	if slices.Equal(answers[0], answers[1]) {
		t.Fatal("the two prompts have the same answer, so a leaked history could not show")
	}
	ps, fs := pooled.MemCacheStats(plm), fresh.MemCacheStats(flm)
	if ps.StatesCreated != 1 || ps.StatesReused != 5 || ps.StatesClosed != 0 || ps.Pooled != 1 {
		t.Fatalf("pooled engine: %+v; want one State built, reused five times, none closed", ps)
	}
	if fs.StatesCreated != 6 || fs.StatesClosed != 6 || fs.Pooled != 0 {
		t.Fatalf("unpooled engine: %+v; want six built and six closed", fs)
	}
	if n := len(pooled.Sessions()); n != 0 {
		t.Fatalf("a pooled State is listed as %d open session(s)", n)
	}
	// An unload closes the pool.
	if _, err := pooled.UnloadModel("m", false); err != nil {
		t.Fatalf("an idle pool kept the model from unloading: %v", err)
	}
	if ps := pooled.MemCacheStats(plm); ps.StatesClosed != 1 || ps.Pooled != 0 {
		t.Fatalf("after the unload: %+v", ps)
	}
}

func (e *Engine) encodeText(t *testing.T, lm *LoadedModel, s string) []int32 {
	t.Helper()
	ids, err := e.encode(lm, Prompt{Kind: PromptText, Text: s}, "")
	if err != nil {
		t.Fatal(err)
	}
	return ids
}

// TestAWarmLoadLeavesAPooledState: a load with Warm runs its prefill and
// decode step on a State that goes into the pool, so the first request
// builds none; it stores nothing and counts nothing as served.
func TestAWarmLoadLeavesAPooledState(t *testing.T) {
	e, lm := ttftEngine(t, smallModel, Config{}, LoadOptions{Warm: true})
	st := e.MemCacheStats(lm)
	if !st.Warmed || st.StatesCreated != 1 || st.Pooled != 1 {
		t.Fatalf("after a warm load: %+v; want one State built and pooled", st)
	}
	if st.Store.Sets != 0 || lm.tokensPrefilled.Load() != 0 || lm.tokensGenerated.Load() != 0 {
		t.Fatalf("the warm-up was counted as served or stored: %+v, %d prefilled", st, lm.tokensPrefilled.Load())
	}
	generate(t, e, GenerateOptions{ModelID: "m", Prompt: Prompt{Kind: PromptText, Text: story}, MaxTokens: 4})
	if st := e.MemCacheStats(lm); st.StatesCreated != 1 || st.StatesReused != 1 {
		t.Fatalf("the first request after a warm load: %+v; want the warm State reused", st)
	}

	_, cold := ttftEngine(t, smallModel, Config{}, LoadOptions{})
	if st := &cold.ttft; st.warmed.Load() || st.statesCreated.Load() != 0 {
		t.Fatal("a load without Warm warmed")
	}
}

// TestAFullQueueIsRefusedWith429 holds one request mid-decode on a model
// whose queue holds one: the next is refused before it builds anything --
// 429 with Retry-After on both HTTP shims, streaming or not, and
// ResourceExhausted over Connect -- and once the first ends a request runs.
func TestAFullQueueIsRefusedWith429(t *testing.T) {
	e, lm := ttftEngine(t, chatModel, Config{MaxQueue: 1, RetryAfter: 3 * time.Second}, LoadOptions{})
	c := serveEngine(t, e)

	hold, holding := make(chan struct{}), make(chan struct{})
	var once sync.Once
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		e.Generate(context.Background(), GenerateOptions{ModelID: "m",
			Prompt: Prompt{Kind: PromptText, Text: story}, MaxTokens: 8},
			func(ev Event) error {
				if ev.Kind == EventToken {
					once.Do(func() { close(holding) })
					<-hold
				}
				return nil
			})
	}()
	<-holding
	created := lm.ttft.statesCreated.Load()

	post := func(path, body string) *http.Response {
		r, err := http.Post(c.url+path, "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { r.Body.Close() })
		return r
	}
	for _, x := range []struct{ path, body, typ string }{
		{"/v1/completions", `{"model":"m","prompt":"hi","max_tokens":2}`, "rate_limit_error"},
		{"/v1/completions", `{"model":"m","prompt":"hi","max_tokens":2,"stream":true}`, "rate_limit_error"},
		{"/v1/messages", `{"model":"m","max_tokens":2,"messages":[{"role":"user","content":"hi"}]}`, "overloaded_error"},
		{"/v1/messages", `{"model":"m","max_tokens":2,"stream":true,"messages":[{"role":"user","content":"hi"}]}`, "overloaded_error"},
	} {
		r := post(x.path, x.body)
		b, _ := io.ReadAll(r.Body)
		if r.StatusCode != http.StatusTooManyRequests || r.Header.Get("Retry-After") != "3" {
			t.Fatalf("%s %s: status %d, Retry-After %q, body %s", x.path, x.body, r.StatusCode, r.Header.Get("Retry-After"), b)
		}
		if !strings.Contains(string(b), x.typ) {
			t.Fatalf("%s: body %s does not name %s", x.path, b, x.typ)
		}
	}
	_, err := c.inference.Complete(context.Background(), req(&v1.GenerateRequest{ModelId: "m", Prompt: text(story), MaxTokens: 2}))
	wantCode(t, "Complete past a full queue", err, connect.CodeResourceExhausted)
	if got := lm.ttft.statesCreated.Load(); got != created {
		t.Fatalf("a refused request built %d State(s)", got-created)
	}
	if st := e.MemCacheStats(lm); st.Refused != 5 {
		t.Fatalf("refused %d, want 5", st.Refused)
	}
	mr, err := http.Get(c.url + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	mb, _ := io.ReadAll(mr.Body)
	mr.Body.Close()
	for _, want := range []string{`jitllm_requests_refused_total{model="m"} 5`, `jitllm_requests_in_flight{model="m"} 1`} {
		if !strings.Contains(string(mb), want) {
			t.Fatalf("/metrics does not carry %q", want)
		}
	}

	close(hold)
	wg.Wait()
	r := post("/v1/completions", `{"model":"m","prompt":"hi","max_tokens":2}`)
	if r.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(r.Body)
		t.Fatalf("after the queue drained: status %d, %s", r.StatusCode, b)
	}
}

// TestACancelledPrefillLeavesThePooledStateClean: a request whose client has
// gone stops its prefill between chunks (ErrPrefillInterrupted becomes the
// context's error, with no Started sent), and the State it leaves in the pool
// answers the next request as a fresh one.
func TestACancelledPrefillLeavesThePooledStateClean(t *testing.T) {
	e, lm := ttftEngine(t, smallModel, Config{NoMemCache: true}, LoadOptions{})
	p := longPrompt(lm.m)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	started := false
	err := e.Generate(ctx, GenerateOptions{ModelID: "m", Prompt: idsPrompt(p), MaxTokens: 8},
		func(ev Event) error { started = started || ev.Kind == EventStarted; return nil })
	if !errors.Is(err, context.Canceled) || started {
		t.Fatalf("a cancelled prefill returned %v, started %v", err, started)
	}
	if lm.tokensPrefilled.Load() != 0 {
		t.Fatalf("the cancelled prefill counted %d positions", lm.tokensPrefilled.Load())
	}
	got := generate(t, e, GenerateOptions{ModelID: "m", Prompt: idsPrompt(p), MaxTokens: 8})
	fresh, _ := ttftEngine(t, smallModel, Config{NoMemCache: true, SessionPool: -1}, LoadOptions{})
	want := generate(t, fresh, GenerateOptions{ModelID: "m", Prompt: idsPrompt(p), MaxTokens: 8})
	if !slices.Equal(want.ids, got.ids) {
		t.Fatalf("after a cancelled prefill the pooled State answered %v, a fresh one %v", got.ids, want.ids)
	}
	if st := e.MemCacheStats(lm); st.StatesReused != 1 {
		t.Fatalf("%+v: the cancelled request's State was not pooled", st)
	}
}

// TestACancelledRequestDoesNoPromptWorkOnAnyPath: a request whose context is
// done before Generate sends no GenerateStarted and returns the context's
// error on each path a prompt can take -- the step loop's row, the prefill
// alone, and speculation -- and computes no position.
func TestACancelledRequestDoesNoPromptWorkOnAnyPath(t *testing.T) {
	for _, arm := range []struct {
		name string
		cfg  Config
		spec *Speculation
	}{
		{"row", Config{NoMemCache: true}, nil},
		{"alone", Config{NoMemCache: true, MaxBatchRows: 1}, nil},
		{"speculation", Config{NoMemCache: true, MaxBatchRows: 1}, &Speculation{Enabled: true, Draft: 4}},
	} {
		t.Run(arm.name, func(t *testing.T) {
			e, lm := ttftEngine(t, smallModel, arm.cfg, LoadOptions{})
			if (lm.loop != nil) != (arm.cfg.MaxBatchRows != 1) {
				t.Fatalf("the %s arm has step loop %v", arm.name, lm.loop != nil)
			}
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			for range 20 {
				started := false
				err := e.Generate(ctx, GenerateOptions{ModelID: "m", Prompt: idsPrompt(longPrompt(lm.m)), MaxTokens: 8,
					Speculation: arm.spec}, func(ev Event) error { started = started || ev.Kind == EventStarted; return nil })
				if !errors.Is(err, context.Canceled) || started {
					t.Fatalf("a cancelled request returned %v, started %v", err, started)
				}
			}
			if n := lm.tokensPrefilled.Load(); n != 0 {
				t.Fatalf("cancelled requests computed %d positions", n)
			}
		})
	}
}

// TestACancelledRowLeavesItsPromptUnrun: the step loop's half of a cancel,
// which the check at Generate's entry cannot reach -- a row the loop admitted
// before its request saw the context go (the request's select between
// admission and cancellation picks either when both are ready). Its prompt,
// begun or not, is not fed; it pushes no GenerateStarted; its State is reset;
// and it ends with errPromptCancelled, which generateBatched turns into the
// context's error.
func TestACancelledRowLeavesItsPromptUnrun(t *testing.T) {
	e, lm := ttftEngine(t, smallModel, Config{NoMemCache: true}, LoadOptions{})
	lp := lm.loop
	if lp == nil {
		t.Fatal("the model has no step loop")
	}
	p := longPrompt(lm.m)
	for _, begun := range []bool{false, true} {
		s, err := e.CreateSession(SessionOptions{ModelID: "m"})
		if err != nil {
			t.Fatal(err)
		}
		fed := 0
		if begun {
			fed = 4
			if _, err := s.st.Prefill(p[:fed]); err != nil {
				t.Fatal(err)
			}
		}
		r := &row{s: s, ids: p, fed: fed, begun: begun, fresh: true, reset: true,
			stream: newStreamText(lm.m.Vocab.NewChatStream().Next, nil),
			done:   make(chan struct{}), notify: make(chan struct{}, 1)}
		r.cancelled.Store(true)
		// Nothing is waiting, so the loop goroutine is parked in awaitWork
		// and its rows are this goroutine's.
		lp.rows = append(lp.rows, r)
		if units := lp.promptUnits(nil, model.MaxStepRows); len(units) != 0 {
			t.Fatalf("begun %v: a cancelled row was fed %d unit(s)", begun, len(units))
		}
		select {
		case <-r.done:
		default:
			t.Fatalf("begun %v: the cancelled row was not finished", begun)
		}
		for _, ev := range r.take() {
			if ev.Kind == EventStarted {
				t.Fatalf("begun %v: a cancelled row sent GenerateStarted", begun)
			}
		}
		if !errors.Is(r.result.err, errPromptCancelled) || r.result.reason != FinishCancelled {
			t.Fatalf("begun %v: the row ended %v, %v", begun, r.result.reason, r.result.err)
		}
		if pos := s.st.Pos(); pos != 0 {
			t.Fatalf("begun %v: the cancelled row left its State at position %d", begun, pos)
		}
		e.CloseSession(s.id)
	}
}

// TestARequestsPriorityFavoursItsModel: priority "high" makes the request's
// model the favoured one with priority on, so it holds all but an eighth of
// the host budget; "normal" changes nothing; anything else is the caller's
// error.
func TestARequestsPriorityFavoursItsModel(t *testing.T) {
	e, lm := ttftEngine(t, smallModel, Config{}, LoadOptions{})
	other, err := e.LoadModel(LoadOptions{Path: lm.path, ModelID: "other"})
	if err != nil {
		t.Fatal(err)
	}
	even := lm.Budget()
	if even != other.Budget() {
		t.Fatalf("two unpinned models split %d and %d", even, other.Budget())
	}
	generate(t, e, GenerateOptions{ModelID: "other", Prompt: Prompt{Kind: PromptText, Text: story}, MaxTokens: 2, Priority: PriorityNormal})
	if e.priority || other.Budget() != even {
		t.Fatal("a normal-priority request moved the budget")
	}
	generate(t, e, GenerateOptions{ModelID: "other", Prompt: Prompt{Kind: PromptText, Text: story}, MaxTokens: 2, Priority: PriorityHigh})
	if !e.priority || e.favored != "other" || other.Budget() <= lm.Budget() {
		t.Fatalf("a high-priority request left priority %v, favoured %q, budgets %d (it) and %d (the other)",
			e.priority, e.favored, other.Budget(), lm.Budget())
	}
	err = e.Generate(context.Background(), GenerateOptions{ModelID: "m", Prompt: Prompt{Kind: PromptText, Text: story},
		MaxTokens: 2, Priority: "urgent"}, func(Event) error { return nil })
	if !errors.Is(err, ErrInvalid) {
		t.Fatalf("priority %q returned %v", "urgent", err)
	}
}

// TestAnthropicCacheUsageIsReported: two identical Anthropic requests with a
// cache_control breakpoint; the first reports its prompt as written to the
// cache, the second as read from it, and both bodies say the same thing.
func TestAnthropicCacheUsageIsReported(t *testing.T) {
	e, _ := ttftEngine(t, chatModel, Config{}, LoadOptions{})
	c := serveEngine(t, e)
	body := `{"model":"m","max_tokens":6,"temperature":0,"system":[{"type":"text","text":"` +
		strings.Repeat("You are a careful assistant who answers briefly. ", 12) +
		`","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":"Say hi."}]}`
	type usage struct {
		Input   int `json:"input_tokens"`
		Read    int `json:"cache_read_input_tokens"`
		Created int `json:"cache_creation_input_tokens"`
	}
	var out [2]struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
		Usage usage `json:"usage"`
	}
	for i := range out {
		r, err := http.Post(c.url+"/v1/messages", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		b, _ := io.ReadAll(r.Body)
		r.Body.Close()
		if r.StatusCode != http.StatusOK {
			t.Fatalf("status %d: %s", r.StatusCode, b)
		}
		if err := json.Unmarshal(b, &out[i]); err != nil {
			t.Fatal(err)
		}
	}
	if u := out[0].Usage; u.Read != 0 || u.Created != u.Input {
		t.Fatalf("the first request's usage %+v: want nothing read and the whole prompt written", u)
	}
	if u := out[1].Usage; u.Read == 0 || u.Read+u.Created != u.Input {
		t.Fatalf("the second request's usage %+v: want the prefix read and the rest written", u)
	}
	if len(out[0].Content) == 0 || len(out[1].Content) == 0 || out[0].Content[0].Text != out[1].Content[0].Text {
		t.Fatalf("the restored request answered differently: %+v vs %+v", out[0].Content, out[1].Content)
	}
}

// TestTheMemCacheLayersOverThePromptStore: a page offered goes to memory and
// disk; once memory has dropped it, a Get finds it on disk and copies it back
// up, so the next Get is a memory hit.
func TestTheMemCacheLayersOverThePromptStore(t *testing.T) {
	disk, err := model.NewFileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	mem := model.NewBoundedMemStore(4096)
	defer mem.Close()
	l := &layered{mem: mem, disk: disk}
	page := bytes.Repeat([]byte{3}, 4096)
	if err := l.Set("a", 0, 0, bytes.NewReader(page)); err != nil {
		t.Fatal(err)
	}
	// A second page pushes the first out of memory.
	if err := l.Set("b", 0, 0, bytes.NewReader(page)); err != nil {
		t.Fatal(err)
	}
	if err := mem.Get("a", 0, 0, io.Discard); !errors.Is(err, model.ErrNoPage) {
		t.Fatalf("memory still holds the first page (%v): the bound never bit", err)
	}
	var got bytes.Buffer
	if err := l.Get("a", 0, 0, &got); err != nil || !bytes.Equal(got.Bytes(), page) {
		t.Fatalf("the layered Get did not fall through to disk: %v", err)
	}
	before := mem.Stats().Hits
	if err := l.Get("a", 0, 0, io.Discard); err != nil || mem.Stats().Hits != before+1 {
		t.Fatalf("a disk hit was not copied up to memory (%v)", err)
	}
	if err := l.Get("c", 0, 0, io.Discard); !errors.Is(err, model.ErrNoPage) {
		t.Fatalf("a page neither holds returned %v", err)
	}
}

// TestGrammarAndSpeculationMeetTheMemCache: a grammar-constrained generate
// runs alone and still restores from the model's memory cache and seals into
// it (a repeat restores, with the same answer); a speculative one prefills
// through its Speculator, which neither restores nor seals, so it is run
// without the cache -- it restores nothing even after the same prompt ran.
func TestGrammarAndSpeculationMeetTheMemCache(t *testing.T) {
	sm := structuredModels[1]
	e, _ := ttftEngine(t, sm.file, Config{}, LoadOptions{})
	prompt := Prompt{Kind: PromptText, Text: sm.prompt}
	g := GenerateOptions{ModelID: "m", Prompt: prompt, MaxTokens: 16, Grammar: `root ::= "zebra quantum lettuce"` + "\n"}
	first := generate(t, e, g)
	again := generate(t, e, g)
	if first.started != 0 || again.started == 0 {
		t.Fatalf("a grammar generate restored %d, then %d: the repeat should restore", first.started, again.started)
	}
	if !slices.Equal(first.ids, again.ids) {
		t.Fatalf("the restored grammar generate answered %v, the first %v", again.ids, first.ids)
	}
	sp := generate(t, e, GenerateOptions{ModelID: "m", Prompt: prompt, MaxTokens: 8,
		Speculation: &Speculation{Enabled: true}})
	if sp.started != 0 {
		t.Fatalf("a speculative generate restored %d positions through a Speculator that does not restore", sp.started)
	}
}

// sampleTrace is the distributions one generate sampled from: the softmax at
// the sampler's temperature of each step's logits.
type sampleTrace struct {
	mu    sync.Mutex
	probs [][]float64
}

func (s *sampleTrace) add(temp float64, lg []float32, next int32) {
	if temp <= 0 {
		return
	}
	mx := lg[0]
	for _, v := range lg {
		mx = max(mx, v)
	}
	p := make([]float64, len(lg))
	var sum float64
	for i, v := range lg {
		p[i] = math.Exp(float64(v-mx) / temp)
		sum += p[i]
	}
	for i := range p {
		p[i] /= sum
	}
	s.mu.Lock()
	s.probs = append(s.probs, p)
	s.mu.Unlock()
}

// maxSampledTV is the most two engines' distributions may part by at a step
// where they drew the same number: a seeded draw with the same u picks a
// different token only inside that much probability mass. The V100 shows
// 0.001 to 0.01 between a restored and a cold prefix (a KV scaled by 1.15 gives 0.17).
const maxSampledTV = 0.05

// compare holds the warm trace to the cold one, up to the first step their
// tokens part: one draw per token in both arms, and at each step the total
// variation distance between the two distributions below maxSampledTV.
func (c *sampleTrace) compare(t *testing.T, w *sampleTrace, cold, warm []int32) {
	t.Helper()
	// One Sample per token, and one more when the last was an end of
	// generation, which the answer leaves out.
	for name, tr := range map[string]struct {
		n, ids int
	}{"cold": {len(c.probs), len(cold)}, "warm": {len(w.probs), len(warm)}} {
		if tr.n == 0 || tr.n < tr.ids || tr.n > tr.ids+1 {
			t.Fatalf("the %s arm sampled %d times for %d tokens: a restored prefix must draw once per token", name, tr.n, tr.ids)
		}
	}
	for s := range min(len(cold), len(warm), len(c.probs), len(w.probs)) {
		var tv float64
		for i := range c.probs[s] {
			tv += math.Abs(c.probs[s][i] - w.probs[s][i])
		}
		tv /= 2
		t.Logf("sampled step %d: total variation %.4f, tokens %d / %d", s, tv, cold[s], warm[s])
		if tv > maxSampledTV {
			t.Fatalf("sampled step %d: the restored prefix's distribution is %.4f away from the cold one (bound %v)", s, tv, maxSampledTV)
		}
		if cold[s] != warm[s] {
			return // the draws part here, inside the mass the distributions disagree by; later steps condition on different tokens
		}
	}
}
