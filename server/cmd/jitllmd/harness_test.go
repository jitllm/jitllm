package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/server"
	"github.com/jitllm/jitllm/server/gen/jitllm/v1/jitllmv1connect"
)

// The harness these gates share.
//
// Every gate in this package drives a real in-process server over real HTTP
// through the generated stubs, so it exercises protobuf encoding, the oneof
// mapping and connect-go's stream framing. Two shapes of server:
//
//	engineServer  the real server.Engine behind server.Handler(). It answers
//	              every read verb and every refusal without a model on disk.
//
//	scriptServer  the real generated InferenceServiceHandler over a scripted
//	              Backend, so the streaming gates run without loading a model
//	              or occupying cores.

// ---------------------------------------------------------------- servers

// pathCount records which RPC each request actually hit. Generate and
// Complete return the same text, so counting the invoked procedure is the
// only way to tell a streaming client from one that called Complete.
type pathCount struct {
	mu sync.Mutex
	n  map[string]int
}

func (p *pathCount) wrap(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p.mu.Lock()
		if p.n == nil {
			p.n = map[string]int{}
		}
		p.n[r.URL.Path]++
		p.mu.Unlock()
		h.ServeHTTP(w, r)
	})
}

func (p *pathCount) get(suffix string) int {
	p.mu.Lock()
	defer p.mu.Unlock()
	total := 0
	for k, v := range p.n {
		if strings.HasSuffix(k, suffix) {
			total += v
		}
	}
	return total
}

func (p *pathCount) paths() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	var out []string
	for k, v := range p.n {
		out = append(out, fmt.Sprintf("%s x%d", k, v))
	}
	return out
}

// engineServer stands up the REAL engine on a real socket and hands back the
// address in the form a person would type.
func engineServer(t *testing.T, cfg server.Config) (string, *server.Engine) {
	t.Helper()
	if cfg.ModelDir == "" {
		cfg.ModelDir = t.TempDir()
	}
	if cfg.Probe == nil {
		cfg.Probe = func() ([]server.DeviceInfo, error) { return nil, nil }
	}
	cfg.Admit = testdevices.Admit()
	e := server.New(cfg)
	t.Cleanup(e.Close)
	s := httptest.NewServer(e.Handler())
	t.Cleanup(s.Close)
	return s.URL, e
}

// scriptServer mounts the real generated handlers: InferenceService over a
// scripted Backend, and ModelService over an empty real engine so the client's
// own model probe has somebody to talk to.
func scriptServer(t *testing.T, b server.Backend) (string, *pathCount) {
	t.Helper()
	e := server.New(server.Config{Admit: testdevices.Admit(),
		ModelDir: t.TempDir(),
		Probe:    func() ([]server.DeviceInfo, error) { return nil, nil },
	})
	t.Cleanup(e.Close)

	mux := http.NewServeMux()
	mux.Handle(jitllmv1connect.NewInferenceServiceHandler(&server.InferenceService{B: b}))
	mux.Handle(jitllmv1connect.NewModelServiceHandler(&server.ModelService{E: e}))

	pc := &pathCount{}
	s := httptest.NewServer(pc.wrap(mux))
	t.Cleanup(s.Close)
	return s.URL, pc
}

// ---------------------------------------------------------------- backend

// scripted is a server.Backend that emits a fixed token sequence and can be
// held between tokens.
type scripted struct {
	tokens []string
	reason server.FinishReason
	fail   error

	// before runs immediately before token i is emitted. The streaming gate
	// uses it to hold token i until the client has printed token i-1.
	before func(i int)

	mu   sync.Mutex
	last server.GenerateOptions
	n    int
}

func (s *scripted) Generate(ctx context.Context, o server.GenerateOptions, emit func(server.Event) error) error {
	s.mu.Lock()
	s.last = o
	s.mu.Unlock()
	if s.fail != nil {
		return s.fail
	}
	if err := emit(server.Event{Kind: server.EventStarted, Started: &server.Started{
		SessionID:    "sess-scripted",
		ModelID:      o.ModelID,
		Ephemeral:    o.SessionID == "",
		PromptTokens: 7,
		QueuedFor:    1500 * time.Millisecond,
		QueueDepth:   2,
		DeviceBlocks: 22,
		HostBlocks:   10,
		DeviceIDs:    []string{"cuda:0"},
		Prefill:      40 * time.Millisecond,
	}}); err != nil {
		return err
	}
	n := 0
	for i, tk := range s.tokens {
		if s.before != nil {
			s.before(i)
		}
		if ctx.Err() != nil {
			break
		}
		if err := emit(server.Event{Kind: server.EventToken, Token: &server.Token{
			ID: int32(100 + i), Text: tk, Index: i,
		}}); err != nil {
			return err
		}
		n++
	}
	return emit(server.Event{Kind: server.EventFinished, Finished: &server.Finished{
		Reason:           s.reason,
		PromptTokens:     7,
		CompletionTokens: n,
		Prefill:          40 * time.Millisecond,
		Decode:           120 * time.Millisecond,
		TokensPerSecond:  33.3,
		BytesPerToken:    816010912,
		Position:         7 + n,
	}})
}

func (s *scripted) ListLoaded() []server.ModelSummary {
	return []server.ModelSummary{{ID: "m-1", Name: "tinyllama", Loaded: true}}
}

func (s *scripted) ListAvailable() []server.ModelSummary { return nil }

func (s *scripted) BindTarget(o *server.GenerateOptions, sessionID, modelName string) error {
	if sessionID != "" {
		o.SessionID = sessionID
		return nil
	}
	o.ModelID = modelName
	return nil
}

func (s *scripted) NextID(prefix string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.n++
	return fmt.Sprintf("%s-%d", prefix, s.n)
}

func (s *scripted) CancelSession(id string) (bool, error) { return true, nil }

func (s *scripted) Embed(ctx context.Context, o server.EmbedOptions) (*server.EmbedResult, error) {
	return nil, fmt.Errorf("%w: the scripted backend embeds nothing", server.ErrInvalid)
}

func (s *scripted) opts() server.GenerateOptions {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.last
}

// ---------------------------------------------------------------- writers

// syncBuffer is a concurrency-safe sink for a verb's stderr.
type syncBuffer struct {
	mu sync.Mutex
	b  strings.Builder
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

// notifyWriter records what the verb printed and announces each write. The
// streaming test's server waits on the announcement before the next token;
// hooking the write rather than the wire catches a client that reads a frame
// and sits on it.
type notifyWriter struct {
	mu sync.Mutex
	b  strings.Builder
	ch chan string
}

func newNotifyWriter(n int) *notifyWriter {
	return &notifyWriter{ch: make(chan string, n+8)}
}

func (w *notifyWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	w.b.Write(p)
	w.mu.Unlock()
	w.ch <- string(p)
	return len(p), nil
}

func (w *notifyWriter) String() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.b.String()
}

// ---------------------------------------------------------------- driving

// runVerb drives a verb in process, under a deadline, so a failing stream
// gate fails rather than hangs until the package timeout.
func runVerb(t *testing.T, fn func(context.Context, cli, []string) error, c cli, args []string, within time.Duration) error {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	done := make(chan error, 1)
	go func() { done <- fn(ctx, c, args) }()
	select {
	case err := <-done:
		return err
	case <-time.After(within):
		cancel()
		t.Fatalf("the verb did not return within %s", within)
		return nil
	}
}

// out drives a verb and returns stdout, failing on error.
func out(t *testing.T, fn func(context.Context, cli, []string) error, args ...string) string {
	t.Helper()
	var o, e syncBuffer
	if err := runVerb(t, fn, cli{out: &o, err: &e}, args, 20*time.Second); err != nil {
		t.Fatalf("%v\nstderr:\n%s", err, e.String())
	}
	return o.String()
}
