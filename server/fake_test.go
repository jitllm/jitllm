package server

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// fakeBackend is a scripted token stream standing in for the engine, so the
// streaming gates (framing, field mapping, flushing) run without loading a
// model or occupying cores. What it cannot gate is listed in
// e2e_test.go.
type fakeBackend struct {
	tokens []string
	reason FinishReason
	stopAt string

	// beforeToken runs before each token is emitted. The streaming gates use
	// it to hold the server between frames and prove the client saw the
	// earlier one, which tells a flushed stream from a buffered response.
	beforeToken func(i int)

	failWith error

	// decide answers /v1/systemone; nil refuses it.

	mu   sync.Mutex
	last GenerateOptions
	// all is every generate's options, in the order they came.
	all []GenerateOptions
	ids int
}

func (f *fakeBackend) Generate(ctx context.Context, o GenerateOptions, emit func(Event) error) error {
	f.mu.Lock()
	f.last = o
	f.all = append(f.all, o)
	f.mu.Unlock()

	if f.failWith != nil {
		return f.failWith
	}
	// A request for several choices (n) gets the script once for each.
	for ch := 0; ch < max(len(o.Seeds), 1); ch++ {
		if err := f.choice(ctx, o, ch, func(ev Event) error {
			ev.Choice = ch
			return emit(ev)
		}); err != nil {
			return err
		}
	}
	return nil
}

func (f *fakeBackend) choice(ctx context.Context, o GenerateOptions, ch int, emit func(Event) error) error {
	// A prompt asked for its progress runs once, before its first choice: two
	// of its seven positions restored, then a chunk of three, then the rest.
	if o.Progress && ch == 0 {
		for _, done := range []int{2, 5, 7} {
			if err := emit(Event{Kind: EventProgress, Progress: &Progress{Total: 7, Cached: 2, Done: done,
				Elapsed: time.Duration(done) * time.Millisecond}}); err != nil {
				return err
			}
		}
	}
	if err := emit(Event{Kind: EventStarted, Started: &Started{
		SessionID:    "sess-fake",
		ModelID:      o.ModelID,
		PromptTokens: 7,
		QueuedFor:    1500 * time.Millisecond,
		QueueDepth:   2,
		DeviceBlocks: 22,
		CPUBlocks:    10,
		DeviceIDs:    []string{"cuda:0"},
		Prefill:      40 * time.Millisecond,
	}}); err != nil {
		return err
	}
	n := 0
	for i, t := range f.tokens {
		if f.beforeToken != nil {
			f.beforeToken(i)
		}
		if ctx.Err() != nil {
			break
		}
		if err := emit(Event{Kind: EventToken, Token: &Token{ID: int32(100 + i), Text: t, Index: i}}); err != nil {
			return err
		}
		n++
	}
	return emit(Event{Kind: EventFinished, Finished: &Finished{
		Reason:           f.reason,
		StopMatched:      f.stopAt,
		PromptTokens:     7,
		CompletionTokens: n,
		Prefill:          40 * time.Millisecond,
		Decode:           120 * time.Millisecond,
		TokensPerSecond:  33.3,
		BytesPerToken:    816010912,
		Position:         7 + n,
	}})
}

func (f *fakeBackend) ListLoaded() []ModelSummary {
	return []ModelSummary{{ID: "m-1", Name: "tinyllama", LoadedAt: time.Unix(1700000000, 0), MaxModelLen: 2048, Loaded: true}}
}

func (f *fakeBackend) ListAvailable() []ModelSummary { return nil }

func (f *fakeBackend) BindTarget(o *GenerateOptions, sessionID, modelName string) error {
	if sessionID != "" {
		o.SessionID = sessionID
		return nil
	}
	if modelName == "" {
		return fmt.Errorf("model is required")
	}
	o.ModelID = modelName
	return nil
}

func (f *fakeBackend) NextID(prefix string) string {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.ids++
	return fmt.Sprintf("%s-%d", prefix, f.ids)
}

func (f *fakeBackend) CancelSession(id string) (bool, error) {
	if id == "" {
		return false, fmt.Errorf("%w: session %q", ErrNotFound, id)
	}
	return true, nil
}

func (f *fakeBackend) Embed(ctx context.Context, o EmbedOptions) (*EmbedResult, error) {
	return nil, fmt.Errorf("%w: the scripted backend embeds nothing", ErrInvalid)
}

func (f *fakeBackend) Decide(ctx context.Context, o DecideOptions) (*DecideResult, error) {
	return nil, fmt.Errorf("%w: the scripted backend decides nothing", ErrInvalid)
}

// took is every generate's options since the last call, in order.
func (f *fakeBackend) took() []GenerateOptions {
	f.mu.Lock()
	defer f.mu.Unlock()
	all := f.all
	f.all = nil
	return all
}

func (f *fakeBackend) opts() GenerateOptions {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.last
}
