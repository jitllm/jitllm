package server

import (
	"context"
	"testing"

	"github.com/jitllm/jitllm/engine/model"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// contextFullSeq is the session the context-full gates generate in: short,
// so running to the end of it takes a moment.
const contextFullSeq = 64

// runsToTheEndOfTheContext asks a session of contextFullSeq positions for a
// reply that never ends on its own (ignore_eos), with each max_tokens in asks,
// and wants every one to stop exactly at the end of the context with
// FINISH_REASON_MAX_TOKENS rather than an error: 0 (none asked) and more than
// the context holds are both cut to the room the prompt left.
func runsToTheEndOfTheContext(t *testing.T, c clients, lm *LoadedModel, wantBatched bool) {
	t.Helper()
	conv, err := lm.m.ChatIDsTools([]model.ChatMessage{{Role: "user", Content: ignoreEOSQuestion}}, nil, true)
	if err != nil {
		t.Fatal(err)
	}
	room := contextFullSeq - len(conv)
	for _, asked := range []int32{0, 10 * contextFullSeq} {
		if _, err := c.session.CreateSession(context.Background(), req(&v1.CreateSessionRequest{
			ModelId: "inst", SessionId: "full", MaxSeq: contextFullSeq})); err != nil {
			t.Fatal(err)
		}
		r := complete(c, &v1.GenerateRequest{SessionId: "full", Prompt: ids(conv...),
			MaxTokens: asked, IgnoreEos: true})
		if r.err != nil {
			t.Fatalf("max_tokens %d in a %d-position session: %v", asked, contextFullSeq, r.err)
		}
		if r.started.GetBatched() != wantBatched {
			t.Fatalf("max_tokens %d: batched %v, want %v", asked, r.started.GetBatched(), wantBatched)
		}
		if got := r.finished.GetReason(); got != v1.FinishReason_FINISH_REASON_MAX_TOKENS || len(r.ids) != room {
			t.Fatalf("max_tokens %d: %v after %d tokens, want FINISH_REASON_MAX_TOKENS after %d "+
				"(a %d-position session, a %d-token prompt)", asked, got, len(r.ids), room, contextFullSeq, len(conv))
		}
		if _, err := c.session.CloseSession(context.Background(), req(&v1.CloseSessionRequest{SessionId: "full"})); err != nil {
			t.Fatal(err)
		}
	}
}

// TestNoMaxTokensRunsToTheEndOfTheContext: a generate with no max_tokens runs
// until the model ends its reply or the session's context is full, and ends
// there cleanly; a limit past the context is cut to it. On the host the
// generate is a row of the model's step loop too (batch.go).
func TestNoMaxTokensRunsToTheEndOfTheContext(t *testing.T) {
	_, lm, c := loadedEngine(t, instructModel, "inst", LoadOptions{})
	runsToTheEndOfTheContext(t, c, lm, true)
}

// TestNoMaxTokensRunsToTheEndOfTheContextBatched is the same for a generate
// that runs as a row of its model's step loop (batch.go).
func TestNoMaxTokensRunsToTheEndOfTheContextBatched(t *testing.T) {
	path := modelPath(t, instructModel)
	e := New(admitConfig(Config{Probe: oneCardProbe, Version: "test"}))
	t.Cleanup(e.Close)
	lm, err := e.LoadModel(LoadOptions{Path: path, ModelID: "inst", DeviceIDs: []string{"gpu:0"}, Sessions: 1})
	if err != nil {
		t.Skipf("NO DEVICE: loading %s onto -devices gpu:0 failed (%v) -- this gate proved nothing", instructModel, err)
	}
	if lm.loop == nil {
		t.Fatal("a model loaded onto a device has no step loop")
	}
	requireWholeOnDevice(t, e, lm)
	runsToTheEndOfTheContext(t, serveEngine(t, e), lm, true)
}
