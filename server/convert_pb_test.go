package server

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"net/http/httptest"
	"testing"

	"connectrpc.com/connect"
	"google.golang.org/protobuf/proto"

	"github.com/jitllm/jitllm/model"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// The Go <-> protobuf mappings and the error codes, gated table by table: a
// value that falls through to UNSPECIFIED or Internal reads as a working
// answer, so every arm is asserted.

// TestEveryEngineErrorHasItsCode. With any arm of connectErr removed its row
// comes back Internal.
func TestEveryEngineErrorHasItsCode(t *testing.T) {
	if connectErr(nil) != nil {
		t.Fatal("connectErr(nil) is not nil: a success would read as a failure")
	}
	already := connect.NewError(connect.CodeUnavailable, errors.New("x"))
	if got := connectErr(fmt.Errorf("wrapped: %w", already)); got != already {
		t.Fatalf("a connect error inside a wrap was re-coded: %v", got)
	}
	for _, tc := range []struct {
		err  error
		want connect.Code
	}{
		{fmt.Errorf("%w: model %q", ErrNotFound, "m"), connect.CodeNotFound},
		{fmt.Errorf("open x.jlm: %w", fs.ErrNotExist), connect.CodeNotFound},
		{fmt.Errorf("%w: model id %q", ErrExists, "m"), connect.CodeAlreadyExists},
		{fmt.Errorf("%w: no prompt", ErrInvalid), connect.CodeInvalidArgument},
		{fmt.Errorf("%w: 1 open session", ErrInUse), connect.CodeFailedPrecondition},
		{ErrQueueTimeout, connect.CodeResourceExhausted},
		{context.Canceled, connect.CodeCanceled},
		{context.DeadlineExceeded, connect.CodeDeadlineExceeded},
		{errors.New("a fault"), connect.CodeInternal},
	} {
		wantCode(t, tc.err.Error(), connectErr(tc.err), tc.want)
	}
	// A GGUF handed to Open names its path in metadata, so a client can offer
	// the conversion as an action rather than parse the message.
	ce := wantCode(t, "NotConvertedError", connectErr(fmt.Errorf("load: %w", &model.NotConvertedError{Path: "/m/x.gguf"})),
		connect.CodeFailedPrecondition)
	if got := ce.Meta().Get("jitllm-not-converted"); got != "/m/x.gguf" {
		t.Fatalf("jitllm-not-converted metadata %q, want the GGUF's path", got)
	}
	wantCode(t, "invalid()", invalid("x %d", 1), connect.CodeInvalidArgument)
	wantCode(t, "unimplemented()", unimplemented("x"), connect.CodeUnimplemented)
}

// TestEveryEnumValueHasItsWireValue: each Go value maps to its own proto
// value, and anything else is UNSPECIFIED rather than a neighbour.
func TestEveryEnumValueHasItsWireValue(t *testing.T) {
	backends := map[string]v1.Backend{
		"cpu": v1.Backend_BACKEND_CPU, "cuda": v1.Backend_BACKEND_CUDA,
		"vulkan": v1.Backend_BACKEND_VULKAN, "metal": v1.Backend_BACKEND_METAL,
		"opencl": v1.Backend_BACKEND_UNSPECIFIED, "": v1.Backend_BACKEND_UNSPECIFIED,
	}
	for in, want := range backends {
		if got := pbBackend(in); got != want {
			t.Errorf("pbBackend(%q) = %v, want %v", in, got, want)
		}
	}
	kinds := map[DeviceKind]v1.DeviceKind{
		KindHost: v1.DeviceKind_DEVICE_KIND_HOST, KindDiscrete: v1.DeviceKind_DEVICE_KIND_DISCRETE,
		KindIntegrated: v1.DeviceKind_DEVICE_KIND_INTEGRATED, KindUnspecified: v1.DeviceKind_DEVICE_KIND_UNSPECIFIED,
		DeviceKind(99): v1.DeviceKind_DEVICE_KIND_UNSPECIFIED,
	}
	for in, want := range kinds {
		if got := pbKind(in); got != want {
			t.Errorf("pbKind(%d) = %v, want %v", in, got, want)
		}
	}
	modes := map[ExecutionMode]v1.ExecutionMode{
		ExecutionParallel: v1.ExecutionMode_EXECUTION_MODE_PARALLEL, ExecutionSerialised: v1.ExecutionMode_EXECUTION_MODE_SERIALISED,
		ExecutionUnspecified: v1.ExecutionMode_EXECUTION_MODE_UNSPECIFIED,
	}
	for in, want := range modes {
		if got := pbExecution(in); got != want {
			t.Errorf("pbExecution(%d) = %v, want %v", in, got, want)
		}
	}
	reasons := map[FinishReason]v1.FinishReason{
		FinishStop: v1.FinishReason_FINISH_REASON_STOP, FinishMaxTokens: v1.FinishReason_FINISH_REASON_MAX_TOKENS,
		FinishEOS: v1.FinishReason_FINISH_REASON_EOS, FinishCancelled: v1.FinishReason_FINISH_REASON_CANCELLED,
		FinishError: v1.FinishReason_FINISH_REASON_ERROR, FinishUnspecified: v1.FinishReason_FINISH_REASON_UNSPECIFIED,
	}
	for in, want := range reasons {
		if got := pbFinish(in); got != want {
			t.Errorf("pbFinish(%d) = %v, want %v", in, got, want)
		}
	}
}

// TestEverySamplingFieldReachesTheSampler: an unset field leaves the
// sampler's zero (greedy, filters off), and a set one arrives unchanged.
func TestEverySamplingFieldReachesTheSampler(t *testing.T) {
	if pbSampling(nil) != nil {
		t.Fatal("no sampling params must be no override, so the session's own sampler runs")
	}
	f32 := func(v float32) *float32 { return &v }
	i32 := func(v int32) *int32 { return &v }
	seed := uint64(7)
	got := pbSampling(&v1.SamplingParams{
		Temperature: f32(0.5), TopP: f32(0.25), TopK: i32(40), MinP: f32(0.125),
		Seed: &seed, RepeatPenalty: f32(1.5), RepeatLastN: i32(32),
	})
	if got.Temp != 0.5 || got.TopP != 0.25 || got.TopK != 40 || got.MinP != 0.125 || got.Seed != 7 ||
		got.RepeatPen != 1.5 || got.RepeatLastN != 32 {
		t.Fatalf("sampling reached the engine as %+v", *got)
	}
	if got := pbSampling(&v1.SamplingParams{}); got.Temp != 0 || got.TopP != 0 || got.TopK != 0 || got.MinP != 0 ||
		got.Seed != 0 || got.RepeatPen != 0 || got.RepeatLastN != 0 {
		t.Fatalf("an empty params message set fields: %+v", *got)
	}
}

// TestGenerateOptionsMapsEveryWireField: the Connect request becomes the one
// GenerateOptions the shims build too.
func TestGenerateOptionsMapsEveryWireField(t *testing.T) {
	o, err := generateOptions(&v1.GenerateRequest{
		SessionId: "s", MaxTokens: 9, Stop: []string{"END"}, EchoPrompt: true,
		QueueTimeoutMillis: 1500, Prompt: ids(3, 4, 5), IgnoreEos: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if o.SessionID != "s" || o.MaxTokens != 9 || o.Stop[0] != "END" || !o.Echo ||
		o.QueueTimeout.Milliseconds() != 1500 || o.Prompt.Kind != PromptIDs || len(o.Prompt.IDs) != 3 ||
		!o.IgnoreEOS {
		t.Fatalf("generateOptions gave %+v", o)
	}
	// A continuation with no new prompt is a legal request at this layer.
	o, err = generateOptions(&v1.GenerateRequest{SessionId: "s", ContinueSession: true})
	if err != nil || o.Prompt.Kind != PromptNone || !o.Continue {
		t.Fatalf("a bare continuation: %+v %v", o, err)
	}
	_, err = generateOptions(&v1.GenerateRequest{SessionId: "s"})
	wantCode(t, "a generate with no prompt", err, connect.CodeInvalidArgument)
}

// TestHumanBytesAndCapacityDeclines: the two string helpers a client shows.
func TestHumanBytesAndCapacityDeclines(t *testing.T) {
	for in, want := range map[uint64]string{
		0: "0 B", 1023: "1023 B", 1024: "1.00 KiB", 1536 << 10: "1.50 MiB", 3 << 30: "3.00 GiB", 5 << 40: "5.00 TiB",
	} {
		if got := humanBytes(in); got != want {
			t.Errorf("humanBytes(%d) = %q, want %q", in, got, want)
		}
	}
	if b := pbBytes(2048); b.GetBytes() != 2048 || b.GetHuman() != "2.00 KiB" {
		t.Fatalf("pbBytes(2048) = %+v", b)
	}
	// Only "does not fit" is a capacity decline; anything else is a kernel
	// that is owed, and must not be filed under memory.
	for why, want := range map[string]bool{
		"over the device budget": true, "out of memory": true, "the block does not fit": true,
		"No room left": true, "partial rotary is not lowered": false, "": false,
	} {
		if got := isCapacityDecline(why); got != want {
			t.Errorf("isCapacityDecline(%q) = %v, want %v", why, got, want)
		}
	}
}

// TestAnSSEErrorFrameIsNamedAndTyped: the only failure report a stream has
// once its status is 200.
func TestAnSSEErrorFrameIsNamedAndTyped(t *testing.T) {
	w := httptest.NewRecorder()
	s, err := newSSE(w)
	if err != nil {
		t.Fatal(err)
	}
	s.sendError(errors.New("the card fell off the bus"))
	s.done()
	got := w.Body.String()
	want := "event: error\ndata: {\"error\":{\"message\":\"the card fell off the bus\",\"type\":\"server_error\"}}\n\n" +
		"data: [DONE]\n\n"
	if got != want {
		t.Fatalf("the stream reads\n%q\nwant\n%q", got, want)
	}
	if ct := w.Header().Get("Content-Type"); ct != "text/event-stream" || w.Header().Get("X-Accel-Buffering") != "no" {
		t.Fatalf("headers %v: a proxy that buffers defeats the flush", w.Header())
	}
}

// TestValidTextMarshals: a completion cut inside a character (its token
// limit, a lone byte token) reaches the wire, which refuses invalid UTF-8;
// validText makes it marshal and leaves valid text alone. The cut text
// itself is the violation.
func TestValidTextMarshals(t *testing.T) {
	cut := "caf\xc3"
	if _, err := proto.Marshal(&v1.CompleteResponse{Text: cut}); err == nil {
		t.Fatal("a text cut inside a character marshalled: the violation proves nothing")
	}
	if _, err := proto.Marshal(&v1.CompleteResponse{Text: validText(cut)}); err != nil {
		t.Fatalf("validText(%q) = %q does not marshal: %v", cut, validText(cut), err)
	}
	if got := validText("caf\u00e9"); got != "caf\u00e9" {
		t.Fatalf("validText changed valid text: %q", got)
	}
	if ev := pbEvent(Event{Kind: EventToken, Token: &Token{Text: cut}}); ev.GetToken().GetText() != validText(cut) {
		t.Fatalf("a token event carries %q", ev.GetToken().GetText())
	}
}
