package server

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"connectrpc.com/connect"

	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
	"github.com/jitllm/jitllm/server/gen/jitllm/v1/jitllmv1connect"
)

// These gates go through the generated stubs over a real HTTP server, so they
// exercise the protobuf encoding, the oneof mapping and connect-go's stream
// framing, not just the handler methods.

func connectServer(t *testing.T, mount func(*http.ServeMux)) string {
	t.Helper()
	mux := http.NewServeMux()
	mount(mux)
	s := httptest.NewServer(mux)
	t.Cleanup(s.Close)
	return s.URL
}

// TestConnectGenerateStreamsStartedTokensFinishedInThatOrder: the started
// event (with queued_millis) must precede any token. With EventStarted
// emitted after the decode loop it fails with "event 0 is not `started`".
func TestConnectGenerateStreamsStartedTokensFinishedInThatOrder(t *testing.T) {
	f := &fakeBackend{tokens: []string{"a", "b", "c"}, reason: FinishEOS}
	url := connectServer(t, func(mux *http.ServeMux) {
		p, h := jitllmv1connect.NewInferenceServiceHandler(&InferenceService{B: f})
		mux.Handle(p, h)
	})
	cl := jitllmv1connect.NewInferenceServiceClient(http.DefaultClient, url)

	st, err := cl.Generate(context.Background(), connect.NewRequest(&v1.GenerateRequest{
		ModelId: "m-1",
		Prompt:  &v1.PromptInput{Input: &v1.PromptInput_Text{Text: "hello"}},
	}))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	defer st.Close()

	var kinds []string
	var text strings.Builder
	var started *v1.GenerateStarted
	var fin *v1.GenerateFinished
	for st.Receive() {
		switch ev := st.Msg().GetEvent().(type) {
		case *v1.GenerateResponse_Started:
			kinds = append(kinds, "started")
			started = ev.Started
		case *v1.GenerateResponse_Token:
			kinds = append(kinds, "token")
			text.WriteString(ev.Token.GetText())
		case *v1.GenerateResponse_Finished:
			kinds = append(kinds, "finished")
			fin = ev.Finished
		default:
			t.Fatalf("unknown event %T", ev)
		}
	}
	if err := st.Err(); err != nil {
		t.Fatalf("stream: %v", err)
	}
	if len(kinds) == 0 || kinds[0] != "started" {
		t.Fatalf("event 0 is not `started`: got %v", kinds)
	}
	if kinds[len(kinds)-1] != "finished" {
		t.Fatalf("the last event is %q, want finished", kinds[len(kinds)-1])
	}
	if got := text.String(); got != "abc" {
		t.Fatalf("streamed text %q, want %q", got, "abc")
	}
	// The queue wait must survive the encoding.
	if started.GetQueuedMillis() != 1500 || started.GetQueueDepthOnEntry() != 2 {
		t.Fatalf("started carries queued=%dms depth=%d, want 1500 and 2",
			started.GetQueuedMillis(), started.GetQueueDepthOnEntry())
	}
	if started.GetDeviceBlocks() != 22 || started.GetHostBlocks() != 10 {
		t.Fatalf("started carries the split %d/%d, want 22/10",
			started.GetDeviceBlocks(), started.GetHostBlocks())
	}
	if fin.GetReason() != v1.FinishReason_FINISH_REASON_EOS {
		t.Fatalf("finish reason %v, want EOS", fin.GetReason())
	}
	if fin.GetBytesPerToken() == 0 {
		t.Fatal("bytes_per_token is 0: a tok/s figure with no byte count behind it is unfalsifiable")
	}
}

// TestCompleteIsTheStreamingPathAccumulated: Complete must return the same
// text as the stream.
func TestCompleteIsTheStreamingPathAccumulated(t *testing.T) {
	script := []string{"The ", "answer", " is 4."}
	f := &fakeBackend{tokens: script, reason: FinishEOS}
	url := connectServer(t, func(mux *http.ServeMux) {
		p, h := jitllmv1connect.NewInferenceServiceHandler(&InferenceService{B: f})
		mux.Handle(p, h)
	})
	cl := jitllmv1connect.NewInferenceServiceClient(http.DefaultClient, url)

	resp, err := cl.Complete(context.Background(), connect.NewRequest(&v1.GenerateRequest{
		ModelId: "m-1",
		Prompt:  &v1.PromptInput{Input: &v1.PromptInput_Text{Text: "2+2?"}},
	}))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got, want := resp.Msg.GetText(), strings.Join(script, ""); got != want {
		t.Fatalf("Complete text %q, want %q", got, want)
	}
	if n := len(resp.Msg.GetTokenIds()); n != len(script) {
		t.Fatalf("Complete returned %d token ids, want %d", n, len(script))
	}
	if resp.Msg.GetStarted() == nil || resp.Msg.GetFinished() == nil {
		t.Fatal("Complete dropped the started/finished envelopes, so a non-streaming caller " +
			"cannot see the queue wait a streaming one can")
	}
}

// TestAChatPromptCarriesItsSystemFieldSeparately: the proto keeps `system`
// as an optional field distinct from the message list, so the engine knows
// which one the caller meant.
func TestAChatPromptCarriesItsSystemFieldSeparately(t *testing.T) {
	sys := "be terse"
	m := &v1.GenerateRequest{
		ModelId: "m-1",
		Prompt: &v1.PromptInput{Input: &v1.PromptInput_Chat{Chat: &v1.ChatPrompt{
			Messages:            []*v1.ChatMessage{{Role: "user", Content: "hi"}},
			System:              &sys,
			AddGenerationPrompt: true,
		}}},
	}
	o, err := generateOptions(m)
	if err != nil {
		t.Fatal(err)
	}
	if o.Prompt.Kind != PromptChat || o.Prompt.Chat == nil {
		t.Fatalf("prompt kind %v, want chat", o.Prompt.Kind)
	}
	if !o.Prompt.Chat.HasSystem || o.Prompt.Chat.System != sys {
		t.Fatalf("system reached the engine as (%v, %q), want (true, %q)",
			o.Prompt.Chat.HasSystem, o.Prompt.Chat.System, sys)
	}
	if len(o.Prompt.Chat.Messages) != 1 {
		t.Fatalf("the system field was folded into the message list: %d messages, want 1",
			len(o.Prompt.Chat.Messages))
	}

	// An absent system must not read as an empty one: a model whose
	// template branches on whether a system turn exists would render a
	// different prompt.
	m.Prompt.GetChat().System = nil
	o2, _ := generateOptions(m)
	if o2.Prompt.Chat.HasSystem {
		t.Fatal("an absent system field arrived as HasSystem=true; an empty system turn is " +
			"not the same prompt as no system turn")
	}
}

// TestHandingAGGUFToLoadReturnsTheConvertCommand: a GGUF must come back as
// FailedPrecondition carrying the convert command. It needs no file, since
// Open checks the extension first. Without the NotConvertedError arm in
// connectErr it fails with "code is internal, want failed_precondition".
func TestHandingAGGUFToLoadReturnsTheConvertCommand(t *testing.T) {
	e := New(admitConfig(Config{ModelDir: t.TempDir(), Probe: noProbe}))
	defer e.Close()
	url := connectServer(t, func(mux *http.ServeMux) {
		p, h := jitllmv1connect.NewModelServiceHandler(&ModelService{E: e})
		mux.Handle(p, h)
	})
	cl := jitllmv1connect.NewModelServiceClient(http.DefaultClient, url)

	_, err := cl.LoadModel(context.Background(), connect.NewRequest(&v1.LoadModelRequest{
		Path: "/mnt/models/tinyllama.gguf",
	}))
	if err == nil {
		t.Fatal("loading a .gguf succeeded; the engine reads one weight format")
	}
	var ce *connect.Error
	if !errors.As(err, &ce) {
		t.Fatalf("error is not a connect error: %T %v", err, err)
	}
	if ce.Code() != connect.CodeFailedPrecondition {
		t.Fatalf("code is %v, want failed_precondition -- a GGUF is a caller mistake, not a "+
			"server fault (message: %v)", ce.Code(), ce)
	}
	if !strings.Contains(ce.Message(), "jitllm convert") {
		t.Fatalf("the error does not carry the convert command, which is the only actionable "+
			"thing in it: %q", ce.Message())
	}
}

// TestAnUnknownModelIsNotFoundRatherThanInternal.
func TestAnUnknownModelIsNotFoundRatherThanInternal(t *testing.T) {
	e := New(admitConfig(Config{ModelDir: t.TempDir(), Probe: noProbe}))
	defer e.Close()
	url := connectServer(t, func(mux *http.ServeMux) {
		p, h := jitllmv1connect.NewModelServiceHandler(&ModelService{E: e})
		mux.Handle(p, h)
	})
	cl := jitllmv1connect.NewModelServiceClient(http.DefaultClient, url)
	_, err := cl.GetModel(context.Background(), connect.NewRequest(&v1.GetModelRequest{ModelId: "nope"}))
	var ce *connect.Error
	if !errors.As(err, &ce) || ce.Code() != connect.CodeNotFound {
		t.Fatalf("GetModel on an unknown id returned %v, want not_found", err)
	}
}

// TestGenerateRefusesBothSessionAndModel: one continues a sequence, the other
// makes a throwaway, so neither is silently picked.
func TestGenerateRefusesBothSessionAndModel(t *testing.T) {
	e := New(admitConfig(Config{ModelDir: t.TempDir(), Probe: noProbe}))
	defer e.Close()
	err := e.Generate(context.Background(), GenerateOptions{
		SessionID: "s", ModelID: "m",
		Prompt: Prompt{Kind: PromptText, Text: "hi"},
	}, func(Event) error { return nil })
	if err == nil || !strings.Contains(err.Error(), "not both") {
		t.Fatalf("Generate with both ids returned %v, want a refusal naming the conflict", err)
	}
}

// TestTheDeviceQueueReportsItsReason: the queue says how sessions share the
// device, so a caller reading it knows nothing waits for the device as a whole.
func TestTheDeviceQueueReportsItsReason(t *testing.T) {
	e := New(admitConfig(Config{ModelDir: t.TempDir(), Probe: noProbe}))
	defer e.Close()
	url := connectServer(t, func(mux *http.ServeMux) {
		p, h := jitllmv1connect.NewSessionServiceHandler(&SessionService{E: e})
		mux.Handle(p, h)
	})
	cl := jitllmv1connect.NewSessionServiceClient(http.DefaultClient, url)
	resp, err := cl.GetDeviceQueue(context.Background(),
		connect.NewRequest(&v1.GetDeviceQueueRequest{DeviceId: "cuda:0"}))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Msg.GetExecution() != v1.ExecutionMode_EXECUTION_MODE_PARALLEL {
		t.Fatalf("execution %v, want PARALLEL", resp.Msg.GetExecution())
	}
	if !strings.Contains(resp.Msg.GetNote(), "a step at a time") {
		t.Fatalf("the note does not say how sessions share the device: %q", resp.Msg.GetNote())
	}
}

// ---------------------------------------------------------------- memory

func noProbe() ([]DeviceInfo, error) { return nil, nil }

// TestSpendableTotalSubtractsAUnifiedHeapInsteadOfAddingIt: an integrated
// GPU's heap is system RAM and must not be added to the host budget. Without
// the CountsTowardHostBudget skip in SpendableTotal it fails with "counts the
// integrated heap".
func TestSpendableTotalSubtractsAUnifiedHeapInsteadOfAddingIt(t *testing.T) {
	host := HostInfo{WeightBudget: 8 << 30}
	devs := []DeviceInfo{
		{ID: "cpu", Kind: KindHost, TotalMemory: 32 << 30, CountsTowardHostBudget: true, Available: true},
		{ID: "cuda:0", Kind: KindDiscrete, TotalMemory: 4 << 30, Available: true},
		{ID: "vulkan:1", Kind: KindIntegrated, TotalMemory: 21 << 30, CountsTowardHostBudget: true, Available: true},
		{ID: "vulkan:2", Kind: KindDiscrete, TotalMemory: 8 << 30, Available: false}, // no compute queue
	}
	got := SpendableTotal(host, devs)
	want := uint64(8<<30 + 4<<30)
	if got != want {
		t.Fatalf("spendable %s counts the integrated heap: %s of host budget + %s of real VRAM is %s",
			humanBytes(got), humanBytes(host.WeightBudget), humanBytes(4<<30), humanBytes(want))
	}
}

// TestAnUnavailableDeviceIsNotSpendable. A Vulkan device with no compute queue
// is enumerated -- so a caller can see why it is not an option -- and its
// memory is not a budget.
func TestAnUnavailableDeviceIsNotSpendable(t *testing.T) {
	host := HostInfo{WeightBudget: 1 << 30}
	devs := []DeviceInfo{{ID: "vulkan:0", Kind: KindDiscrete, TotalMemory: 8 << 30, Available: false}}
	if got := SpendableTotal(host, devs); got != 1<<30 {
		t.Fatalf("spendable %s includes a device that cannot run a kernel", humanBytes(got))
	}
}

// TestNormaliseDeviceIDsDropsBudgetSuffixesAndTheHost: `cuda:0=3G` queues on
// the `cuda:0` gate, not on a gate of its own.
func TestNormaliseDeviceIDsDropsBudgetSuffixesAndTheHost(t *testing.T) {
	got := strings.Join(normaliseDeviceIDs([]string{"cuda:0=3G", "cpu", "", "vulkan:1"}), ",")
	if got != "cuda:0,vulkan:1" {
		t.Fatalf("normalised to %q, want %q", got, "cuda:0,vulkan:1")
	}
}
