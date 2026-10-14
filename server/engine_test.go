package server

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/internal/testmodels/fresh"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
	"github.com/jitllm/jitllm/server/gen/jitllm/v1/jitllmv1connect"
)

// These gates run the six Connect services against a REAL engine: a container
// is opened, sessions hold model.States, and tokens are decoded. The fake
// backend in fake_test.go cannot reach any of that -- the services other than
// InferenceService take an *Engine, not a Backend.
//
// The model is stories260K: five blocks of a 260K-parameter llama, so a load
// and a dozen tokens cost milliseconds and these run in every `go test`
// rather than behind an opt-in variable like e2e_test.go.

// smallModel is the container every real-engine gate here loads.
const smallModel = "stories260K.jlm"

// chatModel carries a chat template, which stories260K does not; the chat
// paths need one to be reachable at all.
const chatModel = "tiny-qwen3moe-f32.jlm"

// modelPath is name in the model directory, or a loud skip naming where it
// looked. A missing model is never a pass.
func modelPath(t *testing.T, name string) string {
	t.Helper()
	p := testmodels.Path(name)
	fresh.Must(t, p)
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	return p
}

// oneCardProbe is a host and one discrete card, so the device surfaces have
// something to report without opening a driver.
func oneCardProbe() ([]DeviceInfo, error) {
	return []DeviceInfo{
		{ID: "cpu", Backend: "cpu", Kind: KindHost, Name: "host", TotalMemory: 32 << 30,
			CountsTowardHostBudget: true, Available: true},
		{ID: "cuda:0", Backend: "cuda", Kind: KindDiscrete, Name: "fake card",
			TotalMemory: 4 << 30, FreeMemory: 3 << 30, Available: true, PhysicalID: "uuid-0"},
	}, nil
}

type clients struct {
	url       string
	model     jitllmv1connect.ModelServiceClient
	session   jitllmv1connect.SessionServiceClient
	placement jitllmv1connect.PlacementServiceClient
	inference jitllmv1connect.InferenceServiceClient
	telemetry jitllmv1connect.TelemetryServiceClient
	device    jitllmv1connect.DeviceServiceClient
}

// serveEngine mounts e.Handler() -- the shipping mux, compat shims and all --
// on a test server and returns a client for every service.
func serveEngine(t *testing.T, e *Engine) clients {
	t.Helper()
	s := httptest.NewServer(e.Handler())
	t.Cleanup(s.Close)
	c := http.DefaultClient
	return clients{
		url:       s.URL,
		model:     jitllmv1connect.NewModelServiceClient(c, s.URL),
		session:   jitllmv1connect.NewSessionServiceClient(c, s.URL),
		placement: jitllmv1connect.NewPlacementServiceClient(c, s.URL),
		inference: jitllmv1connect.NewInferenceServiceClient(c, s.URL),
		telemetry: jitllmv1connect.NewTelemetryServiceClient(c, s.URL),
		device:    jitllmv1connect.NewDeviceServiceClient(c, s.URL),
	}
}

// loadedEngine is an Engine with name loaded under id. The engine closes at
// cleanup, after the server (cleanups run last-registered first).
func loadedEngine(t *testing.T, name, id string, o LoadOptions) (*Engine, *LoadedModel, clients) {
	t.Helper()
	path := modelPath(t, name)
	// The model directory is the whole test model disk: with on-demand loads
	// on, a request naming any of its files would load it. These gates are
	// about the one model they load; library_test.go gates the folder.
	e := New(Config{ModelDir: filepath.Dir(path), Probe: oneCardProbe, Version: "test", NoOnDemand: true})
	t.Cleanup(e.Close)
	o.Path, o.ModelID = path, id
	lm, err := e.LoadModel(o)
	if err != nil {
		// A stale container is a task, not a skip: re-convert it.
		t.Fatalf("LoadModel(%s): %v", path, err)
	}
	return e, lm, serveEngine(t, e)
}

// wantCode asserts err is a Connect error carrying want.
func wantCode(t *testing.T, what string, err error, want connect.Code) *connect.Error {
	t.Helper()
	var ce *connect.Error
	if !errors.As(err, &ce) {
		t.Fatalf("%s returned %v (%T), want a connect error with code %v", what, err, err, want)
	}
	if ce.Code() != want {
		t.Fatalf("%s: code %v, want %v (message: %s)", what, ce.Code(), want, ce.Message())
	}
	return ce
}

func req[T any](m *T) *connect.Request[T] { return connect.NewRequest(m) }

func text(s string) *v1.PromptInput {
	return &v1.PromptInput{Input: &v1.PromptInput_Text{Text: s}}
}

func ids(xs ...int32) *v1.PromptInput {
	return &v1.PromptInput{Input: &v1.PromptInput_TokenIds{TokenIds: &v1.TokenIDs{Ids: xs}}}
}

const story = "Once upon a time"

// ---------------------------------------------------------------- models

// TestModelServiceDescribesARealContainer reads what LoadModel opened back
// through GetModel, and refuses the loads a caller can get wrong.
func TestModelServiceDescribesARealContainer(t *testing.T) {
	_, _, c := loadedEngine(t, smallModel, "small", LoadOptions{})
	ctx := context.Background()

	got, err := c.model.GetModel(ctx, req(&v1.GetModelRequest{ModelId: "small"}))
	if err != nil {
		t.Fatal(err)
	}
	mi := got.Msg.GetModel()
	if mi.GetName() != "stories260K" || mi.GetArchitecture() != "llama" || mi.GetBlockCount() != 5 {
		t.Fatalf("GetModel says name=%q arch=%q blocks=%d, want stories260K/llama/5",
			mi.GetName(), mi.GetArchitecture(), mi.GetBlockCount())
	}
	// A plain llama has every block attention and none recurrent.
	if mi.GetAttentionBlocks() != 5 || mi.GetLinearBlocks() != 0 {
		t.Fatalf("attention/linear blocks %d/%d, want 5/0", mi.GetAttentionBlocks(), mi.GetLinearBlocks())
	}
	if mi.GetChatCapable() || len(mi.GetChatTemplateNames()) != 0 {
		t.Fatalf("stories260K carries no template, yet ChatCapable=%v names=%v",
			mi.GetChatCapable(), mi.GetChatTemplateNames())
	}
	if mi.GetPageSize().GetBytes() == 0 || mi.GetWeightBytes().GetBytes() == 0 {
		t.Fatalf("page size %v and weight bytes %v: a container has both", mi.GetPageSize(), mi.GetWeightBytes())
	}

	_, err = c.model.LoadModel(ctx, req(&v1.LoadModelRequest{Path: smallModel, ModelId: "small"}))
	wantCode(t, "LoadModel under an id already loaded", err, connect.CodeAlreadyExists)

	_, err = c.model.LoadModel(ctx, req(&v1.LoadModelRequest{}))
	wantCode(t, "LoadModel with no path", err, connect.CodeInvalidArgument)

	// A bare name resolves under ModelDir, so this asks for a file that is not
	// there, which is the caller's to fix.
	_, err = c.model.LoadModel(ctx, req(&v1.LoadModelRequest{Path: "no-such-model.jlm"}))
	wantCode(t, "LoadModel of a file that does not exist", err, connect.CodeNotFound)

	_, err = c.model.UnloadModel(ctx, req(&v1.UnloadModelRequest{ModelId: "nope"}))
	wantCode(t, "UnloadModel of an unknown id", err, connect.CodeNotFound)

	// A second load of the same file under its own id is two models, not one.
	two, err := c.model.LoadModel(ctx, req(&v1.LoadModelRequest{Path: smallModel}))
	if err != nil {
		t.Fatal(err)
	}
	if id := two.Msg.GetModel().GetModelId(); id == "" || id == "small" {
		t.Fatalf("a load with no id was given %q; it must mint a fresh one", id)
	}
	list, err := c.model.ListModels(ctx, req(&v1.ListModelsRequest{LoadedOnly: true}))
	if err != nil {
		t.Fatal(err)
	}
	if n := len(list.Msg.GetLoaded()); n != 2 || len(list.Msg.GetFiles()) != 0 {
		t.Fatalf("loaded_only listed %d loaded and %d files, want 2 and 0", n, len(list.Msg.GetFiles()))
	}
}

// TestListModelsNamesTheFixForEveryFileItCannotLoad: a GGUF is listed WITH the
// command that converts it, and a container this build cannot open is listed
// with the re-convert, because a caller who cannot see why a file is missing
// has no way to fix it. With the jlm.Open check removed the broken container
// reads as loadable (no convert command) and this fails.
func TestListModelsNamesTheFixForEveryFileItCannotLoad(t *testing.T) {
	_, _, c := loadedEngine(t, smallModel, "small", LoadOptions{})
	ctx := context.Background()

	// A directory of stand-ins, plus a link to the real container. Nothing
	// here is model bytes, so it does not belong on the model volume.
	dir := t.TempDir()
	if err := os.Symlink(modelPath(t, smallModel), filepath.Join(dir, "real.jlm")); err != nil {
		t.Fatal(err)
	}
	for name, body := range map[string]string{
		"broken.jlm": "not a container",
		"src.gguf":   "",
		"notes.txt":  "ignored",
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Mkdir(filepath.Join(dir, "sub.jlm"), 0o755); err != nil {
		t.Fatal(err)
	}

	resp, err := c.model.ListModels(ctx, req(&v1.ListModelsRequest{Directory: dir}))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Msg.GetDirectory() != dir {
		t.Fatalf("directory %q, want %q", resp.Msg.GetDirectory(), dir)
	}
	if len(resp.Msg.GetLoaded()) != 1 {
		t.Fatalf("%d loaded models listed, want 1", len(resp.Msg.GetLoaded()))
	}
	byName := map[string]*v1.ModelFile{}
	var names []string
	for _, f := range resp.Msg.GetFiles() {
		byName[f.GetName()] = f
		names = append(names, f.GetName())
	}
	// Sorted, and only containers and GGUFs: not the text file, not the
	// directory that happens to end in .jlm.
	if strings.Join(names, ",") != "broken,real,src" {
		t.Fatalf("files %v, want [broken real src]", names)
	}
	if f := byName["real"]; !f.GetIsContainer() || f.GetContainerVersion() != uint32(jlm.Version) ||
		f.GetConvertCommand() != "" {
		t.Fatalf("the real container is listed as %+v, want a container of v%d with nothing to fix", f, jlm.Version)
	}
	if f := byName["broken"]; !f.GetIsContainer() || f.GetContainerVersion() != 0 ||
		!strings.Contains(f.GetConvertCommand(), "jitllm convert <source> "+filepath.Join(dir, "broken.jlm")) {
		t.Fatalf("a container that does not open is listed as %+v; it must carry the re-convert", f)
	}
	want := "jitllm convert " + filepath.Join(dir, "src.gguf") + " " + filepath.Join(dir, "src.jlm")
	if f := byName["src"]; f.GetIsContainer() || f.GetConvertCommand() != want {
		t.Fatalf("the GGUF is listed as %+v, want the command %q", f, want)
	}

	// A directory that does not exist is an empty listing, not an error: a
	// fresh install has no model directory yet.
	gone := filepath.Join(dir, "absent")
	resp, err = c.model.ListModels(ctx, req(&v1.ListModelsRequest{Directory: gone}))
	if err != nil {
		t.Fatalf("listing a missing directory: %v", err)
	}
	if len(resp.Msg.GetFiles()) != 0 || resp.Msg.GetDirectory() != gone {
		t.Fatalf("a missing directory listed %d files under %q", len(resp.Msg.GetFiles()), resp.Msg.GetDirectory())
	}
}

// TestTokenizeAndDetokenizeAreInverse. The pieces are the vocabulary's own
// text per id, so a client can render the split, and Detokenize undoes
// Tokenize.
func TestTokenizeAndDetokenizeAreInverse(t *testing.T) {
	_, lm, c := loadedEngine(t, smallModel, "small", LoadOptions{})
	ctx := context.Background()

	tk, err := c.model.Tokenize(ctx, req(&v1.TokenizeRequest{ModelId: "small", Text: story, AddSpecial: true}))
	if err != nil {
		t.Fatal(err)
	}
	got := tk.Msg.GetTokenIds()
	if len(got) < 2 || len(tk.Msg.GetPieces()) != len(got) {
		t.Fatalf("%d ids and %d pieces for %q", len(got), len(tk.Msg.GetPieces()), story)
	}
	plain, err := c.model.Tokenize(ctx, req(&v1.TokenizeRequest{ModelId: "small", Text: story}))
	if err != nil {
		t.Fatal(err)
	}
	// add_special must reach the tokenizer: this vocabulary adds a BOS.
	if len(plain.Msg.GetTokenIds()) != len(got)-1 || got[0] != lm.m.Vocab.BOS {
		t.Fatalf("with specials %v, without %v: add_special did not add the BOS", got, plain.Msg.GetTokenIds())
	}
	dt, err := c.model.Detokenize(ctx, req(&v1.DetokenizeRequest{ModelId: "small", TokenIds: plain.Msg.GetTokenIds()}))
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(dt.Msg.GetText()) != story {
		t.Fatalf("Detokenize(Tokenize(%q)) = %q", story, dt.Msg.GetText())
	}

	_, err = c.model.Tokenize(ctx, req(&v1.TokenizeRequest{ModelId: "nope", Text: "x"}))
	wantCode(t, "Tokenize on an unknown model", err, connect.CodeNotFound)
	_, err = c.model.Detokenize(ctx, req(&v1.DetokenizeRequest{ModelId: "nope"}))
	wantCode(t, "Detokenize on an unknown model", err, connect.CodeNotFound)
	_, err = c.model.ApplyChatTemplate(ctx, req(&v1.ApplyChatTemplateRequest{ModelId: "nope"}))
	wantCode(t, "ApplyChatTemplate on an unknown model", err, connect.CodeNotFound)

	// A base model has no template, and rendering one anyway would hand back
	// a raw completion dressed as a chat turn.
	_, err = c.model.ApplyChatTemplate(ctx, req(&v1.ApplyChatTemplateRequest{
		ModelId: "small", Messages: []*v1.ChatMessage{{Role: "user", Content: "hi"}},
	}))
	wantCode(t, "ApplyChatTemplate on a model with no template", err, connect.CodeFailedPrecondition)
}

// TestAChatTemplateIsRenderedAndTokenizedAsOnePrompt: the ids ApplyChatTemplate
// returns are the rendered prompt encoded WITHOUT added specials -- the
// template already carries its own -- and a chat Generate prefills exactly
// that many tokens. Pairing the render with add_special=true doubles the BOS.
func TestAChatTemplateIsRenderedAndTokenizedAsOnePrompt(t *testing.T) {
	_, _, c := loadedEngine(t, chatModel, "chat", LoadOptions{})
	ctx := context.Background()

	msgs := []*v1.ChatMessage{{Role: "user", Content: "what is two plus two?"}}
	ap, err := c.model.ApplyChatTemplate(ctx, req(&v1.ApplyChatTemplateRequest{
		ModelId: "chat", Messages: msgs, AddGenerationPrompt: true,
	}))
	if err != nil {
		t.Fatal(err)
	}
	prompt := ap.Msg.GetPrompt()
	if !strings.Contains(prompt, "what is two plus two?") {
		t.Fatalf("the rendered prompt does not carry the message: %q", prompt)
	}
	tk, err := c.model.Tokenize(ctx, req(&v1.TokenizeRequest{ModelId: "chat", Text: prompt}))
	if err != nil {
		t.Fatal(err)
	}
	if a, b := ap.Msg.GetTokenIds(), tk.Msg.GetTokenIds(); !equalIDs(a, b) {
		t.Fatalf("ApplyChatTemplate ids %v differ from the rendered prompt tokenized without specials %v", a, b)
	}

	resp, err := c.inference.Complete(ctx, req(&v1.GenerateRequest{
		ModelId:   "chat",
		MaxTokens: 3,
		Prompt: &v1.PromptInput{Input: &v1.PromptInput_Chat{Chat: &v1.ChatPrompt{
			Messages: msgs, AddGenerationPrompt: true,
		}}},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if n := resp.Msg.GetStarted().GetPromptTokens(); int(n) != len(ap.Msg.GetTokenIds()) {
		t.Fatalf("a chat generate prefilled %d tokens; the template renders to %d", n, len(ap.Msg.GetTokenIds()))
	}
}

func equalIDs(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ---------------------------------------------------------------- sessions

// TestASessionKeepsItsSequenceAcrossGenerates: continue_session appends to
// the history, a fresh generate resets it, and ResetSession empties it. With
// the `if !o.Continue` guard dropped (every generate resets) the continued
// position comes back short and this fails.
func TestASessionKeepsItsSequenceAcrossGenerates(t *testing.T) {
	_, _, c := loadedEngine(t, smallModel, "small", LoadOptions{})
	ctx := context.Background()

	_, err := c.session.CreateSession(ctx, req(&v1.CreateSessionRequest{ModelId: "nope"}))
	wantCode(t, "CreateSession on an unknown model", err, connect.CodeNotFound)

	cr, err := c.session.CreateSession(ctx, req(&v1.CreateSessionRequest{
		ModelId: "small", SessionId: "a", MaxSeq: 128,
	}))
	if err != nil {
		t.Fatal(err)
	}
	s := cr.Msg.GetSession()
	if s.GetSessionId() != "a" || s.GetMaxSeq() != 128 || s.GetPosition() != 0 ||
		s.GetHostBlocks() != 5 || s.GetDeviceBlocks() != 0 {
		t.Fatalf("a fresh host session reads %+v", s)
	}
	// KV commits as the context grows, so a fresh session is charged nothing
	// yet; the generate below is what commits it.
	if b := cr.Msg.GetBudget(); b.GetSessionsOpen() != 1 || b.GetSessionsPerDeviceMeasured() {
		t.Fatalf("budget %+v: one session open, and no measured per-device limit", b)
	}
	_, err = c.session.CreateSession(ctx, req(&v1.CreateSessionRequest{ModelId: "small", SessionId: "a"}))
	wantCode(t, "CreateSession under an id that exists", err, connect.CodeAlreadyExists)

	gen := func(p *v1.PromptInput, cont bool) *v1.CompleteResponse {
		t.Helper()
		r, err := c.inference.Complete(ctx, req(&v1.GenerateRequest{
			SessionId: "a", Prompt: p, MaxTokens: 4, ContinueSession: cont,
		}))
		if err != nil {
			t.Fatal(err)
		}
		return r.Msg
	}
	position := func() int32 {
		t.Helper()
		g, err := c.session.GetSession(ctx, req(&v1.GetSessionRequest{SessionId: "a"}))
		if err != nil {
			t.Fatal(err)
		}
		return g.Msg.GetSession().GetPosition()
	}

	first := gen(text(story), false)
	p1 := position()
	if g, err := c.session.GetSession(ctx, req(&v1.GetSessionRequest{SessionId: "a"})); err != nil {
		t.Fatal(err)
	} else if g.Msg.GetSession().GetKvBytes().GetBytes() == 0 || g.Msg.GetBudget().GetKvBytesInUse().GetBytes() == 0 {
		t.Fatalf("after a generate the session reports no KV: %+v", g.Msg)
	}
	if p1 == 0 || p1 != first.GetFinished().GetPosition() {
		t.Fatalf("after one generate the session reads position %d, the generate finished at %d",
			p1, first.GetFinished().GetPosition())
	}
	if first.GetStarted().GetEphemeralSession() {
		t.Fatal("a generate on a named session reported an ephemeral one")
	}

	// Two ids, no added BOS, so the sum below is exact.
	more := gen(ids(first.GetTokenIds()...), true)
	f := more.GetFinished()
	want := p1 + f.GetPromptTokens() + f.GetCompletionTokens()
	if f.GetReason() == v1.FinishReason_FINISH_REASON_STOP {
		want-- // a stop's last token is never run
	}
	if p2 := position(); p2 != want {
		t.Fatalf("continue_session finished at %d, want %d (%d before + %d prompt + %d generated): "+
			"the history was not kept", p2, want, p1, f.GetPromptTokens(), f.GetCompletionTokens())
	}

	again := gen(text(story), false)
	if p3 := position(); p3 != again.GetFinished().GetPosition() || p3 >= want {
		t.Fatalf("a generate without continue_session read position %d after %d: it did not reset", p3, want)
	}
	if again.GetText() != first.GetText() {
		t.Fatalf("the same greedy prompt on a reset session gave %q, then %q", first.GetText(), again.GetText())
	}

	rs, err := c.session.ResetSession(ctx, req(&v1.ResetSessionRequest{SessionId: "a"}))
	if err != nil {
		t.Fatal(err)
	}
	if rs.Msg.GetSession().GetPosition() != 0 || position() != 0 {
		t.Fatalf("ResetSession left position %d", rs.Msg.GetSession().GetPosition())
	}
	g, err := c.session.GetSession(ctx, req(&v1.GetSessionRequest{SessionId: "a"}))
	if err != nil {
		t.Fatal(err)
	}
	if n := g.Msg.GetSession().GetTokensGenerated(); n != int64(first.GetFinished().GetCompletionTokens()+
		f.GetCompletionTokens()+again.GetFinished().GetCompletionTokens()) {
		t.Fatalf("tokens_generated %d does not add up over the three generates", n)
	}

	_, err = c.session.ResetSession(ctx, req(&v1.ResetSessionRequest{SessionId: "nope"}))
	wantCode(t, "ResetSession on an unknown session", err, connect.CodeNotFound)
	_, err = c.session.GetSession(ctx, req(&v1.GetSessionRequest{SessionId: "nope"}))
	wantCode(t, "GetSession on an unknown session", err, connect.CodeNotFound)
	if _, err = c.session.CloseSession(ctx, req(&v1.CloseSessionRequest{SessionId: "a"})); err != nil {
		t.Fatal(err)
	}
	_, err = c.session.GetSession(ctx, req(&v1.GetSessionRequest{SessionId: "a"}))
	wantCode(t, "GetSession after CloseSession", err, connect.CodeNotFound)
	_, err = c.session.CloseSession(ctx, req(&v1.CloseSessionRequest{SessionId: "a"}))
	wantCode(t, "a second CloseSession", err, connect.CodeNotFound)
}

// TestListSessionsFiltersByModelAndDevice. A host-only session queues on the
// host's gate, so it lists under "cpu" and not under a card it never touches.
func TestListSessionsFiltersByModelAndDevice(t *testing.T) {
	_, _, c := loadedEngine(t, smallModel, "small", LoadOptions{})
	ctx := context.Background()
	for _, id := range []string{"a", ""} {
		if _, err := c.session.CreateSession(ctx, req(&v1.CreateSessionRequest{
			ModelId: "small", SessionId: id, MaxSeq: 32,
		})); err != nil {
			t.Fatal(err)
		}
	}
	count := func(m *v1.ListSessionsRequest) int {
		t.Helper()
		r, err := c.session.ListSessions(ctx, req(m))
		if err != nil {
			t.Fatal(err)
		}
		for _, s := range r.Msg.GetSessions() {
			if s.GetSessionId() == "" {
				t.Fatal("a session created with no id was listed with none; one must be minted")
			}
		}
		return len(r.Msg.GetSessions())
	}
	for _, c := range []struct {
		m    *v1.ListSessionsRequest
		want int
	}{
		{&v1.ListSessionsRequest{}, 2},
		{&v1.ListSessionsRequest{ModelId: "small"}, 2},
		{&v1.ListSessionsRequest{ModelId: "other"}, 0},
		{&v1.ListSessionsRequest{DeviceId: "cpu"}, 2},
		{&v1.ListSessionsRequest{DeviceId: "cuda:0"}, 0},
	} {
		if got := count(c.m); got != c.want {
			t.Errorf("ListSessions(model=%q device=%q) listed %d, want %d",
				c.m.GetModelId(), c.m.GetDeviceId(), got, c.want)
		}
	}
	q, err := c.session.GetDeviceQueue(ctx, req(&v1.GetDeviceQueueRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	// No device id is the host's queue, which says how sessions share it.
	if q.Msg.GetDeviceId() != HostGateID || !strings.Contains(q.Msg.GetNote(), "shared pool") ||
		q.Msg.GetRunning() || q.Msg.GetWaiting() != 0 {
		t.Fatalf("the idle host queue reads %+v", q.Msg)
	}
}

// TestUnloadingAModelWithALiveSessionNeedsForce: every State must be closed
// before its Model, so an unload that would orphan one is refused, and force
// closes them first. With the victims check removed the plain unload succeeds
// and the session is left holding a closed model.
func TestUnloadingAModelWithALiveSessionNeedsForce(t *testing.T) {
	_, _, c := loadedEngine(t, smallModel, "small", LoadOptions{})
	ctx := context.Background()
	if _, err := c.session.CreateSession(ctx, req(&v1.CreateSessionRequest{
		ModelId: "small", SessionId: "live", MaxSeq: 32,
	})); err != nil {
		t.Fatal(err)
	}

	_, err := c.model.UnloadModel(ctx, req(&v1.UnloadModelRequest{ModelId: "small"}))
	ce := wantCode(t, "UnloadModel with a live session and no force", err, connect.CodeFailedPrecondition)
	if !strings.Contains(ce.Message(), "1 open session") || !strings.Contains(ce.Message(), "force") {
		t.Fatalf("the refusal does not say why or what to do: %q", ce.Message())
	}
	if _, err := c.model.GetModel(ctx, req(&v1.GetModelRequest{ModelId: "small"})); err != nil {
		t.Fatalf("a refused unload removed the model anyway: %v", err)
	}
	if _, err := c.session.GetSession(ctx, req(&v1.GetSessionRequest{SessionId: "live"})); err != nil {
		t.Fatalf("a refused unload closed the session anyway: %v", err)
	}

	un, err := c.model.UnloadModel(ctx, req(&v1.UnloadModelRequest{ModelId: "small", Force: true}))
	if err != nil {
		t.Fatal(err)
	}
	if un.Msg.GetSessionsClosed() != 1 {
		t.Fatalf("a forced unload reports %d session(s) closed, want 1", un.Msg.GetSessionsClosed())
	}
	_, err = c.session.GetSession(ctx, req(&v1.GetSessionRequest{SessionId: "live"}))
	wantCode(t, "GetSession after its model was unloaded", err, connect.CodeNotFound)
	_, err = c.model.GetModel(ctx, req(&v1.GetModelRequest{ModelId: "small"}))
	wantCode(t, "GetModel after unload", err, connect.CodeNotFound)
	_, err = c.inference.Complete(ctx, req(&v1.GenerateRequest{SessionId: "live", Prompt: text(story)}))
	wantCode(t, "a generate on a session whose model was unloaded", err, connect.CodeNotFound)
}

// waitFor polls cond until it holds, failing after a generous bound. It
// replaces a sleep, never measures anything.
func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(30 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(time.Millisecond)
	}
}

// TestForceUnloadStopsAGenerateMidDecode: an unload arriving while tokens are
// being produced ends the decode as CANCELLED -- not with a fault from a
// closed model -- and the model closes only after the generate let go.
func TestForceUnloadStopsAGenerateMidDecode(t *testing.T) {
	e, _, _ := loadedEngine(t, smallModel, "small", LoadOptions{})
	s, err := e.CreateSession(SessionOptions{ModelID: "small", SessionID: "busy", MaxSeq: 512})
	if err != nil {
		t.Fatal(err)
	}
	unloaded := make(chan error, 1)
	var fin *Finished
	err = e.Generate(context.Background(), GenerateOptions{
		SessionID: "busy", Prompt: Prompt{Kind: PromptText, Text: story}, MaxTokens: 400,
	}, func(ev Event) error {
		switch ev.Kind {
		case EventToken:
			if ev.Token.Index == 0 {
				go func() {
					_, err := e.UnloadModel("small", true)
					unloaded <- err
				}()
				waitFor(t, "the unload to mark the session closed", s.closed.Load)
			}
		case EventFinished:
			fin = ev.Finished
		}
		return nil
	})
	if err != nil {
		t.Fatalf("the generate failed rather than finishing cancelled: %v", err)
	}
	if fin == nil || fin.Reason != FinishCancelled {
		t.Fatalf("finished %+v, want reason CANCELLED", fin)
	}
	if fin.CompletionTokens >= 400 {
		t.Fatalf("the decode ran all %d tokens: the unload never reached it", fin.CompletionTokens)
	}
	if err := <-unloaded; err != nil {
		t.Fatalf("UnloadModel(force): %v", err)
	}
}

// TestForceUnloadStopsAStepLoopRowWhileItsClientIsBusy: a row of the step
// loop does not wait for its client, so an unload arriving while the client
// is still inside a token's callback must reach the loop then, not when the
// callback returns. The callback holds on until the loop has let go of the
// row, which is the case of a device decoding faster than its client reads (a
// whole model on Metal): with the cancel noticed only by the drain between
// callbacks, the row runs to max_tokens and finishes MAX_TOKENS.
func TestForceUnloadStopsAStepLoopRowWhileItsClientIsBusy(t *testing.T) {
	e, lm, _ := loadedEngine(t, smallModel, "small", LoadOptions{})
	if lm.loop == nil {
		t.Fatal("the model has no step loop: this gate would run the one-at-a-time path")
	}
	lp := lm.loop
	s, err := e.CreateSession(SessionOptions{ModelID: "small", SessionID: "busy", MaxSeq: 512})
	if err != nil {
		t.Fatal(err)
	}
	unloaded := make(chan error, 1)
	var fin *Finished
	err = e.Generate(context.Background(), GenerateOptions{
		SessionID: "busy", Prompt: Prompt{Kind: PromptText, Text: story}, MaxTokens: 400,
	}, func(ev Event) error {
		switch ev.Kind {
		case EventToken:
			if ev.Token.Index == 0 {
				go func() {
					_, err := e.UnloadModel("small", true)
					unloaded <- err
				}()
				waitFor(t, "the unload to mark the session closed", s.closed.Load)
				waitFor(t, "the loop to let go of the row", func() bool {
					live, _ := lp.snapshot()
					return len(live) == 0
				})
			}
		case EventFinished:
			fin = ev.Finished
		}
		return nil
	})
	if err != nil {
		t.Fatalf("the generate failed rather than finishing cancelled: %v", err)
	}
	if fin == nil || fin.Reason != FinishCancelled {
		t.Fatalf("finished %+v, want reason CANCELLED", fin)
	}
	if err := <-unloaded; err != nil {
		t.Fatalf("UnloadModel(force): %v", err)
	}
}

// TestCancelReportsWhetherAGenerateWasRunning: "was it generating" and "does
// the session exist" are different answers. A cancel from inside the stream
// stops the decode at the next token. With cancelGeneration not calling the
// cancel func the generate runs to max_tokens and this fails.
//
// It runs with batching off: alone, the decode waits for each token's emit,
// which is what makes "the next token" exact. A row of the step loop does not
// wait for its client (its tokens go to an outbox), so a cancel reaches it
// some steps on; TestBatchCancelOneRowLeavesTheOthers gates that path.
func TestCancelReportsWhetherAGenerateWasRunning(t *testing.T) {
	path := modelPath(t, smallModel)
	e := New(Config{ModelDir: filepath.Dir(path), Probe: oneCardProbe, Version: "test", MaxBatchRows: 1})
	t.Cleanup(e.Close)
	if _, err := e.LoadModel(LoadOptions{Path: path, ModelID: "small"}); err != nil {
		t.Fatal(err)
	}
	c := serveEngine(t, e)
	ctx := context.Background()
	_, err := c.inference.Cancel(ctx, req(&v1.CancelRequest{SessionId: "nope"}))
	wantCode(t, "Cancel on an unknown session", err, connect.CodeNotFound)

	if _, err := e.CreateSession(SessionOptions{ModelID: "small", SessionID: "c", MaxSeq: 128}); err != nil {
		t.Fatal(err)
	}
	idle, err := c.inference.Cancel(ctx, req(&v1.CancelRequest{SessionId: "c"}))
	if err != nil {
		t.Fatal(err)
	}
	if idle.Msg.GetWasGenerating() {
		t.Fatal("Cancel on an idle session reported a generate in flight")
	}

	var was, batched bool
	var fin *Finished
	err = e.Generate(ctx, GenerateOptions{
		SessionID: "c", Prompt: Prompt{Kind: PromptText, Text: story}, MaxTokens: 64,
	}, func(ev Event) error {
		switch ev.Kind {
		case EventStarted:
			batched = ev.Started.Batched
		case EventToken:
			if ev.Token.Index == 0 {
				r, err := c.inference.Cancel(ctx, req(&v1.CancelRequest{SessionId: "c"}))
				if err != nil {
					return err
				}
				was = r.Msg.GetWasGenerating()
			}
		case EventFinished:
			fin = ev.Finished
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !was {
		t.Fatal("Cancel during a generate reported nothing running")
	}
	if batched {
		t.Fatal("with batching off the generate ran as a row")
	}
	if fin.Reason != FinishCancelled || fin.CompletionTokens != 1 {
		t.Fatalf("finished %v after %d token(s), want CANCELLED after 1", fin.Reason, fin.CompletionTokens)
	}

	// A client that goes away is a cancel too: its context ends the decode.
	cctx, cancel := context.WithCancel(ctx)
	defer cancel()
	fin = nil
	err = e.Generate(cctx, GenerateOptions{
		SessionID: "c", Prompt: Prompt{Kind: PromptText, Text: story}, MaxTokens: 64,
	}, func(ev Event) error {
		if ev.Kind == EventToken && ev.Token.Index == 1 {
			cancel()
		}
		if ev.Kind == EventFinished {
			fin = ev.Finished
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if fin.Reason != FinishCancelled || fin.CompletionTokens != 2 {
		t.Fatalf("a dropped client finished %v after %d token(s), want CANCELLED after 2",
			fin.Reason, fin.CompletionTokens)
	}
}

// ---------------------------------------------------------------- inference

// TestABadGenerateIsTheCallersErrorNotTheServers: every request below is
// malformed by the caller, and a 500-class code tells a client to retry the
// identical request, so each must carry ErrInvalid and not come back as
// CodeInternal.
func TestABadGenerateIsTheCallersErrorNotTheServers(t *testing.T) {
	e, _, c := loadedEngine(t, smallModel, "small", LoadOptions{})
	ctx := context.Background()
	if _, err := e.CreateSession(SessionOptions{ModelID: "small", SessionID: "s", MaxSeq: 64}); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		what string
		m    *v1.GenerateRequest
		want connect.Code
		says string
	}{
		{"no prompt", &v1.GenerateRequest{ModelId: "small"}, connect.CodeInvalidArgument, "prompt"},
		{"an empty id list", &v1.GenerateRequest{ModelId: "small", Prompt: ids()},
			connect.CodeInvalidArgument, "empty"},
		{"continue with nothing to run", &v1.GenerateRequest{SessionId: "s", Prompt: ids(), ContinueSession: true},
			connect.CodeInvalidArgument, "no token"},
		{"both a session and a model", &v1.GenerateRequest{SessionId: "s", ModelId: "small", Prompt: text("x")},
			connect.CodeInvalidArgument, "not both"},
		{"neither a session nor a model", &v1.GenerateRequest{Prompt: text("x")},
			connect.CodeInvalidArgument, "required"},
		{"a chat prompt to a model with no template", &v1.GenerateRequest{ModelId: "small",
			Prompt: &v1.PromptInput{Input: &v1.PromptInput_Chat{Chat: &v1.ChatPrompt{
				Messages: []*v1.ChatMessage{{Role: "user", Content: "hi"}}}}}},
			connect.CodeInvalidArgument, "RAW COMPLETION"},
		{"an unknown model", &v1.GenerateRequest{ModelId: "nope", Prompt: text("x")}, connect.CodeNotFound, "nope"},
		{"an unknown session", &v1.GenerateRequest{SessionId: "nope", Prompt: text("x")}, connect.CodeNotFound, "nope"},
	} {
		_, err := c.inference.Complete(ctx, req(tc.m))
		ce := wantCode(t, tc.what, err, tc.want)
		if !strings.Contains(ce.Message(), tc.says) {
			t.Errorf("%s: the message %q does not say %q", tc.what, ce.Message(), tc.says)
		}
	}
	// None of those may leave a session behind: an ephemeral one is closed
	// even when its generate fails.
	if n := len(e.Sessions()); n != 1 {
		t.Fatalf("%d sessions open after the refusals, want the one created above", n)
	}
}

// TestEchoStopAndEphemeralSessionsOnTheRealEngine: echo returns the prompt
// text as a token with no id, a stop string ends the completion before it, and
// a model_id request runs on a session that is gone afterwards.
func TestEchoStopAndEphemeralSessionsOnTheRealEngine(t *testing.T) {
	e, _, c := loadedEngine(t, smallModel, "small", LoadOptions{})
	ctx := context.Background()

	full, err := c.inference.Complete(ctx, req(&v1.GenerateRequest{ModelId: "small", Prompt: text(story), MaxTokens: 12}))
	if err != nil {
		t.Fatal(err)
	}
	if !full.Msg.GetStarted().GetEphemeralSession() {
		t.Fatal("a model_id generate did not report an ephemeral session")
	}
	if n := len(e.Sessions()); n != 0 {
		t.Fatalf("the ephemeral session outlived its generate: %d open", n)
	}
	out := full.Msg.GetText()
	if len(strings.TrimSpace(out)) < 4 {
		t.Fatalf("too little text to pick a stop string from: %q", out)
	}

	echo, err := c.inference.Complete(ctx, req(&v1.GenerateRequest{
		ModelId: "small", Prompt: text(story), MaxTokens: 12, EchoPrompt: true,
	}))
	if err != nil {
		t.Fatal(err)
	}
	if got := echo.Msg.GetText(); !strings.HasPrefix(strings.TrimSpace(got), story) || !strings.HasSuffix(got, out) {
		t.Fatalf("echo gave %q; want the prompt, then the same completion %q", got, out)
	}
	// The echoed prompt carries no id: token_ids is the completion only.
	if len(echo.Msg.GetTokenIds()) != len(full.Msg.GetTokenIds()) {
		t.Fatalf("echo returned %d ids, the plain generate %d: the echo was counted as a token",
			len(echo.Msg.GetTokenIds()), len(full.Msg.GetTokenIds()))
	}

	// Greedy is deterministic, so a stop taken from the middle of the text
	// must cut it there.
	trimmed := strings.TrimSpace(out)
	stop := trimmed[len(trimmed)/2 : len(trimmed)/2+2]
	cut, err := c.inference.Complete(ctx, req(&v1.GenerateRequest{
		ModelId: "small", Prompt: text(story), MaxTokens: 12, Stop: []string{stop},
	}))
	if err != nil {
		t.Fatal(err)
	}
	f := cut.Msg.GetFinished()
	if f.GetReason() != v1.FinishReason_FINISH_REASON_STOP || f.GetStopMatched() != stop {
		t.Fatalf("stop %q: finished %v matching %q", stop, f.GetReason(), f.GetStopMatched())
	}
	if want := out[:strings.Index(out, stop)]; cut.Msg.GetText() != want {
		t.Fatalf("stop %q cut the text to %q, want %q", stop, cut.Msg.GetText(), want)
	}
}

// TestTheConnectStreamCarriesTheRealEnginesEvents: the streaming RPC against
// the engine, not the fake: started first, finished last, and the token text
// adds up to what Complete returns for the same greedy request.
func TestTheConnectStreamCarriesTheRealEnginesEvents(t *testing.T) {
	_, _, c := loadedEngine(t, smallModel, "small", LoadOptions{})
	ctx := context.Background()
	m := &v1.GenerateRequest{ModelId: "small", Prompt: text(story), MaxTokens: 8}
	st, err := c.inference.Generate(ctx, req(m))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	var kinds []string
	var b strings.Builder
	for st.Receive() {
		switch ev := st.Msg().GetEvent().(type) {
		case *v1.GenerateResponse_Started:
			kinds = append(kinds, "started")
			if ev.Started.GetHostBlocks() != 5 || ev.Started.GetDeviceBlocks() != 0 {
				t.Fatalf("a host-only generate reports %d host / %d device blocks",
					ev.Started.GetHostBlocks(), ev.Started.GetDeviceBlocks())
			}
		case *v1.GenerateResponse_Token:
			kinds = append(kinds, "token")
			b.WriteString(ev.Token.GetText())
		case *v1.GenerateResponse_Finished:
			kinds = append(kinds, "finished")
			if ev.Finished.GetBytesPerToken() == 0 {
				t.Fatal("bytes_per_token is 0 on the real engine")
			}
		}
	}
	if err := st.Err(); err != nil {
		t.Fatal(err)
	}
	if len(kinds) < 3 || kinds[0] != "started" || kinds[len(kinds)-1] != "finished" {
		t.Fatalf("event order %v", kinds)
	}
	one, err := c.inference.Complete(ctx, req(m))
	if err != nil {
		t.Fatal(err)
	}
	if b.String() != one.Msg.GetText() {
		t.Fatalf("the stream spelled %q and Complete %q for the same greedy request", b.String(), one.Msg.GetText())
	}

	// A refusal on the stream arrives as the stream's error, with its code.
	bad, err := c.inference.Generate(ctx, req(&v1.GenerateRequest{ModelId: "nope", Prompt: text("x")}))
	if err == nil {
		for bad.Receive() {
		}
		err = bad.Err()
		bad.Close()
	}
	wantCode(t, "a streaming generate on an unknown model", err, connect.CodeNotFound)
}

// ---------------------------------------------------------------- placement

// TestPlacementRefusesWhatAHostOnlySessionCannotDo: every block is on the
// host, and each request that would need a tier is refused with its reason
// rather than answered on the CPU with a device's name on it.
func TestPlacementRefusesWhatAHostOnlySessionCannotDo(t *testing.T) {
	e, lm, c := loadedEngine(t, smallModel, "small", LoadOptions{})
	ctx := context.Background()
	if _, err := e.CreateSession(SessionOptions{ModelID: "small", SessionID: "p", MaxSeq: 64}); err != nil {
		t.Fatal(err)
	}

	gp, err := c.placement.GetPlacement(ctx, req(&v1.GetPlacementRequest{SessionId: "p"}))
	if err != nil {
		t.Fatal(err)
	}
	pl := gp.Msg.GetPlacement()
	if pl.GetHostBlockCount() != 5 || pl.GetDeviceBlockCount() != 0 || len(pl.GetBlocks()) != 5 {
		t.Fatalf("a host-only placement reads host=%d device=%d over %d blocks",
			pl.GetHostBlockCount(), pl.GetDeviceBlockCount(), len(pl.GetBlocks()))
	}
	for i, b := range pl.GetBlocks() {
		if b.GetIndex() != int32(i) || b.GetLocation() != v1.BlockLocation_BLOCK_LOCATION_HOST ||
			!b.GetHostResident() || b.GetBytes().GetBytes() != lm.m.PageBytes(i) ||
			b.GetKind() != v1.BlockKind_BLOCK_KIND_ATTENTION {
			t.Fatalf("block %d reads %+v", i, b)
		}
	}

	home, err := c.placement.SetPlacement(ctx, req(&v1.SetPlacementRequest{SessionId: "p"}))
	if err != nil {
		t.Fatalf("bringing everything home on a host session: %v", err)
	}
	if home.Msg.GetPlacement().GetDeviceBlockCount() != 0 {
		t.Fatal("everything-home left a block on a device")
	}
	_, err = c.placement.SetPlacement(ctx, req(&v1.SetPlacementRequest{SessionId: "p", DeviceIds: []string{"cuda:0"}}))
	wantCode(t, "SetPlacement onto a device for a host-only model", err, connect.CodeFailedPrecondition)

	for _, tc := range []struct {
		what string
		m    *v1.RelocateBlocksRequest
		want connect.Code
	}{
		{"a negative first block", &v1.RelocateBlocksRequest{SessionId: "p", FirstBlock: -1, LastBlock: 0, ToHost: true},
			connect.CodeInvalidArgument},
		{"a range past the last block", &v1.RelocateBlocksRequest{SessionId: "p", FirstBlock: 0, LastBlock: 5, ToHost: true},
			connect.CodeInvalidArgument},
		{"a reversed range", &v1.RelocateBlocksRequest{SessionId: "p", FirstBlock: 3, LastBlock: 2, ToHost: true},
			connect.CodeInvalidArgument},
		{"no destination", &v1.RelocateBlocksRequest{SessionId: "p", FirstBlock: 0, LastBlock: 1},
			connect.CodeInvalidArgument},
		{"a device for a host-only model", &v1.RelocateBlocksRequest{SessionId: "p", FirstBlock: 0, LastBlock: 1, ToDeviceId: "cuda:0"},
			connect.CodeFailedPrecondition},
		// Nothing is on a device, so no range is the TAIL of the device set.
		{"home, when nothing is away", &v1.RelocateBlocksRequest{SessionId: "p", FirstBlock: 0, LastBlock: 1, ToHost: true},
			connect.CodeUnimplemented},
		{"an unknown session", &v1.RelocateBlocksRequest{SessionId: "nope", ToHost: true}, connect.CodeNotFound},
	} {
		_, err := c.placement.RelocateBlocks(ctx, req(tc.m))
		wantCode(t, "RelocateBlocks with "+tc.what, err, tc.want)
	}

	rl, err := c.placement.SetRelocation(ctx, req(&v1.SetRelocationRequest{SessionId: "p", Enabled: true}))
	if err != nil || !rl.Msg.GetEnabled() {
		t.Fatalf("SetRelocation(true): %v %v", rl, err)
	}

	ts, err := c.placement.TuneSeam(ctx, req(&v1.TuneSeamRequest{SessionId: "p"}))
	if err == nil {
		for ts.Receive() {
		}
		err = ts.Err()
		ts.Close()
	}
	wantCode(t, "TuneSeam on a host-only session", err, connect.CodeFailedPrecondition)

	for what, call := range map[string]func() error{
		"GetPlacement": func() error {
			_, err := c.placement.GetPlacement(ctx, req(&v1.GetPlacementRequest{SessionId: "nope"}))
			return err
		},
		"SetPlacement": func() error {
			_, err := c.placement.SetPlacement(ctx, req(&v1.SetPlacementRequest{SessionId: "nope"}))
			return err
		},
		"SetRelocation": func() error {
			_, err := c.placement.SetRelocation(ctx, req(&v1.SetRelocationRequest{SessionId: "nope"}))
			return err
		},
		"GetResidency": func() error {
			_, err := c.placement.GetResidency(ctx, req(&v1.GetResidencyRequest{ModelId: "nope"}))
			return err
		},
		"SetPageBudget": func() error {
			_, err := c.placement.SetPageBudget(ctx, req(&v1.SetPageBudgetRequest{ModelId: "nope", BudgetBytes: 1}))
			return err
		},
	} {
		wantCode(t, what+" on an unknown id", call(), connect.CodeNotFound)
	}
}

// refusingDevice is a device that serves nothing. It stands in for a tier so
// SetPlacement reaches State.SetDeviceLayers on a model loaded host-only.
type refusingDevice struct{}

func (refusingDevice) MatVec([]float32, quant.Type, []byte, []float32, int, int) bool { return false }

// TestARefusedPlacementLeavesTheSessionUsable: when the State refuses the move
// -- here a binary16 KV cache that already holds history, which cannot migrate
// -- SetPlacement returns FailedPrecondition, and the session's lock must be
// released on that path too, or the next call on the session hangs.
func TestARefusedPlacementLeavesTheSessionUsable(t *testing.T) {
	f16 := true
	e, lm, c := loadedEngine(t, smallModel, "small", LoadOptions{KVF16: &f16})
	ctx := context.Background()
	s, err := e.CreateSession(SessionOptions{ModelID: "small", SessionID: "p", MaxSeq: 64})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.inference.Complete(ctx, req(&v1.GenerateRequest{SessionId: "p", Prompt: text(story), MaxTokens: 2})); err != nil {
		t.Fatal(err)
	}
	// The model gains a "tier" after the session was built host-only, so the
	// session's f16 history is what the move has to carry.
	lm.dev, lm.deviceIDs = refusingDevice{}, []string{"fake:0"}
	defer func() { lm.dev, lm.deviceIDs = nil, nil }()

	_, err = c.placement.SetPlacement(ctx, req(&v1.SetPlacementRequest{SessionId: "p", DeviceIds: []string{"fake:0"}}))
	ce := wantCode(t, "SetPlacement the State refuses", err, connect.CodeFailedPrecondition)
	if !strings.Contains(ce.Message(), "binary16") {
		t.Fatalf("the refusal lost the State's reason: %q", ce.Message())
	}
	// A different device set is refused before the lock is taken at all.
	_, err = c.placement.SetPlacement(ctx, req(&v1.SetPlacementRequest{SessionId: "p", DeviceIds: []string{"cuda:0"}}))
	wantCode(t, "SetPlacement onto a device the tier does not hold", err, connect.CodeFailedPrecondition)

	done := make(chan error, 1)
	go func() {
		_, err := c.session.ResetSession(ctx, req(&v1.ResetSessionRequest{SessionId: "p"}))
		done <- err
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(30 * time.Second):
		// Hand back the lock the handler kept, so this reports instead of
		// hanging the server's shutdown behind the stuck ResetSession.
		s.mu.Unlock()
		t.Fatal("ResetSession hung after a refused SetPlacement: the session's lock was never released")
	}
}

// TestAPageBudgetBelowTheDenseRegionStillPages: the pager reads a budget of
// ZERO as "unlimited", so a share that works out to nothing must not be handed
// over as zero -- that would turn the smallest budget a caller can ask for
// into the largest: a 1-byte SetPageBudget must not report fits=true with
// budget 0, and the generate after it must evict.
func TestAPageBudgetBelowTheDenseRegionStillPages(t *testing.T) {
	e, lm, c := loadedEngine(t, smallModel, "small", LoadOptions{})
	ctx := context.Background()
	page, dense := lm.m.PageSize(), lm.m.DenseBytes()

	res, err := c.placement.GetResidency(ctx, req(&v1.GetResidencyRequest{ModelId: "small"}))
	if err != nil {
		t.Fatal(err)
	}
	if r := res.Msg.GetResidency(); !r.GetFits() || r.GetBlocks() != 5 || r.GetBudget().GetBytes() == 0 {
		t.Fatalf("under the default budget stories260K must fit: %+v", r)
	}

	// Two pages after the dense region: the cliff the response exists to show.
	two, err := c.placement.SetPageBudget(ctx, req(&v1.SetPageBudgetRequest{ModelId: "small", BudgetBytes: dense + 2*page}))
	if err != nil {
		t.Fatal(err)
	}
	if r := two.Msg.GetResidency(); two.Msg.GetFits() || r.GetBudget().GetBytes() != 2*page || r.GetResidentBlocks() > 2 {
		t.Fatalf("a two-page budget reads fits=%v budget=%d resident=%d; want false, %d, <= 2",
			two.Msg.GetFits(), r.GetBudget().GetBytes(), r.GetResidentBlocks(), 2*page)
	}

	tiny, err := c.placement.SetPageBudget(ctx, req(&v1.SetPageBudgetRequest{ModelId: "small", BudgetBytes: 1}))
	if err != nil {
		t.Fatal(err)
	}
	if r := tiny.Msg.GetResidency(); tiny.Msg.GetFits() || r.GetBudget().GetBytes() == 0 {
		t.Fatalf("a 1-byte budget reads fits=%v with budget %d: it became unlimited",
			tiny.Msg.GetFits(), r.GetBudget().GetBytes())
	}
	before := tiny.Msg.GetResidency().GetEvictions()
	if _, err := c.inference.Complete(ctx, req(&v1.GenerateRequest{ModelId: "small", Prompt: text(story), MaxTokens: 2})); err != nil {
		t.Fatal(err)
	}
	after, err := c.placement.GetResidency(ctx, req(&v1.GetResidencyRequest{ModelId: "small"}))
	if err != nil {
		t.Fatal(err)
	}
	if r := after.Msg.GetResidency(); r.GetEvictions() == before || r.GetResidentBlocks() > 1 {
		t.Fatalf("a token under a 1-byte budget evicted %d page(s) and left %d resident: it did not page",
			r.GetEvictions()-before, r.GetResidentBlocks())
	}

	// The same through LoadModel: the session's re-budget subtracts its KV
	// from a share that is already nothing.
	if _, err := e.LoadModel(LoadOptions{Path: lm.path, ModelID: "starved", PageBudgetBytes: 1}); err != nil {
		t.Fatal(err)
	}
	if _, err := c.inference.Complete(ctx, req(&v1.GenerateRequest{ModelId: "starved", Prompt: text(story), MaxTokens: 2})); err != nil {
		t.Fatal(err)
	}
	st, err := c.placement.GetResidency(ctx, req(&v1.GetResidencyRequest{ModelId: "starved"}))
	if err != nil {
		t.Fatal(err)
	}
	if r := st.Msg.GetResidency(); r.GetBudget().GetBytes() == 0 || r.GetEvictions() == 0 || r.GetFits() {
		t.Fatalf("a model loaded under a 1-byte budget reads budget=%d evictions=%d fits=%v: it ran unlimited",
			r.GetBudget().GetBytes(), r.GetEvictions(), r.GetFits())
	}
}

// ---------------------------------------------------------------- telemetry

// TestStatsCountWhatRan: the counters are facts about what ran, so after a
// known generate they read exactly its token counts.
func TestStatsCountWhatRan(t *testing.T) {
	e, _, c := loadedEngine(t, smallModel, "small", LoadOptions{})
	ctx := context.Background()
	if _, err := e.CreateSession(SessionOptions{ModelID: "small", SessionID: "t", MaxSeq: 64}); err != nil {
		t.Fatal(err)
	}
	r, err := c.inference.Complete(ctx, req(&v1.GenerateRequest{SessionId: "t", Prompt: text(story), MaxTokens: 3}))
	if err != nil {
		t.Fatal(err)
	}
	f := r.Msg.GetFinished()

	st, err := c.telemetry.GetStats(ctx, req(&v1.GetStatsRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	if len(st.Msg.GetModels()) != 1 || len(st.Msg.GetSessions()) != 1 {
		t.Fatalf("stats list %d models and %d sessions, want 1 and 1", len(st.Msg.GetModels()), len(st.Msg.GetSessions()))
	}
	ms, ss := st.Msg.GetModels()[0], st.Msg.GetSessions()[0]
	if ms.GetTokensGenerated() != int64(f.GetCompletionTokens()) || ms.GetTokensPrefilled() != int64(f.GetPromptTokens()) ||
		ms.GetSessions() != 1 {
		t.Fatalf("model stats %+v after one generate of %d+%d", ms, f.GetPromptTokens(), f.GetCompletionTokens())
	}
	if ss.GetPosition() != f.GetPosition() || ss.GetGenerating() || ss.GetPool().GetWorkers() == 0 ||
		ss.GetPool().GetParallelRegions()+ss.GetPool().GetInlineRegions() == 0 {
		t.Fatalf("session stats %+v: an idle session that just ran reports its pool", ss)
	}
	if len(st.Msg.GetDevices()) != 2 || st.Msg.GetHost() == nil {
		t.Fatalf("stats carry %d devices and host %v", len(st.Msg.GetDevices()), st.Msg.GetHost())
	}

	none, err := c.telemetry.GetStats(ctx, req(&v1.GetStatsRequest{ModelId: "other"}))
	if err != nil {
		t.Fatal(err)
	}
	if len(none.Msg.GetModels())+len(none.Msg.GetSessions()) != 0 {
		t.Fatal("a model filter that matches nothing returned stats")
	}
	only, err := c.telemetry.GetStats(ctx, req(&v1.GetStatsRequest{SessionId: "other"}))
	if err != nil {
		t.Fatal(err)
	}
	if len(only.Msg.GetSessions()) != 0 || len(only.Msg.GetModels()) != 1 {
		t.Fatal("a session filter must narrow the sessions and leave the models")
	}

	wctx, cancel := context.WithCancel(ctx)
	w, err := c.telemetry.WatchStats(wctx, req(&v1.WatchStatsRequest{ModelId: "small"}))
	if err != nil {
		t.Fatal(err)
	}
	if !w.Receive() {
		t.Fatalf("WatchStats sent no first snapshot: %v", w.Err())
	}
	if len(w.Msg().GetModels()) != 1 {
		t.Fatalf("the watched snapshot lists %d models", len(w.Msg().GetModels()))
	}
	cancel()
	for w.Receive() {
	}
	w.Close()

	info, err := c.telemetry.GetServerInfo(ctx, req(&v1.GetServerInfoRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	in := info.Msg.GetInfo()
	if in.GetVersion() != "test" || in.GetLoadedModels() != 1 || in.GetOpenSessions() != 1 ||
		strings.Join(in.GetAvailableBackends(), ",") != "cpu,cuda:0" || in.GetModelDirectory() == "" {
		t.Fatalf("server info %+v", in)
	}
}

// TestDevicesAreTheProbesAnswer: the device surfaces report the probe, mapped
// field for field, and a card the probe does not know is NotFound.
func TestDevicesAreTheProbesAnswer(t *testing.T) {
	e, _, c := loadedEngine(t, smallModel, "small", LoadOptions{})
	ctx := context.Background()
	if _, err := e.CreateSession(SessionOptions{ModelID: "small", SessionID: "d", MaxSeq: 32}); err != nil {
		t.Fatal(err)
	}

	ld, err := c.device.ListDevices(ctx, req(&v1.ListDevicesRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	if len(ld.Msg.GetDevices()) != 2 {
		t.Fatalf("%d devices, want the probe's 2", len(ld.Msg.GetDevices()))
	}
	host := ld.Msg.GetDevices()[0]
	if host.GetRef().GetBackend() != v1.Backend_BACKEND_CPU || host.GetKind() != v1.DeviceKind_DEVICE_KIND_HOST ||
		host.GetAttachedSessions() != 1 {
		t.Fatalf("the host reads %+v; its one host-only session is attached to it", host)
	}

	gd, err := c.device.GetDevice(ctx, req(&v1.GetDeviceRequest{DeviceId: "cuda:0"}))
	if err != nil {
		t.Fatal(err)
	}
	d := gd.Msg.GetDevice()
	if d.GetName() != "fake card" || d.GetRef().GetBackend() != v1.Backend_BACKEND_CUDA ||
		d.GetKind() != v1.DeviceKind_DEVICE_KIND_DISCRETE || d.GetTotalMemory().GetBytes() != 4<<30 ||
		d.GetFreeMemory().GetBytes() != 3<<30 || d.GetPhysicalId() != "uuid-0" || d.GetAttachedSessions() != 0 ||
		d.GetExecution() != v1.ExecutionMode_EXECUTION_MODE_PARALLEL || !d.GetAvailable() {
		t.Fatalf("cuda:0 reads %+v", d)
	}
	_, err = c.device.GetDevice(ctx, req(&v1.GetDeviceRequest{DeviceId: "cuda:9"}))
	wantCode(t, "GetDevice on a card the probe does not list", err, connect.CodeNotFound)

	top, err := c.device.GetMemoryTopology(ctx, req(&v1.GetMemoryTopologyRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	tp := top.Msg.GetTopology()
	want := tp.GetHost().GetWeightBudget().GetBytes() + 4<<30
	if tp.GetSpendableTotal().GetBytes() != want || len(tp.GetDevices()) != 2 {
		t.Fatalf("spendable %d over %d devices, want the host budget plus the card's 4 GiB = %d",
			tp.GetSpendableTotal().GetBytes(), len(tp.GetDevices()), want)
	}
}

// TestAProbeFailureIsReportedNotHidden: a probe that cannot enumerate the
// hardware is an error on every device surface, never an empty machine.
func TestAProbeFailureIsReportedNotHidden(t *testing.T) {
	e := New(Config{ModelDir: t.TempDir(), Probe: func() ([]DeviceInfo, error) {
		return nil, errors.New("the driver is wedged")
	}})
	defer e.Close()
	c := serveEngine(t, e)
	ctx := context.Background()
	_, err := c.device.ListDevices(ctx, req(&v1.ListDevicesRequest{}))
	ce := wantCode(t, "ListDevices under a failing probe", err, connect.CodeInternal)
	if !strings.Contains(ce.Message(), "wedged") {
		t.Fatalf("the probe's reason was lost: %q", ce.Message())
	}
	_, err = c.device.GetDevice(ctx, req(&v1.GetDeviceRequest{DeviceId: "cpu"}))
	wantCode(t, "GetDevice under a failing probe", err, connect.CodeInternal)
	_, err = c.device.GetMemoryTopology(ctx, req(&v1.GetMemoryTopologyRequest{}))
	wantCode(t, "GetMemoryTopology under a failing probe", err, connect.CodeInternal)

	// Telemetry stays up: stats are about what ran, and the devices are
	// simply absent from them.
	st, err := c.telemetry.GetStats(ctx, req(&v1.GetStatsRequest{}))
	if err != nil || len(st.Msg.GetDevices()) != 0 {
		t.Fatalf("GetStats under a failing probe: %v, %d devices", err, len(st.Msg.GetDevices()))
	}
}

// TestHealthzAndTheCompatShimsAreMounted: Handler mounts every surface on one
// mux, so a typo in a route is a 404 here rather than in a client.
func TestHealthzAndTheCompatShimsAreMounted(t *testing.T) {
	_, _, c := loadedEngine(t, smallModel, "small", LoadOptions{})
	resp, err := http.Get(c.url + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	b := make([]byte, 16)
	n, _ := resp.Body.Read(b)
	resp.Body.Close()
	if resp.StatusCode != 200 || string(b[:n]) != "ok\n" {
		t.Fatalf("/healthz: %d %q", resp.StatusCode, b[:n])
	}
	resp, err = http.Get(c.url + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("/v1/models through Handler: %d", resp.StatusCode)
	}
}

// ---------------------------------------------------------------- convert

// TestConvertStreamsItsPhasesAndTheResultLoads: the conversion lands where the
// response says, carries the container version, and the server can load what
// it wrote. Refusals come back with the code a caller acts on.
func TestConvertStreamsItsPhasesAndTheResultLoads(t *testing.T) {
	src := modelPath(t, "stories260K.gguf")
	// Scratch conversions go beside the models, never under /tmp.
	dir, err := os.MkdirTemp(testmodels.Dir(), ".server-convert-test-")
	if err != nil {
		t.Fatalf("a scratch directory in the model directory: %v", err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	e := New(Config{ModelDir: dir, Probe: noProbe})
	defer e.Close()
	c := serveEngine(t, e)
	ctx := context.Background()

	convert := func(m *v1.ConvertRequest) ([]*v1.ConvertResponse, error) {
		st, err := c.model.Convert(ctx, req(m))
		if err != nil {
			return nil, err
		}
		defer st.Close()
		var out []*v1.ConvertResponse
		for st.Receive() {
			out = append(out, st.Msg())
		}
		return out, st.Err()
	}

	dst := filepath.Join(dir, "out.jlm")
	got, err := convert(&v1.ConvertRequest{SourcePath: src, OutputPath: dst})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) < 2 || got[0].GetPhase() != "started" {
		t.Fatalf("phases %v: the first must be started", got)
	}
	last := got[len(got)-1]
	if !last.GetDone() || last.GetPhase() != "done" || last.GetFraction() != 1 || last.GetOutputPath() != dst ||
		last.GetOutputSize().GetBytes() == 0 || !strings.Contains(last.GetDetail(), "container v") {
		t.Fatalf("the last frame reads %+v", last)
	}
	fi, err := os.Stat(dst)
	if err != nil || uint64(fi.Size()) != last.GetOutputSize().GetBytes() {
		t.Fatalf("the response says %d bytes at %s; the file: %v %v", last.GetOutputSize().GetBytes(), dst, fi, err)
	}
	if _, err := c.model.LoadModel(ctx, req(&v1.LoadModelRequest{Path: "out.jlm"})); err != nil {
		t.Fatalf("the server cannot load the container it just wrote: %v", err)
	}

	_, err = convert(&v1.ConvertRequest{SourcePath: src, OutputPath: dst})
	wantCode(t, "a convert onto an existing file without overwrite", err, connect.CodeAlreadyExists)
	_, err = convert(&v1.ConvertRequest{})
	wantCode(t, "a convert with no source", err, connect.CodeInvalidArgument)
	_, err = convert(&v1.ConvertRequest{SourcePath: "no-such.gguf"})
	wantCode(t, "a convert of a file that does not exist", err, connect.CodeNotFound)
}
