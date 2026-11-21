//go:build linux

package engine

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/jitllm/jitllm/common/api"
	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/server"
)

// The app's model is the API's model: one engine, so a model opened in the
// app is listed by the server and answers a request there beside the chat's
// session, without a second copy of it being opened.
func TestTheAppsModelIsTheServersModel(t *testing.T) {
	e, sh := loadFor(t, func(st *testStore) { st.DeviceSpec.Set("cpu") })
	if !sh.Store.Loaded.Get() {
		t.Fatalf("the model did not load: %s", sh.Store.Problem.Get().Title)
	}
	ms := e.Server().Models()
	if len(ms) != 1 || ms[0].Path() != modelPath() {
		t.Fatalf("the server lists %d model(s), want the app's one at %s", len(ms), modelPath())
	}
	if n := len(e.Server().Sessions()); n != 1 {
		t.Fatalf("the server holds %d session(s), want the chat's one", n)
	}

	var got int
	err := e.Server().Generate(context.Background(), server.GenerateOptions{
		ModelID:   ms[0].ID(),
		Prompt:    server.Prompt{Kind: server.PromptText, Text: "Once upon a time"},
		MaxTokens: 4,
	}, func(ev server.Event) error {
		if ev.Kind == server.EventToken && ev.Token.ID >= 0 {
			got++
		}
		return nil
	})
	if err != nil || got == 0 {
		t.Fatalf("a request on the app's model through the server: %d token(s), %v", got, err)
	}

	// The chat's session is untouched by the request beside it.
	reply := sh.Store.AppendTurn(session.Turn{Role: session.RoleAssistant})
	e.Send(session.ChatRequest{Prompt: "Once upon a time", Reply: reply, MaxTokens: 4})
	pump(t, sh, 90*time.Second, "the turn", func() bool { return !sh.Store.Busy.Get() })
	if tr := sh.Store.Turn(reply); tr.Tokens == 0 {
		t.Fatalf("the chat produced no tokens after the server's request: %+v", tr)
	}
}

// Served as the API, the app's engine lists the app's model over HTTP: the
// front end's toggle serves this engine, not one of its own.
func TestTheAPIServesTheAppsModel(t *testing.T) {
	e, sh := loadFor(t, func(st *testStore) { st.DeviceSpec.Set("cpu") })
	if !sh.Store.Loaded.Get() {
		t.Fatalf("the model did not load: %s", sh.Store.Problem.Get().Title)
	}
	s, err := api.Start("127.0.0.1:0", e.Server())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	r, err := http.Get("http://" + s.Addr() + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	var list struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(r.Body).Decode(&list); err != nil {
		t.Fatal(err)
	}
	// The loaded model, under its id and under its file name, which the shim
	// resolves to the same model.
	lm := e.Server().Models()[0]
	if len(list.Data) != 2 || list.Data[0].ID != lm.ID() || list.Data[1].ID != lm.Name() {
		t.Fatalf("/v1/models lists %+v, want the app's model as %q and %q", list.Data, lm.ID(), lm.Name())
	}
}
