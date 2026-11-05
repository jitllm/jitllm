package server

import (
	"context"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// TestSessionKnobsReachTheStateOverHTTP: the session knobs SessionOptions
// carries are on CreateSession's wire, they reach the instance, and a value no
// session can be made from is a 400 over plain HTTP, not a 500 and not a
// silent default. The server-wide KV width reaches a load that names none.
//
// VIOLATION SIGNATURE. Drop SessionOptions.check from CreateSession and the
// negative max_seq case fails with `status 200` (or a 500 from deeper down);
// drop the PromptCache mapping in CreateSession and "prompt_cache" fails
// with `the session has no prompt store`; drop cmp.Or's e.cfg.KVF16 in
// LoadModel and the KV width cases fail.
func TestSessionKnobsReachTheStateOverHTTP(t *testing.T) {
	path := modelPath(t, smallModel)
	for _, f16 := range []bool{true, false} {
		e := New(admitConfig(Config{ModelDir: filepath.Dir(path), Probe: oneCardProbe, Version: "test",
			KVF16: &f16, PromptStore: engine.NewMemStore()}))
		t.Cleanup(e.Close)
		if _, err := e.LoadModel(LoadOptions{Path: path, ModelID: "small"}); err != nil {
			t.Fatal(err)
		}
		c := serveEngine(t, e)
		post := func(body string) int {
			t.Helper()
			r, err := http.Post(c.url+"/jitllm.v1.SessionService/CreateSession", "application/json", strings.NewReader(body))
			if err != nil {
				t.Fatal(err)
			}
			r.Body.Close()
			return r.StatusCode
		}
		for _, bad := range []string{
			`{"modelId":"small","maxSeq":-1}`,
			`{"modelId":"small","maxDeviceBlocks":-2}`,
			`{"modelId":"small","relocateWhileServing":true,"keepOffHost":true}`,
			`{"modelId":"small","cacheKey":"k"}`,
		} {
			if got := post(bad); got != http.StatusBadRequest {
				t.Errorf("CreateSession %s: status %d, want 400", bad, got)
			}
		}
		if got := post(`{"modelId":"small","sessionId":"c","maxSeq":64,"keepOffHost":true,"promptCache":true,"cacheKey":"k"}`); got != http.StatusOK {
			t.Fatalf("a valid CreateSession: status %d", got)
		}
		s, err := e.Session("c")
		if err != nil {
			t.Fatal(err)
		}
		if !s.cached {
			t.Errorf("prompt_cache: the session has no prompt store")
		}
		if got := s.in().KVIsF16(); got != f16 {
			t.Errorf("Config.KVF16 %v: the session's cache is f16=%v", f16, got)
		}
		// The store is used: a generate writes the prompt's pages.
		if _, err := c.inference.Complete(context.Background(), req(&v1.GenerateRequest{SessionId: "c", Prompt: text(story + story + story + story + story), MaxTokens: 2})); err != nil {
			t.Fatal(err)
		}
	}

	// A server with no store refuses prompt_cache.
	e := New(admitConfig(Config{ModelDir: filepath.Dir(path), Probe: oneCardProbe, Version: "test"}))
	t.Cleanup(e.Close)
	if _, err := e.LoadModel(LoadOptions{Path: path, ModelID: "small"}); err != nil {
		t.Fatal(err)
	}
	c := serveEngine(t, e)
	r, err := http.Post(c.url+"/jitllm.v1.SessionService/CreateSession", "application/json",
		strings.NewReader(`{"modelId":"small","promptCache":true}`))
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
	if r.StatusCode != http.StatusBadRequest {
		t.Errorf("prompt_cache with no store: status %d, want 400", r.StatusCode)
	}
}
