package server

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func fetch(t *testing.T, h http.Handler, path string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", path, nil))
	return rec
}

// TestWebUIIsServedUnderUIAndEverythingElsePassesThrough: the page and its
// script come from the embedded bundle, and a path outside /ui/ reaches the
// wrapped handler untouched.
//
// VIOLATION SIGNATURE. Drop the "/" route in WithWebUI and the API paths 404;
// embed an empty dist and the script fetch is a 404 or an empty body.
func TestWebUIIsServedUnderUIAndEverythingElsePassesThrough(t *testing.T) {
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, "api:"+r.URL.Path)
	})
	h := WithWebUI(next)

	page := fetch(t, h, "/ui/")
	if page.Code != 200 || !strings.Contains(page.Body.String(), `src="app.js"`) {
		t.Fatalf("/ui/ = %d %q", page.Code, page.Body.String())
	}
	js := fetch(t, h, "/ui/app.js")
	if js.Code != 200 || js.Body.Len() < 10_000 {
		t.Fatalf("/ui/app.js = %d, %d bytes: the bundle is not embedded", js.Code, js.Body.Len())
	}
	if r := fetch(t, h, "/ui"); r.Code != http.StatusTemporaryRedirect || r.Header().Get("Location") != "/ui/" {
		t.Fatalf("/ui = %d %q", r.Code, r.Header().Get("Location"))
	}
	if r := fetch(t, h, "/jitllm.v1.ModelService/ListModels"); r.Body.String() != "api:/jitllm.v1.ModelService/ListModels" {
		t.Fatalf("an API path was not passed through: %q", r.Body.String())
	}
}

// TestWebUIFilesNeedNoKeyButTheRPCsStillDo: with an API key set the page must
// load so it can ask for the key; the services behind it stay closed. Host
// checks still apply to the page.
//
// VIOLATION SIGNATURE. Remove /ui from openRoute and /ui/ answers 401, so a
// keyed server's page can never show its login box; open every route and the
// ListModels call answers 200 without a key.
func TestWebUIFilesNeedNoKeyButTheRPCsStillDo(t *testing.T) {
	g := Guard{APIKey: guardKey, Addr: "127.0.0.1:4242"}
	h := g.Wrap(WithWebUI(guardEngine(t).Handler()))
	if w := call(h, http.MethodGet, "/ui/", "127.0.0.1:4242", nil); w.Code != http.StatusOK {
		t.Fatalf("/ui/ answered %d without a key", w.Code)
	}
	if w := call(h, http.MethodPost, "/jitllm.v1.ModelService/ListModels", "127.0.0.1:4242", map[string]string{"Content-Type": "application/json"}); w.Code != http.StatusUnauthorized {
		t.Fatalf("ListModels answered %d without a key", w.Code)
	}
	if w := call(h, http.MethodGet, "/ui/", "evil.example:4242", nil); w.Code == http.StatusOK {
		t.Fatal("/ui/ answered a rebinding Host")
	}
}
