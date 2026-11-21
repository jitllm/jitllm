package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

const guardKey = "s3cret-key"

func guardEngine(t *testing.T) *Engine {
	t.Helper()
	e := New(admitConfig(Config{ModelDir: t.TempDir(), Probe: func() ([]DeviceInfo, error) { return nil, nil }}))
	t.Cleanup(e.Close)
	return e
}

// call sends one request straight into h and returns the recorded answer.
func call(h http.Handler, method, path, host string, hdr map[string]string) *httptest.ResponseRecorder {
	var body *strings.Reader
	if method == http.MethodPost {
		body = strings.NewReader("{}")
	} else {
		body = strings.NewReader("")
	}
	r := httptest.NewRequest(method, path, body)
	if host != "" {
		r.Host = host
	}
	r.Header.Set("Content-Type", "application/json")
	for k, v := range hdr {
		r.Header.Set(k, v)
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}

// routes are one route of each API, with the Connect protocol header it needs.
func guardRoutes() []struct {
	name, method, path string
	hdr                map[string]string
} {
	return []struct {
		name, method, path string
		hdr                map[string]string
	}{
		{"openai", http.MethodGet, "/v1/models", nil},
		{"anthropic", http.MethodPost, "/v1/messages", nil},
		{"connect", http.MethodPost, "/jitllm.v1.TelemetryService/GetServerInfo", map[string]string{"Connect-Protocol-Version": "1"}},
	}
}

// guardMisses runs every check the guard exists for against h and returns the
// ones h fails to make. A guard that works returns none; a handler without
// one (or with one check switched off) returns the checks it lets through.
func guardMisses(h http.Handler) []string {
	var missed []string
	merge := func(a, b map[string]string) map[string]string {
		m := map[string]string{}
		for k, v := range a {
			m[k] = v
		}
		for k, v := range b {
			m[k] = v
		}
		return m
	}
	for _, rt := range guardRoutes() {
		if w := call(h, rt.method, rt.path, "127.0.0.1:4242", rt.hdr); w.Code != http.StatusUnauthorized {
			missed = append(missed, rt.name+": no key was refused")
		}
		if w := call(h, rt.method, rt.path, "127.0.0.1:4242", merge(rt.hdr, map[string]string{"Authorization": "Bearer nope"})); w.Code != http.StatusUnauthorized {
			missed = append(missed, rt.name+": a wrong key was refused")
		}
		if w := call(h, rt.method, rt.path, "evil.example:4242", merge(rt.hdr, map[string]string{"Authorization": "Bearer " + guardKey})); w.Code != http.StatusForbidden {
			missed = append(missed, rt.name+": a rebound Host header was refused")
		}
		if w := call(h, rt.method, rt.path, "127.0.0.1:4242", merge(rt.hdr, map[string]string{
			"Authorization": "Bearer " + guardKey, "Origin": "https://evil.example"})); w.Code != http.StatusForbidden {
			missed = append(missed, rt.name+": a cross-origin request was refused")
		}
	}
	return missed
}

func TestGuardRefusesWhatItExistsFor(t *testing.T) {
	e := guardEngine(t)
	g := Guard{APIKey: guardKey, Addr: "127.0.0.1:4242"}
	if m := guardMisses(g.Wrap(e.Handler())); len(m) != 0 {
		t.Fatalf("the guard let these through: %v", m)
	}
}

// TestGuardGateDiscriminates: the gate above fails when a check is off. The
// bare handler checks nothing; a guard with no Addr checks no Host; one with
// no key checks no key.
func TestGuardGateDiscriminates(t *testing.T) {
	e := guardEngine(t)
	for name, h := range map[string]http.Handler{
		"no guard":      e.Handler(),
		"no host check": Guard{APIKey: guardKey}.Wrap(e.Handler()),
		"no key check":  Guard{Addr: "127.0.0.1:4242"}.Wrap(e.Handler()),
	} {
		if m := guardMisses(h); len(m) == 0 {
			t.Errorf("%s: the gate passed a handler with a check missing", name)
		}
	}
}

func TestGuardAcceptsTheKeyOnEveryAPI(t *testing.T) {
	g := Guard{APIKey: guardKey, Addr: "127.0.0.1:4242"}
	h := g.Wrap(guardEngine(t).Handler())
	for _, rt := range guardRoutes() {
		hdr := map[string]string{"Authorization": "Bearer " + guardKey}
		for k, v := range rt.hdr {
			hdr[k] = v
		}
		if w := call(h, rt.method, rt.path, "localhost:4242", hdr); w.Code == http.StatusUnauthorized || w.Code == http.StatusForbidden {
			t.Errorf("%s: the right key answered %d: %s", rt.name, w.Code, w.Body)
		}
	}
	// The Anthropic API's own header works on its routes, and only there.
	if w := call(h, http.MethodPost, "/v1/messages", "127.0.0.1:4242", map[string]string{"x-api-key": guardKey}); w.Code == http.StatusUnauthorized {
		t.Errorf("x-api-key refused on /v1/messages: %s", w.Body)
	}
	if w := call(h, http.MethodGet, "/v1/models", "127.0.0.1:4242", map[string]string{"x-api-key": guardKey}); w.Code != http.StatusUnauthorized {
		t.Errorf("x-api-key opened an OpenAI route: %d", w.Code)
	}
}

func TestGuardHealthzIsOpen(t *testing.T) {
	h := Guard{APIKey: guardKey, Addr: "127.0.0.1:4242"}.Wrap(guardEngine(t).Handler())
	if w := call(h, http.MethodGet, "/healthz", "127.0.0.1:4242", nil); w.Code != http.StatusOK {
		t.Fatalf("/healthz needs no key, answered %d", w.Code)
	}
	// The key protects the manifest, which names the build and the routes.
	if w := call(h, http.MethodGet, "/v1/capabilities", "127.0.0.1:4242", nil); w.Code != http.StatusUnauthorized {
		t.Fatalf("/v1/capabilities answered %d without a key", w.Code)
	}
}

// TestGuardAnswersInEachAPIsShape: a client library parses a refusal as it
// parses any other error.
func TestGuardAnswersInEachAPIsShape(t *testing.T) {
	h := Guard{APIKey: guardKey, Addr: "127.0.0.1:4242"}.Wrap(guardEngine(t).Handler())
	decode := func(path string, hdr map[string]string) map[string]any {
		w := call(h, http.MethodPost, path, "127.0.0.1:4242", hdr)
		if w.Code != http.StatusUnauthorized {
			t.Fatalf("%s: %d", path, w.Code)
		}
		var m map[string]any
		if err := json.Unmarshal(w.Body.Bytes(), &m); err != nil {
			t.Fatalf("%s: %v: %s", path, err, w.Body)
		}
		return m
	}
	oa := decode("/v1/chat/completions", nil)
	if e, _ := oa["error"].(map[string]any); e == nil || e["code"] != "invalid_api_key" || e["message"] == "" {
		t.Errorf("openai shape: %v", oa)
	}
	an := decode("/v1/messages", nil)
	if e, _ := an["error"].(map[string]any); an["type"] != "error" || e == nil || e["type"] != "authentication_error" {
		t.Errorf("anthropic shape: %v", an)
	}
	cn := decode("/jitllm.v1.ModelService/ListModels", nil)
	if cn["code"] != "unauthenticated" {
		t.Errorf("connect shape: %v", cn)
	}
}

func TestGuardHostAndOriginRules(t *testing.T) {
	loop := Guard{Addr: "127.0.0.1:4242", AllowOrigins: []string{"https://app.example.com"}}.Wrap(guardEngine(t).Handler())
	for _, c := range []struct {
		host, origin string
		want         int
	}{
		{"127.0.0.1:4242", "", 200},
		{"localhost:4242", "", 200},
		{"[::1]:4242", "", 200},
		{"evil.example", "", 403},
		{"127.0.0.1:4242", "http://localhost:3000", 200},
		{"127.0.0.1:4242", "http://127.0.0.1:5173", 200},
		{"127.0.0.1:4242", "https://evil.example", 403},
		{"127.0.0.1:4242", "null", 403},
		{"127.0.0.1:4242", "https://app.example.com", 200},
	} {
		hdr := map[string]string{}
		if c.origin != "" {
			hdr["Origin"] = c.origin
		}
		if w := call(loop, http.MethodGet, "/healthz", c.host, hdr); w.Code != c.want {
			t.Errorf("Host %q Origin %q: %d, want %d", c.host, c.origin, w.Code, c.want)
		}
	}

	// A listed origin gets CORS headers and its preflight is answered before
	// any key is asked for; an unlisted loopback origin gets none.
	keyed := Guard{APIKey: guardKey, Addr: "127.0.0.1:4242", AllowOrigins: []string{"https://app.example.com"}}.Wrap(guardEngine(t).Handler())
	w := call(keyed, http.MethodOptions, "/v1/chat/completions", "127.0.0.1:4242", map[string]string{
		"Origin": "https://app.example.com", "Access-Control-Request-Method": "POST",
		"Access-Control-Request-Headers": "authorization,content-type"})
	if w.Code != http.StatusNoContent || w.Header().Get("Access-Control-Allow-Origin") != "https://app.example.com" {
		t.Errorf("preflight: %d %v", w.Code, w.Header())
	}
	w = call(keyed, http.MethodGet, "/healthz", "127.0.0.1:4242", map[string]string{"Origin": "http://localhost:3000"})
	if w.Header().Get("Access-Control-Allow-Origin") != "" {
		t.Errorf("an unlisted origin got CORS headers")
	}

	// Off loopback the Host header says nothing, so it is not checked.
	open := Guard{Addr: ":4242"}.Wrap(guardEngine(t).Handler())
	if w := call(open, http.MethodGet, "/healthz", "box.lan:4242", nil); w.Code != 200 {
		t.Errorf("a server on every interface refused Host box.lan: %d", w.Code)
	}
}

func TestLoopback(t *testing.T) {
	for addr, want := range map[string]bool{
		"127.0.0.1:4242": true, "localhost:4242": true, "[::1]:4242": true,
		":4242": false, "0.0.0.0:4242": false, "192.0.2.5:4242": false, "[::]:4242": false,
	} {
		if got := Loopback(addr); got != want {
			t.Errorf("Loopback(%q) = %v", addr, got)
		}
	}
}
