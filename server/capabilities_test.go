package server

import (
	"bytes"
	"encoding/json"
	"io"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/capabilities"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestAPIManifestIsGenerated holds internal/capabilities/api.json to the
// route table and request structs: the file must be exactly what API()
// renders now.
func TestAPIManifestIsGenerated(t *testing.T) {
	testmodels.SourceTree(t)
	want, err := capabilities.RenderAPI(API())
	if err != nil {
		t.Fatal(err)
	}
	have, err := os.ReadFile(filepath.Join("..", filepath.FromSlash(capabilities.APIPath)))
	if err != nil {
		t.Fatalf("%v; run: %s", err, capabilities.Command)
	}
	if !bytes.Equal(have, want) {
		t.Fatalf("%s is not what the server's tables give; run: %s", capabilities.APIPath, capabilities.Command)
	}
}

// coverage is the gate's body: every reflected field has a note and every
// note a field.
func coverage(notes map[string]map[string]string) []string {
	var bad []string
	for _, r := range compatRoutes() {
		if r.req == nil {
			continue
		}
		got := map[string]bool{}
		for _, f := range fieldsOf(r.path, r.req) {
			got[f.Name] = true
			if strings.TrimSpace(notes[r.path][f.Name]) == "" {
				bad = append(bad, r.path+": field "+f.Name+" has no note in fieldNotes")
			}
		}
		for n := range notes[r.path] {
			if !got[n] {
				bad = append(bad, r.path+": fieldNotes describes "+n+", which the request struct does not have")
			}
		}
	}
	for p := range notes {
		found := false
		for _, r := range compatRoutes() {
			found = found || (r.path == p && r.req != nil)
		}
		if !found {
			bad = append(bad, "fieldNotes describes route "+p+", which takes no request struct")
		}
	}
	sort.Strings(bad)
	return bad
}

// TestAPIManifestCoversEveryField fails when a request field lands without a
// note saying what it does, or a note outlives its field; and is run against
// both violations.
func TestAPIManifestCoversEveryField(t *testing.T) {
	for _, b := range coverage(fieldNotes()) {
		t.Error(b)
	}
	// Violations: a field with its note taken away, and a note for a field
	// that does not exist.
	drop := map[string]map[string]string{}
	for p, m := range fieldNotes() {
		drop[p] = map[string]string{}
		for k, v := range m {
			drop[p][k] = v
		}
	}
	delete(drop["/v1/chat/completions"], "tools")
	if len(coverage(drop)) == 0 {
		t.Error("a field with no note passed the gate")
	}
	delete(drop["/v1/chat/completions"], "no_such")
	drop["/v1/chat/completions"]["tools"] = "x"
	drop["/v1/chat/completions"]["no_such"] = "x"
	if len(coverage(drop)) == 0 {
		t.Error("a note for no field passed the gate")
	}
}

// TestRouteTableIsWhatIsServed checks every listed route answers on Handler
// rather than falling to the not-found handler, and that an unlisted path
// does not -- so the table and the mux cannot part.
func TestRouteTableIsWhatIsServed(t *testing.T) {
	e := New(Config{Probe: oneCardProbe})
	defer e.Close()
	h := e.Handler()
	for _, ep := range API().Endpoints {
		path := ep.Path
		if strings.HasSuffix(path, "/") {
			path += "x"
		}
		req := httptest.NewRequest(ep.Methods[0], path, strings.NewReader("{}"))
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code == http.StatusNotFound && strings.Contains(rec.Body.String(), "404 page not found") {
			t.Errorf("%s %s is listed and not served", ep.Methods[0], ep.Path)
		}
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/not-a-route", nil))
	if rec.Code != http.StatusNotFound {
		t.Errorf("an unlisted path answered %d: the probe above cannot tell served from not", rec.Code)
	}
}

// TestCapabilitiesEndpoint serves the manifest with the build's identity and
// the route table it is serving.
func TestCapabilitiesEndpoint(t *testing.T) {
	e := New(Config{Probe: oneCardProbe, Version: "v1.2.3"})
	defer e.Close()
	rec := httptest.NewRecorder()
	e.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/capabilities", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var m capabilities.Manifest
	if err := json.Unmarshal(rec.Body.Bytes(), &m); err != nil {
		t.Fatal(err)
	}
	if m.Build == nil || m.Build.Version != "v1.2.3" {
		t.Fatalf("build is %+v", m.Build)
	}
	if len(m.Architectures) == 0 || len(m.API.Endpoints) != len(compatRoutes())+len(engineRoutes()) || len(m.API.Control) == 0 {
		t.Fatalf("the served manifest is short: %d architectures, %d endpoints, %d services",
			len(m.Architectures), len(m.API.Endpoints), len(m.API.Control))
	}
	rec = httptest.NewRecorder()
	e.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/capabilities", nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("POST answered %d", rec.Code)
	}
}

// probeRefusal posts one refusal's probe and says what went wrong, if
// anything.
func probeRefusal(t *testing.T, srv *httptest.Server, path string, r refusal) string {
	t.Helper()
	resp, err := http.Post(srv.URL+path, "application/json", strings.NewReader(r.probe))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusBadRequest {
		return "status " + resp.Status + ": " + string(body)
	}
	// Both APIs carry the text at error.message.
	var e struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &e); err != nil {
		return "the body is not an error object: " + string(body)
	}
	if !strings.Contains(e.Error.Message, r.Message) {
		return "the body does not carry " + r.Message + ": " + string(body)
	}
	return ""
}

// sourceCarries reports whether a production .go file of this package returns
// msg.
func sourceCarries(t *testing.T, msg string) bool {
	t.Helper()
	found := false
	err := fs.WalkDir(os.DirFS("."), ".", func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && p != "." {
			return fs.SkipDir
		}
		if d.IsDir() || !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") || p == "capabilities.go" {
			return nil
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		found = found || bytes.Contains(b, []byte(msg))
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return found
}

// TestRefusalsAreRefused holds the manifest's refusals to the code: each with
// a probe is posted to the route (no engine; the refusal is the route's own)
// and must draw a 400 carrying its message, and each engine-level one must be
// a message the package's source returns. Run against violations: a probe
// that is valid, and a message no code returns.
func TestRefusalsAreRefused(t *testing.T) {
	testmodels.SourceTree(t)
	srv := httptest.NewServer(CompatHandler(&fakeBackend{tokens: []string{"x"}, reason: FinishStop}))
	defer srv.Close()
	for _, r := range refusalList() {
		if len(r.Endpoints) == 0 || r.ID == "" || r.When == "" || r.Message == "" {
			t.Errorf("refusal %+v is incomplete", r.Refusal)
			continue
		}
		if r.probe == "" {
			if !sourceCarries(t, r.Message) {
				t.Errorf("%s: no production source returns %q", r.ID, r.Message)
			}
			continue
		}
		if why := probeRefusal(t, srv, r.Endpoints[0], r); why != "" {
			t.Errorf("%s on %s: %s", r.ID, r.Endpoints[0], why)
		}
	}

	valid := refusalList()[0]
	valid.probe = `{"model":"m","messages":[{"role":"user","content":"hi"}]}`
	if probeRefusal(t, srv, valid.Endpoints[0], valid) == "" {
		t.Error("a valid request passed as a refusal")
	}
	if sourceCarries(t, "speculation with logprobs is a thing nobody refuses") {
		t.Error("a message no code returns was found")
	}
}
