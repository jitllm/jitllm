package server

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"connectrpc.com/connect"

	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// modelsFolder is a temporary models folder holding stories260K under each
// of names (file names without .jlm).
func modelsFolder(t *testing.T, names ...string) string {
	t.Helper()
	src := modelPath(t, smallModel)
	b, err := os.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	for _, n := range names {
		if err := os.WriteFile(filepath.Join(dir, n+".jlm"), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func folderEngine(t *testing.T, dir string) (*Engine, clients) {
	t.Helper()
	e := New(admitConfig(Config{ModelDir: dir, Probe: oneCardProbe, Version: "test"}))
	t.Cleanup(e.Close)
	return e, serveEngine(t, e)
}

type v1Models struct {
	Data []struct {
		ID      string   `json:"id"`
		Loaded  bool     `json:"loaded"`
		Aliases []string `json:"aliases"`
	} `json:"data"`
}

func listV1(t *testing.T, url string) v1Models {
	t.Helper()
	resp := do(t, "GET", url+"/v1/models", "")
	defer resp.Body.Close()
	var ml v1Models
	if err := json.NewDecoder(resp.Body).Decode(&ml); err != nil {
		t.Fatal(err)
	}
	return ml
}

func completeBy(t *testing.T, url, model string) (int, string) {
	t.Helper()
	resp := do(t, "POST", url+oaText, `{"model":"`+model+`","prompt":"`+story+`","max_tokens":3}`)
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b)
}

// TestAModelLoadsOnDemandByAlias: a container in the models folder is listed
// before it is loaded, a request naming it by alias loads it, and the Connect
// API reaches the same model by file name without loading it twice.
func TestAModelLoadsOnDemandByAlias(t *testing.T) {
	dir := modelsFolder(t, "stories260K")
	e, c := folderEngine(t, dir)
	ctx := context.Background()

	if _, err := c.model.SetModelOptions(ctx, req(&v1.SetModelOptionsRequest{Name: "stories260K", AddAliases: []string{"tiny"}})); err != nil {
		t.Fatal(err)
	}
	ml := listV1(t, c.url)
	ids := map[string]bool{}
	for _, d := range ml.Data {
		ids[d.ID] = true
		if d.Loaded {
			t.Errorf("%s listed as loaded before any request", d.ID)
		}
	}
	if !ids["stories260K"] || !ids["tiny"] {
		t.Fatalf("/v1/models lists %+v, want the file and its alias", ml.Data)
	}
	if len(e.Models()) != 0 {
		t.Fatal("a model was loaded by listing")
	}

	if code, body := completeBy(t, c.url, "tiny"); code != http.StatusOK {
		t.Fatalf("completion by alias: %d %s", code, body)
	}
	if ms := e.Models(); len(ms) != 1 || ms[0].ID() != "stories260K" {
		t.Fatalf("after the request, loaded %v, want stories260K", ms)
	}

	st, err := c.inference.Generate(ctx, req(&v1.GenerateRequest{ModelId: "stories260K.jlm", MaxTokens: 2,
		Prompt: text(story)}))
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for st.Receive() {
		n++
	}
	if err := st.Err(); err != nil || n == 0 {
		t.Fatalf("Generate by file name: %d events, %v", n, err)
	}
	if len(e.Models()) != 1 {
		t.Fatalf("Generate by file name loaded a second copy: %d models", len(e.Models()))
	}
	lm, err := c.model.ListModels(ctx, req(&v1.ListModelsRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	if f := lm.Msg.GetFiles(); len(f) != 1 || f[0].GetModelId() != "stories260K" || strings.Join(f[0].GetAliases(), ",") != "tiny" {
		t.Fatalf("ListModels files %v", f)
	}
	if l := lm.Msg.GetLoaded(); len(l) != 1 || strings.Join(l[0].GetAliases(), ",") != "tiny" {
		t.Fatalf("ListModels loaded %v", l)
	}
}

// TestADisabledModelIsNotLoadedOnDemand: a disabled model is listed by
// ListModels, absent from /v1/models, refused by name with a 400 that says
// it is disabled, and still loads when loaded explicitly.
func TestADisabledModelIsNotLoadedOnDemand(t *testing.T) {
	dir := modelsFolder(t, "stories260K")
	e, c := folderEngine(t, dir)
	ctx := context.Background()
	off := false
	if _, err := c.model.SetModelOptions(ctx, req(&v1.SetModelOptionsRequest{Name: "stories260K.jlm", Enabled: &off})); err != nil {
		t.Fatal(err)
	}
	if ml := listV1(t, c.url); len(ml.Data) != 0 {
		t.Errorf("/v1/models lists a disabled model: %+v", ml.Data)
	}
	code, body := completeBy(t, c.url, "stories260K")
	if code != http.StatusBadRequest || !strings.Contains(body, "disabled") {
		t.Fatalf("a disabled model: %d %s, want a 400 naming it disabled", code, body)
	}
	if len(e.Models()) != 0 {
		t.Fatal("a disabled model was loaded on demand")
	}
	lm, err := c.model.ListModels(ctx, req(&v1.ListModelsRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	if f := lm.Msg.GetFiles(); len(f) != 1 || !f[0].GetDisabled() {
		t.Fatalf("ListModels files %v, want the model listed as disabled", f)
	}
	if _, err := c.model.LoadModel(ctx, req(&v1.LoadModelRequest{Path: "stories260K.jlm"})); err != nil {
		t.Fatalf("an explicit load of a disabled model: %v", err)
	}
}

// TestAliasesAreUniqueAndSurviveARestart: an alias may not name another
// model (by its alias, file name or stem), and the aliases and the disabled
// flag are read back by a new engine on the same folder.
func TestAliasesAreUniqueAndSurviveARestart(t *testing.T) {
	dir := modelsFolder(t, "a", "b")
	_, c := folderEngine(t, dir)
	ctx := context.Background()
	if _, err := c.model.SetModelOptions(ctx, req(&v1.SetModelOptionsRequest{Name: "a", AddAliases: []string{"x", "y"}})); err != nil {
		t.Fatal(err)
	}
	for _, a := range []string{"x", "a", "a.jlm"} {
		_, err := c.model.SetModelOptions(ctx, req(&v1.SetModelOptionsRequest{Name: "b", AddAliases: []string{a}}))
		wantCode(t, "alias "+a+" on b", err, connect.CodeAlreadyExists)
	}
	_, err := c.model.SetModelOptions(ctx, req(&v1.SetModelOptionsRequest{Name: "b", AddAliases: []string{"has/slash"}}))
	wantCode(t, "an alias with a slash", err, connect.CodeInvalidArgument)
	off := false
	if _, err := c.model.SetModelOptions(ctx, req(&v1.SetModelOptionsRequest{Name: "y", RemoveAliases: []string{"y"}})); err != nil {
		t.Fatal(err)
	}
	if _, err := c.model.SetModelOptions(ctx, req(&v1.SetModelOptionsRequest{Name: "b", Enabled: &off})); err != nil {
		t.Fatal(err)
	}

	_, c2 := folderEngine(t, dir)
	lm, err := c2.model.ListModels(ctx, req(&v1.ListModelsRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]*v1.ModelFile{}
	for _, f := range lm.Msg.GetFiles() {
		got[f.GetName()] = f
	}
	if strings.Join(got["a"].GetAliases(), ",") != "x" || got["a"].GetDisabled() {
		t.Errorf("a after a restart: %v", got["a"])
	}
	if !got["b"].GetDisabled() {
		t.Errorf("b after a restart is enabled: %v", got["b"])
	}
	if code, body := completeBy(t, c2.url, "x"); code != http.StatusOK {
		t.Fatalf("the alias after a restart: %d %s", code, body)
	}
}

// TestConcurrentRequestsLoadAModelOnce: requests arriving together for a
// model nobody loaded share one load.
func TestConcurrentRequestsLoadAModelOnce(t *testing.T) {
	dir := modelsFolder(t, "stories260K")
	e, c := folderEngine(t, dir)
	errs := make(chan string, 4)
	for range 4 {
		go func() {
			resp, err := http.Post(c.url+oaText, "application/json",
				strings.NewReader(`{"model":"stories260K","prompt":"`+story+`","max_tokens":2}`))
			if err != nil {
				errs <- err.Error()
				return
			}
			resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				errs <- resp.Status
				return
			}
			errs <- ""
		}()
	}
	for range 4 {
		if e := <-errs; e != "" {
			t.Error(e)
		}
	}
	if n := len(e.Models()); n != 1 {
		t.Fatalf("%d models loaded, want one", n)
	}
}

// TestAnUnknownModelNamesTheFolder: a name that is neither loaded nor in the
// folder is a 404 that says where it looked.
func TestAnUnknownModelNamesTheFolder(t *testing.T) {
	dir := modelsFolder(t, "stories260K")
	_, c := folderEngine(t, dir)
	code, body := completeBy(t, c.url, "nope")
	// The message as the client reads it: JSON-decoded, so a Windows path's
	// backslashes are single again, and the folder as the engine names it.
	var e struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(body), &e); err != nil {
		t.Fatalf("an unknown model: %d %s: %v", code, body, err)
	}
	if code != http.StatusNotFound || !strings.Contains(e.Error.Message, filepath.Clean(dir)) {
		t.Fatalf("an unknown model: %d %q, want a 404 naming %s", code, e.Error.Message, filepath.Clean(dir))
	}
}
