package discover

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/common/convertjob"
	"github.com/jitllm/jitllm/common/discover/hubtest"
	"github.com/jitllm/jitllm/convert/hf"
	"github.com/jitllm/jitllm/convert/library"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// fallbackModel is a model the Hub at h holds only if b is not nil.
func fallbackModel(t *testing.T, b []byte) (library.Model, *hubtest.Hub) {
	m := library.Model{Name: "gate", Title: "Gate", Repo: "gate/model", File: "stories260K.gguf", Bytes: int64(len(b))}
	files := map[string][]byte{}
	if b != nil {
		files[m.Repo+"/"+m.File] = b
	}
	h := hubtest.New(files)
	t.Cleanup(h.Close)
	return m, h
}

// The download-first path (a platform that cannot stream) reads from the
// Hub it is given, with the fetch the apps pass it, and a gated repository
// there is the gated error -- not another Hub's answer.
func TestTheFallbackKeepsTheHubsRefusal(t *testing.T) {
	m, h := fallbackModel(t, []byte("x"))
	h.Gated = true
	_, err := Get(context.Background(), m, t.TempDir(), Hub{Endpoint: h.URL, Fetch: HubFetch, NoStream: true}, nil)
	if h.Requests.Load() == 0 {
		t.Fatalf("the fallback never asked the Hub it was given (err %v)", err)
	}
	if !errors.Is(err, hf.ErrAccess) {
		t.Fatalf("a gated model on the fallback returned %v, want the Hub's refusal", err)
	}
	if msg := FailMessage(m, err); !strings.Contains(msg, "gated or private") || !strings.Contains(msg, "HF_TOKEN") {
		t.Errorf("the message does not say the model is gated: %q", msg)
	}
}

// The download-first path ends as a container with its download removed.
func TestTheFallbackEndsAsAContainer(t *testing.T) {
	testmodels.SourceTree(t)
	b, err := os.ReadFile("../../testdata/models/stories260K.gguf")
	if err != nil {
		t.Fatal(err)
	}
	m, h := fallbackModel(t, b)
	dir := t.TempDir()
	if _, err := Get(context.Background(), m, dir, Hub{Endpoint: h.URL, Fetch: HubFetch, NoStream: true}, nil); err != nil {
		t.Fatal(err)
	}
	if h.Served.Load() == 0 {
		t.Fatal("nothing was read from the Hub it was given")
	}
	if !convertjob.LoadableAt(ContainerFor(dir, m)) {
		t.Fatal("no container")
	}
	if _, err := os.Stat(filepath.Join(dir, m.File)); err == nil {
		t.Error("the downloaded GGUF is still beside its container")
	}
}
