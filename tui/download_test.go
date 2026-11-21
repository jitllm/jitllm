package main

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/jitllm/jitllm/common/convertjob"
	"github.com/jitllm/jitllm/common/discover"
	"github.com/jitllm/jitllm/common/discover/hubtest"
	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/convert/library"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// served is a library model whose GGUF is path, on a Hub on loopback that
// the terminal reads from.
func served(t *testing.T, r *root, path string) (library.Model, *hubtest.Hub, int64) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS) -- this gate proves nothing without it", err)
	}
	m := library.Model{Name: "gate", Title: "Gate", Repo: "gate/model", File: filepath.Base(path), Bytes: int64(len(b))}
	h := hubtest.New(map[string][]byte{m.Repo + "/" + m.File: b})
	t.Cleanup(h.Close)
	r.env.hub = h.URL
	return m, h, int64(len(b))
}

// ggufSeen records whether a GGUF was ever in dir, sampled on every range the
// Hub answers.
func ggufSeen(h *hubtest.Hub, dir string) func() []string {
	var mu sync.Mutex
	var seen []string
	h.OnGet = func(*http.Request) {
		es, _ := os.ReadDir(dir)
		mu.Lock()
		defer mu.Unlock()
		for _, e := range es {
			if strings.Contains(e.Name(), ".gguf") && e.Name() != "x.gguf" {
				seen = append(seen, e.Name())
			}
		}
	}
	return func() []string { mu.Lock(); defer mu.Unlock(); return seen }
}

// A download in the terminal streams the GGUF into its container: the model
// folder never holds the GGUF.
func TestDownloadStreamsIntoTheContainer(t *testing.T) {
	r := newTestRoot(t)
	m, h, _ := served(t, r, testmodels.Path("stories260K.gguf"))
	dir := discover.Dir(r.env.dirs, r.env.cfg.DefaultModelDir())
	seen := ggufSeen(h, dir)
	r.env.jobs.download(m)
	dst := discover.ContainerFor(dir, m)
	deadline := time.Now().Add(time.Minute)
	for !convertjob.LoadableAt(dst) {
		if time.Now().After(deadline) {
			t.Fatalf("no container at %s after a minute", dst)
		}
		time.Sleep(10 * time.Millisecond)
	}
	// Where a stream cannot run, the download falls back to fetching the
	// whole file and removing it once converted.
	if s := seen(); len(s) > 0 && gguf.Streamable() {
		t.Errorf("the GGUF was written to the model folder: %v", s)
	}
	if h.Served.Load() == 0 {
		t.Error("nothing was read from the Hub")
	}
}

// ctrl+x on Discover stops a download and keeps what it wrote; the next
// download of the model carries on from it.
func TestStopKeyStopsADownload(t *testing.T) {
	if !gguf.Streamable() {
		t.Skip("this platform downloads the whole GGUF, which ctrl+x does not interrupt")
	}
	r := newTestRoot(t)
	m, h, size := served(t, r, testmodels.Path("Qwen3-0.6B-Q8_0.gguf"))
	dir := discover.Dir(r.env.dirs, r.env.cfg.DefaultModelDir())
	dst := discover.ContainerFor(dir, m)
	r.env.jobs.download(m)
	for h.Served.Load() < size/2 {
		time.Sleep(time.Millisecond)
	}
	r.discover.Update(tea.KeyMsg{Type: tea.KeyCtrlX})
	// Stopped is the Hub going quiet with the container unfinished.
	for last := int64(-1); last != h.Served.Load(); {
		last = h.Served.Load()
		time.Sleep(300 * time.Millisecond)
	}
	if convertjob.LoadableAt(dst) {
		t.Fatal("the download ran to the end: ctrl+x did not stop it")
	}
	if n, ok := jlm.Resumable(dst); !ok || n == 0 {
		t.Fatalf("a stopped download kept nothing to carry on from (journal %d, %v)", n, ok)
	}
}
