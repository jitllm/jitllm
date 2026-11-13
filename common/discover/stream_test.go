//go:build unix

package discover

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/jitllm/jitllm/common/discover/hubtest"
	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/convert/library"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// diskWatch samples the bytes in a directory until stopped: the peak, and
// every name that was ever there.
type diskWatch struct {
	dir  string
	mu   sync.Mutex
	peak int64
	seen map[string]bool
	stop chan struct{}
	done chan struct{}
}

func watchDisk(dir string) *diskWatch {
	w := &diskWatch{dir: dir, seen: map[string]bool{}, stop: make(chan struct{}), done: make(chan struct{})}
	go func() {
		defer close(w.done)
		for {
			w.sample()
			select {
			case <-w.stop:
				return
			case <-time.After(time.Millisecond):
			}
		}
	}()
	return w
}

// sample adds up the blocks the directory's files hold -- not their
// lengths: the container is sized up front and filled.
func (w *diskWatch) sample() {
	es, err := os.ReadDir(w.dir)
	if err != nil {
		return
	}
	var n int64
	w.mu.Lock()
	defer w.mu.Unlock()
	for _, e := range es {
		w.seen[e.Name()] = true
		if fi, err := e.Info(); err == nil {
			n += onDisk(fi)
		}
	}
	w.peak = max(w.peak, n)
}

func (w *diskWatch) end() {
	close(w.stop)
	<-w.done
	w.sample()
}

// onDisk is the blocks a file holds: jlm.Write sizes its part sparsely and
// fills it.
func onDisk(fi os.FileInfo) int64 {
	if st, ok := fi.Sys().(*syscall.Stat_t); ok {
		return st.Blocks * 512
	}
	return fi.Size()
}

// reference is the container src converts to from the disk.
func reference(t *testing.T, src string) []byte {
	t.Helper()
	dst := filepath.Join(t.TempDir(), "ref.jlm")
	host, _ := os.Hostname()
	if _, err := convert.FromGGUF(src, dst, jlm.Fingerprint{Host: host}); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(dst)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func served(t *testing.T, path string) (library.Model, *hubtest.Hub) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS) -- this gate proves nothing without it", err)
	}
	m := library.Model{Name: "gate", Title: "Gate", Repo: "gate/model", File: filepath.Base(path), Bytes: int64(len(b))}
	h := hubtest.New(map[string][]byte{m.Repo + "/" + m.File: b})
	t.Cleanup(h.Close)
	return m, h
}

// checkDisk demands the model folder never held the GGUF, and never more
// than the container: the peak a download-then-convert reaches is the GGUF
// and the container together.
func checkDisk(t *testing.T, w *diskWatch, m library.Model, container int) {
	t.Helper()
	for name := range w.seen {
		if strings.Contains(name, ".gguf") {
			t.Errorf("%s was written to the model folder: the GGUF was stored, not streamed", name)
		}
	}
	limit := int64(container) + int64(container)/20 + 1<<20
	t.Logf("peak %d bytes on disk for a %d byte GGUF and a %d byte container", w.peak, m.Bytes, container)
	if w.peak > limit {
		t.Errorf("the model folder peaked at %d bytes; the container is %d and the GGUF %d", w.peak, container, m.Bytes)
	}
}

// TestGetStreamsTheGGUF is the apps' download: the GGUF is converted as it
// arrives and never written, so the folder never holds more than the
// container, and the container is the one the GGUF converts to from disk.
func TestGetStreamsTheGGUF(t *testing.T) {
	if !gguf.Streamable() {
		t.Skip("this platform cannot stream a GGUF -- run this gate on Linux or macOS")
	}
	testmodels.SourceTree(t)
	src := "../../testdata/models/stories260K.gguf"
	m, hub := served(t, src)
	want := reference(t, src)
	dir := t.TempDir()
	w := watchDisk(dir)
	var stages []string
	line, err := Get(context.Background(), m, dir, Hub{Endpoint: hub.URL}, func(s Step) {
		stages = append(stages, s.Stage)
	})
	w.end()
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(ContainerFor(dir, m))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("the streamed container differs from the GGUF's conversion on disk")
	}
	if len(stages) == 0 || stages[0] != "streaming" {
		t.Errorf("the progress said %v; a stream says streaming", stages)
	}
	if !strings.Contains(line, "written by") {
		t.Errorf("no summary line: %q", line)
	}
	checkDisk(t, w, m, len(want))
}

// TestGetResumesACancelledStream cancels a stream halfway, then asks again:
// the second call reads what the first had not written, and the container is
// the GGUF's conversion. The disk never holds the GGUF across either.
func TestGetResumesACancelledStream(t *testing.T) {
	if !gguf.Streamable() {
		t.Skip("this platform cannot stream a GGUF -- run this gate on Linux or macOS")
	}
	src := testmodels.Path("Qwen3-0.6B-Q8_0.gguf")
	m, hub := served(t, src)
	want := reference(t, src)
	dir := t.TempDir()
	w := watchDisk(dir)

	ctx, cancel := context.WithCancel(context.Background())
	hub.OnGet = func(*http.Request) {
		if hub.Served.Load() > m.Bytes/2 {
			cancel()
		}
	}
	_, err := Get(ctx, m, dir, Hub{Endpoint: hub.URL}, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("a cancelled stream returned %v", err)
	}
	dst := ContainerFor(dir, m)
	if n, ok := jlm.Resumable(dst); !ok || n == 0 {
		t.Fatalf("a cancelled stream left nothing to resume (journal %d, %v)", n, ok)
	}
	first := hub.Served.Load()

	hub.OnGet = nil
	hub.Served.Store(0)
	if _, err := Get(context.Background(), m, dir, Hub{Endpoint: hub.URL}, nil); err != nil {
		t.Fatal(err)
	}
	w.end()
	second := hub.Served.Load()
	t.Logf("first call read %d bytes, the resumed one %d, of %d", first, second, m.Bytes)
	// The resumed call starts at the journal, behind the cancel by up to a
	// journal's spacing and the tensor that was in flight.
	if second > m.Bytes*9/10 || first+second > m.Bytes*3/2 {
		t.Errorf("the resumed call read %d of %d bytes after the first read %d: it started over",
			second, m.Bytes, first)
	}
	got, err := os.ReadFile(dst)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("the resumed container differs from the GGUF's conversion on disk")
	}
	if _, ok := jlm.Resumable(dst); ok {
		t.Error("a finished stream left its journal")
	}
	checkDisk(t, w, m, len(want))
}
