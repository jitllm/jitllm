package screen

import (
	"errors"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/common/convertjob"
	"github.com/jitllm/jitllm/common/discover"
	"github.com/jitllm/jitllm/common/discover/hubtest"
	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/convert/hf"
	"github.com/jitllm/jitllm/convert/library"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/ui/app"
)

// firstText is the first library entry with no vision tower.
func firstText(t *testing.T) (library.Model, int) {
	for i, m := range library.Models() {
		if !m.Vision() {
			return m, i
		}
	}
	t.Fatal("the library has no text model")
	return library.Model{}, -1
}

// waitIdle drains the UI queue until the download screen is no longer busy.
func waitIdle(t *testing.T, sh *app.Shell, d *downloadScreen) {
	t.Helper()
	deadline := time.Now().Add(60 * time.Second)
	for time.Now().Before(deadline) {
		sh.DrainQueue(0)
		if !d.busy.Get() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("still busy after a minute; note %q", d.note.Get())
}

// Discover lists every library model exactly once, whatever the balance.
func TestDiscoverListsTheWholeLibrary(t *testing.T) {
	d := downloadsFor(dropShell(t))
	for pref := range discover.BalanceSteps {
		d.pref.Set(pref)
		seen := map[int]bool{}
		for _, i := range d.order().Get() {
			if seen[i] {
				t.Fatalf("balance %d lists %s twice", pref, library.Models()[i].Title)
			}
			seen[i] = true
		}
		if len(seen) != len(library.Models()) {
			t.Fatalf("balance %d lists %d of %d models", pref, len(seen), len(library.Models()))
		}
	}
}

// The disk is checked before a byte moves.
func TestADownloadThatWouldFillTheDiskDoesNotStart(t *testing.T) {
	sh := dropShell(t)
	d := downloadsFor(sh)
	m, i := firstText(t)
	d.sel.Set(i)
	fetched := false
	defer func(f func(*hf.Client, hf.Ref, string) (string, error), g func(string) (int64, bool)) {
		downloadFetch, downloadFree = f, g
	}(downloadFetch, downloadFree)
	downloadFetch = func(*hf.Client, hf.Ref, string) (string, error) { fetched = true; return "", nil }
	downloadFree = func(string) (int64, bool) { return m.Bytes, true } // half of what it needs

	d.act()
	if fetched || d.busy.Get() {
		t.Fatal("a download started with too little disk")
	}
	if !strings.Contains(d.note.Get(), "Not enough disk space") {
		t.Errorf("the note does not say why: %q", d.note.Get())
	}
}

// hubFor serves m's GGUF as b from a Hub on loopback, for the screen's
// downloads, until the test ends.
func hubFor(t *testing.T, m library.Model, b []byte) *hubtest.Hub {
	h := hubtest.New(map[string][]byte{m.Repo + "/" + m.File: b})
	old := downloadEndpoint
	downloadEndpoint = h.URL
	t.Cleanup(func() { downloadEndpoint = old; h.Close() })
	return h
}

// A gated repository says what to do, not what HTTP said.
// Both ways: streamed, and downloaded first as a platform that cannot stream.
func TestAGatedModelSaysHowToGetIt(t *testing.T) {
	for _, noStream := range []bool{false, true} {
		sh := dropShell(t)
		d := downloadsFor(sh)
		m, i := firstText(t)
		d.sel.Set(i)
		defer func(g func(string) (int64, bool), n bool) { downloadFree, downloadNoStream = g, n }(downloadFree, downloadNoStream)
		downloadFree = func(string) (int64, bool) { return 1 << 50, true }
		downloadNoStream = noStream
		h := hubFor(t, m, nil)
		h.Gated = true

		d.act()
		waitIdle(t, sh, d)
		if h.Requests.Load() == 0 {
			t.Errorf("noStream=%v: the download never asked the Hub it was given", noStream)
		}
		note := d.note.Get()
		for _, want := range []string{"gated or private", "huggingface.co/" + m.Repo, "HF_TOKEN"} {
			if !strings.Contains(note, want) {
				t.Errorf("noStream=%v: the note %q does not say %q", noStream, note, want)
			}
		}
	}
}

// A download ends as a container on the Models tab, the row marked on disk,
// and the GGUF converted as it arrived: it is never written to the model
// folder, so the folder never holds the model twice.
func TestADownloadEndsAsAContainer(t *testing.T) {
	src := testmodels.Path("stories260K.gguf")
	b, err := os.ReadFile(src)
	if err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proves nothing without it", err)
	}
	sh := dropShell(t)
	queueRunsNow(sh)
	d := downloadsFor(sh)
	m, i := firstText(t)
	d.sel.Set(i)
	defer func(f discover.FetchFunc, g func(string) (int64, bool)) {
		downloadFetch, downloadFree = f, g
	}(downloadFetch, downloadFree)
	downloadFree = func(string) (int64, bool) { return 1 << 50, true }
	streams := gguf.Streamable()
	downloadFetch = func(*hf.Client, hf.Ref, string) (string, error) {
		return "", errors.New("the stream fell back to a whole-file download")
	}
	if !streams {
		downloadFetch = nil // this platform downloads, from the same Hub
	}
	h := hubFor(t, m, b)
	var landed []string
	h.OnGet = func(*http.Request) {
		if !streams {
			return
		}
		es, _ := os.ReadDir(d.dir())
		for _, e := range es {
			if strings.Contains(e.Name(), ".gguf") {
				landed = append(landed, e.Name())
			}
		}
	}

	d.act()
	waitIdle(t, sh, d)
	if !convertjob.LoadableAt(d.containerFor(m)) {
		t.Fatalf("no container at %s; note %q", d.containerFor(m), d.note.Get())
	}
	if len(landed) > 0 {
		t.Errorf("the GGUF was written to the model folder while it downloaded: %v", landed)
	}
	if h.Served.Load() == 0 {
		t.Error("nothing was read from the Hub")
	}
	if !d.onDisk(i) || d.statusOf(i) != "On this machine" {
		t.Error("the row does not say the model is on disk")
	}
	if !strings.Contains(d.note.Get(), "is ready") {
		t.Errorf("the note does not say it is ready: %q", d.note.Get())
	}
}

// Stop ends a download with what it wrote kept, and says so.
func TestStopEndsADownload(t *testing.T) {
	if !gguf.Streamable() {
		t.Skip("this platform downloads the whole GGUF, which Stop does not interrupt")
	}
	src := testmodels.Path("Qwen3-0.6B-Q8_0.gguf")
	b, err := os.ReadFile(src)
	if err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proves nothing without it", err)
	}
	sh := dropShell(t)
	queueRunsNow(sh)
	d := downloadsFor(sh)
	m, i := firstText(t)
	d.sel.Set(i)
	defer func(g func(string) (int64, bool)) { downloadFree = g }(downloadFree)
	downloadFree = func(string) (int64, bool) { return 1 << 50, true }
	h := hubFor(t, m, b)
	h.OnGet = func(*http.Request) {
		if h.Served.Load() > int64(len(b))/2 {
			d.cancel()
		}
	}

	d.act()
	waitIdle(t, sh, d)
	if !strings.Contains(d.note.Get(), "Stopped") {
		t.Fatalf("the note does not say the download stopped: %q", d.note.Get())
	}
	if n, ok := jlm.Resumable(d.containerFor(m)); !ok || n == 0 {
		t.Errorf("a stopped download kept nothing to carry on from (journal %d, %v)", n, ok)
	}
	if convertjob.LoadableAt(d.containerFor(m)) {
		t.Error("a stopped download left a container that loads")
	}
}

// queueRunsNow installs a runner that runs each job at once, as the engine's
// queue would with nothing ahead of it.
func queueRunsNow(sh *app.Shell) {
	Attach(sh, Deps{Engine: &fakeEngine{run: func(job func()) bool { job(); return true }}})
}
