package hf

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// The download path, gated against a fake Hub: a gate that needs the network
// skips, and only an httptest.Server can drop a connection at a chosen byte or
// serve a stale revision deterministically. It also counts what it was asked
// for, so "it did not re-download" is an assertion.

// hub is a Hub that serves one file and remembers what it was asked.
type hub struct {
	mu     sync.Mutex
	body   []byte
	etag   string   // the sha256 the Hub reports; "" sends none
	files  []string // what the API says the repo holds
	cut    int64    // >0: write this many bytes of the response and drop the connection
	down   bool     // after a cut, every later GET drops before sending a byte
	isDown bool
	noRng  bool // answer 200 to a Range request, i.e. ignore it
	over   int  // extra bytes to send beyond the announced size
	gets   int
	heads  int
	apis   int
	rngs   []string
	served int64
}

func (h *hub) counts() (gets, heads int, served int64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.gets, h.heads, h.served
}

func (h *hub) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if strings.HasPrefix(r.URL.Path, "/api/models/") {
		h.apis++
		var doc struct {
			Siblings []map[string]string `json:"siblings"`
		}
		for _, f := range h.files {
			doc.Siblings = append(doc.Siblings, map[string]string{"rfilename": f})
		}
		json.NewEncoder(w).Encode(&doc)
		return
	}
	size := int64(len(h.body))
	w.Header().Set("x-linked-size", strconv.FormatInt(size, 10))
	if h.etag != "" {
		w.Header().Set("x-linked-etag", `"`+h.etag+`"`)
	}
	w.Header().Set("Accept-Ranges", "bytes")
	if r.Method == http.MethodHead {
		h.heads++
		w.Header().Set("Content-Length", strconv.FormatInt(size, 10))
		return
	}
	h.gets++
	if h.isDown {
		panic(http.ErrAbortHandler)
	}
	start := int64(0)
	if v := r.Header.Get("Range"); v != "" {
		h.rngs = append(h.rngs, v)
		if !h.noRng {
			fmt.Sscanf(v, "bytes=%d-", &start)
			w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, size-1, size))
			w.Header().Set("Content-Length", strconv.FormatInt(size-start, 10))
			w.WriteHeader(http.StatusPartialContent)
		}
	}
	out := h.body[start:]
	if h.over > 0 {
		// No Content-Length: the point is a body longer than the size the Hub
		// announced, and net/http refuses to write past a declared length.
		out = append(append([]byte{}, out...), bytes.Repeat([]byte{'x'}, h.over)...)
	}
	if h.cut > 0 && int64(len(out)) > h.cut {
		w.Header().Set("Content-Length", strconv.FormatInt(int64(len(out)), 10))
		n, _ := w.Write(out[:h.cut])
		h.served += int64(n)
		h.isDown = h.down
		panic(http.ErrAbortHandler) // the dropped link, reproducibly
	}
	n, _ := w.Write(out)
	h.served += int64(n)
}

// fake starts a hub holding n pseudo-random bytes and returns it with a client
// pointed at it. The bytes vary so a splice shows up as a content mismatch
// rather than as a coincidence.
func fake(t *testing.T, n int) (*hub, *Client, Ref) {
	t.Helper()
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i*7 + i/251)
	}
	sum := sha256.Sum256(b)
	h := &hub{body: b, etag: hex.EncodeToString(sum[:]), files: []string{"m.gguf"}}
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return h, &Client{Endpoint: srv.URL, HTTP: srv.Client()}, Ref{Repo: "o/r", Rev: "main", File: "m.gguf"}
}

func readFile(t *testing.T, p string) []byte {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// TestFetchWritesTheFileAndLeavesNoScraps is the happy path, and it asserts the
// two things a caller depends on: the bytes, and that the .part and its sidecar
// are gone -- a converter handed a path that is still being written to reads a
// truncated GGUF.
func TestFetchWritesTheFileAndLeavesNoScraps(t *testing.T) {
	h, cl, r := fake(t, 300_000)
	dir := t.TempDir()
	got, err := cl.Fetch(r, dir)
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(dir, "m.gguf"); got != want {
		t.Errorf("landed at %q, want %q", got, want)
	}
	if !bytes.Equal(readFile(t, got), h.body) {
		t.Errorf("the downloaded bytes are not the served bytes")
	}
	for _, scrap := range []string{got + ".part", got + ".part.json"} {
		if _, err := os.Stat(scrap); err == nil {
			t.Errorf("%s survived a completed download", scrap)
		}
	}
}

// TestFetchDoesNotDownloadWhatIsAlreadyHere. A GGUF is 0.5-40 GB and the
// commonest invocation of this command is the second one.
func TestFetchDoesNotDownloadWhatIsAlreadyHere(t *testing.T) {
	h, cl, r := fake(t, 100_000)
	dir := t.TempDir()
	if _, err := cl.Fetch(r, dir); err != nil {
		t.Fatal(err)
	}
	if _, err := cl.Fetch(r, dir); err != nil {
		t.Fatal(err)
	}
	gets, heads, served := h.counts()
	if gets != 1 {
		t.Errorf("%d GETs for two Fetches of the same file; the second must not transfer it again", gets)
	}
	if heads != 2 {
		t.Errorf("%d HEADs, want 2: the size is what says the local file is complete", heads)
	}
	if served != int64(len(h.body)) {
		t.Errorf("%d bytes served, want %d", served, len(h.body))
	}
}

// TestFetchResumesInsteadOfStartingOver: the first attempt dies at byte 40000 of
// 120000, the second must ask for the rest and must not re-transfer the prefix.
//
// The assertion is the byte count: a restart also produces the right file.
func TestFetchResumesInsteadOfStartingOver(t *testing.T) {
	h, cl, r := fake(t, 120_000)
	h.cut, h.down = 40_000, true // a link that stays down: in-process retries give up
	noBackoff(t, cl)
	dir := t.TempDir()
	if _, err := cl.Fetch(r, dir); err == nil {
		t.Fatal("a dropped connection was reported as a successful download")
	}
	pth := filepath.Join(dir, "m.gguf.part")
	if fi, err := os.Stat(pth); err != nil || fi.Size() == 0 {
		t.Fatalf("the partial download was not kept: %v", err)
	}
	h.mu.Lock()
	h.cut, h.isDown, h.rngs = 0, false, nil
	h.mu.Unlock()

	got, err := cl.Fetch(r, dir)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(readFile(t, got), h.body) {
		t.Fatal("the resumed file is not the served file")
	}
	// Exactly the file, once: 40000 before the drop and 80000 after it. One byte
	// more is a prefix that was sent twice.
	if _, _, served := h.counts(); served != int64(len(h.body)) {
		t.Errorf("%d bytes served for a %d-byte file: it started over instead of resuming",
			served, len(h.body))
	}
	h.mu.Lock()
	rngs := append([]string(nil), h.rngs...)
	h.mu.Unlock()
	if len(rngs) != 1 {
		t.Fatalf("Range headers %q; the second attempt must ask for exactly one range", rngs)
	}
	if want := fmt.Sprintf("bytes=%d-", 40_000); rngs[0] != want {
		t.Errorf("resumed at %q, want %q -- an offset that is not what is on disk splices", rngs[0], want)
	}
}

// TestFetchRetriesADroppedConnectionInProcess: every response drops after
// 40000 bytes, and one Fetch must still deliver the file, each byte sent once.
func TestFetchRetriesADroppedConnectionInProcess(t *testing.T) {
	h, cl, r := fake(t, 120_000)
	h.cut = 40_000
	noBackoff(t, cl)
	got, err := cl.Fetch(r, t.TempDir())
	if err != nil {
		t.Fatalf("three drops, each after progress, failed the fetch: %v", err)
	}
	if !bytes.Equal(readFile(t, got), h.body) {
		t.Fatal("the retried file is not the served file")
	}
	if _, _, served := h.counts(); served != int64(len(h.body)) {
		t.Errorf("%d bytes served for a %d-byte file: a retry re-sent a prefix", served, len(h.body))
	}
}

// noBackoff has cl retry at once. (Tuning's zero is the shipping second, so the
// least a test can ask for is a nanosecond.)
func noBackoff(t *testing.T, cl *Client) {
	t.Helper()
	cl.Tuning.BackoffBase = time.Nanosecond
}

// TestFetchRefusesToSpliceADifferentRevision is the resume's real hazard. A
// prefix on disk says nothing about which file it is a prefix of, and Range
// would staple the tail of a re-quantized upload onto the head of the old one:
// the result parses, converts, and decodes garbage.
func TestFetchRefusesToSpliceADifferentRevision(t *testing.T) {
	h, cl, r := fake(t, 90_000)
	dir := t.TempDir()
	pth := filepath.Join(dir, "m.gguf.part")
	// A prefix of some other file, with a sidecar describing that other file.
	if err := os.WriteFile(pth, bytes.Repeat([]byte{'Z'}, 30_000), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := writePart(pth+".json", part{URL: r.URL(cl.Endpoint), Size: 90_000,
		SHA256: strings.Repeat("a", 64)}); err != nil {
		t.Fatal(err)
	}
	got, err := cl.Fetch(r, dir)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(readFile(t, got), h.body) {
		t.Fatal("the file is a splice of the stale prefix and the new tail")
	}
	h.mu.Lock()
	rngs := append([]string(nil), h.rngs...)
	h.mu.Unlock()
	if len(rngs) != 0 {
		t.Errorf("Range %q: a prefix of a different file must not be resumed from", rngs)
	}
}

// TestFetchRefusesBytesThatDoNotHashAndKeepsNone: the Hub publishes the file's
// sha256, so it must be checked; a corrupted GGUF converts into a container
// that decodes noise at full speed.
func TestFetchRefusesBytesThatDoNotHashAndKeepsNone(t *testing.T) {
	h, cl, r := fake(t, 50_000)
	h.mu.Lock()
	h.body[17] ^= 0xff // one flipped bit, announced under the original digest
	h.mu.Unlock()
	dir := t.TempDir()
	got, err := cl.Fetch(r, dir)
	if err == nil {
		t.Fatalf("corrupted bytes were accepted and written to %s", got)
	}
	if !strings.Contains(err.Error(), "sha256") {
		t.Errorf("the refusal does not name the checksum: %v", err)
	}
	for _, p := range []string{"m.gguf", "m.gguf.part", "m.gguf.part.json"} {
		if _, err := os.Stat(filepath.Join(dir, p)); err == nil {
			t.Errorf("%s survived a checksum failure; a resume would trust it", p)
		}
	}
}

// TestFetchRefusesAServerThatOverruns: a server that lies about its size
// cannot keep writing onto the model disk.
func TestFetchRefusesAServerThatOverruns(t *testing.T) {
	h, cl, r := fake(t, 20_000)
	h.over = 500_000
	dir := t.TempDir()
	if _, err := cl.Fetch(r, dir); err == nil {
		t.Fatal("a body longer than the announced size was accepted")
	}
	// The assertion is what reached the disk, not the error: the write must
	// stop at the announced size plus the one byte that detects the overrun.
	fi, err := os.Stat(filepath.Join(dir, "m.gguf.part"))
	if err != nil {
		t.Fatalf("the partial write is gone, so what it wrote cannot be checked: %v", err)
	}
	if fi.Size() > int64(len(h.body))+1 {
		t.Errorf("%d bytes written for a %d-byte file: the transfer is not bounded by the "+
			"size the server announced", fi.Size(), len(h.body))
	}
}

// TestFetchStartsOverWhenTheServerIgnoresARange. A proxy that answers 200 to a
// ranged request is handing back the whole file; appending it to the prefix is
// corruption, so the transfer restarts.
func TestFetchStartsOverWhenTheServerIgnoresARange(t *testing.T) {
	h, cl, r := fake(t, 80_000)
	h.cut, h.down = 25_000, true
	noBackoff(t, cl)
	dir := t.TempDir()
	if _, err := cl.Fetch(r, dir); err == nil {
		t.Fatal("a dropped connection was reported as a success")
	}
	h.mu.Lock()
	h.cut, h.isDown, h.noRng = 0, false, true
	h.mu.Unlock()
	got, err := cl.Fetch(r, dir)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(readFile(t, got), h.body) {
		t.Fatal("the file is the prefix with a whole second copy appended")
	}
}

// TestFetchRefusesADifferentFileOfTheSameName is the cost of naming a download
// by its basename, made an error instead of a silent conversion of the wrong
// weights.
func TestFetchRefusesADifferentFileOfTheSameName(t *testing.T) {
	_, cl, r := fake(t, 70_000)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "m.gguf"), []byte("somebody else's model"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := cl.Fetch(r, dir)
	if err == nil {
		t.Fatalf("a file of the wrong size was accepted as the model: %s", got)
	}
	if !strings.Contains(err.Error(), "different file") {
		t.Errorf("the refusal does not say what is wrong: %v", err)
	}
}

// TestResolveTakesTheOnlyGGUFAndRefusesAChoice. One GGUF is an answer; a
// quantization menu is a question, and picking Q4_K_M for someone changes the
// model they measure.
func TestResolveTakesTheOnlyGGUFAndRefusesAChoice(t *testing.T) {
	h, cl, r := fake(t, 1000)
	r.File = ""
	h.files = []string{"README.md", "m.gguf", ".gitattributes"}
	got, err := cl.Resolve(r)
	if err != nil {
		t.Fatal(err)
	}
	if got.File != "m.gguf" {
		t.Errorf("Resolve chose %q, want the repository's only GGUF", got.File)
	}

	h.mu.Lock()
	h.files = []string{"a-Q4_K_M.gguf", "a-Q8_0.gguf", "README.md"}
	h.mu.Unlock()
	_, err = cl.Resolve(r)
	if err == nil {
		t.Fatal("a repository of quantizations was silently reduced to one file")
	}
	for _, want := range []string{"a-Q4_K_M.gguf", "a-Q8_0.gguf"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal does not list %s:\n%v", want, err)
		}
	}
	t.Logf("%v", err)

	h.mu.Lock()
	h.files = []string{"README.md"}
	h.mu.Unlock()
	if _, err := cl.Resolve(r); err == nil {
		t.Fatal("a repository with no GGUF resolved to something")
	}
}

// TestTheDigestComesFromTheHubAndNotFromTheStore reproduces the real Hub's
// three-hop reply (see Stat): the file's sha256 is x-linked-etag on the 302,
// and the final ETag is the object store's own hash of the same shape. Against
// a fallback to the final ETag a byte-perfect download refuses itself.
func TestTheDigestComesFromTheHubAndNotFromTheStore(t *testing.T) {
	body := bytes.Repeat([]byte("jitllm"), 5000)
	sum := sha256.Sum256(body)
	mux := http.NewServeMux()
	mux.HandleFunc("/o/r/resolve/main/m.gguf", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/xet/m.gguf", http.StatusTemporaryRedirect)
	})
	mux.HandleFunc("/xet/m.gguf", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("x-linked-size", strconv.Itoa(len(body)))
		w.Header().Set("x-linked-etag", `"`+hex.EncodeToString(sum[:])+`"`)
		http.Redirect(w, r, "/cdn/m.gguf", http.StatusFound)
	})
	mux.HandleFunc("/cdn/m.gguf", func(w http.ResponseWriter, r *http.Request) {
		// 64 hex characters, and not this file's: the store's hash of its own
		// representation.
		w.Header().Set("ETag", `"`+strings.Repeat("8a", 32)+`"`)
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		if r.Method == http.MethodHead {
			return
		}
		w.Write(body)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	cl := &Client{Endpoint: srv.URL, HTTP: srv.Client()}
	r := Ref{Repo: "o/r", Rev: "main", File: "m.gguf"}
	in, err := cl.Stat(r)
	if err != nil {
		t.Fatal(err)
	}
	if want := hex.EncodeToString(sum[:]); in.SHA256 != want {
		t.Errorf("Stat took the digest %q, want the Hub's %q", in.SHA256, want)
	}
	if in.Size != int64(len(body)) {
		t.Errorf("Stat size %d, want %d", in.Size, len(body))
	}
	got, err := cl.Fetch(r, t.TempDir())
	if err != nil {
		t.Fatalf("a byte-perfect download was refused: %v", err)
	}
	if !bytes.Equal(readFile(t, got), body) {
		t.Error("the wrong bytes were written")
	}
}

// TestFetchUsesTheLocalFileWhenTheHubIsUnreachable: a box with no network and
// the model already on it converts. The absence is printed, never swallowed.
func TestFetchUsesTheLocalFileWhenTheHubIsUnreachable(t *testing.T) {
	dir := t.TempDir()
	body := []byte("a model, locally")
	if err := os.WriteFile(filepath.Join(dir, "m.gguf"), body, 0o644); err != nil {
		t.Fatal(err)
	}
	var log bytes.Buffer
	// Port 1 on the loopback: nothing listens, and the dial fails at once.
	cl := &Client{Endpoint: "http://127.0.0.1:1", Progress: &log}
	got, err := cl.Fetch(Ref{Repo: "o/r", Rev: "main", File: "m.gguf"}, dir)
	if err != nil {
		t.Fatalf("the local file was not used: %v", err)
	}
	if !bytes.Equal(readFile(t, got), body) {
		t.Fatal("the wrong file came back")
	}
	if !strings.Contains(log.String(), "unreachable") {
		t.Errorf("the unreachable Hub was not reported:\n%s", log.String())
	}
}

// TestFetchSendsTheToken, because a private repo is the whole reason the field
// exists and "it 401s" is indistinguishable from "the token was never sent".
func TestFetchSendsTheToken(t *testing.T) {
	var seen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = append(seen, r.Header.Get("Authorization"))
		if r.Header.Get("Authorization") != "Bearer s3cret" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Header().Set("x-linked-size", "2")
		if r.Method == http.MethodHead {
			return
		}
		io.WriteString(w, "hi")
	}))
	defer srv.Close()
	cl := &Client{Endpoint: srv.URL, HTTP: srv.Client(), Token: "s3cret"}
	if _, err := cl.Fetch(Ref{Repo: "o/r", Rev: "main", File: "m.gguf"}, t.TempDir()); err != nil {
		t.Fatal(err)
	}
	for _, h := range seen {
		if h != "Bearer s3cret" {
			t.Errorf("a request went out with Authorization %q", h)
		}
	}

	cl.Token = ""
	_, err := cl.Fetch(Ref{Repo: "o/r", Rev: "main", File: "m.gguf"}, t.TempDir())
	if err == nil {
		t.Fatal("a 401 was reported as a download")
	}
	if !errors.Is(err, ErrAccess) {
		t.Errorf("a 401 does not wrap ErrAccess, so an app cannot tell it from any other failure: %v", err)
	}
	if !strings.Contains(err.Error(), "HF_TOKEN") {
		t.Errorf("the 401 does not say how to fix it: %v", err)
	}
}

// A caller must be told how far a download has got, ending at its size.
func TestFetchReportsProgressToTheCaller(t *testing.T) {
	h, cl, r := fake(t, 3<<20)
	var last, total int64
	calls := 0
	cl.OnProgress = func(done, size int64) {
		if done < last {
			t.Errorf("progress went backwards: %d after %d", done, last)
		}
		last, total = done, size
		calls++
	}
	if _, err := cl.Fetch(r, t.TempDir()); err != nil {
		t.Fatal(err)
	}
	if calls < 2 || last != int64(len(h.body)) || total != int64(len(h.body)) {
		t.Errorf("%d report(s), last %d of %d, want several ending at %d", calls, last, total, len(h.body))
	}
}

// The token is found where the Hub's own tools put it, the environment first.
func TestFindTokenLooksWhereTheHubsToolsLook(t *testing.T) {
	home := t.TempDir()
	os.MkdirAll(filepath.Join(home, ".cache", "huggingface"), 0o755)
	os.WriteFile(filepath.Join(home, ".cache", "huggingface", "token"), []byte("from-file\n"), 0o600)
	homeFn := func() (string, error) { return home, nil }
	env := map[string]string{}
	getenv := func(k string) string { return env[k] }

	if got := FindToken(getenv, homeFn); got != "from-file" {
		t.Errorf("with no environment: %q, want the login file's", got)
	}
	env["HUGGING_FACE_HUB_TOKEN"] = "from-env"
	if got := FindToken(getenv, homeFn); got != "from-env" {
		t.Errorf("the environment did not win: %q", got)
	}
	delete(env, "HUGGING_FACE_HUB_TOKEN")
	env["HF_HOME"] = t.TempDir() // a home with no token file
	if got := FindToken(getenv, homeFn); got != "" {
		t.Errorf("HF_HOME was not honoured: %q", got)
	}
}
