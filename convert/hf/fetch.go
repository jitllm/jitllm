package hf

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Client fetches files from a Hub. The zero value works and talks to
// huggingface.co anonymously.
type Client struct {
	HTTP     *http.Client // nil: a shared client with no whole-request timeout
	Endpoint string       // "" : DefaultEndpoint
	Token    string       // "" : anonymous; a gated or private repo answers 401
	Progress io.Writer    // nil: silent
	// OnProgress, when set, is told the bytes on disk and the file's size as a
	// download advances (size -1 when the Hub would not say). It runs on the
	// downloading goroutine.
	OnProgress func(done, total int64)
	// Conns bounds the range requests in flight at once over every Remote
	// this client opened; 0 is DefaultConns.
	Conns int
	// Tuning sizes and times the range reader; zero is the shipping value.
	Tuning Tuning

	connsOnce sync.Once
	conns     chan struct{}
}

// ErrAccess is what a 401 or 403 wraps: the repository is private, gated, or
// does not exist -- the Hub answers the same way to all three. errors.Is it to
// tell a missing token from every other failure.
var ErrAccess = errors.New("the Hub refused access")

// No Client.Timeout: it covers the whole exchange including the body, so no
// single value suits both a small file and a 40 GB one. The timeouts here are
// per-stall: a connection that produces no header, or a read that stops.
var shared = &http.Client{Transport: sharedTransport()}

func sharedTransport() http.RoundTripper {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.ResponseHeaderTimeout = 60 * time.Second
	// A resumable download does not need a long-lived idle pool; it needs the
	// connection it is on to stay up.
	t.IdleConnTimeout = 90 * time.Second
	return t
}

const userAgent = "jitllm (+https://github.com/jitllm/jitllm)"

// copyBuf is fixed: Content-Length is untrusted input, used for progress and
// completeness, never to size an allocation.
const copyBuf = 1 << 20

func (c *Client) client() *http.Client {
	if c.HTTP != nil {
		return c.HTTP
	}
	return shared
}

func (c *Client) req(method, u string) (*http.Request, error) {
	req, err := http.NewRequest(method, u, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)
	if c.Token != "" {
		// net/http drops Authorization on a redirect to another domain, which
		// is exactly what the Hub does -- it redirects to a CDN host that
		// carries its own signed URL. The token reaches the Hub and not the CDN.
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
	return req, nil
}

func (c *Client) notef(format string, a ...any) {
	if c.Progress != nil {
		fmt.Fprintf(c.Progress, format, a...)
	}
}

// Info is what the Hub says about a file before any of it is transferred.
type Info struct {
	Size   int64  // bytes; -1 when the server would not say
	SHA256 string // lowercase hex, "" when the server gave no usable digest
}

// Stat asks what the file is without transferring it.
//
// The digest is x-linked-etag and nothing else. It appears on a redirect hop:
// the final CDN response's ETag is the object store's own hash, also 64 hex
// characters, and not the file's sha256. net/http returns only the last
// response, so the chain is read through CheckRedirect and a plain ETag is
// ignored. x-linked-size is taken the same way.
//
//	302 huggingface.co/.../xet/...   x-linked-etag: "102f..."  <- the file
//	200 cdn.../transfer/...          etag: "8a05..."           <- the store's
func (c *Client) Stat(r Ref) (Info, error) {
	req, err := c.req("HEAD", r.URL(c.Endpoint))
	if err != nil {
		return Info{}, err
	}
	var linkedSize, linkedETag string
	keep := func(resp *http.Response) {
		if v := resp.Header.Get("x-linked-size"); v != "" {
			linkedSize = v
		}
		if v := resp.Header.Get("x-linked-etag"); v != "" {
			linkedETag = v
		}
	}
	// A copy, so the shared client is not given a per-call closure.
	cl := *c.client()
	cl.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if len(via) >= 10 {
			return fmt.Errorf("stopped after 10 redirects")
		}
		if req.Response != nil {
			keep(req.Response)
		}
		return nil
	}
	resp, err := cl.Do(req)
	if err != nil {
		return Info{}, fmt.Errorf("%s: %w", r, err)
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, io.LimitReader(resp.Body, 4<<10))
	if err := httpErr(r, resp); err != nil {
		return Info{}, err
	}
	keep(resp)
	in := Info{Size: -1}
	if n, err := strconv.ParseInt(linkedSize, 10, 64); err == nil && n >= 0 {
		in.Size = n
	} else if resp.ContentLength >= 0 {
		in.Size = resp.ContentLength
	}
	in.SHA256 = digest(linkedETag)
	return in, nil
}

// digest returns v when it is a plain sha256, and "" otherwise -- a missing
// digest degrades to the size check rather than to a silent pass, which is why
// Fetch checks both.
func digest(v string) string {
	v = strings.TrimPrefix(strings.TrimSpace(v), "W/")
	v = strings.ToLower(strings.TrimPrefix(strings.Trim(v, `"`), "sha256:"))
	if len(v) != 64 {
		return ""
	}
	if _, err := hex.DecodeString(v); err != nil {
		return ""
	}
	return v
}

func httpErr(r Ref, resp *http.Response) error {
	switch {
	case resp.StatusCode == 401 || resp.StatusCode == 403:
		// "Or not there at all": the Hub answers 401 for a repository that
		// does not exist as well as for one you cannot see.
		return fmt.Errorf("%s: HTTP %d -- private, gated, or not there at all; the Hub\n"+
			"  answers the same way to all three. Check the name; set HF_TOKEN (or\n"+
			"  HUGGING_FACE_HUB_TOKEN) to a token that can read it; and accept the\n"+
			"  model's licence on the Hub if it asks for one: %w", r, resp.StatusCode, ErrAccess)
	case resp.StatusCode == 404:
		return fmt.Errorf("%s: HTTP 404 -- no such repository, revision or file.\n"+
			"  Drop the filename to be shown what the repository holds", r)
	case resp.StatusCode/100 != 2:
		return fmt.Errorf("%s: HTTP %d %s", r, resp.StatusCode, http.StatusText(resp.StatusCode))
	}
	return nil
}

// Resolve turns a repository reference into a file reference.
//
// One GGUF is an answer. Several is a question, and the refusal lists them:
// choosing a quantization for someone is not this command's business.
func (c *Client) Resolve(r Ref) (Ref, error) {
	if r.File != "" {
		return r, nil
	}
	doc, err := c.info(r)
	if err != nil {
		return r, err
	}
	var gg []string
	for _, s := range doc.Siblings {
		if strings.HasSuffix(strings.ToLower(s.Name), ".gguf") {
			gg = append(gg, s.Name)
		}
	}
	sort.Strings(gg)
	switch len(gg) {
	case 0:
		return r, fmt.Errorf("%s holds no .gguf file (%d files in total)", r, len(doc.Siblings))
	case 1:
		r.File = gg[0]
		return r, nil
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, "%s holds %d GGUF files and this command will not pick one for you.\n"+
		"  Name the one you want:\n", r, len(gg))
	for i, n := range gg {
		if i == 24 {
			fmt.Fprintf(&sb, "    ... and %d more\n", len(gg)-i)
			break
		}
		fmt.Fprintf(&sb, "    %s/%s\n", r, n)
	}
	return r, fmt.Errorf("%s", strings.TrimRight(sb.String(), "\n"))
}

// Fetch makes r's file present under dir and returns its path.
//
// It is the whole of this package's contract: idempotent, resumable, and it
// never returns a path to bytes it has not checked.
func (c *Client) Fetch(r Ref, dir string) (string, error) {
	r, err := c.Resolve(r)
	if err != nil {
		return "", err
	}
	dest, err := r.Dest(dir)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	in, serr := c.Stat(r)
	if st, err := os.Stat(dest); err == nil {
		switch {
		case st.IsDir():
			return "", fmt.Errorf("%s is a directory", dest)
		case serr != nil:
			// Having the file beats reaching the Hub: with no network and the
			// model already present, convert, and print the Hub error.
			c.notef("have     %s (%s); the Hub is unreachable (%v), using it\n", dest, size(st.Size()), serr)
			return dest, nil
		case in.Size < 0 || st.Size() == in.Size:
			c.notef("have     %s (%s), complete -- not downloading\n", dest, size(st.Size()))
			return dest, nil
		default:
			// The basename is shared across repos (see Ref.Dest). Refuse:
			// overwriting throws away someone's file and resuming would splice
			// two different models into one that converts cleanly.
			return "", fmt.Errorf("%s is %s and %s is %s.\n"+
				"  That is a different file under the same name. Move it, or pass -o DIR "+
				"to download somewhere else", dest, size(st.Size()), r, size(in.Size))
		}
	}
	if serr != nil {
		return "", serr
	}
	if err := c.download(r, in, dest); err != nil {
		return "", err
	}
	return dest, nil
}

// part is the sidecar beside a partial download, recording which remote file
// the prefix on disk came from.
type part struct {
	URL    string `json:"url"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256,omitempty"`
}

func (c *Client) download(r Ref, in Info, dest string) error {
	pth := dest + ".part"
	side := pth + ".json"
	want := part{URL: r.URL(c.Endpoint), Size: in.Size, SHA256: in.SHA256}

	h := sha256.New()
	var have int64
	// A resume is only valid if the remote file is the one the prefix came
	// from; otherwise Range staples a new revision's tail onto an old head. The
	// sidecar must agree on URL, size and digest before any byte is kept.
	if got, err := readPart(side); err == nil && got == want {
		if fi, err := os.Stat(pth); err == nil && fi.Size() > 0 && (in.Size < 0 || fi.Size() <= in.Size) {
			n, err := hashPrefix(h, pth)
			if err == nil {
				have = n
			} else {
				h.Reset()
			}
		}
	}
	if have == 0 {
		os.Remove(pth)
		if err := writePart(side, want); err != nil {
			return err
		}
	}

	flag := os.O_CREATE | os.O_WRONLY
	if have == 0 {
		flag |= os.O_TRUNC
	}
	f, err := os.OpenFile(pth, flag, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	if have > 0 {
		if err := f.Truncate(have); err != nil {
			return err
		}
		if _, err := f.Seek(have, io.SeekStart); err != nil {
			return err
		}
		c.notef("resume   %s at %s of %s\n", filepath.Base(dest), size(have), size(in.Size))
	}

	if in.Size < 0 || have < in.Size {
		err := c.bodyRetrying(r, &have, h, f, in)
		// A server that ignored the Range answered with the whole file, so the
		// prefix on disk is worth nothing: rewind everything -- the offset, the
		// hash and the file -- and take it from zero. Once, so a server that
		// ignores every Range cannot loop.
		var restart errRestart
		if errors.As(err, &restart) {
			c.notef("restart  %s: the server ignored a resume at %s\n", filepath.Base(dest), size(restart.at))
			have, h = 0, sha256.New()
			if err = f.Truncate(0); err == nil {
				if _, err = f.Seek(0, io.SeekStart); err == nil {
					err = c.bodyRetrying(r, &have, h, f, in)
				}
			}
		}
		if err != nil {
			f.Close()
			return err
		}
	}
	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}

	if in.Size >= 0 && have != in.Size {
		// Kept, not deleted: the next run resumes from here.
		return fmt.Errorf("%s: got %s of %s -- the transfer stopped early.\n"+
			"  Run the same command again and it resumes from %s", r, size(have), size(in.Size), pth)
	}
	if want.SHA256 != "" {
		if got := hex.EncodeToString(h.Sum(nil)); got != want.SHA256 {
			os.Remove(pth)
			os.Remove(side)
			return fmt.Errorf("%s: sha256 %s, and the Hub says %s.\n"+
				"  The bytes are wrong and have been deleted rather than converted",
				r, got, want.SHA256)
		}
	}
	if err := os.Rename(pth, dest); err != nil {
		return err
	}
	os.Remove(side)
	c.notef("fetched  %s (%s)\n", dest, size(have))
	return nil
}

// transient marks a transport failure -- the request or the copy -- which a
// Range resume from the bytes already written can recover. An HTTP status or a
// server that ignored the Range is not transient.
type transient struct{ error }

func (e transient) Unwrap() error { return e.error }

// bodyRetrying resumes a dropped transfer in-process rather than failing the
// command. A retry that moved bytes resets the count, so only maxAttempts
// failures in a row give up.
func (c *Client) bodyRetrying(r Ref, have *int64, h io.Writer, f io.Writer, in Info) error {
	for attempt := 0; ; attempt++ {
		before := *have
		err := c.body(r, have, h, f, in)
		var t transient
		if err == nil || !errors.As(err, &t) {
			return err
		}
		if *have > before {
			attempt = 0
		}
		if attempt+1 >= maxAttempts {
			return err
		}
		c.notef("retry    %s at %s: %v\n", filepath.Base(r.File), size(*have), t.error)
		time.Sleep(c.backoff(attempt))
	}
}

// body runs the transfer itself, appending to f and to h.
func (c *Client) body(r Ref, have *int64, h io.Writer, f io.Writer, in Info) error {
	req, err := c.req("GET", r.URL(c.Endpoint))
	if err != nil {
		return err
	}
	if *have > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", *have))
	}
	resp, err := c.client().Do(req)
	if err != nil {
		return transient{fmt.Errorf("%s: %w", r, err)}
	}
	defer resp.Body.Close()
	if err := httpErr(r, resp); err != nil {
		return err
	}
	if *have > 0 {
		switch resp.StatusCode {
		case http.StatusPartialContent:
			// Trust the server's own account of where it started over the
			// request: a proxy is free to answer a different range.
			if want := fmt.Sprintf("bytes %d-", *have); !strings.HasPrefix(resp.Header.Get("Content-Range"), want) {
				return fmt.Errorf("%s: asked to resume at %d and the server answered %q",
					r, *have, resp.Header.Get("Content-Range"))
			}
		default:
			// 200 to a ranged request means the range was ignored and this is
			// the whole file. Starting over is correct; appending is corruption.
			return errRestart{at: *have}
		}
	}

	rest := int64(-1)
	if in.Size >= 0 {
		rest = in.Size - *have
	}
	src := io.Reader(resp.Body)
	if rest >= 0 {
		// +1 so a server sending more than it announced is an error rather than
		// an unbounded write onto the model disk.
		src = io.LimitReader(resp.Body, rest+1)
	}
	p := &progress{w: c.Progress, tty: tty(c.Progress), name: filepath.Base(r.File),
		n: *have, base: *have, total: in.Size, start: time.Now(), on: c.OnProgress}
	n, cerr := io.CopyBuffer(io.MultiWriter(f, h, p), src, make([]byte, copyBuf))
	*have += n
	p.done(cerr == nil)
	if cerr != nil {
		return transient{fmt.Errorf("%s: %w", r, cerr)}
	}
	if rest >= 0 && n > rest {
		return fmt.Errorf("%s: the server sent more than the %s it announced", r, size(in.Size))
	}
	return nil
}

// errRestart says the server ignored a Range. download's caller turns it into a
// second attempt from zero; it is a distinct type so that "start over" cannot be
// confused with "the transfer failed".
type errRestart struct{ at int64 }

func (e errRestart) Error() string {
	return fmt.Sprintf("the server ignored a request to resume at %d", e.at)
}

func hashPrefix(h io.Writer, name string) (int64, error) {
	f, err := os.Open(name)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	return io.CopyBuffer(h, f, make([]byte, copyBuf))
}

func readPart(name string) (part, error) {
	var p part
	b, err := os.ReadFile(name)
	if err != nil {
		return p, err
	}
	return p, json.Unmarshal(b, &p)
}

func writePart(name string, p part) error {
	b, err := json.Marshal(p)
	if err != nil {
		return err
	}
	return os.WriteFile(name, b, 0o644)
}

func size(n int64) string {
	switch {
	case n < 0:
		return "an unknown number of bytes"
	case n < 1<<20:
		return fmt.Sprintf("%.1f KiB", float64(n)/(1<<10))
	case n < 1<<30:
		return fmt.Sprintf("%.1f MiB", float64(n)/(1<<20))
	}
	return fmt.Sprintf("%.2f GiB", float64(n)/(1<<30))
}

// progress is a counting writer in the copy chain.
//
// It is a writer rather than a wrapped reader so that it counts what reached
// the file, not bytes a failed write never stored.
type progress struct {
	w          io.Writer
	name       string
	n, total   int64
	base       int64 // bytes a resume started from: they are not this session's rate
	tty        bool
	start, tik time.Time
	on         func(done, total int64)
}

func (p *progress) Write(b []byte) (int, error) {
	p.n += int64(len(b))
	if p.on != nil {
		p.on(p.n, p.total)
	}
	// \r only where something can rewrite the line; a redirected stderr would
	// become one unreadable line.
	end, every := "\r", 500*time.Millisecond
	if !p.tty {
		end, every = "\n", 5*time.Second
	}
	if p.w != nil && time.Since(p.tik) >= every {
		p.tik = time.Now()
		p.line(end)
	}
	return len(b), nil
}

// tty reports whether w is something that can rewrite its last line.
func tty(w io.Writer) bool {
	f, ok := w.(*os.File)
	if !ok {
		return false
	}
	fi, err := f.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0
}

func (p *progress) line(end string) {
	el := time.Since(p.start).Seconds()
	rate := ""
	if el > 0.2 {
		bps := float64(p.n-p.base) / el
		rate = fmt.Sprintf("  %.1f MB/s", bps/1e6)
		if p.total > p.n && bps > 0 {
			eta := time.Duration(float64(p.total-p.n) / bps * float64(time.Second))
			rate += fmt.Sprintf("  eta %s", eta.Round(time.Second))
		}
	}
	pct := ""
	if p.total > 0 {
		pct = fmt.Sprintf(" %5.1f%% of", 100*float64(p.n)/float64(p.total))
	}
	fmt.Fprintf(p.w, "%-28s%s %s%s%s", trunc(p.name, 28), pct, size(p.total), rate, end)
}

func (p *progress) done(ok bool) {
	if p.w == nil {
		return
	}
	if ok {
		p.line("\n") // the last line is a summary, on a line of its own
		return
	}
	fmt.Fprintln(p.w)
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-3] + "..."
}
