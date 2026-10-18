package hf

import (
	"cmp"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// A streaming converter reads the model where it lives: jlm's writer lays the
// container out from the tensors' shapes and then asks for their bytes one
// tensor at a time, so each safetensors shard is read by byte range from the
// Hub and never stored locally.

// repoInfo is what the Hub's API says about one revision of a repository.
type repoInfo struct {
	SHA      string `json:"sha"`
	Siblings []struct {
		Name string `json:"rfilename"`
	} `json:"siblings"`
}

func (c *Client) info(r Ref) (repoInfo, error) {
	var doc repoInfo
	req, err := c.req("GET", r.APIURL(c.Endpoint))
	if err != nil {
		return doc, err
	}
	resp, err := c.client().Do(req)
	if err != nil {
		return doc, fmt.Errorf("%s: %w", r, err)
	}
	defer resp.Body.Close()
	if err := httpErr(r, resp); err != nil {
		return doc, err
	}
	// The body is untrusted, so it is read under a bound.
	b, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return doc, fmt.Errorf("%s: %w", r, err)
	}
	if err := json.Unmarshal(b, &doc); err != nil {
		return doc, fmt.Errorf("%s: the Hub's file list did not parse: %w", r, err)
	}
	return doc, nil
}

var commitSHA = regexp.MustCompile(`^[0-9a-f]{40}$`)

// Pin resolves r's revision to the commit it names at the time of the call and
// lists that commit's files.
//
// A streamed conversion reads for hours, so "main" is not an address: a push
// mid-conversion would mix two checkpoints. Every range is taken from the
// immutable commit the header was read at.
func (c *Client) Pin(r Ref) (Ref, []string, error) {
	doc, err := c.info(r)
	if err != nil {
		return r, nil, err
	}
	if !commitSHA.MatchString(doc.SHA) {
		return r, nil, fmt.Errorf("%s: the Hub gave no commit for revision %q", r, r.Rev)
	}
	r.Rev = doc.SHA
	files := make([]string, len(doc.Siblings))
	for i, s := range doc.Siblings {
		files[i] = s.Name
	}
	return r, files, nil
}

// Remote is a file on the Hub read by byte range. It is an io.ReaderAt and
// safe for concurrent use.
type Remote struct {
	c    *Client
	r    Ref
	url  string
	size int64
	got  atomic.Int64

	// The Hub answers a file's address with a redirect to a signed CDN
	// address, and counts every answer against a resolver limit: anonymously
	// 3000 per five minutes (`ratelimit-policy: "fixed window";"resolvers";
	// q=3000;w=300`). A 1.5 TB model read in 8 MiB ranges is 186,000 of
	// them, so asking the Hub for every range holds a stream near 80 MB/s and
	// then draws 429s. Where the chain ends is kept and every range goes there
	// until it is refused or older than cdnFresh, so the Hub is asked a few
	// times an hour per file.
	mu     sync.Mutex
	direct string    // the CDN address the last redirect chain ended at
	at     time.Time // when it was learned
}

// OpenRemote stats r's file. r should be pinned (see Pin).
func (c *Client) OpenRemote(r Ref) (*Remote, error) {
	in, err := c.Stat(r)
	if err != nil {
		return nil, err
	}
	if in.Size < 0 {
		return nil, fmt.Errorf("%s: the Hub will not say how big it is, and a file read "+
			"by range has to know", r)
	}
	return &Remote{c: c, r: r, url: r.URL(c.Endpoint), size: in.Size}, nil
}

// Size is the file's length in bytes.
func (m *Remote) Size() int64 { return m.size }

// Transferred is how many bytes have arrived so far, over every ReadAt.
func (m *Remote) Transferred() int64 { return m.got.Load() }

// Name is the file's reference, for messages.
func (m *Remote) Name() string { return m.r.String() }

// ReadAt fills p from bytes [off, off+len(p)) of the file.
//
// It is many small ranges, rangeFanout at a time, each retried and watched on
// its own, so a stall costs one piece and the link is shared across
// connections. The requests in flight over every Remote of one Client are
// bounded by Client.Conns.
func (m *Remote) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 || off+int64(len(p)) > m.size {
		return 0, fmt.Errorf("%s: bytes [%d,%d) of %d", m.r, off, off+int64(len(p)), m.size)
	}
	var (
		wg    sync.WaitGroup
		sem   = make(chan struct{}, rangeFanout)
		mu    sync.Mutex
		first error
	)
	chunk := cmp.Or(m.c.Tuning.RangeChunk, defaultRangeChunk)
	for at := 0; at < len(p); at += chunk {
		piece := p[at:min(at+chunk, len(p))]
		wg.Add(1)
		sem <- struct{}{}
		go func(piece []byte, off int64) {
			defer func() { <-sem; wg.Done() }()
			if err := m.piece(piece, off); err != nil {
				mu.Lock()
				if first == nil {
					first = err
				}
				mu.Unlock()
			}
		}(piece, off+int64(at))
	}
	wg.Wait()
	if first != nil {
		return 0, first
	}
	return len(p), nil
}

// rangeChunk and rangeFanout size the pieces a read is split into; the chunk
// is Client.Tuning.RangeChunk so a gate can split a small file. One connection to the Hub's
// CDN carried about 60 MB/s, so a stream needs many.
//
// Client.Tuning carries the variables a gate would otherwise have to set for
// the process; zero is the shipping value.
const defaultRangeChunk = 16 << 20

const rangeFanout = 16

// DefaultConns is how many range requests a Client keeps in flight at once
// when Client.Conns is 0.
const DefaultConns = 16

// ranges is the client range reads use when Client.HTTP is nil. It speaks
// HTTP/1.1 only: the Hub's CDN offers HTTP/2, on which net/http multiplexes
// every request to a host onto ONE TCP connection, so sixteen ranges in
// flight shared one connection's 60 MB/s -- the first full Kimi-K3 run read
// 30 MB/s through a single socket. One connection per range is what spreads
// the stream, and the idle pool keeps them, so a range does not pay a TLS
// handshake.
var ranges = &http.Client{Transport: rangeTransport()}

func rangeTransport() *http.Transport {
	t := sharedTransport().(*http.Transport)
	t.ForceAttemptHTTP2 = false
	t.TLSNextProto = map[string]func(string, *tls.Conn) http.RoundTripper{}
	// Turning HTTP/2 off is not enough: the clone of a transport that has
	// spoken it carries a TLS config offering "h2", a server that takes it
	// answers in HTTP/2 frames, and the HTTP/1.1 reader calls them a
	// malformed response (the second run's config.json, ten times over).
	if t.TLSClientConfig == nil {
		t.TLSClientConfig = &tls.Config{}
	}
	t.TLSClientConfig.NextProtos = []string{"http/1.1"}
	t.MaxIdleConnsPerHost = 4 * DefaultConns
	t.MaxIdleConns = 4 * DefaultConns
	return t
}

// rangeClient is a copy of the client ranges are read with, which follows no
// redirect (see once).
func (c *Client) rangeClient() http.Client {
	cl := *ranges
	if c.HTTP != nil {
		cl = *c.HTTP
	}
	cl.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	return cl
}

// acquire takes one of the client's connection slots; the func it returns
// gives it back.
func (c *Client) acquire() func() {
	c.connsOnce.Do(func() {
		c.conns = make(chan struct{}, cmp.Or(c.Conns, DefaultConns))
	})
	c.conns <- struct{}{}
	return func() { <-c.conns }
}

// piece reads one range, retrying, and says so on Progress each time.
//
// It keeps what a failed attempt delivered. Only a transport error, a stall, a
// deadline, a short body, a 5xx, a 429 or a refused CDN address is retried; a
// deliberate answer from the Hub (401, 404) is final. A range that fails
// maxAttempts times in a row with nothing delivered is an error, never a hang.
func (m *Remote) piece(p []byte, off int64) error {
	done := 0
	for attempt := 0; done < len(p); attempt++ {
		n, err := m.once(p[done:], off+int64(done))
		done += n
		if err == nil {
			continue
		}
		var perm permanent
		if errors.As(err, &perm) || attempt+1 >= maxAttempts {
			return fmt.Errorf("%s: bytes [%d,%d): gave up after %d attempt(s) at byte %d: %w",
				m.r, off, off+int64(len(p)), attempt+1, off+int64(done), err)
		}
		if n > 0 {
			attempt = 0 // progress resets the budget; only a stuck range gives up
		}
		wait := m.c.backoff(attempt)
		var later retryLater
		if errors.As(err, &later) {
			wait = max(wait, later.after)
		}
		m.c.notef("retry    %s bytes [%d,%d) at %d: %v; attempt %d of %d in %v\n",
			m.r, off, off+int64(len(p)), off+int64(done), err, attempt+2, maxAttempts, wait)
		time.Sleep(wait)
	}
	return nil
}

// maxAttempts is ten failures in a row on one range: with a stall, a minute
// of silence each and the backoff between, about fifteen minutes, every one
// of them logged.
const maxAttempts = 10

func (c *Client) backoff(attempt int) time.Duration {
	return min(cmp.Or(c.Tuning.BackoffBase, time.Second)<<attempt, time.Minute)
}

type permanent struct{ error }

// retryLater is a refusal that says when to ask again: a Retry-After, or the
// Hub's `ratelimit: "resolvers";r=0;t=<seconds>`.
type retryLater struct {
	error
	after time.Duration
}

func (e retryLater) Unwrap() error { return e.error }

var rateLimitReset = regexp.MustCompile(`;\s*t=(\d+)`)

// retryAfterOf is how long a 429 or 5xx asks to be left alone, 0 when it does
// not say. The header is the network's claim, so it is bounded.
func retryAfterOf(h http.Header, bound time.Duration) time.Duration {
	var s int64
	if v, err := strconv.ParseInt(strings.TrimSpace(h.Get("Retry-After")), 10, 64); err == nil {
		s = v
	} else if m := rateLimitReset.FindStringSubmatch(h.Get("RateLimit")); m != nil {
		s, _ = strconv.ParseInt(m[1], 10, 64)
	}
	if s <= 0 {
		return 0
	}
	return min(time.Duration(min(s, 1<<20))*time.Second, bound)
}

// Tuning is the waits and sizes of the range reader. Zero is the shipping
// value of each; a gate sets them small to inject failures without sleeping.
type Tuning struct {
	// RangeChunk is the size of the pieces a read is split into (16 MiB).
	RangeChunk int
	// BackoffBase is the first retry's wait, doubled each attempt (a second).
	BackoffBase time.Duration
	// MaxRetryAfter bounds how long a server may ask a range to wait (5 min).
	MaxRetryAfter time.Duration
	// StallAfter is how long a response may produce nothing before the
	// request is abandoned and retried (a minute).
	StallAfter time.Duration
	// AttemptDeadline is how long one request may take in all: a body that
	// trickles a byte every few seconds never stalls, and would otherwise
	// hold its range for days (10 min).
	AttemptDeadline time.Duration
	// CDNFresh is how long a learned CDN address is used before the Hub is
	// asked again; the Hub signs it for an hour (20 min).
	CDNFresh time.Duration
}

func (c *Client) stall() time.Duration { return cmp.Or(c.Tuning.StallAfter, time.Minute) }
func (c *Client) deadline() time.Duration {
	return cmp.Or(c.Tuning.AttemptDeadline, 10*time.Minute)
}

// target is where the next range goes: the CDN address a redirect led to, if
// one is known and fresh, else the Hub's own address.
func (m *Remote) target() (string, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.direct != "" && time.Since(m.at) < cmp.Or(m.c.Tuning.CDNFresh, 20*time.Minute) {
		return m.direct, true
	}
	return m.url, false
}

func (m *Remote) remember(u string) {
	m.mu.Lock()
	m.direct, m.at = u, time.Now()
	m.mu.Unlock()
}

func (m *Remote) forget(u string) {
	m.mu.Lock()
	if m.direct == u {
		m.direct = ""
	}
	m.mu.Unlock()
}

// maxHops bounds a redirect chain.
const maxHops = 10

func (m *Remote) once(p []byte, off int64) (int, error) {
	defer m.c.acquire()()
	// The transport times out a missing header, but a body that stops
	// mid-stream blocks Read forever. The watchdog cancels the request after
	// StallAfter with no bytes, and each byte that arrives winds it back; the
	// deadline ends one that keeps trickling.
	ctx, cancel := context.WithTimeout(context.Background(), m.c.deadline())
	defer cancel()
	var stalled atomic.Bool
	dog := time.AfterFunc(m.c.stall(), func() { stalled.Store(true); cancel() })
	defer dog.Stop()
	rng := "bytes=" + strconv.FormatInt(off, 10) + "-" + strconv.FormatInt(off+int64(len(p))-1, 10)

	// Redirects are followed here rather than by net/http, so where the chain
	// ends can be kept (see Remote.direct).
	cl := m.c.rangeClient()
	u, cached := m.target()
	var resp *http.Response
	for hop := 0; ; hop++ {
		req, err := m.c.req("GET", u)
		if err != nil {
			return 0, permanent{err}
		}
		// The token is for the Hub. net/http drops it on a redirect to
		// another host; following by hand, so does this.
		if !sameHost(u, m.url) {
			req.Header.Del("Authorization")
		}
		req.Header.Set("Range", rng)
		if resp, err = cl.Do(req.WithContext(ctx)); err != nil {
			if cached {
				m.forget(u)
			}
			return 0, m.c.why(ctx, &stalled, 0, len(p), err)
		}
		if !redirect(resp.StatusCode) || hop == maxHops {
			break
		}
		loc, err := resp.Location()
		resp.Body.Close()
		if err != nil {
			return 0, permanent{fmt.Errorf("a redirect with no usable Location: %w", err)}
		}
		u, cached = loc.String(), false
	}
	defer resp.Body.Close()
	switch {
	case resp.StatusCode == http.StatusPartialContent:
		if !cached && u != m.url {
			m.remember(u)
		}
	case resp.StatusCode == http.StatusTooManyRequests, resp.StatusCode/100 == 5:
		if cached {
			m.forget(u)
		}
		return 0, retryLater{fmt.Errorf("HTTP %d", resp.StatusCode), retryAfterOf(resp.Header, cmp.Or(m.c.Tuning.MaxRetryAfter, 5*time.Minute))}
	case u != m.url && resp.StatusCode/100 == 4 && resp.StatusCode != http.StatusRequestedRangeNotSatisfiable:
		// A signed address expires; the Hub hands out a new one.
		m.forget(u)
		return 0, fmt.Errorf("the CDN refused its address (HTTP %d); asking the Hub again", resp.StatusCode)
	case resp.StatusCode == http.StatusOK:
		// A server that ignores Range sends the whole file from byte 0, and
		// reading p's worth of it would silently be the wrong bytes.
		return 0, permanent{fmt.Errorf("the server ignored the byte range")}
	default:
		if err := httpErr(m.r, resp); err != nil {
			return 0, permanent{err}
		}
		return 0, permanent{fmt.Errorf("HTTP %d to a byte range", resp.StatusCode)}
	}
	n, err := io.ReadFull(watched{resp.Body, dog, m.c.stall()}, p)
	m.got.Add(int64(n))
	if err != nil {
		err = m.c.why(ctx, &stalled, n, len(p), err)
	}
	return n, err
}

// why names what ended a request early: the watchdog, the deadline, a short
// body, or the transport's own error.
func (c *Client) why(ctx context.Context, stalled *atomic.Bool, n, want int, err error) error {
	switch {
	case stalled.Load():
		return fmt.Errorf("no bytes for %v after %d of %d", c.stall(), n, want)
	case ctx.Err() != nil:
		return fmt.Errorf("over the %v deadline with %d of %d bytes in", c.deadline(), n, want)
	case err == io.ErrUnexpectedEOF:
		return fmt.Errorf("the range ended after %d of %d bytes", n, want)
	}
	return err
}

func redirect(code int) bool {
	switch code {
	case http.StatusMovedPermanently, http.StatusFound, http.StatusSeeOther,
		http.StatusTemporaryRedirect, http.StatusPermanentRedirect:
		return true
	}
	return false
}

func sameHost(a, b string) bool {
	ua, err1 := url.Parse(a)
	ub, err2 := url.Parse(b)
	return err1 == nil && err2 == nil && ua.Host == ub.Host
}

// watched winds a stall watchdog back on every read that delivers bytes.
type watched struct {
	r     io.Reader
	dog   *time.Timer
	stall time.Duration
}

func (w watched) Read(p []byte) (int, error) {
	n, err := w.r.Read(p)
	if n > 0 {
		w.dog.Reset(w.stall)
	}
	return n, err
}
