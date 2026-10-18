package hf

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fastFailures shrinks every wait in the range reader so a gate can inject a
// failure of each kind many times over without sleeping through them.
func fastFailures(t *testing.T, c *Client, chunk int) {
	t.Helper()
	c.Tuning = Tuning{BackoffBase: time.Millisecond, StallAfter: 50 * time.Millisecond,
		AttemptDeadline: 300 * time.Millisecond, RangeChunk: chunk, MaxRetryAfter: 20 * time.Millisecond}
}

// readWithin is ReadAt under a watchdog of its own: a reader with no timeout
// hangs on a stalled or trickling body, and the gate has to fail on that
// rather than hang with it.
func readWithin(t *testing.T, rm *Remote, p []byte, off int64) error {
	t.Helper()
	done := make(chan error, 1)
	go func() {
		_, err := rm.ReadAt(p, off)
		done <- err
	}()
	select {
	case err := <-done:
		return err
	case <-time.After(30 * time.Second):
		t.Fatalf("ReadAt(%d, %d) still blocked after 30s: a stalled or trickling body is "+
			"holding its range with no timeout", off, len(p))
		return nil
	}
}

func randomBytes(n int, seed uint64) []byte {
	rng := rand.New(rand.NewPCG(seed, 2))
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(rng.Uint32())
	}
	return b
}

func rangeOf(r *http.Request) (a, b int) {
	fmt.Sscanf(r.Header.Get("Range"), "bytes=%d-%d", &a, &b)
	return a, b
}

// TestRemoteReadsTheRangesItIsAskedFor serves a file with a Hub that fails
// every way a long transfer does -- a 503, a 429 that says when to come back,
// a body that stops with the connection left open, a body that trickles a
// byte at a time and never stops long enough to look stalled, a 206 cut off
// mid-body, and a connection reset before any answer -- and demands every
// range come back exact, each failure kind having fired. Without the stall
// watchdog or the per-request deadline a range hangs and readWithin fails the
// gate. A server that ignores Range must be a failure, not the file's first
// bytes.
func TestRemoteReadsTheRangesItIsAskedFor(t *testing.T) {
	data := randomBytes(1<<20, 1)
	var n atomic.Int64
	var fired [8]atomic.Int64
	ignoreRange := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/resolve/abc/m.safetensors") {
			http.NotFound(w, r)
			return
		}
		if ignoreRange && r.Method == "GET" {
			w.Write(data)
			return
		}
		ranged := r.Header.Get("Range") != ""
		kind := int(n.Add(1) % 11)
		switch {
		case !ranged:
		case kind == 1:
			fired[kind].Add(1)
			http.Error(w, "busy", http.StatusServiceUnavailable)
			return
		case kind == 3:
			fired[kind].Add(1)
			w.Header().Set("RateLimit", `"resolvers";r=0;t=1`)
			http.Error(w, "slow down", http.StatusTooManyRequests)
			return
		case kind == 5:
			// A body that starts and then stops producing bytes, with the
			// connection left open.
			fired[kind].Add(1)
			a, b := rangeOf(r)
			w.Header().Set("Content-Length", fmt.Sprint(b-a+1))
			w.WriteHeader(http.StatusPartialContent)
			w.Write(data[a:min(a+10, b+1)])
			w.(http.Flusher).Flush()
			<-r.Context().Done()
			return
		case kind == 7:
			// A body that never goes quiet for stallAfter and never ends:
			// only a deadline on the whole request catches it.
			fired[kind].Add(1)
			a, b := rangeOf(r)
			w.Header().Set("Content-Length", fmt.Sprint(b-a+1))
			w.WriteHeader(http.StatusPartialContent)
			for i := a; i <= b; i++ {
				if _, err := w.Write(data[i : i+1]); err != nil {
					return
				}
				w.(http.Flusher).Flush()
				select {
				case <-r.Context().Done():
					return
				case <-time.After(5 * time.Millisecond):
				}
			}
			return
		case kind == 2:
			// A 206 whose body stops early, which is what a reset mid-body
			// looks like.
			fired[kind].Add(1)
			c, buf, _ := w.(http.Hijacker).Hijack()
			a, b := rangeOf(r)
			buf.WriteString("HTTP/1.1 206 Partial Content\r\nContent-Length: 1000000\r\n\r\n")
			buf.Write(data[a:min(a+100, b+1)])
			buf.Flush()
			c.Close()
			return
		case kind == 4:
			// A connection reset before any answer.
			fired[kind].Add(1)
			c, _, _ := w.(http.Hijacker).Hijack()
			c.Close()
			return
		}
		http.ServeContent(w, r, "m.safetensors", time.Time{}, bytes.NewReader(data))
	}))
	defer srv.Close()

	var log bytes.Buffer
	var logMu sync.Mutex
	c := &Client{Endpoint: srv.URL, Progress: lockedWriter{&log, &logMu}}
	fastFailures(t, c, 16<<10)
	rm, err := c.OpenRemote(Ref{Repo: "o/r", Rev: "abc", File: "m.safetensors"})
	if err != nil {
		t.Fatal(err)
	}
	if rm.Size() != int64(len(data)) {
		t.Fatalf("size %d, want %d", rm.Size(), len(data))
	}
	rng := rand.New(rand.NewPCG(1, 2))
	for i := 0; i < 40; i++ {
		off := rng.IntN(len(data) - 1)
		l := 1 + rng.IntN(min(len(data)-off, 200000))
		p := make([]byte, l)
		if err := readWithin(t, rm, p, int64(off)); err != nil {
			t.Fatalf("ReadAt(%d, %d): %v", off, l, err)
		}
		if !bytes.Equal(p, data[off:off+l]) {
			t.Fatalf("ReadAt(%d, %d): wrong bytes", off, l)
		}
	}
	for _, k := range []int{1, 2, 3, 4, 5, 7} {
		if fired[k].Load() == 0 {
			t.Errorf("failure kind %d never fired, so the gate did not test surviving it", k)
		}
	}
	logMu.Lock()
	retries := strings.Count(log.String(), "retry    ")
	logMu.Unlock()
	if retries == 0 {
		t.Error("ranges were retried and nothing said so on Progress")
	}
	t.Logf("%d requests, %d retries logged", n.Load(), retries)

	ignoreRange = true
	p := make([]byte, 10)
	if err := readWithin(t, rm, p, 5000); err == nil {
		t.Fatal("a server that ignored Range was read as if it had honoured it")
	}
}

type lockedWriter struct {
	b  *bytes.Buffer
	mu *sync.Mutex
}

func (w lockedWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.b.Write(p)
}

// TestARangeThatKeepsFailingErrorsOut holds a range against a server that
// answers 503 forever: the read must end in an error naming the range, after
// maxAttempts logged retries, rather than retry for ever.
func TestARangeThatKeepsFailingErrorsOut(t *testing.T) {
	data := randomBytes(64<<10, 3)
	broken := atomic.Bool{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if broken.Load() && r.Header.Get("Range") != "" {
			http.Error(w, "down", http.StatusBadGateway)
			return
		}
		http.ServeContent(w, r, "m.safetensors", time.Time{}, bytes.NewReader(data))
	}))
	defer srv.Close()
	var log bytes.Buffer
	var mu sync.Mutex
	c := &Client{Endpoint: srv.URL, Progress: lockedWriter{&log, &mu}}
	fastFailures(t, c, 64<<10)
	rm, err := c.OpenRemote(Ref{Repo: "o/r", Rev: "abc", File: "m.safetensors"})
	if err != nil {
		t.Fatal(err)
	}
	broken.Store(true)
	err = readWithin(t, rm, make([]byte, 1000), 100)
	if err == nil || !strings.Contains(err.Error(), "bytes [100,1100)") || !strings.Contains(err.Error(), "gave up") {
		t.Fatalf("a range the server refuses for ever: %v, want an error naming the range", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if got := strings.Count(log.String(), "retry    "); got != maxAttempts-1 {
		t.Errorf("%d retries logged before giving up, want %d", got, maxAttempts-1)
	}
}

// TestRemoteKeepsTheCDNAddress serves the file the way the Hub does: its own
// address answers a redirect to a signed address on another host, and counts
// each answer against a resolver limit. The ranges must go to the signed
// address once it is known -- not back through the Hub, which would hold a
// 1.5 TB stream to the resolver limit -- and an expired address must be
// replaced, not retried until the range gives up. The token goes to the Hub
// and never to the CDN.
func TestRemoteKeepsTheCDNAddress(t *testing.T) {
	data := randomBytes(1<<20, 5)
	var (
		gen       atomic.Int64 // the current signature
		uses      atomic.Int64 // ranges served under it
		ranges    atomic.Int64
		refused   atomic.Int64
		leaked    atomic.Bool
		resolves  atomic.Int64
		hubAuthed atomic.Bool
	)
	gen.Store(1)
	cdn := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			leaked.Store(true)
		}
		var g int64
		fmt.Sscanf(r.URL.Query().Get("sig"), "%d", &g)
		// A signature is good for 25 ranges, then expires.
		if g != gen.Load() || uses.Add(1) > 25 {
			refused.Add(1)
			http.Error(w, "expired", http.StatusForbidden)
			return
		}
		ranges.Add(1)
		http.ServeContent(w, r, "m.safetensors", time.Time{}, bytes.NewReader(data))
	}))
	defer cdn.Close()
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") == "Bearer tok" {
			hubAuthed.Store(true)
		}
		if r.Method == "HEAD" {
			w.Header().Set("x-linked-size", fmt.Sprint(len(data)))
		} else {
			resolves.Add(1)
		}
		g := gen.Load()
		if uses.Load() > 25 {
			g = gen.Add(1)
			uses.Store(0)
		}
		// Another host name, as the Hub's CDN is: net/http keeps a header
		// across ports of one host.
		http.Redirect(w, r, fmt.Sprintf("%s/xet/obj?sig=%d",
			strings.Replace(cdn.URL, "127.0.0.1", "localhost", 1), g), http.StatusFound)
	}))
	defer hub.Close()

	c := &Client{Endpoint: hub.URL, Token: "tok", Conns: 4}
	fastFailures(t, c, 16<<10)
	rm, err := c.OpenRemote(Ref{Repo: "o/r", Rev: "abc", File: "m.safetensors"})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(7, 8))
	for i := 0; i < 30; i++ {
		off := rng.IntN(len(data) - 1)
		l := 1 + rng.IntN(min(len(data)-off, 100000))
		p := make([]byte, l)
		if err := readWithin(t, rm, p, int64(off)); err != nil {
			t.Fatalf("ReadAt(%d, %d): %v", off, l, err)
		}
		if !bytes.Equal(p, data[off:off+l]) {
			t.Fatalf("ReadAt(%d, %d): wrong bytes", off, l)
		}
	}
	t.Logf("%d ranges from the CDN, %d resolves at the Hub, %d refused as expired",
		ranges.Load(), resolves.Load(), refused.Load())
	if ranges.Load() < 100 {
		t.Fatalf("only %d ranges, too few to tell a kept address from none", ranges.Load())
	}
	if refused.Load() == 0 || gen.Load() < 3 {
		t.Errorf("no signature ever expired (generation %d), so replacing one was not tested", gen.Load())
	}
	// Each signature is asked for by at most the requests in flight when the
	// last one expired.
	if limit := gen.Load() * int64(c.Conns+1); resolves.Load() > limit {
		t.Errorf("%d resolves for %d ranges over %d signatures: the ranges are going back "+
			"through the Hub", resolves.Load(), ranges.Load(), gen.Load())
	}
	if leaked.Load() {
		t.Error("the Hub's token was sent to the CDN")
	}
	if !hubAuthed.Load() {
		t.Error("the token never reached the Hub, so the leak check above proves nothing")
	}
}

// TestRangesSpreadOverConnections reads a file from a TLS server that offers
// HTTP/2, as the Hub's CDN does, with ranges in flight at once: they must
// travel as HTTP/1.1 on several connections. Over HTTP/2 net/http puts every
// request to one host on one TCP connection, and a stream of sixteen ranges
// reads at one connection's rate.
func TestRangesSpreadOverConnections(t *testing.T) {
	data := randomBytes(1<<20, 9)
	var (
		mu    sync.Mutex
		addrs = map[string]bool{}
		h2    atomic.Int64
	)
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		addrs[r.RemoteAddr] = true
		mu.Unlock()
		if r.ProtoMajor == 2 {
			h2.Add(1)
		}
		time.Sleep(20 * time.Millisecond) // so the ranges overlap
		http.ServeContent(w, r, "m.safetensors", time.Time{}, bytes.NewReader(data))
	}))
	srv.EnableHTTP2 = true
	srv.StartTLS()
	defer srv.Close()

	tr := rangeTransport()
	// Only the test certificate is added: the transport's own TLS config is
	// what decides the protocols the server is offered, and a gate that set
	// them itself passed while the real config offered h2 and every range
	// came back as HTTP/2 frames read as a malformed response.
	tr.TLSClientConfig.RootCAs = srv.Client().Transport.(*http.Transport).TLSClientConfig.RootCAs
	defer tr.CloseIdleConnections()

	c := &Client{Endpoint: srv.URL, Conns: 8, HTTP: &http.Client{Transport: tr}}
	fastFailures(t, c, 16<<10)
	rm := &Remote{c: c, r: Ref{Repo: "o/r", Rev: "abc", File: "m.safetensors"},
		url: srv.URL + "/m.safetensors", size: int64(len(data))}
	p := make([]byte, len(data))
	if err := readWithin(t, rm, p, 0); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(p, data) {
		t.Fatal("wrong bytes")
	}
	mu.Lock()
	defer mu.Unlock()
	if h2.Load() > 0 {
		t.Errorf("%d ranges went over HTTP/2, which multiplexes them onto one connection", h2.Load())
	}
	if len(addrs) < 4 {
		t.Errorf("64 ranges, 8 at a time, came over %d connection(s)", len(addrs))
	}
	t.Logf("64 ranges over %d connections", len(addrs))
}

func TestRetryAfterIsRead(t *testing.T) {
	h := func(kv ...string) http.Header {
		x := http.Header{}
		for i := 0; i < len(kv); i += 2 {
			x.Set(kv[i], kv[i+1])
		}
		return x
	}
	for _, c := range []struct {
		h    http.Header
		want time.Duration
	}{
		{h(), 0},
		{h("Retry-After", "7"), 7 * time.Second},
		{h("RateLimit", `"resolvers";r=0;t=256`), 256 * time.Second},
		{h("RateLimit", `"resolvers";r=0;t=99999`), 5 * time.Minute},
		{h("Retry-After", "-3"), 0},
		{h("Retry-After", "Wed, 21 Oct 2015 07:28:00 GMT"), 0},
	} {
		if got := retryAfterOf(c.h, 5*time.Minute); got != c.want {
			t.Errorf("%v: %v, want %v", c.h, got, c.want)
		}
	}
}
