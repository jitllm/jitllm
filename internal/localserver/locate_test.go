package localserver

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestTheDefaultIsLoopbackOn4242: the service listens where it opens nothing
// to the network, and the old port is the one the default moved from.
func TestTheDefaultIsLoopbackOn4242(t *testing.T) {
	if DefaultAddr != "localhost:4242" {
		t.Fatalf("DefaultAddr = %q", DefaultAddr)
	}
	if LegacyAddr != "127.0.0.1:8080" || DefaultAddr == LegacyAddr {
		t.Fatalf("LegacyAddr = %q", LegacyAddr)
	}
}

// fakeJitllmd answers the capability manifest, wanting key when it is set.
func fakeJitllmd(t *testing.T, key string) *httptest.Server {
	t.Helper()
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if key != "" && r.Header.Get("Authorization") != "Bearer "+key {
			http.Error(w, `{"error":{"message":"no key"}}`, http.StatusUnauthorized)
			return
		}
		w.Write([]byte(`{"schema":1,"server_api":1,"build":{"version":"v7"}}`))
	}))
	t.Cleanup(s.Close)
	return s
}

// unusedAddr is an address nothing listens on.
func unusedAddr(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().String()
}

// TestLocateFindsAServiceOnTheOldPort: nothing on the default address, a
// jitllmd on the old one: it is used and marked Legacy. An open port that is
// not a jitllmd is not, and an address the caller typed is never swapped.
func TestLocateFindsAServiceOnTheOldPort(t *testing.T) {
	ctx := context.Background()
	primary := unusedAddr(t)
	old := fakeJitllmd(t, "")
	pair := WithDefaults(primary, old.Listener.Addr().String())

	in, err := Locate(ctx, primary, pair)
	if err != nil || !in.Legacy || in.Version != "v7" || in.Addr != old.Listener.Addr().String() {
		t.Fatalf("Locate: %+v, %v", in, err)
	}
	if in, err = Locate(ctx, "", pair); err != nil || !in.Legacy {
		t.Fatalf("an empty address is the default: %+v, %v", in, err)
	}

	// Not a jitllmd on the old port: an open port is not an answer.
	other := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(`{"hello":1}`)) }))
	defer other.Close()
	if _, err := Locate(ctx, primary, WithDefaults(primary, other.Listener.Addr().String())); err == nil {
		t.Fatal("a server that is not a jitllmd was used")
	}

	// A typed address is not second-guessed.
	if _, err := Locate(ctx, unusedAddr(t), pair); err == nil {
		t.Fatal("an address the caller typed fell back to the old port")
	}

	// The default answering wins, and is not Legacy.
	now := fakeJitllmd(t, "")
	a := now.Listener.Addr().String()
	if in, err := Locate(ctx, a, WithDefaults(a, old.Listener.Addr().String())); err != nil || in.Legacy {
		t.Fatalf("the default answers: %+v, %v", in, err)
	}
}

// TestProbeSendsTheKey: a jitllmd that wants a key is a different answer from
// one that is absent.
func TestProbeSendsTheKey(t *testing.T) {
	ctx := context.Background()
	s := fakeJitllmd(t, "k1")
	if _, err := Probe(ctx, s.URL); !errors.Is(err, ErrUnauthorized) {
		t.Fatalf("no key: %v, want ErrUnauthorized", err)
	}
	if _, err := Probe(ctx, s.URL, WithAPIKey("wrong")); !errors.Is(err, ErrUnauthorized) {
		t.Fatalf("wrong key: %v", err)
	}
	if in, err := Probe(ctx, s.URL, WithAPIKey("k1")); err != nil || in.Version != "v7" {
		t.Fatalf("right key: %+v %v", in, err)
	}
	// A keyed service on the old port is found, and still asks for the key.
	pair := WithDefaults(unusedAddr(t), s.Listener.Addr().String())
	if in, err := Locate(ctx, "", pair); !errors.Is(err, ErrUnauthorized) || !in.Legacy {
		t.Fatalf("old port, no key: %+v %v", in, err)
	}
}

// TestLocalhostIsDialledAtLoopback: the client transport reaches a server on
// 127.0.0.1 through the name localhost.
func TestLocalhostIsDialledAtLoopback(t *testing.T) {
	s := fakeJitllmd(t, "")
	_, port, _ := net.SplitHostPort(s.Listener.Addr().String())
	in, err := Probe(context.Background(), "localhost:"+port)
	if err != nil || in.Version != "v7" {
		t.Fatalf("Probe(localhost): %+v %v", in, err)
	}
}

func TestKeyTransportSendsTheKey(t *testing.T) {
	s := fakeJitllmd(t, "k2")
	c := &http.Client{Transport: KeyTransport("k2", nil)}
	resp, err := c.Get(s.URL)
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("%v %v", resp, err)
	}
	resp.Body.Close()
	if _, ok := KeyTransport("", nil).(*http.Transport); !ok {
		t.Errorf("no key wraps the transport anyway")
	}
}
