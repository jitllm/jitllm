package client_test

import (
	"context"
	"errors"
	"net"
	"net/http/httptest"
	"testing"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/internal/localserver"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/server"
	"github.com/jitllm/jitllm/server/client"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

func guarded(t *testing.T, key string) *httptest.Server {
	t.Helper()
	e := server.New(server.Config{Admit: testdevices.Admit(), ModelDir: t.TempDir(), Version: "v3", Probe: func() ([]server.DeviceInfo, error) { return nil, nil }})
	t.Cleanup(e.Close)
	s := httptest.NewServer(nil)
	s.Config.Handler = server.Guard{APIKey: key, Addr: s.Listener.Addr().String()}.Wrap(e.Handler())
	t.Cleanup(s.Close)
	return s
}

// TestClientSendsItsKey: a keyed jitllmd answers a client that has the key
// on every call, tells one without it apart from an absent one, and the
// generated stubs carry the key too.
func TestClientSendsItsKey(t *testing.T) {
	ctx := context.Background()
	s := guarded(t, "k-3")

	if _, err := client.Discover(ctx, s.URL); !errors.Is(err, localserver.ErrUnauthorized) {
		t.Fatalf("no key: %v, want ErrUnauthorized", err)
	}
	if _, err := client.Discover(ctx, s.URL, localserver.WithAPIKey("bad")); !errors.Is(err, localserver.ErrUnauthorized) {
		t.Fatalf("wrong key: %v", err)
	}
	c, err := client.Discover(ctx, s.URL, localserver.WithAPIKey("k-3"))
	if err != nil || c.Info.GetVersion() != "v3" {
		t.Fatalf("right key: %v %v", c, err)
	}
	if _, err := c.Model.ListModels(ctx, connect.NewRequest(&v1.ListModelsRequest{})); err != nil {
		t.Fatalf("a call after discovery: %v", err)
	}
	// Dial without the key reaches the server and is refused by it.
	_, err = client.Dial(s.URL).Model.ListModels(ctx, connect.NewRequest(&v1.ListModelsRequest{}))
	if connect.CodeOf(err) != connect.CodeUnauthenticated {
		t.Fatalf("an unkeyed call: %v, want unauthenticated", err)
	}
}

func unused(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().String()
}

// TestDiscoverFindsAServiceOnTheOldPort: nothing on the default address and a
// jitllmd on the old one is a connection to the old one, marked Legacy; an
// address the caller typed is not swapped.
func TestDiscoverFindsAServiceOnTheOldPort(t *testing.T) {
	ctx := context.Background()
	old := guarded(t, "")
	primary := unused(t)
	pair := localserver.WithDefaults(primary, old.Listener.Addr().String())

	c, err := client.Discover(ctx, "", pair)
	if err != nil || !c.Legacy || c.Info.GetVersion() != "v3" {
		t.Fatalf("Discover: %v, %v", c, err)
	}
	if c, err = client.Discover(ctx, primary, pair); err != nil || !c.Legacy {
		t.Fatalf("Discover(default): %v, %v", c, err)
	}
	if _, err := client.Discover(ctx, unused(t), pair); err == nil {
		t.Fatal("a typed address fell back to the old port")
	}
	// The default answering wins over the old port.
	now := guarded(t, "")
	a := now.Listener.Addr().String()
	if c, err := client.Discover(ctx, a, localserver.WithDefaults(a, old.Listener.Addr().String())); err != nil || c.Legacy {
		t.Fatalf("the default answers: %v, %v", c, err)
	}
}
