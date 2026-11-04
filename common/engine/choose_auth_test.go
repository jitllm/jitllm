package engine

import (
	"context"
	"net"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/localserver"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/server"
)

// jitllmdAt serves a real engine behind the guard, wanting key when set.
func jitllmdAt(t *testing.T, key string) string {
	t.Helper()
	e := server.New(server.Config{Admit: testdevices.Admit(), ModelDir: t.TempDir(), Version: "9.8.7",
		Probe: func() ([]server.DeviceInfo, error) { return nil, nil }})
	t.Cleanup(e.Close)
	s := httptest.NewServer(nil)
	s.Config.Handler = server.Guard{APIKey: key, Addr: s.Listener.Addr().String()}.Wrap(e.Handler())
	t.Cleanup(s.Close)
	return s.Listener.Addr().String()
}

func unusedAddr(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().String()
}

// TestChooseSendsTheServiceKey: Auto connects to a keyed jitllmd with the key;
// without it the app says why and runs its own engine, never connected to a
// service it cannot use.
func TestChooseSendsTheServiceKey(t *testing.T) {
	ctx := context.Background()
	addr := jitllmdAt(t, "kk")

	ch := Choose(ctx, ModeAuto, addr, localserver.WithAPIKey("kk"))
	if ch.Conn == nil || ch.Conn.Info == nil || ch.Warning != "" {
		t.Fatalf("with the key: %+v", ch)
	}
	ch = Choose(ctx, ModeAuto, addr)
	if ch.Conn != nil || !strings.Contains(ch.Warning, "API key") {
		t.Fatalf("without the key: %+v", ch)
	}
	// Service mode stays connected and shows the reason.
	ch = Choose(ctx, ModeService, addr)
	if ch.Conn == nil || ch.Conn.Info != nil || !strings.Contains(ch.Warning, "API key") {
		t.Fatalf("service mode without the key: %+v", ch)
	}
	// Embedded beside a keyed service still warns that the two share the GPU.
	ch = Choose(ctx, ModeEmbedded, addr)
	if ch.Conn != nil || !strings.Contains(ch.Warning, "share") {
		t.Fatalf("embedded beside a keyed service: %+v", ch)
	}
}

// TestChooseFindsAServiceOnTheOldPort: nothing on the default address and a
// jitllmd on the old one is used, and the warning says it runs there.
func TestChooseFindsAServiceOnTheOldPort(t *testing.T) {
	ctx := context.Background()
	old := jitllmdAt(t, "")
	primary := unusedAddr(t)
	pair := localserver.WithDefaults(primary, old)

	for _, mode := range []string{ModeAuto, ModeService} {
		ch := Choose(ctx, mode, primary, pair)
		if ch.Conn == nil || ch.Conn.Info == nil || ch.Conn.Base != "http://"+old || ch.Warning != localserver.LegacyNotice {
			t.Fatalf("%s: %+v", mode, ch)
		}
	}
	// The default answering wins and says nothing.
	now := jitllmdAt(t, "")
	ch := Choose(ctx, ModeAuto, now, localserver.WithDefaults(now, old))
	if ch.Conn == nil || ch.Conn.Base != "http://"+now || ch.Warning != "" {
		t.Fatalf("default answers: %+v", ch)
	}
	// An address the person set is not swapped.
	if ch := Choose(ctx, ModeAuto, unusedAddr(t), pair); ch.Conn != nil {
		t.Fatalf("a typed address fell back: %+v", ch)
	}
}
