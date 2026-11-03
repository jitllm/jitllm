package main

import (
	"net"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/internal/localserver"
	"github.com/jitllm/jitllm/server"
)

// TestTheDaemonListensOnLoopbackByDefault: `jitllmd serve` and every client
// verb default to the loopback address, not every interface.
func TestTheDaemonListensOnLoopbackByDefault(t *testing.T) {
	if defaultAddr != "localhost:4242" {
		t.Fatalf("defaultAddr = %q", defaultAddr)
	}
	if !server.Loopback(defaultAddr) {
		t.Fatalf("%s is not loopback", defaultAddr)
	}
	if w := exposure(defaultAddr, ""); w != "" {
		t.Errorf("the default warned: %s", w)
	}
}

func TestBindingBeyondLoopbackWithoutAKeyWarns(t *testing.T) {
	for _, addr := range []string{":4242", "0.0.0.0:4242", "192.0.2.9:4242"} {
		w := exposure(addr, "")
		if !strings.Contains(w, "WARNING") || !strings.Contains(w, "-api-key") || !strings.Contains(w, addr) {
			t.Errorf("%s: warning %q", addr, w)
		}
		if w := exposure(addr, "k"); w != "" {
			t.Errorf("%s with a key warned: %s", addr, w)
		}
	}
}

func guardedServer(t *testing.T, key string) string {
	t.Helper()
	e := server.New(server.Config{ModelDir: t.TempDir(), Probe: func() ([]server.DeviceInfo, error) { return nil, nil }})
	t.Cleanup(e.Close)
	s := httptest.NewServer(nil)
	s.Config.Handler = server.Guard{APIKey: key, Addr: s.Listener.Addr().String()}.Wrap(e.Handler())
	t.Cleanup(s.Close)
	return s.URL
}

// TestVerbsSendTheKey: a verb with -api-key (or $JITLLM_API_KEY) reaches a
// keyed daemon, and one without says what is missing.
func TestVerbsSendTheKey(t *testing.T) {
	t.Setenv("JITLLM_API_KEY", "")
	addr := guardedServer(t, "kk")

	var o, e syncBuffer
	err := runVerb(t, cmdDevices, cli{out: &o, err: &e}, []string{"-addr", addr}, 20*time.Second)
	if err == nil || !strings.Contains(err.Error(), "API key") {
		t.Fatalf("no key: %v", err)
	}
	if got := out(t, cmdDevices, "-addr", addr, "-api-key", "kk"); got == "" {
		t.Error("the flag's key got no devices")
	}
	t.Setenv("JITLLM_API_KEY", "kk")
	if got := out(t, cmdDevices, "-addr", addr); got == "" {
		t.Error("the environment's key got no devices")
	}
}

func unusedPort(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().String()
}

// TestVerbsFindAServiceOnTheOldPort: with nothing on the default address and
// a jitllmd on the old one, a verb uses it and says once that it runs there.
func TestVerbsFindAServiceOnTheOldPort(t *testing.T) {
	t.Setenv("JITLLM_API_KEY", "")
	old, _ := engineServer(t, server.Config{})
	primary := unusedPort(t)
	addrOpts := []localserver.Option{localserver.WithDefaults(primary, strings.TrimPrefix(old, "http://"))}

	var o, e syncBuffer
	if err := runVerb(t, cmdDevices, cli{out: &o, err: &e, addrOpts: addrOpts}, []string{"-addr", primary}, 20*time.Second); err != nil {
		t.Fatalf("%v\n%s", err, e.String())
	}
	if o.String() == "" || strings.Count(e.String(), localserver.LegacyNotice) != 1 {
		t.Fatalf("stdout %q, stderr %q", o.String(), e.String())
	}
	// A typed address is not swapped for the old one.
	var o2, e2 syncBuffer
	err := runVerb(t, cmdDevices, cli{out: &o2, err: &e2, addrOpts: addrOpts}, []string{"-addr", unusedPort(t)}, 20*time.Second)
	if err == nil || strings.Contains(e2.String(), localserver.LegacyNotice) {
		t.Fatalf("a typed address fell back: %v %q", err, e2.String())
	}
}
