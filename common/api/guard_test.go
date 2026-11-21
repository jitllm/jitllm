package api

import (
	"net/http"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/server"
)

func get(t *testing.T, url, host, bearer string) int {
	t.Helper()
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		t.Fatal(err)
	}
	if host != "" {
		req.Host = host
	}
	if bearer != "" {
		req.Header.Set("Authorization", "Bearer "+bearer)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	return resp.StatusCode
}

func TestTheAPIDefaultsToLoopback4242(t *testing.T) {
	if DefaultAddr != "localhost:4242" || !server.Loopback(DefaultAddr) {
		t.Fatalf("DefaultAddr = %q", DefaultAddr)
	}
}

// TestTheAPIIsGuarded: the window's and the terminal's API refuses what
// jitllmd refuses: a missing key, a rebound Host, a foreign origin.
func TestTheAPIIsGuarded(t *testing.T) {
	e := server.New(server.Config{Admit: testdevices.Admit(), ModelDir: t.TempDir()})
	defer e.Close()
	s, err := Start("127.0.0.1:0", e, WithKey("kk"))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	base := "http://" + s.Addr()

	if c := get(t, base+"/v1/models", "", ""); c != http.StatusUnauthorized {
		t.Errorf("no key: %d", c)
	}
	if c := get(t, base+"/v1/models", "", "kk"); c != http.StatusOK {
		t.Errorf("the key: %d", c)
	}
	if c := get(t, base+"/healthz", "", ""); c != http.StatusOK {
		t.Errorf("healthz: %d", c)
	}
	if c := get(t, base+"/v1/models", "evil.example", "kk"); c != http.StatusForbidden {
		t.Errorf("rebound Host: %d", c)
	}
	req, _ := http.NewRequest(http.MethodGet, base+"/v1/models", nil)
	req.Header.Set("Authorization", "Bearer kk")
	req.Header.Set("Origin", "https://evil.example")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusForbidden {
		t.Errorf("foreign origin: %d", resp.StatusCode)
	}
}

// TestToggleRefusesAKeyedService: a jitllmd that wants a key is still a
// jitllmd holding the address.
func TestToggleRefusesAKeyedService(t *testing.T) {
	svc := server.New(server.Config{Admit: testdevices.Admit(), ModelDir: t.TempDir(), Version: "9.9.9"})
	defer svc.Close()
	held, err := Start("127.0.0.1:0", svc, WithKey("kk"))
	if err != nil {
		t.Fatal(err)
	}
	defer held.Close()
	e := server.New(server.Config{Admit: testdevices.Admit(), ModelDir: t.TempDir()})
	defer e.Close()
	s, msg, in := Toggle(nil, true, held.Addr(), e)
	if s != nil || in.Addr == "" || msg == "" {
		t.Fatalf("a keyed service did not hold the address: %v %q %+v", s, msg, in)
	}
}
