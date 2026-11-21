package api

import (
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/server"
)

func TestStartServesAndCloses(t *testing.T) {
	e := server.New(server.Config{Admit: testdevices.Admit(), ModelDir: t.TempDir()})
	defer e.Close()
	s, err := Start("127.0.0.1:0", e)
	if err != nil {
		t.Fatal(err)
	}
	r, err := http.Get("http://" + s.Addr() + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(r.Body)
	r.Body.Close()
	if string(b) != "ok\n" {
		t.Fatalf("healthz: %q", b)
	}
	if _, err := Start(s.Addr(), e); err == nil {
		t.Fatal("a second Start on the same address succeeded")
	}
	s.Close()
	if _, err := http.Get("http://" + s.Addr() + "/healthz"); err == nil {
		t.Fatal("still serving after Close")
	}
}

func TestToggle(t *testing.T) {
	e := server.New(server.Config{Admit: testdevices.Admit(), ModelDir: t.TempDir()})
	defer e.Close()
	s, msg, _ := Toggle(nil, true, "127.0.0.1:0", e)
	if s == nil {
		t.Fatal(msg)
	}
	if again, _, _ := Toggle(s, true, "127.0.0.1:0", e); again != s {
		t.Fatal("an unchanged setting restarted the server")
	}
	if off, msg, _ := Toggle(s, false, "", e); off != nil || msg != "API off" {
		t.Fatalf("off: %v %q", off, msg)
	}
}

// TestToggleRefusesAHeldAddress: turning the API on where a jitllmd already
// answers is refused, naming the service, rather than failing on the bind or
// -- where both could bind -- becoming a second engine nobody asked for.
func TestToggleRefusesAHeldAddress(t *testing.T) {
	svc := server.New(server.Config{Admit: testdevices.Admit(), ModelDir: t.TempDir(), Version: "9.9.9"})
	defer svc.Close()
	held, err := Start("127.0.0.1:0", svc)
	if err != nil {
		t.Fatal(err)
	}
	defer held.Close()

	e := server.New(server.Config{Admit: testdevices.Admit(), ModelDir: t.TempDir()})
	defer e.Close()
	s, msg, in := Toggle(nil, true, held.Addr(), e)
	if s != nil {
		s.Close()
		t.Fatalf("started beside a service: %q", msg)
	}
	if in.Addr != held.Addr() || in.Version != "9.9.9" {
		t.Fatalf("held = %+v, want the service at %s, version 9.9.9", in, held.Addr())
	}
	if !strings.Contains(msg, "connect to it") || !strings.Contains(msg, "9.9.9") {
		t.Fatalf("the refusal does not offer the service: %q", msg)
	}
}
