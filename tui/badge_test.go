package main

import (
	"context"
	"github.com/jitllm/jitllm/internal/testbin"
	"net"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/jitllm/jitllm/common/api"
	"github.com/jitllm/jitllm/common/engine"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/server"
)

// footerOf is the status bar's text.
func footerOf(r *root) string {
	lines := strings.Split(stripANSI(r.View()), "\n")
	return lines[len(lines)-1]
}

// The status bar says which engine runs on every screen and at every width,
// and what the API is doing.
func TestTheStatusBarAlwaysSaysWhichEngine(t *testing.T) {
	r := newTestRoot(t)
	for _, w := range []int{160, 80, 60} {
		r.Update(tea.WindowSizeMsg{Width: w, Height: 30})
		for s := range screenNames() {
			r.setScreen(screen(s))
			if f := footerOf(r); !strings.Contains(f, "Embedded · API off") {
				t.Errorf("%s at %d columns: no badge in %q", screenNames()[s], w, f)
			}
		}
	}
	r.Update(tea.WindowSizeMsg{Width: 160, Height: 44})
	r.env.cfg.API, r.env.cfg.APIAddr = true, "127.0.0.1:0"
	r.env.syncAPI()
	defer r.env.api.Close()
	if f := footerOf(r); !strings.Contains(f, "Embedded · API on :") {
		t.Errorf("API on, the badge says %q", f)
	}
	r.env.where = engine.Where{Service: true, Addr: "localhost:4242", Version: "1.2.3"}
	if f := footerOf(r); !strings.Contains(f, "Service · localhost:4242 (jitllmd 1.2.3)") {
		t.Errorf("connected, the badge says %q", f)
	}
}

// Turning the API on where a jitllmd answers is refused, the setting goes
// back off, and the line says how to connect instead.
func TestTheAPIIsRefusedWhereAServiceAnswers(t *testing.T) {
	svc := server.New(server.Config{Admit: testdevices.Admit(), ModelDir: t.TempDir(), Version: "4.5.6"})
	defer svc.Close()
	held, err := api.Start("127.0.0.1:0", svc)
	if err != nil {
		t.Fatal(err)
	}
	defer held.Close()

	r := newTestRoot(t)
	r.env.cfg.API, r.env.cfg.APIAddr = true, held.Addr()
	msg := r.env.syncAPI()
	if r.env.api != nil {
		r.env.api.Close()
		t.Fatalf("an API started beside the service: %q", msg)
	}
	if r.env.cfg.API {
		t.Error("the setting still says the API is on")
	}
	if !strings.Contains(msg, "--connect "+held.Addr()) {
		t.Errorf("the refusal does not say how to connect: %q", msg)
	}
	if f := footerOf(r); !strings.Contains(f, "Embedded · API off") {
		t.Errorf("after the refusal the badge says %q", f)
	}
}

// Connected to a jitllmd, the status bar names it; when it stops answering
// the badge says so and the palette offers to reconnect, and the app keeps
// its connected engine rather than starting one of its own.
func TestTheStatusBarFollowsAConnectedService(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	ln.Close()
	bin := testbin.Build(t, "jitllmd", filepath.Join("..", "server"), "./cmd/jitllmd")
	cmd := exec.Command(bin, "serve", "-addr", addr, "-models", t.TempDir(), "-version", "5.4.3")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { cmd.Process.Kill(); cmd.Wait() }()
	var ch engine.Choice
	for deadline := time.Now().Add(30 * time.Second); time.Now().Before(deadline); time.Sleep(50 * time.Millisecond) {
		if ch = engine.Choose(context.Background(), engine.ModeAuto, addr); ch.Conn != nil {
			break
		}
	}
	if ch.Conn == nil {
		t.Fatalf("Auto did not connect to the jitllmd at %s", addr)
	}

	r := newTestRoot(t)
	rem := engine.NewRemote(r.env.f, r.env.f.st, ch.Conn, ch.Addr, nil)
	defer rem.Close()
	r.env.e, r.env.where = rem, rem.Where()
	if f := footerOf(r); !strings.Contains(f, "Service · "+addr+" (jitllmd 5.4.3)") {
		t.Fatalf("connected, the badge says %q", f)
	}
	cmd.Process.Kill()
	cmd.Wait()
	for deadline := time.Now().Add(10 * time.Second); time.Now().Before(deadline) && !rem.Where().Lost; time.Sleep(20 * time.Millisecond) {
	}
	r.env.where = rem.Where()
	if f := footerOf(r); !strings.Contains(f, "not answering") {
		t.Fatalf("after the loss the badge says %q", f)
	}
	var titles []string
	for _, a := range actions(r) {
		titles = append(titles, a.title)
	}
	if !strings.Contains(strings.Join(titles, "|"), "Reconnect to jitllmd at "+addr) {
		t.Errorf("the palette offers no reconnect: %v", titles)
	}
	if _, ok := r.env.e.(*engine.Remote); !ok {
		t.Error("the app left the service")
	}
}

// exeSuffix is the platform's executable suffix: go build -o names the file
// as given, and Windows runs only one ending in .exe.
func exeSuffix() string {
	if runtime.GOOS == "windows" {
		return ".exe"
	}
	return ""
}
