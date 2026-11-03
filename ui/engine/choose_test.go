//go:build linux

package engine

import (
	"context"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	core "github.com/jitllm/jitllm/common/engine"
	"github.com/jitllm/jitllm/ui/app"
)

// The window's engine choice against a real jitllmd, built from this tree.

func freeAddr(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	a := ln.Addr().String()
	ln.Close()
	return a
}

// startJitllmd builds and serves jitllmd at addr until the test ends.
func startJitllmd(t *testing.T, addr string) (stop func()) {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "jitllmd"+exeSuffix())
	build := exec.Command("go", "build", "-o", bin, "./cmd/jitllmd")
	build.Dir = filepath.Join("..", "..", "server")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building jitllmd: %v\n%s", err, out)
	}
	cmd := exec.Command(bin, "serve", "-addr", addr, "-models", t.TempDir(), "-version", "3.2.1")
	cmd.Stdout, cmd.Stderr = os.Stderr, os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	done := false
	stop = func() {
		if !done {
			done = true
			cmd.Process.Kill()
			cmd.Wait()
		}
	}
	t.Cleanup(stop)
	for deadline := time.Now().Add(30 * time.Second); time.Now().Before(deadline); time.Sleep(50 * time.Millisecond) {
		if ch := core.Choose(context.Background(), core.ModeService, addr); ch.Conn.Info != nil {
			return stop
		}
	}
	t.Fatalf("jitllmd never answered at %s", addr)
	return stop
}

func pumpUntil(t *testing.T, sh *app.Shell, what string, want func() bool) {
	t.Helper()
	for deadline := time.Now().Add(15 * time.Second); time.Now().Before(deadline); time.Sleep(5 * time.Millisecond) {
		sh.DrainQueue(0)
		if want() {
			return
		}
	}
	t.Fatalf("timed out waiting for %s (badge %q, status %q)", what, sh.Badge().Get(), sh.Store.Status.Get())
}

// Under Auto the window connects to a jitllmd that answers and says so in
// its badge; a lost service is a problem with Reconnect and the badge says
// it, and the window does not start an engine of its own.
func TestTheWindowPicksTheServiceFirst(t *testing.T) {
	addr := freeAddr(t)
	stop := startJitllmd(t, addr)

	sh := app.NewShell(nil, app.DefaultConfig(app.Env{}))
	sh.Store.ServiceAddr.Set(addr)
	e := Start(sh)
	defer e.Close()
	if _, ok := e.Driver().Current().(*core.Remote); !ok {
		t.Fatalf("Auto runs %T beside a jitllmd that answers", e.Driver().Current())
	}
	want := "Service · " + addr + " (jitllmd 3.2.1)"
	if got := sh.Badge().Get(); got != want {
		t.Fatalf("badge %q, want %q", got, want)
	}

	stop()
	pumpUntil(t, sh, "the loss", func() bool { return sh.Store.Problem.Get().Seq != 0 })
	if p := sh.Store.Problem.Get(); p.Action != "Reconnect" {
		t.Errorf("the loss shows as %+v", p)
	}
	if got := sh.Badge().Get(); !strings.Contains(got, "not answering") {
		t.Errorf("after the loss the badge says %q", got)
	}
	if _, ok := e.Driver().Current().(*core.Remote); !ok {
		t.Errorf("after the loss the window runs %T", e.Driver().Current())
	}
}

// With nothing answering, Auto runs the window's own engine, and the badge
// says that.
func TestTheWindowFallsBackToItsOwnEngine(t *testing.T) {
	sh := app.NewShell(nil, app.DefaultConfig(app.Env{}))
	sh.Store.ServiceAddr.Set(freeAddr(t))
	e := Start(sh)
	defer e.Close()
	if _, ok := e.Driver().Current().(*core.Engine); !ok {
		t.Fatalf("nothing answers, and Auto runs %T", e.Driver().Current())
	}
	if got := sh.Badge().Get(); got != "Embedded · API off" {
		t.Fatalf("badge %q", got)
	}
}

// Connect, offered where the API was refused, switches the running window to
// the service.
func TestConnectSwitchesToTheService(t *testing.T) {
	addr := freeAddr(t)
	startJitllmd(t, addr)
	sh := app.NewShell(nil, app.DefaultConfig(app.Env{}))
	sh.Store.EngineMode.Set(core.ModeEmbedded)
	e := Start(sh)
	defer e.Close()
	if _, ok := e.Driver().Current().(*core.Engine); !ok {
		t.Fatalf("Embedded runs %T", e.Driver().Current())
	}
	sh.ConnectTo(addr)
	pumpUntil(t, sh, "the switch", func() bool {
		_, ok := e.Driver().Current().(*core.Remote)
		return ok
	})
	if sh.Store.EngineMode.Get() != core.ModeService || sh.Store.ServiceAddr.Get() != addr {
		t.Errorf("the setting is %q at %q", sh.Store.EngineMode.Get(), sh.Store.ServiceAddr.Get())
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
