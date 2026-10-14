//go:build linux

package engine

import (
	"context"
	"github.com/jitllm/jitllm/internal/testmodels/fresh"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jitllm/jitllm/common/session"
)

// The connected engine is gated against a real jitllmd: the binary built from
// this tree, serving on a free port, over the model folder the tests read.

var (
	jitllmdOnce sync.Once
	jitllmdBin  string
	jitllmdErr  error
)

// buildJitllmd builds server/cmd/jitllmd once per test binary.
func buildJitllmd(t *testing.T) string {
	t.Helper()
	jitllmdOnce.Do(func() {
		dir, err := os.MkdirTemp("", "jitllmd-bin")
		if err != nil {
			jitllmdErr = err
			return
		}
		jitllmdBin = filepath.Join(dir, "jitllmd"+exeSuffix())
		cmd := exec.Command("go", "build", "-o", jitllmdBin, "./cmd/jitllmd")
		cmd.Dir = filepath.Join("..", "..", "server")
		if out, err := cmd.CombinedOutput(); err != nil {
			jitllmdErr = err
			jitllmdBin = string(out)
		}
	})
	if jitllmdErr != nil {
		t.Fatalf("building jitllmd: %v\n%s", jitllmdErr, jitllmdBin)
	}
	return jitllmdBin
}

// freeAddr is a loopback address nothing listens on.
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

// startJitllmd serves the test models' folder at addr and returns what stops
// it. It waits until the service answers.
func startJitllmd(t *testing.T, addr string) (stop func()) {
	t.Helper()
	bin := buildJitllmd(t)
	cmd := exec.Command(bin, "serve", "-addr", addr, "-models", filepath.Dir(modelPath), "-version", "9.8.7")
	cmd.Stdout, cmd.Stderr = os.Stderr, os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	stopped := false
	stop = func() {
		if stopped {
			return
		}
		stopped = true
		cmd.Process.Kill()
		cmd.Wait()
	}
	t.Cleanup(stop)
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		if ch := Choose(context.Background(), ModeService, addr); ch.Conn.Info != nil {
			return stop
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("jitllmd never answered at %s", addr)
	return stop
}

func needModel(t *testing.T) {
	t.Helper()
	fresh.Must(t, modelPath)
	if _, err := os.Stat(modelPath); err != nil {
		t.Fatalf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proves nothing without it", err)
	}
}

// Auto connects to a jitllmd that answers, and the app's model list, block
// placement, memory and chat then run on it: a model loaded, a reply streamed
// a token at a time, the model unloaded.
func TestAutoPicksARunningJitllmd(t *testing.T) {
	needModel(t)
	addr := freeAddr(t)
	startJitllmd(t, addr)

	ch := Choose(context.Background(), ModeAuto, addr)
	if ch.Conn == nil || ch.Conn.Info == nil {
		t.Fatalf("Auto did not pick the jitllmd at %s: %+v", addr, ch)
	}
	sh := newTestShell()
	st := sh.Store
	var mu sync.Mutex
	var where Where
	r := NewRemote(sh, sh.state(), ch.Conn, ch.Addr, func(w Where) { mu.Lock(); where = w; mu.Unlock() })
	defer r.Close()
	mu.Lock()
	got := Badge(where, "")
	mu.Unlock()
	if want := "Service · " + addr + " (jitllmd 9.8.7)"; got != want {
		t.Errorf("badge %q, want %q", got, want)
	}

	r.Load(modelPath)
	pump(t, sh, 90*time.Second, "the model to load on the service", func() bool {
		if p := st.Problem.Get(); p.Seq != 0 {
			t.Fatalf("%s: %s", p.Title, p.Detail)
		}
		return st.Loaded.Get()
	})
	pump(t, sh, 10*time.Second, "the models list", func() bool { return len(st.Models.Get()) == 1 })
	m := st.Models.Get()[0]
	if m.Path != modelPath || !m.Active || m.Blocks == 0 {
		t.Errorf("listed %+v", m)
	}
	pump(t, sh, 10*time.Second, "the block map", func() bool { return len(st.BlockMap.Get()) == m.Blocks })
	if a := st.Alloc.Get(); a.NBlocks != m.Blocks || a.HostBlocks+a.DeviceBlocks != m.Blocks {
		t.Errorf("allocation %+v for %d blocks", a, m.Blocks)
	}
	if !strings.Contains(st.ModelSummary.Get(), "jitllmd at "+addr) {
		t.Errorf("the header does not say where the model runs: %q", st.ModelSummary.Get())
	}

	var partials []string
	unsub := st.Stream.SubscribeForever(func(v string) { partials = append(partials, v) })
	defer unsub()
	reply := st.AppendTurn(session.Turn{Role: session.RoleAssistant})
	r.Send(session.ChatRequest{Prompt: "Once upon a time", Reply: reply, MaxTokens: 12})
	pump(t, sh, 90*time.Second, "the reply", func() bool { return !st.Busy.Get() && st.Turn(reply).Text != "" })
	turn := st.Turn(reply)
	if turn.Tokens == 0 || turn.TokPerSec <= 0 {
		t.Errorf("the reply %+v has no tokens or no rate", turn)
	}
	if len(partials) < turn.Tokens/2 {
		t.Errorf("%d partials for %d tokens: the reply did not stream", len(partials), turn.Tokens)
	}

	r.Unload(modelPath)
	pump(t, sh, 10*time.Second, "the unload", func() bool { return len(st.Models.Get()) == 0 && !st.Loaded.Get() })
}

// With no jitllmd answering, Auto runs embedded, and says nothing about a
// service.
func TestAutoFallsBackToEmbedded(t *testing.T) {
	ch := Choose(context.Background(), ModeAuto, freeAddr(t))
	if ch.Conn != nil || ch.Warning != "" {
		t.Fatalf("nothing listens, and Auto chose %+v", ch)
	}
}

// Embedded beside a running jitllmd says that the two share the GPU.
func TestEmbeddedBesideAServiceWarns(t *testing.T) {
	addr := freeAddr(t)
	startJitllmd(t, addr)
	ch := Choose(context.Background(), ModeEmbedded, addr)
	if ch.Conn != nil || !strings.Contains(ch.Warning, "share") {
		t.Fatalf("Embedded beside a service: %+v", ch)
	}
}

// A service that stops answering is a problem on screen with Reconnect, the
// badge says so, and nothing switches to an engine of the app's own; when it
// is back, the app is connected again.
func TestALostServiceIsAProblemNotAFallback(t *testing.T) {
	addr := freeAddr(t)
	stop := startJitllmd(t, addr)
	ch := Choose(context.Background(), ModeAuto, addr)
	if ch.Conn == nil {
		t.Fatal("not connected")
	}
	sh := newTestShell()
	st := sh.Store
	var mu sync.Mutex
	var where Where
	r := NewRemote(sh, sh.state(), ch.Conn, ch.Addr, func(w Where) { mu.Lock(); where = w; mu.Unlock() })
	defer r.Close()

	stop()
	pump(t, sh, 3*heartbeat+5*time.Second, "the loss to show", func() bool { return st.Problem.Get().Seq != 0 })
	p := st.Problem.Get()
	if !strings.Contains(p.Title, "not answering") || p.Action != "Reconnect" || p.Do == nil {
		t.Fatalf("the loss shows as %+v", p)
	}
	mu.Lock()
	w := where
	mu.Unlock()
	if !w.Service || !w.Lost {
		t.Fatalf("after the loss the badge is %q", Badge(w, ""))
	}
	if !r.Where().Service {
		t.Fatal("the app left the service")
	}

	startJitllmd(t, addr)
	p.Do()
	pump(t, sh, 3*heartbeat+5*time.Second, "the reconnect", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return !where.Lost && st.Problem.Get().Seq == 0
	})
}

// Service mode with nothing answering starts connected-and-lost, never
// embedded.
func TestServiceModeWithNoServiceIsLost(t *testing.T) {
	addr := freeAddr(t)
	ch := Choose(context.Background(), ModeService, addr)
	if ch.Conn == nil || ch.Conn.Info != nil {
		t.Fatalf("Service mode chose %+v", ch)
	}
	sh := newTestShell()
	r := NewRemote(sh, sh.state(), ch.Conn, ch.Addr, nil)
	defer r.Close()
	pump(t, sh, 5*time.Second, "the problem", func() bool { return sh.Store.Problem.Get().Seq != 0 })
	if w := r.Where(); !w.Lost || w.Addr != addr {
		t.Fatalf("where %+v", w)
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
