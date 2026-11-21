package main

import (
	"bytes"
	"context"
	"fmt"
	"github.com/jitllm/jitllm/internal/testbin"
	"net"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/internal/localserver"
	"github.com/jitllm/jitllm/internal/localserver/service"
)

// TestServiceRunsUnderSystemd installs a unit of its own name (never the
// person's jitllm.service) in this session's systemd user manager, waits for
// the jitllmd it runs to answer, stops and starts it through `jitllmd
// service`, and removes it. It skips where there is no user manager.
func TestServiceRunsUnderSystemd(t *testing.T) {
	if out, err := exec.Command("systemctl", "--user", "is-system-running").Output(); err != nil &&
		!strings.Contains(string(out), "degraded") {
		t.Skipf("no systemd user manager here (%v: %s)", err, out)
	}
	bin := testbin.Build(t, "jitllmd", ".", ".")
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := l.Addr().String()
	l.Close()

	name := fmt.Sprintf("jitllm-test-%d", os.Getpid())
	cl := cli{thisService: func() (service.Service, error) {
		s, err := cli{}.service()
		s.Name = name
		return s, err
	}}
	s, err := cl.service()
	if err != nil {
		t.Fatal(err)
	}
	if s.Installed() {
		t.Fatalf("%s already exists", s.File())
	}
	defer func() {
		s.Uninstall()
		if s.Installed() {
			t.Errorf("%s left behind", s.File())
		}
	}()

	models := t.TempDir()
	do := func(args ...string) string {
		var out bytes.Buffer
		if err := cmdService(context.Background(), cli{out: &out, err: &out, thisService: cl.thisService}, args); err != nil {
			j, _ := exec.Command("journalctl", "--user", "-u", name, "-n", "30", "--no-pager").CombinedOutput()
			t.Fatalf("jitllmd service %v: %v\n%s\njournal:\n%s", args, err, out.String(), j)
		}
		return out.String()
	}
	out := do("install", "-addr", addr, "-models", models, "-jitllmd", bin)
	if !strings.Contains(out, "running") {
		t.Fatalf("install did not report the service up:\n%s", out)
	}
	if out := do("status", "-addr", addr); !strings.Contains(out, "models    "+models) || !strings.Contains(out, "installed") {
		t.Errorf("status:\n%s", out)
	}
	do("stop", "-addr", addr)
	gone := false
	for i := 0; i < 50 && !gone; i++ {
		_, err := localserver.Probe(context.Background(), addr)
		gone = err != nil
		time.Sleep(100 * time.Millisecond)
	}
	if !gone {
		t.Fatal("still answering after stop")
	}
	if out := do("start", "-addr", addr); !strings.Contains(out, "running") {
		t.Fatalf("start:\n%s", out)
	}
	do("uninstall")
	if s.Installed() {
		t.Fatal("uninstall left the unit")
	}
}
