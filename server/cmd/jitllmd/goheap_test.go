package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/sched"
)

// TestServeCapsTheGoHeap: `jitllmd serve` sets the collector's memory limit
// from the cgroup before it loads anything, as cmd/jitllm does (AGENTS.md
// RULE 2f), through the one shared helper. The daemon runs in a child
// process, because debug.SetMemoryLimit is process-wide.
//
// VIOLATION SIGNATURE. Delete the goheap.Cap call from serve and this fails
// with
//
//	serve came up without setting the go memory limit
func TestServeCapsTheGoHeap(t *testing.T) {
	if os.Getenv("JITLLMD_SERVE_CHILD") == "1" {
		serve([]string{"-addr", "127.0.0.1:0", "-models", t.TempDir()})
		return
	}
	lim := sched.MemLimit()
	if lim <= sched.GCHeadroom {
		t.Skipf("NO CGROUP LIMIT: sched.MemLimit() = %d, so there is nothing to cap -- this gate "+
			"proved nothing; run it under scripts/cap", lim)
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestServeCapsTheGoHeap$")
	cmd.Env = append(os.Environ(), "JITLLMD_SERVE_CHILD=1")
	stderr, err := cmd.StderrPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		cmd.Process.Kill()
		cmd.Wait()
	}()
	lines := make(chan string)
	go func() {
		sc := bufio.NewScanner(stderr)
		for sc.Scan() {
			lines <- sc.Text()
		}
		close(lines)
	}()
	want := fmt.Sprintf("go memory limit %.2f GiB", float64(lim-sched.GCHeadroom)/(1<<30))
	var seen []string
	deadline := time.After(30 * time.Second)
	for {
		select {
		case l, ok := <-lines:
			if !ok {
				t.Fatalf("serve exited before listening: %q", seen)
			}
			seen = append(seen, l)
			if l == want {
				return
			}
			if strings.HasPrefix(l, "jitllmd on ") {
				t.Fatalf("serve came up without setting the go memory limit (want %q): %q", want, seen)
			}
		case <-deadline:
			t.Fatalf("serve said nothing useful in 30s: %q", seen)
		}
	}
}
