package engine

import (
	"os"
	"strings"
	"testing"
)

// procCounts is what /proc says of the process: what a leak gate reads beside
// the live heap where the host has /proc, and leaves out where it has not.
type procCounts struct {
	ok       bool
	rss, fds int
	th       threadCensus
}

func procNow(t *testing.T) procCounts {
	t.Helper()
	return procCounts{ok: true, rss: rssKB(t), fds: openFDs(t), th: censusThreads(t)}
}

// threadCensus splits the process's OS threads into the Go runtime's Ms and
// everyone else's. Every M the runtime makes inherits the process's name, and a
// driver names the threads it starts (CUDA's "cuda0000...", Mesa's queue
// threads), so the name tells them apart.
type threadCensus struct{ goMs, foreign int }

func censusThreads(t *testing.T) threadCensus {
	t.Helper()
	self, err := os.ReadFile("/proc/self/comm")
	if err != nil {
		t.Skipf("no /proc/self/comm: %v", err)
	}
	name := strings.TrimSpace(string(self))
	ents, err := os.ReadDir("/proc/self/task")
	if err != nil {
		t.Skipf("no /proc/self/task: %v", err)
	}
	var c threadCensus
	for _, e := range ents {
		b, err := os.ReadFile("/proc/self/task/" + e.Name() + "/comm")
		if err != nil {
			continue // the thread exited under us
		}
		if strings.TrimSpace(string(b)) == name {
			c.goMs++
		} else {
			c.foreign++
		}
	}
	return c
}
