//go:build linux

package cpu

import (
	"fmt"
	"os"
	"sync"
)

// perfMap is a Host's lines of /tmp/perf-<pid>.map (WithPerfMap), the file
// opened to append, as every Host of the process writes it.
type perfMap struct {
	mu sync.Mutex
	f  *os.File
}

func (p *perfMap) open() {
	f, err := os.OpenFile(fmt.Sprintf("/tmp/perf-%d.map", os.Getpid()), os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o644)
	if err == nil {
		p.f = f
	}
}

func (p *perfMap) add(addr uintptr, size int, name string) {
	if p.f == nil {
		return
	}
	p.mu.Lock()
	fmt.Fprintf(p.f, "%x %x %s\n", addr, size, name)
	p.f.Sync()
	p.mu.Unlock()
}

func (p *perfMap) close() error {
	if p.f == nil {
		return nil
	}
	return p.f.Close()
}
