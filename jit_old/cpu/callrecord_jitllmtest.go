//go:build jitllmtest

package cpu

import (
	"sync"
	"unsafe"
)

// Called is a kernel a recording saw called: its code and how often.
type Called struct {
	Name  string
	Code  []byte
	Calls int
}

// Recording is the kernels called while it is open, each once, in the order
// first called.
type Recording struct {
	mu    sync.Mutex
	seen  map[*byte]int
	calls []Called
}

var recording struct {
	mu  sync.Mutex
	cur *Recording
}

// Record opens a recording of every kernel called until Stop.
func Record() *Recording {
	r := &Recording{seen: map[*byte]int{}}
	recording.mu.Lock()
	recording.cur = r
	recording.mu.Unlock()
	return r
}

// Stop closes the recording and is what it saw.
func (r *Recording) Stop() []Called {
	recording.mu.Lock()
	if recording.cur == r {
		recording.cur = nil
	}
	recording.mu.Unlock()
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.calls
}

func recordCall(c *Code) {
	recording.mu.Lock()
	r := recording.cur
	recording.mu.Unlock()
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if i, ok := r.seen[c.entry]; ok {
		r.calls[i].Calls++
		return
	}
	r.seen[c.entry] = len(r.calls)
	r.calls = append(r.calls, Called{Name: codeName(c), Code: append([]byte(nil), unsafe.Slice(c.entry, c.Size)...), Calls: 1})
}
