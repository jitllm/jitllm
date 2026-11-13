package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// sseWriter is Server-Sent Events for the two compatibility shims.
//
// Every frame is flushed explicitly; otherwise Go's response writer buffers
// and the client receives the whole stream at the end. The gate asserts that
// frames arrive before the generate finishes.
type sseWriter struct {
	w  http.ResponseWriter
	rc *http.ResponseController
	// mu orders the frames against a ping (pings); last is when the stream
	// last wrote.
	mu   sync.Mutex
	last time.Time
}

func newSSE(w http.ResponseWriter) (*sseWriter, error) {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	// Proxies that buffer defeat the flush above just as effectively as Go's
	// own writer does.
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	s := &sseWriter{w: w, rc: http.NewResponseController(w), last: time.Now()}
	if err := s.rc.Flush(); err != nil && !errors.Is(err, http.ErrNotSupported) {
		return nil, err
	}
	return s, nil
}

// send writes one unnamed `data:` frame, which is the OpenAI shape.
func (s *sseWriter) send(v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := fmt.Fprintf(s.w, "data: %s\n\n", b); err != nil {
		return err
	}
	s.rc.Flush()
	s.last = time.Now()
	return nil
}

// sendNamed writes an `event: <name>` frame, which is the Anthropic shape:
// clients dispatch on the event name, and message_stop ends the stream with
// no sentinel. OpenAI's anonymous data: chunks end with data: [DONE].
func (s *sseWriter) sendNamed(name string, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := fmt.Fprintf(s.w, "event: %s\ndata: %s\n\n", name, b); err != nil {
		return err
	}
	s.rc.Flush()
	s.last = time.Now()
	return nil
}

// done writes OpenAI's terminator. It is not part of SSE; clients written
// against the OpenAI spec block waiting for it.
func (s *sseWriter) done() {
	s.raw([]byte("data: [DONE]\n\n"))
}

// raw writes b as it is.
func (s *sseWriter) raw(b []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.w.Write(b)
	s.rc.Flush()
	s.last = time.Now()
}

// pings writes an SSE comment (":") whenever the stream has been silent for
// every, as llama-server does, so a client that times out on silence keeps
// the connection. stop ends them and returns once no ping can be written.
func (s *sseWriter) pings(every time.Duration) (stop func()) {
	quit, exited := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(exited)
		t := time.NewTimer(every)
		defer t.Stop()
		for {
			select {
			case <-quit:
				return
			case <-t.C:
			}
			s.mu.Lock()
			idle := time.Since(s.last)
			if idle >= every {
				fmt.Fprint(s.w, ":\n\n")
				s.rc.Flush()
				s.last, idle = time.Now(), 0
			}
			s.mu.Unlock()
			t.Reset(every - idle)
		}
	}()
	return func() {
		close(quit)
		<-exited
	}
}

// sendError is the only way to report a failure once the status line is
// already 200.
func (s *sseWriter) sendError(err error) {
	s.sendNamed("error", map[string]any{
		"error": map[string]string{"type": "server_error", "message": err.Error()},
	})
}
