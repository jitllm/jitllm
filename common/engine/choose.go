package engine

import (
	"context"
	"errors"
	"sync"

	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/internal/localserver"
	"github.com/jitllm/jitllm/server/client"
)

// Driver is the engine a front end drives: the embedded Engine or a Remote
// connected to a jitllmd. Every method queues and returns.
type Driver interface {
	Load(path string)
	Use(path string)
	Unload(path string)
	Pin(path string, bytes uint64)
	SetPriority(on bool)
	Relocate(n int)
	// MoveToNextDevice moves the chat's device blocks to the next device of
	// its tier, their history with them.
	MoveToNextDevice()
	Run(job func()) bool
	Send(r session.ChatRequest)
	Stop()
	Close()
}

// The engine settings. The zero value is Auto.
const (
	// ModeAuto connects to a jitllmd that answers at the address and runs
	// embedded only when none does.
	ModeAuto = "auto"
	// ModeEmbedded runs the engine in the app's own process.
	ModeEmbedded = "embedded"
	// ModeService connects to the address and nothing else: a service that
	// does not answer is shown as not answering, never replaced.
	ModeService = "service"
)

// Choice is what a front end starts: Conn non-nil is a Remote (Conn.Info nil
// when the service did not answer), nil the embedded engine. Warning is a
// line to show beside either; Addr is the address asked.
type Choice struct {
	Conn    *client.Conn
	Addr    string
	Warning string
	// Beside is an embedded choice with a compatible jitllmd answering at
	// Addr, which a front end offers to connect to.
	Beside bool
}

// Choose decides the engine for a setting. addr empty is the default address
// (the login service's). It asks the network once, within two seconds; ask it
// before the front end's own API starts, or Auto finds the app itself.
//
// A service on the old default port (localserver.LegacyAddr) is found when
// the default address answers nothing, and the Choice's Warning says it runs
// there. localserver.WithAPIKey is the key to ask the service with.
func Choose(ctx context.Context, mode, addr string, opts ...localserver.Option) Choice {
	if addr == "" {
		addr = localserver.DefaultAddr
	}
	ch := Choice{Addr: addr}
	switch mode {
	case ModeEmbedded:
		if in, w, ok := Beside(ctx, addr, opts...); ok {
			ch.Warning = w
			ch.Beside = localserver.Compatible(in.APIVersion) == nil
		}
		return ch
	case ModeService:
		c, err := client.Discover(ctx, addr, opts...)
		if err != nil {
			at := addr
			if c != nil && c.Legacy {
				at = c.Base
			}
			c = client.Dial(at, opts...)
			c.Info = nil
		}
		ch.Conn = c
		switch {
		case errors.Is(err, localserver.ErrIncompatible) || errors.Is(err, localserver.ErrUnauthorized):
			ch.Warning = err.Error()
		case c.Legacy:
			ch.Warning = localserver.LegacyNotice
		}
		return ch
	}
	c, err := client.Discover(ctx, addr, opts...)
	switch {
	case err == nil:
		ch.Conn = c
		if c.Legacy {
			ch.Warning = localserver.LegacyNotice
		}
	case errors.Is(err, localserver.ErrUnauthorized):
		ch.Warning = err.Error() + ". Running this app's own engine beside it; the two share the GPU's memory."
	case errors.Is(err, localserver.ErrIncompatible):
		// Refused, and said: the app runs its own engine beside it.
		ch.Warning = err.Error() + ". Running this app's own engine beside it; the two share the GPU's memory."
	}
	return ch
}

// Switch is a Driver that forwards to the engine in use, so a front end can
// change engines (Settings, Connect) without rebuilding the screens holding
// it.
type Switch struct {
	mu sync.Mutex
	d  Driver
}

// Set makes d the engine in use and closes the one before it.
func (s *Switch) Set(d Driver) {
	s.mu.Lock()
	old := s.d
	s.d = d
	s.mu.Unlock()
	if old != nil {
		old.Close()
	}
}

// Current is the engine in use, nil before the first Set.
func (s *Switch) Current() Driver {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.d
}

// Between engines (Set(nil), then the next) a call does nothing: the
// screens may click while a switch is being chosen.
func (s *Switch) Load(p string)          { s.do(func(d Driver) { d.Load(p) }) }
func (s *Switch) Use(p string)           { s.do(func(d Driver) { d.Use(p) }) }
func (s *Switch) Unload(p string)        { s.do(func(d Driver) { d.Unload(p) }) }
func (s *Switch) Pin(p string, b uint64) { s.do(func(d Driver) { d.Pin(p, b) }) }
func (s *Switch) SetPriority(on bool)    { s.do(func(d Driver) { d.SetPriority(on) }) }
func (s *Switch) Relocate(n int)         { s.do(func(d Driver) { d.Relocate(n) }) }
func (s *Switch) MoveToNextDevice()      { s.do(func(d Driver) { d.MoveToNextDevice() }) }
func (s *Switch) Send(r session.ChatRequest) {
	s.do(func(d Driver) { d.Send(r) })
}
func (s *Switch) Stop() { s.do(func(d Driver) { d.Stop() }) }
func (s *Switch) Run(job func()) bool {
	ok := false
	s.do(func(d Driver) { ok = d.Run(job) })
	return ok
}

func (s *Switch) do(f func(Driver)) {
	if d := s.Current(); d != nil {
		f(d)
	}
}

// Close closes the engine in use.
func (s *Switch) Close() { s.Set(nil) }
