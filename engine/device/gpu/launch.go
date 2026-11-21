package gpu

import (
	"github.com/jitllm/jitllm/backend"
)

// launcher is a submission's launch path: its session, and the slice every
// launch's buffer list is copied into before the call reaches the session. A
// variadic slice handed to an interface method escapes, so a launch with its
// buffers listed at the call site was a heap object per dispatch -- every
// dispatch of a submission that issues its launches rather than replaying a
// graph (Vulkan and Metal always, CUDA when it captures). Through launch the
// caller's list stays on its stack and the copy is reused.
//
// It is a concrete type passed by pointer, never a func value: a call through
// a func value is opaque to escape analysis and its arguments escape too.
type launcher struct {
	// to is the session and the reused list, on the devTier. Behind a pointer
	// because escape analysis does not tell one field from another: the
	// session escapes into its own Launch, and as a field here it would carry
	// the pointers to the caller's locals below to the heap with it.
	to *launchTo
	// err is the submission's latched error.
	err *error
	// g is the submission's device view, for the persistent path's plans
	// (persist.go).
	g *devTier
}

// issue launches one of the composer's forms' kernels.
func (l *launcher) issue(k backend.Kernel, groups, width int, bufs ...backend.Buf) error {
	t := l.to
	if ps := t.ps; ps != nil && ps.on {
		if ps.add(k, groups, width, bufs) {
			return nil
		}
		// Full: what is held goes first, and this launch after it.
		l.persistDrain()
		if *l.err != nil {
			return *l.err
		}
	}
	t.bufs = append(t.bufs[:0], bufs...)
	err := t.s.Launch(k, groups, width, t.bufs...)
	clear(t.bufs)
	return err
}

// launchTo is where a launcher sends its launches: the submission's session
// and the buffer list reused across submissions.
type launchTo struct {
	s    backend.Session
	bufs []backend.Buf
	// ps is the persistent path's collection state (persist.go), nil until
	// the session first runs with Config.Persist; psess the session a
	// collecting submission runs its own calls through, so none overtakes
	// what is held.
	ps    *persister
	psess persistSession
}

// leave drops the session once its submission is done.
func (t *launchTo) leave() { t.s = nil }
