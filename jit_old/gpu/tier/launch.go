package tier

import (
	"errors"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
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
	// err is the submission's latched error; la launches nothing once it is
	// set.
	err *error
	// la's bookkeeping, shared with layersSession: fsrc is the float buffer
	// the current int8 activation was quantized from, recognised by a launch
	// that writes (a, ax), and memoK/memoSrc the ActF16 memo every launch
	// through la forgets.
	fsrc    *backend.Buf
	memoK   *backend.Kernel
	memoSrc *backend.Buf
	a, ax   backend.Buf
	// g is the submission's device view, for the persistent path's plans
	// (persist.go).
	g *devTier
}

// launch is s.Launch through the reused buffer list.
func (l *launcher) launch(k backend.Kernel, groups, width int, bufs ...backend.Buf) error {
	// A kernel a scratch never built is a refusal with a reason: handed to
	// the backend it stalled the device (a launch that never completes).
	if k == nil {
		return errNilKernel
	}
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

// la is the ordinary launch: threads in groups of 128, nothing once the
// submission has failed.
func (l *launcher) la(k backend.Kernel, threads int, bufs ...backend.Buf) {
	l.law(k, threads, 0, bufs...)
}

// law is la in workgroups of width threads (0 is 128): a staged GEMM's
// workgroup is its tile's warps, which an eight-warp tile makes 256.
func (l *launcher) law(k backend.Kernel, threads, width int, bufs ...backend.Buf) {
	if width <= 0 {
		width = 128
	}
	*l.memoK, *l.memoSrc = nil, nil
	if *l.err == nil {
		*l.err = l.launch(k, (threads+width-1)/width, width, bufs...)
	}
	if len(bufs) == 3 && bufs[1] == l.a && bufs[2] == l.ax {
		*l.fsrc = bufs[0]
	}
}

// errNilKernel is a launch of a kernel the scratch did not build.
var errNilKernel = errors.New("tier: a launch of a kernel this block's scratch did not build")
