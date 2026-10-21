package backend

import (
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/gpu/cuda"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/jit_old/gpu/ptx"
)

// cudaDev owns the device from one goroutine and serialises every call onto it.
//
// A CUDA context is current per OS thread, and goroutines migrate between
// threads (t.Run subtests, sched.Pool workers), which yields
// CUDA_ERROR_INVALID_CONTEXT. So one owner goroutine holds a locked thread with
// the context current, and other callers either post a closure to it or run
// inline with the context bound (see do and inline).
type cudaDev struct {
	allocCount
	d *cuda.Device
	// mu guards reqs against Close. do reads it and Close nils it, and without
	// this that is a data race whose benign-looking outcome is a send on a
	// closed channel -- a panic from the caller's goroutine, not an error.
	//
	// It is NOT held across the request: do waits for the owner goroutine to
	// finish, and holding a lock across that would serialise every device call
	// behind every other one and deadlock a request that itself calls do.
	mu   sync.Mutex
	reqs chan func()
	// owner is the OS thread the owner goroutine is locked to (-1 where the
	// platform cannot say); see do.
	owner int64
	// cap is the capture stream, created on first use and reused. A stream is
	// a resource rather than a mode: nothing reads it except the session that
	// is capturing, and that session runs on the owner goroutine.
	cap   *cuda.Stream
	noCap bool // stream creation failed; do not keep asking
	// up is the copy queue (backend.Uploader), a non-blocking stream created
	// on first use; noUp records that it could not be.
	up   *cuda.Stream
	noUp bool
	// tev is a ring of (before, after) event pairs around the last three
	// replays, and ti counts replays timed; see timeReplay.
	tev [3][2]*cuda.Event
	ti  int
	// lastMarks is the last timed replay's marks, settled at the next one.
	lastMarks []kernMark
	// run serialises inline calls (see do); in is the thread holding it, so a
	// call made from inside one runs straight through instead of waiting on
	// itself.
	run sync.Mutex
	in  atomic.Int64
	// life is read-held by every Session on a queue and write-held by Close,
	// so the context is never destroyed under one. queued is the threads
	// running such a Session (nQueued of them), so a call made from inside
	// one runs straight through as from inside an inline one; queues is every
	// queue not yet closed, which Close destroys before the context.
	life sync.RWMutex
	// capturing is read-held by a queue's Session from BeginRecord to
	// EndRecord and write-held by syncAll. cuCtxSynchronize waits for every
	// stream of the context, and waiting for a stream that is capturing is
	// illegal in any capture mode (the CUDA programming guide, "Prohibited
	// and Unhandled Operations"); racing an EndCapture on another thread it
	// faulted inside the driver (TestConcurrentMultiDeviceDoesNotLeak, a
	// lane's rope table written while a queue ended its capture).
	capturing sync.RWMutex
	qmu       sync.Mutex
	queued    map[int64]qRef
	nQueued   atomic.Int32
	queues    map[*cudaQueue]bool
	// posted routes every call through the owner goroutine instead of running
	// it inline; Opts.CUDAPosted, fixed at open.
	posted bool
	// maxRegs is Opts.CUDAMaxRegs, fixed at open; res is what each compiled
	// kernel cost (ResourceReporter).
	maxRegs int
	res     resLedger
	// reqFree is Session's free list; see Session.
	reqMu   sync.Mutex
	reqFree []*sessReq
	// peers is, per other device a CopyPeer has read from, whether peer
	// access is enabled between the two: asked once (peerTo).
	peerMu sync.Mutex
	peers  map[*cudaDev]bool
}

// CopyPeer is backend.PeerCopier: cuMemcpyPeer from another CUDA device's
// buffer into one of this device's, after the source's work has landed. The
// first copy from a device asks the driver whether the two cards address
// each other (NVLink or PCIe peer-to-peer) and enables it both ways; where
// they cannot, the driver stages the copy through the host itself, and
// direct says so.
func (c *cudaDev) CopyPeer(dst Buf, dstOff int, from Device, src Buf, srcOff, n int) (bool, error) {
	f, ok := from.(*cudaDev)
	db, ok1 := dst.(*cudaBuf)
	sb, ok2 := src.(*cudaBuf)
	if !ok || !ok1 || !ok2 || db == nil || sb == nil || db.dev != c || sb.dev != f {
		return false, fmt.Errorf("backend: cuda: a peer copy needs this device's buffer and one of the CUDA device it names")
	}
	if f == c {
		return false, c.Copy(dst, dstOff, src, srcOff, n)
	}
	if err := copyRange(db.b.Len(), dstOff, sb.b.Len(), srcOff, n, false); err != nil || n == 0 {
		return false, err
	}
	if !cuda.PeerCopyAvailable() {
		return false, fmt.Errorf("backend: cuda: this driver has no cuMemcpyPeer")
	}
	err := errCUDAClosed
	f.do(func() { err = f.syncAll() })
	if err != nil {
		return false, err
	}
	direct := c.peerTo(f)
	err = errCUDAClosed
	c.do(func() {
		if c.hasQueues() {
			if err = c.syncAll(); err != nil {
				return
			}
		}
		if err = cuda.CopyPeer(db.b, c.d, dstOff, sb.b, f.d, srcOff, n); err == nil {
			err = c.syncAll()
		}
	})
	return direct, err
}

// PeerAccess is backend.PeerCopier's: peerTo for a CUDA device of the same
// process.
func (c *cudaDev) PeerAccess(from Device) bool {
	f, ok := from.(*cudaDev)
	if !ok || f == c || !cuda.PeerCopyAvailable() {
		return false
	}
	return c.peerTo(f)
}

// peerTo enables peer access between c and f once, both ways, and reports
// whether the driver granted it.
func (c *cudaDev) peerTo(f *cudaDev) bool {
	c.peerMu.Lock()
	defer c.peerMu.Unlock()
	if v, ok := c.peers[f]; ok {
		return v
	}
	ok := c.d.CanAccessPeer(f.d) && f.d.CanAccessPeer(c.d)
	if ok {
		err := errCUDAClosed
		c.do(func() { err = c.d.EnablePeerAccess(f.d) })
		if err == nil {
			err = errCUDAClosed
			f.do(func() { err = f.d.EnablePeerAccess(c.d) })
		}
		ok = err == nil
	}
	if c.peers == nil {
		c.peers = map[*cudaDev]bool{}
	}
	c.peers[f] = ok
	return ok
}

// openCUDA is what backend.Open uses: the first card.
func openCUDA(o Opts) (Device, error) { return OpenCUDAWith(-1, o) }

// CUDACount is how many CUDA devices the driver sees, without opening one.
// Opening costs a context, a locked OS thread and an owner goroutine, so a
// caller asks this first and opens only the ordinals it chose.
func CUDACount() (int, error) { return cuda.Count() }

// OpenCUDA opens one CUDA device by ordinal, with its own context and its own
// owner goroutine. One owner per device keeps each context current on its own
// thread without pushing or popping per call; only the driver binding is
// shared.
func OpenCUDA(ord int) (Device, error) { return OpenCUDAWith(ord, Opts{}) }

// OpenCUDAWith is OpenCUDA with this device's open choices.
//
// An enumerating open of an ordinal below zero is the default pick: device 0, or the first device an
// installed Restriction allows.
func OpenCUDAWith(ord int, opts Opts) (Device, error) {
	if ord >= 0 || opts.Enumerating {
		var err error
		if ord, err = admit("ptx", ord, opts.Enumerating); err != nil {
			return nil, err
		}
	}
	if ord < 0 && opts.Enumerating {
		ord = 0
	}
	reqs := make(chan func(), 1)
	type opened struct {
		d   *cuda.Device
		tid int64
		err error
	}
	out := make(chan opened, 1)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		d, err := cuda.OpenDevice(ord)
		out <- opened{d, threadID(), err}
		if err != nil {
			return
		}
		for {
			f, ok := spinRecv(reqs, ownerSpin)
			if !ok {
				return
			}
			f()
		}
	}()
	o := <-out
	if o.err != nil {
		return nil, o.err
	}
	ownedDevices.Add(1)
	return &cudaDev{d: o.d, reqs: reqs, owner: o.tid, posted: opts.CUDAPosted, maxRegs: opts.CUDAMaxRegs}, nil
}

// Session batches a sequence of device calls into ONE ownership hop.
//
// The request do runs and the session handed to f come off a free list rather
// than a closure and a fresh cudaSession: both escape (do may post to the
// owner, f is the caller's), so building them per call was two heap objects
// on every decode token. Nested sessions take a request each.
func (c *cudaDev) Session(f func(Session)) {
	// Inside a Session on a queue, a nested one is that queue's: on the legacy
	// stream it would not be ordered against the queue's work.
	if q := c.curQueue(); q != nil {
		ownerCalls.Add(1)
		f(&q.s)
		return
	}
	c.reqMu.Lock()
	var r *sessReq
	if n := len(c.reqFree); n > 0 {
		r, c.reqFree = c.reqFree[n-1], c.reqFree[:n-1]
	}
	c.reqMu.Unlock()
	if r == nil {
		r = &sessReq{c: c}
		r.run = r.call
	}
	r.f = f
	c.do(r.run)
	r.f = nil
	c.reqMu.Lock()
	c.reqFree = append(c.reqFree, r)
	c.reqMu.Unlock()
}

// sessReq is one Session call: the caller's body, the session it is handed,
// and the function do runs, built once per request.
type sessReq struct {
	c   *cudaDev
	f   func(Session)
	s   cudaSession
	run func()
}

func (r *sessReq) call() {
	// The legacy stream and a queue's non-blocking stream are not ordered
	// against each other: with queues open, the session's work has landed
	// when it returns, so a queue's later work reads what it wrote. Only the
	// legacy stream is waited for, not the queues, which run on. A failed
	// synchronise is a sticky context error the next call reports.
	r.s = cudaSession{c: r.c}
	r.f(&r.s)
	if r.c.hasQueues() {
		cuda.SyncLegacy()
	}
}

// cudaSession's methods run on the owner goroutine already, so they call
// straight through. It is a distinct type rather than a flag on cudaDev because
// a flag would have to be read by a goroutine that is not the owner, and a
// misread would run a driver call on the wrong thread -- the exact bug this
// whole mechanism exists to prevent.
//
// It is a POINTER so that BeginRecord can redirect launches onto the capture
// stream for the length of a recording. That state lives here rather than on
// cudaDev for the reason above: the value is made, mutated and read inside one
// Session that is already running on the owner.
type cudaSession struct {
	c *cudaDev
	// q is the queue this session runs on, nil on the legacy stream. Every
	// copy, launch, capture and wait of a queued session is on q's stream.
	q *cudaQueue
	// stream is where launches go: zero is the legacy stream, non-zero only
	// while a capture is open.
	stream cuda.CUstream
	// qcap is set while this queued session holds c.capturing for a capture.
	qcap bool
	// marks collects a timed capture's per-launch events; nil otherwise.
	// It points at recMarks, which holds them between BeginRecord and
	// EndRecord.
	marks    *[]kernMark
	recMarks []kernMark
	// trial says this session's launches are a tuner's trial (MarkTrial):
	// the timing instrument names them under TrialPrefix.
	trial bool
}

// MarkTrial implements TrialMarker.
func (s *cudaSession) MarkTrial() { s.trial = true }

func (s *cudaSession) Write(b Buf, p []byte) error { return s.WriteAt(b, 0, p) }

func (s *cudaSession) WriteAt(b Buf, off int, p []byte) error {
	if len(p) == 0 {
		return nil
	}
	if s.q != nil {
		return s.q.write(b.(*cudaBuf).b, off, p)
	}
	return b.(*cudaBuf).b.WriteAt(off, unsafe.Pointer(&p[0]), len(p))
}

func (s *cudaSession) Read(b Buf, p []byte) error {
	if len(p) == 0 {
		return nil
	}
	countRead(len(p))
	if s.q != nil {
		return s.q.read(b.(*cudaBuf).b, p)
	}
	return b.(*cudaBuf).b.Read(unsafe.Pointer(&p[0]), len(p))
}

// Launch fills the module's own argument array in place, so a launch
// allocates nothing. The scratch is per-module state, safe because every call
// here is on one thread; see cuda.Module.args.
func (s *cudaSession) Launch(k Kernel, groups, width int, bufs ...Buf) error {
	ck := k.(*cudaKern)
	if err := checkWidth("ptx", ck.name, ck.group, width); err != nil {
		return err
	}
	if err := checkGrid("ptx", ck.name, ck.grid, groups, ck.resident, ck.residentErr); err != nil {
		return err
	}
	m := ck.m
	args := m.Args(len(bufs))
	if s.q != nil {
		// The module's argument array is shared by every caller of the kernel,
		// and two queues launch it at once: a queued session fills its own.
		args = s.q.args[:min(len(bufs), len(s.q.args))]
		if len(bufs) > len(s.q.args) {
			args = nil
		}
	}
	if args == nil {
		return fmt.Errorf("backend: cuda: %d kernel arguments is too many", len(bufs))
	}
	for i, b := range bufs {
		// A nil buffer is an error, not a panic: every caller can fall back to
		// the host, and a panic here would take a server down.
		cb, ok := b.(*cudaBuf)
		if !ok || cb == nil {
			return fmt.Errorf("backend: cuda: %s: kernel argument %d of %d is nil", ck.name, i, len(bufs))
		}
		args[i] = cb.b.Arg()
	}
	// No Sync: see the Session interface. cuMemcpyDtoH on the legacy stream
	// waits for prior work on it and on every blocking stream, so the readback
	// orders this whether it ran here or inside a replayed graph.
	st := s.stream
	if st == 0 && s.q != nil {
		st = s.q.st.Handle()
	}
	// Outside a capture the timing instrument is an event pair around the
	// launch itself, read back by CUDAKernelTimes.
	var t0 *cuda.Event
	if s.marks == nil && !s.qcap && s.stream == 0 && cudaKernTiming.Load() && cuda.EventsAvailable() {
		if e, err := cuda.NewEvent(); err == nil && e.RecordOn(st) == nil {
			t0 = e
		}
	}
	if err := m.LaunchArgs(st, groups, width, args); err != nil {
		return err
	}
	ck.noteRan()
	name := ck.name
	if s.trial && (t0 != nil || s.marks != nil) {
		name = TrialPrefix + name
	}
	if t0 != nil {
		if e, err := cuda.NewEvent(); err == nil && e.RecordOn(st) == nil {
			directMarks.add(kernPair{t0, e, name})
		}
	}
	if s.marks != nil {
		// Inside a timed capture: an event NODE after the launch, so a replay
		// stamps each kernel's end with no host in the loop.
		if e, err := cuda.NewEvent(); err == nil && e.RecordNode(s.stream) == nil {
			*s.marks = append(*s.marks, kernMark{e, name})
		}
	}
	return nil
}

// cudaKernTiming arms an event pair around every launch outside a capture;
// see SetCUDAKernelTiming.
var cudaKernTiming atomic.Bool

// SetCUDAKernelTiming arms per-kernel device-clock timing on CUDA: a launch
// outside a capture runs between two events, read when CUDAKernelTimes is
// called after the work is done; a capture records an event node after every
// launch, and each replay's kernels are read back one replay later -- the SAME measurement scripts/kshim.c takes of
// llama.cpp's graphs, so the two engines' kernels can be set side by side. It
// applies to graphs recorded after it is armed. A measurement instrument,
// default off: the event nodes cost device time of their own.
func SetCUDAKernelTiming(on bool) { cudaKernTiming.Store(on) }

// kernMark is one captured launch's end event: kernel i's device time is its
// event minus kernel i-1's (the capture's first mark is a bare start event).
type kernMark struct {
	ev   *cuda.Event
	name string
}

// kernPair is one launch outside a capture, between two events.
type kernPair struct {
	start, end *cuda.Event
	name       string
}

// directMarks holds the launches timed outside a capture until
// CUDAKernelTimes reads them.
var directMarks kernPairs

type kernPairs struct {
	sync.Mutex
	p []kernPair
}

func (k *kernPairs) add(p kernPair) {
	k.Lock()
	k.p = append(k.p, p)
	k.Unlock()
}

// settle adds every completed pair to kernTimes and keeps the rest.
func (k *kernPairs) settle() {
	k.Lock()
	defer k.Unlock()
	keep := k.p[:0]
	for _, p := range k.p {
		if !p.end.Done() {
			keep = append(keep, p)
			continue
		}
		if ms, err := cuda.Elapsed(p.start, p.end); err == nil {
			kernTimes.add(p.name, ms*1e3)
		}
		p.start.Destroy()
		p.end.Destroy()
	}
	k.p = keep
}

// settleMarks adds a completed replay's per-kernel times to kernTimes.
func settleMarks(m []kernMark) {
	for i := 1; i < len(m); i++ {
		if ms, err := cuda.Elapsed(m[i-1].ev, m[i].ev); err == nil {
			kernTimes.add(m[i].name, ms*1e3)
		}
	}
}

// kernTimes sums device microseconds and counts per kernel name.
var kernTimes kernTotals

type kernTotals struct {
	sync.Mutex
	us map[string]float64
	n  map[string]int
}

func (t *kernTotals) add(name string, us float32) {
	t.Lock()
	defer t.Unlock()
	if t.us == nil {
		t.us, t.n = map[string]float64{}, map[string]int{}
	}
	t.us[name] += float64(us)
	t.n[name]++
}

// CUDAKernelTimes returns, and clears, the per-kernel device microseconds and
// launch counts SetCUDAKernelTiming collected.
func CUDAKernelTimes() (us map[string]float64, n map[string]int) {
	directMarks.settle()
	kernTimes.Lock()
	defer kernTimes.Unlock()
	us, n = kernTimes.us, kernTimes.n
	kernTimes.us, kernTimes.n = nil, nil
	return us, n
}

func (s *cudaSession) Sync() error {
	if s.q != nil {
		return s.q.sync()
	}
	return s.c.syncAll()
}

// BeginRecord starts capturing this session's launches into a graph instead
// of running them. It redirects the launches rather than intercepting them, so
// a caller records with the same code it would otherwise run and the two
// cannot drift apart (see Record).
func (s *cudaSession) BeginRecord() error {
	if !cuda.GraphsAvailable() {
		return fmt.Errorf("backend: cuda: this driver has no stream capture")
	}
	if s.q != nil {
		// A queue captures on its own stream: it is already not the legacy
		// stream, and its copies are on it too.
		s.c.capturing.RLock()
		if err := s.q.st.BeginCapture(); err != nil {
			s.c.capturing.RUnlock()
			return err
		}
		s.qcap = true
		s.stream = s.q.st.Handle()
		s.armMarks()
		return nil
	}
	if s.c.cap == nil {
		if s.c.noCap {
			return fmt.Errorf("backend: cuda: no capture stream")
		}
		st, err := s.c.d.NewStream()
		if err != nil {
			s.c.noCap = true
			return err
		}
		s.c.cap = st
	}
	if err := s.c.cap.BeginCapture(); err != nil {
		return err
	}
	s.stream = s.c.cap.Handle()
	s.armMarks()
	return nil
}

// armMarks starts a capture's kernel-timing event nodes when
// SetCUDAKernelTiming is on, on whichever stream the capture runs.
func (s *cudaSession) armMarks() {
	s.recMarks = nil
	if cudaKernTiming.Load() && cuda.EventsAvailable() {
		if e, err := cuda.NewEvent(); err == nil && e.RecordNode(s.stream) == nil {
			s.recMarks = []kernMark{{e, ""}}
			s.marks = &s.recMarks
		}
	}
}

// EndRecord closes the capture BeginRecord opened and returns its graph.
func (s *cudaSession) EndRecord() (Recording, error) {
	marks := s.recMarks
	s.stream, s.marks, s.recMarks = 0, nil, nil
	capSt := s.c.cap
	if s.q != nil {
		capSt = s.q.st
	}
	// EndCapture runs even when the launches failed: a stream left capturing
	// refuses every later launch on it, and its error is the informative one.
	g, err := capSt.EndCapture()
	if s.qcap {
		s.qcap = false
		s.c.capturing.RUnlock()
	}
	if err != nil {
		return nil, err
	}
	return &cudaGraph{g: g, dev: s.c, marks: marks}, nil
}

func (s *cudaSession) Replay(r Recording) error {
	cg := r.(*cudaGraph)
	if len(cg.marks) > 0 {
		// The previous timed replay is done: decode read its token back. A
		// session on a queue is no different here; settling only on the
		// legacy stream left both timing instruments silent for every decode
		// that runs on a queue, which is every decode of a session.
		settleMarks(s.c.lastMarks)
		s.c.lastMarks = cg.marks
	}
	st := s.c.cap
	if s.q != nil {
		st = s.q.st
	}
	if !cudaTiming.Load() || !cuda.EventsAvailable() {
		return cg.g.Launch(st)
	}
	return s.c.timeReplay(cg.g, st)
}

// timeReplay is Replay between two events on the stream it runs on. Replay i's
// pair is read at replay i+1: its graph is done by then, because decode reads
// the token back between them (a legacy-stream copy waits for this stream),
// and so is replay i-1's after-event, which makes the gap between two graphs
// -- the device idle while the host takes the token and issues the next --
// a device-clock number too.
func (c *cudaDev) timeReplay(g *cuda.Graph, st *cuda.Stream) error {
	cur := &c.tev[c.ti%3]
	for i := range cur {
		if cur[i] == nil {
			e, err := cuda.NewEvent()
			if err != nil {
				return g.Launch(st)
			}
			cur[i] = e
		}
	}
	if c.ti >= 1 {
		prev := c.tev[(c.ti-1)%3]
		busy, err := cuda.Elapsed(prev[0], prev[1])
		gap := float32(-1)
		if err == nil && c.ti >= 2 {
			if g2, e2 := cuda.Elapsed(c.tev[(c.ti-2)%3][1], prev[0]); e2 == nil {
				gap = g2
			}
		}
		if err == nil {
			graphTimes.add(busy, gap)
		}
	}
	if err := cur[0].Record(st); err != nil {
		return err
	}
	if err := g.Launch(st); err != nil {
		return err
	}
	c.ti++
	return cur[1].Record(st)
}

// graphTimes holds the last replays' device times, in microseconds.
var graphTimes replayTimes

type replayTimes struct {
	sync.Mutex
	busy, gap []float32
}

func (t *replayTimes) add(busyMs, gapMs float32) {
	t.Lock()
	defer t.Unlock()
	if len(t.busy) >= 1<<16 {
		t.busy, t.gap = t.busy[:0], t.gap[:0]
	}
	t.busy = append(t.busy, busyMs*1e3)
	t.gap = append(t.gap, gapMs*1e3)
}

// CUDAGraphTimes returns, and clears, the device-clock time of every timed
// replay and the idle gap before it (-1 where there was no previous one), in
// microseconds. See SetCUDATiming.
func CUDAGraphTimes() (busy, gap []float32) {
	graphTimes.Lock()
	defer graphTimes.Unlock()
	busy, gap = graphTimes.busy, graphTimes.gap
	graphTimes.busy, graphTimes.gap = nil, nil
	return busy, gap
}

// cudaGraph is a Recording. Free hops to the owner goroutine the way cudaBuf's
// does, because destroying a graph is a driver call like any other while the
// caller releasing one is the tier, on whatever goroutine decode runs on.
type cudaGraph struct {
	g     *cuda.Graph
	dev   *cudaDev
	marks []kernMark // SetCUDAKernelTiming's event nodes, if it was armed
}

func (g *cudaGraph) Free() {
	g.dev.do(func() {
		g.g.Destroy()
		if len(g.marks) > 0 && len(g.dev.lastMarks) > 0 && &g.dev.lastMarks[0] == &g.marks[0] {
			settleMarks(g.dev.lastMarks) // freed while pending: its last replay is done by now
			g.dev.lastMarks = nil
		}
		for _, m := range g.marks {
			m.ev.Destroy()
		}
	})
}

// errCUDAClosed is what a call that returns something answers on a closed
// device. Each such call starts its error at this and lets f overwrite it, so
// the dropped call below is a refusal rather than a success with nothing done
// -- an Alloc that answered a buffer with no memory behind it, or a Read that
// left the caller's bytes as they were.
var errCUDAClosed = errors.New("backend: cuda: the device is closed")

// do runs f on the device-owning goroutine and waits for it.
//
// A closed device is a no-op, not a hang: Close nils c.reqs, and a send on a nil
// channel blocks forever. Dropping the work is correct for a release, because
// cuCtxDestroy already reclaimed every allocation, stream and module of the
// context; a call that answers something reports errCUDAClosed instead.
//
// A call from the owner's own thread (that is, from inside a Session) runs
// inline, because posting it would wait on itself forever. Such calls are
// counted (OwnerCalls) so gates can hold the tier to issuing none.
//
// A panic inside a posted request is re-raised on the caller, so it names the
// caller's stack and does not kill the owner.
func (c *cudaDev) do(f func()) {
	tid := threadID()
	if tid >= 0 && (tid == c.owner || tid == c.in.Load() || c.inQueue(tid)) {
		ownerCalls.Add(1)
		f()
		return
	}
	if !c.posted && tid >= 0 && c.inline(f) {
		return
	}
	c.mu.Lock()
	reqs := c.reqs
	c.mu.Unlock()
	if reqs == nil {
		return
	}
	postedCalls.Add(1)
	done := make(chan struct{})
	var pv any
	// The send is on the SNAPSHOT, not c.reqs: Close cannot nil a local, and it
	// cannot close the channel either while its own lock below is held.
	sent := func() (ok bool) {
		defer func() {
			// A Close that won the race closed the channel between the snapshot
			// and the send. The work is moot (cuCtxDestroy reclaimed it), and a
			// panic out of a Free would kill the process.
			if recover() != nil {
				ok = false
			}
		}()
		reqs <- func() {
			defer func() { pv = recover(); close(done) }()
			f()
		}
		return true
	}()
	if !sent {
		return
	}
	spinWait(done, callerSpin)
	if pv != nil {
		panic(pv)
	}
}

// inline runs f on the calling thread with the device's context made current
// there, under run, and reports false only when the device is closed.
//
// Posting to the owner costs two goroutine wakeups per Session, a measurable
// share of a device decode token. A context can be current on any number of
// threads; what the owner provided was serialisation, and run gives that without
// moving the work. The thread is locked for the call so the context stays
// current.
func (c *cudaDev) inline(f func()) bool {
	c.run.Lock()
	defer c.run.Unlock()
	c.mu.Lock()
	open := c.reqs != nil
	c.mu.Unlock()
	if !open {
		return false
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	tid := threadID()
	c.in.Store(tid)
	defer c.in.Store(-1)
	if err := c.d.Bind(); err != nil {
		return false
	}
	f()
	return true
}

// allocRefused counts allocations a device refused for want of memory.
var allocRefused atomic.Int64

// AllocRefused is how many device allocations were refused because the card
// was full -- by this process or, on a shared card, by another one. Placement
// declines around it; a gate that needs a model on the card reads it to tell
// "the card is too small here" from a failure.
func AllocRefused() int64 { return allocRefused.Load() }

// postedCalls counts device calls that took the hop to the owner goroutine.
var postedCalls atomic.Int64

// PostedCalls is how many CUDA calls were posted to an owner goroutine rather
// than run inline; a gate holds it still across a Session.
func PostedCalls() int64 { return postedCalls.Load() }

// ownerCalls counts device calls made from inside a Session, which do runs
// inline rather than deadlocking on; see OwnerCalls.
var ownerCalls atomic.Int64

// OwnerCalls is how many device calls were made from inside a Session. The
// backend completes them, so this is a hygiene count, not an error count.
func OwnerCalls() int64 { return ownerCalls.Load() }

// Both ends of the ownership hop poll before they park, because waking a
// parked goroutine costs tens of microseconds with the card idle meanwhile
// (CUDA's own synchronise spins for the same reason). The owner polls for its
// next request for ownerSpin, and a caller polls for its result for
// callerSpin, which is longer than a decode token.
//
// Each poll yields: the owner still needs a P, and a caller spinning without
// yielding starves it until preemption.
//
// The windows are fixed and cost one host core while a device call runs; make
// them options if a host needs that core back during device decode.
const (
	ownerSpin  = 2 * time.Millisecond
	callerSpin = 50 * time.Millisecond
)

func spinRecv(c chan func(), d time.Duration) (func(), bool) {
	t0 := time.Now()
	for time.Since(t0) < d {
		select {
		case f, ok := <-c:
			return f, ok
		default:
			runtime.Gosched()
		}
	}
	f, ok := <-c
	return f, ok
}

func spinWait(done chan struct{}, d time.Duration) {
	t0 := time.Now()
	for time.Since(t0) < d {
		select {
		case <-done:
			return
		default:
			runtime.Gosched()
		}
	}
	<-done
}

// Name is ordinal, card and PTX target together -- "#0 <card name> (sm_86)".
//
// Target alone is a capability, not an identity: tier.deviceKey hashes
// API()+"/"+Name() for its probe cache, and two cards share a target. The
// ordinal separates two identical cards, which share a name too.
func (c *cudaDev) Name() string { return c.d.Label() }
func (c *cudaDev) API() string  { return "ptx" }

// Ordinal is which CUDA device this is, for a caller placing work across
// several. It is an OPTIONAL interface for the same reason Mem is: the other
// two backends enumerate differently, and a caller that does not need it must
// not have to know.
func (c *cudaDev) Ordinal() int { return c.d.Ordinal() }

// Slots comes from the driver: SM count times max threads per SM.
func (c *cudaDev) Slots() int { return c.d.Slots() }

// Mem is free and total device memory. It is an optional interface rather than
// a Device method so a backend that cannot say need not invent a number.
// CUDA's figure is machine-wide; Vulkan's (VK_EXT_memory_budget) accounts for
// other processes; Metal's counts only this process and is a slice of host RAM.
//
// cuMemGetInfo answers for the current context, so cuda.Device.Mem binds this
// device's context before asking.
func (c *cudaDev) Mem() (free, total uint64, err error) {
	err = errCUDAClosed
	c.do(func() { free, total, err = c.d.Mem() })
	return
}

// Close destroys the context on the owner goroutine and only then releases it,
// so the OS thread that held the context outlives the context; see
// cuda.Device.Close.
func (c *cudaDev) Close() {
	// No inline call and no Session on a queue may be running on the context
	// while it is destroyed.
	c.run.Lock()
	defer c.run.Unlock()
	c.life.Lock()
	defer c.life.Unlock()
	c.mu.Lock()
	if c.reqs == nil {
		c.mu.Unlock()
		return
	}
	c.mu.Unlock()
	done := make(chan struct{})
	c.reqs <- func() {
		// The streams go before the context that owns them.
		c.cap.Destroy()
		c.up.Destroy()
		c.qmu.Lock()
		for q := range c.queues {
			q.st.Destroy()
			if q.stage != nil {
				cuda.FreeHost(q.stage)
				q.stage = nil
			}
		}
		c.queues = nil
		c.qmu.Unlock()
		c.d.Close()
		close(done)
	}
	<-done
	ownedDevices.Add(-1)
	c.mu.Lock()
	close(c.reqs)
	c.reqs = nil
	c.mu.Unlock()
}

func (c *cudaDev) Alloc(n int) (Buf, error) {
	var b *cuda.Buffer
	err := errCUDAClosed
	c.do(func() { b, err = c.d.Alloc(n) })
	if err != nil {
		if errors.Is(err, cuda.ErrOutOfMemory) {
			allocRefused.Add(1)
		}
		return nil, err
	}
	return &cudaBuf{b: b, dev: c, acc: c.take(n, cudaFootprint(n))}, nil
}

// PinHost is cuda.HostAlloc on the owner goroutine (backend.HostPinner).
func (c *cudaDev) PinHost(n int) ([]byte, error) {
	var b []byte
	err := errCUDAClosed
	c.do(func() { b, err = cuda.HostAlloc(n) })
	return b, err
}

// ImportAlign and Import are cuda.Device's host-pointer import
// (backend.HostImport) on the owner goroutine. The buffer is not counted: the
// card did not allocate it.
func (c *cudaDev) ImportAlign() int { return c.d.ImportAlign() }

func (c *cudaDev) Import(p unsafe.Pointer, n int) (Buf, error) {
	var b *cuda.Buffer
	err := errCUDAClosed
	c.do(func() { b, err = c.d.Import(p, n) })
	if err != nil {
		return nil, err
	}
	return &cudaBuf{b: b, dev: c}, nil
}

// UnifiedMemory is CU_DEVICE_ATTRIBUTE_INTEGRATED: a Jetson's memory is the
// host's, a discrete card's is not (backend.Unified).
func (c *cudaDev) UnifiedMemory() bool { return c.d.Integrated() }

var (
	_ HostImport = (*cudaDev)(nil)
	_ Unified    = (*cudaDev)(nil)

	_ ResourceReporter = (*cudaDev)(nil)
)

// KernelResources is every kernel this card compiled, as the driver reports
// ptxas's allocation of it.
func (c *cudaDev) KernelResources() []KernelResources { return c.res.list() }

// UnpinHost gives PinHost's memory back.
func (c *cudaDev) UnpinHost(b []byte) { c.do(func() { cuda.FreeHost(b) }) }

// GuaranteedLanes: a CUDA warp is 32 threads on every architecture NVIDIA has
// shipped, WARP_SIZE is 32 in every CUDA toolkit, and ptx.Lower emits
// shfl.sync with a full member mask and maxLane 31 -- one segment, the whole
// warp. So 32 is guaranteed by the target and nothing else is offered: a width
// this backend cannot produce is refused rather than approximated.
//
// It is not read from the device (CU_DEVICE_ATTRIBUTE_WARP_SIZE): what matters
// is the width the lowerer assumed, a compile-time 32. A device reporting
// anything else would need a different lowering, not a different check.
func (c *cudaDev) GuaranteedLanes(w int) (bool, string) {
	if w == ir.SubgroupLanes {
		return true, ""
	}
	return false, fmt.Sprintf("a CUDA warp is %d lanes and ptx emits shfl.sync over exactly "+
		"those; %d cannot be asked for", ir.SubgroupLanes, w)
}

func (c *cudaDev) Compile(k *ir.Kernel) (Kernel, error) {
	compiles.Add(1)
	if CompileLog != nil {
		CompileLog(k.Name)
	}
	src, err := ptx.Lower(k, c.d.Target)
	if err != nil {
		return nil, err
	}
	if c.maxRegs > 0 {
		src = strings.Replace(src, ")\n{\n", fmt.Sprintf(")\n.maxnreg %d\n{\n", c.maxRegs), 1)
	}
	var m *cuda.Module
	var a cuda.Attrs
	var attrsOK bool
	err = errCUDAClosed
	ck := &cudaKern{dev: c, name: k.Name, group: k.Group, src: k}
	c.do(func() {
		if m, err = c.d.Load(src, k.Name); err != nil {
			return
		}
		a, attrsOK = m.Attrs(k.Group[0])
		if !k.GridSync() {
			return
		}
		// A grid barrier runs cooperatively, on a grid the occupancy
		// calculator says is resident at once (cuda/coop.go).
		m.Coop = true
		ck.resident, ck.residentErr = m.Resident(k.Group[0])
	})
	if err != nil {
		return nil, err
	}
	if attrsOK && a.MaxThreads > 0 && a.MaxThreads < k.Group[0] {
		// The driver refuses the launch (too many resources requested); a
		// refusal here lets the caller keep another kernel instead.
		c.do(m.Unload)
		return nil, fmt.Errorf("%w: %s at %d registers a thread launches at most %d threads, not %d",
			ErrGroupTooWide, k.Name, a.Regs, a.MaxThreads, k.Group[0])
	}
	ck.m = m
	if attrsOK {
		ck.res = KernelResources{Name: k.Name, API: "cuda", Group: k.Group[0], Regs: a.Regs,
			SpillBytes: a.Local, SharedBytes: a.Shared, MaxGroup: a.MaxThreads,
			Resident: a.Blocks, ResidentThreads: a.Blocks * k.Group[0], MaxThreads: c.d.MaxThreadsPerSM()}
		ck.resOK = true
		c.res.note(ck.res)
	}
	return ck, nil
}

// GridLimit is the most workgroups a launch of k may carry when k has a grid
// barrier: the occupancy calculator's resident blocks (cuda.Module.Resident).
func (c *cudaDev) GridLimit(k Kernel) (int, error) {
	ck := k.(*cudaKern)
	if !ck.src.GridSync() {
		return 0, fmt.Errorf("backend: cuda: %s has no grid barrier", ck.name)
	}
	return ck.resident, ck.residentErr
}

type cudaBuf struct {
	b   *cuda.Buffer
	dev *cudaDev
	acc *counted
}

// Write and Read are real device copies here, unlike the mapped backends. That
// asymmetry is why Buf is not a []byte.
func (b *cudaBuf) Write(p []byte) error {
	if len(p) == 0 {
		return nil
	}
	return b.WriteAt(0, p)
}

func (b *cudaBuf) WriteAt(off int, p []byte) error {
	if len(p) == 0 {
		return nil
	}
	if q := b.dev.curQueue(); q != nil {
		return q.write(b.b, off, p)
	}
	r := cudaWrites.Get().(*cudaWrite)
	*r = cudaWrite{b: b, off: off, p: p, err: errCUDAClosed, fn: r.fn}
	b.dev.do(r.fn)
	err := r.err
	*r = cudaWrite{fn: r.fn}
	cudaWrites.Put(r)
	return err
}

// cudaWrite is one WriteAt outside a queue, with the function the owner
// goroutine runs for it, kept as cudaCopy is: a closure over WriteAt's locals
// was a heap object per write.
type cudaWrite struct {
	b   *cudaBuf
	off int
	p   []byte
	err error
	fn  func()
}

var cudaWrites = sync.Pool{New: func() any {
	r := &cudaWrite{}
	r.fn = r.run
	return r
}}

func (r *cudaWrite) run() {
	d := r.b.dev
	// Outside a queue the legacy stream does not wait for one, and a write
	// must land after everything before it.
	queues := d.hasQueues()
	if queues {
		if r.err = d.syncAll(); r.err != nil {
			return
		}
	}
	r.err = r.b.b.WriteAt(r.off, unsafe.Pointer(&r.p[0]), len(r.p))
	// From pageable memory the copy returns once the driver has staged the
	// bytes, not once they have landed: a queue's next launch, on a stream
	// that does not wait for the legacy one, could read the old ones.
	if r.err == nil && queues {
		r.err = cuda.SyncLegacy()
	}
}

func (b *cudaBuf) Read(p []byte) error {
	if len(p) == 0 {
		return nil
	}
	countRead(len(p))
	if q := b.dev.curQueue(); q != nil {
		return q.read(b.b, p)
	}
	err := errCUDAClosed
	b.dev.do(func() {
		if b.dev.hasQueues() {
			if err = b.dev.syncAll(); err != nil {
				return
			}
		}
		err = b.b.Read(unsafe.Pointer(&p[0]), len(p))
	})
	return err
}

func (b *cudaBuf) Free() {
	b.dev.do(func() { b.b.Free() })
	b.acc.give()
}

// Copy is cuMemcpyDtoD on the legacy stream, which waits for every launch
// before it on that stream and on every blocking stream, followed by a
// context synchronise so the bytes have landed when it returns.
func (c *cudaDev) Copy(dst Buf, dstOff int, src Buf, srcOff, n int) error {
	db, ok := dst.(*cudaBuf)
	sb, ok2 := src.(*cudaBuf)
	if !ok || !ok2 || db == nil || sb == nil || db.dev != c || sb.dev != c {
		return fmt.Errorf("backend: cuda: a copy between buffers this device did not allocate")
	}
	if err := copyRange(db.b.Len(), dstOff, sb.b.Len(), srcOff, n, db.b == sb.b); err != nil || n == 0 {
		return err
	}
	r := cudaCopies.Get().(*cudaCopy)
	*r = cudaCopy{c: c, db: db, sb: sb, dOff: dstOff, sOff: srcOff, n: n, err: errCUDAClosed, fn: r.fn}
	c.do(r.fn)
	err := r.err
	*r = cudaCopy{fn: r.fn}
	cudaCopies.Put(r)
	return err
}

// CopyList is Copy for every range of cs, under one ownership hop and one
// wait: the context is synchronised once before the list (as Copy does with
// queues open) and once after it, not around each copy.
func (c *cudaDev) CopyList(cs []CopyRange) error {
	for _, r := range cs {
		db, ok := r.Dst.(*cudaBuf)
		sb, ok2 := r.Src.(*cudaBuf)
		if !ok || !ok2 || db == nil || sb == nil || db.dev != c || sb.dev != c {
			return fmt.Errorf("backend: cuda: a copy between buffers this device did not allocate")
		}
		if err := copyRange(db.b.Len(), r.DstOff, sb.b.Len(), r.SrcOff, r.N, db.b == sb.b); err != nil {
			return err
		}
	}
	l := cudaCopyLists.Get().(*cudaCopyList)
	l.c, l.cs, l.err = c, cs, errCUDAClosed
	c.do(l.fn)
	err := l.err
	l.c, l.cs, l.err = nil, nil, nil
	cudaCopyLists.Put(l)
	return err
}

// cudaCopyList is CopyList's request, kept like cudaCopy's so a list makes no
// heap object per call.
type cudaCopyList struct {
	c   *cudaDev
	cs  []CopyRange
	err error
	fn  func()
}

var cudaCopyLists = sync.Pool{New: func() any {
	l := &cudaCopyList{}
	l.fn = l.run
	return l
}}

func (l *cudaCopyList) run() {
	c := l.c
	if c.hasQueues() {
		if l.err = c.syncAll(); l.err != nil {
			return
		}
	}
	for _, r := range l.cs {
		if r.N == 0 {
			continue
		}
		if l.err = cuda.CopyDtoD(r.Dst.(*cudaBuf).b, r.DstOff, r.Src.(*cudaBuf).b, r.SrcOff, r.N); l.err != nil {
			return
		}
	}
	l.err = c.syncAll()
}

// cudaCopy is one Copy's arguments with the function the owner goroutine runs
// for it, made once a request and the requests kept (cudaCopies): a closure
// over Copy's locals was a heap object per copy, and a recurrent block's
// zeroing makes four a block every prompt.
type cudaCopy struct {
	c             *cudaDev
	db, sb        *cudaBuf
	dOff, sOff, n int
	err           error
	fn            func()
}

var cudaCopies = sync.Pool{New: func() any {
	r := &cudaCopy{}
	r.fn = r.run
	return r
}}

func (r *cudaCopy) run() {
	c := r.c
	// The legacy stream does not wait for a queue's non-blocking stream, so
	// with queues open the copy waits for the whole context first.
	if c.hasQueues() {
		if r.err = c.syncAll(); r.err != nil {
			return
		}
	}
	if r.err = cuda.CopyDtoD(r.db.b, r.dOff, r.sb.b, r.sOff, r.n); r.err == nil {
		r.err = c.syncAll()
	}
}

type cudaKern struct {
	m    *cuda.Module
	dev  *cudaDev
	name string // ir.Kernel.Name, for SetCUDAKernelTiming and checkWidth
	// group is the declared workgroup. CUDA runs the launch's width, so a
	// mismatch computes here and not on Vulkan; checkWidth refuses it on both.
	group [3]int
	// src is the kernel's IR (IRSource); resident and residentErr its
	// co-residency limit when it has a grid barrier (GridLimit).
	src         *ir.Kernel
	resident    int
	grid        bool // src has a grid barrier: decided once at compile, read by every launch
	residentErr error
	// res is what the driver reported for this compile (resOK false where it
	// would not say); see KernelResourcesOf.
	res   KernelResources
	resOK bool
	// ran is set at the first launch, which marks the kernel in the device's
	// ledger as one that ran rather than one compiled and passed over.
	ran atomic.Bool
}

func (k *cudaKern) resources() (KernelResources, bool) { return k.res, k.resOK }

// noteRan is one atomic load a launch; the first launch records it.
func (k *cudaKern) noteRan() {
	if !k.ran.Load() && k.ran.CompareAndSwap(false, true) {
		k.dev.res.ran(k.name)
	}
}

func (k *cudaKern) IR() *ir.Kernel { return k.src }

// Launch synchronises before returning. cuLaunchKernel is asynchronous and the
// Vulkan and Metal arms block, so without this the interface would mean two
// things per backend, and timing a launch would measure only the enqueue.
func (k *cudaKern) Launch(groups, width int, bufs ...Buf) error {
	if err := checkWidth("ptx", k.name, k.group, width); err != nil {
		return err
	}
	if err := checkGrid("ptx", k.name, k.grid, groups, k.resident, k.residentErr); err != nil {
		return err
	}
	cb := make([]*cuda.Buffer, len(bufs))
	for i, b := range bufs {
		cb[i] = b.(*cudaBuf).b
	}
	err := errCUDAClosed
	k.dev.do(func() {
		if err = k.m.Launch(groups, width, cb...); err == nil {
			k.noteRan()
			err = k.dev.syncAll()
		}
	})
	return err
}

func (k *cudaKern) Close() { k.dev.do(func() { k.m.Unload() }) }

// cudaQueue is one non-blocking stream, with the argument array its sessions
// launch from (cuda.Module's own is shared by every caller of the kernel)
// and the session SessionOn hands out, reused so a call allocates nothing.
type cudaQueue struct {
	once
	c    *cudaDev
	st   *cuda.Stream
	mu   sync.Mutex
	args [64]unsafe.Pointer
	s    cudaSession
	// stage is page-locked host memory every copy of this queue goes through,
	// used from the front and reset once the stream has drained. A copy from
	// pageable memory is staged by the driver synchronously, which stalls the
	// other queues' work as well as this one's: two sessions' kernels then run
	// one after another whatever streams they are on.
	stage []byte
	used  int
}

// stageMin is a queue's first staging arena: a decode token's writes (the
// position, the descriptors, the embedding row) and its readback fit many
// times over.
const stageMin = 1 << 20

// stageWrites is how many writes of the largest size seen the arena holds
// before it drains.
const stageWrites = 8

// room returns n bytes of the staging arena, draining the stream first when
// they are not free and growing the arena when n is more than all of it.
func (q *cudaQueue) room(n int) ([]byte, error) {
	if q.used+n > len(q.stage) {
		if err := q.st.Sync(); err != nil {
			return nil, err
		}
		q.used = 0
	}
	if n > len(q.stage) {
		if q.stage != nil {
			cuda.FreeHost(q.stage)
			q.stage = nil
		}
		// Room for stageWrites writes of this size: an arena grown to exactly
		// n is full after one, and every write after it -- the next sheet, or a
		// decode token's few words -- drained the stream first. A hybrid
		// expert cache sending 2.9 MB sheets waited on the card at each.
		b, err := cuda.HostAlloc(max(n*stageWrites, stageMin))
		if err != nil {
			return nil, err
		}
		q.stage = b
	}
	p := q.stage[q.used : q.used+n : q.used+n]
	q.used += n
	return p, nil
}

func (q *cudaQueue) write(b *cuda.Buffer, off int, p []byte) error {
	st, err := q.room(len(p))
	if err != nil {
		return err
	}
	copy(st, p)
	return b.WriteAtOn(q.st, off, unsafe.Pointer(&st[0]), len(p))
}

func (q *cudaQueue) read(b *cuda.Buffer, p []byte) error {
	st, err := q.room(len(p))
	if err != nil {
		return err
	}
	if err := b.ReadOn(q.st, unsafe.Pointer(&st[0]), len(p)); err != nil {
		return err
	}
	copy(p, st)
	q.used = 0 // ReadOn drained the stream
	return nil
}

func (q *cudaQueue) sync() error {
	if err := q.st.Sync(); err != nil {
		return err
	}
	q.used = 0
	return nil
}

// NewQueue is a non-blocking stream on this device (backend.Queued).
func (c *cudaDev) NewQueue() (Queue, error) {
	if !cuda.QueuesAvailable() {
		return nil, fmt.Errorf("backend: cuda: this driver has no non-blocking streams")
	}
	var st *cuda.Stream
	err := errCUDAClosed
	c.do(func() { st, err = c.d.NewQueue() })
	if err != nil {
		return nil, err
	}
	q := &cudaQueue{c: c, st: st}
	ownedQueues.Add(1)
	c.qmu.Lock()
	if c.queues == nil {
		c.queues = map[*cudaQueue]bool{}
	}
	c.queues[q] = true
	c.qmu.Unlock()
	return q, nil
}

// Close destroys the stream, after everything queued on it.
func (q *cudaQueue) Close() {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.release(&ownedQueues)
	c := q.c
	c.qmu.Lock()
	open := c.queues[q]
	delete(c.queues, q)
	c.qmu.Unlock()
	if !open {
		return // the device closed it
	}
	c.do(func() {
		q.st.Sync()
		q.st.Destroy()
		if q.stage != nil {
			cuda.FreeHost(q.stage)
			q.stage = nil
		}
	})
}

// SessionOn runs f on the calling thread, the context bound there, its work on
// q's stream. Unlike Session it takes no device-wide lock: sessions on two
// queues run at once, each waiting on its own stream alone.
func (c *cudaDev) SessionOn(q Queue, f func(Session)) {
	cq := q.(*cudaQueue)
	cq.mu.Lock()
	defer cq.mu.Unlock()
	c.life.RLock()
	defer c.life.RUnlock()
	c.qmu.Lock()
	open := c.queues[cq]
	c.qmu.Unlock()
	if !open {
		return
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	tid := threadID()
	c.enterQueue(tid, cq)
	defer c.leaveQueue(tid)
	if err := c.d.Bind(); err != nil {
		return
	}
	cq.s = cudaSession{c: c, q: cq}
	f(&cq.s)
}

// qRef is the queue a thread's Session runs on, and how deep.
type qRef struct {
	q *cudaQueue
	n int
}

func (c *cudaDev) enterQueue(tid int64, q *cudaQueue) {
	c.qmu.Lock()
	if c.queued == nil {
		c.queued = map[int64]qRef{}
	}
	r := c.queued[tid]
	c.queued[tid] = qRef{q, r.n + 1}
	c.qmu.Unlock()
	c.nQueued.Add(1)
}

func (c *cudaDev) leaveQueue(tid int64) {
	c.nQueued.Add(-1)
	c.qmu.Lock()
	if r := c.queued[tid]; r.n <= 1 {
		delete(c.queued, tid)
	} else {
		c.queued[tid] = qRef{r.q, r.n - 1}
	}
	c.qmu.Unlock()
}

// queueOf is the queue the calling thread is running a Session on, or nil.
func (c *cudaDev) queueOf(tid int64) *cudaQueue {
	if tid < 0 || c.nQueued.Load() == 0 {
		return nil
	}
	c.qmu.Lock()
	defer c.qmu.Unlock()
	return c.queued[tid].q
}

// inQueue reports whether tid is running a Session on a queue.
func (c *cudaDev) inQueue(tid int64) bool { return c.queueOf(tid) != nil }

// curQueue is queueOf for the calling thread.
func (c *cudaDev) curQueue() *cudaQueue {
	if c.nQueued.Load() == 0 {
		return nil
	}
	return c.queueOf(threadID())
}

// syncAll is cuCtxSynchronize, held off while any queue is capturing (see
// capturing). A thread inside a queue's Session holds the read side already
// and calls straight through: waiting for itself would never return.
func (c *cudaDev) syncAll() error {
	if c.nQueued.Load() == 0 || !c.inQueue(threadID()) {
		c.capturing.Lock()
		defer c.capturing.Unlock()
	}
	return cuda.Sync()
}

func (c *cudaDev) hasQueues() bool {
	c.qmu.Lock()
	defer c.qmu.Unlock()
	return len(c.queues) > 0
}

// Upload copies p into b at off on the device's copy queue (backend.Uploader).
// A session runs on a thread with the context current, as the copy needs.
func (s *cudaSession) Upload(b Buf, off int, p []byte) error {
	if len(p) == 0 {
		return nil
	}
	c := s.c
	if c.up == nil {
		if c.noUp {
			return s.WriteAt(b, off, p)
		}
		st, err := c.d.NewQueue()
		if err != nil {
			c.noUp = true
			return s.WriteAt(b, off, p)
		}
		c.up = st
	}
	return b.(*cudaBuf).b.WriteAtOn(c.up, off, unsafe.Pointer(&p[0]), len(p))
}

// UploadWait waits for the copy queue (backend.Uploader).
func (s *cudaSession) UploadWait() error {
	if s.c.up == nil {
		return nil
	}
	return s.c.up.Sync()
}
