//go:build darwin

package sched

import (
	"runtime"

	"github.com/jitllm/jitllm/backend/ffi"
)

// QoS classes (sys/qos.h). macOS exposes no processor affinity; on Apple
// Silicon its scheduler keeps a lowered-QoS thread on the efficiency cluster,
// which is the only way to put work there.
const (
	QoSBackground = 0x09
	QoSUtility    = 0x11
	QoSUserInit   = 0x19

	defaultEQoSClass = QoSUtility
)

// QoS is the calls that put a thread at a QoS class
// (pthread_set_qos_class_self_np, qos_class_self), loaded from libSystem by
// whoever lowers threads, and the class a thread on the E-cores asks for. The
// zero QoS lowers nothing.
type QoS struct {
	set    func(uint32, int32) int32
	self   func() uint32
	eClass uint32
}

// LoadQoS is the QoS whose threads on the E-cores ask for class eClass, 0
// the one the measurements chose (QoSUtility).
func LoadQoS(eClass uint32) QoS {
	q := QoS{eClass: eClass}
	if q.eClass == 0 {
		q.eClass = defaultEQoSClass
	}
	l, err := ffi.Open("/usr/lib/libSystem.B.dylib")
	if err != nil || !l.Has("pthread_set_qos_class_self_np") || !l.Has("qos_class_self") {
		return q
	}
	q.set = ffi.Fn2[int32, uint32, int32](l, "pthread_set_qos_class_self_np")
	q.self = ffi.Fn0[uint32](l, "qos_class_self")
	return q
}

// Ready reports whether q can set a thread's class.
func (q QoS) Ready() bool { return q.set != nil }

// EClass is the QoS class a thread on the E-cores asks for.
func (q QoS) EClass() uint32 { return q.eClass }

// Self is the calling thread's QoS class, 0 where there is none to read.
func (q QoS) Self() uint32 {
	if q.self == nil {
		return 0
	}
	return q.self()
}

// EQoS puts the calling goroutine's thread at the E-core class and locks the
// goroutine to that thread, so what it runs stays on the E-cores. It reports
// whether the class was set. The lock is for the goroutine's life: a thread
// whose class was lowered must not go back to the runtime's pool, where it
// would carry the class to whatever ran on it next (the leak RULE 5 records
// for an affinity mask). A pool's workers call it once (New, on a pool wholly
// on the E-cores).
func (q QoS) EQoS() bool {
	if !q.Ready() {
		return false
	}
	runtime.LockOSThread()
	return q.set(q.eClass, 0) == 0
}

// Lowered is a thread taken down to the E-core class by LowerToE.
type Lowered struct {
	q   QoS
	was uint32
	ok  bool
}

// LowerToE puts the calling goroutine's thread at the E-core class and locks
// the goroutine to it, so a prefill run from an arbitrary goroutine keeps
// participant 0 (the caller of every region) on the efficiency cluster with
// the pool's workers. Raise puts the class back and unlocks. A thread that
// cannot be raised again is not handed back to the runtime: the goroutine
// keeps it locked until it exits, which ends the thread rather than leaving
// a lowered thread in the runtime's pool.
func (q QoS) LowerToE() Lowered {
	if !q.Ready() {
		return Lowered{}
	}
	runtime.LockOSThread()
	was := q.self()
	return Lowered{q: q, was: was, ok: q.set(q.eClass, 0) == 0}
}

// Raise undoes LowerToE.
func (l Lowered) Raise() {
	if !l.q.Ready() {
		return
	}
	if !l.ok || (l.q.set(l.was, 0) == 0 && l.q.self() == l.was) {
		runtime.UnlockOSThread()
	}
}

// CanSplitE reports whether a session can be given the E-cores: a host with
// them and q able to ask for them.
func (q QoS) CanSplitE() bool { return len(ECores()) > 0 && q.Ready() }

// allE reports whether every cpu of a pool is an E-core index (ECores).
func allE(cpus []int) bool {
	e := ECores()
	if len(e) == 0 || len(cpus) == 0 {
		return false
	}
	for _, c := range cpus {
		if c < e[0] {
			return false
		}
	}
	return true
}
