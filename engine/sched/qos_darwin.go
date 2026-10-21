//go:build darwin

package sched

import (
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/jitllm/jitllm/jit_old/gpu/ffi"
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

var (
	qosOnce  sync.Once
	qosSet   func(uint32, int32) int32
	qosSelf  func() uint32
	qosReady bool
	eClass   atomic.Uint32
)

func qosInit() bool {
	qosOnce.Do(func() {
		l, err := ffi.Open("/usr/lib/libSystem.B.dylib")
		if err != nil || !l.Has("pthread_set_qos_class_self_np") || !l.Has("qos_class_self") {
			return
		}
		qosSet = ffi.Fn2[int32, uint32, int32](l, "pthread_set_qos_class_self_np")
		qosSelf = ffi.Fn0[uint32](l, "qos_class_self")
		qosReady = true
	})
	return qosReady
}

// SetEClass picks the QoS class a thread on the E-cores asks for (QoSUtility
// and the like), for threads that ask after the call, and returns the class
// it replaces. It exists so a gate can compare classes; the default is the
// one the measurements chose.
func SetEClass(class uint32) uint32 {
	prev := EClass()
	eClass.Store(class)
	return prev
}

// EClass is the QoS class a thread on the E-cores asks for.
func EClass() uint32 {
	if c := eClass.Load(); c != 0 {
		return c
	}
	return defaultEQoSClass
}

// QoSSelf is the calling thread's QoS class, 0 where there is none to read.
func QoSSelf() uint32 {
	if !qosInit() {
		return 0
	}
	return qosSelf()
}

// EQoS puts the calling goroutine's thread at the E-core class and locks the
// goroutine to that thread, so what it runs stays on the E-cores. It reports
// whether the class was set. The lock is for the goroutine's life: a thread
// whose class was lowered must not go back to the runtime's pool, where it
// would carry the class to whatever ran on it next (the leak RULE 5 records
// for an affinity mask). A pool's workers call it once (New, on a pool wholly
// on the E-cores).
func EQoS() bool {
	if !qosInit() {
		return false
	}
	runtime.LockOSThread()
	return qosSet(EClass(), 0) == 0
}

// Lowered is a thread taken down to the E-core class by LowerToE.
type Lowered struct {
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
func LowerToE() Lowered {
	if !qosInit() {
		return Lowered{}
	}
	runtime.LockOSThread()
	was := qosSelf()
	return Lowered{was: was, ok: qosSet(EClass(), 0) == 0}
}

// Raise undoes LowerToE.
func (l Lowered) Raise() {
	if !qosReady {
		return
	}
	if !l.ok || (qosSet(l.was, 0) == 0 && qosSelf() == l.was) {
		runtime.UnlockOSThread()
	}
}

// CanSplitE reports whether a session can be given the E-cores: a host with
// them and a way to ask for them.
func CanSplitE() bool { return len(ECores()) > 0 && qosInit() }

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
