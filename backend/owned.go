package backend

import "sync/atomic"

// Owned is what the process holds of the drivers' objects through this
// package: devices opened and not yet Closed, and queues made by NewQueue and
// not yet Closed by their owner. A queue counts until its own Close even when
// its device closed first and the driver reclaimed it there: the count is of
// what the engine owns, so a session that forgets its queue shows here even on
// a backend whose device Close tidies up after it.
//
// It exists for leak gates. An OS thread count cannot tell a driver object kept
// alive from the Go runtime keeping an idle thread (the runtime never destroys
// an M except under a goroutine that exits locked), so a gate counts the
// objects themselves.
type Owned struct {
	Devices, Queues int64
}

var ownedDevices, ownedQueues atomic.Int64

// OwnedNow is the process's count of live driver objects, every backend
// together.
func OwnedNow() Owned {
	return Owned{Devices: ownedDevices.Load(), Queues: ownedQueues.Load()}
}

// once is a Close that counts down exactly once.
type once struct{ done atomic.Bool }

func (o *once) release(n *atomic.Int64) {
	if o.done.CompareAndSwap(false, true) {
		n.Add(-1)
	}
}
