// Package cost prices the composer's final IR on a device, ahead of
// running anything: what a placement, a form of a kernel or a request sent
// to another node would cost. It reads a device as numbers -- what its
// probe measured, or an approximation of the hardware -- so a machine can
// price any device, its own or another's.
package cost

import "github.com/jitllm/jitllm/jit"

// Device is a device as the composer forms kernels for it and as an
// estimate weighs it.
type Device struct {
	jit.Device
	// Units is the multiprocessors a GPU runs workgroups on, 0 where
	// nothing measured them.
	Units int
	// Bandwidth is the bytes a second the device's memory serves all its
	// threads, Thread what one thread gets alone; Launch the seconds a
	// launch or a call costs; Link the bytes a second it exchanges with
	// another device; Latency the seconds a read that misses every cache
	// waits for memory, Start what a short stream from memory waits beyond
	// its bytes at the memory's rate; Floor the seconds a GPU kernel holds
	// the device beyond its work, launches issued back to back; Overlap the
	// loads a GPU thread has in flight in a loop that uses each as it goes.
	Bandwidth, Thread, Launch, Link, Latency, Start, Floor, Overlap float64
	// Caches are its cache levels, nearest first.
	Caches [3]Cache
	// Rates are the instructions a second it issues of each class, Issue
	// of a mix of the classes, whose instructions run beside each other.
	Rates [NumClasses]Rate
	Issue Rate
	// Delay is the seconds an instruction of each class takes before an
	// instruction using its result can start.
	Delay [NumClasses]float64
}

// Placement is the device each part of a model is priced on.
type Placement func(jit.Part) Device

// Composer is placement p as the composer places parts: on the devices
// alone, what they cost aside.
func (p Placement) Composer() jit.Placement {
	return func(part jit.Part) jit.Device { return p(part).Device }
}
