package ir

import "fmt"

// The grid-wide barrier.
//
// OpBarrier orders one workgroup. OpGridBarrier orders the whole launch: no
// thread of any workgroup passes it until every thread of every workgroup has
// reached it, and every global write made before it by any thread is visible
// to every global read after it by any other. It is what lets one launch run a
// sequence of dependent phases -- a decode block's norm, matvecs, attention and
// FFN -- that would otherwise be one launch each.
//
// It is built from two documented mechanisms, on every target:
//
//   - an atomic arrive-and-wait counter in a device buffer (the sync buffer,
//     Args[0]), with device-scope release on arrival and acquire on departure,
//     each workgroup waiting until every other has arrived; and
//   - a grid no larger than what the device holds resident at once. A
//     workgroup spinning on the counter waits for workgroups that have not
//     started; if one of those can only start when a spinning one finishes, the
//     launch never ends. None of the three APIs promises forward progress to a
//     workgroup that is not resident (the CUDA Programming Guide's grid
//     synchronisation section requires co-residency and offers the cooperative
//     launch to guarantee it; the Vulkan specification's execution model and
//     Metal's give no inter-workgroup progress guarantee at all), so the
//     co-residency limit is the backend's to state and to enforce at launch:
//     backend.GridLimit.
//
// The sync buffer is GridSyncWords u32 words, zero when allocated; what they
// hold is the lowering's (PTX keeps one 64-bit count that only grows, the
// others a count and a generation word). Every launch leaves it valid for the
// next, so it is never reset. Two launches that may run at once must not share
// one. The barrier owns its buffer: Validate refuses any other access to it.
//
// Every thread of every workgroup must reach the barrier the same number of
// times, so it may not sit inside a loop with a runtime count (whose trip may
// differ between workgroups). A constant-count loop is allowed.
//
// RULE 13's in-place refusal applies per phase: a kernel may read a buffer it
// writes when a grid barrier separates the write from the read, because no
// surplus thread clamped to the last item can then observe its own update.

// GridSyncWords is the sync buffer's size in u32 words.
const GridSyncWords = 2

// GridBarrier synchronises every workgroup of the launch through sync, a u32
// parameter of GridSyncWords words that nothing else in the kernel touches.
func (b *Builder) GridBarrier(sync Value) {
	b.emit(Op{Kind: OpGridBarrier, Args: [3]Value{sync}})
}

// GridSync reports whether the kernel contains a grid barrier or dependency
// counters (depsync.go), and so may only be launched with a grid the device
// holds resident at once.
func (k *Kernel) GridSync() bool {
	for _, o := range k.Ops {
		if o.Kind == OpGridBarrier || o.Kind == OpDepExit {
			return true
		}
	}
	return false
}

// GridSyncParam is the parameter index of the kernel's sync buffer, or -1.
func (k *Kernel) GridSyncParam() int {
	for _, o := range k.Ops {
		if o.Kind == OpGridBarrier || o.Kind == OpDepExit {
			return k.paramOf(o.Args[0])
		}
	}
	return -1
}

// validateGrid checks every grid barrier: its buffer is one u32 parameter,
// the same for every barrier and touched by nothing else, and no barrier sits
// in a runtime-count loop.
func (k *Kernel) validateGrid() error {
	sync := -1
	var loops []bool // per open loop: whether its count is a runtime value
	for i, o := range k.Ops {
		switch o.Kind {
		case OpLoop:
			loops = append(loops, o.Imm < 0)
		case OpEndLoop:
			if len(loops) > 0 {
				loops = loops[:len(loops)-1]
			}
		case OpGridBarrier:
			p := k.paramOf(o.Args[0])
			if p < 0 {
				return fmt.Errorf("ir: op %d: a grid barrier's sync buffer must be a parameter", i)
			}
			if k.Params[p].Elem != U32 {
				return fmt.Errorf("ir: op %d: grid barrier sync buffer %q is %s, not u32", i, k.Params[p].Name, k.Params[p].Elem)
			}
			if sync >= 0 && sync != p {
				return fmt.Errorf("ir: op %d: a second grid sync buffer (param %d after %d); one launch has one", i, p, sync)
			}
			sync = p
			for _, rt := range loops {
				if rt {
					return fmt.Errorf("ir: op %d: a grid barrier inside a loop with a runtime count; "+
						"every workgroup must reach it equally often", i)
				}
			}
		}
	}
	if sync < 0 {
		return nil
	}
	for i, o := range k.Ops {
		switch o.Kind {
		case OpGridBarrier, OpParam:
			continue
		}
		for _, a := range o.Args {
			if a != 0 && k.paramOf(a) == sync {
				return fmt.Errorf("ir: op %d (%s) touches the grid sync buffer %q, which the barrier owns",
					i, o.Kind, k.Params[sync].Name)
			}
		}
		for _, a := range o.Vargs {
			if k.paramOf(a) == sync {
				return fmt.Errorf("ir: op %d (%s) touches the grid sync buffer", i, o.Kind)
			}
		}
	}
	return nil
}
