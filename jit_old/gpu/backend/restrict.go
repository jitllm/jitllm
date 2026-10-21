package backend

import (
	"fmt"
	"sync/atomic"
)

// Restriction decides which devices a process may open. It exists for test
// binaries, which share a box with other jobs and must stay on the card they
// were given: Vulkan ignores CUDA_VISIBLE_DEVICES, so a gate that opens "every
// Vulkan device" reaches cards it does not own. Nothing in a shipped binary
// sets one, and the package reads no environment: the test support package
// (internal/testdevices) reads JITLLM_TEST_DEVICES and installs it.
//
// api is the lowering's name ("ptx", "spirv", "msl", "amdgcn"). ord is the
// ordinal the open asks for, -1 for the backend's own default pick and -2 for
// a Vulkan selector that is a name, whose ordinal is known only after the
// open. enumerated is true when the open is part of finding devices (the
// default pick, a loop over a backend's ordinals) and false when the caller
// named this device. The returned ordinal is the one to open: a default pick
// resolves to the first device the restriction allows. A non-nil error
// refuses the open.
type Restriction func(api string, ord int, enumerated bool) (int, error)

var restriction atomic.Pointer[Restriction]

// Restrict installs r for every later open in this process; nil removes it.
func Restrict(r Restriction) {
	if r == nil {
		restriction.Store(nil)
		return
	}
	restriction.Store(&r)
}

// admit applies the installed restriction. With none, the ordinal is the
// caller's own and a default pick is the backend's.
func admit(api string, ord int, enumerated bool) (int, error) {
	r := restriction.Load()
	if r == nil {
		return ord, nil
	}
	n, err := (*r)(api, ord, enumerated)
	if err != nil {
		return ord, fmt.Errorf("backend: %w", err)
	}
	return n, nil
}

// enumerating is o marked as part of a device search.
func enumerating(o Opts) Opts {
	o.Enumerating = true
	return o
}
