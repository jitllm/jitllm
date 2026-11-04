package backend

import "fmt"

// Restriction decides which devices an open may reach (Opts.Admit). It exists
// for test binaries, which share a box with other jobs and must stay on the
// card they were given: Vulkan ignores CUDA_VISIBLE_DEVICES, so a gate that
// opens "every Vulkan device" reaches cards it does not own. Nothing in a
// shipped binary passes one, and the package reads no environment: the test
// support package (internal/testdevices) reads JITLLM_TEST_DEVICES and builds
// it.
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

// admit applies o's restriction to an open of api's ordinal ord. With none,
// the ordinal is the caller's own and a default pick is the backend's.
func (o Opts) admit(api string, ord int) (int, error) {
	if o.Admit == nil {
		return ord, nil
	}
	n, err := o.Admit(api, ord, o.Enumerating)
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
