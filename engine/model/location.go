package model

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
)

// A block's location is the device that runs it for this State, the CPU
// among them: nn.CPU, or the name the attached tier gives the device holding
// it. onDev and gpuLayers are where the State keeps it, written only by
// markOnDev, unmarkOnDev, clearOnDev, shrinkOnDev and holdCPURuns
// (srcgate.TestBlockLocationIsWrittenByItsSetters).

// WhereIs is the device running block li: nn.CPU for a block this State runs
// on the CPU, else the name of the device holding it ("" for a device with no
// name). A block outside the State's [lo, hi) is "".
func (s *State) WhereIs(li int) string {
	if li < s.lo || li >= s.hi {
		return ""
	}
	if !s.devAt(li) {
		return nn.CPU
	}
	if nd, ok := s.ld.(nn.NamedDevice); ok {
		if at, ok := nd.DeviceOf(li); ok {
			return at
		}
	}
	return ""
}

// Locations is WhereIs for every block below the State's last: element li is
// block li's device, "" for the blocks below lo the State does not run.
func (s *State) Locations() []string {
	out := make([]string, s.hi)
	for li := s.lo; li < s.hi; li++ {
		out[li] = s.WhereIs(li)
	}
	return out
}

// onCPU reports whether to names the CPU.
func onCPU(to string) bool { return nn.DeviceName(to) == nn.CPU }

// cpuListed reports whether the engine may choose the CPU for a block
// (WithCPU): as the destination of a block no card takes, of a relocation, of
// a fault's evacuation.
func (s *State) cpuListed() bool { return !s.m.opt.noCPU }

// errCPUUnlisted is the refusal of a move the engine would have made to the
// CPU, had the caller listed it.
func errCPUUnlisted(what string) error {
	return fmt.Errorf("model: %s, and the CPU is not one of the devices listed (WithCPU)", what)
}
