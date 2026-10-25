package model

import (
	"errors"
	"strings"

	"github.com/jitllm/jitllm/engine/nn"
)

const nFailCauses = int(nn.FailFault) + 1

// devCause is what kind of refusal the device's last call was. A device that
// does not say is a decline.
func (s *State) devCause() nn.FailCause {
	if c, ok := s.ld.(nn.CauseReporter); ok {
		return c.ErrCause()
	}
	return nn.FailDecline
}

// buildRefusal is the error a device call that refused because a kernel did
// not build ends with, nil for any other refusal. No move gives a kernel its
// build, so the caller returns it where it would have demoted or relocated:
// the kernel is the work still to write.
func (s *State) buildRefusal(li int) error {
	if s.devCause() != nn.FailBuild {
		return nil
	}
	s.buildRefused++
	return &nn.BuildError{Layer: li, Err: errors.New(strings.TrimPrefix(s.devWhy(), ": "))}
}

// Evacuations is every way work left a device since the State began, by why.
// A fault is a driver error or a device lost; a decline is a refusal that
// names a limit (room, a width). Only those two move work: a kernel that did
// not build moves nothing and is counted as BuildRefused, the calls it ended
// with an nn.BuildError.
type Evacuations struct {
	// DemotedFault and DemotedDecline are the times every block was moved to
	// the CPU device, by the cause the device gave (DeviceDemotions is their
	// sum).
	DemotedFault, DemotedDecline int
	// Relocated is the blocks handed to the CPU device one at a time for room;
	// a relocation is always a decline.
	Relocated    int
	BuildRefused int
}

// Evacuations reads the State's counters of work moved off a device.
func (s *State) Evacuations() Evacuations {
	return Evacuations{
		DemotedFault:   s.demotedBy[nn.FailFault],
		DemotedDecline: s.demotedBy[nn.FailDecline],
		Relocated:      s.devRelocated,
		BuildRefused:   s.buildRefused,
	}
}
