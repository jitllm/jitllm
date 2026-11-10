package engine

import (
	"errors"
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

const nFailCauses = int(nn.FailFault) + 1

// devCause is what kind of refusal the device's last call was. A device that
// does not say is a decline.
func (s *Instance) devCause() nn.FailCause {
	if c, ok := s.ld.(nn.CauseReporter); ok {
		return c.ErrCause()
	}
	return nn.FailDecline
}

// refusal is the error a device call that refused ends with, nil for a
// fault. A kernel that did not build and a decline -- a limit, or a form the
// device does not serve -- are answered by no move: the caller returns the
// error. Only a fault moves blocks (demoteAll).
func (s *Instance) refusal(li int) error {
	return s.refusalOf(li, strings.TrimPrefix(s.devWhy(), ": "), s.devCause())
}

// refusalOf is refusal for a device that said why and with what cause; a
// block the front end does not state says what of it is not covered.
func (s *Instance) refusalOf(li int, why string, cause nn.FailCause) error {
	if cause == nn.FailDecline {
		if u, err := s.m.UncoveredAt(li); u != model.Covered || err != nil {
			why = fmt.Sprintf("%s (the front end: %v, %v)", why, u, err)
		}
	}
	switch cause {
	case nn.FailBuild:
		s.buildRefused++
		return &nn.BuildError{Layer: li, Err: errors.New(why)}
	case nn.FailDecline, nn.FailRoom:
		s.declined++
		return errDeclined{li, why, cause == nn.FailRoom}
	}
	return nil
}

// errDeclined is a device call that declined block layer: the reason is the
// work still to do, and nothing runs the block elsewhere in its place.
type errDeclined struct {
	layer int
	why   string
	room  bool
}

func (e errDeclined) Error() string {
	return fmt.Sprintf("model: the device declined block %d: %s", e.layer, e.why)
}

// Unwrap is ErrNoRoom for a decline for room.
func (e errDeclined) Unwrap() error {
	if e.room {
		return ErrNoRoom
	}
	return nil
}

// ErrNoRoom is a call the device declined for room for its history: run with
// fewer rows beside it, or once another sequence gives its pages back, it
// may fit.
const ErrNoRoom roomError = "model: the device has no room for the call's history now"

type roomError string

func (e roomError) Error() string { return string(e) }

// Evacuations is every way work left a device since the instance began, by why.
// A fault is a driver error or a device lost, and only a fault moves work. A
// kernel that did not build and a decline move nothing: they are counted
// (BuildRefused, Declined) and the call ends with the error. A relocation
// for room is a placement, made before a call.
type Evacuations struct {
	// DemotedFault and DemotedDecline are the times every block was moved to
	// the CPU device, by the cause the device gave (DeviceDemotions is their
	// sum).
	DemotedFault, DemotedDecline int
	// Relocated is the blocks handed to the CPU device one at a time for room;
	// a relocation is always a decline.
	Relocated    int
	BuildRefused int
	// Declined is the calls the device declined, each ended with errDeclined.
	Declined int
}

// Evacuations reads the instance's counters of work moved off a device.
func (s *Instance) Evacuations() Evacuations {
	return Evacuations{
		DemotedFault:   s.demotedBy[nn.FailFault],
		DemotedDecline: s.demotedBy[nn.FailDecline] + s.demotedBy[nn.FailRoom],
		Relocated:      s.devRelocated,
		BuildRefused:   s.buildRefused,
		Declined:       s.declined,
	}
}
