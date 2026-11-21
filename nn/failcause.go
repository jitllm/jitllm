package nn

import (
	"errors"
	"fmt"
)

// FailCause says what kind of refusal the device's last call was. Only a
// fault moves work: a block whose kernel did not build is missing work to
// write, and no move gives it a kernel.
type FailCause uint8

const (
	// FailDecline is a refusal that names a limit: a width, a setting.
	FailDecline FailCause = iota
	// FailRoom is a decline for room: the device holds no page for the
	// call's history now, and may once other sequences give theirs back.
	FailRoom
	// FailBuild is a kernel that did not build on the device: its form, its
	// lowering, its compile or its link.
	FailBuild
	// FailFault is the device itself failing: a driver error, a device lost.
	FailFault
)

func (c FailCause) String() string {
	switch c {
	case FailRoom:
		return "room"
	case FailBuild:
		return "build"
	case FailFault:
		return "fault"
	}
	return "decline"
}

// CauseReporter says what kind of refusal the device's last call was.
type CauseReporter interface{ ErrCause() FailCause }

// BuildError is a kernel that did not build on a device. Layer is the block
// whose call needed the kernel, or -1 where the code that failed did not know.
type BuildError struct {
	Layer int
	Err   error
}

func (e *BuildError) Error() string {
	if e.Layer < 0 {
		return e.Err.Error()
	}
	return fmt.Sprintf("block %d: a kernel did not build on the device: %v", e.Layer, e.Err)
}

func (e *BuildError) Unwrap() error { return e.Err }

// IsBuildError reports whether err is, or wraps, a BuildError.
func IsBuildError(err error) bool {
	var b *BuildError
	return errors.As(err, &b)
}
