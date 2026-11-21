package server

import (
	"context"
	"errors"
	"fmt"
	"io/fs"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/offheap"
	"github.com/jitllm/jitllm/model"
)

// connectErr maps an engine error onto a Connect code. A GGUF handed to Open
// yields a *model.NotConvertedError carrying the convert command; it becomes
// FailedPrecondition, with the command repeated in an error detail so a
// client can show it as an action. A container of another version
// (*jlm.VersionError) is FailedPrecondition too, its fix -- upgrade or
// convert -- in the metadata.
func connectErr(err error) error {
	if err == nil {
		return nil
	}
	var already *connect.Error
	if errors.As(err, &already) {
		return already
	}

	var nc *model.NotConvertedError
	if errors.As(err, &nc) {
		ce := connect.NewError(connect.CodeFailedPrecondition, err)
		ce.Meta().Set("jitllm-not-converted", nc.Path)
		return ce
	}
	var ve *jlm.VersionError
	if errors.As(err, &ve) {
		ce := connect.NewError(connect.CodeFailedPrecondition, err)
		fix := "convert"
		if ve.Upgradable {
			fix = "upgrade"
		}
		ce.Meta().Set("jitllm-container-version", fix)
		return ce
	}
	switch {
	case errors.Is(err, ErrNotFound), errors.Is(err, fs.ErrNotExist):
		// A file named in the request that is not there is the caller's to
		// fix, the same as an unknown id.
		return connect.NewError(connect.CodeNotFound, err)
	case errors.Is(err, ErrExists):
		return connect.NewError(connect.CodeAlreadyExists, err)
	case errors.Is(err, ErrInvalid):
		return connect.NewError(connect.CodeInvalidArgument, err)
	case errors.Is(err, ErrInUse), errors.Is(err, ErrPlacement):
		return connect.NewError(connect.CodeFailedPrecondition, err)
	case errors.Is(err, ErrOverloaded):
		// Nothing ran; the caller retries after the queue drains.
		return connect.NewError(connect.CodeResourceExhausted, err)
	case errors.Is(err, offheap.ErrNoMemory):
		// The host has no memory for it now: a retry once sessions close or
		// models unload is correct, the identical request unchanged.
		return connect.NewError(connect.CodeResourceExhausted, err)
	case errors.Is(err, ErrQueueTimeout):
		// ResourceExhausted, not Unavailable: the caller stopped waiting for
		// a busy device, the session is still open and a retry is correct.
		return connect.NewError(connect.CodeResourceExhausted, err)
	case errors.Is(err, context.Canceled):
		return connect.NewError(connect.CodeCanceled, err)
	case errors.Is(err, context.DeadlineExceeded):
		return connect.NewError(connect.CodeDeadlineExceeded, err)
	}
	return connect.NewError(connect.CodeInternal, err)
}

// chatErr is a chat render's error as the server answers it: a system message
// the model's chat template has no role for is the caller's to change.
func chatErr(err error) error {
	var sr *model.SystemRoleError
	if errors.As(err, &sr) {
		return fmt.Errorf("%w: %w", ErrInvalid, err)
	}
	return err
}

// systemPromptError is the refusal of a system prompt a request sends apart
// from its messages, for a chat template with no system role.
type systemPromptError struct{ role *model.SystemRoleError }

func (e *systemPromptError) Error() string {
	return "the request's system prompt has no place in this chat template, which has no system role; " +
		"send its text in a user message"
}

func (e *systemPromptError) Unwrap() error { return e.role }

func invalid(format string, a ...any) error {
	return connect.NewError(connect.CodeInvalidArgument, fmt.Errorf(format, a...))
}

func unimplemented(format string, a ...any) error {
	return connect.NewError(connect.CodeUnimplemented, fmt.Errorf(format, a...))
}
