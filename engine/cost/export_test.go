package cost

import (
	"testing"

	"github.com/jitllm/jitllm/jit"
)

var composeOpen func(*testing.T, string) jit.Model

func SetComposeOpen(f func(*testing.T, string) jit.Model) { composeOpen = f }

func composing(t *testing.T, name string) jit.Model {
	t.Helper()
	return composeOpen(t, name)
}
