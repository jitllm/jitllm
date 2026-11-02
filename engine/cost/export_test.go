package cost

import (
	"testing"

	"github.com/jitllm/jitllm/composer"
)

var composeOpen func(*testing.T, string) composer.Model

func SetComposeOpen(f func(*testing.T, string) composer.Model) { composeOpen = f }

func composing(t *testing.T, name string) composer.Model {
	t.Helper()
	return composeOpen(t, name)
}
