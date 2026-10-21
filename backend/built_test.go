package backend

import (
	"testing"

	"github.com/jitllm/jitllm/jit/ir"
)

// built is the kernel b holds, the test failed where b recorded an error.
func built(t testing.TB, b *ir.Builder) *ir.Kernel {
	t.Helper()
	k, err := b.Done()
	if err != nil {
		t.Fatal(err)
	}
	return k
}
