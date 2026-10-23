package ir_test

import (
	"testing"

	"github.com/jitllm/jitllm/jit/ir"
)

// mustBuild is b's kernel, the test failed where the builder recorded an
// error.
func mustBuild(t testing.TB, b *ir.Builder) *ir.Kernel {
	t.Helper()
	k, err := b.Done()
	if err != nil {
		t.Fatal(err)
	}
	return k
}
