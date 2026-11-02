package ir

import "testing"

// built is b's kernel, the test failed where the builder recorded an error.
func built(t testing.TB, b *Builder) *Kernel {
	t.Helper()
	k, err := b.Done()
	if err != nil {
		t.Fatal(err)
	}
	return k
}
