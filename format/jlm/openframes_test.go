package jlm_test

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// framesOf is c's frames, closed when t ends.
func framesOf(t testing.TB, c *jlm.File, opts ...jlm.FramesOption) *jlm.Frames {
	t.Helper()
	f, err := jlm.NewFrames(c, opts...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(f.Close)
	return f
}

// openFrames opens the container at path and its frames, both closed when t
// ends.
func openFrames(t testing.TB, path string, opts ...jlm.FramesOption) *jlm.Frames {
	t.Helper()
	c, err := jlm.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	f, err := jlm.NewFrames(c, opts...)
	if err != nil {
		c.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		f.Close()
		c.Close()
	})
	return f
}
