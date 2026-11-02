//go:build amd64 || arm64

package cpu

import (
	"bytes"
	"testing"
	"unsafe"
)

// TestPrefetchLinesWritesNothing runs the lookahead's kernel over 0 through a
// few pieces of lines, and over lines that end past the buffer (a hint does
// not fault), and asserts the bytes come back as they were: the kernel must
// only hint. A kernel that stores into its lines fails it at one line.
func TestPrefetchLinesWritesNothing(t *testing.T) {
	c := onHost(t)(hostBytes(t)(hostTable().PrefetchLines()), nil)
	const lines = 70
	buf := make([]byte, (lines+2)*PrefetchLine)
	for i := range buf {
		buf[i] = byte(i*7 + 3)
	}
	want := bytes.Clone(buf)
	for _, n := range []int{0, 1, 2, 7, 64, lines} {
		c.Call(&Args{Out: (*float32)(unsafe.Pointer(&buf[0])), K: int64(n)})
		if !bytes.Equal(buf, want) {
			t.Fatalf("%d lines: the kernel wrote into the lines it hinted", n)
		}
	}
	// The last line of the buffer and lines beyond it.
	c.Call(&Args{Out: (*float32)(unsafe.Pointer(&buf[len(buf)-PrefetchLine])), K: 1 << 10})
	if !bytes.Equal(buf, want) {
		t.Fatal("hinting past the buffer wrote into it")
	}
}
