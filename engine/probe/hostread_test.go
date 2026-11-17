//go:build amd64 || arm64

package probe

import (
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/device/cpu"
)

// TestTheReadSumsItsWholeBuffer holds the read the rates time to the words
// it reads: the sum it stores is the wrapped sum of every word of its
// buffer, so a read that covers less than the bytes a rate counts stores
// another sum.
func TestTheReadSumsItsWholeBuffer(t *testing.T) {
	r := &reader{fm: newFormer(cpu.New()), forms: map[int]*formed{}}
	defer r.close()
	for _, n := range []int{64, 192, 4096, 1 << 18} {
		words := make([]uint32, n)
		var want uint32
		for i := range words {
			words[i] = uint32(i)*2654435761 + 1
			want += words[i]
		}
		b, err := r.on(unsafe.Slice((*byte)(unsafe.Pointer(&words[0])), 4*n))
		if err != nil {
			t.Fatal(err)
		}
		b.run()
		if got := b.keep[len(b.keep)-1][0]; got != want {
			t.Errorf("a read of %d words sums %#x, its buffer %#x", n, got, want)
		}
	}
}
