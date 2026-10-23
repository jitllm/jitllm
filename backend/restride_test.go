package backend_test

import (
	"encoding/binary"
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestRestrideMovesExactlyTheBlock: every word of a rows x cols block lands at
// its new stride bit for bit, and every other destination word -- the gaps
// between rows and a guard after the last -- keeps the poison it was filled
// with. Shapes are the KV cache's: a transposed K growing and shrinking, a
// row-major prefix, and ragged sizes around a 128-thread group.
func TestRestrideMovesExactlyTheBlock(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		for _, c := range [][4]int{
			{1024, 257, 257, 513}, // transposed K, 256 -> 512
			{64, 100, 513, 257},   // a shrink keeps a prefix of each row
			{1, 257 * 1024, 257 * 1024, 257 * 1024},
			{3, 7, 9, 11},
			{1, 1, 1, 1},
			{5, 129, 130, 200},
		} {
			restrideCase(t, d, fmt.Sprintf("%s/%dx%d/%d->%d", d.API(), c[0], c[1], c[2], c[3]), c[0], c[1], c[2], c[3])
		}
	}
}

func restrideCase(t *testing.T, d backend.Device, name string, rows, cols, ss, ds int) {
	const poison, guard = 0xDEADBEEF, 0x5EED5EED
	src := make([]byte, 4*rows*ss)
	for i := 0; i < rows*ss; i++ {
		binary.LittleEndian.PutUint32(src[4*i:], uint32(i)*2654435761)
	}
	n := rows * ds
	dst := make([]byte, 4*(n+1))
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint32(dst[4*i:], poison)
	}
	binary.LittleEndian.PutUint32(dst[4*n:], guard)
	kern, err := kernels.Restride(rows, cols, ss, ds)
	if err != nil {
		t.Fatal(err)
	}
	if err := kern.Validate(); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	kk, err := d.Compile(kern)
	if err != nil {
		t.Fatalf("%s: compile: %v", name, err)
	}
	defer kk.Close()
	g := newGPU(t, d)
	defer g.free()
	bs, bd := g.up(src), g.up(dst)
	if err := kk.Launch((rows*cols+127)/128, 128, bs, bd); err != nil {
		t.Fatalf("%s: launch: %v", name, err)
	}
	got := make([]byte, len(dst))
	if err := bd.Read(got); err != nil {
		t.Fatalf("%s: read: %v", name, err)
	}
	bad := 0
	for i := 0; i <= n; i++ {
		want := uint32(guard)
		if i < n {
			want = poison
			if r, c := i/ds, i%ds; c < cols {
				want = uint32(r*ss+c) * 2654435761
			}
		}
		if v := binary.LittleEndian.Uint32(got[4*i:]); v != want {
			if bad++; bad <= 3 {
				t.Errorf("%s: word %d = %#x, want %#x", name, i, v, want)
			}
		}
	}
	if bad > 0 {
		t.Fatalf("%s: %d of %d words wrong", name, bad, n+1)
	}
}
