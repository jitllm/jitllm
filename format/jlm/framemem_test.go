package jlm

import (
	"errors"
	"testing"
	"unsafe"
)

// countedMemory is a FrameMemory that hands out aligned buffers and keeps
// what it handed out, so a gate can see the frames' memory came from it and
// went back to it.
type countedMemory struct {
	live  map[*byte]int
	maps  int
	refus bool
}

func (c *countedMemory) Map(n int) ([]byte, error) {
	if c.refus {
		return nil, errors.New("no memory")
	}
	b := alignedBuf(n)
	if c.live == nil {
		c.live = map[*byte]int{}
	}
	c.live[unsafe.SliceData(b)] = n
	c.maps++
	return b, nil
}

func (c *countedMemory) Unmap(b []byte) { delete(c.live, unsafe.SliceData(b)) }

// TestFramesTakeTheirDevicesMemory holds frames given a memory to it: every
// page they read lands in a buffer it mapped, MappedBytes counts them, and
// every one goes back to it at Close. A memory that refuses leaves the
// frames working on their own.
func TestFramesTakeTheirDevicesMemory(t *testing.T) {
	const nblocks, page = 4, 8 * 4096
	f := newReal(t, nblocks, page, 4096)
	mem := &countedMemory{}
	f.mem = mem
	for i := 0; i < nblocks; i++ {
		if err := f.EnsurePage(i); err != nil {
			t.Fatal(err)
		}
		p := f.Page(i)
		if _, ok := mem.live[unsafe.SliceData(p)]; !ok {
			t.Fatalf("page %d is not in the memory the frames were given", i)
		}
		if p[0] != byte(i+1) {
			t.Fatalf("page %d holds %d, want %d", i, p[0], i+1)
		}
	}
	if mem.maps < nblocks {
		t.Fatalf("the frames mapped %d buffers of their memory for %d pages", mem.maps, nblocks)
	}
	if got := f.MappedBytes(); got < nblocks*page {
		t.Fatalf("MappedBytes is %d, below the %d bytes of pages held", got, nblocks*page)
	}
	f.Close()
	if len(mem.live) != 0 {
		t.Fatalf("%d buffers of the frames' memory were not given back at Close", len(mem.live))
	}

	refusing := &countedMemory{refus: true}
	g := newReal(t, nblocks, page, 4096)
	g.mem = refusing
	if err := g.EnsurePage(0); err != nil {
		t.Fatal(err)
	}
	if p := g.Page(0); len(p) == 0 || p[0] != 1 {
		t.Fatal("a memory that refused left the frames without a page")
	}
	g.Close()
}
