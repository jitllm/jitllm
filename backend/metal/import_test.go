//go:build darwin

package metal

import (
	"os"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/internal/offheap"
)

// newBufferWithBytesNoCopy:, at this package's level. The claim is an identity:
// -contents of the returned buffer must be the pointer handed in, or it is a
// copy and the weights are held twice in one pool. Ctx.Import asserts that on
// every call; these tests check the assertion is reachable and the alignment
// arithmetic matches what the driver wants.

// TestImportAlignIsTheHostsPage pins the number the whole layout depends on.
//
// The page is 16384 on Apple Silicon (4096 on Linux). offheap.HostAlign takes
// the larger of the Vulkan floor and the host page, since arena arrays laid out
// on 4096 would be refused here with a nil return and no error, silently
// forcing a copy.
func TestImportAlignIsTheHostsPage(t *testing.T) {
	c, err := Open()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	defer c.Close()
	page := os.Getpagesize()
	if got := c.ImportAlign(); got != page {
		t.Fatalf("ImportAlign %d, the host's page is %d", got, page)
	}
	if uint64(page) > offheap.HostAlign() {
		t.Fatalf("the host's page is %d and the arena lays out on %d, so no arena array on "+
			"this machine can be imported at all", page, offheap.HostAlign())
	}
	t.Logf("page %d, arena alignment %d, unified %v", page, offheap.HostAlign(), c.UnifiedMemory())
}

// TestImportIsTheSamePointer is the no-copy claim itself.
func TestImportIsTheSamePointer(t *testing.T) {
	c, err := Open()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	defer c.Close()

	// Off-heap memory, because that is what the device imports: mmap'd, page
	// aligned and padded to a whole alignment.
	n := int(offheap.HostAlign())
	mem, mapped, err := offheap.MapOffHeap(n)
	if err != nil || !mapped {
		t.Fatalf("MapOffHeap(%d): %v (mapped %v)", n, err, mapped)
	}
	defer offheap.UnmapOffHeap(mem)
	qs := unsafe.Slice((*uint32)(unsafe.Pointer(&mem[0])), n/4)
	for i := range qs {
		qs[i] = uint32(i * 7)
	}
	at := uintptr(unsafe.Pointer(&mem[0]))
	b, err := c.Import(unsafe.Pointer(&mem[0]), n)
	if err != nil {
		t.Fatalf("Import(%d bytes at %#x): %v", n, at, err)
	}
	defer b.Free()
	if !b.Imported() {
		t.Fatal("the buffer does not report itself as imported")
	}
	// Bytes() must be the mapped memory: checked both ways, because a
	// one-way check passes on a buffer copied once.
	host := b.Bytes()
	if len(host) != n {
		t.Fatalf("the buffer reports %d bytes, want %d", len(host), n)
	}
	if uintptr(unsafe.Pointer(&host[0])) != at {
		t.Fatalf("the buffer's memory is at %#x and the mapping's is at %#x: that is a COPY",
			uintptr(unsafe.Pointer(&host[0])), at)
	}
	qs[0] ^= 0xffffffff
	if got := uint32(host[0]) | uint32(host[1])<<8 | uint32(host[2])<<16 | uint32(host[3])<<24; got != qs[0] {
		t.Fatalf("wrote %#x through the mapping and the buffer reads %#x", qs[0], got)
	}
}

// TestImportRefusesWhatTheDriverWouldRefuse is the violation arm:
// newBufferWithBytesNoCopy: with a misaligned pointer or a partial-page length
// returns nil with no NSError, so Import must refuse these itself.
func TestImportRefusesWhatTheDriverWouldRefuse(t *testing.T) {
	c, err := Open()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	defer c.Close()
	page := c.ImportAlign()

	base, err := c.Alloc(4 * page)
	if err != nil {
		t.Fatal(err)
	}
	defer base.Free()
	// A page-aligned region to misuse: Metal buffer contents are page
	// aligned, so the misaligned case is built deliberately.
	mem := base.Bytes()
	p := unsafe.Pointer(&mem[0])
	if uintptr(p)%uintptr(page) != 0 {
		t.Skipf("the scratch buffer at %p is not page aligned, so this case cannot be built", p)
	}

	for _, c2 := range []struct {
		name string
		p    unsafe.Pointer
		n    int
	}{
		{"a pointer one byte past a page", unsafe.Pointer(&mem[1]), page},
		{"a length that is not whole pages", p, page + 1},
		{"no length at all", p, 0},
		{"a nil pointer", nil, page},
	} {
		b, err := c.Import(c2.p, c2.n)
		if err == nil {
			b.Free()
			t.Errorf("%s was ACCEPTED: the driver refuses these with a nil return and no "+
				"error object, so a caller that does not check them copies forever", c2.name)
			continue
		}
		t.Logf("%-34s refused: %v", c2.name, err)
	}
}
