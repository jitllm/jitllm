package offheap

import (
	"os"
	"testing"
)

// TestReservationCommitsWritesAndDrops fills a range of a reservation once
// it is committed, reads it back, drops it and reads zeros once it is
// committed again, the neighbouring ranges untouched; and maps memory that
// reads zeros and takes writes. Where there is no reservation it checks the
// platform says so.
func TestReservationCommitsWritesAndDrops(t *testing.T) {
	ps := os.Getpagesize()
	r, mapped, err := ReserveOffHeap(4 * ps)
	if err != nil {
		t.Fatal(err)
	}
	if !mapped {
		if r != nil {
			t.Fatalf("no reservation, but %d bytes back", len(r))
		}
		t.Log("no reservation on this platform")
		return
	}
	defer UnmapOffHeap(r)
	if len(r) != 4*ps {
		t.Fatalf("reserved %d bytes, asked %d", len(r), 4*ps)
	}
	fill := func(b []byte, v byte) {
		for i := range b {
			b[i] = v + byte(i)
		}
	}
	page := func(i int) []byte { return r[i*ps : (i+1)*ps] }
	for _, i := range []int{1, 2} {
		if err := CommitOffHeap(page(i)); err != nil {
			t.Fatal(err)
		}
		for j, v := range page(i) {
			if v != 0 {
				t.Fatalf("page %d byte %d is %d before anything wrote it", i, j, v)
			}
		}
		fill(page(i), byte(i))
	}
	DropOffHeap(page(1))
	if err := CommitOffHeap(page(1)); err != nil {
		t.Fatal(err)
	}
	for j, v := range page(1) {
		if v != 0 {
			t.Fatalf("dropped page byte %d is %d, not zero", j, v)
		}
	}
	for j, v := range page(2) {
		if v != byte(2)+byte(j) {
			t.Fatalf("page 2 byte %d is %d after its neighbour was dropped", j, v)
		}
	}

	m, mapped, err := MapOffHeap(3 * ps)
	if err != nil {
		t.Fatal(err)
	}
	if !mapped {
		t.Fatal("a platform that reserves maps nothing")
	}
	defer UnmapOffHeap(m)
	for j, v := range m {
		if v != 0 {
			t.Fatalf("mapped byte %d is %d", j, v)
		}
	}
	fill(m, 7)
	if m[len(m)-1] != byte(7)+byte(len(m)-1) {
		t.Fatal("the mapping did not take a write")
	}
}
