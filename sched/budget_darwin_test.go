//go:build darwin

package sched

import "testing"

// TestMemBudgetIsANumberOnDarwin: a budget of zero reads as unlimited to the
// pager and the server, which is how a Mac once never paged and served a
// default page budget of 0 B. The budget must be nonzero, inside what is
// available, and below the physical memory with headroom.
func TestMemBudgetIsANumberOnDarwin(t *testing.T) {
	total, avail, b := darwinMemsize(), darwinAvailable(), MemBudget()
	t.Logf("hw.memsize %.2f GB, available %.2f GB, budget %.2f GB",
		float64(total)/1e9, float64(avail)/1e9, float64(b)/1e9)
	// A NUL-trimmed hw.memsize read without padding is 1/256 of the truth or
	// garbage; a real Mac has at least a gigabyte.
	if total < 1<<30 {
		t.Fatalf("hw.memsize read as %d bytes", total)
	}
	if avail == 0 || avail > total {
		t.Fatalf("available %d of %d bytes", avail, total)
	}
	if b == 0 {
		t.Fatal("budget is 0 with the memory known: the pager would read it as unlimited")
	}
	if b >= avail {
		t.Errorf("budget %d leaves no headroom under the available %d", b, avail)
	}
}
