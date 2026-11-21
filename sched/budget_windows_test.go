//go:build windows

package sched

import "testing"

// TestMemBudgetIsANumberOnWindows: a budget of zero reads as unlimited to the
// pager and the server, which is how Windows once never paged by budget and
// served a default page budget of 0 B. The budget must be nonzero, inside
// what is available, and below the physical memory with headroom.
func TestMemBudgetIsANumberOnWindows(t *testing.T) {
	total, avail := windowsMemory()
	b := MemBudget()
	t.Logf("physical %.2f GB, available %.2f GB, budget %.2f GB",
		float64(total)/1e9, float64(avail)/1e9, float64(b)/1e9)
	// A struct whose layout or length is wrong fails the call or reads garbage;
	// a real Windows host has at least a gigabyte.
	if total < 1<<30 {
		t.Fatalf("total physical memory read as %d bytes", total)
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
