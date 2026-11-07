package jlm

import "testing"

// TestViewsKeepTheirOwnOrder holds two views of one model's frames -- two
// instances on one CPU device -- each to its own budget and order: A
// retains its three pages while B scans three others under a budget of its
// own; A then claiming a page B holds reads nothing and gives up its own
// most recent page; a page both retain stays when one gives it up. The
// model-wide pager -- one view at the same budget, the frames' one order --
// is the violation: B's scan evicts A's pages.
func TestViewsKeepTheirOwnOrder(t *testing.T) {
	const page = 4096
	a := newFake(8, page, 0, Align)
	a.SetBudget(3 * page)
	b, err := a.View(3 * page)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 3 {
		a.fakeIn(i, 1)
	}
	for i := 5; i < 8; i++ {
		b.fakeIn(i, 2)
	}
	for i := range 3 {
		if a.pages[i] == nil {
			t.Fatalf("B's scan under its own budget evicted A's page %d", i)
		}
	}
	if a.held != 3*page || b.held != 3*page || a.resident != 6*page {
		t.Fatalf("A holds %d bytes and B %d of %d resident, want 3 pages each of 6", a.held, b.held, a.resident)
	}
	fresh := a.fresh
	a.fakeIn(6, 3)
	switch {
	case a.fresh != fresh:
		t.Fatal("A claiming B's page read a frame of its own")
	case a.pages[2] != nil:
		t.Fatal("A claiming a fourth page under three pages kept its most recent one")
	case a.pages[6] == nil || a.held != 3*page || b.held != 3*page:
		t.Fatalf("after A claimed B's page 6: A holds %d bytes and B %d, want 3 pages each", a.held, b.held)
	}
	b.DropPage(6)
	if a.pages[6] == nil || b.held != 2*page {
		t.Fatalf("B giving page 6 up took it from A too (B holds %d bytes)", b.held)
	}
	t.Logf("two views of 3 pages each: B's scan left A's pages in, A read nothing for B's page and gave up its own, B giving a shared page up left it A's")

	one := newFake(8, page, 0, Align)
	one.SetBudget(3 * page)
	for i := range 3 {
		one.fakeIn(i, 1)
	}
	for i := 5; i < 8; i++ {
		one.fakeIn(i, 2)
	}
	gone := 0
	for i := range 3 {
		if one.pages[i] == nil {
			gone++
		}
	}
	t.Logf("violation (one view, one order): the second scan evicted %d of the first's 3 pages", gone)
	if gone == 0 {
		t.Fatal("VIOLATION: one view at the same budget keeps every page: the gate cannot tell the views apart")
	}
}
