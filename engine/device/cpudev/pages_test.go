package cpudev

import "testing"

// TestPagesGrowAndGoBack grows a block's pages out of its arena until it is
// full and then out of the pool, releases them to the arena and the spares,
// grows again from the spares, and hands the rest to the pool, which a second
// block's pages grow out of, zeroed.
func TestPagesGrowAndGoBack(t *testing.T) {
	const pp, slots = 32, 2
	a, err := MapArena(pp, slots, false)
	if err != nil {
		t.Fatal(err)
	}
	defer a.unmap()
	pool := &PagePool{}
	v := NewView(1)
	p := v.Pages(0, pp, false, pool)
	p.UseArena(a, 4)
	for n := range 4 {
		p.Grow(n)
	}
	for n := range 4 {
		if !p.Resident(n) {
			t.Fatalf("page %d is not resident after Grow", n)
		}
		if got, want := p.ArenaPage(n), n < slots; got != want {
			t.Fatalf("page %d in the arena = %v, want %v: the arena's %d slots come first", n, got, want, slots)
		}
	}
	p.K(3)[0] = 7
	p.Release(3, 4)
	p.Release(0, 4)
	if p.Resident(3) || p.Resident(0) || len(p.spare) != 2 {
		t.Fatalf("released: page 3 %v, page 0 %v, %d spares; want neither resident and page 3's two buffers spare",
			p.Resident(3), p.Resident(0), len(p.spare))
	}
	p.Grow(0)
	if !p.ArenaPage(0) {
		t.Fatal("a page grown after its slot went back did not take the slot again")
	}
	p.Grow(3)
	if p.ArenaPage(3) || p.K(3)[0] != 0 || len(p.spare) != 0 {
		t.Fatalf("page 3 grown again: arena %v, first float %v, %d spares; want a zeroed spare",
			p.ArenaPage(3), p.K(3)[0], len(p.spare))
	}
	for n := range 4 {
		if p.ArenaPage(n) {
			p.Give(p.K(n))
		}
	}
	p.Unmap()
	p.K(2)[1] = 9
	pool.Put(p)
	if held := pool.Held(nil); held != 2*2*4*pp {
		t.Fatalf("the pool holds %d bytes, want the two heap pages' %d", held, 2*2*4*pp)
	}
	q := v.Pages(0, pp, false, pool)
	q.Grow(0)
	if q.K(0)[1] != 0 || pool.Held(nil) != 2*4*pp {
		t.Fatalf("a page grown from the pool: float %v, %d bytes left; want zeros and one page's buffers taken",
			q.K(0)[1], pool.Held(nil))
	}
	l := v.Pages(0, pp, true, nil)
	l.Grow(0)
	if &l.K(0)[0] != &l.V(0)[0] {
		t.Fatal("a latent block's value is not its key's memory")
	}
}
